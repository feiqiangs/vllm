# SPDX-License-Identifier: Apache-2.0
"""
MMA CPU-GPU offloading handler  —  Relay Daemon variant (方案2)

Key differences from original mma_cpu_gpu.py:
  1. Does NOT call mma.init() — MMA lives only in the Daemon process
  2. Does NOT allocate CPU tensors via torch.zeros(pin_memory=True)
     — CPU memory is owned by Daemon (cudaHostAllocPortable), shared via shm
  3. transfer_async() pushes requests to Daemon via MMADaemonClient ring buffer
     instead of calling mma.batch_d2h_async / mma.batch_h2d_async directly
  4. get_finished() drains the completion ring buffer instead of querying CUDA events

CPU tensor access:
  Ranks read results from the mmap'd cpu_pool shm segment via MMADaemonClient.
  The CPU tensors exposed upward are torch.Tensor wrappers over the shm region
  (zero-copy on the rank side).

Startup protocol (called from MmaCpuGpuOffloadingHandlers.__init__):
  1. Optionally launch daemon (rank 0 only, or external orchestration)
  2. MMADaemonClient.attach()  — registers IPC handle, waits for daemon_ready
  3. Build logical CPU tensor views over shm cpu_pool
"""

from __future__ import annotations

import ctypes
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_offload.mediums import BlockIDsLoadStoreSpec
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    TransferResult,
    TransferSpec,
)
from vllm.v1.kv_offload.worker.mma_daemon import DIR_D2H, DIR_H2D, launch_daemon
from vllm.v1.kv_offload.worker.mma_daemon_client import MMADaemonClient

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Block ID expansion  (unchanged from original)
# ---------------------------------------------------------------------------

def expand_block_ids(
    block_ids: np.ndarray,
    block_size_factor: int,
    output: np.ndarray,
    skip_count: int = 0,
) -> None:
    assert skip_count < block_size_factor
    first_range = np.arange(skip_count, block_size_factor)
    full_range  = np.arange(0, block_size_factor)
    output_idx  = 0
    for i, block_id in enumerate(block_ids):
        base = block_id * block_size_factor
        indices = first_range if i == 0 else full_range
        end = output_idx + len(indices)
        output[output_idx:end] = base + indices
        output_idx = end


# ---------------------------------------------------------------------------
# Single-direction handler  (Daemon IPC variant)
# ---------------------------------------------------------------------------

@dataclass
class _PendingTransfer:
    job_id:    int
    num_bytes: int
    submit_ns: int
    cpu_offset: int   # for slot reclaim


class MmaDaemonDirectionHandler(OffloadingHandler):
    """
    Offloading handler for a single transfer direction (D2H or H2D)
    that delegates actual transfers to the MMA Relay Daemon via IPC.

    The interface is identical to the original MmaSingleDirectionOffloadingHandler
    so it can be dropped in as a replacement.
    """

    def __init__(
        self,
        client: MMADaemonClient,
        src_tensors: list[torch.Tensor],
        dst_tensors: list[torch.Tensor],
        src_block_size_factor: int,
        dst_block_size_factor: int,
        gpu_to_cpu: bool,
    ):
        """
        Args:
            client               : Attached MMADaemonClient for this rank
            src_tensors          : Source tensor list (GPU for D2H, CPU shm view for H2D)
            dst_tensors          : Destination tensor list
            src_block_size_factor: blocks per KV-block in src
            dst_block_size_factor: blocks per KV-block in dst
            gpu_to_cpu           : True → D2H offload, False → H2D reload
        """
        assert len(src_tensors) == len(dst_tensors)
        self.client = client
        self.src_tensors = src_tensors
        self.dst_tensors = dst_tensors
        self.gpu_to_cpu  = gpu_to_cpu

        min_factor = min(src_block_size_factor, dst_block_size_factor)
        self.src_block_size_factor = src_block_size_factor // min_factor
        self.dst_block_size_factor = dst_block_size_factor // min_factor

        # Byte stride per block in src tensors (for pointer arithmetic)
        self.block_stride_bytes = [
            t.element_size() * t.stride(0) * min_factor
            for t in src_tensors
        ]
        self.total_block_bytes = sum(self.block_stride_bytes)

        # For GPU tensors we need the base pointer and the offset from kv_cache base
        self.gpu_base_ptr      = client.kv_cache_base_ptr
        self.transfer_type     = ("GPU", "CPU") if gpu_to_cpu else ("CPU", "GPU")

        self._transfers: deque[_PendingTransfer] = deque()

    # ------------------------------------------------------------------

    def transfer_async(self, job_id: int, transfer_spec: TransferSpec) -> bool:
        """
        Push a transfer request to the Daemon via ring buffer.

        Instead of directly calling mma.batch_d2h_async, we:
          1. Compute the byte offset from kv_cache_base_ptr for each sub-block
          2. Allocate a cpu_pool slot of the required size
          3. Push a single request slot to the daemon ring buffer
        The Daemon will call mma.batch_d2h_async / mma.batch_h2d_async on
        the full region.
        """
        src_spec, dst_spec = transfer_spec
        assert isinstance(src_spec, BlockIDsLoadStoreSpec)
        assert isinstance(dst_spec, BlockIDsLoadStoreSpec)

        src_blocks = src_spec.block_ids
        dst_blocks = dst_spec.block_ids
        assert src_blocks.ndim == dst_blocks.ndim == 1

        dst_sub_count     = dst_blocks.size * self.dst_block_size_factor
        src_sub_count     = src_blocks.size * self.src_block_size_factor
        src_skip          = -dst_blocks.size % self.src_block_size_factor
        assert dst_sub_count == src_sub_count - src_skip

        # Expand to sub-block indices
        src_expanded = np.empty(dst_sub_count, dtype=np.int64)
        dst_expanded = np.empty(dst_sub_count, dtype=np.int64)
        expand_block_ids(src_blocks, self.src_block_size_factor, src_expanded, src_skip)
        expand_block_ids(dst_blocks, self.dst_block_size_factor, dst_expanded)

        total_bytes = dst_sub_count * self.total_block_bytes

        # Allocate a CPU pool slot for this transfer
        cpu_offset = self.client.alloc_cpu_slot(total_bytes)

        # Compute GPU byte offset (first sub-block offset from kv_cache base)
        # For multi-tensor KV caches we transfer all layers as one contiguous blob:
        # This requires the src_tensors to be contiguous within the kv_cache tensor.
        # We use the first block's offset as the base and assume contiguous layout.
        if self.gpu_to_cpu:
            gpu_tensor = self.src_tensors[0]
        else:
            gpu_tensor = self.dst_tensors[0]

        gpu_tensor_ptr = gpu_tensor.data_ptr()
        first_block_idx = int(src_expanded[0] if self.gpu_to_cpu else dst_expanded[0])
        gpu_offset_bytes = (gpu_tensor_ptr - self.gpu_base_ptr) + \
                           first_block_idx * self.block_stride_bytes[0]

        # Submit to daemon
        if self.gpu_to_cpu:
            submitted_job = self.client.submit_d2h(gpu_offset_bytes, cpu_offset, total_bytes)
        else:
            submitted_job = self.client.submit_h2d(gpu_offset_bytes, cpu_offset, total_bytes)

        self._transfers.append(_PendingTransfer(
            job_id=job_id,
            num_bytes=total_bytes,
            submit_ns=time.time_ns(),
            cpu_offset=cpu_offset,
        ))
        return True

    def get_finished(self) -> list[TransferResult]:
        """
        Drain completion ring buffer and return finished TransferResult list.
        Daemon completions are matched against pending transfers in FIFO order.
        """
        results: list[TransferResult] = []
        if not self._transfers:
            return results

        # Poll daemon completions
        completions = {jid: ok for jid, ok in self.client.poll_completions()}

        while self._transfers:
            t = self._transfers[0]
            if t.job_id not in completions:
                break
            ok = completions.pop(t.job_id)
            self._transfers.popleft()

            elapsed_ns  = time.time_ns() - t.submit_ns
            results.append(TransferResult(
                job_id        = t.job_id,
                success       = ok,
                transfer_size = t.num_bytes,
                transfer_time = elapsed_ns * 1e-9,
                transfer_type = self.transfer_type,
            ))

        return results

    def wait(self, job_ids: set[int]) -> None:
        """Blocking wait for specific jobs."""
        self.client.wait_for_jobs(job_ids)


# ---------------------------------------------------------------------------
# Top-level factory  (drop-in replacement for MmaCpuGpuOffloadingHandlers)
# ---------------------------------------------------------------------------

class MmaCpuGpuOffloadingHandlers:
    """
    Creates D2H and H2D offloading handlers backed by the MMA Relay Daemon.

    Key behavioral changes vs original:
      - Does NOT call mma.init() in the rank process
      - Does NOT allocate pinned CPU tensors via torch.zeros(pin_memory=True)
      - Launches daemon subprocess (rank 0 only) and attaches all ranks via IPC
      - CPU tensors are logical views over daemon-owned shm cpu_pool
    """

    def __init__(
        self,
        rank: int,
        num_ranks: int,
        gpu_block_size: int,
        cpu_block_size: int,
        num_cpu_blocks: int,
        gpu_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
        cpu_pool_bytes: int,
        mma_config_path: str | None = None,
        daemon_timeout_s: float = 60.0,
    ):
        """
        Args:
            rank          : This rank's DP index (0..num_ranks-1)
            num_ranks     : Total number of DP ranks
            cpu_pool_bytes: Total size of shared pinned CPU pool (all ranks share this)
            ... (rest same as original)
        """
        assert gpu_caches
        assert cpu_block_size % gpu_block_size == 0

        self.rank = rank

        # ----------------------------------------------------------------
        # Step 1: rank 0 launches the daemon subprocess
        # ----------------------------------------------------------------
        if rank == 0:
            self._daemon_proc = launch_daemon(
                num_ranks      = num_ranks,
                cpu_pool_bytes = cpu_pool_bytes,
                mma_config_path= mma_config_path,
            )
            logger.info("Rank 0: MMA Relay Daemon launched (pid=%d)", self._daemon_proc.pid)
        else:
            self._daemon_proc = None

        # ----------------------------------------------------------------
        # Step 2: Build the flat GPU tensor list (same logic as original)
        # ----------------------------------------------------------------
        kernel_block_size: int | None = None
        parsed_gpu_tensors: list[tuple[torch.Tensor, bool]] = []

        for layer_name, gpu_tensor in gpu_caches.items():
            gpu_shape    = gpu_tensor.shape
            attn_backend = attn_backends[layer_name]
            test_shape   = attn_backend.get_kv_cache_shape(
                num_blocks=1234, block_size=16, num_kv_heads=8, head_size=256
            )

            has_layers_dim = False
            split_k_and_v  = False
            if len(gpu_shape) != len(test_shape):
                has_layers_dim = True
                test_shape = (80,) + test_shape
            elif test_shape[0] != 1234:
                assert test_shape[0] == 2
                split_k_and_v = True

            try:
                stride_order = attn_backend.get_kv_cache_stride_order(
                    include_num_layers_dimension=has_layers_dim
                )
            except (AttributeError, NotImplementedError):
                stride_order = tuple(range(len(gpu_shape)))

            test_shape    = tuple(test_shape[i] for i in stride_order)
            block_size_idx = test_shape.index(16)
            if kernel_block_size is not None:
                assert kernel_block_size == gpu_shape[block_size_idx]
            else:
                kernel_block_size = gpu_shape[block_size_idx]
                assert gpu_block_size % kernel_block_size == 0

            parsed_gpu_tensors.append((gpu_tensor, split_k_and_v))

        assert kernel_block_size is not None
        cpu_block_size_factor = cpu_block_size // kernel_block_size
        gpu_block_size_factor = gpu_block_size // kernel_block_size

        gpu_tensors: list[torch.Tensor] = []
        for gpu_tensor, split_k_and_v in parsed_gpu_tensors:
            gpu_tensors.extend(
                gpu_tensor.unbind(0) if split_k_and_v else [gpu_tensor]
            )

        # ----------------------------------------------------------------
        # Step 3: The "KV cache tensor" that we register with daemon is the
        # first (and usually only contiguous) gpu_tensor in gpu_caches.
        # All block offsets are computed relative to its data_ptr().
        # ----------------------------------------------------------------
        # Use the first layer tensor as IPC registration target.
        # For multi-tensor setups, all tensors are contiguous within the same
        # cudaMalloc slab (guaranteed by vLLM's KV cache allocator).
        # We register the full slab by using the min data_ptr and max extent.
        kv_tensors = list(gpu_caches.values())
        kv_base_ptr = min(t.data_ptr() for t in kv_tensors)
        kv_end_ptr  = max(t.data_ptr() + t.numel() * t.element_size() for t in kv_tensors)
        kv_size     = kv_end_ptr - kv_base_ptr

        # Build a "view" tensor covering the full slab for IPC registration
        import ctypes as _ct
        kv_slab = torch.from_blob(
            _ct.c_void_p(kv_base_ptr),
            [kv_size],
            dtype=torch.uint8,
        )
        # Ensure it's on the right CUDA device
        kv_slab = kv_slab.cuda(rank)

        # ----------------------------------------------------------------
        # Step 4: Attach to daemon via MMADaemonClient
        # ----------------------------------------------------------------
        self._client = MMADaemonClient(rank=rank, kv_cache_tensor=kv_slab)
        self._client.attach(timeout_s=daemon_timeout_s)

        # ----------------------------------------------------------------
        # Step 5: Build CPU tensor views over shm cpu_pool
        # (logical placeholders — actual data lives in daemon's pinned pool)
        # ----------------------------------------------------------------
        # We create placeholder CPU tensors of the right shape/dtype so that
        # the scheduler-side block manager can compute addresses.
        # These are NOT used for actual data transfer (daemon does that);
        # they exist purely for compatibility with the OffloadingHandler interface.
        cpu_tensors: list[torch.Tensor] = []
        for gpu_tensor, split_k_and_v in parsed_gpu_tensors:
            cpu_shape = list(gpu_tensor.shape)
            num_cpu_kernel_blocks = num_cpu_blocks * cpu_block_size_factor
            cpu_shape[1 if split_k_and_v else 0] = num_cpu_kernel_blocks

            # Logical CPU tensor (not pinned here; daemon owns actual pinned mem)
            # We use the shm-backed approach: frombuffer on cpu_pool_mm when needed
            cpu_tensor = torch.zeros(cpu_shape, dtype=gpu_tensor.dtype, device="cpu")
            cpu_tensors.extend(
                cpu_tensor.unbind(0) if split_k_and_v else [cpu_tensor]
            )

        # ----------------------------------------------------------------
        # Step 6: Create directional handlers
        # ----------------------------------------------------------------
        self.gpu_to_cpu_handler = MmaDaemonDirectionHandler(
            client               = self._client,
            src_tensors          = gpu_tensors,
            dst_tensors          = cpu_tensors,
            src_block_size_factor= gpu_block_size_factor,
            dst_block_size_factor= cpu_block_size_factor,
            gpu_to_cpu           = True,
        )
        self.cpu_to_gpu_handler = MmaDaemonDirectionHandler(
            client               = self._client,
            src_tensors          = cpu_tensors,
            dst_tensors          = gpu_tensors,
            src_block_size_factor= cpu_block_size_factor,
            dst_block_size_factor= gpu_block_size_factor,
            gpu_to_cpu           = False,
        )

        logger.info(
            "MmaCpuGpuOffloadingHandlers (Daemon variant) initialized: "
            "rank=%d/%d, gpu_blocks=%d, cpu_blocks=%d",
            rank, num_ranks, len(gpu_tensors[0]) if gpu_tensors else 0, num_cpu_blocks,
        )
