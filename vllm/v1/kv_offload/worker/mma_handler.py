# SPDX-License-Identifier: Apache-2.0
"""
MMA CPU-GPU offloading handler — thin adapter layer for vLLM.

This module translates vLLM's block-level KV cache transfer requests
into MMAClient API calls. It has zero knowledge of MMA internals.

Key responsibilities:
  1. Parse vLLM's KV cache tensors into per-layer registration format
  2. Translate block-level transfer specs into (layer_idx, byte_offset, size)
  3. Delegate all actual transfers to MMAClient
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_offload.mediums import BlockIDsLoadStoreSpec
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    TransferResult,
    TransferSpec,
)

# The only MMA import vLLM needs — mma_relay is an external package
# provided by the MMA library.  If it is not installed, fail early.
try:
    from mma_relay.client import MMAClient
except ImportError as e:
    raise ImportError(
        "MMA offloading requires the 'mma_relay' package. "
        "Please install MMA first (pip install mma)."
    ) from e

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Block ID expansion  (same as vLLM's cpu_gpu.py)
# ---------------------------------------------------------------------------


def expand_block_ids(
    block_ids: np.ndarray,
    block_size_factor: int,
    output: np.ndarray,
    skip_count: int = 0,
) -> None:
    """Vectorized block ID expansion (no Python for-loop)."""
    assert skip_count < block_size_factor
    n = len(block_ids)
    if n == 0:
        return

    full_range = np.arange(block_size_factor, dtype=np.int64)

    # Broadcast: (N, block_size_factor) -> flatten
    expanded = (block_ids[:, None].astype(np.int64) * block_size_factor
                + full_range[None, :]).ravel()

    # Skip `skip_count` sub-blocks from the first block
    output[:] = expanded[skip_count:skip_count + len(output)]


# ---------------------------------------------------------------------------
# Single-direction handler
# ---------------------------------------------------------------------------


@dataclass
class _PendingTransfer:
    job_id: int
    num_bytes: int
    submit_ns: int
    cpu_offset: int
    cpu_size: int  # actual CPU slot allocation size for deferred free
    mma_job_ids: list[int]  # kept for backward compat, but not used for tracking
    batch_id: int = -1  # MMA batch_id from submit_batch (batch-level tracking)
    completed: bool = False  # set to True when batch completion is received
    all_ok: bool = True  # tracks whether the batch completed successfully


class MmaDirectionHandler(OffloadingHandler):
    """
    Offloading handler for a single transfer direction (D2H or H2D)
    that delegates actual transfers to MMAClient.
    """

    def __init__(
        self,
        client: MMAClient,
        src_tensors: list[torch.Tensor],
        dst_tensors: list[torch.Tensor],
        src_block_size_factor: int,
        dst_block_size_factor: int,
        gpu_to_cpu: bool,
    ):
        assert len(src_tensors) == len(dst_tensors)
        self.client = client
        self.src_tensors = src_tensors
        self.dst_tensors = dst_tensors
        self.gpu_to_cpu = gpu_to_cpu

        min_factor = min(src_block_size_factor, dst_block_size_factor)
        self.src_block_size_factor = src_block_size_factor // min_factor
        self.dst_block_size_factor = dst_block_size_factor // min_factor

        # Byte stride per block in src tensors
        self.block_stride_bytes = [
            t.element_size() * t.stride(0) * min_factor
            for t in src_tensors
        ]
        self.total_block_bytes = sum(self.block_stride_bytes)

        # Per-layer base pointers
        self.layer_base_ptrs: list[int] = [
            t.storage().data_ptr()
            for t in (src_tensors if gpu_to_cpu else dst_tensors)
        ]
        self.transfer_type = ("GPU", "CPU") if gpu_to_cpu else ("CPU", "GPU")

        self._transfers: deque[_PendingTransfer] = deque()
        # Maps batch_id -> owning _PendingTransfer for O(1) completion lookup.
        # MMAClient.poll_completions() returns (batch_id, all_ok) at batch
        # granularity — one batch covers all layers in a single submit_batch.
        self._batch_to_transfer: dict[int, _PendingTransfer] = {}

    # ------------------------------------------------------------------

    def transfer_async(self, job_id: int, transfer_spec: TransferSpec) -> bool:
        """Push a transfer request via MMAClient.

        Block IDs from vLLM are NOT necessarily contiguous — they can be
        arbitrary scattered indices (e.g. [5, 12, 3, 8]).  We must emit
        one MMA transfer item per (layer, sub-block) pair so the Daemon
        performs per-block scatter/gather DMA, not one large contiguous
        memcpy.
        """
        src_spec, dst_spec = transfer_spec
        assert isinstance(src_spec, BlockIDsLoadStoreSpec)
        assert isinstance(dst_spec, BlockIDsLoadStoreSpec)

        src_blocks = src_spec.block_ids
        dst_blocks = dst_spec.block_ids
        assert src_blocks.ndim == dst_blocks.ndim == 1

        dst_sub_count = dst_blocks.size * self.dst_block_size_factor
        src_sub_count = src_blocks.size * self.src_block_size_factor
        src_skip = -dst_blocks.size % self.src_block_size_factor
        assert dst_sub_count == src_sub_count - src_skip

        # Expand to sub-block indices
        src_expanded = np.empty(dst_sub_count, dtype=np.int64)
        dst_expanded = np.empty(dst_sub_count, dtype=np.int64)
        expand_block_ids(
            src_blocks, self.src_block_size_factor, src_expanded, src_skip
        )
        expand_block_ids(dst_blocks, self.dst_block_size_factor, dst_expanded)

        total_bytes = dst_sub_count * self.total_block_bytes

        # Allocate a CPU pool slot — gracefully handle pool exhaustion
        try:
            cpu_offset = self.client.alloc_cpu_slot(total_bytes)
        except RuntimeError:
            logger.warning(
                "CPU pool exhausted for job %d (need %d bytes), "
                "transfer deferred",
                job_id, total_bytes,
            )
            return False

        # Determine GPU tensors and block index arrays for offset computation.
        # For gpu_to_cpu: GPU is src, so gpu_block_indices = src_expanded
        # For cpu_to_gpu: GPU is dst, so gpu_block_indices = dst_expanded
        if self.gpu_to_cpu:
            gpu_tensors = self.src_tensors
            gpu_block_indices = src_expanded
        else:
            gpu_tensors = self.dst_tensors
            gpu_block_indices = dst_expanded

        direction = 0 if self.gpu_to_cpu else 1  # DIR_D2H=0, DIR_H2D=1

        # Build per-(layer, sub-block) parameters for batch submission.
        # Each sub-block is a separate transfer item so the Daemon
        # performs scatter/gather DMA for non-contiguous block IDs.
        layer_indices: list[int] = []
        gpu_offsets: list[int] = []
        cpu_offsets: list[int] = []
        sizes: list[int] = []

        cumulative_cpu_offset = 0
        for layer_i, gpu_tensor in enumerate(gpu_tensors):
            view_offset = gpu_tensor.data_ptr() - self.layer_base_ptrs[layer_i]
            block_stride = self.block_stride_bytes[layer_i]

            for blk_j in range(dst_sub_count):
                gpu_blk_idx = int(gpu_block_indices[blk_j])
                gpu_offset_ij = view_offset + gpu_blk_idx * block_stride
                cpu_offset_ij = cpu_offset + cumulative_cpu_offset

                layer_indices.append(layer_i)
                gpu_offsets.append(gpu_offset_ij)
                cpu_offsets.append(cpu_offset_ij)
                sizes.append(block_stride)
                cumulative_cpu_offset += block_stride

        # Batch submit all (layer, sub-block) items at once.
        # submit_batch returns (job_ids, batch_id) — completion tracking
        # is at batch granularity via batch_id.
        submitted_jobs, batch_id = self.client.submit_batch(
            direction, layer_indices, gpu_offsets, cpu_offsets, sizes
        )

        pending = _PendingTransfer(
            job_id=job_id,
            num_bytes=total_bytes,
            submit_ns=time.time_ns(),
            cpu_offset=cpu_offset,
            cpu_size=total_bytes,
            mma_job_ids=submitted_jobs,
            batch_id=batch_id,
            completed=False,
            all_ok=True,
        )
        self._transfers.append(pending)
        self._batch_to_transfer[batch_id] = pending

        return True

    def get_finished(self) -> list[TransferResult]:
        """Poll completions and return finished TransferResult list.

        MMAClient.poll_completions() returns batch-level results:
        [(batch_id, all_ok)].  Each batch corresponds to exactly one
        vLLM transfer (one submit_batch call per transfer_async call).

        This is more efficient than per-job tracking: one poll result
        covers all layers in a transfer.
        """
        results: list[TransferResult] = []
        if not self._transfers:
            return results

        # Accumulate batch-level completions.
        for batch_id, all_ok in self.client.poll_completions():
            pending = self._batch_to_transfer.pop(batch_id, None)
            if pending is not None:
                pending.completed = True
                pending.all_ok = all_ok

        # Drain completed transfers from the front of the deque.
        while self._transfers:
            t = self._transfers[0]
            if not t.completed:
                break

            self._transfers.popleft()

            # Deferred CPU slot reclamation: free now that all layers
            # have completed, so no in-flight DMA touches this memory.
            self.client.free_cpu_slot(
                t.cpu_offset, t.cpu_size
            )

            elapsed_ns = time.time_ns() - t.submit_ns
            results.append(
                TransferResult(
                    job_id=t.job_id,
                    success=t.all_ok,
                    transfer_size=t.num_bytes,
                    transfer_time=elapsed_ns * 1e-9,
                    transfer_type=self.transfer_type,
                )
            )

        return results

    def wait(self, job_ids: set[int]) -> None:
        """Blocking wait for specific vLLM-level job IDs.

        Collects the underlying MMA batch IDs for the requested vLLM
        transfers and delegates to the client's batch-level wait.

        Note: wait_for_batches() internally calls poll_completions()
        which consumes completion ring entries.  We must NOT call
        poll_completions() again here — instead we mark the waited
        batches as completed directly, since wait_for_batches() only
        returns after all requested batches have finished.
        """
        batch_ids: set[int] = set()
        for t in self._transfers:
            if t.job_id in job_ids and not t.completed:
                batch_ids.add(t.batch_id)
        if not batch_ids:
            return

        # wait_for_batches() spins on poll_completions() internally
        # until all requested batch_ids are drained from the completion
        # ring.  It returns a dict mapping batch_id -> all_ok, and
        # stashes any non-target completions for later poll_completions().
        waited_results = self.client.wait_for_batches(batch_ids)

        # Mark the waited transfers as completed with actual error status.
        for t in self._transfers:
            if t.batch_id in batch_ids and not t.completed:
                # Remove from batch_to_transfer map (already consumed)
                self._batch_to_transfer.pop(t.batch_id, None)
                t.completed = True
                t.all_ok = waited_results.get(t.batch_id, True)


# ---------------------------------------------------------------------------
# Top-level factory  (drop-in replacement for CpuGpuOffloadingHandlers)
# ---------------------------------------------------------------------------


class MmaCpuGpuOffloadingHandlers:
    """
    Creates D2H and H2D offloading handlers backed by MMAClient.

    This is the adapter between vLLM's OffloadingHandler interface and
    MMA's transfer engine. It:
      - Parses KV cache tensor shapes/strides (vLLM-specific logic)
      - Creates MMAClient and delegates all transfer details to it
      - Builds directional handlers for gpu_to_cpu and cpu_to_gpu
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
        timeout_s: float = 60.0,
        device_ids: list[int] | None = None,
    ):
        assert gpu_caches
        assert cpu_block_size % gpu_block_size == 0

        self.rank = rank

        # ----------------------------------------------------------------
        # Step 1: Parse GPU tensor layout  (vLLM-specific logic, unchanged)
        # ----------------------------------------------------------------
        kernel_block_size: int | None = None
        parsed_gpu_tensors: list[tuple[torch.Tensor, bool]] = []

        for layer_name, gpu_tensor in gpu_caches.items():
            gpu_shape = gpu_tensor.shape
            attn_backend = attn_backends[layer_name]
            test_shape = attn_backend.get_kv_cache_shape(
                num_blocks=1234,
                block_size=16,
                num_kv_heads=8,
                head_size=256,
            )

            has_layers_dim = False
            split_k_and_v = False
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

            test_shape = tuple(test_shape[i] for i in stride_order)
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
        # Step 2: Collect per-layer KV caches for registration
        # ----------------------------------------------------------------
        layer_kv_caches: dict[str, torch.Tensor] = {}
        for layer_name, gpu_tensor in gpu_caches.items():
            layer_kv_caches[layer_name] = gpu_tensor

        # ----------------------------------------------------------------
        # Step 3: Create MMAClient and start
        # ----------------------------------------------------------------
        self._client = MMAClient(
            rank=rank,
            num_ranks=num_ranks,
            cpu_pool_bytes=cpu_pool_bytes,
            mma_config_path=mma_config_path,
            device_ids=device_ids,
        )
        self._client.register_kv_layers(layer_kv_caches)
        self._client.start(timeout_s=timeout_s)

        # ----------------------------------------------------------------
        # Step 4: Build CPU tensor views (for interface compatibility)
        # ----------------------------------------------------------------
        num_cpu_kernel_blocks = num_cpu_blocks * cpu_block_size_factor
        cpu_tensors: list[torch.Tensor] = []
        for gpu_tensor, split_k_and_v in parsed_gpu_tensors:
            cpu_shape = list(gpu_tensor.shape)
            cpu_shape[1 if split_k_and_v else 0] = num_cpu_kernel_blocks

            cpu_tensor = torch.zeros(
                cpu_shape, dtype=gpu_tensor.dtype, device="cpu"
            )
            cpu_tensors.extend(
                cpu_tensor.unbind(0) if split_k_and_v else [cpu_tensor]
            )

        # ----------------------------------------------------------------
        # Step 5: Create directional handlers
        # ----------------------------------------------------------------
        self.gpu_to_cpu_handler = MmaDirectionHandler(
            client=self._client,
            src_tensors=gpu_tensors,
            dst_tensors=cpu_tensors,
            src_block_size_factor=gpu_block_size_factor,
            dst_block_size_factor=cpu_block_size_factor,
            gpu_to_cpu=True,
        )
        self.cpu_to_gpu_handler = MmaDirectionHandler(
            client=self._client,
            src_tensors=cpu_tensors,
            dst_tensors=gpu_tensors,
            src_block_size_factor=cpu_block_size_factor,
            dst_block_size_factor=gpu_block_size_factor,
            gpu_to_cpu=False,
        )

        logger.info(
            "MmaCpuGpuOffloadingHandlers initialized: "
            "rank=%d/%d, gpu_layers=%d, cpu_blocks=%d",
            rank,
            num_ranks,
            len(gpu_tensors),
            num_cpu_blocks,
        )
