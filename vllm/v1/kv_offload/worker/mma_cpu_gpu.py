# SPDX-License-Identifier: Apache-2.0
# MMA (Multi-path Memory Access) based CPU<->GPU offloading handler
# Uses MMA multi-path transfer engine for accelerated KV cache transfers
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

logger = init_logger(__name__)

# Try to import MMA
try:
    import mma

    _mma_available = mma.is_available()
except ImportError:
    _mma_available = False
    mma = None


@dataclass
class Transfer:
    job_id: int
    stream: torch.cuda.Stream
    start_event: torch.Event
    end_event: torch.Event
    num_bytes: int


def expand_block_ids(
    block_ids: np.ndarray,
    block_size_factor: int,
    output: np.ndarray,
    skip_count: int = 0,
):
    """
    Convert a list of block IDs to a list of matching block ids,
    assuming each block is composed of actual block_size_factor blocks.
    Outputs to output tensor.
    The first skip_count blocks will be skipped.
    Note that skip_count must be less than block_size_factor.

    For example, if block_ids = [0, 1, 3] and block_size_factor = 4,
    then it yields [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15]
    since 0 maps to [0, 1, 2, 3]
    1 maps to [4, 5, 6, 7]
    and 3 maps to [12, 13, 14, 15]
    """
    assert skip_count < block_size_factor

    first_range = np.arange(skip_count, block_size_factor)
    full_range = np.arange(0, block_size_factor)

    output_idx = 0
    for i, block_id in enumerate(block_ids):
        base_block_id = block_id * block_size_factor
        indices = first_range if i == 0 else full_range
        output_end_idx = output_idx + len(indices)
        output[output_idx:output_end_idx] = base_block_id + indices
        output_idx = output_end_idx


class MmaSingleDirectionOffloadingHandler(OffloadingHandler):
    """
    MMA-based offloading handler for a single transfer direction
    (either CPU->GPU or GPU->CPU).

    Instead of using ops.swap_blocks (single-path DMA), this handler
    uses MMA's batch async API to leverage NVLink multi-path relay
    through peer GPUs for significantly higher transfer bandwidth.

    Transfers are guaranteed to execute in order of submission.
    Each transfer uses a unique CUDA stream, and completion is
    detected via CUDA events recorded after MMA's spin_kernel.
    """

    def __init__(
        self,
        src_tensors: list[torch.Tensor],
        dst_tensors: list[torch.Tensor],
        src_block_size_factor: int,
        dst_block_size_factor: int,
        mma_config_path: str | None = None,
    ):
        """
        Initialize a MMA-based single direction offloading handler.

        Args:
            src_tensors: list of KV cache tensors to copy from.
            dst_tensors: list of KV cache tensors to copy to.
                Order should match src_tensors.
            src_block_size_factor: The number of kernel blocks
                per KV block in a source tensor.
            dst_block_size_factor: The number of kernel blocks
                per KV block in a destination tensor.
            mma_config_path: Path to MMA configuration file (optional).
        """
        assert len(src_tensors) == len(dst_tensors)

        # Initialize MMA in worker process (lazy, idempotent)
        if not _mma_available:
            raise RuntimeError(
                "MMA is not available. Please install MMA package "
                "with CUDA support to use MMA offloading."
            )
        assert mma is not None
        mma.init(mma_config_path)
        logger.info("MMA initialized successfully for offloading handler")

        self.src_tensors: list[torch.Tensor] = src_tensors
        self.dst_tensors: list[torch.Tensor] = dst_tensors
        min_block_size_factor = min(src_block_size_factor, dst_block_size_factor)
        self.src_block_size_factor: int = (
            src_block_size_factor // min_block_size_factor
        )
        self.dst_block_size_factor: int = (
            dst_block_size_factor // min_block_size_factor
        )

        # Precompute per-tensor block stride in bytes
        self.block_stride_bytes = [
            tensor.element_size() * tensor.stride(0) * min_block_size_factor
            for tensor in src_tensors
        ]
        self.total_block_size_in_bytes = sum(self.block_stride_bytes)

        assert len(src_tensors) > 0
        self.gpu_to_cpu: bool = self.src_tensors[0].is_cuda
        self.transfer_type = ("GPU", "CPU") if self.gpu_to_cpu else ("CPU", "GPU")

        # job_id -> event
        self._transfer_events: dict[int, torch.Event] = {}
        # queue of transfers
        self._transfers: deque[Transfer] = deque()
        # CUDA stream pool for reuse
        self._stream_pool: list[torch.cuda.Stream] = []
        # CUDA event pool for reuse
        self._event_pool: list[torch.Event] = []

    def transfer_async(self, job_id: int, transfer_spec: TransferSpec) -> bool:
        """
        Initiate an async transfer using MMA multi-path engine.

        The key difference from SingleDirectionOffloadingHandler:
        - Instead of ops.swap_blocks(src_tensor, dst_tensor, block_size, mapping),
          we compute raw pointers for each block and call mma.batch_h2d_async /
          mma.batch_d2h_async to leverage multi-path relay.
        - Completion is still detected via CUDA events, since MMA's spin_kernel
          blocks the stream until transfer completes.
        """
        assert mma is not None

        src_spec, dst_spec = transfer_spec
        assert isinstance(src_spec, BlockIDsLoadStoreSpec)
        assert isinstance(dst_spec, BlockIDsLoadStoreSpec)

        src_blocks = src_spec.block_ids
        dst_blocks = dst_spec.block_ids
        assert src_blocks.ndim == 1
        assert dst_blocks.ndim == 1

        src_sub_block_count = src_blocks.size * self.src_block_size_factor
        dst_sub_block_count = dst_blocks.size * self.dst_block_size_factor
        src_sub_blocks_to_skip = -dst_blocks.size % self.src_block_size_factor

        assert dst_sub_block_count == src_sub_block_count - src_sub_blocks_to_skip

        # Expand block IDs to sub-block level
        src_expanded = np.empty(dst_sub_block_count, dtype=np.int64)
        dst_expanded = np.empty(dst_sub_block_count, dtype=np.int64)
        expand_block_ids(
            src_blocks,
            self.src_block_size_factor,
            src_expanded,
            skip_count=src_sub_blocks_to_skip,
        )
        expand_block_ids(dst_blocks, self.dst_block_size_factor, dst_expanded)

        # Get stream and events
        stream = (
            self._stream_pool.pop() if self._stream_pool else torch.cuda.Stream()
        )
        start_event = (
            self._event_pool.pop()
            if self._event_pool
            else torch.Event(enable_timing=True)
        )
        end_event = (
            self._event_pool.pop()
            if self._event_pool
            else torch.Event(enable_timing=True)
        )

        if self.gpu_to_cpu:
            # Wait for model computation to finish before offloading
            stream.wait_stream(torch.cuda.current_stream())
        if self._transfers:
            last_transfer: Transfer = self._transfers[-1]
            last_event = last_transfer.end_event
            # Ensure job starts only after the previous one completes
            stream.wait_event(last_event)

        with torch.cuda.stream(stream):
            start_event.record(stream)

            # Core: use MMA batch API instead of ops.swap_blocks
            for src_tensor, dst_tensor, stride_bytes in zip(
                self.src_tensors,
                self.dst_tensors,
                self.block_stride_bytes,
            ):
                # Build raw pointer lists for MMA batch transfer
                src_base = src_tensor.data_ptr()
                dst_base = dst_tensor.data_ptr()

                src_ptrs = [
                    int(src_base + int(blk_id) * stride_bytes)
                    for blk_id in src_expanded
                ]
                dst_ptrs = [
                    int(dst_base + int(blk_id) * stride_bytes)
                    for blk_id in dst_expanded
                ]
                sizes = [stride_bytes] * len(src_ptrs)

                # Bug 9 fix: pass stream.cuda_stream (raw int handle)
                # explicitly instead of the torch.cuda.Stream object,
                # for robustness across different PyTorch versions.
                stream_handle = stream.cuda_stream
                if self.gpu_to_cpu:
                    mma.batch_d2h_async(
                        dst_ptrs, src_ptrs, sizes, stream=stream_handle
                    )
                else:
                    mma.batch_h2d_async(
                        dst_ptrs, src_ptrs, sizes, stream=stream_handle
                    )

            end_event.record(stream)

        self._transfer_events[job_id] = end_event
        self._transfers.append(
            Transfer(
                job_id=job_id,
                stream=stream,
                start_event=start_event,
                end_event=end_event,
                num_bytes=dst_sub_block_count * self.total_block_size_in_bytes,
            )
        )

        return True

    def get_finished(self) -> list[TransferResult]:
        """
        Get transfers finished since last call.
        Uses CUDA event queries - since MMA's spin_kernel blocks the
        stream, the end_event is recorded after spin_kernel exits,
        meaning event completion == MMA transfer completion.
        """
        results: list[TransferResult] = []
        while self._transfers and self._transfers[0].end_event.query():
            transfer = self._transfers.popleft()
            transfer_time = (
                transfer.start_event.elapsed_time(transfer.end_event) * 1e-3
            )  # elapsed_time is in milliseconds
            result = TransferResult(
                job_id=transfer.job_id,
                success=True,
                transfer_size=transfer.num_bytes,
                transfer_time=transfer_time,
                transfer_type=self.transfer_type,
            )

            results.append(result)
            self._stream_pool.append(transfer.stream)
            self._event_pool.append(transfer.end_event)
            self._event_pool.append(transfer.start_event)
            del self._transfer_events[transfer.job_id]
        return results

    def wait(self, job_ids: set[int]):
        """Wait for specific jobs to finish (blocking)."""
        for job_id in job_ids:
            event = self._transfer_events.get(job_id)
            if event is not None:
                event.synchronize()


class MmaCpuGpuOffloadingHandlers:
    """
    Creates GPU->CPU and CPU->GPU MMA-based offloading handlers.

    CPU tensor allocation logic is identical to CpuGpuOffloadingHandlers.
    The only difference is that transfers use MmaSingleDirectionOffloadingHandler
    (MMA multi-path) instead of SingleDirectionOffloadingHandler (ops.swap_blocks).
    """

    def __init__(
        self,
        gpu_block_size: int,
        cpu_block_size: int,
        num_cpu_blocks: int,
        gpu_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
        mma_config_path: str | None = None,
    ):
        assert gpu_caches
        assert cpu_block_size % gpu_block_size == 0

        # Find kernel block size and determine layout per each gpu tensor
        # (same logic as CpuGpuOffloadingHandlers)
        kernel_block_size: int | None = None
        # list of (gpu_tensor, split_k_and_v)
        parsed_gpu_tensors: list[tuple[torch.Tensor, bool]] = []
        for layer_name, gpu_tensor in gpu_caches.items():
            gpu_shape = gpu_tensor.shape
            attn_backend = attn_backends[layer_name]
            test_shape = attn_backend.get_kv_cache_shape(
                num_blocks=1234, block_size=16, num_kv_heads=8, head_size=256
            )

            has_layers_dim = False
            split_k_and_v = False
            if len(gpu_shape) != len(test_shape):
                # cross-layers tensor
                # shape is (num_blocks, ...)
                assert len(gpu_shape) == len(test_shape) + 1
                has_layers_dim = True
                # prepend a dummy num_layers=80 to test_shape
                test_shape = (80,) + test_shape
            elif test_shape[0] != 1234:
                # shape should be (2, num_blocks, ...)
                assert test_shape[0] == 2
                assert test_shape[1] == 1234
                assert gpu_shape[0] == 2
                split_k_and_v = True

            try:
                kv_cache_stride_order = attn_backend.get_kv_cache_stride_order(
                    include_num_layers_dimension=has_layers_dim
                )
                assert len(kv_cache_stride_order) == len(gpu_shape)
            except (AttributeError, NotImplementedError):
                kv_cache_stride_order = tuple(range(len(gpu_shape)))

            # permute test_shape according to stride_order
            test_shape = tuple(test_shape[i] for i in kv_cache_stride_order)

            # find block_size (16) dimension index
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
        num_cpu_kernel_blocks = num_cpu_blocks * cpu_block_size_factor

        # Allocate CPU tensors (pinned memory required for MMA)
        pin_memory = is_pin_memory_available()
        if not pin_memory:
            logger.warning(
                "Pin memory is not available. MMA multi-path transfer "
                "requires pinned memory for optimal performance."
            )
        logger.info(
            "Allocating %d CPU tensors for MMA offloading...",
            len(parsed_gpu_tensors),
        )
        gpu_tensors: list[torch.Tensor] = []
        cpu_tensors: list[torch.Tensor] = []
        for gpu_tensor, split_k_and_v in parsed_gpu_tensors:
            cpu_shape = list(gpu_tensor.shape)
            cpu_shape[1 if split_k_and_v else 0] = num_cpu_kernel_blocks

            logger.debug("Allocating CPU tensor of shape %r", cpu_shape)
            cpu_tensor = torch.zeros(
                cpu_shape,
                dtype=gpu_tensor.dtype,
                device="cpu",
                pin_memory=pin_memory,
            )

            gpu_tensors.extend(
                gpu_tensor.unbind(0) if split_k_and_v else [gpu_tensor]
            )
            cpu_tensors.extend(
                cpu_tensor.unbind(0) if split_k_and_v else [cpu_tensor]
            )

        self.gpu_to_cpu_handler = MmaSingleDirectionOffloadingHandler(
            src_tensors=gpu_tensors,
            dst_tensors=cpu_tensors,
            src_block_size_factor=gpu_block_size_factor,
            dst_block_size_factor=cpu_block_size_factor,
            mma_config_path=mma_config_path,
        )

        self.cpu_to_gpu_handler = MmaSingleDirectionOffloadingHandler(
            src_tensors=cpu_tensors,
            dst_tensors=gpu_tensors,
            src_block_size_factor=cpu_block_size_factor,
            dst_block_size_factor=gpu_block_size_factor,
            mma_config_path=mma_config_path,
        )
