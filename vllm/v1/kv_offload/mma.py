# SPDX-License-Identifier: Apache-2.0
# MMA (Multi-path Memory Access) offloading spec for vLLM
# Uses MMA multi-path transfer engine to accelerate GPU<->CPU KV cache transfers
from collections.abc import Iterator

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.abstract import LoadStoreSpec, OffloadingManager
from vllm.v1.kv_offload.arc_manager import ARCOffloadingManager
from vllm.v1.kv_offload.backends.cpu import CPUBackend
from vllm.v1.kv_offload.lru_manager import LRUOffloadingManager
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec, GPULoadStoreSpec
from vllm.v1.kv_offload.spec import OffloadingSpec
from vllm.v1.kv_offload.worker.mma_cpu_gpu import MmaCpuGpuOffloadingHandlers
from vllm.v1.kv_offload.worker.worker import OffloadingHandler

logger = init_logger(__name__)


class MMAOffloadingSpec(OffloadingSpec):
    """
    Offloading spec that uses MMA (Multi-path Memory Access) engine
    for GPU<->CPU KV cache transfers.

    MMA accelerates data transfers by leveraging NVLink peer-to-peer
    multi-path relay through other GPUs, achieving significantly higher
    bandwidth than single-path DMA.

    Configuration (via kv_connector_extra_config):
        - cpu_bytes_to_use: Total CPU memory budget for offloaded KV cache.
        - block_size: Offloaded block size (default: gpu_block_size).
        - eviction_policy: "lru" or "arc" (default: "lru").
        - mma_config_path: Path to MMA configuration file (optional).
    """

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, kv_cache_config)

        cpu_bytes_to_use = self.extra_config.get("cpu_bytes_to_use")
        if not cpu_bytes_to_use:
            raise Exception(
                "cpu_bytes_to_use must be specified in kv_connector_extra_config"
            )

        # MMA-specific config
        self.mma_config_path: str | None = self.extra_config.get(
            "mma_config_path", None
        )

        # Calculate kv_bytes_per_offloaded_block (same logic as CPUOffloadingSpec)
        assert kv_cache_config is not None
        page_sizes = {
            kv_cache_group.kv_cache_spec.page_size_bytes
            for kv_cache_group in kv_cache_config.kv_cache_groups
        }
        assert len(page_sizes) == 1
        page_size_bytes = page_sizes.pop()
        kv_bytes_per_block = (
            page_size_bytes
            * len(kv_cache_config.kv_cache_tensors)
            * vllm_config.parallel_config.world_size
        )
        kv_bytes_per_offloaded_block = kv_bytes_per_block * (
            self.offloaded_block_size // self.gpu_block_size
        )

        self.num_blocks = (
            int(cpu_bytes_to_use) // kv_bytes_per_offloaded_block
            if kv_bytes_per_offloaded_block > 0
            else 0
        )

        # Scheduler-side (lazy init)
        self._manager: OffloadingManager | None = None

        # Worker-side (lazy init)
        self._handlers: MmaCpuGpuOffloadingHandlers | None = None

        self.eviction_policy: str = self.extra_config.get("eviction_policy", "lru")

        logger.info(
            "MMAOffloadingSpec initialized: num_blocks=%d, "
            "offloaded_block_size=%d, eviction_policy=%s, "
            "mma_config_path=%s",
            self.num_blocks,
            self.offloaded_block_size,
            self.eviction_policy,
            self.mma_config_path,
        )

    def get_manager(self) -> OffloadingManager:
        """
        Get the offloading manager (scheduler-side).
        Reuses the same LRU/ARC managers as CPUOffloadingSpec since
        MMA only changes the transfer engine, not the eviction strategy.
        """
        if not self._manager:
            kv_events_config = self.vllm_config.kv_events_config
            enable_events = (
                kv_events_config is not None
                and kv_events_config.enable_kv_cache_events
            )

            backend = CPUBackend(
                block_size=self.offloaded_block_size, num_blocks=self.num_blocks
            )

            if self.eviction_policy == "lru":
                self._manager = LRUOffloadingManager(
                    backend=backend, enable_events=enable_events
                )
            elif self.eviction_policy == "arc":
                self._manager = ARCOffloadingManager(
                    backend=backend, enable_events=enable_events
                )
            else:
                raise ValueError(
                    f"Unknown eviction policy: {self.eviction_policy}. "
                    f"Supported policies: lru, arc"
                )
        return self._manager

    def get_handlers(
        self,
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ) -> Iterator[
        tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]
    ]:
        """
        Get MMA-based offloading handlers (worker-side).
        Creates MmaCpuGpuOffloadingHandlers which use MMA's multi-path
        transfer engine instead of ops.swap_blocks.
        """
        if not self._handlers:
            if not current_platform.is_cuda_alike():
                raise Exception(
                    "MMA Offloading is currently only supported on CUDA-alike GPUs"
                )

            self._handlers = MmaCpuGpuOffloadingHandlers(
                attn_backends=attn_backends,
                gpu_block_size=self.gpu_block_size,
                cpu_block_size=self.offloaded_block_size,
                num_cpu_blocks=self.num_blocks,
                gpu_caches=kv_caches,
                mma_config_path=self.mma_config_path,
            )

        assert self._handlers is not None
        yield GPULoadStoreSpec, CPULoadStoreSpec, self._handlers.gpu_to_cpu_handler
        yield CPULoadStoreSpec, GPULoadStoreSpec, self._handlers.cpu_to_gpu_handler


# ---------------------------------------------------------------------------
# MMADaemonOffloadingSpec  (方案2 - Relay Daemon variant)
# ---------------------------------------------------------------------------

from vllm.v1.kv_offload.worker.mma_cpu_gpu_daemon import (
    MmaCpuGpuOffloadingHandlers as MmaDaemonCpuGpuOffloadingHandlers,
)


class MMADaemonOffloadingSpec(OffloadingSpec):
    """
    Offloading spec that routes KV cache transfers through a dedicated
    MMA Relay Daemon process (方案2).

    The Daemon process:
      - Sees ALL physical GPUs (launched without CUDA_VISIBLE_DEVICES isolation)
      - Owns all pinned CPU memory (cudaHostAllocPortable)
      - Executes NVLink multi-path relay via MMA engine

    DP ranks communicate with the Daemon via lock-free shared-memory ring
    buffers.  MMA is never imported or initialized in rank processes.

    Configuration (via kv_connector_extra_config):
        - cpu_bytes_to_use  : Total CPU memory budget (required)
        - num_ranks         : Number of DP ranks (required, e.g. 8)
        - rank              : This rank's index (required, 0..num_ranks-1)
        - eviction_policy   : "lru" or "arc" (default: "lru")
        - mma_config_path   : Path to MMA config file (optional)
        - daemon_timeout_s  : Seconds to wait for daemon ready (default: 60)
    """

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, kv_cache_config)

        cpu_bytes_to_use = self.extra_config.get("cpu_bytes_to_use")
        if not cpu_bytes_to_use:
            raise ValueError(
                "cpu_bytes_to_use must be specified in kv_connector_extra_config"
            )

        self.mma_config_path: str | None = self.extra_config.get("mma_config_path")
        self.eviction_policy: str = self.extra_config.get("eviction_policy", "lru")
        self.daemon_timeout_s: float = float(
            self.extra_config.get("daemon_timeout_s", 60.0)
        )

        # rank / num_ranks — DP-aware
        parallel_cfg = vllm_config.parallel_config
        self.num_ranks: int = int(
            self.extra_config.get("num_ranks", parallel_cfg.data_parallel_size or 1)
        )
        self.rank: int = int(
            self.extra_config.get("rank", parallel_cfg.data_parallel_rank or 0)
        )

        # Calculate num_cpu_blocks (same formula as CPUOffloadingSpec)
        assert kv_cache_config is not None
        page_sizes = {
            g.kv_cache_spec.page_size_bytes
            for g in kv_cache_config.kv_cache_groups
        }
        assert len(page_sizes) == 1
        page_size_bytes = page_sizes.pop()
        kv_bytes_per_block = (
            page_size_bytes
            * len(kv_cache_config.kv_cache_tensors)
            * vllm_config.parallel_config.world_size
        )
        kv_bytes_per_offloaded_block = kv_bytes_per_block * (
            self.offloaded_block_size // self.gpu_block_size
        )
        self.num_blocks = (
            int(cpu_bytes_to_use) // kv_bytes_per_offloaded_block
            if kv_bytes_per_offloaded_block > 0
            else 0
        )
        self.cpu_bytes_to_use: int = int(cpu_bytes_to_use)

        self._manager: OffloadingManager | None = None
        self._handlers: MmaDaemonCpuGpuOffloadingHandlers | None = None

        logger.info(
            "MMADaemonOffloadingSpec: rank=%d/%d, num_blocks=%d, "
            "cpu_pool=%.1f GB, eviction=%s",
            self.rank, self.num_ranks, self.num_blocks,
            self.cpu_bytes_to_use / 1e9, self.eviction_policy,
        )

    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            kv_events_config = self.vllm_config.kv_events_config
            enable_events = (
                kv_events_config is not None
                and kv_events_config.enable_kv_cache_events
            )
            backend = CPUBackend(
                block_size=self.offloaded_block_size,
                num_blocks=self.num_blocks,
            )
            if self.eviction_policy == "lru":
                self._manager = LRUOffloadingManager(
                    backend=backend, enable_events=enable_events
                )
            elif self.eviction_policy == "arc":
                self._manager = ARCOffloadingManager(
                    backend=backend, enable_events=enable_events
                )
            else:
                raise ValueError(f"Unknown eviction policy: {self.eviction_policy}")
        return self._manager

    def get_handlers(
        self,
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        if not self._handlers:
            if not current_platform.is_cuda_alike():
                raise RuntimeError(
                    "MMADaemonOffloadingSpec requires a CUDA-alike GPU"
                )
            self._handlers = MmaDaemonCpuGpuOffloadingHandlers(
                rank            = self.rank,
                num_ranks       = self.num_ranks,
                gpu_block_size  = self.gpu_block_size,
                cpu_block_size  = self.offloaded_block_size,
                num_cpu_blocks  = self.num_blocks,
                gpu_caches      = kv_caches,
                attn_backends   = attn_backends,
                cpu_pool_bytes  = self.cpu_bytes_to_use,
                mma_config_path = self.mma_config_path,
                daemon_timeout_s= self.daemon_timeout_s,
            )

        assert self._handlers is not None
        yield GPULoadStoreSpec, CPULoadStoreSpec, self._handlers.gpu_to_cpu_handler
        yield CPULoadStoreSpec, GPULoadStoreSpec, self._handlers.cpu_to_gpu_handler
