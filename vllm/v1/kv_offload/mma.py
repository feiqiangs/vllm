# SPDX-License-Identifier: Apache-2.0
"""
MMA (Multi-path Memory Access) offloading spec for vLLM.

Uses MMA multi-path transfer engine to accelerate GPU<->CPU KV cache
transfers via NVLink peer-to-peer relay through other GPUs.

Configuration (via kv_connector_extra_config):
    - cpu_bytes_to_use  : Total CPU memory budget (required)
    - num_ranks         : Number of DP ranks (optional, auto-detected)
    - rank              : This rank's index (optional, auto-detected)
    - eviction_policy   : "lru" or "arc" (default: "lru")
    - mma_config_path   : Path to MMA config file (optional)
    - timeout_s         : Seconds to wait for MMA ready (default: 60)
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

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
from vllm.v1.kv_offload.worker.worker import OffloadingHandler

logger = init_logger(__name__)


class MMAOffloadingSpec(OffloadingSpec):
    """
    MMA offloading spec that routes KV cache transfers through
    the MMA multi-path transfer engine.

    DP ranks interact with MMA exclusively via MMAClient, which handles
    all initialization and transfer coordination internally.
    """

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, kv_cache_config)

        cpu_bytes_to_use = self.extra_config.get("cpu_bytes_to_use")
        if not cpu_bytes_to_use:
            raise ValueError(
                "cpu_bytes_to_use must be specified in kv_connector_extra_config"
            )

        self.mma_config_path: str | None = self.extra_config.get(
            "mma_config_path"
        )
        self.eviction_policy: str = self.extra_config.get(
            "eviction_policy", "lru"
        )
        self.timeout_s: float = float(
            self.extra_config.get("timeout_s", 60.0)
        )

        # rank / num_ranks — DP-aware
        parallel_cfg = vllm_config.parallel_config
        self.num_ranks: int = int(
            self.extra_config.get(
                "num_ranks", parallel_cfg.data_parallel_size or 1
            )
        )
        self.rank: int = int(
            self.extra_config.get(
                "rank", parallel_cfg.data_parallel_rank or 0
            )
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

        # Compute global physical device IDs for this rank.
        # In DP=4 TP=2 on 8 GPUs, rank 1 might use local devices [0,1]
        # which map to physical devices [2,3].  We pass these explicitly
        # to MMAClient so the Daemon opens IPC handles on the correct GPUs,
        # regardless of whether CUDA_VISIBLE_DEVICES is set.
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.device_ids: list[int] = [
            current_platform.device_id_to_physical_device_id(i)
            for i in range(tp_size)
        ]

        self._manager: OffloadingManager | None = None
        self._handlers: MmaCpuGpuOffloadingHandlers | None = None  # noqa: F821

        logger.info(
            "MMAOffloadingSpec: rank=%d/%d, num_blocks=%d, "
            "cpu_pool=%.1f GB, eviction=%s, device_ids=%s",
            self.rank,
            self.num_ranks,
            self.num_blocks,
            self.cpu_bytes_to_use / 1e9,
            self.eviction_policy,
            self.device_ids,
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
                raise ValueError(
                    f"Unknown eviction policy: {self.eviction_policy}"
                )
        return self._manager

    def get_handlers(
        self,
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ) -> Iterator[
        tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]
    ]:
        if not self._handlers:
            if not current_platform.is_cuda_alike():
                raise RuntimeError(
                    "MMAOffloadingSpec requires a CUDA-alike GPU"
                )
            # Lazy import: mma_handler depends on the external mma_relay
            # package which is provided by MMA.
            from vllm.v1.kv_offload.worker.mma_handler import (
                MmaCpuGpuOffloadingHandlers,
            )
            self._handlers = MmaCpuGpuOffloadingHandlers(
                rank=self.rank,
                num_ranks=self.num_ranks,
                gpu_block_size=self.gpu_block_size,
                cpu_block_size=self.offloaded_block_size,
                num_cpu_blocks=self.num_blocks,
                gpu_caches=kv_caches,
                attn_backends=attn_backends,
                cpu_pool_bytes=self.cpu_bytes_to_use,
                mma_config_path=self.mma_config_path,
                timeout_s=self.timeout_s,
                device_ids=self.device_ids,
            )

        assert self._handlers is not None
        yield (
            GPULoadStoreSpec,
            CPULoadStoreSpec,
            self._handlers.gpu_to_cpu_handler,
        )
        yield (
            CPULoadStoreSpec,
            GPULoadStoreSpec,
            self._handlers.cpu_to_gpu_handler,
        )
