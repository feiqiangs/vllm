# SPDX-License-Identifier: Apache-2.0
"""
MMA Relay Daemon Client  (runs inside each DP rank process)

Responsibilities:
  1. Startup:  register per-layer kv_cache IPC handles with daemon
               obtain pinned CPU pool mapping (mmap shm)
               wait for daemon_ready_flag
  2. Runtime:  push transfer requests via lock-free ring buffer
               poll completion ring buffer for finished jobs
  3. Shutdown: cleanup mmap handles

Address translation (hot path):
  GPU side : rank passes (layer_idx, gpu_byte_offset)
             daemon maps = kv_layer_ptrs[rank][layer_idx] + gpu_byte_offset
             Each layer_ptr is opened independently via cudaIpcOpenMemHandle,
             corresponding to a single cudaMalloc allocation in the rank process.
             This matches vLLM's KV cache layout where each layer is allocated
             independently (uniform layout) or in groups (hybrid layout).
  CPU side : client allocates slots from cpu_pool (bump allocator)
             daemon writes/reads pinned memory at cpu_pool_ptr + offset
             rank reads results from mmap'd cpu_pool shm at same offset
"""

from __future__ import annotations

import ctypes
import mmap
import os
import struct
import threading
import time
from typing import Optional

import torch
import torch.cuda

from vllm.logger import init_logger
from vllm.v1.kv_offload.worker.mma_daemon import (
    _META_SIZE,
    _META_OFF_NUM_RANKS,
    _META_OFF_CPU_TOTAL,
    _META_OFF_CPU_SHM_SZ,
    _META_OFF_NUM_LAYERS,
    _META_OFF_RANK_READY,
    _META_OFF_DAEMON_RDY,
    _IPC_HANDLE_SIZE,
    _HANDLE_ENTRY_SIZE,
    _RING_TOTAL_BYTES,
    _RING_HDR_HEAD,
    _RING_HDR_TAIL,
    _RING_HDR_COMPL_HEAD,
    _RING_HDR_COMPL_TAIL,
    _RING_BODY_OFFSET,
    _RING_COMPL_OFFSET,
    _RING_SLOT_BYTES,
    _RING_SLOTS,
    _SLOT_SIZE,
    _SLOT_OFF_JOB_ID,
    _SLOT_OFF_DIRECTION,
    _SLOT_OFF_STATUS,
    _SLOT_OFF_SRC_GPU,
    _SLOT_OFF_LAYER_IDX,
    _SLOT_OFF_GPU_OFFSET,
    _SLOT_OFF_CPU_OFFSET,
    _SLOT_OFF_SIZE,
    _SHM_META_NAME,
    _SHM_CPU_POOL_NAME,
    _SHM_RING_NAME,
    _SHM_HANDLES_NAME,
    _MAX_LAYERS,
    DIR_D2H,
    DIR_H2D,
    STATUS_PENDING,
    STATUS_DONE,
    STATUS_ERROR,
    _shm_open,
    _shm_create,
    _read_u64,
    _write_u64,
    _get_ipc_handle,
    _slot_base,
)

logger = init_logger(__name__)


class MMADaemonClient:
    """
    Per-DP-rank client that communicates with the MMA Relay Daemon.

    Key change from v1: GPU KV cache is registered per-layer.
    Each layer tensor corresponds to one independent cudaMalloc allocation
    in vLLM (uniform KV cache layout). The daemon maps each layer separately
    via cudaIpcOpenMemHandle, so gpu_offset in each transfer slot is always
    within the bounds of a single allocation — no cross-allocation pointer
    arithmetic, no undefined behavior.

    Typical lifecycle:
        client = MMADaemonClient(rank=local_rank)
        client.register_kv_layers(kv_caches)   # dict[layer_name → Tensor]
        client.attach(timeout_s=30)             # blocks until daemon is ready
        # ... during inference:
        job_id = client.submit_d2h(layer_idx, gpu_byte_offset, cpu_slot_id, size_bytes)
        finished = client.poll_completions()
        client.detach()
    """

    def __init__(self, rank: int):
        """
        Args:
            rank : DP rank index (0..num_ranks-1), also physical GPU id.
        """
        assert rank >= 0
        self.rank = rank

        # Per-layer info (populated by register_kv_layers before attach)
        self._layer_names: list[str] = []           # ordered layer names
        self._layer_tensors: list[torch.Tensor] = []
        self._layer_base_ptrs: list[int] = []       # data_ptr() of each layer
        self._layer_sizes: list[int] = []           # numel * element_size
        self._num_layers: int = 0

        # Shared memory handles (opened at attach time)
        self.meta_mm: Optional[mmap.mmap] = None
        self.ring_mm: Optional[mmap.mmap] = None
        self.cpu_pool_mm: Optional[mmap.mmap] = None

        # CPU pool slot allocator (simple bump allocator)
        self._cpu_pool_size: int = 0
        self._cpu_pool_lock = threading.Lock()
        self._cpu_slot_next: int = 0

        # Job tracking
        self._pending_jobs: dict[int, int] = {}   # job_id → cpu_offset
        self._job_counter: int = 0

        self._attached = False

    # ------------------------------------------------------------------
    # Layer registration (must call before attach)
    # ------------------------------------------------------------------

    def register_kv_layers(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """
        Register per-layer KV cache tensors.

        Each tensor must be:
          - On CUDA (this rank's GPU)
          - The direct result of cudaMalloc (i.e. the raw allocation tensor from
            vLLM's _allocate_kv_cache, before reshape/permute).
            If you only have the reshaped tensors (post _reshape_kv_cache),
            pass the contiguous base tensor for each layer.

        Args:
            kv_caches: dict mapping layer_name → kv_cache tensor.
                       Ordering is preserved; layer_idx in transfer slots
                       corresponds to the insertion order of this dict.
        """
        assert not self._attached, "register_kv_layers must be called before attach()"
        assert len(kv_caches) > 0, "kv_caches must not be empty"
        assert len(kv_caches) <= _MAX_LAYERS, \
            f"Too many layers ({len(kv_caches)} > _MAX_LAYERS={_MAX_LAYERS})"

        self._layer_names = list(kv_caches.keys())
        self._layer_tensors = []
        self._layer_base_ptrs = []
        self._layer_sizes = []

        for layer_name, tensor in kv_caches.items():
            assert tensor.is_cuda, f"Layer {layer_name}: tensor must be on CUDA"
            # Get the contiguous base for IPC registration.
            # cudaIpcGetMemHandle requires the pointer returned by cudaMalloc;
            # for non-contiguous views we use the storage data_ptr (base of alloc).
            base_ptr = tensor.storage().data_ptr()
            size_bytes = tensor.storage().nbytes()
            self._layer_tensors.append(tensor)
            self._layer_base_ptrs.append(base_ptr)
            self._layer_sizes.append(size_bytes)

        self._num_layers = len(self._layer_names)
        logger.info(
            "[Rank %d] Registered %d KV cache layers for IPC export",
            self.rank, self._num_layers,
        )

    def layer_index(self, layer_name: str) -> int:
        """Return the layer_idx for a given layer_name (for use in submit_*)."""
        return self._layer_names.index(layer_name)

    # ------------------------------------------------------------------
    # Attach / detach
    # ------------------------------------------------------------------

    def attach(self, timeout_s: float = 60.0) -> None:
        """
        Connect to the running daemon:
          1. Open meta + ring shm
          2. Write per-layer IPC handles into handles shm
          3. Set rank_ready_flag
          4. Wait for daemon_ready_flag
          5. mmap cpu_pool shm
        """
        if self._attached:
            return
        assert self._num_layers > 0, "Must call register_kv_layers() before attach()"

        deadline = time.monotonic() + timeout_s

        # 1. Wait for and open meta shm (daemon must have created it)
        while time.monotonic() < deadline:
            try:
                self.meta_mm = _shm_open(_SHM_META_NAME, _META_SIZE)
                break
            except FileNotFoundError:
                time.sleep(0.05)
        else:
            raise TimeoutError(f"[Rank {self.rank}] Timed out waiting for daemon meta shm")

        # 2. Open ring shm
        self.ring_mm = _shm_open(_SHM_RING_NAME.format(rank=self.rank), _RING_TOTAL_BYTES)

        # 3. Write per-layer IPC handles
        self._register_kv_layers()

        # 4. Wait for daemon to finish mapping all IPC handles
        self._wait_daemon_ready(deadline)

        # 5. mmap cpu_pool
        cpu_pool_size = _read_u64(self.meta_mm, _META_OFF_CPU_SHM_SZ)
        self._cpu_pool_size = cpu_pool_size
        self.cpu_pool_mm = _shm_open(_SHM_CPU_POOL_NAME, cpu_pool_size)

        self._attached = True
        logger.info(
            "[Rank %d] Attached to MMA Relay Daemon. %d layers, CPU pool: %.1f GB",
            self.rank, self._num_layers, cpu_pool_size / 1e9,
        )

    def detach(self) -> None:
        """Close shm mappings."""
        if self.meta_mm:     self.meta_mm.close()
        if self.ring_mm:     self.ring_mm.close()
        if self.cpu_pool_mm: self.cpu_pool_mm.close()
        self._attached = False

    # ------------------------------------------------------------------
    # Transfer submission
    # ------------------------------------------------------------------

    def submit_d2h(
        self,
        layer_idx: int,
        gpu_byte_offset: int,
        cpu_slot_offset: int,
        size_bytes: int,
    ) -> int:
        """
        Submit a GPU→CPU transfer request for one layer.

        Args:
            layer_idx       : index into registered layer list (see layer_index())
            gpu_byte_offset : byte offset within layer tensor's allocation
                              (e.g. block_id * block_stride_bytes)
            cpu_slot_offset : byte offset into shared CPU pool (from alloc_cpu_slot())
            size_bytes      : transfer size in bytes

        Returns job_id.
        """
        return self._push_request(DIR_D2H, layer_idx, gpu_byte_offset, cpu_slot_offset, size_bytes)

    def submit_h2d(
        self,
        layer_idx: int,
        gpu_byte_offset: int,
        cpu_slot_offset: int,
        size_bytes: int,
    ) -> int:
        """Submit a CPU→GPU transfer request. Returns job_id."""
        return self._push_request(DIR_H2D, layer_idx, gpu_byte_offset, cpu_slot_offset, size_bytes)

    def _push_request(
        self,
        direction: int,
        layer_idx: int,
        gpu_offset: int,
        cpu_offset: int,
        size_bytes: int,
    ) -> int:
        """
        Lock-free push to request ring buffer.
        Spins if ring is full (should be rare with 256 slots).
        """
        assert self._attached
        assert self.ring_mm is not None
        assert 0 <= layer_idx < self._num_layers, \
            f"layer_idx={layer_idx} out of range [0, {self._num_layers})"

        mm = self.ring_mm
        job_id = self._next_job_id()

        # Spin until slot available
        while True:
            tail = _read_u64(mm, _RING_HDR_TAIL)
            head = _read_u64(mm, _RING_HDR_HEAD)
            if tail - head < _RING_SLOTS:
                break
            time.sleep(0)

        slot_off = _slot_base(mm, False, tail)

        # Write slot fields. Write status=PENDING last to commit the slot.
        mm.seek(slot_off + _SLOT_OFF_JOB_ID)
        mm.write(struct.pack("<Q", job_id))           # job_id (8 bytes)
        mm.seek(slot_off + _SLOT_OFF_DIRECTION)
        mm.write(struct.pack("BBB", direction, 0, self.rank))   # direction, status_tmp=0, src_gpu
        mm.seek(slot_off + _SLOT_OFF_LAYER_IDX)
        mm.write(struct.pack("<H", layer_idx))        # layer_idx (uint16)
        mm.seek(slot_off + _SLOT_OFF_GPU_OFFSET)
        mm.write(struct.pack("<QQQ", gpu_offset, cpu_offset, size_bytes))

        # Commit: write STATUS_PENDING into status byte
        mm.seek(slot_off + _SLOT_OFF_STATUS)
        mm.write(bytes([STATUS_PENDING]))

        # Advance tail
        _write_u64(mm, _RING_HDR_TAIL, tail + 1)
        mm.flush()

        self._pending_jobs[job_id] = cpu_offset
        return job_id

    # ------------------------------------------------------------------
    # Completion polling
    # ------------------------------------------------------------------

    def poll_completions(self) -> list[tuple[int, bool]]:
        """
        Non-blocking. Returns list of (job_id, success) for all completed jobs
        since last call.
        """
        assert self._attached
        assert self.ring_mm is not None

        mm = self.ring_mm
        results: list[tuple[int, bool]] = []

        compl_head = _read_u64(mm, _RING_HDR_COMPL_HEAD)
        compl_tail = _read_u64(mm, _RING_HDR_COMPL_TAIL)

        while compl_head != compl_tail:
            slot_off = _slot_base(mm, True, compl_head)
            mm.seek(slot_off)
            raw = mm.read(24)
            job_id = struct.unpack_from("<Q", raw, 0)[0]
            status = raw[8]
            ok = (status == STATUS_DONE)
            results.append((job_id, ok))

            if job_id in self._pending_jobs:
                del self._pending_jobs[job_id]

            compl_head += 1

        if results:
            _write_u64(mm, _RING_HDR_COMPL_HEAD, compl_head)
            mm.flush()

        return results

    def wait_for_jobs(self, job_ids: set[int], timeout_s: float = 30.0) -> None:
        """Blocking wait for specific job IDs."""
        remaining = set(job_ids)
        deadline = time.monotonic() + timeout_s
        while remaining and time.monotonic() < deadline:
            for job_id, ok in self.poll_completions():
                remaining.discard(job_id)
            if remaining:
                time.sleep(0)

        if remaining:
            raise TimeoutError(f"[Rank {self.rank}] Timed out waiting for jobs: {remaining}")

    # ------------------------------------------------------------------
    # CPU pool slot allocator
    # ------------------------------------------------------------------

    def alloc_cpu_slot(self, size_bytes: int) -> int:
        """
        Allocate a CPU pool slot. Returns byte offset into shared CPU pool.
        Simple bump allocator; wraps around (circular).
        Thread-safe within one rank process.
        """
        with self._cpu_pool_lock:
            offset = self._cpu_slot_next
            self._cpu_slot_next += size_bytes
            if self._cpu_slot_next > self._cpu_pool_size:
                # Wrap around (simplistic; production code should track in-use slots)
                self._cpu_slot_next = size_bytes
                offset = 0
        return offset

    def get_cpu_tensor(self, cpu_offset: int, size_bytes: int, dtype: torch.dtype) -> torch.Tensor:
        """
        Return a CPU torch.Tensor backed by the shared pinned pool at cpu_offset.
        Uses torch.frombuffer on the mmap'd region (zero-copy).
        """
        assert self.cpu_pool_mm is not None
        self.cpu_pool_mm.seek(cpu_offset)
        raw_bytes = self.cpu_pool_mm.read(size_bytes)
        return torch.frombuffer(raw_bytes, dtype=dtype)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_job_id(self) -> int:
        self._job_counter += 1
        return self._job_counter

    def _register_kv_layers(self) -> None:
        """
        Export per-layer CUDA IPC handles and write them into a dedicated
        handles shm segment. Then set num_layers and rank_ready_flag in meta.

        Shm layout: /mma_relay_handles_{rank}
          num_layers * _HANDLE_ENTRY_SIZE bytes
          Each entry (80 bytes):
            [0:8]   uint64  layer_size_bytes  (storage().nbytes())
            [8:72]  bytes   cudaIpcMemHandle_t (64 bytes)
            [72:80] uint64  reserved
        """
        assert self.meta_mm is not None

        num_layers = self._num_layers
        handles_shm_size = num_layers * _HANDLE_ENTRY_SIZE
        handles_mm = _shm_create(_SHM_HANDLES_NAME.format(rank=self.rank), handles_shm_size)

        for layer_idx, (base_ptr, size_bytes) in enumerate(
            zip(self._layer_base_ptrs, self._layer_sizes)
        ):
            handle_bytes = _get_ipc_handle(base_ptr)
            assert len(handle_bytes) == _IPC_HANDLE_SIZE

            entry_off = layer_idx * _HANDLE_ENTRY_SIZE
            _write_u64(handles_mm, entry_off, size_bytes)          # layer size
            handles_mm.seek(entry_off + 8)
            handles_mm.write(handle_bytes)                          # IPC handle
            # reserved bytes are zero-initialized by shm_create

            logger.info(
                "[Rank %d] Layer %d (%s): exported IPC handle, base_ptr=0x%x, size=%.1f MB",
                self.rank, layer_idx,
                self._layer_names[layer_idx] if layer_idx < len(self._layer_names) else "?",
                base_ptr, size_bytes / 1e6,
            )

        handles_mm.flush()
        handles_mm.close()

        # Write num_layers into meta shm
        _write_u64(self.meta_mm, _META_OFF_NUM_LAYERS + self.rank * 8, num_layers)

        # Set rank_ready_flag
        flag_off = _META_OFF_RANK_READY + self.rank * 8
        _write_u64(self.meta_mm, flag_off, 1)
        self.meta_mm.flush()

        logger.info(
            "[Rank %d] Wrote %d layer IPC handles to shm, set rank_ready_flag",
            self.rank, num_layers,
        )

    def _wait_daemon_ready(self, deadline: float) -> None:
        assert self.meta_mm is not None
        while time.monotonic() < deadline:
            if _read_u64(self.meta_mm, _META_OFF_DAEMON_RDY) == 1:
                logger.info("[Rank %d] Daemon ready signal received.", self.rank)
                return
            time.sleep(0.01)
        raise TimeoutError(f"[Rank {self.rank}] Timed out waiting for daemon_ready_flag")
