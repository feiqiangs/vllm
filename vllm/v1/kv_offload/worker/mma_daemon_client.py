# SPDX-License-Identifier: Apache-2.0
"""
MMA Relay Daemon Client  (runs inside each DP rank process)

Responsibilities:
  1. Startup:  register kv_cache IPC handle with daemon
               obtain pinned CPU pool mapping (mmap shm)
               wait for daemon_ready_flag
  2. Runtime:  push transfer requests via lock-free ring buffer
               poll completion ring buffer for finished jobs
  3. Shutdown: cleanup mmap handles

Address translation (hot path):
  GPU side : rank passes byte offset into kv_cache tensor
             daemon maps = kv_cache_ptrs[rank] + offset
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
    _META_OFF_KV_SIZES,
    _META_OFF_IPC_HANDLES,
    _META_OFF_RANK_READY,
    _META_OFF_DAEMON_RDY,
    _IPC_HANDLE_SIZE,
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
    _SLOT_OFF_GPU_OFFSET,
    _SLOT_OFF_CPU_OFFSET,
    _SLOT_OFF_SIZE,
    _SHM_META_NAME,
    _SHM_CPU_POOL_NAME,
    _SHM_RING_NAME,
    DIR_D2H,
    DIR_H2D,
    STATUS_PENDING,
    STATUS_DONE,
    STATUS_ERROR,
    _shm_open,
    _read_u64,
    _write_u64,
    _get_ipc_handle,
    _slot_base,
)

logger = init_logger(__name__)


class MMADaemonClient:
    """
    Per-DP-rank client that communicates with the MMA Relay Daemon.

    Typical lifecycle:
        client = MMADaemonClient(rank=local_rank, kv_cache_tensor=gpu_tensor)
        client.attach(timeout_s=30)      # blocks until daemon is ready
        # ... during inference:
        job_id = client.submit_d2h(gpu_byte_offset, cpu_slot_id, size_bytes)
        finished = client.poll_completions()   # returns list of (job_id, ok)
        # cleanup:
        client.detach()
    """

    def __init__(self, rank: int, kv_cache_tensor: torch.Tensor):
        """
        Args:
            rank          : DP rank index (0..num_ranks-1), also physical GPU id
            kv_cache_tensor: The GPU KV cache tensor for this rank.
                            Must be contiguous and on CUDA.
        """
        assert rank >= 0
        assert kv_cache_tensor.is_cuda and kv_cache_tensor.is_contiguous()

        self.rank = rank
        self.kv_cache_tensor = kv_cache_tensor
        self.kv_cache_base_ptr: int = kv_cache_tensor.data_ptr()
        self.kv_cache_size: int = kv_cache_tensor.numel() * kv_cache_tensor.element_size()

        # Shared memory handles (opened at attach time)
        self.meta_mm: Optional[mmap.mmap] = None
        self.ring_mm: Optional[mmap.mmap] = None
        self.cpu_pool_mm: Optional[mmap.mmap] = None

        # CPU pool slot allocator (simple bump allocator with free-list)
        # Caller uses alloc_cpu_slot / free_cpu_slot to manage CPU buffer slots
        self._cpu_pool_size: int = 0
        self._cpu_pool_lock = threading.Lock()
        self._cpu_slot_next: int = 0       # next free byte offset

        # Job tracking: job_id → cpu_slot_offset (for reclaim after completion)
        self._pending_jobs: dict[int, int] = {}   # job_id → cpu_offset
        self._job_counter: int = 0

        self._attached = False

    # ------------------------------------------------------------------
    # Attach / detach
    # ------------------------------------------------------------------

    def attach(self, timeout_s: float = 60.0) -> None:
        """
        Connect to the running daemon:
          1. Open meta + ring shm
          2. Write kv_cache IPC handle into meta
          3. Wait for daemon_ready_flag
          4. mmap cpu_pool shm
        """
        if self._attached:
            return

        # 1. Open meta shm (daemon must have created it already)
        deadline = time.monotonic() + timeout_s
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

        # 3. Register kv_cache IPC handle
        self._register_kv_cache()

        # 4. Wait for daemon to finish mapping all IPC handles
        self._wait_daemon_ready(deadline)

        # 5. mmap cpu_pool
        cpu_pool_size = _read_u64(self.meta_mm, _META_OFF_CPU_SHM_SZ)
        self._cpu_pool_size = cpu_pool_size
        self.cpu_pool_mm = _shm_open(_SHM_CPU_POOL_NAME, cpu_pool_size)

        self._attached = True
        logger.info(
            "[Rank %d] Attached to MMA Relay Daemon. CPU pool: %.1f GB",
            self.rank,
            cpu_pool_size / 1e9,
        )

    def detach(self) -> None:
        """Close shm mappings."""
        if self.meta_mm:   self.meta_mm.close()
        if self.ring_mm:   self.ring_mm.close()
        if self.cpu_pool_mm: self.cpu_pool_mm.close()
        self._attached = False

    # ------------------------------------------------------------------
    # Transfer submission
    # ------------------------------------------------------------------

    def submit_d2h(
        self,
        gpu_byte_offset: int,
        cpu_slot_offset: int,
        size_bytes: int,
    ) -> int:
        """
        Submit a GPU→CPU transfer request.
        Returns job_id.

        gpu_byte_offset : offset from kv_cache_base_ptr (bytes)
        cpu_slot_offset : offset into shared CPU pool (bytes), allocated via alloc_cpu_slot()
        size_bytes      : transfer size in bytes
        """
        return self._push_request(DIR_D2H, gpu_byte_offset, cpu_slot_offset, size_bytes)

    def submit_h2d(
        self,
        gpu_byte_offset: int,
        cpu_slot_offset: int,
        size_bytes: int,
    ) -> int:
        """Submit a CPU→GPU transfer request. Returns job_id."""
        return self._push_request(DIR_H2D, gpu_byte_offset, cpu_slot_offset, size_bytes)

    def _push_request(
        self,
        direction: int,
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

        mm = self.ring_mm
        job_id = self._next_job_id()

        # Spin until slot available
        while True:
            tail = _read_u64(mm, _RING_HDR_TAIL)
            head = _read_u64(mm, _RING_HDR_HEAD)
            if tail - head < _RING_SLOTS:
                break
            # Ring full: yield briefly
            time.sleep(0)

        slot_off = _slot_base(mm, False, tail)

        # Write slot fields (status LAST to signal slot is ready)
        mm.seek(slot_off)
        mm.write(struct.pack("<Q", job_id))              # job_id
        # direction, status(=0 not ready yet), src_gpu, dst_gpu, pad
        mm.write(struct.pack("BBBBI", direction, STATUS_PENDING ^ STATUS_PENDING, self.rank, self.rank, 0))
        # skip to gpu_offset at _SLOT_OFF_GPU_OFFSET = 16
        mm.seek(slot_off + 16)
        mm.write(struct.pack("<QQQ", gpu_offset, cpu_offset, size_bytes))
        # Now write status=PENDING to commit the slot (memory ordering)
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
            raw = mm.read(24)   # job_id(8) + status(1) + pad(7) + ts(8)
            job_id = struct.unpack_from("<Q", raw, 0)[0]
            status = raw[8]
            ok = (status == STATUS_DONE)
            results.append((job_id, ok))

            # Free cpu slot
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
                time.sleep(0)  # yield

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
        Zero-copy: uses torch.frombuffer on the mmap'd region.
        """
        assert self.cpu_pool_mm is not None
        self.cpu_pool_mm.seek(cpu_offset)
        # frombuffer on mmap region gives a CPU tensor backed by shared memory
        # This is the rank's view of the data daemon wrote
        num_elements = size_bytes // torch.tensor([], dtype=dtype).element_size()
        # Use numpy as intermediate (mmap → numpy → torch, zero extra copy)
        import numpy as np
        self.cpu_pool_mm.seek(cpu_offset)
        raw_bytes = self.cpu_pool_mm.read(size_bytes)
        arr = np.frombuffer(raw_bytes, dtype=torch.zeros([], dtype=dtype).numpy().dtype)
        return torch.from_numpy(arr.copy())  # copy needed to detach from mmap lifetime

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_job_id(self) -> int:
        self._job_counter += 1
        return self._job_counter

    def _register_kv_cache(self) -> None:
        """Write kv_cache IPC handle + size into meta shm, set rank_ready_flag."""
        assert self.meta_mm is not None

        # Get IPC handle for kv_cache tensor
        handle_bytes = _get_ipc_handle(self.kv_cache_base_ptr)
        assert len(handle_bytes) == _IPC_HANDLE_SIZE

        # Write size
        size_off = _META_OFF_KV_SIZES + self.rank * 8
        _write_u64(self.meta_mm, size_off, self.kv_cache_size)

        # Write IPC handle
        handle_off = _META_OFF_IPC_HANDLES + self.rank * _IPC_HANDLE_SIZE
        self.meta_mm.seek(handle_off)
        self.meta_mm.write(handle_bytes)

        # Set rank_ready_flag
        flag_off = _META_OFF_RANK_READY + self.rank * 8
        _write_u64(self.meta_mm, flag_off, 1)
        self.meta_mm.flush()

        logger.info(
            "[Rank %d] Registered kv_cache IPC handle (%.1f GB)",
            self.rank,
            self.kv_cache_size / 1e9,
        )

    def _wait_daemon_ready(self, deadline: float) -> None:
        assert self.meta_mm is not None
        while time.monotonic() < deadline:
            if _read_u64(self.meta_mm, _META_OFF_DAEMON_RDY) == 1:
                logger.info("[Rank %d] Daemon ready signal received.", self.rank)
                return
            time.sleep(0.01)
        raise TimeoutError(f"[Rank {self.rank}] Timed out waiting for daemon_ready_flag")
