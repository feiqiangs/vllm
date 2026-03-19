# SPDX-License-Identifier: Apache-2.0
"""
MMA Relay Daemon (方案2 - Relay Daemon)

Architecture:
  - One Daemon process sees ALL physical GPUs (no CUDA_VISIBLE_DEVICES isolation)
  - Each DP rank registers per-layer GPU KV cache IPC handles at startup
  - Daemon owns and manages all pinned CPU memory
  - DP ranks communicate via lock-free shared-memory ring buffers (control plane)
  - Daemon executes NVLink relay transfers (data plane) using MMA engine

Memory model:
  Control plane  : POSIX shm  ring buffers (one per rank, 64-byte slots)
  GPU data plane : CUDA IPC   - rank exports per-layer handles → daemon maps each
  CPU data plane : cudaHostAlloc(Portable) in daemon → mmap(MAP_SHARED) to ranks

Performance targets:
  Control plane latency : < 500 ns  (atomic CAS, no syscall in hot path)
  Transfer bandwidth    : 150-200 GB/s  (MMA NVLink relay, 8×H20)
"""

from __future__ import annotations

import ctypes
import mmap
import os
import struct
import time
import threading
import multiprocessing as mp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.cuda

from vllm.logger import init_logger

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_DP_RANKS      = 8
_MAX_LAYERS        = 256          # max number of KV cache layers per rank
_RING_SLOTS        = 256          # must be power-of-2
_RING_SLOT_BYTES   = 64           # one cacheline per slot
_SHM_RING_NAME     = "/mma_relay_ring_{rank}"
_SHM_CPU_POOL_NAME = "/mma_relay_cpu_pool"
_SHM_META_NAME     = "/mma_relay_meta"

# slot directions
DIR_D2H = 0
DIR_H2D = 1

# slot status (written into slot.status by daemon on completion)
STATUS_FREE      = 0
STATUS_PENDING   = 1
STATUS_DONE      = 2
STATUS_ERROR     = 3

# ---------------------------------------------------------------------------
# Shared-memory ring buffer slot  (64 bytes, one cacheline)
# ---------------------------------------------------------------------------
# Layout (all little-endian):
#   0  : uint64  job_id
#   8  : uint8   direction  (DIR_D2H / DIR_H2D)
#   9  : uint8   status     (STATUS_*)
#  10  : uint8   src_gpu    (physical GPU id)
#  11  : uint8   dst_gpu    (physical GPU id, same as src_gpu for D2H)
#  12  : uint16  layer_idx  (index into per-rank layer handle table)
#  14  : uint16  _pad
#  16  : uint64  gpu_offset  (byte offset within the layer's IPC mapping)
#  24  : uint64  cpu_slot_offset (byte offset into shared pinned CPU pool)
#  32  : uint64  size_bytes
#  40  : uint64  completion_ts_ns  (filled by daemon on done)
#  48  : uint64  _reserved[2]

_SLOT_SIZE = 64

_SLOT_OFF_JOB_ID      = 0
_SLOT_OFF_DIRECTION   = 8
_SLOT_OFF_STATUS      = 9
_SLOT_OFF_SRC_GPU     = 10
_SLOT_OFF_DST_GPU     = 11
_SLOT_OFF_LAYER_IDX   = 12   # uint16, NEW: index into layer handle table
_SLOT_OFF_GPU_OFFSET  = 16
_SLOT_OFF_CPU_OFFSET  = 24
_SLOT_OFF_SIZE        = 32
_SLOT_OFF_TS          = 40

# Ring buffer header (one page = 4096 bytes, cacheline-aligned head/tail)
_RING_HDR_SIZE    = 4096
_RING_HDR_HEAD    = 0    # uint64, daemon read ptr (consume index)
_RING_HDR_TAIL    = 64   # uint64, rank write ptr  (produce index)
_RING_HDR_COMPL_HEAD = 128  # uint64, rank read ptr for completion ring
_RING_HDR_COMPL_TAIL = 192  # uint64, daemon write ptr for completion ring
_RING_BODY_OFFSET = _RING_HDR_SIZE
_RING_COMPL_OFFSET = _RING_BODY_OFFSET + _RING_SLOTS * _SLOT_SIZE
_RING_TOTAL_BYTES  = _RING_COMPL_OFFSET + _RING_SLOTS * _SLOT_SIZE

# ---------------------------------------------------------------------------
# Meta shared memory: per-layer IPC handle registry + CPU pool base
# ---------------------------------------------------------------------------
# Layout:
#   0   : uint32  num_ranks
#   4   : uint32  _pad
#   8   : uint64  cpu_pool_total_bytes
#  16   : uint64  cpu_pool_shm_size
#  24   : uint64[8]  num_layers_per_rank      (filled by each rank at registration)
#  88   : uint64[8]  rank_ready_flags         (set to 1 when rank has written all handles)
# 152   : uint64    daemon_ready_flag          (set to 1 when daemon finished mapping)
# 216   : (reserved / padding to 4096)
#
# Per-layer IPC handles are stored in a separate shm segment:
#   /mma_relay_handles_{rank}   size = _MAX_LAYERS * (_IPC_HANDLE_SIZE + 16)
#   Each entry (80 bytes):
#     0  : uint64  layer_size_bytes
#     8  : uint8[64]  ipc_handle  (cudaIpcMemHandle_t)
#    72  : uint64  _reserved
#
# This design allows an arbitrary number of layers without inflating the
# fixed-size meta shm.

_META_SIZE              = 4096
_META_OFF_NUM_RANKS     = 0
_META_OFF_CPU_TOTAL     = 8
_META_OFF_CPU_SHM_SZ    = 16
_META_OFF_NUM_LAYERS    = 24     # 8 × uint64  (one per rank)
_META_OFF_RANK_READY    = 88     # 8 × uint64
_META_OFF_DAEMON_RDY    = 152    # uint64

_IPC_HANDLE_SIZE        = 64     # sizeof(cudaIpcMemHandle_t)
_HANDLE_ENTRY_SIZE      = 80     # layer_size(8) + ipc_handle(64) + reserved(8)
_SHM_HANDLES_NAME       = "/mma_relay_handles_{rank}"

# ---------------------------------------------------------------------------
# Low-level helpers  (ctypes / mmap wrappers)
# ---------------------------------------------------------------------------

def _shm_create(name: str, size: int) -> mmap.mmap:
    """Create and zero-fill a POSIX shared memory segment."""
    path = f"/dev/shm{name}"
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o600)
    os.ftruncate(fd, size)
    mm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    os.close(fd)
    return mm


def _shm_open(name: str, size: int) -> mmap.mmap:
    """Open an existing POSIX shared memory segment (read-write)."""
    path = f"/dev/shm{name}"
    fd = os.open(path, os.O_RDWR)
    mm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    os.close(fd)
    return mm


def _shm_unlink(name: str) -> None:
    path = f"/dev/shm{name}"
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def _read_u64(mm: mmap.mmap, offset: int) -> int:
    mm.seek(offset)
    return struct.unpack_from("<Q", mm.read(8))[0]


def _write_u64(mm: mmap.mmap, offset: int, value: int) -> None:
    mm.seek(offset)
    mm.write(struct.pack("<Q", value))


def _read_u8(mm: mmap.mmap, offset: int) -> int:
    mm.seek(offset)
    return struct.unpack_from("B", mm.read(1))[0]


def _write_u8(mm: mmap.mmap, offset: int, value: int) -> None:
    mm.seek(offset)
    mm.write(struct.pack("B", value))


def _slot_base(ring_mm: mmap.mmap, is_completion: bool, idx: int) -> int:
    """Byte offset of slot[idx] in request ring or completion ring."""
    base = _RING_COMPL_OFFSET if is_completion else _RING_BODY_OFFSET
    return base + (idx % _RING_SLOTS) * _SLOT_SIZE


# ---------------------------------------------------------------------------
# CUDA IPC helpers  (via ctypes → libcudart)
# ---------------------------------------------------------------------------

_libcuda: Optional[ctypes.CDLL] = None

def _load_libcuda() -> ctypes.CDLL:
    global _libcuda
    if _libcuda is None:
        _libcuda = ctypes.CDLL("libcuda.so.1", use_errno=True)
    return _libcuda


def _get_ipc_handle(device_ptr: int) -> bytes:
    """
    Call cudaIpcGetMemHandle and return the 64-byte handle as bytes.
    Must be called in the DP rank process that owns the pointer.
    device_ptr must be the base pointer returned by cudaMalloc (i.e. tensor.data_ptr()
    of the original allocation, NOT a pointer into the middle of a slab).
    """
    lib = ctypes.CDLL("libcudart.so", use_errno=True)
    handle_buf = ctypes.create_string_buffer(_IPC_HANDLE_SIZE)
    ret = lib.cudaIpcGetMemHandle(handle_buf, ctypes.c_void_p(device_ptr))
    if ret != 0:
        raise RuntimeError(f"cudaIpcGetMemHandle failed: error {ret} for ptr=0x{device_ptr:x}")
    return bytes(handle_buf)


def _open_ipc_handle(handle_bytes: bytes, gpu_id: int) -> int:
    """
    Call cudaIpcOpenMemHandle on *gpu_id* and return the mapped device pointer.
    Must be called in the Daemon process (which sees all GPUs).
    Returns the base pointer of the mapped region (= cudaMalloc base of rank's tensor).
    """
    lib = ctypes.CDLL("libcudart.so", use_errno=True)
    handle_buf = ctypes.create_string_buffer(handle_bytes, _IPC_HANDLE_SIZE)
    mapped_ptr = ctypes.c_void_p()
    # cudaIpcMemLazyEnablePeerAccess = 1
    ret = lib.cudaIpcOpenMemHandle(
        ctypes.byref(mapped_ptr),
        handle_buf,
        ctypes.c_uint(1),
    )
    if ret != 0:
        raise RuntimeError(
            f"cudaIpcOpenMemHandle failed for GPU {gpu_id}: error {ret}"
        )
    return mapped_ptr.value  # type: ignore[return-value]


def _close_ipc_handle(mapped_ptr: int) -> None:
    lib = ctypes.CDLL("libcudart.so", use_errno=True)
    lib.cudaIpcCloseMemHandle(ctypes.c_void_p(mapped_ptr))


def _cuda_host_alloc_portable(size: int) -> int:
    """
    Allocate page-locked host memory with cudaHostAllocPortable flag.
    Returns pointer as int. Must be called in Daemon process.
    cudaHostAllocPortable = 0x04
    """
    lib = ctypes.CDLL("libcudart.so", use_errno=True)
    ptr = ctypes.c_void_p()
    ret = lib.cudaHostAlloc(ctypes.byref(ptr), ctypes.c_size_t(size), ctypes.c_uint(0x04))
    if ret != 0:
        raise RuntimeError(f"cudaHostAlloc(Portable) failed: error {ret}")
    return ptr.value  # type: ignore[return-value]


def _cuda_host_free(ptr: int) -> None:
    lib = ctypes.CDLL("libcudart.so", use_errno=True)
    lib.cudaFreeHost(ctypes.c_void_p(ptr))


# ---------------------------------------------------------------------------
# DaemonProcess  (runs as a separate OS process, owns all 8 GPUs)
# ---------------------------------------------------------------------------

class MMARelayDaemon:
    """
    Relay Daemon main class. Instantiate and call run() in a dedicated process.

    Startup protocol:
      1. Create /dev/shm/mma_relay_meta, /dev/shm/mma_relay_ring_N (N=0..num_ranks-1)
      2. Allocate pinned CPU pool, create /dev/shm/mma_relay_cpu_pool
      3. Wait for all rank_ready_flags → then IpcOpenMemHandle for each layer of each rank
      4. Write daemon_ready_flag = 1 → ranks can start sending requests
      5. Enter main poll loop: drain ring buffers, dispatch MMA transfers

    Per-layer handle layout (kv_layer_ptrs):
      kv_layer_ptrs[rank][layer_idx] = daemon-side mapped pointer for that layer
      kv_layer_sizes[rank][layer_idx] = size in bytes of that layer tensor
    """

    def __init__(self, num_ranks: int, cpu_pool_bytes: int, mma_config_path: Optional[str] = None):
        self.num_ranks = num_ranks
        self.cpu_pool_bytes = cpu_pool_bytes
        self.mma_config_path = mma_config_path

        # Allocated at startup
        self.cpu_pool_ptr: int = 0
        self.cpu_pool_mm: Optional[mmap.mmap] = None

        self.meta_mm: Optional[mmap.mmap] = None
        self.ring_mms: list[mmap.mmap] = []

        # Per-rank, per-layer mapped pointers
        # kv_layer_ptrs[rank] = list of daemon-side mapped ptrs (one per layer)
        self.kv_layer_ptrs: list[list[int]] = [[] for _ in range(num_ranks)]
        self.kv_layer_sizes: list[list[int]] = [[] for _ in range(num_ranks)]

        # MMA import (only in daemon process)
        self._mma = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Create shm, allocate CPU pool, init MMA, wait for ranks."""
        self._create_shm()
        self._alloc_cpu_pool()
        self._init_mma()
        self._wait_for_ranks()
        self._map_ipc_handles()
        self._signal_ready()
        logger.info("[Daemon] Ready. Entering relay loop.")

    def run(self) -> None:
        """Blocking main loop. Call after start()."""
        self.start()
        self._poll_loop()

    def shutdown(self) -> None:
        logger.info("[Daemon] Shutting down.")
        for rank in range(self.num_ranks):
            for ptr in self.kv_layer_ptrs[rank]:
                if ptr:
                    _close_ipc_handle(ptr)
        if self.cpu_pool_ptr:
            _cuda_host_free(self.cpu_pool_ptr)
        _shm_unlink(_SHM_META_NAME)
        _shm_unlink(_SHM_CPU_POOL_NAME)
        for rank in range(self.num_ranks):
            _shm_unlink(_SHM_RING_NAME.format(rank=rank))
            _shm_unlink(_SHM_HANDLES_NAME.format(rank=rank))

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _create_shm(self) -> None:
        # Meta
        self.meta_mm = _shm_create(_SHM_META_NAME, _META_SIZE)
        _write_u64(self.meta_mm, _META_OFF_NUM_RANKS, self.num_ranks)
        _write_u64(self.meta_mm, _META_OFF_CPU_TOTAL, self.cpu_pool_bytes)

        # Ring buffers (one per rank)
        for rank in range(self.num_ranks):
            mm = _shm_create(_SHM_RING_NAME.format(rank=rank), _RING_TOTAL_BYTES)
            self.ring_mms.append(mm)

        logger.info("[Daemon] Created %d ring shm segments + meta", self.num_ranks)

    def _alloc_cpu_pool(self) -> None:
        self.cpu_pool_ptr = _cuda_host_alloc_portable(self.cpu_pool_bytes)
        cpu_pool_mm = _shm_create(_SHM_CPU_POOL_NAME, self.cpu_pool_bytes)
        self.cpu_pool_mm = cpu_pool_mm

        assert self.meta_mm is not None
        _write_u64(self.meta_mm, _META_OFF_CPU_TOTAL, self.cpu_pool_bytes)
        _write_u64(self.meta_mm, _META_OFF_CPU_SHM_SZ, self.cpu_pool_bytes)

        logger.info(
            "[Daemon] Pinned CPU pool: %.1f GB at 0x%x",
            self.cpu_pool_bytes / 1e9,
            self.cpu_pool_ptr,
        )

    def _init_mma(self) -> None:
        try:
            import mma as _mma_mod
            _mma_mod.init(self.mma_config_path)
            self._mma = _mma_mod
            logger.info("[Daemon] MMA initialized with %d GPUs visible", self._get_gpu_count())
        except Exception as e:
            raise RuntimeError(f"[Daemon] MMA init failed: {e}") from e

    def _get_gpu_count(self) -> int:
        lib = ctypes.CDLL("libcudart.so", use_errno=True)
        count = ctypes.c_int(0)
        lib.cudaGetDeviceCount(ctypes.byref(count))
        return count.value

    def _wait_for_ranks(self, timeout_s: float = 120.0) -> None:
        assert self.meta_mm is not None
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            all_ready = True
            for rank in range(self.num_ranks):
                flag_off = _META_OFF_RANK_READY + rank * 8
                if _read_u64(self.meta_mm, flag_off) == 0:
                    all_ready = False
                    break
            if all_ready:
                logger.info("[Daemon] All %d ranks registered.", self.num_ranks)
                return
            time.sleep(0.01)
        raise TimeoutError("[Daemon] Timed out waiting for rank registrations")

    def _map_ipc_handles(self) -> None:
        """
        For each rank, open the per-layer handles shm and call
        cudaIpcOpenMemHandle for every layer independently.
        This is the correct approach: each layer was allocated via a separate
        cudaMalloc call in vLLM, so each needs its own IPC handle.
        """
        assert self.meta_mm is not None
        for rank in range(self.num_ranks):
            num_layers = int(_read_u64(self.meta_mm, _META_OFF_NUM_LAYERS + rank * 8))
            handles_shm_size = num_layers * _HANDLE_ENTRY_SIZE
            handles_mm = _shm_open(_SHM_HANDLES_NAME.format(rank=rank), handles_shm_size)

            gpu_id = rank  # daemon sees physical GPU ids without CUDA_VISIBLE_DEVICES
            layer_ptrs: list[int] = []
            layer_sizes: list[int] = []

            for layer_idx in range(num_layers):
                entry_off = layer_idx * _HANDLE_ENTRY_SIZE
                layer_size = _read_u64(handles_mm, entry_off)
                handles_mm.seek(entry_off + 8)
                handle_bytes = handles_mm.read(_IPC_HANDLE_SIZE)

                mapped_ptr = _open_ipc_handle(handle_bytes, gpu_id)
                layer_ptrs.append(mapped_ptr)
                layer_sizes.append(layer_size)
                logger.info(
                    "[Daemon] Rank %d layer %d: mapped at 0x%x (%.1f MB)",
                    rank, layer_idx, mapped_ptr, layer_size / 1e6,
                )

            self.kv_layer_ptrs[rank] = layer_ptrs
            self.kv_layer_sizes[rank] = layer_sizes
            handles_mm.close()
            logger.info("[Daemon] Rank %d: %d layers mapped.", rank, num_layers)

    def _signal_ready(self) -> None:
        assert self.meta_mm is not None
        _write_u64(self.meta_mm, _META_OFF_DAEMON_RDY, 1)
        self.meta_mm.flush()
        logger.info("[Daemon] daemon_ready_flag set. Ranks can proceed.")

    # ------------------------------------------------------------------
    # Main poll loop
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        assert self._mma is not None
        idle_count = 0

        while True:
            did_work = False
            for rank in range(self.num_ranks):
                dispatched = self._drain_ring(rank)
                if dispatched:
                    did_work = True
                    idle_count = 0

            if not did_work:
                idle_count += 1
                if idle_count > 100:
                    time.sleep(0.0001)  # 100μs backoff when truly idle

    def _drain_ring(self, rank: int) -> bool:
        """Read pending slots from rank's request ring. Returns True if any processed."""
        mm = self.ring_mms[rank]
        head = _read_u64(mm, _RING_HDR_HEAD)
        tail = _read_u64(mm, _RING_HDR_TAIL)

        if head == tail:
            return False

        did_work = False
        while head != tail:
            slot_off = _slot_base(mm, False, head)

            mm.seek(slot_off)
            raw = mm.read(_SLOT_SIZE)
            job_id     = struct.unpack_from("<Q", raw, _SLOT_OFF_JOB_ID)[0]
            direction  = raw[_SLOT_OFF_DIRECTION]
            status     = raw[_SLOT_OFF_STATUS]
            layer_idx  = struct.unpack_from("<H", raw, _SLOT_OFF_LAYER_IDX)[0]
            gpu_offset = struct.unpack_from("<Q", raw, _SLOT_OFF_GPU_OFFSET)[0]
            cpu_offset = struct.unpack_from("<Q", raw, _SLOT_OFF_CPU_OFFSET)[0]
            size_bytes = struct.unpack_from("<Q", raw, _SLOT_OFF_SIZE)[0]

            if status != STATUS_PENDING:
                break

            ok = self._dispatch_transfer(
                rank=rank,
                job_id=job_id,
                direction=direction,
                layer_idx=layer_idx,
                gpu_offset=gpu_offset,
                cpu_offset=cpu_offset,
                size_bytes=size_bytes,
            )

            self._write_completion(rank, job_id, STATUS_DONE if ok else STATUS_ERROR)

            head += 1
            _write_u64(mm, _RING_HDR_HEAD, head)
            did_work = True

        return did_work

    def _dispatch_transfer(
        self,
        rank: int,
        job_id: int,
        direction: int,
        layer_idx: int,
        gpu_offset: int,
        cpu_offset: int,
        size_bytes: int,
    ) -> bool:
        """
        Issue a MMA batch transfer.

        GPU pointer = kv_layer_ptrs[rank][layer_idx] + gpu_offset
          - kv_layer_ptrs[rank][layer_idx] is the daemon-side base ptr of that
            layer's cudaMalloc allocation (obtained via cudaIpcOpenMemHandle).
          - gpu_offset is the byte offset within that single layer tensor
            (e.g. block_id * block_stride), always within bounds of that allocation.

        CPU pointer = cpu_pool_ptr + cpu_offset
        """
        assert self._mma is not None
        assert self.cpu_pool_mm is not None

        if layer_idx >= len(self.kv_layer_ptrs[rank]):
            logger.error(
                "[Daemon] Invalid layer_idx=%d for rank=%d (num_layers=%d)",
                layer_idx, rank, len(self.kv_layer_ptrs[rank]),
            )
            return False

        layer_base_ptr = self.kv_layer_ptrs[rank][layer_idx]
        gpu_ptr = layer_base_ptr + gpu_offset
        cpu_ptr = self.cpu_pool_ptr + cpu_offset

        try:
            if direction == DIR_D2H:
                self._mma.batch_d2h_async(
                    [cpu_ptr], [gpu_ptr], [size_bytes], stream=None
                )
            else:
                self._mma.batch_h2d_async(
                    [gpu_ptr], [cpu_ptr], [size_bytes], stream=None
                )
            return True
        except Exception as e:
            logger.error("[Daemon] Transfer failed rank=%d job=%d layer=%d: %s",
                         rank, job_id, layer_idx, e)
            return False

    def _write_completion(self, rank: int, job_id: int, status: int) -> None:
        """Push a completion notification into the rank's completion ring."""
        mm = self.ring_mms[rank]
        compl_tail = _read_u64(mm, _RING_HDR_COMPL_TAIL)
        slot_off = _slot_base(mm, True, compl_tail)

        mm.seek(slot_off)
        ts = time.time_ns()
        mm.write(struct.pack("<Q", job_id))
        mm.write(bytes([status, 0, 0, 0, 0, 0, 0, 0]))
        mm.write(struct.pack("<Q", ts))

        compl_tail += 1
        _write_u64(mm, _RING_HDR_COMPL_TAIL, compl_tail)
        mm.flush()


# ---------------------------------------------------------------------------
# Helper to launch the daemon as a subprocess
# ---------------------------------------------------------------------------

def _daemon_entry(num_ranks: int, cpu_pool_bytes: int, mma_config_path: Optional[str]) -> None:
    """Entry point for daemon subprocess (spawn context)."""
    daemon = MMARelayDaemon(num_ranks, cpu_pool_bytes, mma_config_path)
    try:
        daemon.run()
    except KeyboardInterrupt:
        pass
    finally:
        daemon.shutdown()


def launch_daemon(
    num_ranks: int,
    cpu_pool_bytes: int,
    mma_config_path: Optional[str] = None,
) -> mp.Process:
    """
    Spawn the MMA Relay Daemon as a background process.
    Returns the Process object. Caller is responsible for terminating it.
    """
    proc = mp.Process(
        target=_daemon_entry,
        args=(num_ranks, cpu_pool_bytes, mma_config_path),
        daemon=True,
        name="mma-relay-daemon",
    )
    proc.start()
    logger.info("MMA Relay Daemon started (pid=%d)", proc.pid)
    return proc
