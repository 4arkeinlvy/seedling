"""ctypes bindings for the C core in csrc/. Build it first with `make`.

ctypes releases the GIL while C code runs, so other Python threads keep going.
"""

from __future__ import annotations

import ctypes
import os
from functools import cache
from pathlib import Path

import numpy as np

LIB_PATH = Path(__file__).resolve().parent.parent / "build" / "libseedling.so"


@cache
def lib() -> ctypes.CDLL:
    if not LIB_PATH.exists():
        raise FileNotFoundError(f"{LIB_PATH} not found: run `make` in the repo root first")
    so = ctypes.CDLL(str(LIB_PATH))
    so.sl_version.argtypes = []
    so.sl_version.restype = ctypes.c_char_p
    so.sl_sum_f32.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    so.sl_sum_f32.restype = ctypes.c_int
    so.sl_ring_allreduce_f32.argtypes = [
        ctypes.c_int,  # left_fd
        ctypes.c_int,  # right_fd
        ctypes.c_void_p,  # buf
        ctypes.c_size_t,  # n
        ctypes.c_int,  # rank
        ctypes.c_int,  # world
    ]
    so.sl_ring_allreduce_f32.restype = ctypes.c_int
    return so


def _check(rc: int, name: str) -> None:
    if rc != 0:
        raise OSError(-rc, f"{name}: {os.strerror(-rc)}")


def _f32_ptr(arr: np.ndarray) -> int:
    if arr.dtype != np.float32 or not arr.flags.c_contiguous:
        raise TypeError("expected a C-contiguous float32 array")
    if not arr.flags.writeable:
        raise ValueError("array is read-only")
    return arr.ctypes.data


def version() -> str:
    return lib().sl_version().decode()


def sum_f32(dst: np.ndarray, src: np.ndarray) -> None:
    """dst += src, in C. Both float32, same size, not overlapping."""
    if dst.size != src.size:
        raise ValueError(f"size mismatch: {dst.size} vs {src.size}")
    if np.may_share_memory(dst, src):
        raise ValueError("dst and src overlap")
    if src.dtype != np.float32 or not src.flags.c_contiguous:
        raise TypeError("expected a C-contiguous float32 array")
    _check(lib().sl_sum_f32(_f32_ptr(dst), src.ctypes.data, dst.size), "sl_sum_f32")


def ring_allreduce_f32(left_fd: int, right_fd: int, buf: np.ndarray, rank: int, world: int) -> None:
    """In-place SUM all-reduce of a float32 array over the ring, in C."""
    _check(
        lib().sl_ring_allreduce_f32(left_fd, right_fd, _f32_ptr(buf), buf.size, rank, world),
        "sl_ring_allreduce_f32",
    )
