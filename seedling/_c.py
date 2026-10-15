"""ctypes bindings for the C core in csrc/. Build it first with `make`.

The only module that talks to C: everything else calls these typed wrappers.
ctypes releases the GIL while C code runs, so other Python threads keep going.
"""

from __future__ import annotations

import ctypes
import os
from functools import cache
from pathlib import Path

import numpy as np

LIB_PATH = Path(__file__).resolve().parent.parent / "build" / "libseedling.so"

_c_int, _c_size, _c_ptr = ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p
_c_u32, _c_u64 = ctypes.c_uint32, ctypes.c_uint64
_SIGNATURES = {
    "sl_version": ([], ctypes.c_char_p),
    "sl_sum_f32": ([_c_ptr, _c_ptr, _c_size], _c_int),
    "sl_ring_allreduce_f32": ([_c_int, _c_int, _c_ptr, _c_size, _c_int, _c_int], _c_int),
    "sl_philox4x32": ([_c_ptr, _c_ptr, _c_ptr], _c_int),
    "sl_rademacher_i8": ([_c_u64, _c_u32, _c_u64, _c_ptr, _c_size], _c_int),
    "sl_apply_rademacher_i32": ([_c_ptr, _c_size, _c_ptr, _c_ptr, _c_size, _c_u32], _c_int),
    "sl_xxh64_into": ([_c_ptr, _c_size, _c_u64, _c_ptr], _c_int),
    "sl_set_num_threads": ([_c_int], _c_int),
}


@cache
def lib() -> ctypes.CDLL:
    if not LIB_PATH.exists():
        raise FileNotFoundError(f"{LIB_PATH} not found: run `make` in the repo root first")
    so = ctypes.CDLL(str(LIB_PATH))
    for name, (argtypes, restype) in _SIGNATURES.items():
        fn = getattr(so, name)
        fn.argtypes = argtypes
        fn.restype = restype
    return so


def _check(rc: int, name: str) -> None:
    if rc != 0:
        raise OSError(-rc, f"{name}: {os.strerror(-rc)}")


def _ptr(arr: np.ndarray, dtype: type, writable: bool = False) -> int:
    if arr.dtype != dtype or not arr.flags.c_contiguous:
        raise TypeError(f"expected a C-contiguous {np.dtype(dtype).name} array")
    if writable and not arr.flags.writeable:
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
    _check(
        lib().sl_sum_f32(_ptr(dst, np.float32, writable=True), _ptr(src, np.float32), dst.size),
        "sl_sum_f32",
    )


def ring_allreduce_f32(left_fd: int, right_fd: int, buf: np.ndarray, rank: int, world: int) -> None:
    """In-place SUM all-reduce of a float32 array over the ring, in C."""
    _check(
        lib().sl_ring_allreduce_f32(
            left_fd, right_fd, _ptr(buf, np.float32, writable=True), buf.size, rank, world
        ),
        "sl_ring_allreduce_f32",
    )


def philox4x32(counter: tuple[int, int, int, int], key: tuple[int, int]) -> tuple[int, ...]:
    c = np.array(counter, dtype=np.uint32)
    k = np.array(key, dtype=np.uint32)
    out = np.zeros(4, dtype=np.uint32)
    _check(lib().sl_philox4x32(c.ctypes.data, k.ctypes.data, out.ctypes.data), "sl_philox4x32")
    return tuple(int(v) for v in out)


def rademacher_i8(seed: int, n: int, stream: int = 0, offset: int = 0) -> np.ndarray:
    out = np.empty(n, dtype=np.int8)
    _check(lib().sl_rademacher_i8(seed, stream, offset, out.ctypes.data, n), "sl_rademacher_i8")
    return out


def apply_rademacher_i32(state: np.ndarray, seeds: np.ndarray, coefs: np.ndarray, stream: int = 0) -> None:
    """state += sum_j coefs[j] * sign(seeds[j]), in place, saturated to int32."""
    if seeds.size != coefs.size:
        raise ValueError("one coefficient per seed")
    _check(
        lib().sl_apply_rademacher_i32(
            _ptr(state, np.int32, writable=True),
            state.size,
            _ptr(seeds, np.uint64),
            _ptr(coefs, np.int32),
            seeds.size,
            stream,
        ),
        "sl_apply_rademacher_i32",
    )


def xxh64(data: bytes | np.ndarray, seed: int = 0) -> int:
    view = np.frombuffer(memoryview(data).cast("B"), dtype=np.uint8)
    out = np.zeros(1, dtype=np.uint64)
    _check(lib().sl_xxh64_into(view.ctypes.data, view.size, seed, out.ctypes.data), "sl_xxh64_into")
    return int(out[0])


def set_num_threads(threads: int) -> None:
    _check(lib().sl_set_num_threads(threads), "sl_set_num_threads")
