"""Slow, obviously-correct Python versions of the C core: the oracles for the tests.

They follow the C code (csrc/philox.c, csrc/xxhash.c, csrc/apply.c) line by line.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
SIGNS_PER_BLOCK = 128

PHILOX_M0, PHILOX_M1 = 0xD2511F53, 0xCD9E8D57
PHILOX_W0, PHILOX_W1 = 0x9E3779B9, 0xBB67AE85


def philox4x32(counter: Sequence[int], key: Sequence[int], rounds: int = 10) -> list[int]:
    c0, c1, c2, c3 = counter
    k0, k1 = key
    for r in range(rounds):
        if r > 0:
            k0 = (k0 + PHILOX_W0) & MASK32
            k1 = (k1 + PHILOX_W1) & MASK32
        p0 = PHILOX_M0 * c0
        p1 = PHILOX_M1 * c2
        c0, c1, c2, c3 = (p1 >> 32) ^ c1 ^ k0, p1 & MASK32, (p0 >> 32) ^ c3 ^ k1, p0 & MASK32
    return [c0, c1, c2, c3]


def sign_at(seed: int, stream: int, i: int) -> int:
    block, bit = divmod(i, SIGNS_PER_BLOCK)
    words = philox4x32(
        [block & MASK32, (block >> 32) & MASK32, stream, 0], [seed & MASK32, (seed >> 32) & MASK32]
    )
    return 1 if (words[bit // 32] >> (bit % 32)) & 1 else -1


def rademacher(seed: int, n: int, stream: int = 0, offset: int = 0) -> np.ndarray:
    return np.array([sign_at(seed, stream, offset + j) for j in range(n)], dtype=np.int8)


def apply_rademacher(
    state: np.ndarray, seeds: Sequence[int], coefs: Sequence[int], stream: int = 0
) -> np.ndarray:
    acc = state.astype(np.int64)
    for seed, coef in zip(seeds, coefs, strict=True):
        acc += int(coef) * rademacher(seed, state.size, stream).astype(np.int64)
    return np.clip(acc, np.iinfo(np.int32).min, np.iinfo(np.int32).max).astype(np.int32)


P1, P2, P3 = 0x9E3779B185EBCA87, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9
P4, P5 = 0x85EBCA77C2B2AE63, 0x27D4EB2F165667C5


def _rotl(x: int, r: int) -> int:
    return ((x << r) | (x >> (64 - r))) & MASK64


def _round(acc: int, lane: int) -> int:
    return (_rotl((acc + lane * P2) & MASK64, 31) * P1) & MASK64


def _merge(acc: int, value: int) -> int:
    return ((acc ^ _round(0, value)) * P1 + P4) & MASK64


def xxh64(data: bytes, seed: int = 0) -> int:
    n = len(data)
    p = 0
    if n >= 32:
        v = [(seed + P1 + P2) & MASK64, (seed + P2) & MASK64, seed, (seed - P1) & MASK64]
        while p <= n - 32:
            for lane in range(4):
                v[lane] = _round(v[lane], int.from_bytes(data[p + 8 * lane : p + 8 * lane + 8], "little"))
            p += 32
        h = (_rotl(v[0], 1) + _rotl(v[1], 7) + _rotl(v[2], 12) + _rotl(v[3], 18)) & MASK64
        for lane in range(4):
            h = _merge(h, v[lane])
    else:
        h = (seed + P5) & MASK64
    h = (h + n) & MASK64
    while p + 8 <= n:
        h = (_rotl(h ^ _round(0, int.from_bytes(data[p : p + 8], "little")), 27) * P1 + P4) & MASK64
        p += 8
    if p + 4 <= n:
        h = (_rotl(h ^ ((int.from_bytes(data[p : p + 4], "little") * P1) & MASK64), 23) * P2 + P3) & MASK64
        p += 4
    while p < n:
        h = (_rotl(h ^ ((data[p] * P5) & MASK64), 11) * P1) & MASK64
        p += 1
    h ^= h >> 33
    h = (h * P2) & MASK64
    h ^= h >> 29
    h = (h * P3) & MASK64
    h ^= h >> 32
    return h


def apply_lowrank(
    delta: np.ndarray, seeds: Sequence[int], coefs: Sequence[int], stream_a: int, stream_b: int
) -> np.ndarray:
    rows, cols = delta.shape
    acc = delta.astype(np.int64)
    for seed, coef in zip(seeds, coefs, strict=True):
        a = rademacher(seed, cols, stream_a).astype(np.int64)
        b = rademacher(seed, rows, stream_b).astype(np.int64)
        acc += int(coef) * np.outer(b, a)
    return np.clip(acc, np.iinfo(np.int32).min, np.iinfo(np.int32).max).astype(np.int32)
