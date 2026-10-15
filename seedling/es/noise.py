"""Seed-reconstructible Gaussian noise: we only ever store or send the seed."""

from __future__ import annotations

import torch


def noise(seed: int, n: int) -> torch.Tensor:
    """epsilon ~ N(0, I) of length n. Same seed, same bits.

    Always generated on the CPU: CUDA's generator gives a different stream for the same seed.
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n, generator=generator)
