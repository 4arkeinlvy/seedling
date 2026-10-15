"""Weights stored as int32 fixed-point numbers, so that updates are exact everywhere."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from seedling import _c
from seedling.exact.record import StepRecord

INT32 = np.iinfo(np.int32)


@dataclass
class FixedPointState:
    """value = q * 2**-frac_bits for every int32 q in `values`."""

    values: np.ndarray
    frac_bits: int = 20

    def __post_init__(self) -> None:
        if self.values.dtype != np.int32 or not self.values.flags.c_contiguous:
            raise TypeError("values must be a contiguous int32 array")

    @classmethod
    def from_float(cls, theta: torch.Tensor | np.ndarray, frac_bits: int = 20) -> FixedPointState:
        scaled = np.rint(np.asarray(theta, dtype=np.float64) * 2.0**frac_bits)
        if scaled.min(initial=0) < INT32.min or scaled.max(initial=0) > INT32.max:
            raise OverflowError(f"values do not fit int32 with {frac_bits} fractional bits")
        return cls(scaled.astype(np.int32), frac_bits)

    @property
    def unit(self) -> float:
        """The smallest step: 2**-frac_bits."""
        return 2.0**-self.frac_bits

    def to_float(self) -> torch.Tensor:
        return torch.from_numpy((self.values.astype(np.float64) * self.unit).astype(np.float32))

    def apply(self, record: StepRecord) -> None:
        """values += sum_j coefs[j] * sign(seeds[j]): exact, any machine, any thread count."""
        _c.apply_rademacher_i32(
            self.values,
            np.asarray(record.seeds, dtype=np.uint64),
            np.asarray(record.coefs, dtype=np.int32),
        )

    def digest(self) -> str:
        """xxHash64 of the raw int32 bytes, as 16 hex digits."""
        return f"{_c.xxh64(self.values):016x}"

    def copy(self) -> FixedPointState:
        return FixedPointState(self.values.copy(), self.frac_bits)
