"""One committed training step: everything needed to replay it, and nothing more."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StepRecord:
    step: int
    seeds: tuple[int, ...]
    coefs: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.seeds) != len(self.coefs):
            raise ValueError("one integer coefficient per seed")

    def to_json(self) -> dict[str, Any]:
        return {"step": self.step, "seeds": list(self.seeds), "coefs": list(self.coefs)}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> StepRecord:
        return cls(int(data["step"]), tuple(int(s) for s in data["seeds"]), tuple(int(c) for c in data["coefs"]))

    def nbytes(self) -> int:
        """Size on the wire in a compact binary form: 8-byte seed + 4-byte coefficient each."""
        return 8 + 12 * len(self.seeds)
