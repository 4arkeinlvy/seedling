"""Everything that defines a run. Coordinator and workers must agree on all of it."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class RunConfig:
    task: str
    steps: int
    task_args: dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    pop: int = 32
    sigma: float = 0.05
    lr: float = 0.05
    frac_bits: int = 20
    pairs_per_task: int = 4
    deadline: float = 30.0
    snapshot_every: int = 50

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> RunConfig:
        return cls(**data)
