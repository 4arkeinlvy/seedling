"""What the runtime trains. A Workload owns four decisions:

    initial_state   the int32 state every replica starts from
    apply           how a StepRecord changes that state (exactly)
    score           how a replica scores seeds (any precision, any device)
    evaluate        metrics for logs

The coordinator and the workers only talk to this interface, so new kinds of models
(a small vector model, an LLM with low-rank noise) plug in without touching them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from seedling.es.evaluate import evaluate_batched
from seedling.exact.integer_es import rademacher_noise
from seedling.exact.record import StepRecord
from seedling.exact.state import FixedPointState
from seedling.tasks.base import Task


@runtime_checkable
class Workload(Protocol):
    name: str

    def initial_state(self, seed: int, frac_bits: int) -> FixedPointState: ...

    def apply(self, state: FixedPointState, record: StepRecord) -> None: ...

    def score(
        self, state: FixedPointState, step: int, seeds: Sequence[int], sigma: float
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def evaluate(self, state: FixedPointState, step: int) -> dict[str, float]: ...


class VectorWorkload:
    """Adapter: any Task (a model small enough to hold the whole population) with +-1 noise."""

    def __init__(self, task: Task):
        self.task = task
        self.name = task.name

    def initial_state(self, seed: int, frac_bits: int) -> FixedPointState:
        return FixedPointState.from_float(self.task.init(seed), frac_bits)

    def apply(self, state: FixedPointState, record: StepRecord) -> None:
        state.apply(record)

    def score(
        self, state: FixedPointState, step: int, seeds: Sequence[int], sigma: float
    ) -> tuple[np.ndarray, np.ndarray]:
        return evaluate_batched(
            lambda thetas: self.task.fitness(thetas, step),
            state.to_float(),
            seeds,
            sigma,
            noise_fn=rademacher_noise,
        )

    def evaluate(self, state: FixedPointState, step: int) -> dict[str, float]:
        return self.task.evaluate(state.to_float(), step)


def replay_workload(workload: Workload, state: FixedPointState, records: Sequence[StepRecord]) -> FixedPointState:
    """Apply records in order to a copy of `state`, with the workload's update rule."""
    result = state.copy()
    for record in records:
        workload.apply(result, record)
    return result
