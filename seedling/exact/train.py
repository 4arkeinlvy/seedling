"""Train any Task with IntegerES in one process, keeping the replayable step records."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from seedling.es.evaluate import evaluate_batched
from seedling.exact.integer_es import IntegerES, rademacher_noise
from seedling.exact.record import StepRecord
from seedling.exact.state import FixedPointState
from seedling.tasks.base import Task


@dataclass
class Run:
    es: IntegerES
    initial: FixedPointState
    records: list[StepRecord] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)


def train_integer_es(
    task: Task,
    steps: int,
    pop: int = 32,
    sigma: float = 0.05,
    lr: float = 0.05,
    seed: int = 0,
    frac_bits: int = 20,
    eval_every: int = 10,
) -> Run:
    initial = FixedPointState.from_float(task.init(seed), frac_bits)
    es = IntegerES(initial.copy(), pop=pop, sigma=sigma, lr=lr, seed=seed)
    run = Run(es, initial)
    for step in range(steps):
        seeds = es.ask()
        f_plus, f_minus = evaluate_batched(
            lambda thetas: task.fitness(thetas, step), es.theta, seeds, sigma, noise_fn=rademacher_noise
        )
        run.records.append(es.tell(seeds, f_plus, f_minus))
        if (step + 1) % eval_every == 0 or step + 1 == steps:
            metrics = task.evaluate(es.theta, step)
            run.history.append(
                {"step": step + 1, "population_mean": float(np.mean(np.r_[f_plus, f_minus])), **metrics}
            )
    return run
