"""Train any Workload in one process: the same steps the coordinator and workers take,
without the network. Useful for experiments and for checking a distributed run."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from seedling.exact.integer_es import IntegerES
from seedling.exact.record import StepRecord
from seedling.exact.state import FixedPointState
from seedling.runtime.config import RunConfig
from seedling.runtime.workload import Workload


@dataclass
class LocalRun:
    initial: FixedPointState
    state: FixedPointState
    records: list[StepRecord] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)


def train_local(
    workload: Workload,
    config: RunConfig,
    eval_every: int = 0,
    log: Callable[[dict], None] = lambda row: None,
) -> LocalRun:
    initial = workload.initial_state(config.seed, config.frac_bits)
    es = IntegerES(initial.copy(), config.pop, config.sigma, config.lr, config.seed, apply=workload.apply)
    run = LocalRun(initial, es.state)
    for step in range(config.steps):
        t0 = time.perf_counter()
        seeds = es.ask()
        f_plus, f_minus = workload.score(es.state, step, seeds, config.sigma)
        run.records.append(es.tell(seeds, f_plus, f_minus))
        row = {
            "step": step + 1,
            "mean_score": float(np.mean(np.r_[f_plus, f_minus])),
            "best_score": float(np.max(np.r_[f_plus, f_minus])),
            "seconds": time.perf_counter() - t0,
        }
        if eval_every and ((step + 1) % eval_every == 0 or step + 1 == config.steps):
            row.update(workload.evaluate(es.state, step + 1))
        run.history.append(row)
        log(row)
    return run
