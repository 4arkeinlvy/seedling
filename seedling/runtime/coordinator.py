"""The coordinator: owns the run's step log and hands out work. No sockets in here.

One step's life:  open (seeds = step_seeds(seed, step, pop))
               -> hand out pairs of seeds to workers that ask (expired ones go out again)
               -> collect scores (first answer per pair wins, digest checked)
               -> commit (integer coefficients, journal append, apply) -> next step
"""

from __future__ import annotations

import base64
import time
from collections.abc import Callable

import numpy as np

from seedling.es.es import step_seeds
from seedling.exact.integer_es import integer_coefficients, replay
from seedling.exact.record import StepRecord
from seedling.exact.state import FixedPointState
from seedling.runtime import protocol as p
from seedling.runtime.config import RunConfig
from seedling.runtime.journal import Journal
from seedling.runtime.tasks import create_task

Clock = Callable[[], float]
WAIT_SECONDS = 0.05


def initial_state(config: RunConfig, task_factory: Callable = create_task) -> FixedPointState:
    task = task_factory(config.task, **config.task_args)
    return FixedPointState.from_float(task.init(config.seed), config.frac_bits)


class Coordinator:
    def __init__(
        self,
        config: RunConfig,
        journal: Journal,
        clock: Clock = time.monotonic,
        task_factory: Callable = create_task,
    ):
        self.config = config
        self.journal = journal
        self.clock = clock
        self.initial = initial_state(config, task_factory)
        self.initial_digest = self.initial.digest()
        journal.open(config, self.initial_digest)
        self.records = journal.records()
        self.state = self._recover()
        self._open_step()
        self._handlers: dict[type, Callable[[p.Message], p.Message]] = {
            p.Hello: self._hello,
            p.Sync: self._sync,
            p.GetWork: self._get_work,
            p.Submit: self._submit,
        }

    @property
    def step(self) -> int:
        return len(self.records)

    @property
    def finished(self) -> bool:
        return self.step >= self.config.steps

    def handle(self, message: p.Message) -> p.Message:
        handler = self._handlers.get(type(message))
        if handler is None:
            return p.Error(f"coordinator does not accept {type(message).__name__}")
        return handler(message)

    # ---------------------------------------------------------------- recovery

    def _recover(self) -> FixedPointState:
        """Rebuild the state from the latest snapshot plus the records after it."""
        snapshot = self.journal.snapshot()
        if snapshot is not None and snapshot[0] <= self.step:
            start, state = snapshot
        else:
            start, state = 0, self.initial
        return replay(state, self.records[start:])

    # ---------------------------------------------------------------- steps

    def _open_step(self) -> None:
        self.seeds = step_seeds(self.config.seed, self.step, self.config.pop)
        self.assigned: dict[int, float] = {}
        self.results: dict[int, tuple[float, float]] = {}

    def _pick_pairs(self, now: float) -> list[int]:
        free = [i for i in range(self.config.pop) if i not in self.results and i not in self.assigned]
        if not free:
            free = [
                i
                for i, at in self.assigned.items()
                if i not in self.results and now - at > self.config.deadline
            ]
        return sorted(free)[: self.config.pairs_per_task]

    def _commit(self) -> None:
        pairs = sorted(self.results)
        f_plus = np.array([self.results[i][0] for i in pairs])
        f_minus = np.array([self.results[i][1] for i in pairs])
        coefs = integer_coefficients(f_plus, f_minus, self.config.lr, self.config.sigma, self.config.frac_bits)
        record = StepRecord(self.step, tuple(self.seeds[i] for i in pairs), tuple(coefs))
        self.journal.append(record)
        self.state.apply(record)
        self.records.append(record)
        if self.config.snapshot_every and self.step % self.config.snapshot_every == 0:
            self.journal.write_snapshot(self.state, self.step)
        self._open_step()

    # ---------------------------------------------------------------- handlers

    def _hello(self, message: p.Hello) -> p.Message:
        return p.Welcome(self.config.to_json(), self.initial_digest)

    def _sync(self, message: p.Sync) -> p.Message:
        behind = self.step - message.from_step
        if message.from_step < 0 or behind < 0:
            return p.Error(f"cannot sync from step {message.from_step}")
        if self.config.snapshot_every and behind > self.config.snapshot_every:
            values = base64.b64encode(self.state.values.tobytes()).decode()
            return p.Snapshot(self.step, self.state.frac_bits, values, self.state.digest())
        return p.Records([r.to_json() for r in self.records[message.from_step :]])

    def _get_work(self, message: p.GetWork) -> p.Message:
        if self.finished:
            return p.Finished(self.step, self.state.digest())
        now = self.clock()
        pairs = self._pick_pairs(now)
        if not pairs:
            return p.Wait(WAIT_SECONDS)
        for i in pairs:
            self.assigned[i] = now
        return p.Work(self.step, pairs)

    def _submit(self, message: p.Submit) -> p.Message:
        if message.step != self.step or self.finished:
            return p.Ack(False, "stale")
        if message.digest != self.state.digest():
            return p.Ack(False, "resync")
        if not (len(message.pairs) == len(message.f_plus) == len(message.f_minus)):
            return p.Ack(False, "malformed")
        for i, fp, fm in zip(message.pairs, message.f_plus, message.f_minus, strict=True):
            if 0 <= i < self.config.pop and i not in self.results:
                self.results[i] = (float(fp), float(fm))
        if len(self.results) == self.config.pop:
            self._commit()
        return p.Ack(True)
