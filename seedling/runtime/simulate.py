"""Run a coordinator against simulated workers, some of them dishonest. No sockets, no threads.

Each round, the live workers ask for work in a random order, "score" their pairs and submit.
A simulated worker does not compute anything: the true scores of a step are computed once
(the coordinator's state is every honest replica's state, bit for bit) and each worker's
behaviour turns them into what it reports. Deterministic for a given `seed`.

    honest    the true scores, times (1 + jitter * noise): a different device's rounding
    flip      swaps F+ and F-: pushes the model the wrong way (the most harmful bounded lie)
    inflate   F+ = 1e9, F- = -1e9: claims its noise is the best direction
    lazy      skips the work and sends zeros (a free rider)
    random    the true scores of some other pair: plausible, but wrong
    sleeper   honest for its first `sleeper_after` pairs (through probation), then flips

Deterministic lies collude for free: two flippers auditing each other report the same lie.
"""

from __future__ import annotations

import random
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field

from seedling.runtime import protocol as p
from seedling.runtime.audit import AuditPolicy
from seedling.runtime.config import RunConfig
from seedling.runtime.coordinator import Coordinator
from seedling.runtime.journal import Journal
from seedling.runtime.tasks import create_workload
from seedling.runtime.workload import Workload

Scores = tuple[float, float]
BEHAVIOURS = ("honest", "flip", "inflate", "lazy", "random", "sleeper")


@dataclass
class SimWorker:
    name: str
    behaviour: str = "honest"
    worker_id: str = ""
    pairs_scored: int = 0
    banned_at_step: int | None = None

    @property
    def honest(self) -> bool:
        return self.behaviour == "honest"


@dataclass
class SimResult:
    workers: list[SimWorker]
    steps: int
    finished: bool
    pairs_scored: int  # by all workers, audits included
    lies_committed: int  # committed pairs whose scores were not the true ones
    pairs_dropped: int
    pop: int
    metrics: dict[str, float] = field(default_factory=dict)
    digest: str = ""

    @property
    def cost(self) -> float:
        """Pairs scored per pair committed: 1.0 without audits."""
        committed = self.steps * self.pop - self.pairs_dropped
        return self.pairs_scored / max(committed, 1)

    def banned(self, honest: bool) -> int:
        return sum(w.banned_at_step is not None for w in self.workers if w.honest == honest)


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _Recorder(Coordinator):
    """Counts committed lies: the simulation knows the true scores, the coordinator does not."""

    truth: dict[int, list[Scores]]
    lies = 0
    dropped_total = 0

    def _commit(self) -> None:
        truth = self.truth[self.step]
        for i, (fp, fm) in self.results.items():
            self.lies += not (self.audit.close(fp, truth[i][0]) and self.audit.close(fm, truth[i][1]))
        self.dropped_total += len(self.dropped)
        super()._commit()


def simulate(
    config: RunConfig,
    behaviours: list[str],
    audit: AuditPolicy = AuditPolicy(),
    seed: int = 0,
    jitter: float = 0.0,
    sleeper_after: int = 16,
    evaluate: bool = True,
    workload_factory: Callable[..., Workload] = create_workload,
    max_idle_rounds: int = 3,
) -> SimResult:
    """Train `config` with one simulated worker per entry of `behaviours`."""
    unknown = set(behaviours) - set(BEHAVIOURS)
    if unknown:
        raise ValueError(f"unknown behaviours {sorted(unknown)}; known: {BEHAVIOURS}")
    rng = random.Random(seed)
    clock = _Clock()
    with tempfile.TemporaryDirectory() as run_dir:
        coordinator = _Recorder(
            config, Journal(run_dir), clock, workload_factory, audit, rng=random.Random(rng.randrange(2**63))
        )
        coordinator.truth = {}
        workers = [SimWorker(f"w{k}", b) for k, b in enumerate(behaviours)]
        for w in workers:
            welcome = coordinator.handle(p.Hello(w.name))
            assert isinstance(welcome, p.Welcome)
            w.worker_id = welcome.worker_id
        pairs_scored, idle = 0, 0
        while not coordinator.finished and idle < max_idle_rounds:
            live = [w for w in workers if w.banned_at_step is None]
            rng.shuffle(live)
            worked = False
            for w in live:
                if coordinator.finished:
                    break
                reply = coordinator.handle(p.GetWork(w.worker_id))
                if isinstance(reply, p.Error):
                    w.banned_at_step = coordinator.step
                    continue
                if not isinstance(reply, p.Work):
                    continue
                truth = _truth(coordinator, reply.step)
                reports = [_report(w, truth, i, rng, jitter, sleeper_after) for i in reply.pairs]
                w.pairs_scored += len(reply.pairs)
                pairs_scored += len(reply.pairs)
                worked = True
                digest = coordinator.state.digest()
                f_plus, f_minus = [r[0] for r in reports], [r[1] for r in reports]
                ack = coordinator.handle(p.Submit(w.worker_id, reply.step, reply.pairs, f_plus, f_minus, digest))
                if isinstance(ack, p.Error):
                    w.banned_at_step = coordinator.step
            for w in workers:  # bans can also land while someone else submits
                if w.banned_at_step is None and w.worker_id in coordinator.ledger.banned:
                    w.banned_at_step = coordinator.step
            if worked:
                idle = 0
            else:  # everyone is waiting on expired work: let the deadline pass
                idle += 1
                clock.now += config.deadline + 1
        metrics = coordinator.workload.evaluate(coordinator.state, coordinator.step) if evaluate else {}
        return SimResult(
            workers=workers,
            steps=coordinator.step,
            finished=coordinator.finished,
            pairs_scored=pairs_scored,
            lies_committed=coordinator.lies,
            pairs_dropped=coordinator.dropped_total,
            metrics=metrics,
            digest=coordinator.state.digest(),
            pop=config.pop,
        )


def _truth(coordinator: _Recorder, step: int) -> list[Scores]:
    if step not in coordinator.truth:
        config = coordinator.config
        f_plus, f_minus = coordinator.workload.score(coordinator.state, step, coordinator.seeds, config.sigma)
        coordinator.truth[step] = [(float(a), float(b)) for a, b in zip(f_plus, f_minus, strict=True)]
    return coordinator.truth[step]


def _report(w: SimWorker, truth: list[Scores], i: int, rng: random.Random, jitter: float, after: int) -> Scores:
    fp, fm = truth[i]
    behaviour = w.behaviour
    if behaviour == "sleeper":
        behaviour = "honest" if w.pairs_scored < after else "flip"
    if behaviour == "honest":
        if jitter:
            return fp * (1 + jitter * rng.uniform(-1, 1)), fm * (1 + jitter * rng.uniform(-1, 1))
        return fp, fm
    if behaviour == "flip":
        return fm, fp
    if behaviour == "inflate":
        return 1e9, -1e9
    if behaviour == "lazy":
        return 0.0, 0.0
    if behaviour == "random":
        other = truth[rng.randrange(len(truth))]
        return other if other != truth[i] else (other[1], other[0])
    raise AssertionError(behaviour)
