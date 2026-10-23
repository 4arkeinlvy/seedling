"""The coordinator: owns the run's step log and hands out work. No sockets in here.

One step's life:  open (seeds = step_seeds(seed, step, pop))
               -> hand out pairs of seeds to workers that ask (expired ones go out again)
               -> collect scores (first answer per pair wins, digest checked)
               -> commit (integer coefficients, journal append, apply) -> next step

With an AuditPolicy (M8, see audit.py) some pairs need more than one opinion: they go out
again to other workers, settle by vote, and outvoted workers are banned before the commit.
With a referee (M10), the coordinator settles disputes by scoring the pair itself.
"""

from __future__ import annotations

import base64
import random
import secrets
import time
from collections.abc import Callable

import numpy as np

from seedling.es.es import step_seeds
from seedling.exact.integer_es import integer_coefficients
from seedling.exact.record import StepRecord
from seedling.exact.state import FixedPointState
from seedling.runtime import protocol as p
from seedling.runtime.access import Access
from seedling.runtime.audit import AuditPolicy, Ledger, Opinion, agree, vote
from seedling.runtime.config import RunConfig
from seedling.runtime.journal import Journal
from seedling.runtime.tasks import create_workload
from seedling.runtime.workload import Workload, replay_workload

Clock = Callable[[], float]
WAIT_SECONDS = 0.05
REFEREE = "referee"
MAX_EVIDENCE = 4 * 2**20  # characters of evidence per pair: a real claim is tens of KB


def build_workload(config: RunConfig, factory: Callable[..., Workload] = create_workload) -> Workload:
    return factory(config.task, **config.task_args)


def initial_state(config: RunConfig, factory: Callable[..., Workload] = create_workload) -> FixedPointState:
    return build_workload(config, factory).initial_state(config.seed, config.frac_bits)


class Coordinator:
    def __init__(
        self,
        config: RunConfig,
        journal: Journal,
        clock: Clock = time.monotonic,
        workload_factory: Callable[..., Workload] = create_workload,
        audit: AuditPolicy = AuditPolicy(),
        rng: random.Random | None = None,
        access: Access = Access(),
    ):
        self.config = config
        self.journal = journal
        self.clock = clock
        self.audit = audit
        # Which pairs get audited must be secret, so this never derives from the run seed.
        self.rng = rng or random.Random(secrets.randbits(64))
        # Counters for /metrics (M10).
        self.referee_scored = 0
        self.opinions_total = 0
        self.pairs_committed_total = 0
        self.last_seen: dict[str, float] = {}
        self.last_commit_at: float | None = None
        self.step_seconds = 0.0
        self.access = access
        # Secure mode: ids are issued (and checked) by this coordinator, not chosen by workers.
        self.secure = audit.enabled or access.required
        self.ledger = Ledger.load(journal.dir / "audit.json") if self.secure else Ledger()
        self.workload = build_workload(config, workload_factory)
        self.initial = self.workload.initial_state(config.seed, config.frac_bits)
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
        return replay_workload(self.workload, state, self.records[start:])

    # ---------------------------------------------------------------- steps

    def _open_step(self) -> None:
        pop = self.config.pop
        self.seeds = step_seeds(self.config.seed, self.step, pop)
        self.assigned: list[dict[str, float]] = [{} for _ in range(pop)]  # worker -> when
        self.opinions: list[list[Opinion]] = [[] for _ in range(pop)]
        self.audited: list[bool | None] = [None] * pop  # None: not drawn yet
        self.results: dict[int, tuple[float, float]] = {}  # settled pairs
        self.dropped: set[int] = set()  # no majority after max_opinions
        self.verdicts: dict[int, Opinion] = {}  # the referee's own scores of disputed pairs
        self.evidence: dict[tuple[int, str], str] = {}  # (pair, worker) -> its evidence, audited pairs only
        self.graded: dict[tuple[int, str], tuple[float, float] | None] = {}  # the referee's grade of it
        self.audit_assignments: set[tuple[int, str]] = set()  # handed out while the pair had an opinion
        self.order = self.rng.sample(range(pop), pop) if self.audit.enabled else list(range(pop))

    def _open(self, i: int) -> bool:
        return i not in self.results and i not in self.dropped

    def _needed(self, i: int) -> int:
        """Opinions pair i needs before it can settle: one, two if audited, one more per tie."""
        if self.audit.median:
            return self.audit.replicas
        n = len(self.opinions[i])
        if not self.audited[i]:
            return 1
        return 2 if n < 2 else n + 1

    def _pick_pairs(self, worker: str, now: float) -> list[int]:
        """Pairs that still need an opinion nobody is working on, and not one from `worker`.
        Pairs that already have opinions (audits, ties) go first: they hold up the step.
        A slot freed only by an expired assignment is a retry, handed out after all others."""
        fresh, retry = [], []
        for i in sorted(self.order, key=lambda i: not self.opinions[i]):
            scored = {o.worker for o in self.opinions[i]}
            mine = self.assigned[i].get(worker)
            if not self._open(i) or worker in scored or (mine is not None and now - mine <= self.config.deadline):
                continue
            pending = [at for w, at in self.assigned[i].items() if w != worker and w not in scored]
            live = sum(now - at <= self.config.deadline for at in pending)
            have, needed = len(scored), self._needed(i)
            if have + live < needed:
                (fresh if have + len(pending) < needed else retry).append(i)
        return (fresh + retry)[: self.config.pairs_per_task]

    def _settle(self, i: int) -> None:
        """Settle pair i if its opinions allow it; ban whoever a settled vote outvoted."""
        opinions = self.opinions[i]
        if not opinions:
            return
        if self.audit.median:
            if len(opinions) >= self.audit.replicas:
                self.results[i] = (
                    float(np.median([o.f_plus for o in opinions])),
                    float(np.median([o.f_minus for o in opinions])),
                )
            return
        if self.audited[i] is None:
            chance = self.ledger.audit_chance(self.audit, opinions[0].worker) if self.audit.enabled else 0.0
            self.audited[i] = self.rng.random() < chance
        if not self.audited[i]:
            self.results[i] = opinions[0].scores
            self.evidence.pop((i, opinions[0].worker), None)  # nobody will check it
            return
        if i in self.verdicts or (self.audit.referee and len(opinions) >= 2):
            self._settle_by_referee(i)
            return
        if len(opinions) < 2:
            return
        winner = vote(self.audit, opinions)
        if winner is None:
            if len(opinions) >= self.audit.max_opinions:
                self.dropped.add(i)
            return
        self.results[i] = opinions[winner].scores
        for o in opinions:
            if not agree(self.audit, o, opinions[winner]):
                self._ban(o.worker, f"outvoted on pair {i} of step {self.step}")

    def _settle_by_referee(self, i: int) -> None:
        """Agreement settles a pair; any disagreement goes to the referee, whose word is final."""
        opinions = self.opinions[i]
        if i not in self.verdicts and all(agree(self.audit, o, opinions[0]) for o in opinions):
            self.results[i] = opinions[0].scores
            return
        if hasattr(self.workload, "check"):
            self._settle_by_evidence(i)
            return
        if i not in self.verdicts:
            self.verdicts[i] = self._scored_verdict(i)
            self.referee_scored += 1
        verdict = self.verdicts[i]
        self.results[i] = verdict.scores
        for o in opinions:
            if not agree(self.audit, o, verdict):
                self._ban(o.worker, f"contradicted by the referee on pair {i} of step {self.step}")

    def _settle_by_evidence(self, i: int) -> None:
        """For workloads that send evidence (generated text): judge each opinion by its OWN
        evidence. Two honest workers may generate different text where the model nearly ties,
        and both hold up; a liar's either fails the check or does not match its own scores."""
        held = [o for o in self.opinions[i] if self._holds_up(i, o)]
        if i not in self.verdicts:
            graded = self.graded.get((i, held[0].worker)) if held else None
            self.verdicts[i] = Opinion(REFEREE, *graded) if graded else self._scored_verdict(i)
            self.referee_scored += 1
        self.results[i] = self.verdicts[i].scores
        for o in list(self.opinions[i]):
            if o not in held:
                self._ban(o.worker, f"its evidence does not hold up on pair {i} of step {self.step}")

    def _holds_up(self, i: int, o: Opinion) -> bool:
        """Does this opinion's evidence pass the check, and do its scores match the grade of it?"""
        key = (i, o.worker)
        if key not in self.graded:
            claim = self.evidence.get(key, "")
            seeds, sigma = [self.seeds[i]], self.config.sigma
            self.graded[key] = self.workload.check(self.state, self.step, seeds, sigma, [claim])[0] if claim else None
        graded = self.graded[key]
        return graded is not None and agree(self.audit, o, Opinion(REFEREE, *graded))

    def _scored_verdict(self, i: int) -> Opinion:
        f_plus, f_minus = self.workload.score(self.state, self.step, [self.seeds[i]], self.config.sigma)
        return Opinion(REFEREE, float(f_plus[0]), float(f_minus[0]))

    def _ban(self, worker: str, reason: str) -> None:
        """Ban `worker` and take back everything it said in the open step."""
        if worker in self.ledger.banned:
            return
        self.ledger.ban(worker, reason)
        token = self.ledger.token_of.get(worker)
        if token and token not in self.ledger.revoked:
            self.ledger.revoke(token, f"{worker} {reason}")
        for i in range(self.config.pop):
            self.assigned[i].pop(worker, None)
            if all(o.worker != worker for o in self.opinions[i]):
                continue
            self.opinions[i] = [o for o in self.opinions[i] if o.worker != worker]
            self.evidence.pop((i, worker), None)
            self.results.pop(i, None)
            self.dropped.discard(i)
            if not self.opinions[i]:
                self.audited[i] = None  # a fresh pair again: the next scorer gets a fresh draw
            self._settle(i)

    def _commit(self) -> None:
        if self.audit.enabled:
            # Dissenters were banned and removed: whoever is left on an audited pair passed.
            for i in self.results:
                for o in self.opinions[i] if self.audited[i] else []:
                    self.ledger.passed[o.worker] = self.ledger.passed.get(o.worker, 0) + 1
            self.ledger.save()
        pairs = sorted(self.results)
        now = self.clock()
        if self.last_commit_at is not None:
            self.step_seconds = now - self.last_commit_at
        self.last_commit_at = now
        self.pairs_committed_total += len(pairs)
        f_plus = np.array([self.results[i][0] for i in pairs])
        f_minus = np.array([self.results[i][1] for i in pairs])
        coefs = integer_coefficients(f_plus, f_minus, self.config.lr, self.config.sigma, self.config.frac_bits)
        record = StepRecord(self.step, tuple(self.seeds[i] for i in pairs), tuple(coefs))
        self.journal.append(record)
        self.workload.apply(self.state, record)
        self.records.append(record)
        if self.config.snapshot_every and self.step % self.config.snapshot_every == 0:
            self.journal.write_snapshot(self.state, self.step)
        self._open_step()

    # ---------------------------------------------------------------- handlers

    def _hello(self, message: p.Hello) -> p.Message:
        if len(message.worker_id) > 128 or len(message.token) > 256:
            return p.Error("worker name or token too long")
        if not self.secure:
            return p.Welcome(self.config.to_json(), self.initial_digest, message.worker_id)
        token = ""
        if self.access.required:
            token = self.access.name_of(message.token) or ""
            if not token:
                return p.Error("invalid join token")
            if token in self.ledger.revoked:
                return p.Error(f"token revoked: {self.ledger.revoked[token]}")
            if self.ledger.ids_of(token) >= self.access.max_ids_per_token:
                return p.Error(f"token {token} already has {self.access.max_ids_per_token} worker ids")
        if len(self.ledger.names) >= self.access.max_ids:
            return p.Error("this run has issued all the worker ids it allows")
        worker = self.ledger.issue(message.worker_id, token)
        return p.Welcome(self.config.to_json(), self.initial_digest, worker)

    def _refuse(self, worker: str) -> p.Error | None:
        """In secure mode, only ids this coordinator issued, not banned or revoked, may work."""
        if not self.secure:
            return None
        if worker in self.ledger.banned:
            return p.Error(f"banned: {self.ledger.banned[worker]}")
        if not self.ledger.known(worker):
            return p.Error("unknown worker id: send Hello first")
        token = self.ledger.token_of.get(worker)
        if token in self.ledger.revoked:
            return p.Error(f"token revoked: {self.ledger.revoked[token]}")
        return None

    def _sync(self, message: p.Sync) -> p.Message:
        behind = self.step - message.from_step
        if message.from_step < 0 or behind < 0:
            return p.Error(f"cannot sync from step {message.from_step}")
        if self.config.snapshot_every and behind > self.config.snapshot_every:
            values = base64.b64encode(self.state.values.tobytes()).decode()
            return p.Snapshot(self.step, self.state.frac_bits, values, self.state.digest())
        return p.Records([r.to_json() for r in self.records[message.from_step :]])

    def _get_work(self, message: p.GetWork) -> p.Message:
        if refused := self._refuse(message.worker_id):
            return refused
        self.last_seen[message.worker_id] = self.clock()
        if self.finished:
            return p.Finished(self.step, self.state.digest())
        now = self.clock()
        pairs = self._pick_pairs(message.worker_id, now)
        if not pairs:
            return p.Wait(WAIT_SECONDS)
        for i in pairs:
            self.assigned[i][message.worker_id] = now
            if self.opinions[i]:
                self.audit_assignments.add((i, message.worker_id))
        # An audit carries the first scorer's evidence, so the auditor can check instead of redo.
        claims = [self.evidence.get((i, self.opinions[i][0].worker), "") if self.opinions[i] else "" for i in pairs]
        return p.Work(self.step, pairs, claims if any(claims) else [])

    def _submit(self, message: p.Submit) -> p.Message:
        if refused := self._refuse(message.worker_id):
            return refused
        if message.step != self.step or self.finished:
            return p.Ack(False, "stale")
        if message.digest != self.state.digest():
            return p.Ack(False, "resync")
        pairs = message.pairs
        if not (len(pairs) == len(message.f_plus) == len(message.f_minus)) or len(set(pairs)) != len(pairs):
            return p.Ack(False, "malformed")
        if message.evidence and len(message.evidence) != len(pairs):
            return p.Ack(False, "malformed")
        if any(len(e) > MAX_EVIDENCE for e in message.evidence):
            return p.Ack(False, "malformed")
        if not all(0 <= i < self.config.pop for i in pairs):
            return p.Ack(False, "malformed")
        worker = message.worker_id
        for k, (i, fp, fm) in enumerate(zip(pairs, message.f_plus, message.f_minus, strict=True)):
            if not self._open(i) or any(o.worker == worker for o in self.opinions[i]):
                continue
            if self.audit.enabled and worker not in self.assigned[i]:
                continue  # unsolicited: otherwise a colluder could pre-empt an audit
            if self.opinions[i] and not self.audit.median and (i, worker) not in self.audit_assignments:
                continue  # a late answer to a fresh assignment (it expired and went out again): not an audit
            self.opinions[i].append(Opinion(worker, fp, fm))
            self.opinions_total += 1
            if message.evidence and message.evidence[k]:
                self.evidence[(i, worker)] = message.evidence[k]
            self._settle(i)
        if len(self.results) + len(self.dropped) == self.config.pop:
            self._commit()
        return p.Ack(True)

    # ---------------------------------------------------------------- metrics

    def metrics_text(self, active_window: float = 60.0) -> str:
        """The run's health in the Prometheus text format (served at GET /metrics)."""
        now = self.clock()
        active = sum(now - seen <= active_window for seen in list(self.last_seen.values()))
        work = self.opinions_total / max(self.pairs_committed_total, 1)
        values = [
            ("seedling_step", "gauge", "Steps committed so far.", self.step),
            ("seedling_steps_target", "gauge", "Steps the run is configured for.", self.config.steps),
            ("seedling_finished", "gauge", "1 once the run is over.", int(self.finished)),
            ("seedling_workers_active", "gauge", f"Workers seen in the last {active_window:g}s.", active),
            ("seedling_workers_banned", "gauge", "Worker ids banned for lying.", len(self.ledger.banned)),
            ("seedling_tokens_revoked", "gauge", "Join tokens revoked for lying.", len(self.ledger.revoked)),
            ("seedling_opinions_total", "counter", "Pair scores received (audits included).", self.opinions_total),
            ("seedling_pairs_committed_total", "counter", "Pairs that went into a step.", self.pairs_committed_total),
            ("seedling_referee_scored_total", "counter", "Disputed pairs the referee scored.", self.referee_scored),
            ("seedling_work_per_pair", "gauge", "Scores received per committed pair (1 = no audit cost).", work),
            ("seedling_step_seconds", "gauge", "Duration of the last step.", self.step_seconds),
        ]
        lines = []
        for name, kind, help_text, value in values:
            lines += [f"# HELP {name} {help_text}", f"# TYPE {name} {kind}", f"{name} {value:g}"]
        return "\n".join(lines) + "\n"
