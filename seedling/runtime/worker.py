"""A worker: keep a replica of the state in sync, score the pairs it is given, report back.

It never receives weights or noise: seeds are recomputed from (run seed, step), and the
state follows the coordinator's step records, applied exactly (M4).
"""

from __future__ import annotations

import base64
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from seedling.es.es import step_seeds
from seedling.exact.record import StepRecord
from seedling.exact.state import FixedPointState
from seedling.runtime import protocol as p
from seedling.runtime.config import RunConfig
from seedling.runtime.coordinator import build_workload
from seedling.runtime.tasks import create_workload
from seedling.runtime.transport import Transport
from seedling.runtime.workload import Workload


class ReplicaMismatch(RuntimeError):
    """This worker's state does not match the coordinator's: different code or a bug."""


def expect(reply: p.Message, kind: type[p.Message] | tuple[type[p.Message], ...]) -> p.Message:
    if not isinstance(reply, kind):
        raise p.ProtocolError(f"unexpected reply {type(reply).__name__}")
    return reply


@dataclass
class WorkerStats:
    pairs_scored: int = 0
    submits_accepted: int = 0
    reconnects: int = 0
    resyncs: int = 0


class Worker:
    def __init__(
        self,
        transport: Transport,
        worker_id: str,
        workload_factory: Callable[..., Workload] = create_workload,
        max_retry_seconds: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
        token: str = "",
    ):
        self.transport = transport
        self.worker_id = worker_id
        self.token = token
        self.workload_factory = workload_factory
        self.max_retry_seconds = max_retry_seconds
        self.sleep = sleep
        self.stats = WorkerStats()

    # ---------------------------------------------------------------- talking

    def _call(self, message: p.Message) -> p.Message:
        """Send with retries: a restarting coordinator looks like a short outage."""
        delay, waited = 0.1, 0.0
        while True:
            try:
                reply = self.transport.request(message)
            except ConnectionError:
                if waited >= self.max_retry_seconds:
                    raise
                self.stats.reconnects += 1
                self.sleep(delay)
                waited += delay
                delay = min(delay * 2, 2.0)
                continue
            if isinstance(reply, p.Error):
                raise RuntimeError(f"coordinator refused {type(message).__name__}: {reply.reason}")
            return reply

    # ---------------------------------------------------------------- state

    def _join(self) -> None:
        welcome = expect(self._call(p.Hello(self.worker_id, self.token)), p.Welcome)
        self.worker_id = welcome.worker_id  # with audits on, the coordinator issues our id
        self.config = RunConfig.from_json(welcome.config)
        self.workload = build_workload(self.config, self.workload_factory)
        self.initial = self.workload.initial_state(self.config.seed, self.config.frac_bits)
        if self.initial.digest() != welcome.initial_digest:
            raise ReplicaMismatch("initial state differs from the coordinator's")
        fingerprint = getattr(self.workload, "fingerprint", None)
        self.fingerprint = fingerprint() if fingerprint else ""
        self._reset()

    def _reset(self) -> None:
        self.state = self.initial.copy()
        self.step = 0

    def _sync(self) -> None:
        reply = expect(self._call(p.Sync(self.step)), (p.Snapshot, p.Records))
        if isinstance(reply, p.Snapshot):
            values = np.frombuffer(base64.b64decode(reply.values_b64), dtype=np.int32).copy()
            self.state = FixedPointState(values, reply.frac_bits)
            if self.state.digest() != reply.digest:
                raise ReplicaMismatch("snapshot digest mismatch")
            self.step = reply.step
            return
        for data in reply.records:
            record = StepRecord.from_json(data)
            if record.step != self.step:
                raise ReplicaMismatch(f"expected record {self.step}, got {record.step}")
            self.workload.apply(self.state, record)
            self.step += 1

    # ---------------------------------------------------------------- work

    def _score(self, work: p.Work) -> p.Submit:
        """Score the pairs. A pair with a claim (an audit) is checked against the claimed evidence
        if the workload can (M10); others are scored, with evidence if the workload sends it."""
        seeds = step_seeds(self.config.seed, work.step, self.config.pop)
        claims = work.claims or [""] * len(work.pairs)
        sigma, n = self.config.sigma, len(work.pairs)
        f_plus, f_minus, evidence = np.zeros(n), np.zeros(n), [""] * n
        check = [k for k in range(n) if claims[k] and hasattr(self.workload, "verify")]
        score = [k for k in range(n) if k not in check]
        if check:
            chosen = [seeds[work.pairs[k]] for k in check]
            fp, fm, packs = self.workload.verify(self.state, work.step, chosen, sigma, [claims[k] for k in check])
            f_plus[check], f_minus[check] = fp, fm
            for k, pack in zip(check, packs, strict=True):
                evidence[k] = pack
        if score:
            chosen = [seeds[work.pairs[k]] for k in score]
            if hasattr(self.workload, "score_with_evidence"):
                fp, fm, packs = self.workload.score_with_evidence(self.state, work.step, chosen, sigma)
                for k, pack in zip(score, packs, strict=True):
                    evidence[k] = pack
            else:
                fp, fm = self.workload.score(self.state, work.step, chosen, sigma)
            f_plus[score], f_minus[score] = fp, fm
        self.stats.pairs_scored += n
        return p.Submit(
            self.worker_id, work.step, list(work.pairs), f_plus.tolist(), f_minus.tolist(), self.state.digest(),
            evidence if any(evidence) else [],
        )  # fmt: skip

    def run(self) -> str:
        """Work until the run is finished; returns the final state digest."""
        self._join()
        while True:
            self._sync()
            reply = expect(self._call(p.GetWork(self.worker_id, self.fingerprint)), (p.Finished, p.Wait, p.Work))
            if isinstance(reply, p.Finished):
                self._sync()
                if self.state.digest() != reply.digest:
                    raise ReplicaMismatch("final digest differs from the coordinator's")
                return reply.digest
            if isinstance(reply, p.Wait):
                self.sleep(reply.seconds)
                continue
            if reply.step != self.step:
                self._sync()
                if reply.step != self.step:
                    continue
            ack = expect(self._call(self._score(reply)), p.Ack)
            if ack.accepted:
                self.stats.submits_accepted += 1
            elif ack.reason == "resync":
                self.stats.resyncs += 1
                self._reset()
