"""M8: spot checks against dishonest workers.

A worker's only input to training is its scores, so a dishonest worker lies about scores:
it reverses them (pushing the model the wrong way), inflates them, or skips the work and
sends zeros. The coordinator cannot recompute scores itself (it may have no GPU), but other
workers can:

    audit      with probability `rate`, a pair someone scored also goes to a second worker.
               Work messages look the same either way and fresh pairs go out in a secret
               random order, so a worker cannot tell an audit from new work.
    probation  a new worker id has *all* its pairs audited until `probation` of them passed.
               Ids are free to make, so this is what makes getting caught expensive: every
               new identity first has to do `probation` pairs of honest work.
    vote       two opinions agree when both scores match within rtol/atol (scores may come
               from different devices). An audited pair settles once a strict majority of
               at least two opinions agree; until then one more worker is asked, up to
               `max_opinions`, after which the pair is left out of the step.
    ban        a worker outvoted on a pair is banned for the rest of the run, and its other
               opinions in the open step are dropped before the step commits.

This assumes the workers auditing a pair are mostly honest. Colluding workers who happen to
audit each other can still pass a lie; centered ranks then bound what it can do to the
update, whatever the lie is (`influence_bound`).

aggregate="median" is the baseline from robust aggregation (as in CyBeR-0, Egger et al. 2025):
every pair is scored by `replicas` different workers and the coordinate-wise median counts.
Nobody is banned; a lie passes only when most of a pair's scorers tell it. It costs
`replicas` times the work, against about 1.1x for spot checks.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

from seedling.runtime.journal import atomic_write


@dataclass(frozen=True)
class AuditPolicy:
    rate: float = 0.0
    probation: int = 0
    rtol: float = 1e-3
    atol: float = 1e-6
    max_opinions: int = 5
    aggregate: str = "vote"  # "vote": spot checks and bans; "median": replicate every pair
    replicas: int = 3

    def __post_init__(self) -> None:
        if not 0.0 <= self.rate <= 1.0:
            raise ValueError("audit rate must be in [0, 1]")
        if self.probation < 0 or self.max_opinions < 3:
            raise ValueError("probation >= 0 and max_opinions >= 3 (a tie needs a third opinion)")
        if self.aggregate not in ("vote", "median") or self.replicas < 1:
            raise ValueError("aggregate is 'vote' or 'median', and replicas >= 1")

    @property
    def enabled(self) -> bool:
        return self.rate > 0 or self.probation > 0 or self.median

    @property
    def median(self) -> bool:
        return self.aggregate == "median"

    def close(self, a: float, b: float) -> bool:
        return abs(a - b) <= self.atol + self.rtol * max(abs(a), abs(b))


class Opinion(NamedTuple):
    worker: str
    f_plus: float
    f_minus: float

    @property
    def scores(self) -> tuple[float, float]:
        return self.f_plus, self.f_minus


def agree(policy: AuditPolicy, a: Opinion, b: Opinion) -> bool:
    return policy.close(a.f_plus, b.f_plus) and policy.close(a.f_minus, b.f_minus)


def vote(policy: AuditPolicy, opinions: list[Opinion]) -> int | None:
    """Index of the opinion a strict majority (at least two) agrees with, or None."""
    best, support = None, 0
    for k, a in enumerate(opinions):
        n = sum(agree(policy, a, b) for b in opinions)
        if n > support:
            best, support = k, n
    return best if support >= 2 and 2 * support > len(opinions) else None


def influence_bound(pairs: int, dishonest: int) -> float:
    """Most the antithetic weights w_i = s+_i - s-_i can move, in L1, when `dishonest` of
    `pairs` pairs report anything at all, with centered ranks as the shaper.

    Each shaped score lies in [-1/2, 1/2] with spacing 1/(2N-1). A dishonest pair's weight
    can change by at most 2. An honest score's rank moves by at most 2m (only the 2m
    dishonest scores can cross it), so each honest weight moves by at most 4m/(2N-1).
    Every coordinate of the update moves by at most this times lr/(2 N sigma): the noise
    is +-1, so ||delta theta||_inf <= lr/(2 N sigma) * ||delta w||_1.
    """
    n, m = pairs, dishonest
    if n < 1 or not 0 <= m <= n:
        raise ValueError("need 0 <= dishonest <= pairs, pairs >= 1")
    if n == 1:
        return 2.0 * m
    return 2.0 * m + (n - m) * 4.0 * m / (2 * n - 1)


@dataclass
class Ledger:
    """Worker ids the coordinator issued, how many audits each passed, and who is banned.

    Saved as audit.json in the run directory, so a coordinator restart is not an amnesty.
    """

    path: Path | None = None
    names: dict[str, str] = field(default_factory=dict)
    passed: dict[str, int] = field(default_factory=dict)
    banned: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> Ledger:
        if not path.exists():
            return cls(path)
        data = json.loads(path.read_text())
        return cls(path, data["names"], data["passed"], data["banned"])

    def save(self) -> None:
        if self.path is not None:
            data = {"names": self.names, "passed": self.passed, "banned": self.banned}
            atomic_write(self.path, json.dumps(data, indent=1, sort_keys=True).encode())

    def issue(self, name: str) -> str:
        """A fresh, unguessable id: nobody can submit (or be framed) under another's id."""
        worker = f"{name}~{secrets.token_hex(8)}"
        self.names[worker] = name
        self.passed[worker] = 0
        self.save()
        return worker

    def known(self, worker: str) -> bool:
        return worker in self.names

    def audit_chance(self, policy: AuditPolicy, worker: str) -> float:
        return 1.0 if self.passed.get(worker, 0) < policy.probation else policy.rate

    def ban(self, worker: str, reason: str) -> None:
        self.banned[worker] = reason
        self.save()
