"""M8 spec: dishonest workers. Run: uv run pytest tests/test_m8_byzantine.py -x"""

import json
import random
import threading

import numpy as np
import pytest

from seedling.es.es import centered_ranks, raw_fitness
from seedling.exact.integer_es import replay
from seedling.runtime import Coordinator, Journal, RunConfig, Worker, initial_state, make_server, serialized
from seedling.runtime import protocol as p
from seedling.runtime.audit import AuditPolicy, Opinion, influence_bound, vote
from seedling.runtime.simulate import simulate
from seedling.runtime.transport import TcpTransport, _respond


def quadratic(steps: int = 3, pop: int = 4, **overrides) -> RunConfig:
    args = dict(task="quadratic", steps=steps, pop=pop, sigma=0.1, lr=0.1, pairs_per_task=2, deadline=5.0)
    args.update(overrides)
    return RunConfig(**args)


class Script(random.Random):
    """An rng whose random() plays back a script: decides exactly which pairs get audited."""

    def __init__(self, draws: list[float]):
        super().__init__(0)
        self.draws = list(draws)

    def random(self) -> float:
        return self.draws.pop(0)

    def getrandbits(self, k: int) -> int:  # defined, so sample() does not draw from the script
        return super().getrandbits(k)


def submit(c: Coordinator, worker: str, pairs: list[int], scores: list[tuple[float, float]]) -> p.Message:
    f_plus, f_minus = [s[0] for s in scores], [s[1] for s in scores]
    return c.handle(p.Submit(worker, c.step, pairs, f_plus, f_minus, c.state.digest()))


def join(c: Coordinator, *names: str) -> list[str]:
    return [c.handle(p.Hello(n)).worker_id for n in names]


# ------------------------------------------------------------------ protocol: nothing malformed gets in


def _submit_json(**body) -> bytes:
    fields = dict(worker_id="w", step=0, pairs=[0], f_plus=[1.0], f_minus=[0.0], digest="00")
    fields.update(body)
    return json.dumps({"v": p.PROTOCOL_VERSION, "type": "Submit", "body": fields}).encode()


@pytest.mark.parametrize(
    "raw",
    [
        _submit_json(pairs=[0.5]),
        _submit_json(pairs=[True]),
        _submit_json(pairs=["0"]),
        _submit_json(pairs=None),
        _submit_json(f_plus=["1.0"]),
        _submit_json(f_plus=[True]),
        _submit_json(step="0"),
        _submit_json(worker_id=7),
        _submit_json().replace(b"[1.0]", b"[NaN]"),
        _submit_json().replace(b"[1.0]", b"[Infinity]"),
        json.dumps({"v": p.PROTOCOL_VERSION, "type": "Submit", "body": [1, 2]}).encode(),
    ],
)
def test_decode_checks_every_field(raw):
    with pytest.raises(p.ProtocolError):
        p.decode(raw)


def test_ints_are_accepted_where_floats_are_expected():
    message = p.decode(_submit_json(f_plus=[1], f_minus=[-2]))
    assert message.f_plus == [1.0] and isinstance(message.f_plus[0], float)


def test_hostile_submissions_cannot_wedge_the_run(tmp_path):
    """Before M8, pairs like 0.5 were stored as results and every later commit crashed."""
    c = Coordinator(quadratic(steps=1), Journal(tmp_path))
    digest = c.state.digest()
    for pairs in ([0.5], [1.5], [2.5], [3.5]):
        reply = p.decode(_respond(c.handle, _submit_json(pairs=pairs, digest=digest)))
        assert isinstance(reply, p.Error)
    for pairs in ([0, 0], [-1], [4]):
        assert submit(c, "w", pairs, [(1.0, 0.0)] * len(pairs)) == p.Ack(False, "malformed")
    assert submit(c, "w", [0, 1, 2, 3], [(float(i), 0.0) for i in range(4)]) == p.Ack(True)
    assert c.finished


# ------------------------------------------------------------------ vote and the influence bound


def test_vote_needs_a_strict_majority_of_at_least_two():
    policy = AuditPolicy(rtol=1e-3)
    a, a2, b, c = Opinion("x", 1.0, 2.0), Opinion("y", 1.0002, 2.0), Opinion("z", 2.0, 1.0), Opinion("u", 0, 0)
    assert vote(policy, [a]) is None
    assert vote(policy, [a, a2]) == 0, "within rtol: agreement"
    assert vote(policy, [a, b]) is None, "a tie needs a third opinion"
    assert vote(policy, [b, a, a2]) == 1
    assert vote(policy, [a, b, c]) is None
    assert vote(policy, [a, b, c, a2]) is None, "2 of 4 is not a majority"


@pytest.mark.parametrize("trial", range(50))
def test_centered_ranks_bound_what_any_lie_can_do(trial):
    rng = np.random.default_rng(trial)
    n = int(rng.integers(2, 40))
    m = int(rng.integers(0, n + 1))
    honest = rng.normal(size=2 * n).round(1)  # rounding makes ties
    lie = honest.copy()
    liars = rng.choice(n, size=m, replace=False)
    wild = np.array([1e300, -1e300, 0.0, honest.max(), honest.min(), *honest[:4]])
    for i in liars:
        lie[i], lie[n + i] = rng.choice(wild), rng.choice(wild)

    def weights(f):
        s = centered_ranks(f)
        return s[:n] - s[n:]

    moved = np.abs(weights(lie) - weights(honest)).sum()
    assert moved <= influence_bound(n, m) + 1e-9


def test_raw_scores_have_no_such_bound():
    honest = np.linspace(-1, 1, 20)
    lie = honest.copy()
    lie[0] = 1e9
    moved = np.abs((raw_fitness(lie) - raw_fitness(honest))[:10]).sum()
    assert moved > 1e6 * influence_bound(10, 1), "one lie, any size: without ranks it is unbounded"


# ------------------------------------------------------------------ coordinator with audits


def test_audited_runs_issue_ids_and_refuse_strangers(tmp_path):
    c = Coordinator(quadratic(), Journal(tmp_path), audit=AuditPolicy(probation=2))
    a, b = join(c, "laptop", "laptop")
    assert a != b and a.startswith("laptop~"), "ids are issued, unique, and keep the name"
    assert isinstance(c.handle(p.GetWork("laptop")), p.Error), "a made-up id cannot work"
    assert isinstance(submit(c, "laptop", [0], [(1.0, 0.0)]), p.Error), "nor be framed"
    assert isinstance(c.handle(p.GetWork(a)), p.Work)


def test_a_new_worker_is_audited_by_someone_else(tmp_path):
    c = Coordinator(quadratic(pop=4), Journal(tmp_path), audit=AuditPolicy(probation=100))
    a, b = join(c, "a", "b")
    work = c.handle(p.GetWork(a))
    assert submit(c, a, work.pairs, [(1.0, 0.0)] * 2) == p.Ack(True)
    assert c.results == {}, "on probation: one opinion is not enough"
    again = c.handle(p.GetWork(a))
    assert not set(again.pairs) & set(work.pairs), "nobody audits their own pairs"
    assert isinstance(c.handle(p.GetWork(a)), p.Wait), "nor gets pairs it is already working on"
    audit = c.handle(p.GetWork(b))
    assert audit.pairs == work.pairs, "audits go out first: they hold up the step"
    submit(c, b, audit.pairs, [(1.0, 0.0)] * 2)
    assert set(c.results) == set(work.pairs)


def test_a_liar_is_outvoted_banned_and_its_scores_thrown_away(tmp_path):
    c = Coordinator(quadratic(pop=2, pairs_per_task=2), Journal(tmp_path), audit=AuditPolicy(probation=100))
    liar, h1, h2 = join(c, "liar", "h1", "h2")
    truth = [(5.0, 1.0), (2.0, 3.0)]
    pairs = c.handle(p.GetWork(liar)).pairs
    submit(c, liar, pairs, [truth[i][::-1] for i in pairs])
    submit(c, h1, c.handle(p.GetWork(h1)).pairs, [truth[i] for i in pairs])
    assert c.results == {} and c.step == 0, "1 against 1: ask a third worker"
    tiebreak = c.handle(p.GetWork(h2))
    assert sorted(tiebreak.pairs) == [0, 1]
    submit(c, h2, tiebreak.pairs, [truth[i] for i in tiebreak.pairs])
    assert liar in c.ledger.banned and h1 not in c.ledger.banned
    assert c.step == 1, "both pairs settled 2 to 1: the step committed"
    assert c.ledger.passed[h1] == c.ledger.passed[h2] == 2
    assert c.handle(p.GetWork(liar)).reason.startswith("banned")


def test_a_ban_takes_back_the_liars_unaudited_scores(tmp_path):
    # rate 0.5: the scripted draws make pair 0 unaudited (0.9) and pair 1 audited (0.1).
    rng = Script([0.9, 0.1, 0.9])
    config = quadratic(pop=2, pairs_per_task=1)
    c = Coordinator(config, Journal(tmp_path), audit=AuditPolicy(rate=0.5), rng=rng)
    c.order = [0, 1]
    liar, h1, h2 = join(c, "liar", "h1", "h2")
    assert c.handle(p.GetWork(liar)).pairs == [0]
    submit(c, liar, [0], [(9.0, -9.0)])
    assert c.results == {0: (9.0, -9.0)}, "unaudited: accepted on one opinion"
    assert c.handle(p.GetWork(liar)).pairs == [1]
    submit(c, liar, [1], [(9.0, -9.0)])
    submit(c, h1, c.handle(p.GetWork(h1)).pairs, [(1.0, 2.0)])
    submit(c, h2, c.handle(p.GetWork(h2)).pairs, [(1.0, 2.0)])
    assert liar in c.ledger.banned
    assert 0 not in c.results, "the ban reopened the pair it slipped through"
    assert c.handle(p.GetWork(h1)).pairs == [0]
    submit(c, h1, [0], [(3.0, 4.0)])
    assert c.step == 1
    assert len(c.records[0].seeds) == 2


def test_unsolicited_opinions_do_not_count(tmp_path):
    c = Coordinator(quadratic(pop=2), Journal(tmp_path), audit=AuditPolicy(probation=100))
    a, b = join(c, "a", "b")
    pairs = c.handle(p.GetWork(a)).pairs
    submit(c, a, pairs, [(1.0, 2.0)] * 2)
    submit(c, b, pairs, [(1.0, 2.0)] * 2)  # b never asked for these: it could be a colluder
    assert c.results == {}


def test_bans_and_ids_survive_a_coordinator_restart(tmp_path):
    config, policy = quadratic(pop=2), AuditPolicy(probation=100)
    c = Coordinator(config, Journal(tmp_path), audit=policy)
    liar, h1, h2 = join(c, "liar", "h1", "h2")
    c._ban(liar, "test")
    again = Coordinator(config, Journal(tmp_path), audit=policy)
    assert isinstance(again.handle(p.GetWork(liar)), p.Error)
    assert isinstance(again.handle(p.GetWork(h1)), p.Work)


# ------------------------------------------------------------------ simulated attacks


SIM = quadratic(steps=12, pop=16)
AUDIT = AuditPolicy(rate=0.2, probation=4)


@pytest.mark.parametrize("attack", ["flip", "inflate", "lazy", "random", "sleeper"])
def test_spot_checks_ban_every_liar_and_no_honest_worker(attack):
    result = simulate(SIM, ["honest"] * 6 + [attack] * 2, AUDIT, seed=3, jitter=1e-6, sleeper_after=8)
    assert result.finished
    assert result.banned(honest=False) == 2
    assert result.banned(honest=True) == 0
    assert result.cost < 1.5, f"audits cost {result.cost:.2f} evaluations per committed pair"


def test_without_spot_checks_lies_go_through():
    result = simulate(SIM, ["honest"] * 6 + ["flip"] * 2, seed=3)
    assert result.finished and result.lies_committed > 0 and result.banned(honest=False) == 0


def test_honest_rounding_differences_never_get_anyone_banned():
    result = simulate(SIM, ["honest"] * 6, AuditPolicy(rate=1.0, rtol=1e-3), seed=0, jitter=1e-5)
    assert result.finished and result.banned(honest=True) == 0
    assert result.lies_committed == 0 and result.cost == pytest.approx(2.0)


# ------------------------------------------------------------------ end to end: real workers, real sockets


class FlippingWorker(Worker):
    """A real worker that does the work, then reports F+ and F- swapped."""

    def _score(self, work: p.Work) -> p.Submit:
        honest = super()._score(work)
        return p.Submit(honest.worker_id, honest.step, honest.pairs, honest.f_minus, honest.f_plus, honest.digest)


def test_a_lying_worker_is_banned_over_tcp_and_the_rest_finish_exactly(tmp_path):
    config = quadratic(steps=6, pop=8)
    coordinator = Coordinator(config, Journal(tmp_path), audit=AuditPolicy(rate=0.25, probation=4))
    server = make_server("tcp", serialized(coordinator.handle), "127.0.0.1", 0).start()
    outcomes: dict[str, object] = {}

    def run(name: str, cls: type[Worker]) -> None:
        try:
            outcomes[name] = cls(TcpTransport("127.0.0.1", server.port), name, max_retry_seconds=5).run()
        except RuntimeError as e:
            outcomes[name] = e

    threads = [threading.Thread(target=run, args=(f"h{i}", Worker)) for i in range(3)]
    threads.append(threading.Thread(target=run, args=("liar", FlippingWorker)))
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    server.shutdown()
    assert coordinator.finished
    assert "banned" in str(outcomes["liar"])
    assert [outcomes[f"h{i}"] for i in range(3)] == [coordinator.state.digest()] * 3
    rebuilt = replay(initial_state(config), Journal(tmp_path).records())
    assert rebuilt.digest() == coordinator.state.digest(), "the journal still replays bit for bit"
