"""M10 spec: what a production run needs. Run: uv run pytest tests/test_m10_production.py -x"""

import pytest

from seedling.runtime import Coordinator, Journal, RunConfig
from seedling.runtime import protocol as p
from seedling.runtime.audit import AuditPolicy
from seedling.runtime.simulate import simulate


def quadratic(steps: int = 3, pop: int = 4, **overrides) -> RunConfig:
    args = dict(task="quadratic", steps=steps, pop=pop, sigma=0.1, lr=0.1, pairs_per_task=2, deadline=5.0)
    args.update(overrides)
    return RunConfig(**args)


def submit(c: Coordinator, worker: str, pairs: list[int], scores: list[tuple[float, float]]) -> p.Message:
    f_plus, f_minus = [s[0] for s in scores], [s[1] for s in scores]
    return c.handle(p.Submit(worker, c.step, pairs, f_plus, f_minus, c.state.digest()))


def true_scores(c: Coordinator, pairs: list[int]) -> list[tuple[float, float]]:
    f_plus, f_minus = c.workload.score(c.state, c.step, [c.seeds[i] for i in pairs], c.config.sigma)
    return list(zip(f_plus.tolist(), f_minus.tolist(), strict=True))


# ------------------------------------------------------------------ the referee (Verde-style)


REFEREE = AuditPolicy(probation=100, referee="self")


def test_the_referee_settles_a_dispute_so_two_liars_cannot_frame_an_honest_worker(tmp_path):
    c = Coordinator(quadratic(pop=1, pairs_per_task=1), Journal(tmp_path), audit=REFEREE)
    honest, liar1, liar2 = (c.handle(p.Hello(n)).worker_id for n in ("honest", "liar1", "liar2"))
    pairs = c.handle(p.GetWork(honest)).pairs
    truth = true_scores(c, pairs)
    submit(c, honest, pairs, truth)
    assert c.handle(p.GetWork(liar1)).pairs == pairs, "the audit"
    submit(c, liar1, pairs, [(fm, fp) for fp, fm in truth])
    assert c.step == 1, "the referee settled the dispute at once: no third opinion"
    assert liar1 in c.ledger.banned and honest not in c.ledger.banned
    assert c.referee_scored == 1
    assert isinstance(c.handle(p.GetWork(liar2)), p.Work), "liar2 never got the chance to outvote anyone"


def test_agreement_needs_no_referee(tmp_path):
    c = Coordinator(quadratic(pop=2), Journal(tmp_path), audit=REFEREE)
    a, b = (c.handle(p.Hello(n)).worker_id for n in ("a", "b"))
    pairs = c.handle(p.GetWork(a)).pairs
    submit(c, a, pairs, true_scores(c, pairs))
    submit(c, b, c.handle(p.GetWork(b)).pairs, true_scores(c, pairs))
    assert c.step == 1 and c.referee_scored == 0


def test_with_a_referee_colluding_sleepers_frame_nobody():
    """M8 measured 1-4 honest bans per 3 runs here (sleepers outvoting honest workers 2 to 1)."""
    config = quadratic(steps=30, pop=16)
    behaviours = ["honest"] * 6 + ["sleeper"] * 4
    policy = AuditPolicy(rate=0.1, probation=4, referee="self")
    for seed in range(3):
        result = simulate(config, behaviours, policy, seed=seed, sleeper_after=8)
        assert result.finished
        assert result.banned(honest=True) == 0
        assert result.banned(honest=False) == 4


def test_referee_setting_is_checked():
    with pytest.raises(ValueError):
        AuditPolicy(referee="anyone")


# ------------------------------------------------------------------ join tokens


SECRET_A, SECRET_B = "a" * 32, "b" * 32


@pytest.fixture
def tokens(tmp_path):
    path = tmp_path / "tokens.txt"
    path.write_text(f"# who may join\nalice:{SECRET_A}\nbob:{SECRET_B}  # bob's laptop\n")
    return path


def test_token_files_are_parsed_and_checked(tokens, tmp_path):
    from seedling.runtime.access import Access

    access = Access.from_file(tokens)
    assert access.required and access.name_of(SECRET_A) == "alice" and access.name_of("nope") is None
    assert SECRET_A not in str(access.tokens), "only hashes are kept"
    bad = tmp_path / "bad.txt"
    bad.write_text("alice:short\n")
    with pytest.raises(ValueError):
        Access.from_file(bad)


def test_joining_needs_a_valid_token_and_a_lie_revokes_it(tokens, tmp_path):
    from seedling.runtime.access import Access

    access = Access.from_file(tokens, max_ids_per_token=2)
    config = quadratic(pop=1, pairs_per_task=1)
    c = Coordinator(config, Journal(tmp_path / "run"), audit=REFEREE, access=access)
    assert isinstance(c.handle(p.Hello("x", "wrong")), p.Error)
    assert isinstance(c.handle(p.Hello("x" * 10_000, SECRET_A)), p.Error), "names are short: they go to disk"
    honest = c.handle(p.Hello("gpu0", SECRET_A)).worker_id
    liar, liar2 = (c.handle(p.Hello(f"gpu{k}", SECRET_B)).worker_id for k in (1, 2))
    assert isinstance(c.handle(p.Hello("gpu3", SECRET_B)), p.Error), "at most 2 ids per token"

    pairs = c.handle(p.GetWork(honest)).pairs
    truth = true_scores(c, pairs)
    submit(c, honest, pairs, truth)
    c.handle(p.GetWork(liar))
    submit(c, liar, pairs, [(fm, fp) for fp, fm in truth])
    assert "bob" in c.ledger.revoked
    assert "token revoked" in c.handle(p.GetWork(liar2)).reason, "every id of the token is out"
    assert isinstance(c.handle(p.Hello("gpu9", SECRET_B)), p.Error), "and the token cannot join again"
    restarted = Coordinator(config, Journal(tmp_path / "run"), audit=REFEREE, access=access)
    assert isinstance(restarted.handle(p.GetWork(liar2)), p.Error), "revocations survive a restart"
    assert isinstance(restarted.handle(p.GetWork(honest)), p.Message)


def test_tokens_alone_turn_on_issued_ids(tokens, tmp_path):
    from seedling.runtime.access import Access

    c = Coordinator(quadratic(), Journal(tmp_path), access=Access.from_file(tokens))
    worker = c.handle(p.Hello("w", SECRET_A)).worker_id
    assert worker.startswith("w~") and isinstance(c.handle(p.GetWork("w")), p.Error)
    assert isinstance(c.handle(p.GetWork(worker)), p.Work)


# ------------------------------------------------------------------ TLS, frame limits, metrics (real sockets)


@pytest.fixture(scope="module")
def certificate(tmp_path_factory):
    import shutil
    import subprocess

    if shutil.which("openssl") is None:
        pytest.skip("needs the openssl command")
    d = tmp_path_factory.mktemp("tls")
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1", "-subj", "/CN=localhost",
         "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1",
         "-keyout", str(d / "key.pem"), "-out", str(d / "cert.pem")],
        check=True, capture_output=True,
    )  # fmt: skip
    return str(d / "cert.pem"), str(d / "key.pem")


def _run_workers(server, url: str, n: int, tls=None, token: str = "", shutdown: bool = True) -> list:
    import threading

    from seedling.runtime import Worker
    from seedling.runtime.transport import connect

    out: list = []

    def run(k):
        try:
            out.append(Worker(connect(url, tls), f"w{k}", max_retry_seconds=2, token=token).run())
        except (RuntimeError, ConnectionError) as e:
            out.append(e)

    threads = [threading.Thread(target=run, args=(k,)) for k in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    if shutdown:
        server.shutdown()
    return out


@pytest.mark.parametrize("scheme", ["tcps", "https"])
def test_workers_train_over_tls(tmp_path, certificate, scheme):
    from seedling.runtime import make_server, serialized
    from seedling.runtime.access import client_tls, server_tls

    c = Coordinator(quadratic(steps=4, pop=6), Journal(tmp_path))
    server = make_server(scheme, serialized(c.handle), "127.0.0.1", 0, tls=server_tls(*certificate)).start()
    digests = _run_workers(server, f"{scheme}://127.0.0.1:{server.port}", 2, tls=client_tls(certificate[0]))
    assert c.finished and digests == [c.state.digest()] * 2


def test_a_worker_that_cannot_verify_the_coordinator_does_not_talk_to_it(tmp_path, certificate):
    from seedling.runtime import make_server, serialized
    from seedling.runtime.access import server_tls

    c = Coordinator(quadratic(), Journal(tmp_path))
    server = make_server("tcps", serialized(c.handle), "127.0.0.1", 0, tls=server_tls(*certificate)).start()
    (outcome,) = _run_workers(server, f"tcps://127.0.0.1:{server.port}", 1)  # system CAs only
    assert isinstance(outcome, ConnectionError) and c.step == 0


def test_an_oversized_frame_is_refused_without_allocating_it(tmp_path):
    import socket
    import struct

    from seedling.foundation.wire import recv_msg, send_msg
    from seedling.runtime import make_server, serialized

    c = Coordinator(quadratic(), Journal(tmp_path))
    server = make_server("tcp", serialized(c.handle), "127.0.0.1", 0).start()
    with socket.create_connection(("127.0.0.1", server.port), timeout=5) as sock:
        sock.sendall(struct.pack("!Q", 2**62))
        assert sock.recv(1) == b"", "the server hung up"
    with socket.create_connection(("127.0.0.1", server.port), timeout=5) as sock:
        send_msg(sock, p.encode(p.Hello("w")))
        assert isinstance(p.decode(recv_msg(sock)), p.Welcome), "and keeps serving everyone else"
    server.shutdown()


def test_metrics_are_served_in_prometheus_format(tmp_path):
    import urllib.request

    from seedling.runtime import make_server, serialized

    c = Coordinator(quadratic(steps=4, pop=6), Journal(tmp_path), audit=AuditPolicy(probation=2))
    server = make_server("http", serialized(c.handle), "127.0.0.1", 0, metrics=c.metrics_text).start()
    url = f"http://127.0.0.1:{server.port}"
    digests = _run_workers(server, url, 3, shutdown=False)
    with urllib.request.urlopen(f"{url}/metrics", timeout=5) as response:
        content_type, text = response.headers["Content-Type"], response.read().decode()
    server.shutdown()
    assert c.finished and digests == [c.state.digest()] * 3
    assert content_type.startswith("text/plain")
    values = dict(line.split() for line in text.splitlines() if not line.startswith("#"))
    assert values["seedling_step"] == "4" and values["seedling_finished"] == "1"
    assert float(values["seedling_work_per_pair"]) > 1.0, "probation audits cost extra work"


# ------------------------------------------------------------------ export


def test_an_llm_run_exports_to_a_standard_hugging_face_folder(tmp_path):
    import numpy as np
    import torch

    transformers = pytest.importorskip("transformers")
    from seedling.llm.lowrank import delta_view
    from seedling.llm.workload import CountdownWorkload
    from seedling.runtime.export import export_run
    from seedling.runtime.local import train_local
    from tests.test_m6_llm import StubCodec, tiny_qwen

    def workload(*_, **__):
        return CountdownWorkload(
            last_layers=1, prompts_per_step=2, max_new_tokens=3, eval_problems=2, device="cpu", dtype="float32",
            model=tiny_qwen(), codec=StubCodec(),
        )  # fmt: skip

    config = RunConfig(task="countdown", steps=2, pop=2, sigma=0.05, lr=0.05, frac_bits=16)
    run = train_local(workload(), config, journal=Journal(tmp_path / "run"))
    summary = export_run(str(tmp_path / "run"), str(tmp_path / "out"), workload_factory=workload)
    assert summary["digest"] == run.state.digest() and summary["steps"] == 2

    exported = transformers.AutoModelForCausalLM.from_pretrained(tmp_path / "out")
    base = tiny_qwen()
    w = workload()
    for spec in w.specs:
        delta = torch.from_numpy(delta_view(run.state, spec).astype(np.float32)) * run.state.unit
        want = base.get_submodule(spec.name).weight + delta
        assert torch.allclose(exported.get_submodule(spec.name).weight, want), spec.name
    assert np.abs(run.state.values).sum() > 0, "the run did change the weights"
    out = exported.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=2, do_sample=False)
    assert out.shape == (1, 5)


def test_a_vector_run_exports_its_parameters(tmp_path):
    import numpy as np

    from seedling.runtime.export import export_run
    from seedling.runtime.local import train_local
    from seedling.runtime.tasks import create_workload

    config = quadratic(steps=3, pop=4)
    run = train_local(create_workload("quadratic"), config, journal=Journal(tmp_path / "run"))
    export_run(str(tmp_path / "run"), str(tmp_path / "out"))
    assert np.array_equal(np.load(tmp_path / "out" / "params.npy"), run.state.to_float().numpy())


def test_a_tinylm_run_exports_weights_and_vocabulary(tmp_path):
    import json

    from safetensors.torch import load_file

    from seedling.runtime.export import export_run
    from seedling.runtime.local import train_local
    from seedling.runtime.tasks import create_workload

    args = {"n_layer": 1, "n_embd": 32, "n_head": 2, "block_size": 16, "batch": 2}
    config = RunConfig(task="tinylm", steps=2, pop=2, sigma=0.01, lr=0.002, frac_bits=24, task_args=args)
    train_local(create_workload("tinylm", **args), config, journal=Journal(tmp_path / "run"))
    export_run(str(tmp_path / "run"), str(tmp_path / "out"))
    weights = load_file(tmp_path / "out" / "model.safetensors")
    assert "blocks.0.attn.qkv.weight" in weights and not any(".base." in k for k in weights), "plain Linear names"
    assert len(json.loads((tmp_path / "out" / "vocab.json").read_text())) == 62


# ------------------------------------------------------------------ audits by prefill (TOPLOC-style)


def test_evidence_round_trips_and_rejects_garbage():
    import base64
    import zlib

    import numpy as np

    from seedling.llm import evidence

    tokens = np.arange(12, dtype=np.int32).reshape(3, 4)
    assert np.array_equal(evidence.unpack(evidence.pack(tokens)), tokens)
    bomb = base64.b64encode(zlib.compress(np.array([64, 2**20], np.int32).tobytes() + bytes(2**28))).decode()
    assert len(bomb) < 2**20, "a 256 MB claim compresses to a few hundred KB"
    with pytest.raises(ValueError):
        evidence.unpack(bomb)  # refused without inflating it
    for bad in ["not base64!", base64.b64encode(b"short").decode(),
                base64.b64encode(zlib.compress(np.array([5, 5], np.int32).tobytes())).decode(),
                base64.b64encode(zlib.compress(np.array([10**6, 10**6], np.int32).tobytes())).decode()]:  # fmt: skip
        with pytest.raises(ValueError):
            evidence.unpack(bad)


def test_text_after_the_first_end_token_is_ignored():
    import numpy as np

    from seedling.llm import evidence

    tokens, valid = evidence.canonical(np.array([[5, 9, 7, 7], [5, 6, 6, 6]]), eos=[9], pad=0)
    assert tokens.tolist() == [[5, 9, 0, 0], [5, 6, 6, 6]]
    assert valid.tolist() == [[True, True, False, False], [True, True, True, True]]


def _countdown(model=None, **overrides):
    from seedling.llm.workload import CountdownWorkload
    from tests.test_m6_llm import StubCodec, tiny_qwen

    args = dict(
        last_layers=1, prompts_per_step=3, max_new_tokens=6, device="cpu", dtype="float32",
        model=model or tiny_qwen(), codec=StubCodec(),
    )  # fmt: skip
    args.update(overrides)
    return CountdownWorkload(**args)


def test_an_auditor_confirms_honest_text_with_one_forward_pass():
    import numpy as np

    worker, auditor = _countdown(), _countdown()
    state = worker.initial_state(0, 20)
    f_plus, f_minus, claims = worker.score_with_evidence(state, 0, [4, 9], 0.05)
    got_plus, got_minus, evidence = auditor.verify(state, 0, [4, 9], 0.05, claims)
    assert np.array_equal(got_plus, f_plus) and np.array_equal(got_minus, f_minus)
    assert (auditor.verified, auditor.regenerated) == (2, 0), "no generation needed"
    assert evidence == claims, "what it checked is its evidence"


def test_forged_or_truncated_text_fails_the_check():
    import numpy as np

    from seedling.llm import evidence

    worker, auditor = _countdown(), _countdown()
    state = worker.initial_state(0, 20)
    honest_plus, _, (claim,) = worker.score_with_evidence(state, 0, [4], 0.05)
    tokens = evidence.unpack(claim)
    with torch_no_grad():
        logits = _row_logits(auditor, state, tokens)
    forged = tokens.copy()
    forged[0, 2] = int(logits[0, 2].argmin())  # the least likely token: never within tolerance
    truncated = tokens[:, :-2]
    for bad in (evidence.pack(forged), evidence.pack(truncated), "garbage"):
        fresh = _countdown()
        assert fresh.check(state, 0, [4], 0.05, [bad]) == [None]
        got_plus, _, (own,) = fresh.verify(state, 0, [4], 0.05, [bad])
        assert fresh.regenerated == 1 and np.array_equal(got_plus, honest_plus), "it generated for itself"
        assert own != bad, "and sends its own evidence"


def torch_no_grad():
    import torch

    return torch.no_grad()


def _row_logits(workload, state, tokens):
    """Teacher-forced logits of the claimed tokens of seed 4 (the + rows and - rows)."""
    import torch

    problems = workload.step_problems(0)
    prompts = [workload.prompt(p) for p in problems] * 2
    net = workload.perturbable()
    net.load(state)
    ids, mask = workload.codec().encode(prompts)
    width = ids.shape[1]
    full = torch.cat([ids, torch.from_numpy(tokens).long()], 1)
    full_mask = torch.cat([mask, torch.ones_like(torch.from_numpy(tokens)).long()], 1)
    with net.population([4], 0.05, len(problems)):
        return workload.forward_logits(full, full_mask)[:, width - 1 : width - 1 + tokens.shape[1]]


def test_a_liar_sending_true_text_with_false_scores_is_caught_by_checking_auditors(tmp_path):
    import threading

    from bench.gpu.dishonest_worker import LyingWorker
    from seedling.runtime import Worker, make_server, serialized
    from seedling.runtime.transport import TcpTransport

    from seedling.llm import countdown

    made = []
    problems = [countdown.Problem((k, 2, 3), k + 5) for k in range(1, 30)]

    def factory(name, **args):  # TinyZero scoring: a tagged answer earns 0.1, so scores vary
        made.append(_countdown(protocol="tinyzero", problems=problems, eval_problems=4))
        return made[-1]

    config = RunConfig(task="countdown", steps=2, pop=4, sigma=0.5, lr=0.05, frac_bits=16, pairs_per_task=2)
    coordinator = Coordinator(
        config, Journal(tmp_path), workload_factory=factory, audit=AuditPolicy(probation=100, referee="self")
    )
    server = make_server("tcp", serialized(coordinator.handle), "127.0.0.1", 0).start()
    outcome: dict = {}

    def run(name, cls, **kw):
        try:
            outcome[name] = cls(TcpTransport("127.0.0.1", server.port), name, workload_factory=factory, **kw).run()
        except RuntimeError as e:
            outcome[name] = e

    threads = [threading.Thread(target=run, args=(f"h{k}", Worker)) for k in range(2)]
    threads.append(threading.Thread(target=run, args=("liar", LyingWorker), kwargs={"behaviour": "flip"}))
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    server.shutdown()
    assert coordinator.finished and "banned" in str(outcome["liar"])
    assert outcome["h0"] == outcome["h1"] == coordinator.state.digest()
    assert sum(w.verified for w in made) > 0, "audits of honest work were settled by checking, not generating"


class _CheckableQuadratic:
    """The quadratic workload plus a scripted `check`: evidence "A" and "B" grade to two
    different (both honest) score pairs, as two GPUs breaking a near-tie differently would."""

    GRADES = {"A": (1.0, 2.0), "B": (3.0, 4.0)}

    def __init__(self):
        from seedling.runtime.tasks import create_workload

        self.inner = create_workload("quadratic")
        self.name = "quadratic"

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def check(self, state, step, seeds, sigma, claims):
        return [self.GRADES.get(c) for c in claims]


def _submit_evidence(c, worker, pairs, scores, evidence):
    f_plus, f_minus = [s[0] for s in scores], [s[1] for s in scores]
    return c.handle(p.Submit(worker, c.step, pairs, f_plus, f_minus, c.state.digest(), evidence))


def test_the_referee_judges_each_opinion_by_its_own_evidence(tmp_path):
    config = quadratic(pop=1, pairs_per_task=1)
    c = Coordinator(config, Journal(tmp_path), workload_factory=lambda *a, **k: _CheckableQuadratic(), audit=REFEREE)
    h1, h2, liar = (c.handle(p.Hello(n)).worker_id for n in ("h1", "h2", "liar"))
    c.handle(p.GetWork(h1))
    _submit_evidence(c, h1, [0], [(1.0, 2.0)], ["A"])
    work = c.handle(p.GetWork(h2))
    assert work.claims == ["A"], "the audit carries h1's evidence"
    _submit_evidence(c, h2, [0], [(3.0, 4.0)], ["B"])  # generated its own, different, honest text
    assert c.step == 1 and not c.ledger.banned, "both texts hold up: nobody lied"

    c.handle(p.GetWork(h1))
    _submit_evidence(c, h1, [0], [(1.0, 2.0)], ["A"])
    c.handle(p.GetWork(liar))
    _submit_evidence(c, liar, [0], [(2.0, 1.0)], ["A"])  # true text, false scores
    assert list(c.ledger.banned) == [liar]


def test_a_late_answer_to_an_expired_assignment_is_not_an_audit(tmp_path):
    class Clock:
        now = 0.0

        def __call__(self):
            return self.now

    clock = Clock()
    c = Coordinator(quadratic(pop=1, pairs_per_task=1), Journal(tmp_path), clock=clock, audit=AuditPolicy(probation=100))
    slow, fast, auditor = (c.handle(p.Hello(n)).worker_id for n in ("slow", "fast", "auditor"))
    c.handle(p.GetWork(slow))
    clock.now = 10.0  # the deadline (5 s) passes: the pair goes out again, fresh
    assert c.handle(p.GetWork(fast)).claims == []
    submit(c, fast, [0], true_scores(c, [0]))
    submit(c, slow, [0], [(9.0, 9.0)])  # the slow worker's late answer to its fresh assignment
    assert len(c.opinions[0]) == 1 and not c.ledger.banned, "ignored, not counted as a disagreeing audit"
    c.handle(p.GetWork(auditor))
    submit(c, auditor, [0], true_scores(c, [0]))
    assert c.step == 1


# ------------------------------------------------------------------ the same base model everywhere


def test_a_fingerprint_names_the_base_weights_not_the_trained_delta():
    from seedling.llm.base import model_fingerprint
    from tests.test_m6_llm import tiny_qwen

    assert model_fingerprint("m", tiny_qwen(0)) == model_fingerprint("m", tiny_qwen(0))
    assert model_fingerprint("m", tiny_qwen(0)) != model_fingerprint("m", tiny_qwen(1))
    w = _countdown()
    before = w.fingerprint()
    state = w.initial_state(0, 20)
    state.values[:] = 7  # a trained Delta, loaded into the weights
    w.perturbable().load(state)
    assert w.fingerprint() == before, "training does not change which base model it is"


def test_a_worker_with_other_base_weights_is_refused(tmp_path):
    config = quadratic()
    c = Coordinator(config, Journal(tmp_path))
    assert isinstance(c.handle(p.GetWork("a", "model#1111")), p.Work), "the first fingerprint sets the run's"
    refused = c.handle(p.GetWork("b", "model#2222"))
    assert isinstance(refused, p.Error) and "different base model" in refused.reason
    assert isinstance(c.handle(p.GetWork("c")), p.Message), "workloads without a fingerprint send none"
    again = Coordinator(config, Journal(tmp_path))
    assert isinstance(again.handle(p.GetWork("b", "model#2222")), p.Error), "remembered across restarts"


def test_real_workers_with_different_models_cannot_mix(tmp_path):
    import threading

    from seedling.runtime import Worker, make_server, serialized
    from seedling.runtime.transport import TcpTransport
    from tests.test_m6_llm import StubCodec, tiny_qwen

    def factory_for(seed):
        def factory(name, **args):
            from seedling.llm.workload import CountdownWorkload

            return CountdownWorkload(
                last_layers=1, prompts_per_step=2, max_new_tokens=3, device="cpu", dtype="float32",
                model=tiny_qwen(seed), codec=StubCodec(),
            )  # fmt: skip

        return factory

    config = RunConfig(task="countdown", steps=1, pop=2, sigma=0.05, lr=0.05, frac_bits=16, pairs_per_task=2)
    coordinator = Coordinator(config, Journal(tmp_path), workload_factory=factory_for(0))
    server = make_server("tcp", serialized(coordinator.handle), "127.0.0.1", 0).start()
    outcome = {}

    def run(name, seed):
        try:
            worker = Worker(TcpTransport("127.0.0.1", server.port), name, workload_factory=factory_for(seed))
            outcome[name] = worker.run()
        except RuntimeError as e:
            outcome[name] = str(e)

    first = threading.Thread(target=run, args=("same", 0))
    first.start()
    first.join(timeout=60)
    other = threading.Thread(target=run, args=("other", 1))
    other.start()
    other.join(timeout=60)
    server.shutdown()
    assert outcome["same"] == coordinator.state.digest()
    assert "different base model" in outcome["other"]


def test_shutting_a_server_down_hangs_up_on_connected_workers(tmp_path):
    from seedling.runtime import make_server, serialized
    from seedling.runtime.transport import TcpTransport

    c = Coordinator(quadratic(), Journal(tmp_path))
    server = make_server("tcp", serialized(c.handle), "127.0.0.1", 0).start()
    transport = TcpTransport("127.0.0.1", server.port, timeout=5)
    assert isinstance(transport.request(p.Hello("w")), p.Welcome), "connected"
    server.shutdown()
    with pytest.raises(ConnectionError):
        transport.request(p.GetWork("w"))  # before the fix, the open connection was still served
