"""M9 spec: bigger workloads and the defense baselines they are compared with.

Run: uv run pytest tests/test_m9_scale.py -x
"""

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


# ------------------------------------------------------------------ replicate + median (CyBeR-0-style baseline)


MEDIAN = AuditPolicy(aggregate="median", replicas=3)


def test_median_mode_needs_every_replica_and_bans_nobody(tmp_path):
    c = Coordinator(quadratic(pop=2, pairs_per_task=2), Journal(tmp_path), audit=MEDIAN)
    ids = [c.handle(p.Hello(n)).worker_id for n in ("liar", "h1", "h2")]
    truth = [(5.0, 1.0), (2.0, 3.0)]
    for k, worker in enumerate(ids):
        work = c.handle(p.GetWork(worker))
        assert sorted(work.pairs) == [0, 1], "each replica goes to a different worker"
        lie = k == 0
        submit(c, worker, work.pairs, [truth[i][::-1] if lie else truth[i] for i in work.pairs])
        if k < 2:
            assert c.step == 0, "fewer than 3 opinions: nothing settles"
    assert c.step == 1
    assert c.ledger.banned == {}, "the median outvotes a liar without banning anyone"
    assert isinstance(c.handle(p.GetWork(ids[0])), p.Work)


def test_median_mode_rejects_bad_settings():
    with pytest.raises(ValueError):
        AuditPolicy(aggregate="mean")
    with pytest.raises(ValueError):
        AuditPolicy(aggregate="median", replicas=0)


def test_median_replication_costs_its_replica_count_and_stops_a_minority():
    config = quadratic(steps=10, pop=16)
    workers = ["honest"] * 8 + ["flip"] * 2
    result = simulate(config, workers, MEDIAN, seed=1)
    undefended = simulate(config, workers, seed=1)
    assert result.finished and result.banned(honest=True) == result.banned(honest=False) == 0
    assert result.cost == pytest.approx(3.0)
    # A lie passes only when both flippers score the same pair: C(2,2)*C(8,1)/C(10,3) = 6.7%,
    # against the 20% of pairs the flippers own without a defense.
    assert result.lies_committed < 0.5 * undefended.lies_committed


# ------------------------------------------------------------------ MuJoCo locomotion


def _gym_env(gym_id: str):
    gym = pytest.importorskip("gymnasium")
    pytest.importorskip("mujoco")
    return gym.make(gym_id).unwrapped


@pytest.mark.parametrize("name, gym_id", [("halfcheetah", "HalfCheetah-v5"), ("hopper", "Hopper-v5")])
def test_simulator_matches_gymnasium_v5_bit_for_bit(name, gym_id):
    import numpy as np

    from seedling.tasks.mujoco import ENVIRONMENTS, Simulator

    env = _gym_env(gym_id)
    obs, _ = env.reset(seed=3)
    sim = Simulator(ENVIRONMENTS[name])
    data = sim.datas(1)[0]
    sim.set_state(data, env.data.qpos.copy(), env.data.qvel.copy())
    assert np.array_equal(sim.observe(data), obs)
    rng = np.random.default_rng(0)
    for _ in range(200):
        action = rng.uniform(-1, 1, size=sim.act_dim)
        obs, reward, terminated, _, _ = env.step(action)
        assert sim.step(data, action) == (reward, terminated)
        assert np.array_equal(sim.observe(data), obs)
        if terminated:
            break


def test_a_policy_scores_the_same_whatever_else_is_in_its_batch():
    """Workers score different subsets of a step's pairs: scores must not depend on that."""
    pytest.importorskip("mujoco")
    import torch

    from seedling.tasks.mujoco import LocomotionTask

    task = LocomotionTask("halfcheetah", episodes=2, horizon=40)
    thetas = torch.stack([task.init(s) for s in range(6)])
    full = task.fitness(thetas, step=4)
    assert torch.equal(task.fitness(thetas[2:5], step=4), full[2:5])
    assert torch.equal(task.fitness(thetas, step=4), full), "deterministic"
    assert not torch.equal(task.fitness(thetas, step=5), full), "the step picks the start states"


def test_locomotion_trains_through_the_runtime_and_replays_exactly():
    pytest.importorskip("mujoco")
    from seedling.runtime.local import train_local
    from seedling.runtime.tasks import create_workload
    from seedling.runtime.workload import replay_workload

    config = RunConfig(task="hopper", steps=3, pop=4, sigma=0.05, lr=0.05, task_args={"horizon": 30})
    workload = create_workload(config.task, **config.task_args)
    run = train_local(workload, config)
    again = train_local(create_workload(config.task, **config.task_args), config)
    assert run.state.digest() == again.state.digest()
    assert replay_workload(workload, run.initial, run.records).digest() == run.state.digest()


# ------------------------------------------------------------------ SST-2 by label-word likelihood (offline: tiny Qwen)


def _tiny_qwen(seed: int = 0):
    import torch

    transformers = pytest.importorskip("transformers")
    torch.manual_seed(seed)
    config = transformers.Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
    )  # fmt: skip
    return transformers.Qwen2ForCausalLM(config).float().eval()


class _CharCodec:
    """Characters -> ids in the tiny vocabulary, left-padded like the real codec."""

    pad_token_id = 0

    def encode(self, prompts):
        import torch

        ids = [[1 + ord(c) % 60 for c in p[-20:]] for p in prompts]
        width = max(len(x) for x in ids)
        input_ids = torch.tensor([[0] * (width - len(x)) + x for x in ids])
        return input_ids, (input_ids != 0).long()

    def decode(self, token_ids):
        return ["" for _ in token_ids]


def _sst2(model=None, **overrides):
    from seedling.llm.classify import Example, Sst2Workload

    train = [Example(f"movie number {i} was {'fine' if i % 2 else 'bad'}", i % 2) for i in range(40)]
    args = dict(
        last_layers=2, examples_per_step=4, eval_examples=8, device="cpu", model=model or _tiny_qwen(),
        codec=_CharCodec(), train=train, validation=train[:8], label_token_ids=(5, 9),
    )  # fmt: skip
    args.update(overrides)
    return Sst2Workload(**args)


def test_sst2_prompts_follow_the_mezo_template():
    from seedling.llm.classify import Example

    assert Example("  a gripping film ", 1).prompt() == "a gripping film It was"


def test_sst2_population_scores_equal_each_member_scored_alone():
    import copy

    import numpy as np
    import torch

    from seedling.llm.lowrank import signs

    model = _tiny_qwen()
    reference = copy.deepcopy(model)
    workload = _sst2(model)
    state = workload.initial_state(seed=0, frac_bits=20)
    seeds, sigma = [3, 8], 0.05
    f_plus, f_minus = workload.score(state, step=2, seeds=seeds, sigma=sigma)
    assert f_plus.shape == f_minus.shape == (2,) and np.all(f_plus <= 0) and np.all(f_minus <= 0)

    examples = workload.step_examples(2)
    labels = torch.tensor([e.label for e in examples])
    for k, seed in enumerate(seeds):
        for sign, got in ((1.0, f_plus[k]), (-1.0, f_minus[k])):
            alone = copy.deepcopy(reference)
            with torch.no_grad():
                for spec in workload.specs:
                    a, b = signs(seed, spec.streams[0], spec.cols), signs(seed, spec.streams[1], spec.rows)
                    alone.get_submodule(spec.name).weight += sign * sigma * torch.outer(b, a)
            solo = _sst2(alone)
            want = float(solo._label_logprobs([e.prompt() for e in examples], labels).double().mean())
            assert got == pytest.approx(want, abs=1e-5)


def test_sst2_scores_do_not_depend_on_the_other_members_of_the_batch():
    import numpy as np

    workload = _sst2(max_rows=8)  # 4 examples x 2 signs: one member per chunk
    state = workload.initial_state(seed=0, frac_bits=20)
    f_plus, f_minus = workload.score(state, step=1, seeds=[1, 2, 3], sigma=0.05)
    sub_plus, sub_minus = _sst2(_tiny_qwen(), max_rows=64).score(state, step=1, seeds=[2], sigma=0.05)
    assert np.allclose(sub_plus, f_plus[1:2], rtol=1e-5) and np.allclose(sub_minus, f_minus[1:2], rtol=1e-5)


def test_sst2_trains_and_evaluates_through_the_runtime_machinery():
    from seedling.runtime.local import train_local

    workload = _sst2()
    config = RunConfig(task="sst2", steps=2, pop=3, sigma=0.05, lr=1e-3, frac_bits=24)
    run = train_local(workload, config, eval_every=2)
    assert run.records and set(run.history[-1]) >= {"accuracy", "label_logprob"}
    assert 0 <= run.history[-1]["accuracy"] <= 1
    assert workload.language_loss(run.state, ["some held out text", "more text here"]) > 0


# ------------------------------------------------------------------ Countdown with the TinyZero protocol (Qiu et al.)


def test_tinyzero_score_is_strict_about_tags_and_numbers():
    from seedling.llm import countdown

    problem = countdown.Problem((44, 19, 35), 98)
    assert countdown.tinyzero_reward("... <answer> 44 + 19 + 35 </answer>", problem) == 1.0
    assert countdown.tinyzero_reward("<answer> 44 + 19 </answer>", problem) == 0.1, "any answer: format credit"
    assert countdown.tinyzero_reward("44 + 19 + 35", problem) == 0.0, "no tags, no credit"
    assert countdown.tinyzero_prompt(problem).startswith("Using the numbers [44, 19, 35], create an equation that equals 98.")
    assert countdown.tinyzero_rows({"nums": [[1, 2], [3, 4, 5]], "target": [3, 12]})[1] == countdown.Problem((3, 4, 5), 12)


def test_tinyzero_protocol_holds_out_the_last_problems():
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from test_m6_llm import StubCodec, tiny_qwen

    from seedling.llm import countdown
    from seedling.llm.workload import CountdownWorkload

    problems = [countdown.Problem((i, 1, 2), i + 3) for i in range(1, 41)]
    w = CountdownWorkload(
        last_layers=1, prompts_per_step=3, max_new_tokens=4, eval_problems=8, device="cpu",
        model=tiny_qwen(), codec=StubCodec(), protocol="tinyzero", problems=problems,
    )  # fmt: skip
    assert w.held_out_problems() == problems[-8:]
    for step in range(20):
        assert not set(w.step_problems(step)) & set(problems[-8:]), "training never sees held-out problems"
    assert w.step_problems(5) == w.step_problems(5)
    f_plus, f_minus = w.score(w.initial_state(0, 20), step=0, seeds=[1, 2], sigma=0.01)
    assert f_plus.shape == (2,)


# ------------------------------------------------------------------ journaled local runs (GPU experiments resume)


def test_a_journaled_local_run_resumes_bit_for_bit(tmp_path):
    from seedling.runtime.local import train_local
    from seedling.runtime.tasks import create_workload

    short, full = quadratic(steps=3, pop=6), quadratic(steps=5, pop=6)
    train_local(create_workload("quadratic"), short, journal=Journal(tmp_path))
    fresh = train_local(create_workload("quadratic"), full)
    assert len(Journal(tmp_path).records()) == 3
    again = train_local(create_workload("quadratic"), full, journal=Journal(tmp_path))
    assert len(Journal(tmp_path).records()) == 5 and len(again.history) == 2, "only the missing steps ran"
    assert again.state.digest() == fresh.state.digest(), "extending a run changes nothing else"


# ------------------------------------------------------------------ a real lying worker (for experiments)


def test_the_lying_worker_is_caught_by_a_real_coordinator(tmp_path):
    import threading

    from bench.gpu.dishonest_worker import LyingWorker
    from seedling.runtime import Worker, make_server, serialized
    from seedling.runtime.transport import TcpTransport

    config = quadratic(steps=4, pop=8)
    coordinator = Coordinator(config, Journal(tmp_path), audit=AuditPolicy(rate=0.2, probation=4))
    server = make_server("tcp", serialized(coordinator.handle), "127.0.0.1", 0).start()
    outcome: dict[str, object] = {}

    def run(name, make):
        try:
            outcome[name] = make(TcpTransport("127.0.0.1", server.port), name).run()
        except RuntimeError as e:
            outcome[name] = e

    workers = [("h0", Worker), ("h1", Worker), ("h2", Worker)]
    workers.append(("lazy", lambda t, n: LyingWorker(t, n, behaviour="lazy", max_retry_seconds=5)))
    threads = [threading.Thread(target=run, args=w) for w in workers]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    server.shutdown()
    assert coordinator.finished and "banned" in str(outcome["lazy"])
