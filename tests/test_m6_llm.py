"""M6 spec: low-rank ES on a language model. Run: uv run pytest tests/test_m6_llm.py -x

Offline: a tiny random Qwen2 (built from a config, no download) and a stub codec.
The `llm` tests use the real Qwen2.5-0.5B-Instruct on the GPU: uv run pytest -m llm
"""

import copy

import numpy as np
import pytest
import torch

from seedling import _c
from seedling.exact import reference as ref
from seedling.exact.record import StepRecord
from seedling.exact.state import FixedPointState
from seedling.llm import countdown
from seedling.llm.lowrank import PerturbableModel, PopulationLinear, apply_lowrank, plan_layers, signs
from seedling.llm.workload import CountdownWorkload, linear_shapes

transformers = pytest.importorskip("transformers")


def tiny_qwen(seed: int = 0) -> torch.nn.Module:
    torch.manual_seed(seed)
    config = transformers.Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    return transformers.Qwen2ForCausalLM(config).float().eval()


class StubCodec:
    """Characters -> ids modulo the tiny vocabulary; enough to drive generate()."""

    pad_token_id = 0

    def encode(self, prompts):
        ids = [[1 + ord(c) % 60 for c in p[:12]] for p in prompts]
        width = max(len(x) for x in ids)
        input_ids = torch.tensor([[0] * (width - len(x)) + x for x in ids])
        return input_ids, (input_ids != 0).long()

    def decode(self, token_ids):
        return ["<answer> 1 + 2 </answer>" if row[0] % 2 else "no answer" for row in token_ids.tolist()]


# ------------------------------------------------------------------ Countdown


def test_generated_problems_are_solvable_and_reproducible():
    problems = countdown.generate(50, seed=3)
    assert problems == countdown.generate(50, seed=3)
    assert all(len(p.numbers) == 3 and 0 < p.target <= 400 for p in problems)


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("7 * 3 + 25", (True, True)),
        ("25 + 3 * 7", (True, True)),
        ("(25 - 3) * 7", (True, False)),
        ("7 * 3 + 25 + 0", (False, False)),
        ("3 + 7", (False, False)),
        ("7 ** 3 + 25", (False, False)),
        ("25 / 0 + 3 * 7", (False, False)),
        ('__import__("os").system("rm -rf /")', (False, False)),
        ("((((", (False, False)),
    ],
)
def test_answers_are_checked_without_eval(expression, expected):
    assert countdown.check(expression, countdown.Problem((3, 7, 25), 46)) == expected


def test_reward_levels():
    problem = countdown.Problem((3, 7, 25), 46)
    assert countdown.reward("work... <answer> 7*3+25 </answer>", problem) == 1.0
    assert 0.1 < countdown.reward("<answer> 3+7+25 </answer>", problem) < 0.5, "35 is close to 46"
    assert countdown.reward("<answer> 3*7*25 </answer>", problem) == 0.1, "525 is far from 46"
    assert countdown.reward("I don't know", problem) == 0.0
    assert countdown.extract_answer("<answer>1</answer> then <answer>2</answer>") == "2"
    assert countdown.extract_answer("Let me see.\n(5 - 1) * 8 = 40") == "(5 - 1) * 8"
    assert countdown.reward("(25 - 3) * 7 = 46", problem) == 0.1, "untagged answers count too"


# ------------------------------------------------------------------ exact low-rank update


def test_lowrank_c_matches_python_oracle():
    delta = np.arange(-90, 90, dtype=np.int32).reshape(12, 15)
    seeds, coefs = [3, 2**62, 77], [11, -4, 50_000]
    want = ref.apply_lowrank(delta, seeds, coefs, 8, 9)
    _c.apply_lowrank_i32(delta, np.array(seeds, np.uint64), np.array(coefs, np.int32), 8, 9)
    assert np.array_equal(delta, want)


def test_apply_lowrank_updates_each_layer_with_its_own_streams():
    specs = plan_layers([("a", 4, 6), ("b", 5, 3)])
    state = FixedPointState(np.zeros(4 * 6 + 5 * 3, np.int32))
    apply_lowrank(state, StepRecord(0, (42,), (10,)), specs)
    first = state.values[:24].reshape(4, 6)
    a, b = signs(42, 0, 6), signs(42, 1, 4)
    assert np.array_equal(first, 10 * np.outer(b, a).astype(np.int32))
    assert np.abs(state.values[24:]).sum() > 0, "the second layer moves too, with its own streams"


# ------------------------------------------------------------------ the population trick


def test_population_linear_equals_explicit_perturbed_weights():
    torch.manual_seed(0)
    base = torch.nn.Linear(6, 4)
    layer = PopulationLinear(base)
    x = torch.randn(3, 5, 6)
    a, b = torch.randn(3, 6).sign(), torch.randn(3, 4).sign()
    scale = torch.tensor([0.1, -0.1, 0.05])
    layer.set_population(a, b, scale)
    got = layer(x)
    for r in range(3):
        weight = base.weight + scale[r] * torch.outer(b[r], a[r])
        assert torch.allclose(got[r], torch.nn.functional.linear(x[r], weight, base.bias), atol=1e-5)


def test_one_batched_forward_equals_each_member_alone():
    model = tiny_qwen()
    specs = plan_layers(linear_shapes(model, last_layers=2, modules=("self_attn.o_proj", "mlp.down_proj")))
    reference = copy.deepcopy(model)
    net = PerturbableModel(model, specs)
    seeds, sigma, repeat = [5, 9], 0.05, 2
    tokens = torch.randint(1, 64, (repeat, 7), generator=torch.Generator().manual_seed(1))
    with torch.no_grad(), net.population(seeds, sigma, repeat):
        batched = model(tokens.repeat(2 * len(seeds), 1)).logits

    member = 0
    for seed in seeds:
        for sign in (1.0, -1.0):
            alone = copy.deepcopy(reference)
            with torch.no_grad():
                for spec in specs:
                    linear = alone.get_submodule(spec.name)
                    a, b = signs(seed, spec.streams[0], spec.cols), signs(seed, spec.streams[1], spec.rows)
                    linear.weight += sign * sigma * torch.outer(b, a)
                logits = alone(tokens).logits
            rows = slice(member * repeat, (member + 1) * repeat)
            assert torch.allclose(batched[rows], logits, atol=1e-4), f"member {member} differs"
            member += 1


def test_load_writes_base_plus_delta():
    model = tiny_qwen()
    specs = plan_layers(linear_shapes(model, 1, ("mlp.down_proj",)))
    original = model.get_submodule(specs[0].name).weight.detach().clone()
    net = PerturbableModel(model, specs)
    state = FixedPointState(np.zeros(specs[0].size, np.int32), frac_bits=10)
    apply_lowrank(state, StepRecord(0, (1,), (3,)), specs)
    net.load(state)
    delta = torch.from_numpy(state.values.reshape(specs[0].rows, specs[0].cols)).float() * 2**-10
    assert torch.allclose(net.layers[0].base.weight, original + delta)


# ------------------------------------------------------------------ the workload end to end (tiny)


def test_countdown_workload_scores_a_population():
    workload = CountdownWorkload(
        last_layers=1, prompts_per_step=3, max_new_tokens=4, device="cpu", model=tiny_qwen(), codec=StubCodec()
    )
    state = workload.initial_state(seed=0, frac_bits=20)
    assert state.values.size == sum(s.size for s in workload.specs) and not state.values.any()
    f_plus, f_minus = workload.score(state, step=0, seeds=[1, 2, 3], sigma=0.01)
    assert f_plus.shape == f_minus.shape == (3,)
    assert np.all((0 <= f_plus) & (f_plus <= 1))
    metrics = CountdownWorkload(
        last_layers=1, eval_problems=4, max_new_tokens=4, device="cpu", model=tiny_qwen(), codec=StubCodec()
    ).evaluate(state, step=0)
    assert set(metrics) == {"accuracy", "well_formed"}


# ------------------------------------------------------------------ real Qwen (GPU, opt-in)


@pytest.fixture(scope="module")
def qwen(cuda):
    return CountdownWorkload(prompts_per_step=4, max_new_tokens=96, eval_problems=8, eval_batch=8)


@pytest.mark.llm
def test_qwen_layer_shapes_come_from_the_config_alone(qwen):
    assert len(qwen.specs) == 16
    assert {(s.rows, s.cols) for s in qwen.specs} == {(896, 896), (896, 4864)}


@pytest.mark.llm
def test_qwen_scores_and_evaluates(qwen):
    state = qwen.initial_state(seed=0, frac_bits=24)
    f_plus, f_minus = qwen.score(state, step=0, seeds=[1, 2], sigma=0.002)
    assert f_plus.shape == (2,) and np.all((0 <= f_plus) & (f_plus <= 1))
    metrics = qwen.evaluate(state, step=0)
    assert 0 <= metrics["accuracy"] <= metrics["well_formed"] <= 1
