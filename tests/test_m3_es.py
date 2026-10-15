"""M3 spec: evolution strategies. Run: uv run pytest tests/test_m3_es.py -x"""

import math

import numpy as np
import pytest
import torch

from seedling.es import (
    ES,
    SGD,
    Adam,
    BatchedMLP,
    assign,
    centered_ranks,
    evaluate_batched,
    evaluate_inplace,
    flatten,
    noise,
    perturb_,
    raw_fitness,
)
from seedling.tasks import cartpole
from seedling.tasks.tiny_gpt import GPTConfig, TinyGPT


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(a @ b / (a.norm() * b.norm()))


# ------------------------------------------------------------------ building blocks


def test_noise_is_rebuilt_from_its_seed():
    assert torch.equal(noise(42, 1000), noise(42, 1000))
    assert not torch.equal(noise(42, 1000), noise(43, 1000))
    big = noise(7, 200_000)
    assert abs(big.mean().item()) < 0.01 and abs(big.std().item() - 1) < 0.01


def test_centered_ranks_keep_only_the_order():
    shaped = centered_ranks(np.array([10.0, -3.0, 1e9, 0.5]))
    assert np.allclose(shaped, [1 / 6, -0.5, 0.5, -1 / 6])
    assert abs(shaped.sum()) < 1e-12
    assert np.array_equal(np.argsort(shaped), np.argsort([10.0, -3.0, 1e9, 0.5]))


def test_es_gradient_points_along_the_true_gradient():
    dim = 20
    a = torch.linspace(-1, 1, dim)
    theta = torch.zeros(dim)
    es = ES(theta, SGD(lr=0.0), pop=2000, sigma=0.1, shaper=raw_fitness)
    seeds = es.ask()
    f_plus, f_minus = evaluate_batched(lambda t: t @ a, theta, seeds, es.sigma)
    grad = es.tell(seeds, f_plus, f_minus)
    assert cosine(grad, a) > 0.98, "for F(theta) = a . theta the ES estimate should be about a"


def test_es_finds_the_optimum_of_a_quadratic():
    target = torch.linspace(-2, 2, 10)
    es = ES(torch.zeros(10), Adam(lr=0.05), pop=16, sigma=0.1)
    start = (es.theta - target).norm().item()
    for _ in range(300):
        seeds = es.ask()
        f_plus, f_minus = evaluate_batched(
            lambda t: -((t - target) ** 2).sum(dim=1), es.theta, seeds, es.sigma
        )
        es.tell(seeds, f_plus, f_minus)
    assert (es.theta - target).norm().item() < 0.05 * start


def test_optimizers_step_uphill():
    theta = torch.zeros(3)
    SGD(lr=0.5).step(theta, torch.tensor([1.0, -2.0, 0.0]))
    assert torch.equal(theta, torch.tensor([0.5, -1.0, 0.0]))
    theta = torch.zeros(3)
    Adam(lr=0.1).step(theta, torch.tensor([3.0, -0.01, 5.0]))
    assert torch.allclose(theta, torch.tensor([0.1, -0.1, 0.1]), atol=1e-5), "Adam's 1st step ~ lr*sign"


def test_batched_mlp_equals_one_model_at_a_time():
    mlp = BatchedMLP([5, 7, 3])
    thetas = torch.randn(4, mlp.size, generator=torch.Generator().manual_seed(0))
    x = torch.randn(6, 5, generator=torch.Generator().manual_seed(1))
    together = mlp(thetas, x)
    for p in range(4):
        assert torch.allclose(together[p], mlp(thetas[p : p + 1], x)[0], atol=1e-6)


def test_assign_copies_instead_of_sharing_memory():
    torch.manual_seed(0)
    model = TinyGPT(GPTConfig(vocab_size=10, block_size=8))
    theta = flatten(model) + 1.0
    assign(model, theta)
    before = theta.clone()
    with torch.no_grad():
        next(model.parameters()).add_(5.0)
    assert torch.equal(theta, before), "changing the model must not change theta"


def test_inplace_evaluation_perturbs_then_restores_exactly():
    torch.manual_seed(0)
    model = TinyGPT(GPTConfig(vocab_size=10, block_size=8))
    theta = flatten(model)
    theta_before = theta.clone()
    seen = []
    f_plus, f_minus = evaluate_inplace(
        model, lambda m: seen.append(flatten(m)) or 0.0, theta, seeds=[1, 2, 3], sigma=0.05
    )
    assert len(seen) == 6 and len(f_plus) == len(f_minus) == 3
    assert torch.allclose(seen[0] - theta, 0.05 * noise(1, theta.numel()), atol=1e-6)
    assert torch.allclose(seen[1] - theta, -0.05 * noise(1, theta.numel()), atol=1e-6)
    assert torch.equal(theta, theta_before), "evaluation must never touch theta"
    assert torch.equal(flatten(model), theta), "the model must end holding exactly theta"


def test_add_then_subtract_in_float32_drifts_a_little():
    weights = torch.randn(100_000, generator=torch.Generator().manual_seed(0))
    w = weights.clone()
    for seed in range(16):
        eps = noise(seed, w.numel())
        perturb_([w], eps, 0.05)
        perturb_([w], eps, -0.05)
    drift = (w - weights).abs().max().item()
    assert 0 < drift < 1e-5, f"drift {drift}: rounding should leave a tiny, non-zero error"


def test_two_runs_with_the_same_seed_are_bit_identical():
    a, _ = cartpole.train_es(generations=3, seed=5)
    b, _ = cartpole.train_es(generations=3, seed=5)
    assert torch.equal(a.theta, b.theta)


# ------------------------------------------------------------------ CartPole physics


def test_one_physics_step_matches_the_equations():
    x, x_dot, angle, angle_dot = 0.01, 0.02, 0.05, -0.03
    force = 10.0
    temp = (force + 0.05 * angle_dot**2 * math.sin(angle)) / 1.1
    angle_acc = (9.8 * math.sin(angle) - math.cos(angle) * temp) / (
        0.5 * (4 / 3 - 0.1 * math.cos(angle) ** 2 / 1.1)
    )
    x_acc = temp - 0.05 * angle_acc * math.cos(angle) / 1.1
    want = [x + 0.02 * x_dot, x_dot + 0.02 * x_acc, angle + 0.02 * angle_dot, angle_dot + 0.02 * angle_acc]
    got = cartpole.physics_step(torch.tensor([x, x_dot, angle, angle_dot], dtype=torch.float64), torch.tensor(True))
    assert torch.allclose(got, torch.tensor(want, dtype=torch.float64), atol=1e-12)


def test_episode_ends_when_the_pole_or_cart_goes_too_far():
    assert cartpole.has_failed(torch.tensor([0.0, 0.0, 0.21, 0.0]))
    assert cartpole.has_failed(torch.tensor([2.5, 0.0, 0.0, 0.0]))
    assert not cartpole.has_failed(torch.tensor([2.3, 0.0, 0.2, 0.0]))


def test_es_learns_to_balance_the_pole():
    es, history = cartpole.train_es(generations=30, seed=0)
    assert history[0]["policy_return"] < 100, "the random starting policy should be bad"
    assert history[-1]["policy_return"] >= 475, "ES should balance the pole (max 500)"


# ------------------------------------------------------------------ MNIST and tiny GPT


@pytest.fixture(scope="module")
def mnist_data():
    from seedling.tasks import mnist

    try:
        return mnist.load()
    except ConnectionError as e:
        pytest.skip(f"MNIST not cached and download failed: {e}")


def test_es_learns_mnist(mnist_data):
    from seedling.tasks import mnist

    _, history = mnist.train_es(mnist_data, steps=300, eval_every=300)
    assert history[-1]["test_accuracy"] >= 0.80


def test_es_lowers_the_tiny_gpt_loss():
    from seedling.tasks.es_gpt import train_es_gpt

    _, history = train_es_gpt(steps=60, device="cpu")
    first = np.mean([r["train_loss"] for r in history[:5]])
    last = np.mean([r["train_loss"] for r in history[-5:]])
    assert last < first - 0.5, f"loss {first:.2f} -> {last:.2f}"
