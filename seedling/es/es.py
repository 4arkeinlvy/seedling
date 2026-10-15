"""OpenAI-style evolution strategies with an ask/tell interface.

    es = ES(theta, Adam(lr=0.05), pop=32, sigma=0.1)
    for step in range(steps):
        seeds = es.ask()
        f_plus, f_minus = evaluate_batched(fitness, es.theta, seeds, es.sigma)
        es.tell(seeds, f_plus, f_minus)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch

from seedling.es.noise import noise
from seedling.es.optim import Optimizer

FitnessShaper = Callable[[np.ndarray], np.ndarray]
NoiseFn = Callable[[int, int], torch.Tensor]


def step_seeds(seed: int, step: int, pop: int) -> list[int]:
    """The seeds of one step: a pure function of (run seed, step), so anyone can rebuild them."""
    rng = np.random.default_rng([seed, step])
    return [int(s) for s in rng.integers(0, 2**63 - 1, size=pop, dtype=np.int64)]


def centered_ranks(fitness: np.ndarray) -> np.ndarray:
    """Replace each value by its rank, scaled to [-0.5, 0.5]: only the order matters."""
    fitness = np.asarray(fitness, dtype=np.float64).ravel()
    if fitness.size < 2:
        return np.zeros_like(fitness)
    ranks = np.empty(fitness.size)
    ranks[np.argsort(fitness, kind="stable")] = np.arange(fitness.size)
    return ranks / (fitness.size - 1) - 0.5


def raw_fitness(fitness: np.ndarray) -> np.ndarray:
    return np.asarray(fitness, dtype=np.float64).ravel()


def estimate_gradient(
    seeds: Sequence[int],
    f_plus: np.ndarray,
    f_minus: np.ndarray,
    sigma: float,
    dim: int,
    shaper: FitnessShaper = centered_ranks,
    noise_fn: NoiseFn = noise,
) -> torch.Tensor:
    """Antithetic ES estimate: g = 1/(2 N sigma) * sum_i (F+_i - F-_i) * eps_i."""
    n = len(seeds)
    shaped = shaper(np.concatenate([np.asarray(f_plus), np.asarray(f_minus)]))
    weights = shaped[:n] - shaped[n:]
    grad = torch.zeros(dim)
    for seed, weight in zip(seeds, weights, strict=True):
        grad.add_(noise_fn(seed, dim), alpha=float(weight))
    return grad / (2 * n * sigma)


class ES:
    def __init__(
        self,
        theta: torch.Tensor,
        optimizer: Optimizer,
        pop: int = 32,
        sigma: float = 0.05,
        seed: int = 0,
        shaper: FitnessShaper = centered_ranks,
    ):
        self.theta = theta.detach().to("cpu", torch.float32).clone()
        self.optimizer = optimizer
        self.pop = pop
        self.sigma = sigma
        self.seed = seed
        self.shaper = shaper
        self.step = 0

    def ask(self) -> list[int]:
        """The seeds for this step. Each seed is evaluated at theta + sigma*eps and theta - sigma*eps."""
        return step_seeds(self.seed, self.step, self.pop)

    def tell(self, seeds: Sequence[int], f_plus: np.ndarray, f_minus: np.ndarray) -> torch.Tensor:
        """Update theta from the scores of this step's seeds; returns the gradient estimate."""
        grad = estimate_gradient(seeds, f_plus, f_minus, self.sigma, self.theta.numel(), self.shaper)
        self.optimizer.step(self.theta, grad)
        self.step += 1
        return grad
