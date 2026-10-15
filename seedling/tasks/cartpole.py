"""M3: CartPole from its physics equations (no gym), vectorized: P policies x E episodes at once.

Same constants and rules as gym's CartPole-v1: push the cart left or right, keep the pole
within 12 degrees and the cart within 2.4 m, at most 500 steps, reward 1 per step.

    run:  uv run python -m seedling.tasks.es_demo --task cartpole
"""

from __future__ import annotations

import math

import numpy as np
import torch

from seedling.es import ES, Adam, BatchedMLP, evaluate_batched

GRAVITY = 9.8
MASS_CART = 1.0
MASS_POLE = 0.1
HALF_POLE = 0.5
FORCE = 10.0
DT = 0.02
X_LIMIT = 2.4
ANGLE_LIMIT = math.radians(12)
MAX_STEPS = 500

TOTAL_MASS = MASS_CART + MASS_POLE
POLE_MOMENT = MASS_POLE * HALF_POLE

POLICY = BatchedMLP([4, 16, 2])


def physics_step(state: torch.Tensor, push_right: torch.Tensor) -> torch.Tensor:
    """One Euler step. state (..., 4) = (x, x_dot, angle, angle_dot)."""
    x, x_dot, angle, angle_dot = state.unbind(-1)
    force = torch.where(push_right, FORCE, -FORCE)
    cos, sin = torch.cos(angle), torch.sin(angle)
    temp = (force + POLE_MOMENT * angle_dot**2 * sin) / TOTAL_MASS
    angle_acc = (GRAVITY * sin - cos * temp) / (HALF_POLE * (4 / 3 - MASS_POLE * cos**2 / TOTAL_MASS))
    x_acc = temp - POLE_MOMENT * angle_acc * cos / TOTAL_MASS
    return torch.stack(
        [x + DT * x_dot, x_dot + DT * x_acc, angle + DT * angle_dot, angle_dot + DT * angle_acc],
        dim=-1,
    )


def has_failed(state: torch.Tensor) -> torch.Tensor:
    return (state[..., 0].abs() > X_LIMIT) | (state[..., 2].abs() > ANGLE_LIMIT)


def episode_returns(thetas: torch.Tensor, episodes: int = 5, seed: int = 0) -> torch.Tensor:
    """Average return of each of the P policies in thetas (P, d), on the same E start states."""
    generator = torch.Generator().manual_seed(seed)
    start = (torch.rand(episodes, 4, generator=generator) - 0.5) * 0.1
    state = start.expand(thetas.shape[0], episodes, 4).clone()
    alive = torch.ones(state.shape[:2], dtype=torch.bool)
    returns = torch.zeros(state.shape[:2])
    for _ in range(MAX_STEPS):
        logits = POLICY(thetas, state)
        new_state = physics_step(state, logits[..., 1] > logits[..., 0])
        returns += alive
        state = torch.where(alive.unsqueeze(-1), new_state, state)
        alive &= ~has_failed(state)
        if not alive.any():
            break
    return returns.mean(dim=1)


def train_es(
    generations: int = 60,
    pop: int = 32,
    sigma: float = 0.1,
    lr: float = 0.05,
    episodes: int = 5,
    seed: int = 0,
) -> tuple[ES, list[dict]]:
    theta = POLICY.init(torch.Generator().manual_seed(seed))
    es = ES(theta, Adam(lr=lr), pop=pop, sigma=sigma, seed=seed)
    history = []
    for generation in range(generations):
        seeds = es.ask()
        f_plus, f_minus = evaluate_batched(
            lambda thetas: episode_returns(thetas, episodes, seed=generation), es.theta, seeds, sigma
        )
        es.tell(seeds, f_plus, f_minus)
        score = episode_returns(es.theta[None], episodes=20, seed=10_000 + generation)[0]
        history.append(
            {
                "step": generation,
                "population_mean": float(np.mean(np.concatenate([f_plus, f_minus]))),
                "policy_return": float(score),
            }
        )
    return es, history
