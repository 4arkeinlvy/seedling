"""Two ways to score the population of one ES step.

evaluate_batched: small models. Build all 2N perturbed vectors as one (2N, d) tensor and
                  score them in a single call.
evaluate_inplace: big models (the MeZO trick). One model in memory: add sigma*eps to its
                  weights, score, subtract 2*sigma*eps, score. Before every seed the
                  weights are copied back from theta, so float rounding never piles up.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn

from seedling.es.noise import noise
from seedling.es.vector import assign


def _numpy(values) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float64)


def evaluate_batched(
    fitness: Callable[[torch.Tensor], torch.Tensor],
    theta: torch.Tensor,
    seeds: Sequence[int],
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """fitness maps a (2N, d) batch of parameter vectors to 2N scores (higher is better)."""
    eps = torch.stack([noise(seed, theta.numel()) for seed in seeds])
    scores = _numpy(fitness(torch.cat([theta + sigma * eps, theta - sigma * eps])))
    n = len(seeds)
    return scores[:n], scores[n:]


@torch.no_grad()
def perturb_(params: Sequence[torch.Tensor], flat: torch.Tensor, scale: float) -> None:
    """params += scale * flat, where flat is split across the params in order."""
    offset = 0
    for p in params:
        n = p.numel()
        p.add_(flat[offset : offset + n].view_as(p), alpha=scale)
        offset += n


@torch.no_grad()
def evaluate_inplace(
    model: nn.Module,
    fitness: Callable[[nn.Module], float],
    theta: torch.Tensor,
    seeds: Sequence[int],
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """fitness scores the model as it currently is. The model ends holding exactly theta."""
    params = list(model.parameters())
    device = params[0].device
    f_plus, f_minus = [], []
    for seed in seeds:
        assign(model, theta)
        eps = noise(seed, theta.numel()).to(device)
        perturb_(params, eps, sigma)
        f_plus.append(fitness(model))
        perturb_(params, eps, -2 * sigma)
        f_minus.append(fitness(model))
    assign(model, theta)
    return _numpy(f_plus), _numpy(f_minus)
