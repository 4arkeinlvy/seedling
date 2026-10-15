"""The smallest possible Task: climb to a fixed target point. Fast, exact, good for tests."""

from __future__ import annotations

import torch


class QuadraticTask:
    name = "quadratic"

    def __init__(self, dim: int = 16):
        self.dim = dim
        self.target = torch.linspace(-1, 1, dim)

    def init(self, seed: int) -> torch.Tensor:
        return torch.zeros(self.dim)

    def fitness(self, thetas: torch.Tensor, step: int) -> torch.Tensor:
        return -((thetas - self.target) ** 2).sum(dim=1)

    def evaluate(self, theta: torch.Tensor, step: int) -> dict[str, float]:
        return {"distance": float((theta - self.target).norm())}
