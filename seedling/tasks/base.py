"""The interface every ES workload implements.

A task scores parameter vectors. Everything that is random about a score (episodes,
minibatches) depends only on the step number, so any worker scoring the same vector at the
same step gets the same number.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class Task(Protocol):
    name: str
    dim: int

    def init(self, seed: int) -> torch.Tensor:
        """Starting parameter vector (float32, shape (dim,))."""

    def fitness(self, thetas: torch.Tensor, step: int) -> torch.Tensor:
        """Scores of P parameter vectors (P, dim) at this step; higher is better."""

    def evaluate(self, theta: torch.Tensor, step: int) -> dict[str, float]:
        """Metrics of one parameter vector, for logs and plots."""
