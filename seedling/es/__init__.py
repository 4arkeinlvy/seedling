"""M3: evolution strategies (ES): training with forward passes only.

Explained in docs/M3-explained.md.
"""

from seedling.es.batched_mlp import BatchedMLP
from seedling.es.es import ES, centered_ranks, estimate_gradient, raw_fitness, step_seeds
from seedling.es.evaluate import evaluate_batched, evaluate_inplace, perturb_
from seedling.es.noise import noise
from seedling.es.optim import SGD, Adam, Optimizer
from seedling.es.vector import assign, flatten

__all__ = [
    "ES",
    "SGD",
    "Adam",
    "BatchedMLP",
    "Optimizer",
    "assign",
    "centered_ranks",
    "estimate_gradient",
    "evaluate_batched",
    "evaluate_inplace",
    "flatten",
    "noise",
    "perturb_",
    "raw_fitness",
    "step_seeds",
]
