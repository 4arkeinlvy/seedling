"""Move between an nn.Module and one flat parameter vector."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector


def flatten(model: nn.Module) -> torch.Tensor:
    """All parameters as one flat float32 CPU vector (a copy)."""
    return parameters_to_vector(model.parameters()).detach().to("cpu", torch.float32).clone()


@torch.no_grad()
def assign(model: nn.Module, theta: torch.Tensor) -> None:
    """Copy a flat vector into the model's parameters.

    Copies on purpose: torch's vector_to_parameters makes the parameters *views* of the
    vector, so later in-place changes to the model would silently change theta too.
    """
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.copy_(theta[offset : offset + n].view_as(p))
        offset += n
    if offset != theta.numel():
        raise ValueError(f"theta has {theta.numel()} values, the model has {offset}")
