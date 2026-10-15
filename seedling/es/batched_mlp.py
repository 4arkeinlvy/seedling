"""An MLP whose weights come from a (P, d) matrix: P different models in one forward pass."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch


class BatchedMLP:
    def __init__(self, sizes: Sequence[int], activation: Callable[[torch.Tensor], torch.Tensor] = torch.tanh):
        self.sizes = tuple(sizes)
        self.activation = activation
        self.layers = list(zip(self.sizes[:-1], self.sizes[1:], strict=True))
        self.size = sum(n_in * n_out + n_out for n_in, n_out in self.layers)

    def init(self, generator: torch.Generator | None = None) -> torch.Tensor:
        """One flat parameter vector: weights ~ N(0, 1/fan_in), biases 0."""
        parts = []
        for n_in, n_out in self.layers:
            parts.append(torch.randn(n_in * n_out, generator=generator) / n_in**0.5)
            parts.append(torch.zeros(n_out))
        return torch.cat(parts)

    def __call__(self, thetas: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """thetas (P, d); x (B, in) shared by all P models, or (P, B, in). Returns (P, B, out)."""
        p = thetas.shape[0]
        offset = 0
        h = x
        for k, (n_in, n_out) in enumerate(self.layers):
            w = thetas[:, offset : offset + n_in * n_out].view(p, n_in, n_out)
            offset += n_in * n_out
            b = thetas[:, offset : offset + n_out].view(p, 1, n_out)
            offset += n_out
            h = h @ w + b
            if k < len(self.layers) - 1:
                h = self.activation(h)
        return h
