"""Optimizers for a flat parameter vector. They step UP the gradient (we maximize fitness)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Optimizer(ABC):
    @abstractmethod
    def step(self, theta: torch.Tensor, grad: torch.Tensor) -> None:
        """Move theta along +grad, in place."""


class SGD(Optimizer):
    def __init__(self, lr: float, momentum: float = 0.0):
        self.lr = lr
        self.momentum = momentum
        self._velocity: torch.Tensor | None = None

    def step(self, theta: torch.Tensor, grad: torch.Tensor) -> None:
        if self._velocity is None:
            self._velocity = torch.zeros_like(theta)
        self._velocity.mul_(self.momentum).add_(grad)
        theta.add_(self._velocity, alpha=self.lr)


class Adam(Optimizer):
    def __init__(self, lr: float, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self._t = 0
        self._m: torch.Tensor | None = None
        self._v: torch.Tensor | None = None

    def step(self, theta: torch.Tensor, grad: torch.Tensor) -> None:
        if self._m is None or self._v is None:
            self._m = torch.zeros_like(theta)
            self._v = torch.zeros_like(theta)
        self._t += 1
        self._m.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
        self._v.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
        m_hat = self._m / (1 - self.beta1**self._t)
        v_hat = self._v / (1 - self.beta2**self._t)
        theta.add_(m_hat / (v_hat.sqrt() + self.eps), alpha=self.lr)
