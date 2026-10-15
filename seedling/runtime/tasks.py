"""Workload registry: coordinator and workers build the same Workload from (name, args)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from seedling.runtime.workload import VectorWorkload, Workload

WorkloadFactory = Callable[..., Workload]
_REGISTRY: dict[str, WorkloadFactory] = {}


def register(name: str) -> Callable[[WorkloadFactory], WorkloadFactory]:
    def decorate(factory: WorkloadFactory) -> WorkloadFactory:
        _REGISTRY[name] = factory
        return factory

    return decorate


def create_workload(name: str, **args: Any) -> Workload:
    if name not in _REGISTRY:
        raise KeyError(f"unknown workload {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**args)


@register("quadratic")
def _quadratic(**args: Any) -> Workload:
    from seedling.tasks.quadratic import QuadraticTask

    return VectorWorkload(QuadraticTask(**args))


@register("cartpole")
def _cartpole(**args: Any) -> Workload:
    from seedling.tasks.cartpole import CartPoleTask

    return VectorWorkload(CartPoleTask(**args))


@register("mnist")
def _mnist(**args: Any) -> Workload:
    from seedling.tasks.mnist import MnistTask

    return VectorWorkload(MnistTask(**args))


@register("countdown")
def _countdown(**args: Any) -> Workload:
    from seedling.llm.workload import CountdownWorkload

    return CountdownWorkload(**args)
