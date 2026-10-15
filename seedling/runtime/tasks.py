"""Task registry: workers and coordinator build the same Task from (name, args)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from seedling.tasks.base import Task

TaskFactory = Callable[..., Task]
_REGISTRY: dict[str, TaskFactory] = {}


def register(name: str) -> Callable[[TaskFactory], TaskFactory]:
    def decorate(factory: TaskFactory) -> TaskFactory:
        _REGISTRY[name] = factory
        return factory

    return decorate


def create_task(name: str, **args: Any) -> Task:
    if name not in _REGISTRY:
        raise KeyError(f"unknown task {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**args)


@register("quadratic")
def _quadratic(**args: Any) -> Task:
    from seedling.tasks.quadratic import QuadraticTask

    return QuadraticTask(**args)


@register("cartpole")
def _cartpole(**args: Any) -> Task:
    from seedling.tasks.cartpole import CartPoleTask

    return CartPoleTask(**args)


@register("mnist")
def _mnist(**args: Any) -> Task:
    from seedling.tasks.mnist import MnistTask

    return MnistTask(**args)
