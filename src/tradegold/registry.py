"""Minimal decorator-based registry.

Lets new data providers, feature blocks, labelers, models and threshold
policies be plugged in by decorating a function/class, without any core
module needing to import or know about them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def decorator(obj: T) -> T:
            key = name.lower()
            if key in self._items:
                raise ValueError(f"{self._kind} '{name}' is already registered")
            self._items[key] = obj
            return obj

        return decorator

    def get(self, name: str) -> T:
        key = name.lower()
        if key not in self._items:
            raise KeyError(
                f"Unknown {self._kind} '{name}'. Available: {', '.join(sorted(self._items))}"
            )
        return self._items[key]

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name.lower() in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._items))
