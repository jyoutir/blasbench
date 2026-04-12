"""Model adapter registry for extensible ASR evaluation.

Uses the registry pattern from lm-evaluation-harness: string names map to
lazily-imported adapter classes. Adding a new model backend requires only:
1. Create an adapter class implementing BaseAdapter
2. Register it with @model_registry.register("backend_name")
"""

from __future__ import annotations

from typing import Any


class Registry:
    """Generic registry mapping string names to factory callables."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._registry: dict[str, type[Any]] = {}

    def register(self, name: str) -> Any:
        """Decorator to register a class under a name."""

        def decorator(cls: type[Any]) -> type[Any]:
            if name in self._registry:
                raise ValueError(f"{self._name} registry: {name!r} already registered")
            self._registry[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> type[Any]:
        """Get a registered class by name."""
        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(f"{self._name} registry: {name!r} not found. Available: {available}")
        return self._registry[name]

    def list(self) -> list[str]:
        """List all registered names."""
        return sorted(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._registry


model_registry: Registry = Registry("model")
dataset_registry: Registry = Registry("dataset")
