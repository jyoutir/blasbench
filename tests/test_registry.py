"""Tests for the registry pattern."""

from __future__ import annotations

import pytest

from blasbench.registry import Registry, model_registry


class TestRegistry:
    def test_register_and_get(self) -> None:
        reg = Registry("test")

        @reg.register("foo")
        class Foo:
            pass

        assert reg.get("foo") is Foo

    def test_duplicate_raises(self) -> None:
        reg = Registry("test")

        @reg.register("bar")
        class Bar:
            pass

        with pytest.raises(ValueError, match="already registered"):

            @reg.register("bar")
            class Bar2:
                pass

    def test_unknown_raises(self) -> None:
        reg = Registry("test")
        with pytest.raises(KeyError, match="not found"):
            reg.get("nonexistent")

    def test_list(self) -> None:
        reg = Registry("test")

        @reg.register("b")
        class B:
            pass

        @reg.register("a")
        class A:
            pass

        assert reg.list() == ["a", "b"]

    def test_contains(self) -> None:
        reg = Registry("test")

        @reg.register("x")
        class X:
            pass

        assert "x" in reg
        assert "y" not in reg


class TestModelRegistry:
    def test_transformers_registered(self) -> None:
        # Import the adapter to trigger registration
        import blasbench.adapters.transformers_adapter  # noqa: F401

        assert "transformers" in model_registry

    def test_azure_registered(self) -> None:
        import blasbench.adapters.azure_adapter  # noqa: F401

        assert "azure" in model_registry
