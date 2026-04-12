"""Tests for configuration models."""

from __future__ import annotations

import tempfile

from blasbench.config import (
    DatasetConfig,
    Dialect,
    EvalConfig,
    ModelConfig,
    NormalizerConfig,
    NormalizerType,
    OutputFormat,
)


class TestEvalConfig:
    def test_defaults(self) -> None:
        config = EvalConfig()
        assert config.model.name == "openai/whisper-large-v3"
        assert config.normalizer.type == NormalizerType.IRISH
        assert config.dataset.config == "ga-IE"
        assert config.compute_cer is True
        assert config.per_dialect is True
        assert config.output_format == OutputFormat.TABLE

    def test_yaml_roundtrip(self) -> None:
        config = EvalConfig(
            model=ModelConfig(name="openai/whisper-small", batch_size=8),
            dataset=DatasetConfig(name="google/fleurs", config="ga_ie"),
            normalizer=NormalizerConfig(type=NormalizerType.IRISH),
        )
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            config.to_yaml(f.name)
            loaded = EvalConfig.from_yaml(f.name)

        assert loaded.model.name == "openai/whisper-small"
        assert loaded.model.batch_size == 8
        assert loaded.dataset.name == "google/fleurs"

    def test_normalizer_remove_diacritics_default_false(self) -> None:
        config = NormalizerConfig()
        assert config.remove_diacritics is False


class TestDialect:
    def test_enum_values(self) -> None:
        assert Dialect.ULSTER.value == "ulster"
        assert Dialect.CONNACHT.value == "connacht"
        assert Dialect.MUNSTER.value == "munster"
