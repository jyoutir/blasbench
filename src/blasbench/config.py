"""Configuration models for the Blas Voice evaluation harness."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class Dialect(str, Enum):
    """Irish dialect categories."""

    ULSTER = "ulster"
    CONNACHT = "connacht"
    MUNSTER = "munster"
    OTHER = "other"
    UNKNOWN = "unknown"


class NormalizerType(str, Enum):
    """Available normalizer strategies."""

    IRISH = "irish"
    BASIC_WHISPER = "basic_whisper"
    NONE = "none"


class DatasetType(str, Enum):
    """Supported dataset sources."""

    HUGGINGFACE = "huggingface"
    CSV = "csv"
    TSV = "tsv"


class OutputFormat(str, Enum):
    """Report output formats."""

    TABLE = "table"
    JSON = "json"
    CSV = "csv"
    MARKDOWN = "markdown"


class NormalizerConfig(BaseModel):
    """Configuration for text normalization."""

    type: NormalizerType = NormalizerType.IRISH
    remove_diacritics: bool = False
    lowercase: bool = True
    remove_punctuation: bool = True


class DatasetConfig(BaseModel):
    """Configuration for a dataset source."""

    type: DatasetType = DatasetType.HUGGINGFACE
    name: str = "mozilla-foundation/common_voice_17_0"
    config: str = "ga-IE"
    split: str = "test"
    audio_column: str = "audio"
    text_column: str = "sentence"
    dialect_column: str | None = None
    speaker_column: str | None = Field(default=None, description="Column for speaker ID")
    max_samples: int | None = Field(default=None, description="Limit samples for testing")
    path: str | None = Field(default=None, description="Path for CSV/TSV files")


class ModelConfig(BaseModel):
    """Configuration for an ASR model."""

    name: str = "openai/whisper-large-v3"
    language: str = "irish"
    task: str = "transcribe"
    batch_size: int = 16
    chunk_length_s: float = 30.0
    torch_dtype: str = "float16"
    device: str = "auto"
    generate_kwargs: dict[str, Any] = Field(default_factory=dict)


class EvalConfig(BaseModel):
    """Top-level evaluation configuration."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    normalizer: NormalizerConfig = Field(default_factory=NormalizerConfig)
    output_format: OutputFormat = OutputFormat.TABLE
    output_path: str | None = None
    compute_cer: bool = True
    per_dialect: bool = True
    per_speaker: bool = False
    num_workers: int = 4
    save_experiment: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvalConfig:
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.model_dump(mode="json"), f, default_flow_style=False)


