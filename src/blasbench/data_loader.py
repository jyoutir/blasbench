"""Data loading for ASR evaluation datasets.

Provides an abstract base class and concrete implementations for loading
evaluation data from HuggingFace Hub, CSV, and TSV files.

Each loader yields (audio_array, sample_rate, reference_text, metadata) tuples.
"""

from __future__ import annotations

import csv
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Sample:
    """A single evaluation sample."""

    audio_array: Any  # numpy array
    sample_rate: int
    reference: str
    metadata: dict[str, Any] = field(default_factory=dict)
    sample_id: str = ""


class DataLoader(ABC):
    """Abstract base class for evaluation data loaders."""

    @abstractmethod
    def __iter__(self) -> Iterator[Sample]:
        """Iterate over samples."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable dataset name."""
        ...


class HuggingFaceLoader(DataLoader):
    """Load datasets from HuggingFace Hub.

    Supports Common Voice, FLEURS, and any dataset with audio + text columns.
    """

    def __init__(
        self,
        dataset_name: str = "mozilla-foundation/common_voice_17_0",
        config: str = "ga-IE",
        split: str = "test",
        audio_column: str = "audio",
        text_column: str = "sentence",
        dialect_column: str | None = None,
        speaker_column: str | None = None,
        max_samples: int | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        self._dataset_name = dataset_name
        self._config = config
        self._split = split
        self._audio_column = audio_column
        self._text_column = text_column
        self._dialect_column = dialect_column
        self._speaker_column = speaker_column
        self._max_samples = max_samples
        self._trust_remote_code = trust_remote_code
        self._dataset: Any = None

    def _load(self) -> None:
        """Lazily load the dataset."""
        if self._dataset is not None:
            return

        from datasets import load_dataset

        logger.info(
            "Loading %s (config=%s, split=%s)", self._dataset_name, self._config, self._split
        )
        ds = load_dataset(
            self._dataset_name,
            self._config,
            split=self._split,
            trust_remote_code=self._trust_remote_code,
        )
        if self._max_samples is not None:
            ds = ds.select(range(min(self._max_samples, len(ds))))
        self._dataset = ds

    def __len__(self) -> int:
        self._load()
        return len(self._dataset)

    def __iter__(self) -> Iterator[Sample]:
        self._load()
        for i, row in enumerate(self._dataset):
            audio = row[self._audio_column]
            metadata: dict[str, Any] = {}
            if self._dialect_column and self._dialect_column in row:
                metadata["dialect"] = row[self._dialect_column]
            if self._speaker_column and self._speaker_column in row:
                metadata["speaker"] = row[self._speaker_column]
            # Common Voice stores client_id as speaker proxy
            if "client_id" in row and "speaker" not in metadata:
                metadata["speaker"] = row["client_id"]

            yield Sample(
                audio_array=audio["array"],
                sample_rate=audio["sampling_rate"],
                reference=row[self._text_column],
                metadata=metadata,
                sample_id=str(i),
            )

    @property
    def name(self) -> str:
        return f"{self._dataset_name}/{self._config}:{self._split}"


class CSVLoader(DataLoader):
    """Load evaluation data from a CSV file.

    Expected columns: 'audio_path', 'reference' (text).
    Optional columns: 'dialect', 'speaker'.
    """

    def __init__(
        self,
        path: str | Path,
        audio_column: str = "audio_path",
        text_column: str = "reference",
        dialect_column: str | None = "dialect",
        speaker_column: str | None = "speaker",
        max_samples: int | None = None,
        delimiter: str = ",",
    ) -> None:
        self._path = Path(path)
        self._audio_column = audio_column
        self._text_column = text_column
        self._dialect_column = dialect_column
        self._speaker_column = speaker_column
        self._max_samples = max_samples
        self._delimiter = delimiter
        self._rows: list[dict[str, str]] | None = None

    def _load(self) -> None:
        if self._rows is not None:
            return
        with open(self._path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=self._delimiter)
            self._rows = list(reader)
        if self._max_samples is not None:
            self._rows = self._rows[: self._max_samples]

    def __len__(self) -> int:
        self._load()
        assert self._rows is not None
        return len(self._rows)

    def __iter__(self) -> Iterator[Sample]:
        import numpy as np
        import soundfile as sf

        self._load()
        assert self._rows is not None
        for i, row in enumerate(self._rows):
            audio_path = row[self._audio_column]
            audio_array, sr = sf.read(audio_path, dtype="float32")
            if audio_array.ndim > 1:
                audio_array = np.mean(audio_array, axis=1)

            metadata: dict[str, Any] = {}
            if self._dialect_column and self._dialect_column in row:
                metadata["dialect"] = row[self._dialect_column]
            if self._speaker_column and self._speaker_column in row:
                metadata["speaker"] = row[self._speaker_column]

            yield Sample(
                audio_array=audio_array,
                sample_rate=sr,
                reference=row[self._text_column],
                metadata=metadata,
                sample_id=str(i),
            )

    @property
    def name(self) -> str:
        return self._path.stem


class TSVLoader(CSVLoader):
    """Load evaluation data from a TSV file."""

    def __init__(
        self,
        path: str | Path,
        audio_column: str = "audio_path",
        text_column: str = "reference",
        dialect_column: str | None = "dialect",
        speaker_column: str | None = "speaker",
        max_samples: int | None = None,
    ) -> None:
        super().__init__(
            path=path,
            audio_column=audio_column,
            text_column=text_column,
            dialect_column=dialect_column,
            speaker_column=speaker_column,
            max_samples=max_samples,
            delimiter="\t",
        )
