"""Base adapter interface for ASR model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class TranscriptionResult:
    """Result from a single transcription."""

    text: str
    language: str = ""
    confidence: float | None = None
    segments: list[dict[str, Any]] | None = None
    audio_duration_s: float = 0.0
    processing_time_s: float = 0.0


class BaseAdapter(ABC):
    """Abstract base class for ASR model adapters.

    All model backends must implement this interface. The evaluation pipeline
    calls load() once, then transcribe_batch() for each batch of audio.
    """

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory. Called once before evaluation."""
        ...

    @abstractmethod
    def transcribe_batch(
        self,
        audio_arrays: list[Any],
        sample_rates: list[int],
    ) -> list[TranscriptionResult]:
        """Transcribe a batch of audio arrays.

        Args:
            audio_arrays: List of numpy arrays (float32, mono).
            sample_rates: Sample rate for each audio array.

        Returns:
            List of TranscriptionResult objects.
        """
        ...

    @abstractmethod
    def unload(self) -> None:
        """Free model resources."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""
        ...

    @property
    def num_parameters(self) -> int | None:
        """Number of model parameters, if known."""
        return None
