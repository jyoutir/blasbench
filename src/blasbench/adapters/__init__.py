"""Base adapter interface for ASR model backends."""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import librosa
import numpy as np
import soundfile as sf


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


def to_wav_bytes(audio_array: Any, sample_rate: int) -> bytes:
    audio = np.asarray(audio_array, dtype=np.float32)
    if sample_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
    buf = io.BytesIO()
    sf.write(buf, audio, 16000, format="WAV", subtype="PCM_16")
    return buf.getvalue()
