"""HuggingFace Transformers adapter for Whisper and other ASR models."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from blasbench.adapters import BaseAdapter, TranscriptionResult
from blasbench.registry import model_registry

logger = logging.getLogger(__name__)


@model_registry.register("transformers")
class TransformersAdapter(BaseAdapter):
    """Adapter for HuggingFace Transformers ASR pipeline.

    Supports Whisper, Wav2Vec2, and any model with the
    automatic-speech-recognition pipeline.
    """

    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3",
        language: str = "irish",
        task: str = "transcribe",
        batch_size: int = 16,
        chunk_length_s: float = 30.0,
        torch_dtype: str = "float16",
        device: str = "auto",
        generate_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._model_name = model_name
        self._language = language
        self._task = task
        self._batch_size = batch_size
        self._chunk_length_s = chunk_length_s
        self._torch_dtype = torch_dtype
        self._device = device
        self._generate_kwargs = generate_kwargs or {}
        self._pipe: Any = None
        self._num_parameters: int | None = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        device = self._device
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        dtype = torch.float16 if self._torch_dtype == "float16" else torch.float32
        if device == "cpu":
            dtype = torch.float32

        logger.info("Loading model %s on %s (%s)", self._model_name, device, dtype)

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self._model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model.to(device)
        self._num_parameters = sum(p.numel() for p in model.parameters())

        processor = AutoProcessor.from_pretrained(self._model_name)  # type: ignore[no-untyped-call]

        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=self._chunk_length_s,
            batch_size=self._batch_size,
            torch_dtype=dtype,
            device=device,
        )
        logger.info(
            "Model loaded: %s (%s params)",
            self._model_name,
            f"{self._num_parameters:,}" if self._num_parameters else "unknown",
        )

    def transcribe_batch(
        self,
        audio_arrays: list[Any],
        sample_rates: list[int],
    ) -> list[TranscriptionResult]:
        if self._pipe is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        import librosa

        target_sr = 16000
        audios: list[dict[str, Any]] = []
        durations: list[float] = []

        for audio, sr in zip(audio_arrays, sample_rates):
            arr = np.asarray(audio, dtype=np.float32)
            if sr != target_sr:
                arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr)
            audios.append({"raw": arr, "sampling_rate": target_sr})
            durations.append(len(arr) / target_sr)

        generate_kwargs: dict[str, Any] = {
            "language": self._language,
            "task": self._task,
        }
        generate_kwargs.update(self._generate_kwargs)

        start = time.perf_counter()
        raw_results = self._pipe(audios, generate_kwargs=generate_kwargs)
        elapsed = time.perf_counter() - start

        per_sample_time = elapsed / max(len(audios), 1)

        results: list[TranscriptionResult] = []
        for i, r in enumerate(raw_results):
            text = r["text"] if isinstance(r, dict) else r[0]["text"]
            results.append(
                TranscriptionResult(
                    text=text,
                    language=self._language,
                    audio_duration_s=durations[i],
                    processing_time_s=per_sample_time,
                )
            )
        return results

    def unload(self) -> None:
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def num_parameters(self) -> int | None:
        return self._num_parameters
