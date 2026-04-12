"""Meta MMS (Massively Multilingual Speech) adapter for Irish ASR.

Supports facebook/mms-1b-all with target_lang="gle" (Irish).
Uses CTC decoding like wav2vec2 models.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from blasbench.adapters import BaseAdapter, TranscriptionResult
from blasbench.registry import model_registry

logger = logging.getLogger(__name__)


@model_registry.register("mms")
class MMSAdapter(BaseAdapter):
    """Adapter for Meta's MMS-1B-All model.

    MMS supports 1000+ languages including Irish (gle).
    The model uses CTC decoding with language-specific tokenizers.
    Setting target_lang loads the correct CTC head and tokenizer for Irish.
    """

    def __init__(
        self,
        model_name: str = "facebook/mms-1b-all",
        target_lang: str = "gle",
        torch_dtype: str = "float32",
        device: str = "auto",
        **kwargs: Any,
    ) -> None:
        self._model_name = model_name
        self._target_lang = target_lang
        self._torch_dtype = torch_dtype
        self._device = device
        self._model: Any = None
        self._processor: Any = None
        self._num_parameters: int | None = None
        self._resolved_device: str = "cpu"

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, Wav2Vec2ForCTC

        device = self._device
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._resolved_device = device

        logger.info("Loading MMS model: %s (lang=%s)", self._model_name, self._target_lang)

        self._processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
            self._model_name, target_lang=self._target_lang
        )
        self._model = Wav2Vec2ForCTC.from_pretrained(
            self._model_name,
            target_lang=self._target_lang,
            ignore_mismatched_sizes=True,
        )
        self._model.to(device)  # type: ignore[arg-type]
        self._model.eval()  # type: ignore[no-untyped-call]

        self._num_parameters = sum(p.numel() for p in self._model.parameters())
        logger.info(
            "MMS loaded: %s (%s params, lang=%s)",
            self._model_name,
            f"{self._num_parameters:,}" if self._num_parameters else "unknown",
            self._target_lang,
        )

    def transcribe_batch(
        self,
        audio_arrays: list[Any],
        sample_rates: list[int],
    ) -> list[TranscriptionResult]:
        if self._model is None or self._processor is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        import librosa
        import torch

        target_sr = 16000
        results: list[TranscriptionResult] = []

        for audio, sr in zip(audio_arrays, sample_rates):
            arr = np.asarray(audio, dtype=np.float32)
            if sr != target_sr:
                arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr)

            duration = len(arr) / target_sr

            inputs = self._processor(
                arr, sampling_rate=target_sr, return_tensors="pt", padding=True
            )
            input_values = inputs.input_values.to(self._resolved_device)

            start = time.perf_counter()
            with torch.no_grad():
                logits = self._model(input_values).logits

            # Greedy CTC decoding
            predicted_ids = torch.argmax(logits, dim=-1)
            text = self._processor.batch_decode(predicted_ids)[0]
            elapsed = time.perf_counter() - start

            results.append(
                TranscriptionResult(
                    text=text,
                    language="irish",
                    audio_duration_s=duration,
                    processing_time_s=elapsed,
                )
            )

        return results

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def num_parameters(self) -> int | None:
        return self._num_parameters
