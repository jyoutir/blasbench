"""SeamlessM4T adapter for multilingual ASR.

Supports facebook/seamless-m4t-v2-large for Irish ASR using the
S2TT (speech-to-text translation) pipeline with source=target language
to perform ASR rather than translation.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from blasbench.adapters import BaseAdapter, TranscriptionResult
from blasbench.registry import model_registry

logger = logging.getLogger(__name__)


@model_registry.register("seamless")
class SeamlessM4TAdapter(BaseAdapter):
    """Adapter for Meta's SeamlessM4T v2 model.

    For Irish ASR, we use the model in S2TT mode with tgt_lang="gle"
    (ISO 639-3 code for Irish) and generate_speech=False.
    """

    def __init__(
        self,
        model_name: str = "facebook/seamless-m4t-v2-large",
        tgt_lang: str = "gle",
        torch_dtype: str = "float16",
        device: str = "auto",
        **kwargs: Any,
    ) -> None:
        self._model_name = model_name
        self._tgt_lang = tgt_lang
        self._torch_dtype = torch_dtype
        self._device = device
        self._model: Any = None
        self._processor: Any = None
        self._num_parameters: int | None = None
        self._resolved_device: str = "cpu"

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, SeamlessM4Tv2Model

        device = self._device
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._resolved_device = device

        dtype = torch.float16 if self._torch_dtype == "float16" else torch.float32
        if device == "cpu":
            dtype = torch.float32

        logger.info("Loading SeamlessM4T model: %s", self._model_name)

        self._processor = AutoProcessor.from_pretrained(self._model_name)  # type: ignore[no-untyped-call]
        self._model = SeamlessM4Tv2Model.from_pretrained(self._model_name, torch_dtype=dtype)
        self._model.to(device)  # type: ignore[arg-type]
        self._model.eval()  # type: ignore[no-untyped-call]

        self._num_parameters = sum(p.numel() for p in self._model.parameters())
        logger.info(
            "SeamlessM4T loaded: %s (%s params)",
            self._model_name,
            f"{self._num_parameters:,}" if self._num_parameters else "unknown",
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

            # Process audio
            inputs = self._processor(audios=arr, sampling_rate=target_sr, return_tensors="pt")
            inputs = {k: v.to(self._resolved_device) for k, v in inputs.items()}

            start = time.perf_counter()
            with torch.no_grad():
                output_tokens = self._model.generate(
                    **inputs,
                    tgt_lang=self._tgt_lang,
                    generate_speech=False,
                )

            # Decode tokens
            text = self._processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
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
