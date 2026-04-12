"""Meta Omnilingual ASR adapter for Irish (gle_Latn).

Requires the `omnilingual-asr` package (install via `pip install -e ".[omnilingual]"`).
Weights download from Meta's CDN to ~/.cache/fairseq2/assets/ on first load.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from blasbench.adapters import BaseAdapter, TranscriptionResult
from blasbench.registry import model_registry


@model_registry.register("omnilingual")
class OmniASRAdapter(BaseAdapter):
    def __init__(
        self,
        model_card: str = "omniASR_LLM_7B_v2",
        language: str = "gle_Latn",
        batch_size: int = 2,
    ) -> None:
        self._model_card = model_card
        self._language = language
        self._batch_size = batch_size
        self._pipeline: Any = None

    def load(self) -> None:
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

        self._pipeline = ASRInferencePipeline(model_card=self._model_card)

    def transcribe_batch(
        self, audio_arrays: list[Any], sample_rates: list[int]
    ) -> list[TranscriptionResult]:
        assert self._pipeline is not None, "call load() first"

        audio_data = [
            {"waveform": np.asarray(a, dtype=np.float32), "sample_rate": int(sr)}
            for a, sr in zip(audio_arrays, sample_rates)
        ]
        langs = [self._language] * len(audio_data)
        transcriptions = self._pipeline.transcribe(
            audio_data, lang=langs, batch_size=self._batch_size
        )
        return [TranscriptionResult(text=t, language=self._language) for t in transcriptions]

    def unload(self) -> None:
        self._pipeline = None

    @property
    def name(self) -> str:
        return f"omnilingual/{self._model_card}/{self._language}"
