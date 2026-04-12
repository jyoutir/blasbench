"""OpenAI Whisper API adapter for Irish."""

from __future__ import annotations

import io
import os
from typing import Any

from blasbench.adapters import BaseAdapter, TranscriptionResult, to_wav_bytes
from blasbench.registry import model_registry


@model_registry.register("openai")
class OpenAIAdapter(BaseAdapter):
    def __init__(self, model_name: str = "whisper-1") -> None:
        key = os.environ.get("OPENAI_API_KEY")
        assert key, "OPENAI_API_KEY env var required"
        self._key = key
        self._model_name = model_name
        self._client: Any = None

    def load(self) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=self._key)

    def transcribe_batch(
        self, audio_arrays: list[Any], sample_rates: list[int]
    ) -> list[TranscriptionResult]:
        assert self._client is not None, "call load() first"

        results: list[TranscriptionResult] = []
        for audio, sr in zip(audio_arrays, sample_rates):
            wav = to_wav_bytes(audio, sr)
            buf = io.BytesIO(wav)
            buf.name = "audio.wav"
            resp = self._client.audio.transcriptions.create(
                model=self._model_name, file=buf, language="ga"
            )
            results.append(TranscriptionResult(text=resp.text, language="ga"))
        return results

    def unload(self) -> None:
        self._client = None

    @property
    def name(self) -> str:
        return f"openai/{self._model_name}"
