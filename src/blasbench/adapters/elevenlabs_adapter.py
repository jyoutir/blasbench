"""ElevenLabs Scribe adapter for Irish (gle)."""

from __future__ import annotations

import io
import os
from typing import Any

from blasbench.adapters import BaseAdapter, TranscriptionResult, to_wav_bytes
from blasbench.registry import model_registry


@model_registry.register("elevenlabs")
class ElevenLabsAdapter(BaseAdapter):
    def __init__(self, model_id: str = "scribe_v1") -> None:
        key = os.environ.get("ELEVENLABS_API_KEY")
        assert key, "ELEVENLABS_API_KEY env var required"
        self._key = key
        self._model_id = model_id
        self._client: Any = None

    def load(self) -> None:
        from elevenlabs.client import ElevenLabs

        self._client = ElevenLabs(api_key=self._key)

    def transcribe_batch(
        self, audio_arrays: list[Any], sample_rates: list[int]
    ) -> list[TranscriptionResult]:
        assert self._client is not None, "call load() first"

        results: list[TranscriptionResult] = []
        for audio, sr in zip(audio_arrays, sample_rates):
            wav = to_wav_bytes(audio, sr)
            buf = io.BytesIO(wav)
            buf.name = "audio.wav"
            resp = self._client.speech_to_text.convert(
                file=buf, model_id=self._model_id, language_code="gle"
            )
            results.append(TranscriptionResult(text=resp.text, language="gle"))
        return results

    def unload(self) -> None:
        self._client = None

    @property
    def name(self) -> str:
        return f"elevenlabs/{self._model_id}"
