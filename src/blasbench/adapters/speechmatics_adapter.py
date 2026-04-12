"""Speechmatics batch adapter for Irish (ga)."""

from __future__ import annotations

import asyncio
import os
import tempfile
from typing import Any

from blasbench.adapters import BaseAdapter, TranscriptionResult, to_wav_bytes
from blasbench.registry import model_registry


@model_registry.register("speechmatics")
class SpeechmaticsAdapter(BaseAdapter):
    def __init__(self) -> None:
        key = os.environ.get("SPEECHMATICS_API_KEY")
        assert key, "SPEECHMATICS_API_KEY env var required"
        self._key = key
        self._language = "ga"
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    def transcribe_batch(
        self, audio_arrays: list[Any], sample_rates: list[int]
    ) -> list[TranscriptionResult]:
        assert self._loaded, "call load() first"
        return asyncio.run(self._transcribe_batch(audio_arrays, sample_rates))

    async def _transcribe_batch(
        self, audio_arrays: list[Any], sample_rates: list[int]
    ) -> list[TranscriptionResult]:
        from speechmatics.batch import AsyncClient, TranscriptionConfig

        results: list[TranscriptionResult] = []
        async with AsyncClient(api_key=self._key) as client:
            for audio, sr in zip(audio_arrays, sample_rates):
                wav = to_wav_bytes(audio, sr)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(wav)
                    path = f.name
                try:
                    resp = await client.transcribe(
                        audio_file=path,
                        transcription_config=TranscriptionConfig(language=self._language),
                    )
                    results.append(
                        TranscriptionResult(text=resp.transcript_text, language=self._language)
                    )
                finally:
                    os.unlink(path)
        return results

    def unload(self) -> None:
        self._loaded = False

    @property
    def name(self) -> str:
        return f"speechmatics/{self._language}"
