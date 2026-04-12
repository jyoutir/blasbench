"""ElevenLabs Scribe adapter for Irish (gle)."""

from __future__ import annotations

import io
import os
from typing import Any

import librosa
import numpy as np
import soundfile as sf

from blasbench.adapters import BaseAdapter, TranscriptionResult
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
            wav = _to_wav_bytes(np.asarray(audio, dtype=np.float32), sr)
            buf = io.BytesIO(wav)
            buf.name = "audio.wav"
            resp = self._client.speech_to_text.convert(
                file=buf, model_id=self._model_id, language_code="gle"
            )
            text = getattr(resp, "text", str(resp))
            results.append(TranscriptionResult(text=text, language="gle"))
        return results

    def unload(self) -> None:
        self._client = None

    @property
    def name(self) -> str:
        return f"elevenlabs/{self._model_id}"


def _to_wav_bytes(audio: np.ndarray[Any, Any], sr: int) -> bytes:
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    buf = io.BytesIO()
    sf.write(buf, audio, 16000, format="WAV", subtype="PCM_16")
    return buf.getvalue()
