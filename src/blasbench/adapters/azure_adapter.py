"""Azure Speech Services adapter for Irish (ga-IE)."""

from __future__ import annotations

import os
import tempfile
from typing import Any

from blasbench.adapters import BaseAdapter, TranscriptionResult, to_wav_bytes
from blasbench.registry import model_registry


@model_registry.register("azure")
class AzureAdapter(BaseAdapter):
    def __init__(self, language: str = "ga-IE") -> None:
        key = os.environ.get("AZURE_SPEECH_KEY")
        region = os.environ.get("AZURE_SPEECH_REGION")
        assert key, "AZURE_SPEECH_KEY env var required"
        assert region, "AZURE_SPEECH_REGION env var required"
        self._key = key
        self._region = region
        self._language = language
        self._config: Any = None

    def load(self) -> None:
        import azure.cognitiveservices.speech as speechsdk

        self._config = speechsdk.SpeechConfig(subscription=self._key, region=self._region)
        self._config.speech_recognition_language = self._language

    def transcribe_batch(
        self, audio_arrays: list[Any], sample_rates: list[int]
    ) -> list[TranscriptionResult]:
        assert self._config is not None, "call load() first"
        import azure.cognitiveservices.speech as speechsdk

        results: list[TranscriptionResult] = []
        for audio, sr in zip(audio_arrays, sample_rates):
            wav = to_wav_bytes(audio, sr)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(wav)
                path = f.name
            try:
                audio_input = speechsdk.AudioConfig(filename=path)
                recognizer = speechsdk.SpeechRecognizer(
                    speech_config=self._config, audio_config=audio_input
                )
                r = recognizer.recognize_once_async().get()
                text = r.text if r.reason == speechsdk.ResultReason.RecognizedSpeech else ""
                results.append(TranscriptionResult(text=text, language=self._language))
            finally:
                os.unlink(path)
        return results

    def unload(self) -> None:
        self._config = None

    @property
    def name(self) -> str:
        return f"azure/{self._language}"
