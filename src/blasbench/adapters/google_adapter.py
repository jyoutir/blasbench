"""Google Cloud Speech-to-Text V2 adapter for Irish."""

from __future__ import annotations

import os
from typing import Any

from blasbench.adapters import BaseAdapter, TranscriptionResult, to_wav_bytes
from blasbench.registry import model_registry


@model_registry.register("google")
class GoogleAdapter(BaseAdapter):
    def __init__(self) -> None:
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        credentials = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        assert project, "GOOGLE_CLOUD_PROJECT env var required"
        assert credentials, "GOOGLE_APPLICATION_CREDENTIALS env var required"
        self._project = project
        self._language = "ga-IE"
        self._model_id = "chirp_2"
        self._location = "europe-west4"
        self._client: Any = None
        self._recognizer = f"projects/{self._project}/locations/{self._location}/recognizers/_"

    def load(self) -> None:
        from google.api_core.client_options import ClientOptions
        from google.cloud.speech_v2 import SpeechClient

        self._client = SpeechClient(
            client_options=ClientOptions(
                api_endpoint=f"{self._location}-speech.googleapis.com",
            )
        )

    def transcribe_batch(
        self, audio_arrays: list[Any], sample_rates: list[int]
    ) -> list[TranscriptionResult]:
        assert self._client is not None, "call load() first"
        from google.cloud.speech_v2.types import cloud_speech

        config = cloud_speech.RecognitionConfig(
            auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
            language_codes=[self._language],
            model=self._model_id,
        )

        results: list[TranscriptionResult] = []
        for audio, sr in zip(audio_arrays, sample_rates):
            response = self._client.recognize(
                request=cloud_speech.RecognizeRequest(
                    recognizer=self._recognizer,
                    config=config,
                    content=to_wav_bytes(audio, sr),
                )
            )
            text = " ".join(result.alternatives[0].transcript for result in response.results)
            results.append(TranscriptionResult(text=text, language=self._language))
        return results

    def unload(self) -> None:
        self._client = None

    @property
    def name(self) -> str:
        return f"google/{self._model_id}/{self._language}"
