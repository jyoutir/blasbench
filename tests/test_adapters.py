"""Tests for model adapter base types and API adapters (env-driven)."""

from __future__ import annotations

import os

import pytest

from blasbench.adapters import TranscriptionResult


class TestTranscriptionResult:
    def test_defaults(self) -> None:
        r = TranscriptionResult(text="hello")
        assert r.text == "hello"
        assert r.confidence is None
        assert r.audio_duration_s == 0.0


class TestAzureAdapter:
    def test_requires_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from blasbench.adapters.azure_adapter import AzureAdapter

        monkeypatch.delenv("AZURE_SPEECH_KEY", raising=False)
        monkeypatch.delenv("AZURE_SPEECH_REGION", raising=False)
        with pytest.raises(AssertionError):
            AzureAdapter()

    def test_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from blasbench.adapters.azure_adapter import AzureAdapter

        monkeypatch.setenv("AZURE_SPEECH_KEY", "fake")
        monkeypatch.setenv("AZURE_SPEECH_REGION", "eastus")
        a = AzureAdapter()
        assert a.name == "azure/ga-IE"


class TestOpenAIAdapter:
    def test_requires_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from blasbench.adapters.openai_adapter import OpenAIAdapter

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(AssertionError):
            OpenAIAdapter()

    def test_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from blasbench.adapters.openai_adapter import OpenAIAdapter

        monkeypatch.setenv("OPENAI_API_KEY", "fake")
        a = OpenAIAdapter()
        assert a.name == "openai/whisper-1"


class TestElevenLabsAdapter:
    def test_requires_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from blasbench.adapters.elevenlabs_adapter import ElevenLabsAdapter

        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
        with pytest.raises(AssertionError):
            ElevenLabsAdapter()

    def test_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from blasbench.adapters.elevenlabs_adapter import ElevenLabsAdapter

        monkeypatch.setenv("ELEVENLABS_API_KEY", "fake")
        a = ElevenLabsAdapter()
        assert a.name == "elevenlabs/scribe_v1"


class TestSpeechmaticsAdapter:
    def test_requires_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from blasbench.adapters.speechmatics_adapter import SpeechmaticsAdapter

        monkeypatch.delenv("SPEECHMATICS_API_KEY", raising=False)
        with pytest.raises(AssertionError):
            SpeechmaticsAdapter()

    def test_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from blasbench.adapters.speechmatics_adapter import SpeechmaticsAdapter

        monkeypatch.setenv("SPEECHMATICS_API_KEY", "fake")
        a = SpeechmaticsAdapter()
        assert a.name == "speechmatics/ga"


class TestGoogleAdapter:
    def test_requires_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from blasbench.adapters.google_adapter import GoogleAdapter

        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        with pytest.raises(AssertionError):
            GoogleAdapter()

    def test_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from blasbench.adapters.google_adapter import GoogleAdapter

        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "demo-project")
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/fake.json")
        a = GoogleAdapter()
        assert a.name == "google/chirp_2/ga-IE"


# Ensure stale env doesn't leak into the whole test session.
@pytest.fixture(autouse=True)
def _clear_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in (
        "AZURE_SPEECH_KEY",
        "AZURE_SPEECH_REGION",
        "OPENAI_API_KEY",
        "ELEVENLABS_API_KEY",
        "SPEECHMATICS_API_KEY",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_APPLICATION_CREDENTIALS",
    ):
        if k in os.environ:
            monkeypatch.delenv(k, raising=False)
