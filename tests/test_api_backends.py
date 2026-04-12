"""Tests for API backend construction through the runner."""

from __future__ import annotations

import pytest

from blasbench.config import ModelConfig
from blasbench.runner import build_adapter


class TestBuildAPIAdapters:
    def test_builds_azure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AZURE_SPEECH_KEY", "fake")
        monkeypatch.setenv("AZURE_SPEECH_REGION", "eastus")
        adapter = build_adapter(
            ModelConfig(name="azure/speech-ga-IE", generate_kwargs={"backend": "azure"})
        )
        assert adapter.name == "azure/ga-IE"

    def test_builds_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "fake")
        adapter = build_adapter(
            ModelConfig(name="openai/whisper-1", generate_kwargs={"backend": "openai"})
        )
        assert adapter.name == "openai/whisper-1"

    def test_builds_elevenlabs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ELEVENLABS_API_KEY", "fake")
        adapter = build_adapter(
            ModelConfig(name="elevenlabs/scribe_v1", generate_kwargs={"backend": "elevenlabs"})
        )
        assert adapter.name == "elevenlabs/scribe_v1"

    def test_builds_speechmatics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SPEECHMATICS_API_KEY", "fake")
        adapter = build_adapter(
            ModelConfig(name="speechmatics/ga", generate_kwargs={"backend": "speechmatics"})
        )
        assert adapter.name == "speechmatics/ga"

    def test_builds_google(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "demo-project")
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/fake.json")
        adapter = build_adapter(
            ModelConfig(name="google/chirp_2-ga-IE", generate_kwargs={"backend": "google"})
        )
        assert adapter.name == "google/chirp_2/ga-IE"
