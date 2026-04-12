#!/usr/bin/env python3
"""Smoke-test commercial adapters on one local audio sample."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from blasbench.config import ModelConfig
from blasbench.runner import build_adapter

CASES = (
    ("azure", "azure/speech-ga-IE", ("AZURE_SPEECH_KEY", "AZURE_SPEECH_REGION")),
    ("openai", "openai/whisper-1", ("OPENAI_API_KEY",)),
    ("elevenlabs", "elevenlabs/scribe_v1", ("ELEVENLABS_API_KEY",)),
    ("speechmatics", "speechmatics/ga", ("SPEECHMATICS_API_KEY",)),
    (
        "google",
        "google/chirp_2-ga-IE",
        ("GOOGLE_CLOUD_PROJECT", "GOOGLE_APPLICATION_CREDENTIALS"),
    ),
)


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key, value.strip().strip("'\""))


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test commercial ASR adapters")
    parser.add_argument("audio_path", help="Path to a short mono WAV sample")
    args = parser.parse_args()

    load_env(Path(".env"))

    audio, sample_rate = sf.read(args.audio_path)
    for backend, name, required_env in CASES:
        print(f"\n== {backend} ==")
        missing = [key for key in required_env if not os.environ.get(key)]
        if missing:
            print(f"skipped: missing {', '.join(missing)}")
            continue
        adapter = build_adapter(ModelConfig(name=name, generate_kwargs={"backend": backend}))
        adapter.load()
        try:
            [result] = adapter.transcribe_batch([audio], [sample_rate])
        finally:
            adapter.unload()
        print(result.text.strip() or "<empty>")


if __name__ == "__main__":
    main()
