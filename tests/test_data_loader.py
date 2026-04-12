"""Tests for data loaders."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from blasbench.data_loader import CSVLoader, Sample, TSVLoader


@pytest.fixture()
def audio_file(tmp_path: Path) -> Path:
    """Create a temporary WAV file."""
    audio = np.random.randn(16000).astype(np.float32)  # 1 second of audio
    path = tmp_path / "test.wav"
    sf.write(str(path), audio, 16000)
    return path


@pytest.fixture()
def csv_file(tmp_path: Path, audio_file: Path) -> Path:
    """Create a temporary CSV file with audio paths."""
    path = tmp_path / "test.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_path", "reference", "dialect"])
        writer.writeheader()
        writer.writerow(
            {
                "audio_path": str(audio_file),
                "reference": "Tá sé go maith",
                "dialect": "munster",
            }
        )
        writer.writerow(
            {
                "audio_path": str(audio_file),
                "reference": "Céad míle fáilte",
                "dialect": "connacht",
            }
        )
    return path


@pytest.fixture()
def tsv_file(tmp_path: Path, audio_file: Path) -> Path:
    """Create a temporary TSV file."""
    path = tmp_path / "test.tsv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_path", "reference"], delimiter="\t")
        writer.writeheader()
        writer.writerow({"audio_path": str(audio_file), "reference": "Dia duit"})
    return path


class TestCSVLoader:
    def test_loads_samples(self, csv_file: Path) -> None:
        loader = CSVLoader(path=csv_file)
        assert len(loader) == 2

    def test_iterates_samples(self, csv_file: Path) -> None:
        loader = CSVLoader(path=csv_file)
        samples = list(loader)
        assert len(samples) == 2
        assert isinstance(samples[0], Sample)
        assert samples[0].reference == "Tá sé go maith"
        assert samples[0].metadata["dialect"] == "munster"
        assert samples[0].sample_rate == 16000

    def test_max_samples(self, csv_file: Path) -> None:
        loader = CSVLoader(path=csv_file, max_samples=1)
        assert len(loader) == 1

    def test_audio_is_float32(self, csv_file: Path) -> None:
        loader = CSVLoader(path=csv_file)
        sample = next(iter(loader))
        assert sample.audio_array.dtype == np.float32


class TestTSVLoader:
    def test_loads_tsv(self, tsv_file: Path) -> None:
        loader = TSVLoader(path=tsv_file)
        samples = list(loader)
        assert len(samples) == 1
        assert samples[0].reference == "Dia duit"

    def test_name(self, tsv_file: Path) -> None:
        loader = TSVLoader(path=tsv_file)
        assert loader.name == "test"
