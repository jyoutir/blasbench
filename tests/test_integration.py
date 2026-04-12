"""Integration tests for the full evaluation pipeline.

These tests verify the end-to-end flow using synthetic data and
the CSV loader (no network access required). They test:
- Full pipeline from data loading through metric computation
- Normalizer + metrics interaction (fadas affecting WER)
- Report generation from real EvalResults
- CLI integration
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

from blasbench.cli import app
from blasbench.data_loader import CSVLoader
from blasbench.metrics import EvalResult, evaluate, evaluate_grouped
from blasbench.normalizer import IrishNormalizer
from blasbench.report import format_result

runner = CliRunner()


@pytest.fixture()
def irish_dataset(tmp_path: Path) -> Path:
    """Create a realistic Irish evaluation dataset with dialect labels."""
    # Create audio files (1 second each, 16kHz)
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    utterances = [
        # (filename, reference text, dialect)
        ("utt001.wav", "Tá sé go maith", "munster"),
        ("utt002.wav", "Céad míle fáilte", "connacht"),
        ("utt003.wav", "An bhfuil cead agam dul amach?", "ulster"),
        ("utt004.wav", "Dia duit, a chara", "munster"),
        ("utt005.wav", "Cén t-ainm atá ort?", "connacht"),
        ("utt006.wav", "Tá mé i mo chónaí i nGaillimh", "connacht"),
        ("utt007.wav", "Is maith liom caife", "munster"),
        ("utt008.wav", "Ar an mbád mór", "ulster"),
        ("utt009.wav", "Mo mháthair agus m'athair", "ulster"),
        ("utt010.wav", "Slán go fóill", "munster"),
    ]

    for filename, _, _ in utterances:
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        sf.write(str(audio_dir / filename), audio, 16000)

    csv_path = tmp_path / "irish_test.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_path", "reference", "dialect"])
        writer.writeheader()
        for filename, ref, dialect in utterances:
            writer.writerow(
                {
                    "audio_path": str(audio_dir / filename),
                    "reference": ref,
                    "dialect": dialect,
                }
            )

    return csv_path


class TestFullPipelineNoModel:
    """Test the evaluation pipeline using pre-generated hypotheses (no model needed)."""

    def test_perfect_transcription(self) -> None:
        """WER should be 0 when hypothesis matches reference exactly."""
        norm = IrishNormalizer()
        refs = [
            "Tá sé go maith",
            "Céad míle fáilte",
            "An bhfuil cead agam dul amach?",
        ]
        hyps = [
            "Tá sé go maith",
            "Céad míle fáilte",
            "An bhfuil cead agam dul amach?",
        ]
        norm_refs = norm.normalize_batch(refs)
        norm_hyps = norm.normalize_batch(hyps)
        result = evaluate(norm_refs, norm_hyps)

        assert result.wer == 0.0
        assert result.cer == 0.0
        assert result.num_utterances == 3

    def test_fada_errors_detected(self) -> None:
        """Missing fadas should count as errors."""
        norm = IrishNormalizer()
        refs = ["Céad míle fáilte"]
        hyps = ["Cead mile failte"]  # Missing all fadas
        norm_refs = norm.normalize_batch(refs)
        norm_hyps = norm.normalize_batch(hyps)
        result = evaluate(norm_refs, norm_hyps)

        # All 3 words have fada errors → 3 substitutions out of 3 words
        assert result.wer == 1.0
        assert result.substitutions == 3

    def test_lenition_errors_detected(self) -> None:
        """Missing lenition should count as error."""
        norm = IrishNormalizer()
        refs = ["an bhean"]  # Correct: lenited
        hyps = ["an bean"]  # Wrong: not lenited
        norm_refs = norm.normalize_batch(refs)
        norm_hyps = norm.normalize_batch(hyps)
        result = evaluate(norm_refs, norm_hyps)

        assert result.wer > 0  # bhean ≠ bean
        assert result.substitutions == 1

    def test_eclipsis_errors_detected(self) -> None:
        """Missing eclipsis should count as error."""
        norm = IrishNormalizer()
        refs = ["ar an mbád"]  # Correct: eclipsis
        hyps = ["ar an bád"]  # Wrong: no eclipsis
        norm_refs = norm.normalize_batch(refs)
        norm_hyps = norm.normalize_batch(hyps)
        result = evaluate(norm_refs, norm_hyps)

        assert result.wer > 0  # mbád ≠ bád
        assert result.substitutions == 1

    def test_punctuation_does_not_affect_wer(self) -> None:
        """Punctuation differences should not affect WER after normalization."""
        norm = IrishNormalizer()
        refs = ["Dia duit, a chara!"]
        hyps = ["dia duit a chara"]
        norm_refs = norm.normalize_batch(refs)
        norm_hyps = norm.normalize_batch(hyps)
        result = evaluate(norm_refs, norm_hyps)

        assert result.wer == 0.0  # Punctuation and case normalized away

    def test_per_dialect_evaluation(self) -> None:
        """Per-dialect breakdown should work correctly."""
        norm = IrishNormalizer()
        refs = [
            "tá sé go maith",  # munster
            "tá sí go maith",  # connacht
            "tá siad go maith",  # ulster
            "tá mé go maith",  # munster
        ]
        hyps = [
            "tá sé go maith",  # correct
            "tá sé go maith",  # wrong: sí→sé
            "tá siad go maith",  # correct
            "tá me go maith",  # wrong: mé→me (fada error)
        ]
        dialects = ["munster", "connacht", "ulster", "munster"]

        result = evaluate_grouped(
            norm.normalize_batch(refs),
            norm.normalize_batch(hyps),
            dialects,
            group_name="dialect",
        )

        assert result.overall.wer > 0

        # Check per-dialect
        dialect_map = {g.group_value: g for g in result.per_group}

        # Connacht had an error
        assert dialect_map["connacht"].metrics.wer > 0
        assert dialect_map["connacht"].num_samples == 1

        # Ulster was perfect
        assert dialect_map["ulster"].metrics.wer == 0.0

    def test_report_json_output(self) -> None:
        """JSON report should be valid and contain expected fields."""
        norm = IrishNormalizer()
        refs = ["tá sé go maith", "céad míle fáilte"]
        hyps = ["tá sé go maith", "cead míle fáilte"]
        norm_refs = norm.normalize_batch(refs)
        norm_hyps = norm.normalize_batch(hyps)
        result_metrics = evaluate(norm_refs, norm_hyps)

        eval_result = EvalResult(
            overall=result_metrics,
            model_name="test-model",
            dataset_name="test-dataset",
            normalizer_name="irish",
        )

        json_str = format_result(eval_result, fmt="json")
        parsed = json.loads(json_str)

        assert parsed["model"] == "test-model"
        assert parsed["overall"]["wer"] > 0
        assert parsed["overall"]["num_utterances"] == 2

    def test_report_csv_output(self) -> None:
        """CSV report should contain header and data rows."""
        norm = IrishNormalizer()
        refs = ["tá sé"]
        hyps = ["tá sé"]
        result_metrics = evaluate(norm.normalize_batch(refs), norm.normalize_batch(hyps))
        eval_result = EvalResult(
            overall=result_metrics,
            model_name="test",
            dataset_name="test",
        )
        csv_str = format_result(eval_result, fmt="csv")
        lines = csv_str.strip().split("\n")
        assert len(lines) >= 2
        assert "wer" in lines[0]


class TestCSVLoaderIntegration:
    """Test CSV loader with realistic Irish data."""

    def test_loads_irish_dataset(self, irish_dataset: Path) -> None:
        loader = CSVLoader(path=irish_dataset)
        samples = list(loader)

        assert len(samples) == 10
        assert samples[0].reference == "Tá sé go maith"
        assert samples[0].metadata["dialect"] == "munster"
        assert samples[1].reference == "Céad míle fáilte"
        assert samples[1].metadata["dialect"] == "connacht"

    def test_full_eval_with_csv(self, irish_dataset: Path) -> None:
        """Simulate a full evaluation using CSV data with fake hypotheses."""
        loader = CSVLoader(path=irish_dataset)
        samples = list(loader)

        norm = IrishNormalizer()
        refs = [s.reference for s in samples]
        # Simulate some typical ASR errors
        hyps = [
            "tá sé go maith",  # correct
            "cead mile failte",  # fada errors
            "an bhfuil cead agam dul amach",  # correct (punctuation removed)
            "dia duit a chara",  # correct (punctuation removed)
            "cén t-ainm atá ort",  # correct
            "tá mé i mo chónaí i nGaillimh",  # correct
            "is maith liom caife",  # correct
            "ar an bád mór",  # eclipsis error: mbád→bád
            "mo mháthair agus m'athair",  # correct
            "slán go fóill",  # correct
        ]

        norm_refs = norm.normalize_batch(refs)
        norm_hyps = norm.normalize_batch(hyps)

        dialects = [s.metadata.get("dialect", "unknown") for s in samples]
        result = evaluate_grouped(norm_refs, norm_hyps, dialects, group_name="dialect")

        # Should have some errors (fada errors + eclipsis error)
        assert result.overall.wer > 0
        assert result.overall.num_utterances == 10

        # Check dialect breakdown exists
        assert len(result.per_group) == 3  # munster, connacht, ulster

        # Connacht should have errors (fada errors in "céad míle fáilte")
        connacht = [g for g in result.per_group if g.group_value == "connacht"][0]
        assert connacht.metrics.wer > 0


class TestCLIIntegration:
    """Test CLI commands work end-to-end."""

    def test_normalize_command(self) -> None:
        result = runner.invoke(app, ["normalize", "Céad míle fáilte!"])
        assert result.exit_code == 0
        assert "céad míle fáilte" in result.stdout

    def test_normalize_lenition(self) -> None:
        result = runner.invoke(app, ["normalize", "An bhean mhór"])
        assert result.exit_code == 0
        assert "an bhean mhór" in result.stdout

    def test_normalize_eclipsis(self) -> None:
        result = runner.invoke(app, ["normalize", "Ar an mbád"])
        assert result.exit_code == 0
        assert "ar an mbád" in result.stdout

    def test_list_commands(self) -> None:
        result = runner.invoke(app, ["list-datasets"])
        assert result.exit_code == 0
        assert "common-voice-ga" in result.stdout

        result = runner.invoke(app, ["list-models"])
        assert result.exit_code == 0
        assert "whisper-large-v3" in result.stdout

    def test_evaluate_help(self) -> None:
        result = runner.invoke(app, ["evaluate", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.stdout
        assert "--dataset" in result.stdout
