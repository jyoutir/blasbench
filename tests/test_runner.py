"""Tests for experiment tracking."""

from __future__ import annotations

import json
from pathlib import Path

from blasbench.runner import (
    Prediction,
    RunMetadata,
    build_metadata,
    generate_leaderboard,
    save_experiment,
    update_summary_csv,
)


class TestRunMetadata:
    def test_build_metadata(self) -> None:
        meta = build_metadata(
            model_name="whisper-test",
            model_source="openai/whisper-test",
            dataset_name="test-dataset",
            dataset_split="test",
            normalizer="irish",
            config_dict={"key": "value"},
            num_samples=100,
        )
        assert "whisper-test" in meta.run_id
        assert meta.model_name == "whisper-test"
        assert meta.dataset_name == "test-dataset"
        assert meta.config_hash.startswith("sha256:")
        assert meta.python_version != ""

    def test_to_dict(self) -> None:
        meta = RunMetadata(run_id="test", model_name="test-model")
        d = meta.to_dict()
        assert d["run_id"] == "test"
        assert d["model_name"] == "test-model"


class TestSaveExperiment:
    def test_saves_all_files(self, tmp_path: Path) -> None:
        meta = RunMetadata(run_id="test_save", model_name="test")
        predictions = [
            Prediction(
                sample_id="0",
                reference="céad míle fáilte",
                hypothesis="cead mile failte",
                wer=1.0,
                cer=0.15,
            ),
        ]
        results_dict = {"overall": {"wer": 50.0, "cer": 15.0}}
        config_dict = {"model": {"name": "test"}}
        md_report = "# Test Report\n"

        run_dir = save_experiment(
            metadata=meta,
            results_dict=results_dict,
            predictions=predictions,
            config_dict=config_dict,
            markdown_report=md_report,
            experiments_dir=tmp_path,
        )

        assert (run_dir / "metadata.json").exists()
        assert (run_dir / "config.yaml").exists()
        assert (run_dir / "results.json").exists()
        assert (run_dir / "results.md").exists()
        assert (run_dir / "predictions" / "test.jsonl").exists()

        # Verify JSONL contains fadas
        with open(run_dir / "predictions" / "test.jsonl") as f:
            line = json.loads(f.readline())
            assert line["reference"] == "céad míle fáilte"


class TestUpdateSummaryCSV:
    def test_creates_and_appends(self, tmp_path: Path) -> None:
        meta = RunMetadata(
            run_id="run1",
            model_name="model1",
            dataset_name="ds1",
            timestamp="2026-01-01",
        )
        results = {
            "overall": {
                "wer": 15.0,
                "cer": 8.0,
                "num_utterances": 100,
                "substitutions": 10,
                "insertions": 3,
                "deletions": 2,
            }
        }

        csv_path = update_summary_csv(meta, results, results_dir=tmp_path)
        assert csv_path.exists()

        # Append another run
        meta2 = RunMetadata(
            run_id="run2",
            model_name="model2",
            dataset_name="ds1",
            timestamp="2026-01-02",
        )
        results2 = {
            "overall": {
                "wer": 20.0,
                "cer": 12.0,
                "num_utterances": 100,
                "substitutions": 15,
                "insertions": 5,
                "deletions": 3,
            }
        }
        update_summary_csv(meta2, results2, results_dir=tmp_path)

        import pandas as pd

        df = pd.read_csv(csv_path)
        assert len(df) == 2
        assert df.iloc[0]["model"] == "model1"
        assert df.iloc[1]["model"] == "model2"


class TestGenerateLeaderboard:
    def test_no_results(self, tmp_path: Path) -> None:
        lb = generate_leaderboard(results_dir=tmp_path)
        assert "No results found" in lb

    def test_with_results(self, tmp_path: Path) -> None:
        import pandas as pd

        df = pd.DataFrame(
            [
                {
                    "run_id": "r1",
                    "model": "whisper-large",
                    "dataset": "cv-ga",
                    "wer": 15.0,
                    "cer": 8.0,
                    "rtfx": 2.5,
                    "num_utterances": 100,
                },
                {
                    "run_id": "r2",
                    "model": "whisper-small",
                    "dataset": "cv-ga",
                    "wer": 25.0,
                    "cer": 14.0,
                    "rtfx": 5.0,
                    "num_utterances": 100,
                },
            ]
        )
        df.to_csv(tmp_path / "summary.csv", index=False)

        lb = generate_leaderboard(results_dir=tmp_path)
        assert "whisper-large" in lb
        assert "whisper-small" in lb
        assert "15.00" in lb
