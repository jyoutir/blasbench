"""Tests for report generation."""

from __future__ import annotations

import json

from blasbench.metrics import EvalResult, MetricsResult, PerGroupResult
from blasbench.report import format_result, result_to_dict, to_csv, to_json


def _make_result() -> EvalResult:
    overall = MetricsResult(
        wer=0.15,
        cer=0.08,
        num_utterances=100,
        total_ref_words=500,
        substitutions=40,
        insertions=15,
        deletions=20,
        hits=460,
    )
    per_group = [
        PerGroupResult(
            group_name="dialect",
            group_value="munster",
            metrics=MetricsResult(
                wer=0.10,
                cer=0.05,
                num_utterances=50,
                total_ref_words=250,
                substitutions=15,
                insertions=5,
                deletions=5,
                hits=230,
            ),
            num_samples=50,
        ),
        PerGroupResult(
            group_name="dialect",
            group_value="connacht",
            metrics=MetricsResult(
                wer=0.20,
                cer=0.11,
                num_utterances=50,
                total_ref_words=250,
                substitutions=25,
                insertions=10,
                deletions=15,
                hits=230,
            ),
            num_samples=50,
        ),
    ]
    return EvalResult(
        overall=overall,
        per_group=per_group,
        model_name="openai/whisper-large-v3",
        dataset_name="common_voice/ga-IE",
        normalizer_name="irish",
    )


class TestResultToDict:
    def test_has_required_keys(self) -> None:
        d = result_to_dict(_make_result())
        assert "model" in d
        assert "dataset" in d
        assert "overall" in d
        assert d["overall"]["wer"] == 15.0

    def test_per_group(self) -> None:
        d = result_to_dict(_make_result())
        assert "per_group" in d
        assert len(d["per_group"]) == 2


class TestToJson:
    def test_valid_json(self) -> None:
        s = to_json(_make_result())
        parsed = json.loads(s)
        assert parsed["overall"]["wer"] == 15.0

    def test_preserves_fadas_in_json(self) -> None:
        """ensure_ascii=False so fadas come through."""
        result = _make_result()
        result.model_name = "test_fáda"
        s = to_json(result)
        assert "fáda" in s


class TestToCsv:
    def test_csv_has_header(self) -> None:
        s = to_csv(_make_result())
        lines = s.strip().split("\n")
        assert len(lines) >= 2  # header + at least 1 row
        assert "wer" in lines[0]

    def test_csv_overall_row(self) -> None:
        s = to_csv(_make_result())
        assert "overall" in s


class TestFormatResult:
    def test_json_format(self) -> None:
        result = format_result(_make_result(), fmt="json")
        assert json.loads(result)

    def test_csv_format(self) -> None:
        result = format_result(_make_result(), fmt="csv")
        assert "wer" in result

    def test_unknown_format_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="Unknown format"):
            format_result(_make_result(), fmt="xml")
