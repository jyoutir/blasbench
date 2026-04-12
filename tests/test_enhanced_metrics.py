"""Tests for enhanced metrics: distributions, RTFx, markdown output."""

from __future__ import annotations

from blasbench.metrics import (
    EvalResult,
    MetricsResult,
    compute_distribution,
    evaluate,
)
from blasbench.report import to_markdown


class TestDistributionStats:
    def test_basic_distribution(self) -> None:
        values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        dist = compute_distribution(values)
        assert dist.min == 0.0
        assert dist.max == 1.0
        assert abs(dist.median - 0.5) < 1e-6
        assert dist.p90 > dist.median
        assert dist.p99 > dist.p90

    def test_empty_distribution(self) -> None:
        dist = compute_distribution([])
        assert dist.mean == 0.0
        assert dist.median == 0.0

    def test_single_value(self) -> None:
        dist = compute_distribution([0.5])
        assert dist.mean == 0.5
        assert dist.median == 0.5
        assert dist.std == 0.0

    def test_to_dict(self) -> None:
        dist = compute_distribution([0.1, 0.2, 0.3])
        d = dist.to_dict()
        assert "mean" in d
        assert "median" in d
        assert "p90" in d
        assert "p95" in d
        assert "p99" in d


class TestEvaluateWithDistributions:
    def test_per_utterance_wer(self) -> None:
        refs = ["tá sé go maith", "céad míle fáilte", "dia duit"]
        hyps = ["tá sé go maith", "cead míle fáilte", "dia duit"]
        result = evaluate(refs, hyps, compute_distributions=True)

        assert len(result.per_utterance_wer) == 3
        assert result.per_utterance_wer[0] == 0.0  # Perfect match
        assert result.per_utterance_wer[1] > 0.0  # Fada error
        assert result.per_utterance_wer[2] == 0.0  # Perfect match

    def test_per_utterance_cer(self) -> None:
        refs = ["tá sé"]
        hyps = ["ta sé"]  # Missing fada on á
        result = evaluate(refs, hyps, compute_distributions=True)

        assert len(result.per_utterance_cer) == 1
        assert result.per_utterance_cer[0] > 0.0

    def test_distribution_stats_available(self) -> None:
        refs = ["tá sé go maith", "céad míle fáilte"]
        hyps = ["tá sé go maith", "cead míle fáilte"]
        result = evaluate(refs, hyps, compute_distributions=True)

        wer_dist = result.wer_distribution
        assert wer_dist is not None
        assert wer_dist.min == 0.0
        assert wer_dist.max > 0.0

    def test_no_distributions_when_disabled(self) -> None:
        refs = ["tá sé"]
        hyps = ["tá sé"]
        result = evaluate(refs, hyps, compute_distributions=False)
        assert result.wer_distribution is None
        assert result.cer_distribution is None


class TestEvalResultRTFx:
    def test_rtfx_field(self) -> None:
        result = EvalResult(
            overall=MetricsResult(
                wer=0.1,
                cer=0.05,
                num_utterances=10,
                total_ref_words=50,
                substitutions=3,
                insertions=1,
                deletions=1,
                hits=45,
            ),
            rtfx=3.5,
            total_audio_duration_s=100.0,
            total_processing_time_s=28.5,
        )
        assert result.rtfx == 3.5


class TestMarkdownOutput:
    def test_markdown_has_frontmatter(self) -> None:
        result = EvalResult(
            overall=MetricsResult(
                wer=0.15,
                cer=0.08,
                num_utterances=100,
                total_ref_words=500,
                substitutions=40,
                insertions=15,
                deletions=20,
                hits=460,
                per_utterance_wer=[0.0, 0.1, 0.2, 0.5, 1.0],
            ),
            model_name="openai/whisper-large-v3",
            dataset_name="common_voice/ga-IE",
            normalizer_name="irish",
            rtfx=2.5,
        )
        md = to_markdown(result)

        assert md.startswith("---")
        assert "type: experiment-run" in md
        assert "wer: 15.00" in md
        assert "cer: 8.00" in md
        assert "rtfx: 2.50" in md

    def test_markdown_has_distribution_table(self) -> None:
        result = EvalResult(
            overall=MetricsResult(
                wer=0.15,
                cer=0.08,
                num_utterances=5,
                total_ref_words=25,
                substitutions=2,
                insertions=1,
                deletions=1,
                hits=21,
                per_utterance_wer=[0.0, 0.1, 0.2, 0.3, 0.15],
            ),
            model_name="test",
            dataset_name="test",
            normalizer_name="irish",
        )
        md = to_markdown(result)
        assert "WER Distribution" in md
        assert "Median" in md
        assert "P90" in md
