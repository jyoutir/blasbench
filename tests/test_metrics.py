"""Tests for WER/CER computation."""

from __future__ import annotations

import pytest

from blasbench.metrics import (
    EvalResult,
    MetricsResult,
    bootstrap_error_ci,
    compute_cer,
    compute_wer,
    evaluate,
    evaluate_grouped,
)


class TestComputeWer:
    def test_perfect_match(self) -> None:
        out = compute_wer(["tá sé go maith"], ["tá sé go maith"])
        assert out.wer == 0.0

    def test_complete_mismatch(self) -> None:
        out = compute_wer(["tá sé"], ["níl aon"])
        assert out.wer == 1.0

    def test_substitution(self) -> None:
        out = compute_wer(["céad míle fáilte"], ["cead míle fáilte"])
        # 1 substitution out of 3 words
        assert abs(out.wer - 1 / 3) < 1e-6

    def test_fada_matters_for_wer(self) -> None:
        """céad (hundred) vs cead (permission) — must be counted as error."""
        out = compute_wer(["céad"], ["cead"])
        assert out.wer == 1.0

    def test_multiple_utterances_global_aggregation(self) -> None:
        """WER must be globally aggregated, not averaged per sentence."""
        refs = ["a b c", "d"]
        hyps = ["a b c", "e"]
        out = compute_wer(refs, hyps)
        # 1 substitution out of 4 total words = 0.25
        assert abs(out.wer - 0.25) < 1e-6

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            compute_wer([], [])

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            compute_wer(["a"], ["a", "b"])

    def test_insertion(self) -> None:
        out = compute_wer(["tá sé"], ["tá sé anseo"])
        assert out.insertions == 1

    def test_deletion(self) -> None:
        out = compute_wer(["tá sé anseo"], ["tá sé"])
        assert out.deletions == 1


class TestComputeCer:
    def test_perfect_match(self) -> None:
        out = compute_cer(["tá"], ["tá"])
        assert out.cer == 0.0

    def test_fada_cer(self) -> None:
        """Single character difference: á vs a."""
        out = compute_cer(["tá"], ["ta"])
        assert out.cer > 0.0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            compute_cer([], [])


class TestEvaluate:
    def test_returns_metrics_result(self) -> None:
        result = evaluate(["tá sé"], ["tá sé"])
        assert isinstance(result, MetricsResult)
        assert result.wer == 0.0
        assert result.cer == 0.0

    def test_skip_cer(self) -> None:
        result = evaluate(["tá sé"], ["tá sé"], compute_cer_flag=False)
        assert result.cer is None

    def test_error_breakdown(self) -> None:
        result = evaluate(["céad míle fáilte"], ["cead míle fáilte"])
        breakdown = result.error_breakdown
        assert breakdown["substitution_rate"] > 0
        assert breakdown["insertion_rate"] == 0
        assert breakdown["deletion_rate"] == 0

    def test_total_errors(self) -> None:
        result = evaluate(["a b c"], ["a x c y"])
        assert result.total_errors == result.substitutions + result.insertions + result.deletions


class TestBootstrapErrorCi:
    """Bootstrap 95% CI on aggregate error rate."""

    def test_deterministic_with_same_seed(self) -> None:
        errs = [(1, 0, 1, 5), (0, 1, 0, 4), (2, 0, 0, 6), (0, 0, 1, 3)]
        a = bootstrap_error_ci(errs, n_iterations=200, seed=42)
        b = bootstrap_error_ci(errs, n_iterations=200, seed=42)
        assert a == b

    def test_different_seeds_give_different_results(self) -> None:
        errs = [(i % 3, (i + 1) % 2, (i + 2) % 3, 5 + i) for i in range(20)]
        a = bootstrap_error_ci(errs, n_iterations=500, seed=1)
        b = bootstrap_error_ci(errs, n_iterations=500, seed=2)
        assert a != b

    def test_brackets_point_estimate(self) -> None:
        """The 95% CI should bracket the global aggregate WER on a well-conditioned sample."""
        errs = [(1, 0, 1, 5), (0, 1, 0, 4), (2, 0, 0, 6), (0, 0, 1, 3)]
        total_s = sum(e[0] for e in errs)
        total_i = sum(e[1] for e in errs)
        total_d = sum(e[2] for e in errs)
        total_n = sum(e[3] for e in errs)
        point = (total_s + total_i + total_d) / total_n
        lo, hi = bootstrap_error_ci(errs, n_iterations=500, seed=42)
        assert lo <= point <= hi

    def test_empty_input(self) -> None:
        assert bootstrap_error_ci([], n_iterations=100) == (0.0, 0.0)

    def test_zero_error_input(self) -> None:
        zero = [(0, 0, 0, 5), (0, 0, 0, 4), (0, 0, 0, 3)]
        assert bootstrap_error_ci(zero, n_iterations=200) == (0.0, 0.0)

    def test_smaller_n_gives_wider_ci(self) -> None:
        """The whole point of bootstrapping: small N → wide CI, big N → narrow CI."""
        small = [(1, 0, 1, 5), (0, 1, 0, 4)]  # n=2
        big = [(1, 0, 1, 5), (0, 1, 0, 4)] * 100  # n=200, identical distribution
        lo_s, hi_s = bootstrap_error_ci(small, n_iterations=500, seed=1)
        lo_b, hi_b = bootstrap_error_ci(big, n_iterations=500, seed=1)
        assert (hi_s - lo_s) > (hi_b - lo_b)

    def test_evaluate_attaches_ci_when_requested(self) -> None:
        result = evaluate(
            ["tá sé go maith", "céad míle fáilte"],
            ["tá sé go maith", "cead míle fáilte"],
            compute_ci=True,
            n_bootstrap=200,
        )
        assert result.wer_ci_lo is not None
        assert result.wer_ci_hi is not None
        assert result.wer_ci_lo <= result.wer <= result.wer_ci_hi
        assert result.cer_ci_lo is not None
        assert result.cer_ci_hi is not None

    def test_evaluate_omits_ci_when_disabled(self) -> None:
        result = evaluate(
            ["tá sé"],
            ["tá sé"],
            compute_ci=False,
        )
        assert result.wer_ci_lo is None
        assert result.wer_ci_hi is None
        assert result.cer_ci_lo is None
        assert result.cer_ci_hi is None


class TestEvaluateGrouped:
    def test_per_dialect(self) -> None:
        refs = ["tá sé", "tá sí", "tá mé"]
        hyps = ["tá sé", "tá sé", "tá mé"]
        groups = ["munster", "connacht", "munster"]

        result = evaluate_grouped(refs, hyps, groups, group_name="dialect")
        assert isinstance(result, EvalResult)
        assert len(result.per_group) == 2  # munster and connacht

        # Find connacht group — it should have WER > 0 (sí→sé substitution)
        connacht = [g for g in result.per_group if g.group_value == "connacht"][0]
        assert connacht.metrics.wer > 0
        assert connacht.num_samples == 1

        # Find munster group — perfect match
        munster = [g for g in result.per_group if g.group_value == "munster"][0]
        assert munster.metrics.wer == 0.0
        assert munster.num_samples == 2
