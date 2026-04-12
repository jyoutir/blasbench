"""WER and CER with bootstrap 95% CIs.

WER is aggregated globally (sum S+I+D / sum N). Bootstrap resamples
utterances with replacement and recomputes that aggregate.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field

import jiwer


@dataclass(frozen=True)
class DistributionStats:
    mean: float
    median: float
    std: float
    p90: float
    p95: float
    p99: float
    min: float
    max: float

    def to_dict(self) -> dict[str, float]:
        return {k: round(v, 4) for k, v in self.__dict__.items()}


def compute_distribution(values: list[float]) -> DistributionStats:
    if not values:
        return DistributionStats(*([0.0] * 8))
    v = sorted(values)
    n = len(v)

    def pct(p: float) -> float:
        k = (n - 1) * p / 100.0
        f = int(k)
        return v[f] + (k - f) * (v[min(f + 1, n - 1)] - v[f])

    return DistributionStats(
        mean=statistics.mean(values),
        median=statistics.median(values),
        std=statistics.stdev(values) if n > 1 else 0.0,
        p90=pct(90),
        p95=pct(95),
        p99=pct(99),
        min=v[0],
        max=v[-1],
    )


@dataclass(frozen=True)
class MetricsResult:
    wer: float
    cer: float | None
    num_utterances: int
    total_ref_words: int
    substitutions: int
    insertions: int
    deletions: int
    hits: int
    per_utterance_wer: list[float] = field(default_factory=list)
    per_utterance_cer: list[float] = field(default_factory=list)
    wer_ci_lo: float | None = None
    wer_ci_hi: float | None = None
    cer_ci_lo: float | None = None
    cer_ci_hi: float | None = None

    @property
    def total_errors(self) -> int:
        return self.substitutions + self.insertions + self.deletions

    @property
    def error_breakdown(self) -> dict[str, float]:
        n = self.total_ref_words or 1
        return {
            "substitution_rate": self.substitutions / n,
            "insertion_rate": self.insertions / n,
            "deletion_rate": self.deletions / n,
        }

    @property
    def wer_distribution(self) -> DistributionStats | None:
        return compute_distribution(self.per_utterance_wer) if self.per_utterance_wer else None

    @property
    def cer_distribution(self) -> DistributionStats | None:
        return compute_distribution(self.per_utterance_cer) if self.per_utterance_cer else None


@dataclass
class PerGroupResult:
    group_name: str
    group_value: str
    metrics: MetricsResult
    num_samples: int


@dataclass
class EvalResult:
    overall: MetricsResult
    per_group: list[PerGroupResult] = field(default_factory=list)
    model_name: str = ""
    dataset_name: str = ""
    normalizer_name: str = ""
    rtfx: float | None = None
    total_audio_duration_s: float = 0.0
    total_processing_time_s: float = 0.0


def _check(refs: list[str], hyps: list[str], label: str) -> None:
    if len(refs) != len(hyps):
        raise ValueError(f"lists must have same length, got {len(refs)} and {len(hyps)}")
    if not refs:
        raise ValueError(f"Cannot compute {label} on empty lists")


def compute_wer(references: list[str], hypotheses: list[str]) -> jiwer.WordOutput:
    _check(references, hypotheses, "WER")
    return jiwer.process_words(references, hypotheses)


def compute_cer(references: list[str], hypotheses: list[str]) -> jiwer.CharacterOutput:
    _check(references, hypotheses, "CER")
    return jiwer.process_characters(references, hypotheses)


def _per_utt_rate(refs: list[str], hyps: list[str], char: bool) -> list[float]:
    fn = jiwer.cer if char else jiwer.wer
    return [fn(r, h) if r.strip() else (0.0 if not h.strip() else 1.0) for r, h in zip(refs, hyps)]


def _per_utt_errors(
    refs: list[str], hyps: list[str], char: bool
) -> list[tuple[int, int, int, int]]:
    out: list[tuple[int, int, int, int]] = []
    for r, h in zip(refs, hyps):
        if not r.strip():
            out.append((0, len(h) if char else len(h.split()), 0, 0))
            continue
        o = jiwer.process_characters([r], [h]) if char else jiwer.process_words([r], [h])
        out.append(
            (o.substitutions, o.insertions, o.deletions, o.hits + o.substitutions + o.deletions)
        )
    return out


def bootstrap_error_ci(
    per_utterance_errors: list[tuple[int, int, int, int]],
    n_iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    n = len(per_utterance_errors)
    if n == 0:
        return (0.0, 0.0)

    rng = random.Random(seed)
    rates: list[float] = []
    for _ in range(n_iterations):
        s = i = d = nn = 0
        for _j in range(n):
            es, ei, ed, en = per_utterance_errors[rng.randrange(n)]
            s += es
            i += ei
            d += ed
            nn += en
        if nn > 0:
            rates.append((s + i + d) / nn)
    if not rates:
        return (0.0, 0.0)
    rates.sort()
    a = (1.0 - confidence) / 2.0
    lo = max(0, int(a * len(rates)))
    hi = min(len(rates) - 1, int((1.0 - a) * len(rates)) - 1)
    return (rates[lo], rates[hi])


def evaluate(
    references: list[str],
    hypotheses: list[str],
    compute_cer_flag: bool = True,
    compute_distributions: bool = True,
    compute_ci: bool = True,
    n_bootstrap: int = 1000,
    bootstrap_seed: int = 42,
) -> MetricsResult:
    word = compute_wer(references, hypotheses)
    cer_val = compute_cer(references, hypotheses).cer if compute_cer_flag else None

    per_wer = _per_utt_rate(references, hypotheses, char=False) if compute_distributions else []
    per_cer = (
        _per_utt_rate(references, hypotheses, char=True)
        if compute_distributions and compute_cer_flag
        else []
    )

    wer_lo = wer_hi = cer_lo = cer_hi = None
    if compute_ci:
        wer_lo, wer_hi = bootstrap_error_ci(
            _per_utt_errors(references, hypotheses, char=False),
            n_iterations=n_bootstrap,
            seed=bootstrap_seed,
        )
        if compute_cer_flag:
            cer_lo, cer_hi = bootstrap_error_ci(
                _per_utt_errors(references, hypotheses, char=True),
                n_iterations=n_bootstrap,
                seed=bootstrap_seed,
            )

    return MetricsResult(
        wer=word.wer,
        cer=cer_val,
        num_utterances=len(references),
        total_ref_words=word.substitutions + word.deletions + word.hits,
        substitutions=word.substitutions,
        insertions=word.insertions,
        deletions=word.deletions,
        hits=word.hits,
        per_utterance_wer=per_wer,
        per_utterance_cer=per_cer,
        wer_ci_lo=wer_lo,
        wer_ci_hi=wer_hi,
        cer_ci_lo=cer_lo,
        cer_ci_hi=cer_hi,
    )


def evaluate_grouped(
    references: list[str],
    hypotheses: list[str],
    groups: list[str],
    group_name: str = "dialect",
    compute_cer_flag: bool = True,
    compute_distributions: bool = True,
) -> EvalResult:
    overall = evaluate(references, hypotheses, compute_cer_flag, compute_distributions)
    by_group: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        by_group.setdefault(g, []).append(i)

    per_group = [
        PerGroupResult(
            group_name=group_name,
            group_value=v,
            metrics=evaluate(
                [references[i] for i in by_group[v]],
                [hypotheses[i] for i in by_group[v]],
                compute_cer_flag,
                compute_distributions,
            ),
            num_samples=len(by_group[v]),
        )
        for v in sorted(by_group)
    ]
    return EvalResult(overall=overall, per_group=per_group)
