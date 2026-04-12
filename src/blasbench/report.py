"""JSON / CSV / Markdown formatting of EvalResult."""

from __future__ import annotations

import json
import time
from io import StringIO
from typing import Any

import pandas as pd
from rich.console import Console

from blasbench.metrics import EvalResult, MetricsResult


def _metrics_dict(m: MetricsResult) -> dict[str, Any]:
    b = m.error_breakdown
    out: dict[str, Any] = {
        "wer": round(m.wer * 100, 2),
        "cer": round(m.cer * 100, 2) if m.cer is not None else None,
        "num_utterances": m.num_utterances,
        "total_ref_words": m.total_ref_words,
        "substitutions": m.substitutions,
        "insertions": m.insertions,
        "deletions": m.deletions,
        "hits": m.hits,
        "substitution_rate": round(b["substitution_rate"] * 100, 2),
        "insertion_rate": round(b["insertion_rate"] * 100, 2),
        "deletion_rate": round(b["deletion_rate"] * 100, 2),
    }
    if m.wer_distribution is not None:
        out["wer_distribution"] = m.wer_distribution.to_dict()
    if m.cer_distribution is not None:
        out["cer_distribution"] = m.cer_distribution.to_dict()
    return out


def result_to_dict(result: EvalResult) -> dict[str, Any]:
    out: dict[str, Any] = {
        "model": result.model_name,
        "dataset": result.dataset_name,
        "normalizer": result.normalizer_name,
        "overall": _metrics_dict(result.overall),
    }
    if result.rtfx is not None:
        out["rtfx"] = round(result.rtfx, 2)
    if result.total_audio_duration_s > 0:
        out["total_audio_duration_s"] = round(result.total_audio_duration_s, 2)
    if result.total_processing_time_s > 0:
        out["total_processing_time_s"] = round(result.total_processing_time_s, 2)
    if result.per_group:
        out["per_group"] = [
            {
                "group_name": g.group_name,
                "group_value": g.group_value,
                "num_samples": g.num_samples,
                **_metrics_dict(g.metrics),
            }
            for g in result.per_group
        ]
    return out


def to_json(result: EvalResult, indent: int = 2) -> str:
    return json.dumps(result_to_dict(result), indent=indent, ensure_ascii=False)


def to_csv(result: EvalResult) -> str:
    def flatten(d: dict[str, Any]) -> dict[str, Any]:
        for k in ("wer_distribution", "cer_distribution"):
            dist = d.pop(k, None)
            if dist:
                for sk, sv in dist.items():
                    d[f"{k}_{sk}"] = sv
        return d

    base = {"model": result.model_name, "dataset": result.dataset_name}
    rows = [flatten({**_metrics_dict(result.overall), "group": "overall", **base})]
    for g in result.per_group:
        rows.append(
            flatten(
                {
                    **_metrics_dict(g.metrics),
                    "group": f"{g.group_name}={g.group_value}",
                    "num_samples": g.num_samples,
                    **base,
                }
            )
        )
    buf = StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    return buf.getvalue()


def to_markdown(result: EvalResult) -> str:
    m = result.overall
    fm = [
        "---",
        "type: experiment-run",
        f"model: {result.model_name}",
        f"dataset: {result.dataset_name}",
        f"normalizer: {result.normalizer_name}",
        f"wer: {m.wer * 100:.2f}",
    ]
    if m.cer is not None:
        fm.append(f"cer: {m.cer * 100:.2f}")
    if result.rtfx is not None:
        fm.append(f"rtfx: {result.rtfx:.2f}")
    fm += [f"num_utterances: {m.num_utterances}", f"date: {time.strftime('%Y-%m-%d')}", "---", ""]

    body = [
        f"# {result.model_name} on {result.dataset_name}",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| WER (%) | {m.wer * 100:.2f} |",
    ]
    if m.cer is not None:
        body.append(f"| CER (%) | {m.cer * 100:.2f} |")
    if result.rtfx is not None:
        body.append(f"| RTFx | {result.rtfx:.2f} |")
    body += [
        f"| Utterances | {m.num_utterances} |",
        f"| Substitutions | {m.substitutions} |",
        f"| Insertions | {m.insertions} |",
        f"| Deletions | {m.deletions} |",
        "",
    ]

    d = m.wer_distribution
    if d is not None:
        body += [
            "## WER Distribution",
            "",
            "| Statistic | Value |",
            "|-----------|-------|",
            f"| Mean | {d.mean * 100:.2f}% |",
            f"| Median | {d.median * 100:.2f}% |",
            f"| P90 | {d.p90 * 100:.2f}% |",
            f"| P95 | {d.p95 * 100:.2f}% |",
            f"| Max | {d.max * 100:.2f}% |",
            "",
        ]

    if result.per_group:
        gn = result.per_group[0].group_name
        body += [
            f"## Results by {gn.title()}",
            "",
            f"| {gn.title()} | Samples | WER (%) | CER (%) |",
            "|-----|-----|-----|-----|",
        ]
        for g in result.per_group:
            cer = f"{g.metrics.cer * 100:.2f}" if g.metrics.cer is not None else "—"
            body.append(
                f"| {g.group_value} | {g.num_samples} | {g.metrics.wer * 100:.2f} | {cer} |"
            )
        body.append("")
    return "\n".join(fm + body)


def format_result(result: EvalResult, fmt: str = "table", console: Console | None = None) -> str:
    match fmt:
        case "json":
            return to_json(result)
        case "csv":
            return to_csv(result)
        case "markdown" | "table":
            md = to_markdown(result)
            if fmt == "table" and console is not None:
                console.print(md)
                return ""
            return md
        case _:
            raise ValueError(f"Unknown format: {fmt!r}")
