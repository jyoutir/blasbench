#!/usr/bin/env python3
"""Generate comparison figures from baseline benchmark results.

Produces four charts:
1. WER comparison bar chart (grouped by dataset)
2. CER comparison bar chart (grouped by dataset)
3. WER vs model size scatter plot
4. Error type breakdown stacked bar chart

Reads from results/baselines_summary.csv (produced by run_baselines.py).

Usage:
    python scripts/generate_figures.py
    python scripts/generate_figures.py --csv results/baselines_summary.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("results/figures")


def _short_name(model: str) -> str:
    """Shorten model name for chart labels."""
    name = model.split("/")[-1]
    # Trim common prefixes
    for prefix in ["whisper-", "wav2vec2-large-xlsr-53-", "seamless-m4t-", "mms-"]:
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name


def generate_wer_bar_chart(df: pd.DataFrame, output_dir: Path) -> Path:
    """Bar chart comparing WER across models, grouped by dataset."""
    import matplotlib.pyplot as plt
    import numpy as np

    datasets = df["dataset"].unique()
    models = df["model"].unique()
    short_names = [_short_name(m) for m in models]

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(models))
    width = 0.8 / len(datasets)

    for i, ds in enumerate(datasets):
        subset = df[df["dataset"] == ds]
        wers = []
        for m in models:
            row = subset[subset["model"] == m]
            wers.append(row["wer"].values[0] if len(row) > 0 else 0)
        bars = ax.bar(x + i * width, wers, width, label=ds, zorder=3)
        # Add value labels
        for bar, val in zip(bars, wers):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_xlabel("Model")
    ax.set_ylabel("WER (%)")
    ax.set_title("Irish ASR Baselines: Word Error Rate")
    ax.set_xticks(x + width * (len(datasets) - 1) / 2)
    ax.set_xticklabels(short_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    path = output_dir / "wer_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def generate_cer_bar_chart(df: pd.DataFrame, output_dir: Path) -> Path:
    """Bar chart comparing CER across models, grouped by dataset."""
    import matplotlib.pyplot as plt
    import numpy as np

    datasets = df["dataset"].unique()
    models = df["model"].unique()
    short_names = [_short_name(m) for m in models]

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(models))
    width = 0.8 / len(datasets)

    for i, ds in enumerate(datasets):
        subset = df[df["dataset"] == ds]
        cers = []
        for m in models:
            row = subset[subset["model"] == m]
            cers.append(row["cer"].values[0] if len(row) > 0 else 0)
        bars = ax.bar(x + i * width, cers, width, label=ds, zorder=3)
        for bar, val in zip(bars, cers):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_xlabel("Model")
    ax.set_ylabel("CER (%)")
    ax.set_title("Irish ASR Baselines: Character Error Rate")
    ax.set_xticks(x + width * (len(datasets) - 1) / 2)
    ax.set_xticklabels(short_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    path = output_dir / "cer_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def generate_wer_vs_size_scatter(df: pd.DataFrame, output_dir: Path) -> Path:
    """Scatter plot of WER vs model size (parameters)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 7))

    datasets = df["dataset"].unique()
    markers = ["o", "s", "D", "^", "v"]

    for i, ds in enumerate(datasets):
        subset = df[df["dataset"] == ds]
        params = subset["num_parameters"] / 1e9  # Convert to billions
        wers = subset["wer"]
        ax.scatter(params, wers, label=ds, marker=markers[i % len(markers)], s=80, zorder=3)

        # Annotate points
        for _, row in subset.iterrows():
            ax.annotate(
                _short_name(row["model"]),
                (row["num_parameters"] / 1e9, row["wer"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
                alpha=0.8,
            )

    ax.set_xlabel("Parameters (Billions)")
    ax.set_ylabel("WER (%)")
    ax.set_title("Irish ASR: WER vs Model Size")
    ax.legend()
    ax.grid(alpha=0.3, zorder=0)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)

    fig.tight_layout()
    path = output_dir / "wer_vs_model_size.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def generate_error_breakdown_chart(df: pd.DataFrame, output_dir: Path) -> Path:
    """Stacked bar chart showing substitution/insertion/deletion breakdown."""
    import matplotlib.pyplot as plt
    import numpy as np

    # Use first dataset for the breakdown
    dataset = df["dataset"].unique()[0]
    subset = df[df["dataset"] == dataset].copy()
    subset = subset.sort_values("wer")

    models = subset["model"].values
    short_names = [_short_name(m) for m in models]

    sub_rates = subset["sub_rate"].values
    ins_rates = subset["ins_rate"].values
    del_rates = subset["del_rate"].values

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(models))

    ax.bar(x, sub_rates, label="Substitutions", color="#e74c3c", zorder=3)
    ax.bar(x, ins_rates, bottom=sub_rates, label="Insertions", color="#3498db", zorder=3)
    ax.bar(
        x,
        del_rates,
        bottom=sub_rates + ins_rates,
        label="Deletions",
        color="#2ecc71",
        zorder=3,
    )

    ax.set_xlabel("Model")
    ax.set_ylabel("Error Rate (%)")
    ax.set_title(f"Error Type Breakdown — {dataset}")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, zorder=0)

    fig.tight_layout()
    path = output_dir / "error_breakdown.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def generate_all_figures(csv_path: Path | None = None) -> list[Path]:
    """Generate all comparison figures from the summary CSV.

    Args:
        csv_path: Path to baselines_summary.csv. Defaults to results/baselines_summary.csv.

    Returns:
        List of paths to generated figure files.
    """
    import pandas as pd

    if csv_path is None:
        csv_path = RESULTS_DIR / "baselines_summary.csv"

    if not csv_path.exists():
        logger.warning("Summary CSV not found: %s — skipping figure generation", csv_path)
        return []

    df = pd.read_csv(csv_path)
    if df.empty:
        logger.warning("Summary CSV is empty — skipping figure generation")
        return []

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Use non-interactive backend
    import matplotlib

    matplotlib.use("Agg")

    figures: list[Path] = []
    figures.append(generate_wer_bar_chart(df, FIGURES_DIR))
    figures.append(generate_cer_bar_chart(df, FIGURES_DIR))
    figures.append(generate_wer_vs_size_scatter(df, FIGURES_DIR))

    # Error breakdown needs sub_rate/ins_rate/del_rate columns
    if all(col in df.columns for col in ["sub_rate", "ins_rate", "del_rate"]):
        figures.append(generate_error_breakdown_chart(df, FIGURES_DIR))
    else:
        logger.warning("Skipping error breakdown chart — missing rate columns in CSV")

    logger.info("Generated %d figures in %s", len(figures), FIGURES_DIR)
    return figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate baseline comparison figures")
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to baselines_summary.csv",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    figures = generate_all_figures(csv_path=args.csv)
    if figures:
        print(f"Generated {len(figures)} figures:")
        for f in figures:
            print(f"  {f}")
    else:
        print("No figures generated. Run benchmarks first to produce results/baselines_summary.csv")


if __name__ == "__main__":
    main()
