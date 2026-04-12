#!/usr/bin/env python3
"""Run all baseline benchmarks for Irish ASR.

Iterates model x dataset pairs, writes per-run experiment artefacts plus
an aggregated summary CSV. Commercial API models are skipped cleanly if
their required env vars are missing.

Usage:
    python scripts/run_baselines.py
    python scripts/run_baselines.py --max-samples 50
    python scripts/run_baselines.py --models whisper-small
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from blasbench.config import (
    DatasetConfig,
    DatasetType,
    EvalConfig,
    ModelConfig,
    NormalizerConfig,
    NormalizerType,
)
from blasbench.runner import run_evaluation

logger = logging.getLogger("baselines")

MODELS: list[dict[str, Any]] = [
    {"name": "openai/whisper-tiny", "backend": "transformers", "language": "irish"},
    {"name": "openai/whisper-small", "backend": "transformers", "language": "irish"},
    {"name": "openai/whisper-medium", "backend": "transformers", "language": "irish"},
    {"name": "openai/whisper-large-v2", "backend": "transformers", "language": "irish"},
    {"name": "openai/whisper-large-v3", "backend": "transformers", "language": "irish"},
    {"name": "openai/whisper-large-v3-turbo", "backend": "transformers", "language": "irish"},
    {"name": "cpierse/wav2vec2-large-xlsr-53-irish", "backend": "wav2vec2"},
    {"name": "Aditya3107/wav2vec2-large-xls-r-1b-ga-ie", "backend": "wav2vec2"},
    {"name": "jimregan/wav2vec2-large-xlsr-irish-basic", "backend": "wav2vec2"},
    {"name": "kingabzpro/wav2vec2-large-xls-r-1b-Irish", "backend": "wav2vec2"},
    {"name": "facebook/seamless-m4t-v2-large", "backend": "seamless"},
    {"name": "facebook/mms-1b-all", "backend": "mms"},
    {
        "name": "azure/speech-ga-IE",
        "backend": "azure",
        "requires_env": ("AZURE_SPEECH_KEY", "AZURE_SPEECH_REGION"),
    },
    {"name": "openai/whisper-1-api", "backend": "openai", "requires_env": ("OPENAI_API_KEY",)},
    {
        "name": "elevenlabs/scribe_v1",
        "backend": "elevenlabs",
        "requires_env": ("ELEVENLABS_API_KEY",),
    },
    {
        "name": "speechmatics/ga",
        "backend": "speechmatics",
        "requires_env": ("SPEECHMATICS_API_KEY",),
    },
    {
        "name": "google/chirp_2-ga-IE",
        "backend": "google",
        "requires_env": ("GOOGLE_CLOUD_PROJECT", "GOOGLE_APPLICATION_CREDENTIALS"),
    },
]

DATASETS: list[dict[str, Any]] = [
    {
        "name": "mozilla-foundation/common_voice_17_0",
        "config": "ga-IE",
        "split": "test",
        "short_name": "commonvoice-ga",
    },
    {
        "name": "google/fleurs",
        "config": "ga_ie",
        "split": "test",
        "short_name": "fleurs-ga",
        "text_column": "transcription",
    },
]


def build_config(
    model: dict[str, Any], dataset: dict[str, Any], max_samples: int | None
) -> EvalConfig:
    generate_kwargs: dict[str, Any] = {"backend": model["backend"]}
    return EvalConfig(
        model=ModelConfig(
            name=model["name"],
            language=model.get("language", "irish"),
            generate_kwargs=generate_kwargs,
        ),
        dataset=DatasetConfig(
            type=DatasetType.HUGGINGFACE,
            name=dataset["name"],
            config=dataset["config"],
            split=dataset["split"],
            text_column=dataset.get("text_column", "sentence"),
            max_samples=max_samples,
        ),
        normalizer=NormalizerConfig(type=NormalizerType.IRISH),
        save_experiment=True,
        per_dialect=False,
    )


def env_missing(model: dict[str, Any]) -> list[str]:
    required: tuple[str, ...] = model.get("requires_env", ())
    return [v for v in required if not os.environ.get(v)]


def main() -> None:
    p = argparse.ArgumentParser(description="Run Irish ASR baselines")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--dry-run", action="store_true", help="Skip execution, just print plan")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    logger.setLevel(logging.INFO)

    models = MODELS
    if args.models:
        models = [m for m in MODELS if any(a in m["name"] for a in args.models)]

    datasets = DATASETS
    if args.datasets:
        datasets = [d for d in DATASETS if d["short_name"] in args.datasets]

    total = len(models) * len(datasets)
    done = failed = skipped = 0

    for i, model in enumerate(models):
        for j, dataset in enumerate(datasets):
            idx = i * len(datasets) + j + 1
            logger.info("[%d/%d] %s on %s", idx, total, model["name"], dataset["short_name"])

            missing = env_missing(model)
            if missing:
                logger.warning("Skipping %s: missing env %s", model["name"], missing)
                skipped += 1
                continue

            if args.dry_run:
                logger.info("DRY-RUN: %s on %s", model["name"], dataset["short_name"])
                continue

            start = time.time()
            try:
                result = run_evaluation(
                    build_config(model, dataset, args.max_samples), save_experiment_flag=True
                )
                elapsed = time.time() - start
                logger.info("DONE in %.1fs: WER=%.2f%%", elapsed, result.overall.wer * 100)
                done += 1
            except Exception as e:
                logger.exception("FAILED %s on %s: %s", model["name"], dataset["short_name"], e)
                failed += 1

    logger.info("Summary: %d done, %d failed, %d skipped of %d", done, failed, skipped, total)


if __name__ == "__main__":
    main()
