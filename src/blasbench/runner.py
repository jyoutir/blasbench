"""End-to-end evaluation runner: data -> adapter -> normalize -> metrics -> artefacts."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from blasbench.adapters import BaseAdapter
from blasbench.adapters.azure_adapter import AzureAdapter
from blasbench.adapters.elevenlabs_adapter import ElevenLabsAdapter
from blasbench.adapters.google_adapter import GoogleAdapter
from blasbench.adapters.mms_adapter import MMSAdapter
from blasbench.adapters.openai_adapter import OpenAIAdapter
from blasbench.adapters.seamless_adapter import SeamlessM4TAdapter
from blasbench.adapters.speechmatics_adapter import SpeechmaticsAdapter
from blasbench.adapters.transformers_adapter import TransformersAdapter
from blasbench.adapters.wav2vec2_adapter import Wav2Vec2Adapter
from blasbench.config import DatasetConfig, EvalConfig, ModelConfig, NormalizerConfig
from blasbench.data_loader import CSVLoader, DataLoader, HuggingFaceLoader, Sample, TSVLoader
from blasbench.metrics import EvalResult, evaluate, evaluate_grouped
from blasbench.normalizer import BaseNormalizer, get_normalizer
from blasbench.report import result_to_dict, to_markdown

logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = Path("experiments")
RESULTS_DIR = Path("results")


@dataclass
class RunMetadata:
    run_id: str = ""
    timestamp: str = ""
    duration_seconds: float = 0.0
    model_name: str = ""
    model_source: str = ""
    model_num_parameters: int | None = None
    model_dtype: str = ""
    dataset_name: str = ""
    dataset_split: str = ""
    dataset_num_samples: int = 0
    dataset_total_audio_hours: float = 0.0
    normalizer: str = ""
    config_hash: str = ""
    python_version: str = ""
    platform_info: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Prediction:
    sample_id: str
    reference: str
    hypothesis: str
    wer: float
    cer: float | None = None
    audio_duration_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


def _config_hash(config_dict: dict[str, Any]) -> str:
    serialized = json.dumps(config_dict, sort_keys=True, default=str)
    return f"sha256:{hashlib.sha256(serialized.encode()).hexdigest()[:16]}"


def build_metadata(
    model_name: str,
    model_source: str,
    dataset_name: str,
    dataset_split: str,
    normalizer: str,
    config_dict: dict[str, Any],
    num_samples: int = 0,
    total_audio_hours: float = 0.0,
    model_num_parameters: int | None = None,
    model_dtype: str = "",
) -> RunMetadata:
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"{ts}_{model_name.replace('/', '_')}_{dataset_name.replace('/', '_')}_{dataset_split}"
    return RunMetadata(
        run_id=run_id,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        model_name=model_name,
        model_source=model_source,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        dataset_num_samples=num_samples,
        dataset_total_audio_hours=total_audio_hours,
        normalizer=normalizer,
        config_hash=_config_hash(config_dict),
        python_version=platform.python_version(),
        platform_info=platform.platform(),
        model_num_parameters=model_num_parameters,
        model_dtype=model_dtype,
    )


def save_experiment(
    metadata: RunMetadata,
    results_dict: dict[str, Any],
    predictions: list[Prediction],
    config_dict: dict[str, Any],
    markdown_report: str,
    experiments_dir: Path = EXPERIMENTS_DIR,
) -> Path:
    import yaml

    run_dir = experiments_dir / metadata.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "metadata.json").write_text(
        json.dumps(metadata.to_dict(), indent=2, ensure_ascii=False)
    )
    (run_dir / "config.yaml").write_text(
        yaml.dump(config_dict, default_flow_style=False, allow_unicode=True)
    )
    (run_dir / "results.json").write_text(json.dumps(results_dict, indent=2, ensure_ascii=False))
    (run_dir / "results.md").write_text(markdown_report)

    pred_dir = run_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)
    with open(pred_dir / "test.jsonl", "w") as f:
        for p in predictions:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")

    logger.info("Experiment saved: %s (%d predictions)", run_dir, len(predictions))
    return run_dir


def update_summary_csv(
    metadata: RunMetadata,
    results_dict: dict[str, Any],
    results_dir: Path = RESULTS_DIR,
) -> Path:
    import pandas as pd

    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "summary.csv"

    overall = results_dict.get("overall", {})
    row = {
        "run_id": metadata.run_id,
        "timestamp": metadata.timestamp,
        "model": metadata.model_name,
        "dataset": metadata.dataset_name,
        "split": metadata.dataset_split,
        "normalizer": metadata.normalizer,
        "wer": overall.get("wer"),
        "cer": overall.get("cer"),
        "num_utterances": overall.get("num_utterances"),
        "substitutions": overall.get("substitutions"),
        "insertions": overall.get("insertions"),
        "deletions": overall.get("deletions"),
        "rtfx": results_dict.get("rtfx"),
        "duration_s": metadata.duration_seconds,
    }

    df_new = pd.DataFrame([row])
    if csv_path.exists():
        df = pd.concat([pd.read_csv(csv_path), df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(csv_path, index=False)
    return csv_path


def generate_leaderboard(results_dir: Path = RESULTS_DIR) -> str:
    import pandas as pd

    csv_path = results_dir / "summary.csv"
    if not csv_path.exists():
        return "No results found. Run an evaluation first.\n"

    df = pd.read_csv(csv_path).sort_values("wer", ascending=True)
    lines = [
        "# Irish ASR Leaderboard",
        "",
        f"*Generated: {time.strftime('%Y-%m-%d %H:%M UTC')}*",
        "",
        "| Rank | Model | Dataset | WER (%) | CER (%) | RTFx | Utterances |",
        "|------|-------|---------|---------|---------|------|------------|",
    ]
    for i, row in df.iterrows():
        rank = int(i) + 1
        wer = f"{row['wer']:.2f}" if pd.notna(row.get("wer")) else "—"
        cer = f"{row['cer']:.2f}" if pd.notna(row.get("cer")) else "—"
        rtfx = f"{row['rtfx']:.1f}" if pd.notna(row.get("rtfx")) else "—"
        n = str(int(row["num_utterances"])) if pd.notna(row.get("num_utterances")) else "—"
        lines.append(
            f"| {rank} | {row['model']} | {row['dataset']} | {wer} | {cer} | {rtfx} | {n} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_data_loader(config: DatasetConfig) -> DataLoader:
    match config.type.value:
        case "huggingface":
            return HuggingFaceLoader(
                dataset_name=config.name,
                config=config.config,
                split=config.split,
                audio_column=config.audio_column,
                text_column=config.text_column,
                dialect_column=config.dialect_column,
                speaker_column=config.speaker_column,
                max_samples=config.max_samples,
            )
        case "csv":
            assert config.path is not None, "CSV dataset requires 'path'"
            return CSVLoader(
                path=config.path,
                audio_column=config.audio_column,
                text_column=config.text_column,
                max_samples=config.max_samples,
            )
        case "tsv":
            assert config.path is not None, "TSV dataset requires 'path'"
            return TSVLoader(
                path=config.path,
                audio_column=config.audio_column,
                text_column=config.text_column,
                max_samples=config.max_samples,
            )
        case _:
            raise ValueError(f"Unknown dataset type: {config.type.value}")


def build_normalizer(config: NormalizerConfig) -> BaseNormalizer:
    return get_normalizer(config.type.value, remove_diacritics=config.remove_diacritics)


def build_adapter(config: ModelConfig) -> BaseAdapter:
    backend = config.generate_kwargs.get("backend", "transformers")
    kwargs = {k: v for k, v in config.generate_kwargs.items() if k != "backend"}

    match backend:
        case "transformers":
            return TransformersAdapter(
                model_name=config.name,
                language=config.language,
                task=config.task,
                batch_size=config.batch_size,
                chunk_length_s=config.chunk_length_s,
                torch_dtype=config.torch_dtype,
                device=config.device,
                generate_kwargs=kwargs,
            )
        case "wav2vec2":
            return Wav2Vec2Adapter(
                model_name=config.name,
                language=config.language,
                torch_dtype=config.torch_dtype,
                device=config.device,
            )
        case "seamless":
            return SeamlessM4TAdapter(
                model_name=config.name,
                torch_dtype=config.torch_dtype,
                device=config.device,
            )
        case "mms":
            return MMSAdapter(
                model_name=config.name,
                torch_dtype=config.torch_dtype,
                device=config.device,
            )
        case "azure":
            return AzureAdapter()
        case "elevenlabs":
            assert config.name.startswith("elevenlabs/"), (
                "elevenlabs backend expects elevenlabs/<model>"
            )
            return ElevenLabsAdapter(model_id=config.name.removeprefix("elevenlabs/"))
        case "google":
            return GoogleAdapter()
        case "openai":
            assert config.name.startswith("openai/"), "openai backend expects openai/<model>"
            return OpenAIAdapter(model_name=config.name.removeprefix("openai/"))
        case "speechmatics":
            return SpeechmaticsAdapter()
        case _:
            raise ValueError(f"Unknown backend: {backend}")


def run_evaluation(config: EvalConfig, save_experiment_flag: bool = False) -> EvalResult:
    run_start = time.time()
    logger.info("Evaluation: model=%s dataset=%s", config.model.name, config.dataset.name)

    loader = build_data_loader(config.dataset)
    samples = list(loader)

    adapter = build_adapter(config.model)
    adapter.load()
    try:
        start = time.time()
        audios = [s.audio_array for s in samples]
        srs = [s.sample_rate for s in samples]
        transcriptions = adapter.transcribe_batch(audios, srs)
        transcribe_elapsed = time.time() - start
    finally:
        adapter.unload()

    hypotheses = [t.text for t in transcriptions]
    total_audio_s = sum(len(np.asarray(s.audio_array)) / s.sample_rate for s in samples)
    rtfx = total_audio_s / transcribe_elapsed if transcribe_elapsed > 0 else None

    normalizer = build_normalizer(config.normalizer)
    norm_refs = normalizer.normalize_batch([s.reference for s in samples])
    norm_hyps = normalizer.normalize_batch(hypotheses)

    dialects = [s.metadata.get("dialect", "unknown") for s in samples]
    has_dialects = any(d != "unknown" for d in dialects)

    if config.per_dialect and has_dialects:
        result = evaluate_grouped(
            norm_refs,
            norm_hyps,
            dialects,
            group_name="dialect",
            compute_cer_flag=config.compute_cer,
        )
    elif config.per_speaker:
        speakers = [s.metadata.get("speaker", "unknown") for s in samples]
        result = evaluate_grouped(
            norm_refs,
            norm_hyps,
            speakers,
            group_name="speaker",
            compute_cer_flag=config.compute_cer,
        )
    else:
        overall = evaluate(norm_refs, norm_hyps, compute_cer_flag=config.compute_cer)
        result = EvalResult(overall=overall)

    result.model_name = config.model.name
    result.dataset_name = f"{config.dataset.name}/{config.dataset.config}"
    result.normalizer_name = config.normalizer.type.value
    result.rtfx = rtfx
    result.total_audio_duration_s = total_audio_s
    result.total_processing_time_s = transcribe_elapsed

    if save_experiment_flag:
        _save(config, result, samples, norm_refs, norm_hyps, time.time() - run_start)

    return result


def _save(
    config: EvalConfig,
    result: EvalResult,
    samples: list[Sample],
    norm_refs: list[str],
    norm_hyps: list[str],
    run_duration: float,
) -> None:
    import jiwer

    config_dict = config.model_dump(mode="json")
    metadata = build_metadata(
        model_name=config.model.name,
        model_source=config.model.name,
        dataset_name=config.dataset.name,
        dataset_split=config.dataset.split,
        normalizer=config.normalizer.type.value,
        config_dict=config_dict,
        num_samples=len(samples),
        total_audio_hours=result.total_audio_duration_s / 3600,
        model_dtype=config.model.torch_dtype,
    )
    metadata.duration_seconds = run_duration

    predictions: list[Prediction] = []
    for i, (ref, hyp, sample) in enumerate(zip(norm_refs, norm_hyps, samples)):
        utt_wer = jiwer.wer(ref, hyp) if ref.strip() else (0.0 if not hyp.strip() else 1.0)
        utt_cer: float | None = None
        if config.compute_cer:
            utt_cer = jiwer.cer(ref, hyp) if ref.strip() else (0.0 if not hyp.strip() else 1.0)
        predictions.append(
            Prediction(
                sample_id=sample.sample_id or str(i),
                reference=ref,
                hypothesis=hyp,
                wer=utt_wer,
                cer=utt_cer,
                metadata=sample.metadata,
            )
        )

    results_dict = result_to_dict(result)
    save_experiment(
        metadata=metadata,
        results_dict=results_dict,
        predictions=predictions,
        config_dict=config_dict,
        markdown_report=to_markdown(result),
    )
    update_summary_csv(metadata, results_dict)
