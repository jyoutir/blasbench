"""CLI interface for the Blas Voice evaluation harness.

Usage:
    blasbench evaluate --model whisper-large-v3 --dataset common-voice-ga
    blasbench benchmark --config benchmark_suite.yaml
    blasbench normalize "Céad míle fáilte!"
    blasbench leaderboard
    blasbench list-datasets
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console

from blasbench.config import (
    BenchmarkSuiteConfig,
    DatasetConfig,
    EvalConfig,
    ModelConfig,
    NormalizerConfig,
    NormalizerType,
    OutputFormat,
)
from blasbench.normalizer import get_normalizer
from blasbench.report import format_result

app = typer.Typer(
    name="blasbench",
    help="Irish (Gaeilge) ASR evaluation harness",
    no_args_is_help=True,
)
console = Console()

# Well-known model shortcuts
MODEL_ALIASES: dict[str, str] = {
    "whisper-tiny": "openai/whisper-tiny",
    "whisper-base": "openai/whisper-base",
    "whisper-small": "openai/whisper-small",
    "whisper-medium": "openai/whisper-medium",
    "whisper-large": "openai/whisper-large-v3",
    "whisper-large-v2": "openai/whisper-large-v2",
    "whisper-large-v3": "openai/whisper-large-v3",
    "whisper-large-v3-turbo": "openai/whisper-large-v3-turbo",
}

# Well-known dataset shortcuts
DATASET_ALIASES: dict[str, tuple[str, str]] = {
    "common-voice-ga": ("mozilla-foundation/common_voice_17_0", "ga-IE"),
    "common-voice-ga-16": ("mozilla-foundation/common_voice_16_0", "ga-IE"),
    "fleurs-ga": ("google/fleurs", "ga_ie"),
}


def _resolve_model(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


def _resolve_dataset(name: str) -> tuple[str, str]:
    return DATASET_ALIASES.get(name, (name, "ga-IE"))


@app.command()
def evaluate(  # noqa: B008
    model: str = typer.Option("whisper-large-v3", "--model", "-m", help="Model name or alias"),
    dataset: str = typer.Option("common-voice-ga", "--dataset", "-d", help="Dataset name or alias"),
    split: str = typer.Option("test", "--split", "-s", help="Dataset split"),
    normalizer: str = typer.Option(
        "irish", "--normalizer", "-n", help="Normalizer: irish, basic_whisper, none"
    ),
    output_format: str = typer.Option(
        "table", "--format", "-f", help="Output: table, json, csv, markdown"
    ),
    output_path: str | None = typer.Option(None, "--output", "-o", help="Save output to file"),
    max_samples: int | None = typer.Option(
        None, "--max-samples", help="Limit samples (for testing)"
    ),
    batch_size: int = typer.Option(16, "--batch-size", "-b", help="Batch size"),
    no_cer: bool = typer.Option(False, "--no-cer", help="Skip CER computation"),
    per_dialect: bool = typer.Option(
        True, "--per-dialect/--no-per-dialect", help="Per-dialect breakdown"
    ),
    per_speaker: bool = typer.Option(False, "--per-speaker", help="Per-speaker"),
    device: str = typer.Option("auto", "--device", help="Device: auto, cpu, cuda:0"),
    save_exp: bool = typer.Option(False, "--save-experiment", help="Save experiment artifacts"),
    config_file: Path | None = typer.Option(None, "--config", "-c", help="YAML config file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
) -> None:
    """Run ASR evaluation on an Irish speech dataset."""
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    # Build config
    if config_file:
        eval_config = EvalConfig.from_yaml(config_file)
    else:
        model_name = _resolve_model(model)
        dataset_name, dataset_config = _resolve_dataset(dataset)

        eval_config = EvalConfig(
            model=ModelConfig(
                name=model_name,
                batch_size=batch_size,
                device=device,
            ),
            dataset=DatasetConfig(
                name=dataset_name,
                config=dataset_config,
                split=split,
                max_samples=max_samples,
            ),
            normalizer=NormalizerConfig(type=NormalizerType(normalizer)),
            output_format=OutputFormat(output_format),
            output_path=output_path,
            compute_cer=not no_cer,
            per_dialect=per_dialect,
            per_speaker=per_speaker,
            save_experiment=save_exp,
        )

    from blasbench.runner import run_evaluation

    result = run_evaluation(eval_config, save_experiment_flag=eval_config.save_experiment)

    # Output
    formatted = format_result(result, fmt=eval_config.output_format.value, console=console)

    if eval_config.output_path:
        Path(eval_config.output_path).write_text(formatted)
        console.print(f"[green]Results saved to {eval_config.output_path}[/green]")
    elif formatted:
        console.print(formatted)


@app.command()
def benchmark(  # noqa: B008
    config_file: Path = typer.Argument(..., help="Benchmark suite YAML config"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
) -> None:
    """Run a benchmark suite (model x dataset matrix evaluation)."""
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    suite = BenchmarkSuiteConfig.from_yaml(config_file)
    configs = suite.to_eval_configs()
    console.print(
        f"[bold]Running benchmark suite: "
        f"{len(suite.models)} models x {len(suite.datasets)} datasets "
        f"= {len(configs)} evaluations[/bold]"
    )

    from blasbench.runner import run_evaluation

    for i, eval_config in enumerate(configs, 1):
        console.print(
            f"\n[bold cyan][{i}/{len(configs)}][/bold cyan] "
            f"{eval_config.model.name} on {eval_config.dataset.name}"
        )
        try:
            result = run_evaluation(eval_config, save_experiment_flag=eval_config.save_experiment)
            format_result(result, fmt="table", console=console)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            continue

    console.print("\n[bold green]Benchmark suite complete.[/bold green]")

    # Generate leaderboard if experiments were saved
    if suite.save_experiments:
        from blasbench.runner import generate_leaderboard

        lb = generate_leaderboard()
        console.print(lb)


@app.command()
def normalize(
    text: str = typer.Argument(..., help="Text to normalize"),
    normalizer_name: str = typer.Option("irish", "--normalizer", "-n", help="Normalizer to use"),
) -> None:
    """Normalize text using an Irish-specific normalizer."""
    norm = get_normalizer(normalizer_name)
    result = norm(text)
    console.print(f"[bold]Input:[/bold]  {text}")
    console.print(f"[bold]Output:[/bold] {result}")


@app.command("leaderboard")
def leaderboard() -> None:
    """Display the leaderboard from previous benchmark runs."""
    from blasbench.runner import generate_leaderboard

    lb = generate_leaderboard()
    console.print(lb)


@app.command("list-datasets")
def list_datasets() -> None:
    """List available dataset aliases."""
    console.print("[bold]Available dataset aliases:[/bold]")
    for alias, (name, config) in DATASET_ALIASES.items():
        console.print(f"  {alias:25s} -> {name} ({config})")


@app.command("list-models")
def list_models() -> None:
    """List available model aliases."""
    console.print("[bold]Available model aliases:[/bold]")
    for alias, name in MODEL_ALIASES.items():
        console.print(f"  {alias:30s} -> {name}")


if __name__ == "__main__":
    app()
