"""llem run — primary command for running LLM efficiency experiments."""

from __future__ import annotations

import signal
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import ValidationError
from tqdm.auto import tqdm

from llenergymeasure import run_experiment
from llenergymeasure.cli._display import (
    format_error,
    format_validation_error,
    print_dry_run,
    print_experiment_header,
    print_result_summary,
)
from llenergymeasure.cli._vram import estimate_vram, get_gpu_vram_gb
from llenergymeasure.config.loader import load_experiment_config
from llenergymeasure.exceptions import (
    BackendError,
    ConfigError,
    ExperimentError,
    PreFlightError,
)

# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------


def run(
    config: Annotated[
        Path | None,
        typer.Argument(help="Path to experiment YAML config"),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Model name or HuggingFace path"),
    ] = None,
    backend: Annotated[
        str | None,
        typer.Option("--backend", "-b", help="Inference backend (pytorch, vllm, tensorrt)"),
    ] = None,
    dataset: Annotated[
        str | None,
        typer.Option("--dataset", "-d", help="Dataset name"),
    ] = None,
    n: Annotated[
        int | None,
        typer.Option("-n", help="Number of prompts to run"),
    ] = None,
    batch_size: Annotated[
        int | None,
        typer.Option("--batch-size", help="Batch size (PyTorch backend)"),
    ] = None,
    precision: Annotated[
        str | None,
        typer.Option(
            "--precision", "-p", help="Floating point precision (fp32, fp16, bf16, int8, int4)"
        ),
    ] = None,
    output: Annotated[
        str | None,
        typer.Option("--output", "-o", help="Output directory for results"),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Validate config and estimate VRAM without running"),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Suppress progress bars"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show detailed output and tracebacks"),
    ] = False,
) -> None:
    """Run an LLM efficiency experiment."""

    # Install SIGINT handler so Ctrl-C exits with code 130
    def _handle_sigint(signum: int, frame: Any) -> None:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        _run_impl(
            config=config,
            model=model,
            backend=backend,
            dataset=dataset,
            n=n,
            batch_size=batch_size,
            precision=precision,
            output=output,
            dry_run=dry_run,
            quiet=quiet,
            verbose=verbose,
        )
    except ConfigError as e:
        print(format_error(e, verbose=verbose), file=sys.stderr)
        raise typer.Exit(code=2) from None
    except (PreFlightError, ExperimentError, BackendError) as e:
        print(format_error(e, verbose=verbose), file=sys.stderr)
        raise typer.Exit(code=1) from None
    except ValidationError as e:
        print(format_validation_error(e), file=sys.stderr)
        raise typer.Exit(code=2) from None
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130) from None


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------


def _run_impl(
    config: Path | None,
    model: str | None,
    backend: str | None,
    dataset: str | None,
    n: int | None,
    batch_size: int | None,
    precision: str | None,
    output: str | None,
    dry_run: bool,
    quiet: bool,
    verbose: bool,
) -> None:
    """Core implementation — separated for clean error handling in run()."""
    # Build CLI overrides dict — only include flags the user explicitly passed
    cli_overrides: dict[str, Any] = {}
    if model is not None:
        cli_overrides["model"] = model
    if backend is not None:
        cli_overrides["backend"] = backend
    if dataset is not None:
        cli_overrides["dataset"] = dataset
    if n is not None:
        cli_overrides["n"] = n
    if batch_size is not None:
        # Dotted key for _unflatten() in loader — maps to pytorch.batch_size
        cli_overrides["pytorch.batch_size"] = batch_size
    if precision is not None:
        cli_overrides["precision"] = precision
    if output is not None:
        cli_overrides["output_dir"] = output

    # Validate we have enough information to resolve a config
    if config is None and model is None:
        raise ConfigError(
            "Provide a config file or --model flag.\n"
            "  Examples:\n"
            "    llem run experiment.yaml\n"
            "    llem run --model gpt2 --backend pytorch"
        )

    # Load/resolve the experiment config
    experiment_config = load_experiment_config(
        path=config,
        cli_overrides=cli_overrides if cli_overrides else None,
    )

    # --- Dry-run branch ---
    if dry_run:
        vram = estimate_vram(experiment_config)
        gpu_vram_gb = get_gpu_vram_gb()
        print_dry_run(experiment_config, vram, gpu_vram_gb, verbose=verbose)
        return

    # --- Run branch ---
    print_experiment_header(experiment_config)

    with tqdm(
        total=None,
        desc="Measuring",
        file=sys.stderr,
        disable=quiet or not sys.stderr.isatty(),
    ) as pbar:
        result = run_experiment(experiment_config)
        pbar.set_description("Done")

    print_result_summary(result)

    # Save output if output_dir specified
    if experiment_config.output_dir:
        output_dir = Path(experiment_config.output_dir)
        ts_source = output_dir / result.timeseries if result.timeseries else None
        result.save(output_dir, timeseries_source=ts_source)
        # Clean up stale flat timeseries file after copy into subdirectory
        if ts_source is not None:
            ts_source.unlink(missing_ok=True)
        print(f"Saved: {experiment_config.output_dir}", file=sys.stderr)
