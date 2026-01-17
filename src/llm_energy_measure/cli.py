"""Command-line interface for llm-energy-measure.

Provides commands for:
- Running experiments
- Aggregating raw results
- Validating configurations
- Listing and inspecting results
"""

from __future__ import annotations

# Load .env file BEFORE any llm_energy_measure imports (constants reads env vars at import time)
from dotenv import load_dotenv

load_dotenv()  # Loads from .env in current directory or parents

# ruff: noqa: E402 - imports must come after load_dotenv()
import contextlib
import copy
import json
import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape as rich_escape
from rich.prompt import Confirm
from rich.table import Table

from llm_energy_measure.config.loader import (
    ConfigWarning,
    has_blocking_warnings,
    load_config,
    validate_config,
)
from llm_energy_measure.config.models import (
    DEFAULT_DATASET,
    ExperimentConfig,
    HuggingFacePromptSource,
)
from llm_energy_measure.constants import (
    COMPLETION_MARKER_PREFIX,
    GRACEFUL_SHUTDOWN_TIMEOUT_SEC,
    PRESETS,
    SCHEMA_VERSION,
)
from llm_energy_measure.core.dataset_loader import (
    list_builtin_datasets,
    load_prompts_from_file,
    load_prompts_from_source,
)
from llm_energy_measure.domain.experiment import AggregatedResult, RawProcessResult
from llm_energy_measure.exceptions import AggregationError, ConfigurationError
from llm_energy_measure.logging import setup_logging
from llm_energy_measure.results.aggregation import aggregate_results, calculate_efficiency_metrics
from llm_energy_measure.results.repository import FileSystemRepository
from llm_energy_measure.state.experiment_state import (
    ExperimentState,
    ExperimentStatus,
    ProcessProgress,
    ProcessStatus,
    StateManager,
    compute_config_hash,
)

app = typer.Typer(
    name="llm-energy-measure",
    help="LLM inference efficiency measurement framework",
    add_completion=False,
)
config_app = typer.Typer(help="Configuration management commands", invoke_without_command=True)
results_app = typer.Typer(help="Results inspection commands", invoke_without_command=True)
app.add_typer(config_app, name="config")
app.add_typer(results_app, name="results")

# Respect NO_COLOR environment variable for testing and accessibility
console = Console(no_color=os.environ.get("NO_COLOR") == "1")


# =============================================================================
# Config Display Helpers
# =============================================================================


def _format_field(
    name: str,
    value: Any,
    is_default: bool,
    nested: bool = False,
) -> tuple[str, str]:
    """Format field name and value with appropriate styling.

    Args:
        name: Field name
        value: Field value
        is_default: Whether value is the default (dim if True)
        nested: Whether this is a nested field (indented)

    Returns:
        Tuple of (formatted_name, formatted_value) for table row
    """
    indent = "  " if nested else ""
    if is_default:
        return f"[dim]{indent}{name}[/dim]", f"[dim]{value}[/dim]"
    else:
        style = "cyan" if nested else "green"
        return f"[{style}]{indent}{name}[/{style}]", str(value)


def _add_section_header(table: Table, name: str) -> None:
    """Add a bold section header row to the table."""
    table.add_row(f"[bold]{name}[/bold]", "")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge override into base, returning a new dict."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _apply_cli_overrides(
    config_dict: dict[str, Any],
    overrides: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply CLI overrides to config dict, tracking what was overridden.

    Returns:
        Tuple of (merged config dict, dict of overridden values with original values)
    """
    tracked_overrides: dict[str, Any] = {}

    for key, value in overrides.items():
        if value is None:
            continue

        # Handle nested keys like "batching_options.batch_size"
        if "." in key:
            parts = key.split(".")
            parent_key, child_key = parts[0], ".".join(parts[1:])

            if parent_key not in config_dict:
                config_dict[parent_key] = {}

            if isinstance(config_dict[parent_key], dict):
                original = config_dict[parent_key].get(child_key.split(".")[0])
                if child_key in config_dict[parent_key]:
                    original = config_dict[parent_key][child_key]
                config_dict[parent_key][child_key.split(".")[0]] = value
                tracked_overrides[key] = {"new": value, "original": original}
        else:
            original = config_dict.get(key)
            config_dict[key] = value
            tracked_overrides[key] = {"new": value, "original": original}

    return config_dict, tracked_overrides


def _print_value(name: str, value: Any, is_default: bool, indent: int = 2) -> None:
    """Print a config value with dim styling for defaults."""
    spaces = " " * indent
    if is_default:
        console.print(f"[dim]{spaces}{name}: {value}[/dim]")
    else:
        console.print(f"{spaces}[cyan]{name}[/cyan]: {value}")


def _display_config_summary(
    config: ExperimentConfig,
    overrides: dict[str, Any],
    preset_name: str | None = None,
) -> None:
    """Display config summary with override visibility.

    Shows ALL configuration parameters with visual styling:
    - Bold: Section headers
    - Cyan: Non-default values
    - Dim: Default values
    """
    console.print(f"\n[bold]Experiment: {config.config_name}[/bold]")
    if preset_name:
        console.print(f"  [cyan]Preset: {preset_name}[/cyan]")

    # Core settings
    console.print("  [bold]core:[/bold]")
    _print_value("model", config.model_name, False, indent=4)
    _print_value("adapter", config.adapter, config.adapter is None, indent=4)
    _print_value("backend", config.backend, config.backend == "pytorch", indent=4)
    _print_value("precision", config.fp_precision, config.fp_precision == "float16", indent=4)
    _print_value(
        "processes",
        f"{config.num_processes} on GPUs {config.gpu_list}",
        config.num_processes == 1,
        indent=4,
    )
    _print_value("task", config.task_type, config.task_type == "text_generation", indent=4)
    _print_value(
        "inference", config.inference_type, config.inference_type == "pure_generative", indent=4
    )
    _print_value("seed", config.random_seed, config.random_seed is None, indent=4)

    # Token/prompt settings
    console.print("  [bold]tokens:[/bold]")
    _print_value("num_prompts", config.num_input_prompts, config.num_input_prompts == 1, indent=4)
    _print_value("max_input", config.max_input_tokens, config.max_input_tokens == 512, indent=4)
    _print_value("max_output", config.max_output_tokens, config.max_output_tokens == 128, indent=4)
    _print_value("min_output", config.min_output_tokens, config.min_output_tokens == 0, indent=4)

    # Streaming settings
    console.print("  [bold]streaming:[/bold]")
    _print_value("enabled", config.streaming, config.streaming is False, indent=4)
    _print_value(
        "warmup_requests",
        config.streaming_warmup_requests,
        config.streaming_warmup_requests == 2,
        indent=4,
    )

    # Batching
    console.print("  [bold]batching:[/bold]")
    batch = config.batching_options
    _print_value("batch_size", batch.batch_size, batch.batch_size == 1, indent=4)
    _print_value("strategy", batch.strategy, batch.strategy == "static", indent=4)
    _print_value(
        "max_tokens_per_batch",
        batch.max_tokens_per_batch,
        batch.max_tokens_per_batch is None,
        indent=4,
    )

    # Sharding
    console.print("  [bold]sharding:[/bold]")
    shard = config.sharding_config
    _print_value("strategy", shard.strategy, shard.strategy == "none", indent=4)
    _print_value("num_shards", shard.num_shards, shard.num_shards == 1, indent=4)

    # Traffic/pacing settings
    console.print("  [bold]traffic:[/bold]")
    _print_value("query_rate", config.query_rate, config.query_rate == 1.0, indent=4)
    sim = config.latency_simulation
    _print_value("simulation", sim.enabled, sim.enabled is False, indent=4)
    _print_value("mode", sim.mode, sim.mode == "poisson", indent=4)
    _print_value("target_qps", sim.target_qps, sim.target_qps == 1.0, indent=4)
    _print_value("sim_seed", sim.seed, sim.seed is None, indent=4)

    # Decoder config
    console.print("  [bold]decoder:[/bold]")
    decoder = config.decoder_config
    _print_value("preset", decoder.preset, decoder.preset is None, indent=4)
    mode = "deterministic (greedy)" if decoder.is_deterministic else "sampling"
    _print_value("mode", mode, decoder.temperature == 1.0 and decoder.do_sample, indent=4)
    _print_value("temperature", decoder.temperature, decoder.temperature == 1.0, indent=4)
    _print_value("do_sample", decoder.do_sample, decoder.do_sample is True, indent=4)
    _print_value("top_p", decoder.top_p, decoder.top_p == 1.0, indent=4)
    _print_value("top_k", decoder.top_k, decoder.top_k == 50, indent=4)
    _print_value("min_p", decoder.min_p, decoder.min_p == 0.0, indent=4)
    _print_value(
        "repetition_penalty",
        decoder.repetition_penalty,
        decoder.repetition_penalty == 1.0,
        indent=4,
    )
    _print_value(
        "no_repeat_ngram_size",
        decoder.no_repeat_ngram_size,
        decoder.no_repeat_ngram_size == 0,
        indent=4,
    )

    # Quantization
    console.print("  [bold]quantization:[/bold]")
    q = config.quantization_config
    _print_value("quantization", q.quantization, q.quantization is False, indent=4)
    _print_value("load_in_4bit", q.load_in_4bit, q.load_in_4bit is False, indent=4)
    _print_value("load_in_8bit", q.load_in_8bit, q.load_in_8bit is False, indent=4)
    _print_value(
        "bnb_4bit_compute_dtype",
        q.bnb_4bit_compute_dtype,
        q.bnb_4bit_compute_dtype == "float16",
        indent=4,
    )
    _print_value(
        "bnb_4bit_quant_type", q.bnb_4bit_quant_type, q.bnb_4bit_quant_type == "nf4", indent=4
    )
    _print_value(
        "bnb_4bit_use_double_quant",
        q.bnb_4bit_use_double_quant,
        q.bnb_4bit_use_double_quant is False,
        indent=4,
    )

    # Schedule config
    console.print("  [bold]schedule:[/bold]")
    _print_value("cycles", config.num_cycles, config.num_cycles == 1, indent=4)
    sched = config.schedule_config
    _print_value("cron_enabled", sched.enabled, sched.enabled is False, indent=4)
    _print_value("interval", sched.interval, sched.interval is None, indent=4)
    _print_value("at", sched.at, sched.at is None, indent=4)
    days_str = ", ".join(sched.days) if sched.days else None
    _print_value("days", days_str, sched.days is None, indent=4)
    _print_value("total_duration", sched.total_duration, sched.total_duration == "24h", indent=4)

    # Prompt source (if configured)
    if config.prompt_source is not None:
        console.print("  [bold]prompts:[/bold]")
        ps = config.prompt_source
        _print_value("type", ps.type, False, indent=4)
        if ps.type == "file":
            _print_value("path", ps.path, False, indent=4)
        else:  # huggingface
            _print_value("dataset", ps.dataset, False, indent=4)
            _print_value("split", ps.split, ps.split == "train", indent=4)
            _print_value("subset", ps.subset, ps.subset is None, indent=4)
            _print_value("column", ps.column, ps.column is None, indent=4)
            _print_value("sample_size", ps.sample_size, ps.sample_size is None, indent=4)
            _print_value("shuffle", ps.shuffle, ps.shuffle is False, indent=4)
            _print_value("seed", ps.seed, ps.seed == 42, indent=4)

    # Show overrides
    if overrides:
        console.print("\n[dim]CLI overrides:[/dim]")
        for key, info in overrides.items():
            original = info.get("original")
            new = info.get("new")
            if original is not None and original != new:
                console.print(f"  {key}: {new} [dim](was: {original})[/dim]")
            else:
                console.print(f"  {key}: {new}")


def _update_process_state_from_markers(
    state: ExperimentState,
    state_manager: StateManager,
    results_dir: Path,
) -> ExperimentState:
    """Update experiment state by scanning completion markers.

    Args:
        state: Current experiment state.
        state_manager: State manager for persistence.
        results_dir: Base results directory.

    Returns:
        Updated experiment state.
    """
    raw_dir = results_dir / "raw" / state.experiment_id

    for i in range(state.num_processes):
        marker_path = raw_dir / f"{COMPLETION_MARKER_PREFIX}{i}"
        if marker_path.exists():
            try:
                marker_data = json.loads(marker_path.read_text())
                state.process_progress[i] = ProcessProgress(
                    process_index=i,
                    status=ProcessStatus.COMPLETED,
                    gpu_id=marker_data.get("gpu_id"),
                    completed_at=datetime.fromisoformat(marker_data["timestamp"]),
                )
            except Exception:
                # Marker exists but couldn't parse - still mark as completed
                state.process_progress[i] = ProcessProgress(
                    process_index=i,
                    status=ProcessStatus.COMPLETED,
                )
        elif i not in state.process_progress:
            # No marker and not previously tracked = failed or didn't start
            state.process_progress[i] = ProcessProgress(
                process_index=i,
                status=ProcessStatus.FAILED,
                error_message="No completion marker found",
            )

    state_manager.save(state)
    return state


def _display_incomplete_experiment(state: ExperimentState) -> None:
    """Display information about an incomplete experiment."""
    console.print("\n[yellow]Incomplete experiment detected:[/yellow]")
    console.print(f"  Experiment ID: [cyan]{state.experiment_id}[/cyan]")
    if state.config_path:
        console.print(f"  Config: {state.config_path}")
    if state.model_name:
        console.print(f"  Model: {state.model_name}")
    if state.prompt_args:
        dataset = state.prompt_args.get("dataset", "")
        sample_size = state.prompt_args.get("sample_size", "")
        if dataset:
            console.print(f"  Dataset: {dataset}" + (f" (n={sample_size})" if sample_size else ""))
    if state.started_at:
        console.print(f"  Started: {state.started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    console.print(f"  Status: {state.status.value}")
    console.print(
        f"  Progress: {state.processes_completed}/{state.num_processes} processes completed"
    )


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        console.print(f"llm-energy-measure v{SCHEMA_VERSION}")
        raise typer.Exit()


@app.callback()  # type: ignore[misc]
def main(
    version: Annotated[
        bool,
        typer.Option("--version", "-v", callback=version_callback, is_eager=True),
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", help="Enable debug logging")] = False,
) -> None:
    """LLM inference efficiency measurement framework."""
    setup_logging(level="DEBUG" if verbose else "INFO")


@app.command()  # type: ignore[misc]
def run(
    config_path: Annotated[Path, typer.Argument(help="Path to experiment config file")],
    prompts_file: Annotated[
        Path | None, typer.Option("--prompts", "-p", help="Path to prompts file (one per line)")
    ] = None,
    dataset: Annotated[
        str | None,
        typer.Option("--dataset", "-d", help="HuggingFace dataset (default: ai-energy-score)"),
    ] = None,
    dataset_split: Annotated[str, typer.Option("--split", help="Dataset split")] = "train",
    dataset_column: Annotated[
        str | None, typer.Option("--column", help="Dataset column for prompts")
    ] = None,
    sample_size: Annotated[
        int | None, typer.Option("--sample-size", "-n", help="Limit number of prompts")
    ] = None,
    results_dir: Annotated[
        Path | None, typer.Option("--results-dir", "-o", help="Results output directory")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Validate config without running experiment")
    ] = False,
) -> None:
    """Run an LLM efficiency experiment.

    Loads the configuration, runs inference, measures energy consumption,
    and saves raw per-process results. Use 'aggregate' command to combine
    results from multi-GPU experiments.

    Prompts can be specified via:
    - --prompts <file.txt>: One prompt per line
    - --dataset <name>: HuggingFace dataset or alias (default: ai-energy-score)
    - prompt_source in config file
    """
    try:
        # Load and validate config
        config = load_config(config_path)
        warnings = validate_config(config)

        for warning in warnings:
            console.print(f"[yellow]Warning:[/yellow] {warning}")

        console.print(f"[green]✓[/green] Config loaded: {config.config_name}")
        console.print(f"  Model: {config.model_name}")
        console.print(f"  Processes: {config.num_processes}")
        console.print(f"  GPUs: {config.gpu_list}")

        if dry_run:
            console.print("[blue]Dry run - skipping experiment execution[/blue]")
            raise typer.Exit()

        # Resolve prompts (CLI > config > default)
        prompts = _resolve_prompts(
            config=config,
            prompts_file=prompts_file,
            dataset=dataset,
            dataset_split=dataset_split,
            dataset_column=dataset_column,
            sample_size=sample_size,
        )
        console.print(f"  Prompts: {len(prompts)}")

        # Initialize repository (will be used when full experiment wiring is complete)
        _ = FileSystemRepository(results_dir)

        # For now, this requires the full experiment infrastructure to be wired up
        # The actual run would use:
        # - ExperimentOrchestrator with DI components
        # - accelerate launch for multi-GPU
        # This is a placeholder that shows the intended interface
        console.print("\n[bold]Experiment execution requires accelerate launch.[/bold]")
        console.print("Use: accelerate launch --num_processes N <script> --config <config>")
        console.print("\nFor programmatic execution, see orchestration/launcher.py")

    except typer.Exit:
        raise  # Re-raise typer exits (including dry run)
    except ConfigurationError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None


def _resolve_prompts(
    config: ExperimentConfig,
    prompts_file: Path | None,
    dataset: str | None,
    dataset_split: str,
    dataset_column: str | None,
    sample_size: int | None,
) -> list[str]:
    """Resolve prompts from CLI args or config.

    Priority: CLI --dataset > CLI --prompts > config.prompt_source > default
    """
    # CLI --dataset takes highest priority
    if dataset:
        source = HuggingFacePromptSource(
            dataset=dataset,
            split=dataset_split,
            column=dataset_column,
            sample_size=sample_size,
        )
        return load_prompts_from_source(source)

    # CLI --prompts file
    if prompts_file and prompts_file.exists():
        prompts = load_prompts_from_file(prompts_file)
        if sample_size:
            prompts = prompts[:sample_size]
        return prompts

    # Config-defined prompt source
    if config.prompt_source:
        prompts = load_prompts_from_source(config.prompt_source)
        if sample_size:  # CLI sample_size can further limit
            prompts = prompts[:sample_size]
        return prompts

    # Default: use AI Energy Score dataset (standardised benchmark)
    console.print(f"[dim]Using default dataset: {DEFAULT_DATASET}[/dim]")
    source = HuggingFacePromptSource(
        dataset=DEFAULT_DATASET,
        sample_size=sample_size,
    )
    return load_prompts_from_source(source)


@app.command("experiment")  # type: ignore[misc]
def experiment(
    config_path: Annotated[
        Path | None, typer.Argument(help="Path to experiment config file (optional with --preset)")
    ] = None,
    # Prompt source options
    dataset: Annotated[
        str | None,
        typer.Option("--dataset", "-d", help="HuggingFace dataset (default: ai-energy-score)"),
    ] = None,
    prompts_file: Annotated[
        Path | None, typer.Option("--prompts", "-p", help="Path to prompts file")
    ] = None,
    sample_size: Annotated[
        int | None, typer.Option("--sample-size", "-n", help="Limit number of prompts")
    ] = None,
    dataset_split: Annotated[str, typer.Option("--split", help="Dataset split")] = "train",
    dataset_column: Annotated[
        str | None, typer.Option("--column", help="Dataset column for prompts")
    ] = None,
    # Preset and model for config-free mode
    preset: Annotated[
        str | None,
        typer.Option("--preset", help=f"Built-in preset ({', '.join(PRESETS.keys())})"),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="HuggingFace model name (required with --preset)"),
    ] = None,
    # Workflow parameters (recommended)
    max_tokens: Annotated[
        int | None,
        typer.Option("--max-tokens", help="Override max_output_tokens"),
    ] = None,
    seed: Annotated[
        int | None,
        typer.Option("--seed", help="Random seed for reproducibility"),
    ] = None,
    backend: Annotated[
        str | None,
        typer.Option("--backend", help="Inference backend (pytorch, vllm)"),
    ] = None,
    cycles: Annotated[
        int | None,
        typer.Option("--cycles", "-c", help="Number of cycles for statistical robustness (1-10)"),
    ] = None,
    # Streaming latency measurement
    streaming: Annotated[
        bool | None,
        typer.Option(
            "--streaming/--no-streaming",
            help="Enable streaming for TTFT/ITL latency measurement",
        ),
    ] = None,
    streaming_warmup: Annotated[
        int | None,
        typer.Option(
            "--streaming-warmup", help="Number of warmup requests for streaming (default: 3)"
        ),
    ] = None,
    # DEPRECATED: Testable params - use YAML config for formal experiments
    batch_size: Annotated[
        int | None,
        typer.Option(
            "--batch-size", "-b", help="[Deprecated] Use YAML config for formal experiments"
        ),
    ] = None,
    precision: Annotated[
        str | None,
        typer.Option("--precision", help="[Deprecated] Use YAML config for formal experiments"),
    ] = None,
    num_processes: Annotated[
        int | None,
        typer.Option("--num-processes", help="[Deprecated] Use YAML config for formal experiments"),
    ] = None,
    gpu_list: Annotated[
        str | None,
        typer.Option("--gpu-list", help="[Deprecated] Use YAML config for formal experiments"),
    ] = None,
    temperature: Annotated[
        float | None,
        typer.Option("--temperature", help="[Deprecated] Use YAML config for formal experiments"),
    ] = None,
    quantization: Annotated[
        bool | None,
        typer.Option("--quantization/--no-quantization", help="[Deprecated] Use YAML config"),
    ] = None,
    # Workflow control options (Phases 1-5)
    no_aggregate: Annotated[
        bool, typer.Option("--no-aggregate", help="Skip auto-aggregation after experiment")
    ] = False,
    fresh: Annotated[
        bool, typer.Option("--fresh", help="Start fresh, ignore incomplete experiments")
    ] = False,
    resume: Annotated[str | None, typer.Option("--resume", help="Resume experiment by ID")] = None,
    results_dir: Annotated[
        Path | None, typer.Option("--results-dir", "-o", help="Results directory")
    ] = None,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Skip confirmation prompts (auto-accept warnings)")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show config and exit without running experiment")
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Run despite blocking config errors")
    ] = False,
) -> None:
    """Run experiment with automatic accelerate handling and auto-aggregation.

    Supports three modes:

    1. Config file (formal experiments):
       llm-energy-measure experiment config.yaml --dataset alpaca -n 100

    2. Preset + model (quick exploration):
       llm-energy-measure experiment --preset quick-test --model TinyLlama/... -d alpaca

    3. Config + CLI overrides (sweeps):
       llm-energy-measure experiment config.yaml -b 8 --precision int8

    Multi-cycle mode for statistical robustness (academic benchmarking standard):
       llm-energy-measure experiment config.yaml --cycles 5 --dataset alpaca -n 100

    Precedence: CLI > Config file > Preset > Defaults

    By default, results are auto-aggregated on success. Use --no-aggregate to skip.
    Interrupted experiments can be resumed with --resume <exp_id>.
    """
    import tempfile

    import yaml

    state_manager = StateManager()
    repo = FileSystemRepository(results_dir)
    actual_results_dir = repo._base  # Get resolved results dir

    # Track subprocess for signal handling
    subprocess_handle: subprocess.Popen[bytes] | None = None
    current_state: ExperimentState | None = None
    interrupt_in_progress = False

    def _handle_interrupt(signum: int, frame: Any) -> None:
        """Handle SIGINT/SIGTERM gracefully.

        Uses process groups to ensure all child processes (accelerate workers)
        are terminated, not just the parent process.
        """
        nonlocal subprocess_handle, current_state, interrupt_in_progress

        # Prevent re-entry from multiple Ctrl+C presses
        if interrupt_in_progress:
            console.print("[dim]Already shutting down, please wait...[/dim]")
            return
        interrupt_in_progress = True

        console.print("\n[yellow]Interrupt received, shutting down...[/yellow]")

        if subprocess_handle and subprocess_handle.poll() is None:
            # Get the process group ID (same as PID when start_new_session=True)
            pgid = os.getpgid(subprocess_handle.pid)

            # Send SIGTERM to entire process group first
            console.print(
                f"[dim]Waiting up to {GRACEFUL_SHUTDOWN_TIMEOUT_SEC}s for subprocess group...[/dim]"
            )
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGTERM)

            # Wait in 1-second increments for responsiveness
            for i in range(GRACEFUL_SHUTDOWN_TIMEOUT_SEC):
                try:
                    subprocess_handle.wait(timeout=1)
                    break  # Subprocess exited
                except subprocess.TimeoutExpired:
                    if i < GRACEFUL_SHUTDOWN_TIMEOUT_SEC - 1:
                        console.print(f"[dim]...{GRACEFUL_SHUTDOWN_TIMEOUT_SEC - i - 1}s[/dim]")
            else:
                # Timeout expired, force kill entire process group
                console.print("[red]Timeout, sending SIGKILL to process group...[/red]")
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(pgid, signal.SIGKILL)

                # Brief wait to let kernel reap the process, but don't block indefinitely
                # SIGKILL cannot be caught, so the process will die eventually
                with contextlib.suppress(subprocess.TimeoutExpired):
                    subprocess_handle.wait(timeout=2)

        # Update state to INTERRUPTED
        if current_state:
            current_state.status = ExperimentStatus.INTERRUPTED
            current_state.error_message = "Interrupted by user (SIGINT/SIGTERM)"
            state_manager.save(current_state)
            console.print("\n[yellow]Experiment interrupted. Resume with:[/yellow]")
            console.print(f"  llm-energy-measure experiment --resume {current_state.experiment_id}")

        raise typer.Exit(130)  # Standard exit code for SIGINT

    try:
        # Clean up stale running states
        stale = state_manager.cleanup_stale()
        if stale:
            console.print(f"[dim]Cleaned up {len(stale)} stale experiment states[/dim]")

        # Handle --resume flag (Phase 2)
        existing_state: ExperimentState | None = None
        if resume:
            existing_state = state_manager.load(resume)
            if not existing_state:
                console.print(f"[red]Error:[/red] No state found for experiment {resume}")
                raise typer.Exit(1)
            if existing_state.status == ExperimentStatus.AGGREGATED:
                console.print(
                    f"[green]Experiment {resume} already completed and aggregated[/green]"
                )
                raise typer.Exit(0)

            # Restore config and prompt args from saved state
            _display_incomplete_experiment(existing_state)
            console.print(f"[green]Resuming experiment {resume}[/green]\n")

            if existing_state.config_path:
                config_path = Path(existing_state.config_path)
            if existing_state.prompt_args:
                # Restore prompt source args
                if "dataset" in existing_state.prompt_args and not dataset:
                    dataset = existing_state.prompt_args["dataset"]
                if "sample_size" in existing_state.prompt_args and sample_size is None:
                    sample_size = existing_state.prompt_args["sample_size"]

        # Validate input modes
        if not config_path and not preset and not resume:
            console.print("[red]Error:[/red] Provide config file, --preset, or --resume")
            raise typer.Exit(1)

        if preset and not config_path and not model:
            console.print(
                "[red]Error:[/red] --model is required when using --preset without config"
            )
            raise typer.Exit(1)

        if preset and preset not in PRESETS:
            console.print(f"[red]Error:[/red] Unknown preset '{preset}'")
            console.print(f"Available presets: {', '.join(PRESETS.keys())}")
            raise typer.Exit(1)

        # Build config from sources (precedence: CLI > config > preset > defaults)
        config_dict: dict[str, Any] = {}
        preset_name: str | None = None

        # 1. Start with preset if provided
        if preset:
            preset_name = preset
            config_dict = copy.deepcopy(PRESETS[preset])
            config_dict["config_name"] = f"preset-{preset}"

        # 2. Merge config file on top (config overrides preset)
        if config_path:
            file_config = load_config(config_path)
            file_dict = file_config.model_dump()
            config_dict = _deep_merge(config_dict, file_dict)

        # 3. Apply model if specified (required for preset-only mode)
        if model:
            config_dict["model_name"] = model

        # Validate we have a model
        if "model_name" not in config_dict or not config_dict["model_name"]:
            console.print("[red]Error:[/red] model_name is required (from config or --model)")
            raise typer.Exit(1)

        # 4. Prepare CLI overrides and emit deprecation warnings for testable params
        # Philosophy: CLI = workflow/meta params, YAML = testable experiment params
        deprecated_flags: list[str] = []
        cli_overrides_dict: dict[str, Any] = {}

        # Workflow params (recommended) - no deprecation warning
        if max_tokens is not None:
            cli_overrides_dict["max_output_tokens"] = max_tokens
        if seed is not None:
            cli_overrides_dict["random_seed"] = seed
        if backend is not None:
            cli_overrides_dict["backend"] = backend
        if streaming is not None:
            cli_overrides_dict["streaming"] = streaming
        if streaming_warmup is not None:
            cli_overrides_dict["streaming_warmup_requests"] = streaming_warmup

        # Deprecated testable params - warn but still apply
        if batch_size is not None:
            deprecated_flags.append("--batch-size")
            cli_overrides_dict["batching_options.batch_size"] = batch_size
        if precision is not None:
            deprecated_flags.append("--precision")
            cli_overrides_dict["fp_precision"] = precision
        if num_processes is not None:
            deprecated_flags.append("--num-processes")
            cli_overrides_dict["num_processes"] = num_processes
        if gpu_list is not None:
            deprecated_flags.append("--gpu-list")
            cli_overrides_dict["gpu_list"] = [int(g.strip()) for g in gpu_list.split(",")]
        if temperature is not None:
            deprecated_flags.append("--temperature")
            cli_overrides_dict["decoder_config.temperature"] = temperature
        if quantization is not None:
            deprecated_flags.append("--quantization")
            cli_overrides_dict["quantization_config.quantization"] = quantization

        # Emit deprecation warning if any deprecated flags used
        if deprecated_flags:
            console.print(
                f"[yellow]Deprecation warning:[/yellow] {', '.join(deprecated_flags)} "
                "are testable params - use YAML config for formal experiments"
            )

        # 5. Apply CLI overrides and track them
        config_dict, tracked_overrides = _apply_cli_overrides(config_dict, cli_overrides_dict)

        # 6. Create final config
        config = ExperimentConfig(**config_dict)
        final_num_processes = config.num_processes

        # 6b. Resolve results_dir with proper precedence:
        #     CLI --results-dir > config io.results_dir > .env > default
        if results_dir is None and config.io_config.results_dir:
            # Config specifies results_dir and CLI didn't override
            repo = FileSystemRepository(Path(config.io_config.results_dir))
            actual_results_dir = repo._base

        # 7. Validate config and handle warnings
        config_warnings = validate_config(config)
        if config_warnings:
            # Display warnings with severity-appropriate styling
            console.print("\n[yellow]Configuration warnings:[/yellow]")
            for warning in config_warnings:
                severity_styles = {"error": "red", "warning": "yellow", "info": "dim"}
                style = severity_styles.get(warning.severity, "yellow")
                console.print(f"  [{style}]{rich_escape(str(warning))}[/{style}]")

            # Check for blocking errors (error severity)
            if has_blocking_warnings(config_warnings):
                if not force:
                    console.print(
                        "\n[red]Blocking config errors detected. Use --force to run anyway.[/red]"
                    )
                    raise typer.Exit(1)
                else:
                    console.print("\n[yellow]--force: Running despite blocking errors[/yellow]")

            # Handle non-blocking warnings with --yes / interactive prompt
            elif not yes:
                # Check if we're in interactive mode
                if not sys.stdin.isatty():
                    # Non-interactive mode without --yes: fail
                    console.print("\n[red]Config warnings detected in non-interactive mode.[/red]")
                    console.print("[yellow]Use --yes to proceed anyway.[/yellow]")
                    raise typer.Exit(1)

                # Interactive mode: prompt user
                if not Confirm.ask("\nContinue with this configuration?", default=True):
                    console.print("[dim]Aborted by user[/dim]")
                    raise typer.Abort()

        # Convert warnings to strings for embedding in results
        warning_strings = [w.to_result_string() for w in config_warnings] if config_warnings else []

        # Compute config hash for incomplete experiment detection (Phase 4)
        config_hash = compute_config_hash(config_dict)

        # Check for incomplete experiments with same config (Phase 4)
        if not fresh and not resume:
            existing = state_manager.find_by_config_hash(config_hash)
            if existing and existing.status != ExperimentStatus.AGGREGATED:
                _display_incomplete_experiment(existing)
                if Confirm.ask("\nResume this experiment?", default=True):
                    resume = existing.experiment_id
                    console.print(f"[green]Resuming experiment {resume}[/green]")
                    # TODO: Actual resume logic
                else:
                    console.print("[yellow]Starting fresh experiment[/yellow]")

        # Display config with override visibility
        _display_config_summary(config, tracked_overrides, preset_name)

        # Resolve cycles: CLI > config.num_cycles > default (1)
        effective_cycles = cycles if cycles is not None else config.num_cycles
        if effective_cycles < 1 or effective_cycles > 10:
            console.print("[red]Error:[/red] cycles must be between 1 and 10")
            raise typer.Exit(1)
        if effective_cycles > 1:
            source = "CLI" if cycles is not None else "config"
            console.print(f"  [cyan]Multi-cycle mode:[/cyan] {effective_cycles} cycles ({source})")

        # Check for MIG instances and warn about energy measurement (Phase: MIG support)
        from llm_energy_measure.core.gpu_info import detect_gpu_topology, validate_gpu_selection

        topology = detect_gpu_topology()
        mig_warnings = validate_gpu_selection(config.gpu_list, topology)
        for mig_warning in mig_warnings:
            console.print(f"[yellow]Warning:[/yellow] {mig_warning}")

        # Dry-run: show config and exit without running
        if dry_run:
            console.print("\n[cyan]--dry-run: Exiting without running experiment[/cyan]")
            raise typer.Exit(0)

        # Build launch command based on backend
        # vLLM manages its own multiprocessing (spawn), incompatible with accelerate (fork)
        backend_name = config.backend if hasattr(config, "backend") else "pytorch"
        subprocess_env: dict[str, str] | None = None  # Environment for subprocess

        if backend_name == "vllm":
            # Direct launch for vLLM - it handles its own distribution
            cmd = [
                sys.executable,
                "-m",
                "llm_energy_measure.orchestration.launcher",
            ]
            # vLLM v1 multiprocessing can have issues with CUDA initialization
            # Disable V1 multiprocessing to use simpler process model
            subprocess_env = os.environ.copy()
            subprocess_env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
            subprocess_env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            # Disable torch.compile - requires C compiler not present in minimal Docker images
            subprocess_env["TORCH_COMPILE_DISABLE"] = "1"
            # Set CUDA_VISIBLE_DEVICES to the configured GPU list
            # This ensures vLLM only sees the intended GPUs
            subprocess_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in config.gpu_list)
        else:
            # Accelerate launch for PyTorch backend
            cmd = [
                sys.executable,
                "-m",
                "accelerate.commands.launch",
                "--num_processes",
                str(final_num_processes),
                "-m",
                "llm_energy_measure.orchestration.launcher",
            ]

        # Generate experiment ID early so CLI and subprocess use the same ID
        # If resuming, reuse the existing experiment ID; otherwise generate new one
        if resume and existing_state:
            experiment_id = existing_state.experiment_id
        else:
            from llm_energy_measure.core.distributed import get_persistent_unique_id

            experiment_id = get_persistent_unique_id()

        # Generate temp config file with _metadata (Phase 0)
        # Always generate temp file to include _metadata for effective_config/cli_overrides tracking
        effective_config = config.model_dump()
        metadata: dict[str, dict[str, Any]] = {
            "_metadata": {
                "experiment_id": experiment_id,  # Pass to subprocess
                "effective_config": effective_config,
                "cli_overrides": tracked_overrides,
                "original_config_path": str(config_path) if config_path else None,
                "cycle_id": None,  # Set for multi-cycle experiments
                "config_warnings": warning_strings,  # Embed in results for traceability
                "results_dir": str(actual_results_dir),  # Pass custom results dir to subprocess
            }
        }
        config_with_metadata = {**effective_config, **metadata}

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, prefix="llm_energy_"
        ) as tmp:
            yaml.dump(config_with_metadata, tmp, default_flow_style=False)
            tmp_config_path = tmp.name

        cmd.extend(["--config", tmp_config_path])
        console.print(f"[dim]Generated config with metadata: {tmp_config_path}[/dim]")

        # Add prompt source args
        if dataset:
            cmd.extend(["--dataset", dataset])
            if dataset_split != "train":
                cmd.extend(["--split", dataset_split])
            if dataset_column:
                cmd.extend(["--column", dataset_column])
        elif prompts_file:
            cmd.extend(["--prompts", str(prompts_file.resolve())])

        if sample_size:
            cmd.extend(["--sample-size", str(sample_size)])

        if backend_name == "vllm":
            console.print(
                "\n[dim]$ python -m llm_energy_measure.orchestration.launcher ...[/dim]\n"
            )
        else:
            console.print(
                f"\n[dim]$ accelerate launch --num_processes {final_num_processes} ...[/dim]\n"
            )

        # Multi-cycle support: run experiment multiple times for statistical robustness
        cycle_results: list[AggregatedResult] = []
        cycle_metadata_list: list[Any] = []
        base_experiment_id = experiment_id

        for cycle_idx in range(effective_cycles):
            # Generate cycle-specific experiment ID
            if effective_cycles > 1:
                cycle_experiment_id = f"{base_experiment_id}_c{cycle_idx}"
                console.print(
                    f"\n[bold cyan]━━━ Cycle {cycle_idx + 1}/{effective_cycles} ━━━[/bold cyan]"
                )
            else:
                cycle_experiment_id = experiment_id

            # Collect cycle metadata (temperature, load) before each cycle
            from llm_energy_measure.results.cycle_statistics import (
                create_cycle_metadata,
                try_get_gpu_temperature,
                try_get_system_load,
            )

            cycle_meta = create_cycle_metadata(
                cycle_id=cycle_idx,
                timestamp=datetime.now(),
                gpu_temperature_c=try_get_gpu_temperature(),
                system_load=try_get_system_load(),
            )
            cycle_metadata_list.append(cycle_meta)

            # Update temp config with cycle-specific experiment ID
            metadata["_metadata"]["experiment_id"] = cycle_experiment_id
            if effective_cycles > 1:
                metadata["_metadata"]["cycle_id"] = cycle_idx
            config_with_metadata = {**effective_config, **metadata}

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False, prefix="llm_energy_"
            ) as tmp:
                yaml.dump(config_with_metadata, tmp, default_flow_style=False)
                cycle_tmp_config_path = tmp.name

            # Update command with cycle-specific config
            cycle_cmd = cmd.copy()
            # Replace config path in command
            config_idx = cycle_cmd.index("--config")
            cycle_cmd[config_idx + 1] = cycle_tmp_config_path

            # Create experiment state before launch (Phase 2)
            current_state = ExperimentState(
                experiment_id=cycle_experiment_id,
                status=ExperimentStatus.RUNNING,
                config_path=str(config_path) if config_path else None,
                config_hash=config_hash,
                model_name=config.model_name,
                prompt_args={
                    "dataset": dataset,
                    "sample_size": sample_size,
                    "prompts_file": str(prompts_file) if prompts_file else None,
                },
                num_processes=final_num_processes,
                started_at=datetime.now(),
            )

            # Register signal handlers (Phase 3)
            original_sigint = signal.signal(signal.SIGINT, _handle_interrupt)
            original_sigterm = signal.signal(signal.SIGTERM, _handle_interrupt)

            try:
                # Run with Popen for better control (Phase 3)
                # start_new_session=True creates a new process group, allowing us to
                # kill all child processes (accelerate workers) with os.killpg()
                subprocess_handle = subprocess.Popen(
                    cycle_cmd, start_new_session=True, env=subprocess_env
                )
                current_state.subprocess_pid = subprocess_handle.pid
                state_manager.save(current_state)

                # Wait for completion
                exit_code = subprocess_handle.wait()

            finally:
                # Restore signal handlers
                signal.signal(signal.SIGINT, original_sigint)
                signal.signal(signal.SIGTERM, original_sigterm)

            # Handle subprocess result
            if exit_code == 0:
                # Update process state from completion markers (Phase 5)
                current_state = _update_process_state_from_markers(
                    current_state, state_manager, actual_results_dir
                )

                if current_state.can_aggregate():
                    current_state.status = ExperimentStatus.COMPLETED
                    state_manager.save(current_state)

                    # Auto-aggregate if not disabled (Phase 1)
                    if not no_aggregate:
                        try:
                            console.print("[dim]Auto-aggregating cycle results...[/dim]")
                            raw_results = repo.load_all_raw(current_state.experiment_id)
                            agg_result = aggregate_results(
                                experiment_id=current_state.experiment_id,
                                raw_results=raw_results,
                                expected_processes=final_num_processes,
                                results_dir=actual_results_dir,
                                strict=True,
                            )
                            result_path = repo.save_aggregated(agg_result)
                            current_state.status = ExperimentStatus.AGGREGATED
                            state_manager.delete(current_state.experiment_id)
                            cycle_results.append(agg_result)
                            if effective_cycles == 1:
                                console.print(f"\n[green]Results saved to:[/green] {result_path}")
                                # Show brief summary including streaming latency if present
                                if agg_result.latency_stats is not None:
                                    lat = agg_result.latency_stats
                                    ttft_p99 = (
                                        lat.get("ttft_p99_ms")
                                        if isinstance(lat, dict)
                                        else lat.ttft_p99_ms
                                    )
                                    itl_p99 = (
                                        lat.get("itl_p99_ms")
                                        if isinstance(lat, dict)
                                        else lat.itl_p99_ms
                                    )
                                    console.print(
                                        f"  [dim]Streaming: TTFT p99={ttft_p99:.1f}ms"
                                        + (f"  ITL p99={itl_p99:.1f}ms" if itl_p99 else "")
                                        + "[/dim]"
                                    )
                        except AggregationError as e:
                            console.print(f"[yellow]Aggregation warning:[/yellow] {e}")
                            console.print(
                                f"Raw results available. Aggregate manually with: "
                                f"llm-energy-measure aggregate {current_state.experiment_id}"
                            )
                            state_manager.save(current_state)
                            if effective_cycles > 1:
                                console.print("[red]Cycle failed, stopping multi-cycle run[/red]")
                                raise typer.Exit(1) from None
                    else:
                        console.print(
                            f"[green]Cycle {cycle_idx + 1} complete.[/green] "
                            f"Aggregate with: llm-energy-measure aggregate {current_state.experiment_id}"
                        )
                        state_manager.save(current_state)
                else:
                    # Exit 0 but incomplete - anomaly
                    console.print(
                        "[yellow]Warning: Process exited successfully but results incomplete[/yellow]"
                    )
                    console.print(
                        f"  Completed: {current_state.processes_completed}/{current_state.num_processes}"
                    )
                    if effective_cycles > 1:
                        console.print("[red]Cycle incomplete, stopping multi-cycle run[/red]")
                        raise typer.Exit(1)
                    console.print(
                        f"  Use 'llm-energy-measure aggregate {current_state.experiment_id} --force' "
                        "to aggregate partial results"
                    )
                    state_manager.save(current_state)
                    raise typer.Exit(0)
            else:
                # Non-zero exit - check what we got
                current_state = _update_process_state_from_markers(
                    current_state, state_manager, actual_results_dir
                )
                current_state.status = ExperimentStatus.FAILED
                current_state.error_message = f"Subprocess exited with code {exit_code}"
                state_manager.save(current_state)

                console.print(f"[red]Experiment failed (exit code {exit_code})[/red]")
                console.print(
                    f"  Completed: {current_state.processes_completed}/{current_state.num_processes}"
                )
                if current_state.processes_completed > 0:
                    console.print(
                        f"  Partial results available. Resume with: "
                        f"llm-energy-measure experiment --resume {current_state.experiment_id}"
                    )

                raise typer.Exit(exit_code)

        # Multi-cycle aggregation and statistics
        if effective_cycles > 1 and len(cycle_results) == effective_cycles:
            from llm_energy_measure.results.cycle_statistics import create_multi_cycle_result

            console.print("\n[bold cyan]━━━ Multi-Cycle Statistics ━━━[/bold cyan]")

            multi_cycle_result = create_multi_cycle_result(
                experiment_id=base_experiment_id,
                cycle_results=cycle_results,
                cycle_metadata=cycle_metadata_list,
                effective_config=effective_config,
            )

            # Save multi-cycle result
            multi_cycle_path = actual_results_dir / "multi_cycle" / f"{base_experiment_id}.json"
            multi_cycle_path.parent.mkdir(parents=True, exist_ok=True)
            multi_cycle_path.write_text(multi_cycle_result.model_dump_json(indent=2))

            # Display statistics
            stats = multi_cycle_result.statistics
            console.print(f"\n[green]✓ Completed {effective_cycles} cycles[/green]")
            console.print("\n[bold]Energy:[/bold]")
            console.print(f"  Mean: {stats.energy_mean_j:.2f} J ± {stats.energy_std_j:.2f}")
            console.print(
                f"  95% CI: [{stats.energy_ci_95_lower:.2f}, {stats.energy_ci_95_upper:.2f}] J"
            )
            console.print(f"  CV: {stats.energy_cv:.1%}")

            console.print("\n[bold]Throughput:[/bold]")
            console.print(
                f"  Mean: {stats.throughput_mean_tps:.2f} tok/s ± {stats.throughput_std_tps:.2f}"
            )
            console.print(
                f"  95% CI: [{stats.throughput_ci_95_lower:.2f}, {stats.throughput_ci_95_upper:.2f}] tok/s"
            )
            console.print(f"  CV: {stats.throughput_cv:.1%}")

            console.print("\n[bold]Efficiency:[/bold]")
            console.print(
                f"  Mean: {stats.efficiency_mean_tpj:.2f} tok/J ± {stats.efficiency_std_tpj:.2f}"
            )

            console.print(f"\n[green]Multi-cycle results saved to:[/green] {multi_cycle_path}")

        raise typer.Exit(0)

    except typer.Exit:
        raise  # Re-raise typer exits
    except ConfigurationError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        raise typer.Exit(1) from None
    except FileNotFoundError:
        console.print(
            "[red]Error:[/red] accelerate not found. Install with: pip install accelerate"
        )
        raise typer.Exit(1) from None


@app.command("datasets")  # type: ignore[misc]
def list_datasets_cmd() -> None:
    """List built-in dataset aliases for prompts.

    These aliases can be used with --dataset option or in config files.
    """
    table = Table(title="Built-in Dataset Aliases")
    table.add_column("Alias", style="cyan")
    table.add_column("HuggingFace Path", style="green")
    table.add_column("Column")
    table.add_column("Description")

    descriptions = {
        "alpaca": "Instruction-following prompts (52k)",
        "sharegpt": "Real user conversations",
        "gsm8k": "Primary school maths reasoning",
        "mmlu": "Multi-task knowledge questions",
    }

    for alias, info in list_builtin_datasets().items():
        table.add_row(
            alias,
            info["path"],
            info.get("column", "auto"),
            descriptions.get(alias, ""),
        )

    console.print(table)
    console.print("\n[dim]Usage: llm-energy-measure run config.yaml --dataset alpaca -n 1000[/dim]")


@app.command("presets")  # type: ignore[misc]
def list_presets_cmd() -> None:
    """List built-in experiment presets.

    Presets provide sensible defaults for common experiment scenarios.
    Use with: llm-energy-measure experiment --preset <name> --model <model>
    """
    table = Table(title="Built-in Presets")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Key Settings")

    preset_descriptions = {
        "quick-test": "Fast validation runs",
        "benchmark": "Formal benchmark measurements",
        "throughput": "Throughput-optimised testing",
    }

    for name, config in PRESETS.items():
        settings = []
        if "max_input_tokens" in config:
            settings.append(f"max_in={config['max_input_tokens']}")
        if "max_output_tokens" in config:
            settings.append(f"max_out={config['max_output_tokens']}")
        if "batching_options" in config:
            settings.append(f"batch={config['batching_options'].get('batch_size', 1)}")
        if "fp_precision" in config:
            settings.append(f"precision={config['fp_precision']}")

        table.add_row(
            name,
            preset_descriptions.get(name, ""),
            ", ".join(settings),
        )

    console.print(table)
    console.print(
        "\n[dim]Usage: llm-energy-measure experiment --preset quick-test --model <model> -d alpaca[/dim]"
    )


@app.command("gpus")  # type: ignore[misc]
def list_gpus_cmd() -> None:
    """Show GPU topology including MIG instances.

    Displays all visible CUDA devices with their configuration,
    including Multi-Instance GPU (MIG) partitions if present.
    """
    from llm_energy_measure.core.gpu_info import (
        detect_gpu_topology,
        format_gpu_topology,
    )

    topology = detect_gpu_topology()

    if not topology.devices:
        console.print("[yellow]No CUDA devices detected[/yellow]")
        console.print("\nPossible causes:")
        console.print("  - No NVIDIA GPU installed")
        console.print("  - CUDA drivers not installed")
        console.print("  - CUDA_VISIBLE_DEVICES set to empty")
        raise typer.Exit(1)

    console.print(format_gpu_topology(topology))

    # Show usage hint
    if len(topology.devices) > 1:
        indices = ",".join(str(d.index) for d in topology.devices[:2])
        console.print(
            f"\n[dim]Use --gpu-list to select devices: llm-energy-measure experiment config.yaml --gpu-list {indices}[/dim]"
        )


@app.command()  # type: ignore[misc]
def aggregate(
    experiment_id: Annotated[str | None, typer.Argument(help="Experiment ID to aggregate")] = None,
    all_pending: Annotated[
        bool, typer.Option("--all", help="Aggregate all experiments without aggregated results")
    ] = False,
    results_dir: Annotated[
        Path | None, typer.Option("--results-dir", "-o", help="Results directory")
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Re-aggregate even if result exists")
    ] = False,
    strict: Annotated[
        bool,
        typer.Option("--strict/--no-strict", help="Fail if results incomplete (default: strict)"),
    ] = True,
    allow_mixed_backends: Annotated[
        bool,
        typer.Option(
            "--allow-mixed-backends",
            help="Allow aggregating results from different backends (not recommended)",
        ),
    ] = False,
) -> None:
    """Aggregate raw per-process results into final experiment result.

    For multi-GPU experiments, this combines results from each GPU/process
    into a single aggregated result with proper energy summation and
    throughput averaging.

    Use --no-strict to aggregate partial results (when some processes failed).
    """
    repo = FileSystemRepository(results_dir)

    if all_pending:
        # Find experiments with raw results but no aggregated results
        all_experiments = repo.list_experiments()
        pending = [exp_id for exp_id in all_experiments if not repo.has_aggregated(exp_id) or force]

        if not pending:
            console.print("[green]No pending experiments to aggregate[/green]")
            raise typer.Exit()

        console.print(f"Aggregating {len(pending)} experiments...")
        for exp_id in pending:
            _aggregate_one(repo, exp_id, force, strict, allow_mixed_backends)
    elif experiment_id:
        _aggregate_one(repo, experiment_id, force, strict, allow_mixed_backends)
    else:
        console.print("[red]Error:[/red] Provide experiment ID or use --all")
        raise typer.Exit(1)


def _aggregate_one(
    repo: FileSystemRepository,
    experiment_id: str,
    force: bool,
    strict: bool = True,
    allow_mixed_backends: bool = False,
) -> None:
    """Aggregate a single experiment."""
    if repo.has_aggregated(experiment_id) and not force:
        console.print(
            f"[yellow]Skipping[/yellow] {experiment_id} (already aggregated, use --force)"
        )
        return

    raw_results = repo.load_all_raw(experiment_id)
    if not raw_results:
        console.print(f"[red]No raw results found:[/red] {experiment_id}")
        return

    try:
        # Pass results_dir for completeness validation (Phase 5)
        aggregated = aggregate_results(
            experiment_id,
            raw_results,
            results_dir=repo._base,
            strict=strict,
            allow_mixed_backends=allow_mixed_backends,
        )
        path = repo.save_aggregated(aggregated)
        console.print(
            f"[green]✓[/green] Aggregated {experiment_id} ({len(raw_results)} processes) → {path}"
        )

        # Show summary
        metrics = calculate_efficiency_metrics(aggregated)
        console.print(f"  Tokens: {aggregated.total_tokens:,}")
        console.print(f"  Energy: {aggregated.total_energy_j:.2f} J")
        console.print(f"  Throughput: {metrics['tokens_per_second']:.2f} tok/s")
        console.print(f"  Efficiency: {metrics['tokens_per_joule']:.2f} tok/J")

        # Show streaming latency if present
        if aggregated.latency_stats is not None:
            lat = aggregated.latency_stats
            ttft_mean = lat.get("ttft_mean_ms") if isinstance(lat, dict) else lat.ttft_mean_ms
            ttft_p99 = lat.get("ttft_p99_ms") if isinstance(lat, dict) else lat.ttft_p99_ms
            itl_mean = lat.get("itl_mean_ms") if isinstance(lat, dict) else lat.itl_mean_ms
            itl_p99 = lat.get("itl_p99_ms") if isinstance(lat, dict) else lat.itl_p99_ms
            console.print(f"  TTFT: mean={ttft_mean:.1f}ms p99={ttft_p99:.1f}ms")
            if itl_mean is not None:
                console.print(f"  ITL: mean={itl_mean:.1f}ms p99={itl_p99:.1f}ms (trimmed)")

    except AggregationError as e:
        if strict:
            console.print(f"[red]Aggregation failed:[/red] {experiment_id} - {e}")
            console.print("[dim]Use --no-strict to aggregate partial results[/dim]")
        else:
            console.print(f"[yellow]Aggregation warning:[/yellow] {e}")
    except Exception as e:
        console.print(f"[red]Aggregation failed:[/red] {experiment_id} - {e}")


@config_app.callback()  # type: ignore[misc]
def config_callback(ctx: typer.Context) -> None:
    """Configuration management commands."""
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


@results_app.callback()  # type: ignore[misc]
def results_callback(ctx: typer.Context) -> None:
    """Results inspection commands."""
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


@config_app.command("validate")  # type: ignore[misc]
def config_validate(
    config_path: Annotated[Path, typer.Argument(help="Path to config file")],
) -> None:
    """Validate an experiment configuration file.

    Checks for:
    - Valid YAML/JSON syntax
    - Required fields
    - Value constraints (e.g., min_tokens <= max_tokens)
    - Config inheritance (_extends) resolution
    """
    try:
        config = load_config(config_path)
        warnings = validate_config(config)

        console.print(f"[green]✓[/green] Valid configuration: {config.config_name}")
        _display_config_summary(config, {})

        # Show warnings with severity
        if warnings:
            console.print()
        for warning in warnings:
            severity_style = "yellow" if warning.severity == "warning" else "dim"
            console.print(f"[{severity_style}]{rich_escape(str(warning))}[/{severity_style}]")

    except ConfigurationError as e:
        console.print(f"[red]Invalid configuration:[/red] {e}")
        raise typer.Exit(1) from None


@config_app.command("show")  # type: ignore[misc]
def config_show(
    config_path: Annotated[Path, typer.Argument(help="Path to config file")],
) -> None:
    """Display resolved configuration with inheritance applied.

    Shows ALL configuration parameters with visual styling:
    - Bold: Section headers
    - Bright/coloured: Explicitly set values
    - Dim: Default values
    """
    try:
        config = load_config(config_path)

        table = Table(title=f"Configuration: {config.config_name}")
        table.add_column("Field")
        table.add_column("Value")

        # =================================================================
        # Core settings (always shown)
        # =================================================================
        table.add_row(*_format_field("model_name", config.model_name, False))
        table.add_row(
            *_format_field("num_processes", config.num_processes, config.num_processes == 1)
        )
        table.add_row(*_format_field("gpu_list", config.gpu_list, config.gpu_list == [0]))
        table.add_row(
            *_format_field(
                "max_input_tokens", config.max_input_tokens, config.max_input_tokens == 512
            )
        )
        table.add_row(
            *_format_field(
                "max_output_tokens", config.max_output_tokens, config.max_output_tokens == 128
            )
        )
        table.add_row(
            *_format_field(
                "min_output_tokens", config.min_output_tokens, config.min_output_tokens == 0
            )
        )
        table.add_row(
            *_format_field(
                "num_input_prompts", config.num_input_prompts, config.num_input_prompts == 1
            )
        )
        table.add_row(
            *_format_field("fp_precision", config.fp_precision, config.fp_precision == "float16")
        )
        table.add_row(*_format_field("backend", config.backend, config.backend == "pytorch"))
        table.add_row(
            *_format_field("task_type", config.task_type, config.task_type == "text_generation")
        )
        table.add_row(
            *_format_field(
                "inference_type", config.inference_type, config.inference_type == "pure_generative"
            )
        )
        table.add_row(
            *_format_field(
                "is_encoder_decoder", config.is_encoder_decoder, config.is_encoder_decoder is False
            )
        )
        table.add_row(
            *_format_field("save_outputs", config.save_outputs, config.save_outputs is False)
        )
        table.add_row(
            *_format_field(
                "decode_token_to_text",
                config.decode_token_to_text,
                config.decode_token_to_text is False,
            )
        )
        table.add_row(*_format_field("num_cycles", config.num_cycles, config.num_cycles == 1))
        table.add_row(*_format_field("query_rate", config.query_rate, config.query_rate == 1.0))
        table.add_row(*_format_field("random_seed", config.random_seed, config.random_seed is None))

        # =================================================================
        # Batching config (always shown)
        # =================================================================
        _add_section_header(table, "batching")
        batch = config.batching_options
        table.add_row(
            *_format_field("batch_size", batch.batch_size, batch.batch_size == 1, nested=True)
        )
        table.add_row(
            *_format_field("strategy", batch.strategy, batch.strategy == "static", nested=True)
        )
        table.add_row(
            *_format_field(
                "max_tokens_per_batch",
                batch.max_tokens_per_batch,
                batch.max_tokens_per_batch is None,
                nested=True,
            )
        )

        # =================================================================
        # Sharding config (always shown)
        # =================================================================
        _add_section_header(table, "sharding")
        shard = config.sharding_config
        table.add_row(
            *_format_field("strategy", shard.strategy, shard.strategy == "none", nested=True)
        )
        table.add_row(
            *_format_field("num_shards", shard.num_shards, shard.num_shards == 1, nested=True)
        )

        # =================================================================
        # Traffic simulation (always shown)
        # =================================================================
        _add_section_header(table, "traffic_simulation")
        sim = config.latency_simulation
        table.add_row(*_format_field("enabled", sim.enabled, sim.enabled is False, nested=True))
        table.add_row(*_format_field("mode", sim.mode, sim.mode == "poisson", nested=True))
        table.add_row(
            *_format_field("target_qps", sim.target_qps, sim.target_qps == 1.0, nested=True)
        )
        table.add_row(*_format_field("seed", sim.seed, sim.seed is None, nested=True))

        # =================================================================
        # Schedule config (always shown)
        # =================================================================
        _add_section_header(table, "schedule")
        sched = config.schedule_config
        table.add_row(*_format_field("enabled", sched.enabled, sched.enabled is False, nested=True))
        table.add_row(
            *_format_field("interval", sched.interval, sched.interval is None, nested=True)
        )
        table.add_row(*_format_field("at", sched.at, sched.at is None, nested=True))
        days_str = ", ".join(sched.days) if sched.days else None
        table.add_row(*_format_field("days", days_str, sched.days is None, nested=True))
        table.add_row(
            *_format_field(
                "total_duration", sched.total_duration, sched.total_duration == "24h", nested=True
            )
        )

        # =================================================================
        # Decoder config (always shown)
        # =================================================================
        _add_section_header(table, "decoder")
        decoder = config.decoder_config
        table.add_row(*_format_field("preset", decoder.preset, decoder.preset is None, nested=True))
        mode = "deterministic (greedy)" if decoder.is_deterministic else "sampling"
        table.add_row(
            *_format_field(
                "mode", mode, decoder.temperature == 1.0 and decoder.do_sample, nested=True
            )
        )
        table.add_row(
            *_format_field(
                "temperature", decoder.temperature, decoder.temperature == 1.0, nested=True
            )
        )
        table.add_row(
            *_format_field("do_sample", decoder.do_sample, decoder.do_sample is True, nested=True)
        )
        table.add_row(*_format_field("top_p", decoder.top_p, decoder.top_p == 1.0, nested=True))
        table.add_row(*_format_field("top_k", decoder.top_k, decoder.top_k == 50, nested=True))
        table.add_row(*_format_field("min_p", decoder.min_p, decoder.min_p == 0.0, nested=True))
        table.add_row(
            *_format_field(
                "repetition_penalty",
                decoder.repetition_penalty,
                decoder.repetition_penalty == 1.0,
                nested=True,
            )
        )
        table.add_row(
            *_format_field(
                "no_repeat_ngram_size",
                decoder.no_repeat_ngram_size,
                decoder.no_repeat_ngram_size == 0,
                nested=True,
            )
        )

        # =================================================================
        # Quantization config (always shown)
        # =================================================================
        _add_section_header(table, "quantization")
        q = config.quantization_config
        table.add_row(
            *_format_field("quantization", q.quantization, q.quantization is False, nested=True)
        )
        table.add_row(
            *_format_field("load_in_4bit", q.load_in_4bit, q.load_in_4bit is False, nested=True)
        )
        table.add_row(
            *_format_field("load_in_8bit", q.load_in_8bit, q.load_in_8bit is False, nested=True)
        )
        table.add_row(
            *_format_field(
                "bnb_4bit_compute_dtype",
                q.bnb_4bit_compute_dtype,
                q.bnb_4bit_compute_dtype == "float16",
                nested=True,
            )
        )
        table.add_row(
            *_format_field(
                "bnb_4bit_quant_type",
                q.bnb_4bit_quant_type,
                q.bnb_4bit_quant_type == "nf4",
                nested=True,
            )
        )
        table.add_row(
            *_format_field(
                "bnb_4bit_use_double_quant",
                q.bnb_4bit_use_double_quant,
                q.bnb_4bit_use_double_quant is False,
                nested=True,
            )
        )

        # =================================================================
        # Prompt source (if configured)
        # =================================================================
        if config.prompt_source is not None:
            _add_section_header(table, "prompts")
            ps = config.prompt_source
            table.add_row(*_format_field("type", ps.type, False, nested=True))
            if ps.type == "file":
                table.add_row(*_format_field("path", ps.path, False, nested=True))
            else:  # huggingface
                table.add_row(*_format_field("dataset", ps.dataset, False, nested=True))
                table.add_row(*_format_field("split", ps.split, ps.split == "train", nested=True))
                table.add_row(*_format_field("subset", ps.subset, ps.subset is None, nested=True))
                table.add_row(*_format_field("column", ps.column, ps.column is None, nested=True))
                table.add_row(
                    *_format_field(
                        "sample_size", ps.sample_size, ps.sample_size is None, nested=True
                    )
                )
                table.add_row(
                    *_format_field("shuffle", ps.shuffle, ps.shuffle is False, nested=True)
                )
                table.add_row(*_format_field("seed", ps.seed, ps.seed == 42, nested=True))

        console.print(table)

    except ConfigurationError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None


@config_app.command("new")  # type: ignore[misc]
def config_new(
    output_path: Annotated[
        Path | None, typer.Option("--output", "-o", help="Output path for config file")
    ] = None,
    base_preset: Annotated[str | None, typer.Option("--preset", help="Start from a preset")] = None,
) -> None:
    """Interactive config builder for creating new experiment configs.

    Guides you through creating a valid configuration file with sensible defaults.
    """
    from rich.prompt import Confirm, Prompt

    console.print("[bold]Create new experiment configuration[/bold]\n")

    # Config name
    config_name = Prompt.ask("Config name", default="my-experiment")

    # Model name
    model_name = Prompt.ask(
        "Model name (HuggingFace path)",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    )

    # Start from preset?
    if base_preset is None:
        use_preset = Confirm.ask("Start from a preset?", default=False)
        if use_preset:
            preset_choices = ", ".join(PRESETS.keys())
            base_preset = Prompt.ask(f"Preset ({preset_choices})", default="benchmark")

    # Build config
    config_dict: dict[str, Any] = {}
    if base_preset and base_preset in PRESETS:
        config_dict = copy.deepcopy(PRESETS[base_preset])
        console.print(f"[dim]Starting from preset: {base_preset}[/dim]")

    config_dict["config_name"] = config_name
    config_dict["model_name"] = model_name

    # GPU configuration
    num_gpus = int(Prompt.ask("Number of GPUs", default="1"))
    config_dict["num_processes"] = num_gpus
    if num_gpus > 1:
        gpu_list = [
            int(g)
            for g in Prompt.ask(
                "GPU indices (comma-separated)",
                default=",".join(str(i) for i in range(num_gpus)),
            ).split(",")
        ]
        config_dict["gpu_list"] = gpu_list
    else:
        config_dict["gpu_list"] = [0]

    # Precision
    precision = Prompt.ask(
        "Precision (float32/float16/bfloat16)",
        default=config_dict.get("fp_precision", "float16"),
    )
    config_dict["fp_precision"] = precision

    # Batch size
    batch_size = int(
        Prompt.ask(
            "Batch size",
            default=str(config_dict.get("batching_options", {}).get("batch_size", 1)),
        )
    )
    if "batching_options" not in config_dict:
        config_dict["batching_options"] = {}
    config_dict["batching_options"]["batch_size"] = batch_size

    # Token limits
    max_input = int(
        Prompt.ask(
            "Max input tokens",
            default=str(config_dict.get("max_input_tokens", 512)),
        )
    )
    max_output = int(
        Prompt.ask(
            "Max output tokens",
            default=str(config_dict.get("max_output_tokens", 128)),
        )
    )
    config_dict["max_input_tokens"] = max_input
    config_dict["max_output_tokens"] = max_output

    # Quantization
    use_quant = Confirm.ask("Enable quantization?", default=False)
    if use_quant:
        quant_bits = Prompt.ask("Quantization bits (4/8)", default="4")
        config_dict["quantization_config"] = {
            "quantization": True,
            "load_in_4bit": quant_bits == "4",
            "load_in_8bit": quant_bits == "8",
        }

    # Validate config
    try:
        config = ExperimentConfig(**config_dict)
    except Exception as e:
        console.print(f"[red]Invalid configuration:[/red] {e}")
        raise typer.Exit(1) from None

    # Determine output path
    if output_path is None:
        output_path = Path(f"configs/{config_name}.yaml")

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write YAML
    import yaml

    with open(output_path, "w") as f:
        yaml.dump(
            config.model_dump(exclude_defaults=True), f, default_flow_style=False, sort_keys=False
        )

    console.print(f"\n[green]✓[/green] Created: {output_path}")
    console.print(
        f"\n[dim]Run with: llm-energy-measure experiment {output_path} --dataset alpaca -n 100[/dim]"
    )


@config_app.command("generate-grid")  # type: ignore[misc]
def config_generate_grid(
    base_config: Annotated[Path, typer.Argument(help="Base config file to vary")],
    vary: Annotated[
        list[str] | None,
        typer.Option("--vary", help="Parameter=values to vary (e.g., batch_size=1,2,4,8)"),
    ] = None,
    output_dir: Annotated[
        Path, typer.Option("--output-dir", "-o", help="Output directory for generated configs")
    ] = Path("configs/grid"),
    validate: Annotated[
        bool,
        typer.Option("--validate", help="Only generate valid configs (skip those with errors)"),
    ] = False,
    strict: Annotated[
        bool, typer.Option("--strict", help="Fail if any generated config would be invalid")
    ] = False,
) -> None:
    """Generate a grid of configs from a base config with parameter variations.

    Creates Cartesian product of all --vary parameters.

    Example:
        llm-energy-measure config generate-grid base.yaml \\
            --vary batch_size=1,2,4,8 \\
            --vary fp_precision=float16,float32 \\
            --output-dir ./grid/
    """
    import itertools

    import yaml

    if not vary:
        console.print("[red]Error:[/red] At least one --vary parameter is required")
        raise typer.Exit(1)

    # Load base config
    try:
        base = load_config(base_config)
        base_dict = base.model_dump()
    except ConfigurationError as e:
        console.print(f"[red]Error loading base config:[/red] {e}")
        raise typer.Exit(1) from None

    # Shorthand mappings for common parameters (match CLI flag behavior)
    param_shortcuts: dict[str, str] = {
        "batch_size": "batching_options.batch_size",
        "temperature": "decoder_config.temperature",
        "quantization": "quantization_config.quantization",
    }

    # Parse variations
    variations: dict[str, list[Any]] = {}
    for v in vary:
        if "=" not in v:
            console.print(f"[red]Error:[/red] Invalid --vary format: {v}")
            console.print("Expected: parameter=value1,value2,...")
            raise typer.Exit(1)

        param, values_str = v.split("=", 1)
        # Apply shorthand mappings
        param = param_shortcuts.get(param, param)
        values: list[Any] = []
        for val in values_str.split(","):
            val = val.strip()
            # Try to parse as number
            try:
                if "." in val:
                    values.append(float(val))
                else:
                    values.append(int(val))
            except ValueError:
                values.append(val)
        variations[param] = values

    # Generate Cartesian product
    param_names = list(variations.keys())
    param_values = list(variations.values())
    combinations = list(itertools.product(*param_values))

    console.print(f"Generating {len(combinations)} configs from {len(param_names)} parameters...")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate configs with validation
    # Build all config variations first (for --strict validation before write)
    config_variations: list[tuple[Path, dict[str, Any]]] = []
    base_name = base_config.stem

    for combo in combinations:
        config_dict = copy.deepcopy(base_dict)

        # Build suffix for filename
        suffix_parts = []
        for param, value in zip(param_names, combo, strict=False):
            # Apply the variation
            if "." in param:
                # Nested parameter like batching_options.batch_size
                parts = param.split(".")
                target = config_dict
                for part in parts[:-1]:
                    if part not in target:
                        target[part] = {}
                    target = target[part]
                target[parts[-1]] = value
            else:
                config_dict[param] = value

            # Clean value for filename
            clean_val = str(value).replace(".", "_")
            suffix_parts.append(f"{param.split('.')[-1]}_{clean_val}")

        # Update config name
        suffix = "_".join(suffix_parts)
        config_dict["config_name"] = f"{base_name}_{suffix}"
        output_path = output_dir / f"{base_name}_{suffix}.yaml"
        config_variations.append((output_path, config_dict))

    # Validate all configs first
    invalid_configs: list[tuple[Path, list[ConfigWarning] | str]] = []
    valid_configs: list[tuple[Path, dict[str, Any]]] = []

    for output_path, config_dict in config_variations:
        try:
            config_obj = ExperimentConfig(**config_dict)
            config_warnings = validate_config(config_obj)

            if has_blocking_warnings(config_warnings):
                invalid_configs.append((output_path, config_warnings))
            else:
                valid_configs.append((output_path, config_dict))

        except ValidationError as e:
            invalid_configs.append((output_path, str(e)))

    # --strict: fail before writing if any invalid
    if strict and invalid_configs:
        console.print(
            f"\n[yellow]⚠ {len(invalid_configs)} config(s) with blocking errors:[/yellow]"
        )
        for path, warnings_or_err in invalid_configs[:5]:
            console.print(f"  {path.name}:")
            if isinstance(warnings_or_err, list):
                for w in warnings_or_err:
                    if w.severity == "error":
                        console.print(f"    [red]{rich_escape(str(w))}[/red]")
            else:
                console.print(f"    [red]{rich_escape(str(warnings_or_err))}[/red]")
        if len(invalid_configs) > 5:
            console.print(f"  ... and {len(invalid_configs) - 5} more")
        console.print("\n[red]Error:[/red] --strict: Some configs have blocking errors")
        console.print("[dim]No configs were written.[/dim]")
        raise typer.Exit(1)

    # Write configs
    generated: list[Path] = []
    skipped_count = 0

    for output_path, config_dict in config_variations:
        # Check if this config was invalid
        is_invalid = any(p == output_path for p, _ in invalid_configs)

        if is_invalid and validate:
            skipped_count += 1
            continue  # Don't write invalid config

        # Write config
        with open(output_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        generated.append(output_path)

    # Report results
    console.print(f"[green]✓[/green] Generated {len(generated)} configs in {output_dir}/")

    if invalid_configs:
        console.print(
            f"\n[yellow]⚠ {len(invalid_configs)} config(s) with blocking errors:[/yellow]"
        )
        for path, warnings_or_err in invalid_configs[:5]:
            console.print(f"  {path.name}:")
            if isinstance(warnings_or_err, list):
                for w in warnings_or_err:
                    if w.severity == "error":
                        console.print(f"    [red]{rich_escape(str(w))}[/red]")
            else:
                console.print(f"    [red]{rich_escape(str(warnings_or_err))}[/red]")
        if len(invalid_configs) > 5:
            console.print(f"  ... and {len(invalid_configs) - 5} more")

    if validate and skipped_count:
        console.print(f"\n[dim]Skipped {skipped_count} invalid configs due to --validate[/dim]")

    if generated:
        console.print(
            f"\n[dim]Run with: llm-energy-measure batch {output_dir}/*.yaml --dataset alpaca -n 100[/dim]"
        )


@app.command("batch")  # type: ignore[misc]
def batch_run(
    config_pattern: Annotated[
        str, typer.Argument(help="Glob pattern for config files (e.g., configs/*.yaml)")
    ],
    dataset: Annotated[
        str | None,
        typer.Option("--dataset", "-d", help="HuggingFace dataset for all runs"),
    ] = None,
    sample_size: Annotated[
        int | None, typer.Option("--sample-size", "-n", help="Limit prompts for all runs")
    ] = None,
    parallel: Annotated[
        int | None,
        typer.Option("--parallel", help="Run N configs in parallel (default: sequential)"),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="List configs without running")
    ] = False,
) -> None:
    """Run multiple experiment configs in batch.

    Supports sequential (default) or parallel execution.

    Examples:
        # Sequential
        llm-energy-measure batch configs/*.yaml --dataset alpaca -n 100

        # Parallel (4 at a time)
        llm-energy-measure batch configs/*.yaml --parallel 4 --dataset alpaca -n 100
    """
    import glob as glob_module
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # Find matching configs
    config_paths = sorted(glob_module.glob(config_pattern))
    if not config_paths:
        console.print(f"[red]Error:[/red] No configs match pattern: {config_pattern}")
        raise typer.Exit(1)

    console.print(f"[bold]Batch run: {len(config_paths)} configs[/bold]")

    # Validate all configs first
    valid_configs: list[tuple[Path, ExperimentConfig]] = []
    for path in config_paths:
        try:
            config = load_config(Path(path))
            valid_configs.append((Path(path), config))
            console.print(f"  [green]✓[/green] {path}: {config.config_name}")
        except ConfigurationError as e:
            console.print(f"  [red]✗[/red] {path}: {e}")

    if len(valid_configs) < len(config_paths):
        console.print(
            f"\n[yellow]Warning:[/yellow] {len(config_paths) - len(valid_configs)} invalid configs skipped"
        )

    if not valid_configs:
        console.print("[red]Error:[/red] No valid configs to run")
        raise typer.Exit(1)

    if dry_run:
        console.print(f"\n[blue]Dry run - would execute {len(valid_configs)} experiments[/blue]")
        raise typer.Exit()

    # Build command template
    def run_config(config_path: Path) -> tuple[str, int]:
        """Run a single config and return (name, exit_code)."""
        cmd = [
            sys.executable,
            "-m",
            "llm_energy_measure.cli",
            "experiment",
            str(config_path),
        ]
        if dataset:
            cmd.extend(["--dataset", dataset])
        if sample_size:
            cmd.extend(["--sample-size", str(sample_size)])

        result = subprocess.run(cmd, capture_output=True, check=False)
        return config_path.stem, result.returncode

    console.print(f"\n[bold]Running {len(valid_configs)} experiments...[/bold]\n")

    results: dict[str, int] = {}

    if parallel and parallel > 1:
        # Parallel execution
        console.print(f"[dim]Parallel mode: {parallel} concurrent runs[/dim]\n")
        with ProcessPoolExecutor(max_workers=parallel) as executor:
            futures = {
                executor.submit(run_config, config_path): config_path
                for config_path, _ in valid_configs
            }
            for future in as_completed(futures):
                name, exit_code = future.result()
                results[name] = exit_code
                status = "[green]✓[/green]" if exit_code == 0 else f"[red]✗ ({exit_code})[/red]"
                console.print(f"  {status} {name}")
    else:
        # Sequential execution
        for config_path, config in valid_configs:
            console.print(f"[dim]Running: {config.config_name}[/dim]")
            name, exit_code = run_config(config_path)
            results[name] = exit_code
            status = "[green]✓[/green]" if exit_code == 0 else f"[red]✗ ({exit_code})[/red]"
            console.print(f"  {status} {name}")

    # Summary
    succeeded = sum(1 for code in results.values() if code == 0)
    failed = len(results) - succeeded

    console.print(f"\n[bold]Batch complete:[/bold] {succeeded} succeeded, {failed} failed")

    if failed > 0:
        raise typer.Exit(1)


@results_app.command("list")  # type: ignore[misc]
def results_list(
    results_dir: Annotated[
        Path | None, typer.Option("--results-dir", "-o", help="Results directory")
    ] = None,
    show_all: Annotated[bool, typer.Option("--all", "-a", help="Show all experiments")] = False,
) -> None:
    """List all experiments with results.

    By default shows experiments with aggregated results.
    Use --all to include experiments with only raw results.
    """
    repo = FileSystemRepository(results_dir)

    all_experiments = set(repo.list_experiments())
    aggregated = set(repo.list_aggregated())

    if not all_experiments and not aggregated:
        console.print("[yellow]No experiments found[/yellow]")
        raise typer.Exit()

    table = Table(title="Experiments")
    table.add_column("ID", style="cyan")
    table.add_column("Raw", justify="center")
    table.add_column("Aggregated", justify="center")
    table.add_column("Processes", justify="right")

    # Show aggregated experiments
    for exp_id in sorted(aggregated):
        raw_count = len(repo.list_raw(exp_id))
        table.add_row(
            exp_id,
            "[green]✓[/green]" if raw_count > 0 else "[dim]-[/dim]",
            "[green]✓[/green]",
            str(raw_count),
        )

    # Optionally show non-aggregated experiments
    if show_all:
        pending = all_experiments - aggregated
        for exp_id in sorted(pending):
            raw_count = len(repo.list_raw(exp_id))
            table.add_row(
                exp_id,
                "[green]✓[/green]",
                "[yellow]pending[/yellow]",
                str(raw_count),
            )

    console.print(table)

    if not show_all:
        pending_count = len(all_experiments - aggregated)
        if pending_count > 0:
            console.print(
                f"\n[dim]{pending_count} experiments pending aggregation (use --all to show)[/dim]"
            )


@results_app.command("show")  # type: ignore[misc]
def results_show(
    experiment_id: Annotated[str, typer.Argument(help="Experiment ID")],
    results_dir: Annotated[
        Path | None, typer.Option("--results-dir", "-o", help="Results directory")
    ] = None,
    raw: Annotated[bool, typer.Option("--raw", help="Show raw per-process results")] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Output as JSON")] = False,
) -> None:
    """Show detailed results for an experiment."""
    repo = FileSystemRepository(results_dir)

    if raw:
        raw_results = repo.load_all_raw(experiment_id)
        if not raw_results:
            console.print(f"[red]No raw results found:[/red] {experiment_id}")
            raise typer.Exit(1)

        if json_output:
            import json

            data = [r.model_dump(mode="json") for r in raw_results]
            console.print_json(json.dumps(data, default=str))
        else:
            for i, result in enumerate(raw_results):
                # Show config only for first result to avoid repetition
                _show_raw_result(result, show_config=(i == 0))
    else:
        aggregated = repo.load_aggregated(experiment_id)
        if not aggregated:
            console.print(f"[yellow]No aggregated result for:[/yellow] {experiment_id}")
            console.print("Run 'llm-energy-measure aggregate' first, or use --raw")
            raise typer.Exit(1)

        if json_output:
            console.print_json(aggregated.model_dump_json())
        else:
            _show_aggregated_result(aggregated)


def _format_dict_field(
    name: str,
    value: Any,
    default: Any,
    nested: bool = False,
) -> tuple[str, str]:
    """Format field from dict config with default comparison."""
    is_default = value == default
    indent = "  " if nested else ""
    if is_default:
        return f"[dim]{indent}{name}[/dim]", f"[dim]{value}[/dim]"
    else:
        style = "cyan" if nested else "green"
        return f"[{style}]{indent}{name}[/{style}]", str(value)


def _show_effective_config(
    config: dict[str, Any], cli_overrides: dict[str, Any] | None = None
) -> None:
    """Display effective configuration from results.

    Shows ALL configuration parameters with visual styling:
    - Bold: Section headers
    - Bright/coloured: Non-default values
    - Dim: Default values
    """
    table = Table(title="Experiment Configuration")
    table.add_column("Field")
    table.add_column("Value")

    # =================================================================
    # Core settings
    # =================================================================
    _add_section_header(table, "core")
    table.add_row(*_format_dict_field("model", config.get("model_name", "N/A"), None, nested=True))
    table.add_row(*_format_dict_field("adapter", config.get("adapter"), None, nested=True))
    table.add_row(
        *_format_dict_field("backend", config.get("backend", "pytorch"), "pytorch", nested=True)
    )
    table.add_row(
        *_format_dict_field(
            "precision", config.get("fp_precision", "float16"), "float16", nested=True
        )
    )
    num_procs = config.get("num_processes", 1)
    gpu_list = config.get("gpu_list", [0])
    table.add_row(
        *_format_dict_field(
            "processes", f"{num_procs} on GPUs {gpu_list}", "1 on GPUs [0]", nested=True
        )
    )
    table.add_row(
        *_format_dict_field(
            "task", config.get("task_type", "text_generation"), "text_generation", nested=True
        )
    )
    table.add_row(
        *_format_dict_field(
            "inference",
            config.get("inference_type", "pure_generative"),
            "pure_generative",
            nested=True,
        )
    )
    table.add_row(*_format_dict_field("seed", config.get("random_seed"), None, nested=True))

    # =================================================================
    # Token settings
    # =================================================================
    _add_section_header(table, "tokens")
    table.add_row(
        *_format_dict_field("num_prompts", config.get("num_input_prompts", 1), 1, nested=True)
    )
    table.add_row(
        *_format_dict_field("max_input", config.get("max_input_tokens", 512), 512, nested=True)
    )
    table.add_row(
        *_format_dict_field("max_output", config.get("max_output_tokens", 128), 128, nested=True)
    )
    table.add_row(
        *_format_dict_field("min_output", config.get("min_output_tokens", 0), 0, nested=True)
    )

    # =================================================================
    # Streaming settings
    # =================================================================
    _add_section_header(table, "streaming")
    table.add_row(
        *_format_dict_field("enabled", config.get("streaming", False), False, nested=True)
    )
    table.add_row(
        *_format_dict_field(
            "warmup_requests", config.get("streaming_warmup_requests", 2), 2, nested=True
        )
    )

    # =================================================================
    # Batching config
    # =================================================================
    _add_section_header(table, "batching")
    batch = config.get("batching_options", {})
    table.add_row(*_format_dict_field("batch_size", batch.get("batch_size", 1), 1, nested=True))
    table.add_row(
        *_format_dict_field("strategy", batch.get("strategy", "static"), "static", nested=True)
    )
    table.add_row(
        *_format_dict_field(
            "max_tokens_per_batch", batch.get("max_tokens_per_batch"), None, nested=True
        )
    )

    # =================================================================
    # Sharding config
    # =================================================================
    _add_section_header(table, "sharding")
    shard = config.get("sharding_config", {})
    table.add_row(
        *_format_dict_field("strategy", shard.get("strategy", "none"), "none", nested=True)
    )
    table.add_row(*_format_dict_field("num_shards", shard.get("num_shards", 1), 1, nested=True))

    # =================================================================
    # Traffic/pacing settings
    # =================================================================
    _add_section_header(table, "traffic")
    table.add_row(
        *_format_dict_field("query_rate", config.get("query_rate", 1.0), 1.0, nested=True)
    )
    latency = config.get("latency_simulation", {})
    table.add_row(
        *_format_dict_field("simulation", latency.get("enabled", False), False, nested=True)
    )
    table.add_row(
        *_format_dict_field("mode", latency.get("mode", "poisson"), "poisson", nested=True)
    )
    table.add_row(
        *_format_dict_field("target_qps", latency.get("target_qps", 1.0), 1.0, nested=True)
    )
    table.add_row(*_format_dict_field("sim_seed", latency.get("seed"), None, nested=True))

    # =================================================================
    # Schedule config
    # =================================================================
    _add_section_header(table, "schedule")
    table.add_row(*_format_dict_field("cycles", config.get("num_cycles", 1), 1, nested=True))
    schedule = config.get("schedule_config", {})
    table.add_row(
        *_format_dict_field("cron_enabled", schedule.get("enabled", False), False, nested=True)
    )
    table.add_row(*_format_dict_field("interval", schedule.get("interval"), None, nested=True))
    table.add_row(*_format_dict_field("at", schedule.get("at"), None, nested=True))
    days = schedule.get("days")
    days_str = ", ".join(days) if days else None
    table.add_row(*_format_dict_field("days", days_str, None, nested=True))
    table.add_row(
        *_format_dict_field(
            "total_duration", schedule.get("total_duration", "24h"), "24h", nested=True
        )
    )

    # =================================================================
    # Decoder config (always shown)
    # =================================================================
    _add_section_header(table, "decoder")
    decoder = config.get("decoder_config", {})
    table.add_row(*_format_dict_field("preset", decoder.get("preset"), None, nested=True))
    is_deterministic = decoder.get("temperature", 1.0) == 0.0 or not decoder.get("do_sample", True)
    mode = "deterministic (greedy)" if is_deterministic else "sampling"
    default_mode = decoder.get("temperature", 1.0) == 1.0 and decoder.get("do_sample", True)
    table.add_row(
        *_format_dict_field("mode", mode, "sampling" if default_mode else mode, nested=True)
    )
    table.add_row(
        *_format_dict_field("temperature", decoder.get("temperature", 1.0), 1.0, nested=True)
    )
    table.add_row(
        *_format_dict_field("do_sample", decoder.get("do_sample", True), True, nested=True)
    )
    table.add_row(*_format_dict_field("top_p", decoder.get("top_p", 1.0), 1.0, nested=True))
    table.add_row(*_format_dict_field("top_k", decoder.get("top_k", 50), 50, nested=True))
    table.add_row(*_format_dict_field("min_p", decoder.get("min_p", 0.0), 0.0, nested=True))
    table.add_row(
        *_format_dict_field(
            "repetition_penalty", decoder.get("repetition_penalty", 1.0), 1.0, nested=True
        )
    )
    table.add_row(
        *_format_dict_field(
            "no_repeat_ngram_size", decoder.get("no_repeat_ngram_size", 0), 0, nested=True
        )
    )

    # =================================================================
    # Quantization config (always shown)
    # =================================================================
    _add_section_header(table, "quantization")
    quant = config.get("quantization_config", {})
    table.add_row(
        *_format_dict_field("quantization", quant.get("quantization", False), False, nested=True)
    )
    table.add_row(
        *_format_dict_field("load_in_4bit", quant.get("load_in_4bit", False), False, nested=True)
    )
    table.add_row(
        *_format_dict_field("load_in_8bit", quant.get("load_in_8bit", False), False, nested=True)
    )
    table.add_row(
        *_format_dict_field(
            "bnb_4bit_compute_dtype",
            quant.get("bnb_4bit_compute_dtype", "float16"),
            "float16",
            nested=True,
        )
    )
    table.add_row(
        *_format_dict_field(
            "bnb_4bit_quant_type", quant.get("bnb_4bit_quant_type", "nf4"), "nf4", nested=True
        )
    )
    table.add_row(
        *_format_dict_field(
            "bnb_4bit_use_double_quant",
            quant.get("bnb_4bit_use_double_quant", False),
            False,
            nested=True,
        )
    )

    # =================================================================
    # Prompt source (if in results)
    # =================================================================
    ps = config.get("prompt_source")
    if ps:
        _add_section_header(table, "prompts")
        table.add_row(*_format_dict_field("type", ps.get("type", "file"), None, nested=True))
        if ps.get("type") == "file":
            table.add_row(*_format_dict_field("path", ps.get("path"), None, nested=True))
        else:  # huggingface
            table.add_row(*_format_dict_field("dataset", ps.get("dataset"), None, nested=True))
            table.add_row(
                *_format_dict_field("split", ps.get("split", "train"), "train", nested=True)
            )
            table.add_row(*_format_dict_field("subset", ps.get("subset"), None, nested=True))
            table.add_row(*_format_dict_field("column", ps.get("column"), None, nested=True))
            table.add_row(
                *_format_dict_field("sample_size", ps.get("sample_size"), None, nested=True)
            )
            table.add_row(
                *_format_dict_field("shuffle", ps.get("shuffle", False), False, nested=True)
            )
            table.add_row(*_format_dict_field("seed", ps.get("seed", 42), 42, nested=True))

    console.print(table)

    # Show CLI overrides if any
    if cli_overrides:
        console.print("\n[dim]CLI overrides:[/dim]")
        for key, override in cli_overrides.items():
            if isinstance(override, dict):
                console.print(f"  {key}: {override.get('original')} → {override.get('new')}")
            else:
                console.print(f"  {key}: {override}")


def _show_raw_result(result: RawProcessResult, show_config: bool = True) -> None:
    """Display a raw process result."""
    console.print(f"\n[bold cyan]Process {result.process_index}[/bold cyan] (GPU {result.gpu_id})")
    if result.gpu_name:
        console.print(f"GPU: {result.gpu_name}")
    if result.gpu_is_mig:
        console.print(f"MIG Profile: {result.gpu_mig_profile}")
    if result.energy_measurement_warning:
        console.print(f"[yellow]⚠ {result.energy_measurement_warning}[/yellow]")

    table = Table(show_header=False, box=None)
    table.add_column("Field", style="dim")
    table.add_column("Value")

    table.add_row("Duration", f"{result.timestamps.duration_sec:.2f}s")
    table.add_row("Tokens", f"{result.inference_metrics.total_tokens:,}")
    table.add_row("Throughput", f"{result.inference_metrics.tokens_per_second:.2f} tok/s")
    table.add_row("Energy", f"{result.energy_metrics.total_energy_j:.2f} J")
    table.add_row("FLOPs", f"{result.compute_metrics.flops_total:.2e}")

    console.print(table)

    # Show effective config (only for first result to avoid repetition)
    if show_config and result.effective_config:
        _show_effective_config(result.effective_config, result.cli_overrides)

    # Show config warnings if present
    if show_config and result.config_warnings:
        console.print("\n[yellow]Config warnings at runtime:[/yellow]")
        for warning in result.config_warnings:
            console.print(f"  [dim]{rich_escape(str(warning))}[/dim]")


def _show_aggregated_result(result: AggregatedResult) -> None:
    """Display an aggregated result."""
    console.print(f"\n[bold cyan]Experiment: {result.experiment_id}[/bold cyan]")
    console.print(f"Schema version: {result.schema_version}")
    console.print(f"Processes: {result.aggregation.num_processes}")

    metrics = calculate_efficiency_metrics(result)

    # Main metrics table
    table = Table(title="Aggregated Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")

    table.add_row("Total Tokens", f"{result.total_tokens:,}")
    table.add_row("Total Energy", f"{result.total_energy_j:.2f} J")
    table.add_row("Duration", f"{result.duration_sec:.2f} s")
    table.add_row("Throughput", f"{metrics['tokens_per_second']:.2f} tok/s")
    table.add_row("Efficiency", f"{metrics['tokens_per_joule']:.2f} tok/J")
    table.add_row("Energy/Token", f"{metrics['joules_per_token']:.4f} J/tok")
    table.add_row("Total FLOPs", f"{result.total_flops:.2e}")

    console.print(table)

    # Show streaming latency stats if present
    if result.latency_stats is not None:
        # Handle both dict (from JSON) and object (LatencyStatistics)
        lat = result.latency_stats
        if isinstance(lat, dict):
            ttft_mean = lat.get("ttft_mean_ms")
            ttft_median = lat.get("ttft_median_ms")
            ttft_p95 = lat.get("ttft_p95_ms")
            ttft_p99 = lat.get("ttft_p99_ms")
            ttft_samples = lat.get("ttft_samples")
            itl_mean = lat.get("itl_mean_ms")
            itl_median = lat.get("itl_median_ms")
            itl_p95 = lat.get("itl_p95_ms")
            itl_p99 = lat.get("itl_p99_ms")
            itl_samples = lat.get("itl_samples")
            itl_full_mean = lat.get("itl_full_mean_ms")
            itl_full_p99 = lat.get("itl_full_p99_ms")
        else:
            ttft_mean = lat.ttft_mean_ms
            ttft_median = lat.ttft_median_ms
            ttft_p95 = lat.ttft_p95_ms
            ttft_p99 = lat.ttft_p99_ms
            ttft_samples = lat.ttft_samples
            itl_mean = lat.itl_mean_ms
            itl_median = lat.itl_median_ms
            itl_p95 = lat.itl_p95_ms
            itl_p99 = lat.itl_p99_ms
            itl_samples = lat.itl_samples
            itl_full_mean = lat.itl_full_mean_ms
            itl_full_p99 = lat.itl_full_p99_ms

        console.print("\n[bold]Streaming Latency[/bold]")
        console.print(
            f"  TTFT:  mean={ttft_mean:.1f}ms  "
            f"median={ttft_median:.1f}ms  "
            f"p95={ttft_p95:.1f}ms  "
            f"p99={ttft_p99:.1f}ms  "
            f"(n={ttft_samples})"
        )
        if itl_mean is not None:
            console.print(
                f"  ITL:   mean={itl_mean:.1f}ms  "
                f"median={itl_median:.1f}ms  "
                f"p95={itl_p95:.1f}ms  "
                f"p99={itl_p99:.1f}ms  "
                f"(n={itl_samples} tokens, trimmed)"
            )
        if itl_full_mean is not None:
            console.print(
                f"  [dim]ITL (full): mean={itl_full_mean:.1f}ms  " f"p99={itl_full_p99:.1f}ms[/dim]"
            )

    # Show effective config if available
    if result.effective_config:
        _show_effective_config(result.effective_config, result.cli_overrides)

    # Show config warnings if present
    if result.config_warnings:
        console.print("\n[yellow]Config warnings at runtime:[/yellow]")
        for warning in result.config_warnings:
            console.print(f"  [dim]{rich_escape(str(warning))}[/dim]")

    # Aggregation metadata
    meta = result.aggregation
    console.print(f"\n[dim]Aggregation method: {meta.method}[/dim]")
    if meta.temporal_overlap_verified:
        console.print("[green]✓ Temporal overlap verified[/green]")
    if meta.gpu_attribution_verified:
        console.print("[green]✓ GPU attribution verified[/green]")
    for warning in meta.warnings:
        console.print(f"[yellow]⚠ {rich_escape(str(warning))}[/yellow]")


def _parse_duration(duration_str: str) -> float:
    """Parse duration string to seconds.

    Supports: '30s', '5m', '2h', '1d', '1w'
    """
    import re

    match = re.match(r"^(\d+(?:\.\d+)?)\s*([smhdw])$", duration_str.lower().strip())
    if not match:
        raise ValueError(f"Invalid duration format: {duration_str}. Use e.g., '30m', '6h', '1d'")

    value = float(match.group(1))
    unit = match.group(2)

    multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
    return value * multipliers[unit]


def _format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.0f}m"
    elif seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    else:
        return f"{seconds / 86400:.1f}d"


@app.command("schedule")  # type: ignore[misc]
def schedule_experiment(
    config_path: Annotated[
        Path | None, typer.Argument(help="Path to experiment config file")
    ] = None,
    # Scheduling options (CLI overrides config)
    interval: Annotated[
        str | None,
        typer.Option("--interval", "-i", help="Interval between runs (e.g., '6h', '30m', '1d')"),
    ] = None,
    at_time: Annotated[
        str | None,
        typer.Option("--at", help="Specific time of day to run (e.g., '09:00', '14:30')"),
    ] = None,
    days: Annotated[
        str | None,
        typer.Option("--days", help="Days to run on (e.g., 'mon,wed,fri' or 'weekdays')"),
    ] = None,
    total_duration: Annotated[
        str,
        typer.Option("--duration", "-d", help="Total duration to run daemon (e.g., '24h', '7d')"),
    ] = "24h",
    # Prompt source options
    dataset: Annotated[
        str | None,
        typer.Option("--dataset", help="Built-in dataset alias or HuggingFace path"),
    ] = None,
    sample_size: Annotated[
        int | None,
        typer.Option("--sample-size", "-n", help="Number of prompts to use"),
    ] = None,
    prompts_file: Annotated[
        Path | None,
        typer.Option("--prompts", "-p", help="Path to prompts file"),
    ] = None,
    # Other options
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="HuggingFace model name"),
    ] = None,
    preset: Annotated[
        str | None,
        typer.Option("--preset", help=f"Built-in preset ({', '.join(PRESETS.keys())})"),
    ] = None,
    results_dir: Annotated[
        Path | None,
        typer.Option("--results-dir", "-o", help="Results directory"),
    ] = None,
) -> None:
    """Run experiments on a schedule for temporal variation studies.

    Daemon mode runs experiments at specified intervals or times, useful for
    studying how energy consumption varies by time of day or day of week.

    Examples:

        # Run every 6 hours for 24 hours (4 experiments)
        llm-energy-measure schedule config.yaml --interval 6h --duration 24h

        # Run daily at 9am for a week
        llm-energy-measure schedule config.yaml --at 09:00 --duration 7d

        # Run at 9am on weekdays only for 2 weeks
        llm-energy-measure schedule config.yaml --at 09:00 --days weekdays --duration 14d

        # Run every 12 hours on weekends
        llm-energy-measure schedule config.yaml --interval 12h --days sat,sun --duration 48h

    Configuration can also be set in YAML:

        schedule_config:
          enabled: true
          interval: "6h"
          at: "09:00"
          days: ["mon", "wed", "fri"]
          total_duration: "7d"
    """
    import signal
    import time
    from datetime import datetime

    import schedule as sched

    from llm_energy_measure.config import load_config
    from llm_energy_measure.config.models import DAY_ALIASES, VALID_DAYS

    # Validate inputs
    if not config_path and not preset:
        console.print("[red]Error:[/red] Provide config file or --preset")
        raise typer.Exit(1)

    # Load config
    if config_path:
        config = load_config(config_path)
        if model:
            config = config.model_copy(update={"model_name": model})
    else:
        from llm_energy_measure.config.models import ExperimentConfig

        if not model:
            console.print(
                "[red]Error:[/red] --model is required when using --preset without config"
            )
            raise typer.Exit(1)
        preset_config = {**PRESETS[preset], "config_name": f"preset-{preset}", "model_name": model}  # type: ignore[index]
        config = ExperimentConfig(**preset_config)

    # Resolve schedule settings: CLI > config > error
    schedule_cfg = config.schedule_config
    effective_interval = interval or schedule_cfg.interval
    effective_at = at_time or schedule_cfg.at
    effective_duration = total_duration if total_duration != "24h" else schedule_cfg.total_duration

    # Parse days
    effective_days: list[str] | None = None
    if days:
        effective_days = []
        for day in days.split(","):
            day_lower = day.strip().lower()
            if day_lower in DAY_ALIASES:
                effective_days.extend(DAY_ALIASES[day_lower])
            elif day_lower in VALID_DAYS:
                effective_days.append(day_lower)
            else:
                console.print(f"[red]Error:[/red] Invalid day '{day}'")
                console.print(f"Valid: {sorted(VALID_DAYS)} or {list(DAY_ALIASES.keys())}")
                raise typer.Exit(1)
    elif schedule_cfg.days:
        effective_days = schedule_cfg.days

    # Validate we have timing
    if not effective_interval and not effective_at:
        console.print("[red]Error:[/red] Specify --interval or --at (or set in config)")
        raise typer.Exit(1)

    # Parse durations
    try:
        duration_sec = _parse_duration(effective_duration)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None

    interval_sec: float | None = None
    if effective_interval:
        try:
            interval_sec = _parse_duration(effective_interval)
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1) from None

    # Display schedule info
    console.print("\n[bold cyan]━━━ Scheduled Experiment Mode ━━━[/bold cyan]\n")
    console.print(f"  [cyan]Model:[/cyan] {config.model_name}")
    console.print(f"  [cyan]Duration:[/cyan] {effective_duration}")

    if effective_interval:
        console.print(f"  [cyan]Interval:[/cyan] {effective_interval}")
    if effective_at:
        console.print(f"  [cyan]Time:[/cyan] {effective_at}")
    if effective_days:
        console.print(f"  [cyan]Days:[/cyan] {', '.join(effective_days)}")

    # Calculate expected runs
    if interval_sec:
        expected_runs = int(duration_sec / interval_sec) + 1
        console.print(f"  [cyan]Expected runs:[/cyan] ~{expected_runs}")

    console.print()

    # State tracking
    run_count = 0
    stop_requested = False
    start_time = time.time()
    end_time = start_time + duration_sec

    def _should_run_today() -> bool:
        """Check if we should run based on day filter."""
        if not effective_days:
            return True
        today = datetime.now().strftime("%a").lower()[:3]
        return today in effective_days

    def _run_experiment() -> None:
        """Run a single experiment."""
        nonlocal run_count

        if not _should_run_today():
            console.print("[dim]Skipping - not a scheduled day[/dim]")
            return

        run_count += 1
        run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        console.print(f"\n[bold cyan]━━━ Run #{run_count} at {run_time} ━━━[/bold cyan]")

        # Build experiment command
        cmd = [
            "llm-energy-measure",
            "experiment",
        ]

        if config_path:
            cmd.append(str(config_path))
        if preset:
            cmd.extend(["--preset", preset])
        if model:
            cmd.extend(["--model", model])
        if dataset:
            cmd.extend(["--dataset", dataset])
        if sample_size:
            cmd.extend(["--sample-size", str(sample_size)])
        if prompts_file:
            cmd.extend(["--prompts", str(prompts_file)])
        if results_dir:
            cmd.extend(["--results-dir", str(results_dir)])

        console.print(f"[dim]$ {' '.join(cmd)}[/dim]\n")

        # Run experiment as subprocess
        import subprocess

        result = subprocess.run(cmd)

        if result.returncode == 0:
            console.print(f"\n[green]✓ Run #{run_count} completed successfully[/green]")
        else:
            console.print(
                f"\n[yellow]⚠ Run #{run_count} exited with code {result.returncode}[/yellow]"
            )

    def _handle_shutdown(signum: int, frame: Any) -> None:
        """Handle graceful shutdown."""
        nonlocal stop_requested
        stop_requested = True
        console.print("\n[yellow]Shutdown requested, will stop after current run...[/yellow]")

    # Register signal handlers
    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    # Run first experiment immediately
    console.print("[cyan]Running first experiment immediately...[/cyan]")
    _run_experiment()

    if stop_requested:
        console.print("\n[yellow]Stopped after first run[/yellow]")
        raise typer.Exit(0)

    # Set up schedule
    if effective_at and not interval_sec:
        # Daily at specific time
        sched.every().day.at(effective_at).do(_run_experiment)
        console.print(f"\n[cyan]Scheduled daily at {effective_at}[/cyan]")
    elif interval_sec:
        # Interval-based
        sched.every(int(interval_sec)).seconds.do(_run_experiment)
        console.print(f"\n[cyan]Scheduled every {effective_interval}[/cyan]")

    # Main loop
    console.print(
        f"[dim]Daemon running until {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}[/dim]"
    )
    console.print("[dim]Press Ctrl+C to stop gracefully[/dim]\n")

    while time.time() < end_time and not stop_requested:
        sched.run_pending()

        # Show next run time
        next_run = sched.next_run()
        if next_run:
            remaining = (next_run - datetime.now()).total_seconds()
            if remaining > 0:
                console.print(
                    f"\r[dim]Next run in {_format_duration(remaining)} "
                    f"(total: {run_count} runs, {_format_duration(end_time - time.time())} remaining)[/dim]",
                    end="",
                )

        time.sleep(10)  # Check every 10 seconds

    # Summary
    console.print("\n\n[bold cyan]━━━ Schedule Complete ━━━[/bold cyan]")
    console.print(f"  [cyan]Total runs:[/cyan] {run_count}")
    console.print(f"  [cyan]Duration:[/cyan] {_format_duration(time.time() - start_time)}")

    if stop_requested:
        console.print("  [yellow]Stopped early by user request[/yellow]")

    raise typer.Exit(0)


if __name__ == "__main__":
    app()
