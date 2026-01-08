"""Command-line interface for llm-energy-measure.

Provides commands for:
- Running experiments
- Aggregating raw results
- Validating configurations
- Listing and inspecting results
"""

from __future__ import annotations

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
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table

from llm_energy_measure.config.loader import load_config, validate_config
from llm_energy_measure.config.models import ExperimentConfig, HuggingFacePromptSource
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

console = Console()


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


def _display_config_summary(
    config: ExperimentConfig,
    overrides: dict[str, Any],
    preset_name: str | None = None,
) -> None:
    """Display config summary with override visibility."""
    console.print(f"\n[bold]Experiment: {config.config_name}[/bold]")
    console.print(f"  Model: {config.model_name}")
    console.print(f"  Processes: {config.num_processes} on GPUs {config.gpu_list}")
    console.print(f"  Precision: {config.fp_precision}")
    console.print(f"  Batch size: {config.batching_options.batch_size}")

    if preset_name:
        console.print(f"  Preset: [cyan]{preset_name}[/cyan]")

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
        typer.Option(
            "--dataset", "-d", help="HuggingFace dataset (alias: alpaca, gsm8k, mmlu, sharegpt)"
        ),
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
    - --dataset <name>: HuggingFace dataset (alpaca, gsm8k, mmlu, sharegpt, or full path)
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

    # Default fallback
    console.print("[yellow]Warning:[/yellow] No prompts specified, using default")
    return ["Hello, how are you?"]


@app.command("experiment")  # type: ignore[misc]
def experiment(
    config_path: Annotated[
        Path | None, typer.Argument(help="Path to experiment config file (optional with --preset)")
    ] = None,
    # Prompt source options
    dataset: Annotated[
        str | None,
        typer.Option("--dataset", "-d", help="HuggingFace dataset (alpaca, gsm8k, mmlu, sharegpt)"),
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
    # CLI overrides for config parameters
    batch_size: Annotated[
        int | None,
        typer.Option("--batch-size", "-b", help="Override batch size"),
    ] = None,
    precision: Annotated[
        str | None,
        typer.Option("--precision", help="Override fp_precision (float32/float16/bfloat16)"),
    ] = None,
    max_tokens: Annotated[
        int | None,
        typer.Option("--max-tokens", help="Override max_output_tokens"),
    ] = None,
    seed: Annotated[
        int | None,
        typer.Option("--seed", help="Random seed for reproducibility"),
    ] = None,
    num_processes: Annotated[
        int | None,
        typer.Option("--num-processes", help="Override number of processes"),
    ] = None,
    gpu_list: Annotated[
        str | None,
        typer.Option("--gpu-list", help="Override GPU list (comma-separated, e.g., '0,1,2')"),
    ] = None,
    temperature: Annotated[
        float | None,
        typer.Option("--temperature", help="Override decoder temperature"),
    ] = None,
    quantization: Annotated[
        bool | None,
        typer.Option("--quantization/--no-quantization", help="Enable/disable quantization"),
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
) -> None:
    """Run experiment with automatic accelerate handling and auto-aggregation.

    Supports three modes:

    1. Config file (formal experiments):
       llm-energy-measure experiment config.yaml --dataset alpaca -n 100

    2. Preset + model (quick exploration):
       llm-energy-measure experiment --preset quick-test --model TinyLlama/... -d alpaca

    3. Config + CLI overrides (sweeps):
       llm-energy-measure experiment config.yaml -b 8 --precision int8

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

        # 4. Prepare CLI overrides
        cli_overrides_dict: dict[str, Any] = {}
        if batch_size is not None:
            cli_overrides_dict["batching_options.batch_size"] = batch_size
        if precision is not None:
            cli_overrides_dict["fp_precision"] = precision
        if max_tokens is not None:
            cli_overrides_dict["max_output_tokens"] = max_tokens
        if seed is not None:
            cli_overrides_dict["random_seed"] = seed
        if num_processes is not None:
            cli_overrides_dict["num_processes"] = num_processes
        if gpu_list is not None:
            cli_overrides_dict["gpu_list"] = [int(g.strip()) for g in gpu_list.split(",")]
        if temperature is not None:
            cli_overrides_dict["decoder_config.temperature"] = temperature
        if quantization is not None:
            cli_overrides_dict["quantization_config.quantization"] = quantization

        # 5. Apply CLI overrides and track them
        config_dict, tracked_overrides = _apply_cli_overrides(config_dict, cli_overrides_dict)

        # 6. Create final config
        config = ExperimentConfig(**config_dict)
        final_num_processes = config.num_processes

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

        # Check for MIG instances and warn about energy measurement (Phase: MIG support)
        from llm_energy_measure.core.gpu_info import detect_gpu_topology, validate_gpu_selection

        topology = detect_gpu_topology()
        mig_warnings = validate_gpu_selection(config.gpu_list, topology)
        for warning in mig_warnings:
            console.print(f"[yellow]Warning:[/yellow] {warning}")

        # Build accelerate command
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
        metadata = {
            "_metadata": {
                "experiment_id": experiment_id,  # Pass to subprocess
                "effective_config": effective_config,
                "cli_overrides": tracked_overrides,
                "original_config_path": str(config_path) if config_path else None,
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

        console.print(
            f"\n[dim]$ accelerate launch --num_processes {final_num_processes} ...[/dim]\n"
        )

        # Create experiment state before launch (Phase 2)
        current_state = ExperimentState(
            experiment_id=experiment_id,
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
            # Run accelerate with Popen for better control (Phase 3)
            # start_new_session=True creates a new process group, allowing us to
            # kill all child processes (accelerate workers) with os.killpg()
            subprocess_handle = subprocess.Popen(cmd, start_new_session=True)
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
                        console.print("\n[dim]Auto-aggregating results...[/dim]")
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
                        console.print(f"\n[green]Results saved to:[/green] {result_path}")
                    except AggregationError as e:
                        console.print(f"[yellow]Aggregation warning:[/yellow] {e}")
                        console.print(
                            f"Raw results available. Aggregate manually with: "
                            f"llm-energy-measure aggregate {current_state.experiment_id}"
                        )
                        state_manager.save(current_state)
                else:
                    console.print(
                        f"\n[green]Experiment complete.[/green] "
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
            _aggregate_one(repo, exp_id, force, strict)
    elif experiment_id:
        _aggregate_one(repo, experiment_id, force, strict)
    else:
        console.print("[red]Error:[/red] Provide experiment ID or use --all")
        raise typer.Exit(1)


def _aggregate_one(
    repo: FileSystemRepository,
    experiment_id: str,
    force: bool,
    strict: bool = True,
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
        console.print(f"  Model: {config.model_name}")
        console.print(f"  Processes: {config.num_processes} on GPUs {config.gpu_list}")

        if config.quantization_config.quantization:
            q = config.quantization_config
            bits = "4-bit" if q.load_in_4bit else "8-bit" if q.load_in_8bit else "unknown"
            console.print(f"  Quantization: {bits}")

        for warning in warnings:
            console.print(f"[yellow]Warning:[/yellow] {warning}")

    except ConfigurationError as e:
        console.print(f"[red]Invalid configuration:[/red] {e}")
        raise typer.Exit(1) from None


@config_app.command("show")  # type: ignore[misc]
def config_show(
    config_path: Annotated[Path, typer.Argument(help="Path to config file")],
) -> None:
    """Display resolved configuration with inheritance applied."""
    try:
        config = load_config(config_path)

        table = Table(title=f"Configuration: {config.config_name}")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("model_name", config.model_name)
        table.add_row("num_processes", str(config.num_processes))
        table.add_row("gpu_list", str(config.gpu_list))
        table.add_row("max_input_tokens", str(config.max_input_tokens))
        table.add_row("max_output_tokens", str(config.max_output_tokens))
        table.add_row("fp_precision", config.fp_precision)
        table.add_row("backend", config.backend)

        if config.quantization_config.quantization:
            q = config.quantization_config
            quant = "4-bit" if q.load_in_4bit else "8-bit" if q.load_in_8bit else "enabled"
            table.add_row("quantization", quant)

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

    # Generate configs
    generated = []
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
        base_name = base_config.stem
        suffix = "_".join(suffix_parts)
        config_dict["config_name"] = f"{base_name}_{suffix}"

        # Write config
        output_path = output_dir / f"{base_name}_{suffix}.yaml"
        with open(output_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        generated.append(output_path)

    console.print(f"[green]✓[/green] Generated {len(generated)} configs in {output_dir}/")
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
            for result in raw_results:
                _show_raw_result(result)
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


def _show_raw_result(result: RawProcessResult) -> None:
    """Display a raw process result."""
    console.print(f"\n[bold cyan]Process {result.process_index}[/bold cyan] (GPU {result.gpu_id})")

    table = Table(show_header=False, box=None)
    table.add_column("Field", style="dim")
    table.add_column("Value")

    table.add_row("Duration", f"{result.timestamps.duration_sec:.2f}s")
    table.add_row("Tokens", f"{result.inference_metrics.total_tokens:,}")
    table.add_row("Throughput", f"{result.inference_metrics.tokens_per_second:.2f} tok/s")
    table.add_row("Energy", f"{result.energy_metrics.total_energy_j:.2f} J")
    table.add_row("FLOPs", f"{result.compute_metrics.flops_total:.2e}")

    console.print(table)


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

    # Aggregation metadata
    meta = result.aggregation
    console.print(f"\n[dim]Aggregation method: {meta.method}[/dim]")
    if meta.temporal_overlap_verified:
        console.print("[green]✓ Temporal overlap verified[/green]")
    if meta.gpu_attribution_verified:
        console.print("[green]✓ GPU attribution verified[/green]")
    for warning in meta.warnings:
        console.print(f"[yellow]⚠ {warning}[/yellow]")


if __name__ == "__main__":
    app()
