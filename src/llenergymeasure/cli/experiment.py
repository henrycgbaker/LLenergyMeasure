"""Experiment execution commands.

This module contains CLI commands for running experiments, batch processing,
scheduling, and related utilities.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.markup import escape as rich_escape
from rich.prompt import Confirm

from llenergymeasure.cli.display import (
    console,
    display_config_summary,
    display_incomplete_experiment,
)
from llenergymeasure.cli.results import aggregate_one
from llenergymeasure.config.loader import has_blocking_warnings, load_config, validate_config
from llenergymeasure.config.models import (
    DEFAULT_DATASET,
    ExperimentConfig,
    HuggingFacePromptSource,
)
from llenergymeasure.constants import GRACEFUL_SHUTDOWN_TIMEOUT_SEC, PRESETS
from llenergymeasure.core.dataset_loader import (
    load_prompts_from_file,
    load_prompts_from_source,
)
from llenergymeasure.domain.experiment import AggregatedResult
from llenergymeasure.exceptions import ConfigurationError
from llenergymeasure.results.repository import FileSystemRepository
from llenergymeasure.state.experiment_state import (
    ExperimentState,
    ExperimentStatus,
    StateManager,
    compute_config_hash,
)


def _display_measurement_summary(
    repo: FileSystemRepository,
    experiment_id: str,
    effective_cycles: int,
    cycle_results: list[AggregatedResult],
) -> None:
    """Display Phase 1 measurement summary after experiment.

    Shows environment metadata, thermal throttle warnings, energy breakdown,
    and warmup convergence status from the aggregated result.

    Args:
        repo: Results repository to load aggregated result from.
        experiment_id: Base experiment ID (or cycle-specific for single-cycle).
        effective_cycles: Number of cycles executed.
        cycle_results: Already-loaded cycle results (for multi-cycle mode).
    """
    try:
        # Load the aggregated result
        result: AggregatedResult | None
        if effective_cycles > 1 and cycle_results:
            result = cycle_results[0]  # Use first cycle for environment info
        else:
            result = repo.load_aggregated(experiment_id)

        if result is None:
            return

        # Environment summary
        if result.environment is not None:
            console.print(f"  Environment: {result.environment.summary_line}")

        # Thermal throttle warning
        if result.thermal_throttle is not None and result.thermal_throttle.detected:
            console.print(
                f"  [yellow]Warning: Thermal throttling detected "
                f"({result.thermal_throttle.throttle_duration_sec:.1f}s)[/yellow]"
            )

        # Energy breakdown
        if result.energy_breakdown is not None and result.energy_breakdown.adjusted_j is not None:
            console.print(
                f"  Energy: {result.energy_breakdown.raw_j:.2f}J raw, "
                f"{result.energy_breakdown.adjusted_j:.2f}J adjusted "
                f"(baseline: {result.energy_breakdown.baseline_power_w:.1f}W)"
            )

        # Warmup status (from first process result)
        if result.process_results:
            first_process = result.process_results[0]
            if first_process.warmup_result is not None:
                wr = first_process.warmup_result
                status = "converged" if wr.converged else "not converged"
                console.print(
                    f"  Warmup: {wr.iterations_completed} prompts "
                    f"({status}, CV={wr.final_cv:.3f})"
                )
    except Exception:
        pass  # Measurement summary display is non-fatal


def resolve_prompts(
    config: ExperimentConfig,
    prompts_file: Path | None,
    dataset: str | None,
    dataset_split: str,
    dataset_column: str | None,
    sample_size: int | None,
) -> list[str]:
    """Resolve prompts from CLI args or config.

    Priority: CLI --dataset > CLI --prompts > config.dataset > config.prompts > default
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

    # Config simple dataset field (new recommended approach)
    if config.dataset:
        source = HuggingFacePromptSource(
            dataset=config.dataset.name,
            split=dataset_split if dataset_split != "train" else config.dataset.split,
            column=dataset_column or config.dataset.column,
            sample_size=sample_size or config.dataset.sample_size,
        )
        return load_prompts_from_source(source)

    # Config advanced prompt source
    if config.prompts:
        prompts = load_prompts_from_source(config.prompts)
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


def run_cmd(
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
        console.print(f"  Backend: {config.backend}")
        console.print(f"  GPUs: {config.gpus}")

        if dry_run:
            console.print("[blue]Dry run - skipping experiment execution[/blue]")
            raise typer.Exit()

        # Resolve prompts (CLI > config > default)
        prompts = resolve_prompts(
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


def experiment_cmd(
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
       lem experiment config.yaml --dataset alpaca -n 100

    2. Preset + model (quick exploration):
       lem experiment --preset quick-test --model TinyLlama/... -d alpaca

    3. Config + CLI overrides (sweeps):
       lem experiment config.yaml -b 8 --precision int8

    Multi-cycle mode for statistical robustness (academic benchmarking standard):
       lem experiment config.yaml --cycles 5 --dataset alpaca -n 100

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
            current_state.transition_to(
                ExperimentStatus.INTERRUPTED,
                error_message="Interrupted by user (SIGINT/SIGTERM)",
            )
            state_manager.save(current_state)
            console.print("\n[yellow]Experiment interrupted. Resume with:[/yellow]")
            console.print(f"  lem experiment --resume {current_state.experiment_id}")

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
            display_incomplete_experiment(existing_state)
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
        # Philosophy: CLI = workflow/meta params, YAML = testable experiment params
        deprecated_flags: list[str] = []
        cli_overrides_dict: dict[str, Any] = {}

        # For preset-only mode, generate a config_name
        if preset and not config_path:
            cli_overrides_dict["config_name"] = f"preset-{preset}"

        # Apply model if specified (required for preset-only mode)
        if model:
            cli_overrides_dict["model_name"] = model

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
        if cycles is not None:
            cli_overrides_dict["num_cycles"] = cycles

        # Deprecated testable params - warn but still apply
        if batch_size is not None:
            deprecated_flags.append("--batch-size")
            cli_overrides_dict["batching.batch_size"] = batch_size
        if precision is not None:
            deprecated_flags.append("--precision")
            cli_overrides_dict["fp_precision"] = precision
        if num_processes is not None:
            deprecated_flags.append("--num-processes")
            cli_overrides_dict["num_processes"] = num_processes
        if gpu_list is not None:
            deprecated_flags.append("--gpu-list")
            cli_overrides_dict["gpus"] = [int(g.strip()) for g in gpu_list.split(",")]
        if temperature is not None:
            deprecated_flags.append("--temperature")
            cli_overrides_dict["decoder.temperature"] = temperature
        if quantization is not None:
            deprecated_flags.append("--quantization")
            cli_overrides_dict["quantization.quantization"] = quantization

        # Emit deprecation warning if any deprecated flags used
        if deprecated_flags:
            console.print(
                f"[yellow]Deprecation warning:[/yellow] {', '.join(deprecated_flags)} "
                "are testable params - use YAML config for formal experiments"
            )

        # Load config with full provenance tracking
        from llenergymeasure.config.loader import load_config_with_provenance

        resolved = load_config_with_provenance(
            path=config_path,
            preset_name=preset,
            cli_overrides=cli_overrides_dict,
        )
        config = resolved.config
        preset_chain = resolved.preset_chain

        # Build tracked_overrides for backward compatibility
        tracked_overrides = {
            p.path: {"new": p.value, "original": None} for p in resolved.get_cli_overrides()
        }

        # Build provenance dict for serialization
        parameter_provenance = {
            path: {
                "value": p.value,
                "source": p.source.value,
                "source_detail": p.source_detail,
            }
            for path, p in resolved.provenance.items()
        }

        # Validate we have a model
        if not config.model_name or config.model_name == "__defaults__":
            console.print("[red]Error:[/red] model_name is required (from config or --model)")
            raise typer.Exit(1)

        # Determine effective launcher processes based on backend capabilities
        # vLLM/TensorRT manage parallelism internally = 1 launcher process
        # PyTorch with Accelerate = config.num_processes launcher processes
        from llenergymeasure.orchestration.launcher import get_effective_launcher_processes

        final_num_processes = get_effective_launcher_processes(config)

        # 6b. Resolve results_dir with proper precedence:
        #     CLI --results-dir > config io.results_dir > .env > default
        if results_dir is None and config.io.results_dir:
            # Config specifies results_dir and CLI didn't override
            repo = FileSystemRepository(Path(config.io.results_dir))
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
        config_hash = compute_config_hash(config.model_dump())

        # Check for incomplete experiments with same config (Phase 4)
        if not fresh and not resume:
            existing = state_manager.find_by_config_hash(config_hash)
            if existing and existing.status != ExperimentStatus.AGGREGATED:
                display_incomplete_experiment(existing)
                # Respect --yes flag for automated campaigns
                if yes:
                    resume = existing.experiment_id
                    console.print(f"[dim]Auto-resuming experiment {resume} (--yes flag)[/dim]")
                elif Confirm.ask("\nResume this experiment?", default=True):
                    resume = existing.experiment_id
                    console.print(f"[green]Resuming experiment {resume}[/green]")
                    # TODO: Actual resume logic
                else:
                    console.print("[yellow]Starting fresh experiment[/yellow]")

        # Display config with override visibility
        display_config_summary(config, tracked_overrides, preset)

        # Resolve cycles: CLI > config.num_cycles > default (1)
        effective_cycles = cycles if cycles is not None else config.num_cycles
        if effective_cycles < 1 or effective_cycles > 10:
            console.print("[red]Error:[/red] cycles must be between 1 and 10")
            raise typer.Exit(1)
        if effective_cycles > 1:
            source = "CLI" if cycles is not None else "config"
            console.print(f"  [cyan]Multi-cycle mode:[/cyan] {effective_cycles} cycles ({source})")
        else:
            console.print(
                "  [dim]Single cycle:[/dim] confidence intervals and robustness "
                "metrics require >= 3 cycles (--cycles 3)"
            )

        # Check for MIG instances and warn about energy measurement (Phase: MIG support)
        from llenergymeasure.core.gpu_info import detect_gpu_topology, validate_gpu_selection

        topology = detect_gpu_topology()
        mig_warnings = validate_gpu_selection(config.gpus, topology)
        for mig_warning in mig_warnings:
            console.print(f"[yellow]Warning:[/yellow] {mig_warning}")

        # Dry-run: show config and exit without running
        if dry_run:
            console.print("\n[cyan]--dry-run: Exiting without running experiment[/cyan]")
            raise typer.Exit(0)

        # Build launch command based on backend
        # vLLM manages its own multiprocessing (spawn), incompatible with accelerate (fork)
        backend_name = config.backend if hasattr(config, "backend") else "pytorch"

        # Always set subprocess environment to pass verbosity and HF credentials
        subprocess_env = os.environ.copy()
        subprocess_env["LLM_ENERGY_VERBOSITY"] = os.environ.get("LLM_ENERGY_VERBOSITY", "normal")

        # Explicitly propagate HuggingFace token to child processes
        # This is critical for multi-process PyTorch experiments with gated models
        # torch.distributed.run may not inherit env vars reliably with spawn method
        if hf_token := os.environ.get("HF_TOKEN"):
            subprocess_env["HF_TOKEN"] = hf_token
            # Also set legacy env var for older transformers versions
            subprocess_env["HUGGING_FACE_HUB_TOKEN"] = hf_token

        if backend_name == "vllm":
            # Direct launch for vLLM - it handles its own distribution
            cmd = [
                sys.executable,
                "-m",
                "llenergymeasure.orchestration.launcher",
            ]
            # vLLM v1 multiprocessing can have issues with CUDA initialization
            # Disable V1 multiprocessing to use simpler process model
            subprocess_env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
            subprocess_env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            # Disable torch.compile - requires C compiler not present in minimal Docker images
            subprocess_env["TORCH_COMPILE_DISABLE"] = "1"
            # Set CUDA_VISIBLE_DEVICES to the configured GPU list
            # This ensures vLLM only sees the intended GPUs
            subprocess_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in config.gpus)
        else:
            # Accelerate launch for PyTorch backend
            # Map fp_precision to accelerate's mixed_precision values
            mixed_precision_map = {"float16": "fp16", "bfloat16": "bf16", "float32": "no"}
            mixed_precision = mixed_precision_map.get(config.fp_precision, "no")

            # Determine dynamo backend from pytorch.torch_compile config
            dynamo_backend = "no"
            if config.pytorch and config.pytorch.torch_compile:
                dynamo_backend = "inductor"

            cmd = [
                sys.executable,
                "-m",
                "accelerate.commands.launch",
                "--num_processes",
                str(final_num_processes),
                "--num_machines",
                "1",
                "--mixed_precision",
                mixed_precision,
                "--dynamo_backend",
                dynamo_backend,
                "-m",
                "llenergymeasure.orchestration.launcher",
            ]

        # Generate experiment ID early so CLI and subprocess use the same ID
        # If resuming, reuse the existing experiment ID; otherwise generate new one
        if resume and existing_state:
            experiment_id = existing_state.experiment_id
        else:
            from llenergymeasure.core.distributed import get_persistent_unique_id

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
                "parameter_provenance": parameter_provenance,  # Full provenance tracking
                "preset_chain": preset_chain,  # Presets applied in order
            }
        }
        config_with_metadata = {**effective_config, **metadata}

        # Add prompt source args to base command (before config, as config is cycle-specific)
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

        # Store base command before adding config (config is per-cycle)
        base_cmd = list(cmd)

        if backend_name == "vllm":
            console.print("\n[dim]$ python -m llenergymeasure.orchestration.launcher ...[/dim]\n")
        else:
            console.print(
                f"\n[dim]$ accelerate launch --num_processes {final_num_processes} ...[/dim]\n"
            )

        # Multi-cycle support: run experiment multiple times for statistical robustness
        from llenergymeasure.progress import CycleProgress

        cycle_results: list[AggregatedResult] = []
        cycle_metadata_list: list[Any] = []
        base_experiment_id = experiment_id

        # Use cycle progress for multi-cycle runs
        cycle_progress = CycleProgress(effective_cycles)

        with cycle_progress:
            for cycle_idx in range(effective_cycles):
                # Generate cycle-specific experiment ID
                if effective_cycles > 1:
                    cycle_experiment_id = f"{base_experiment_id}_c{cycle_idx}"
                    cycle_progress.info(f"\n━━━ Cycle {cycle_idx + 1}/{effective_cycles} ━━━")
                else:
                    cycle_experiment_id = experiment_id

                # Collect cycle metadata (temperature, load) before each cycle
                from llenergymeasure.results.cycle_statistics import (
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
                    tmp_config_path = tmp.name
                    cmd_cycle = [*base_cmd, "--config", tmp_config_path]
                    console.print(f"[dim]Generated config with metadata: {tmp_config_path}[/dim]")

                # Create state for this cycle
                current_state = ExperimentState(
                    experiment_id=cycle_experiment_id,
                    status=ExperimentStatus.RUNNING,
                    num_processes=final_num_processes,
                    config_hash=config_hash,
                    config_path=str(config_path) if config_path else None,
                    model_name=config.model_name,
                    prompt_args={
                        "dataset": dataset,
                        "sample_size": sample_size,
                    },
                )
                state_manager.save(current_state)

                # Register signal handlers
                signal.signal(signal.SIGINT, _handle_interrupt)
                signal.signal(signal.SIGTERM, _handle_interrupt)

                # Run subprocess with its own process group for clean signal handling
                subprocess_handle = subprocess.Popen(
                    cmd_cycle,
                    env=subprocess_env,
                    start_new_session=True,  # Creates new process group
                )

                # Wait for subprocess
                exit_code = subprocess_handle.wait()

                if exit_code != 0:
                    current_state.transition_to(
                        ExperimentStatus.FAILED,
                        error_message=f"Subprocess exited with code {exit_code}",
                    )
                    state_manager.save(current_state)
                    console.print(f"[red]Experiment failed[/red] (exit code {exit_code})")
                    raise typer.Exit(exit_code)

                # Mark cycle as completed
                current_state.transition_to(ExperimentStatus.COMPLETED)
                state_manager.save(current_state)

                # Auto-aggregate unless --no-aggregate
                if not no_aggregate:
                    console.print("\n[dim]Auto-aggregating results...[/dim]")
                    aggregate_one(repo, cycle_experiment_id, force=False, strict=False)

                    # Load aggregated result for multi-cycle statistics
                    if effective_cycles > 1:
                        agg = repo.load_aggregated(cycle_experiment_id)
                        if agg:
                            cycle_results.append(agg)

                    # Update state to AGGREGATED
                    current_state.transition_to(ExperimentStatus.AGGREGATED)
                    state_manager.save(current_state)

                cycle_progress.advance()

        # Multi-cycle statistics
        if effective_cycles > 1 and cycle_results:
            from llenergymeasure.results.cycle_statistics import (
                create_cycle_statistics,
                create_multi_cycle_result,
            )

            console.print("\n[bold cyan]━━━ Multi-Cycle Statistics ━━━[/bold cyan]")
            stats = create_cycle_statistics(cycle_results)

            console.print(
                f"  Throughput: {stats.throughput_mean_tps:.2f} ± {stats.throughput_std_tps:.2f} tok/s"
            )
            console.print(f"  Energy: {stats.energy_mean_j:.2f} ± {stats.energy_std_j:.2f} J")
            console.print(
                f"  Efficiency: {stats.efficiency_mean_tpj:.2f} ± {stats.efficiency_std_tpj:.2f} tok/J"
            )
            console.print(f"  CoV (throughput): {stats.throughput_cv:.1%}")

            # Create and save multi-cycle result
            multi_cycle_result = create_multi_cycle_result(
                base_experiment_id,
                cycle_results,
                cycle_metadata_list,
                effective_config,
            )
            stats_path = actual_results_dir / f"{base_experiment_id}_multi_cycle.json"
            stats_path.write_text(multi_cycle_result.model_dump_json(indent=2))
            console.print(f"\n[dim]Statistics saved: {stats_path}[/dim]")

        # --- Phase 1: Display environment/measurement summary ---
        _display_measurement_summary(repo, base_experiment_id, effective_cycles, cycle_results)

        console.print("\n[green]✓ Experiment completed successfully[/green]")

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


def aggregate_cmd(
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
            aggregate_one(repo, exp_id, force, strict, allow_mixed_backends)
    elif experiment_id:
        aggregate_one(repo, experiment_id, force, strict, allow_mixed_backends)
    else:
        console.print("[red]Error:[/red] Provide experiment ID or use --all")
        raise typer.Exit(1)


__all__ = [
    "aggregate_cmd",
    "experiment_cmd",
    "resolve_prompts",
    "run_cmd",
]
