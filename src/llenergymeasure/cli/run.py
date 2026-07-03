"""llem run - primary command for running LLM efficiency experiments."""

from __future__ import annotations

import signal
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer
from pydantic import ValidationError

from llenergymeasure.api import run_experiment
from llenergymeasure.cli._display import (
    format_error,
    format_validation_error,
    print_dry_run,
    print_result_summary,
)
from llenergymeasure.cli._vram import estimate_vram, get_gpu_vram_gb
from llenergymeasure.config.loader import load_experiment_config
from llenergymeasure.config.ssot import (
    RUNNER_DOCKER,
    RUNNER_LOCAL,
    Engine,
)
from llenergymeasure.utils.exceptions import (
    ConfigError,
    EngineError,
    ExperimentError,
    PreFlightError,
    StudyError,
)

if TYPE_CHECKING:
    from llenergymeasure.cli._step_display import _CompletedRow

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
    engine: Annotated[
        str | None,
        typer.Option("--engine", "-e", help="Inference engine (transformers, vllm, tensorrt)"),
    ] = None,
    dataset: Annotated[
        str | None,
        typer.Option("--dataset", "-d", help="Dataset name"),
    ] = None,
    n_prompts: Annotated[
        int | None,
        typer.Option("--n-prompts", "-n", help="Number of prompts to run"),
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
        int,
        typer.Option("--verbose", "-v", count=True, help="Increase verbosity (-v=INFO, -vv=DEBUG)"),
    ] = 0,
    cycles: Annotated[
        int | None,
        typer.Option("--cycles", help="Number of cycles (study mode)"),
    ] = None,
    order: Annotated[
        str | None,
        typer.Option(
            "--order",
            help="Experiment ordering: sequential, interleave, shuffle, reverse, latin_square (study mode)",
        ),
    ] = None,
    no_gaps: Annotated[
        bool,
        typer.Option("--no-gaps", help="Disable thermal gaps between experiments (study mode)"),
    ] = False,
    skip_preflight: Annotated[
        bool,
        typer.Option(
            "--skip-preflight",
            help="Skip Docker pre-flight checks (GPU visibility, CUDA/driver compatibility)",
        ),
    ] = False,
    resume: Annotated[
        bool,
        typer.Option("--resume", help="Resume most recent interrupted study"),
    ] = False,
    resume_dir: Annotated[
        Path | None,
        typer.Option("--resume-dir", help="Resume a specific study directory"),
    ] = None,
    fail_fast: Annotated[
        bool,
        typer.Option(
            "--fail-fast", help="Abort study on first failure (circuit breaker threshold=1)"
        ),
    ] = False,
    no_circuit_breaker: Annotated[
        bool,
        typer.Option("--no-circuit-breaker", help="Disable circuit breaker entirely"),
    ] = False,
    timeout: Annotated[
        float | None,
        typer.Option("--timeout", help="Study wall-clock timeout in hours (e.g. 24, 1.5)"),
    ] = None,
    no_lock: Annotated[
        bool,
        typer.Option("--no-lock", help="Disable GPU lock files (advanced)"),
    ] = False,
    no_dedup: Annotated[
        bool,
        typer.Option(
            "--no-dedup",
            help=(
                "Disable library-resolution mechanism sweep dedup. Every declared "
                "config runs regardless of measurement equivalence (study mode)."
            ),
        ),
    ] = False,
) -> None:
    """Run an LLM efficiency experiment."""

    from llenergymeasure.cli import _setup_logging

    _setup_logging(verbose)
    verbose_on = verbose > 0

    # Install SIGINT handler so Ctrl-C exits with code 130
    def _handle_sigint(signum: int, frame: Any) -> None:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        _run_impl(
            config=config,
            model=model,
            engine=engine,
            dataset=dataset,
            n_prompts=n_prompts,
            output=output,
            dry_run=dry_run,
            quiet=quiet,
            verbose=verbose_on,
            cycles=cycles,
            order=order,
            no_gaps=no_gaps,
            skip_preflight=skip_preflight,
            resume=resume,
            resume_dir=resume_dir,
            fail_fast=fail_fast,
            no_circuit_breaker=no_circuit_breaker,
            timeout=timeout,
            no_lock=no_lock,
            no_dedup=no_dedup,
        )
    except ConfigError as e:
        print(format_error(e, verbose=verbose_on), file=sys.stderr)
        raise typer.Exit(code=2) from None
    except (PreFlightError, ExperimentError, EngineError, StudyError) as e:
        print(format_error(e, verbose=verbose_on), file=sys.stderr)
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
    engine: str | None,
    dataset: str | None,
    n_prompts: int | None,
    output: str | None,
    dry_run: bool,
    quiet: bool,
    verbose: bool,
    cycles: int | None = None,
    order: str | None = None,
    no_gaps: bool = False,
    skip_preflight: bool = False,
    resume: bool = False,
    resume_dir: Path | None = None,
    fail_fast: bool = False,
    no_circuit_breaker: bool = False,
    timeout: float | None = None,
    no_lock: bool = False,
    no_dedup: bool = False,
) -> None:
    """Core implementation - separated for clean error handling in run()."""
    # Build CLI overrides dict - only include flags the user explicitly passed
    cli_overrides: dict[str, Any] = {}
    if model is not None:
        cli_overrides["task.model"] = model
    if engine is not None:
        cli_overrides["engine"] = engine
    if dataset is not None:
        cli_overrides["task.dataset.source"] = dataset
    if n_prompts is not None:
        cli_overrides["task.dataset.n_prompts"] = n_prompts

    # Validate we have enough information to resolve a config
    if config is None and model is None:
        raise ConfigError(
            "Provide a config file or --model flag.\n"
            "  Examples:\n"
            "    llem run experiment.yaml\n"
            "    llem run --model gpt2 --engine transformers"
        )

    # Study detection: YAML with sweep: or experiments: keys is a study
    is_study = False
    if config is not None:
        import yaml

        try:
            raw = yaml.safe_load(config.read_text())
            if isinstance(raw, dict) and ("sweep" in raw or "experiments" in raw):
                is_study = True
        except Exception:
            pass  # Fall through to normal experiment path - loader will raise if invalid

    # Route to study execution path
    if is_study:
        assert config is not None  # Guarded by study detection above
        _run_study_impl(
            config=config,
            cli_overrides=cli_overrides,
            cycles=cycles,
            order=order,
            no_gaps=no_gaps,
            quiet=quiet,
            verbose=verbose,
            skip_preflight=skip_preflight,
            dry_run=dry_run,
            output=output,
            resume=resume,
            resume_dir=resume_dir,
            fail_fast=fail_fast,
            no_circuit_breaker=no_circuit_breaker,
            timeout=timeout,
            no_lock=no_lock,
            no_dedup=no_dedup,
        )
        return

    # Single-experiment path: warn about any study-only flags the user set.
    # These are parsed by run() for study mode but silently ignored here, so
    # --cycles 5 on a single experiment would run once with no feedback.
    _warn_ignored_study_flags(
        cycles=cycles,
        order=order,
        no_gaps=no_gaps,
        resume=resume,
        resume_dir=resume_dir,
        fail_fast=fail_fast,
        no_circuit_breaker=no_circuit_breaker,
        timeout=timeout,
        no_lock=no_lock,
        no_dedup=no_dedup,
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
        print_dry_run(experiment_config, vram, gpu_vram_gb, verbose=verbose, output_dir=output)
        return

    # --- Run branch ---
    # Build experiment header string
    runner_tag = _resolve_runner_tag(experiment_config)
    header = _build_header(experiment_config, runner_tag=runner_tag)

    effective_mode = _resolve_progress_mode(quiet, verbose)

    # Create progress display (None in quiet mode).
    # Steps are pre-registered with a fixed count so [x/y] counters are
    # stable. Steps that don't apply are shown as SKIP.
    progress = None
    display = None
    if effective_mode != "quiet":
        from llenergymeasure.cli._step_display import StepDisplay
        from llenergymeasure.domain.progress import docker_steps

        display = StepDisplay(
            header=f"Experiment: {header}",
            force_plain=effective_mode == "plain",
        )
        # Pre-register: Docker path is the common case (auto-elevation).
        # Local path is rare (only when runner explicitly set to local).
        # Single-experiment CLI has no study-level image prep, so include
        # image_check/pull. host_baseline tracks where STEP_BASELINE sits
        # relative to container_start (see docker_steps() docstring).
        host_baseline = (
            experiment_config.measurement.baseline.enabled
            and experiment_config.measurement.baseline.strategy != "fresh"
        )
        display.register_steps(docker_steps(images_prepared=False, host_baseline=host_baseline))
        display.start()
        progress = display

    result = None
    try:
        result = run_experiment(
            experiment_config,
            skip_preflight=skip_preflight,
            progress=progress,
            output_dir=output,
        )
    finally:
        if display is not None:
            energy = getattr(result, "total_energy_j", None) if result is not None else None
            throughput = (
                getattr(result, "avg_tokens_per_second", None) if result is not None else None
            )
            display.finish(energy_j=energy, throughput_tok_s=throughput)

    print_result_summary(result)

    if output:
        print(f"Saved: {output}", file=sys.stderr)


def _warn_ignored_study_flags(
    *,
    cycles: int | None,
    order: str | None,
    no_gaps: bool,
    resume: bool,
    resume_dir: Path | None,
    fail_fast: bool,
    no_circuit_breaker: bool,
    timeout: float | None,
    no_lock: bool,
    no_dedup: bool,
) -> None:
    """Warn when study-only flags are set on a single-experiment run.

    These flags only take effect in study mode (a YAML with ``sweep:`` or
    ``experiments:``). On a single experiment they are silently ignored, which
    hides mistakes like ``--cycles 5`` on a one-off run.
    """
    set_flags = [
        name
        for name, is_set in (
            ("--cycles", cycles is not None),
            ("--order", order is not None),
            ("--no-gaps", no_gaps),
            ("--resume", resume),
            ("--resume-dir", resume_dir is not None),
            ("--fail-fast", fail_fast),
            ("--no-circuit-breaker", no_circuit_breaker),
            ("--timeout", timeout is not None),
            ("--no-lock", no_lock),
            ("--no-dedup", no_dedup),
        )
        if is_set
    ]
    if set_flags:
        print(
            f"Warning: study-only flag(s) ignored on a single-experiment run: "
            f"{', '.join(set_flags)}. These apply only to study configs "
            f"(YAML with sweep: or experiments:).",
            file=sys.stderr,
        )


def _resolve_progress_mode(quiet: bool, verbose: bool) -> str:
    """Resolve effective progress mode: CLI flags > user config > default."""
    if quiet:
        return "quiet"
    if verbose:
        return "plain"
    from llenergymeasure.config.user_config import load_user_config

    return load_user_config().ui.progress_mode


def _resolve_runner_tag(config: Any) -> str:
    """Determine the runner tag string for display from config.runner.

    Returns "local" or "docker" based on the runner field.
    """
    runner = getattr(config, "runner", "auto")
    if runner == RUNNER_LOCAL:
        return RUNNER_LOCAL
    if runner == RUNNER_DOCKER or (isinstance(runner, str) and runner.startswith("docker:")):
        return RUNNER_DOCKER
    # auto: transformers defaults to local, vllm/tensorrt default to docker
    engine = getattr(config, "engine", Engine.TRANSFORMERS)
    return RUNNER_LOCAL if engine == Engine.TRANSFORMERS else RUNNER_DOCKER


def _build_header(config: Any, runner_tag: str = RUNNER_LOCAL) -> str:
    """Build compact experiment header: model | engine [runner] + deviation fields.

    Args:
        config: ExperimentConfig with model, engine, dtype, dataset fields.
        runner_tag: Runner tag string ("local" or "docker").
    """
    from llenergymeasure.config.models import DatasetConfig

    _ds_fields = DatasetConfig.model_fields
    default_n = _ds_fields["n_prompts"].default
    default_source = _ds_fields["source"].default

    # Strip HuggingFace org prefix (meta-llama/Llama-3.2-1B-Instruct -> Llama-3.2-1B-Instruct)
    model = config.task.model.split("/")[-1] if "/" in config.task.model else config.task.model
    parts = [f"{model} | {config.engine}"]
    # Deviation fields (only when non-default/explicit)
    # dtype lives on the active engine's engine_params; only show when set.
    engine_dtype = getattr(config.active_engine_params(), "dtype", None)
    if engine_dtype is not None:
        parts.append(engine_dtype)
    if config.task.dataset.n_prompts != default_n:
        parts.append(f"n_prompts={config.task.dataset.n_prompts}")
    if config.task.dataset.source != default_source:
        parts.append(config.task.dataset.source)
    return f"{' | '.join(parts)} [{runner_tag}]"


# ---------------------------------------------------------------------------
# Study execution path
# ---------------------------------------------------------------------------


def _resolve_resume_target(
    resume: bool, resume_dir: Path | None, output: str | None
) -> tuple[Path | None, Any, bool]:
    """Resolve the resume target directory and load its manifest (best-effort).

    Returns ``(resume_dir, resume_manifest, is_resume)``. Raises typer.BadParameter
    when an explicit --resume-dir is not a study directory, or --resume finds nothing.
    """
    resume_manifest = None
    is_resume = resume or resume_dir is not None
    if resume_dir is not None:
        if not (resume_dir / "manifest.json").exists():
            raise typer.BadParameter(
                f"No manifest.json in {resume_dir} - not a valid study directory.",
                param_hint="--resume-dir",
            )
        try:
            from llenergymeasure.api import load_resume_state

            resume_manifest, _ = load_resume_state(resume_dir)
        except Exception:
            pass  # Best-effort: display will work without manifest data
    elif resume:
        from llenergymeasure.api import find_resumable_study

        _output = Path(output or "./results")
        resume_dir = find_resumable_study(_output)
        if resume_dir is None:
            raise typer.BadParameter(
                f"No resumable study found in {_output}. Run a study first or use --resume-dir.",
                param_hint="--resume",
            )
        try:
            from llenergymeasure.api import load_resume_state

            resume_manifest, _ = load_resume_state(resume_dir)
        except Exception:
            pass  # Best-effort: display will work without manifest data
    return resume_dir, resume_manifest, is_resume


def _build_study_cli_overrides(
    cli_overrides: dict[str, Any],
    cycles: int | None,
    order: str | None,
    no_gaps: bool,
    fail_fast: bool,
    no_circuit_breaker: bool,
    timeout: float | None,
    no_dedup: bool,
    yaml_execution: dict[str, Any],
) -> dict[str, Any]:
    """Merge CLI flags into the study override dict passed to load_study.

    Applies the CLI-layer effective defaults (n_cycles=3, experiment_order="shuffle")
    only when neither the YAML study_execution block nor a CLI flag sets them; the
    Pydantic defaults are intentionally more conservative (n_cycles=1).
    """
    exec_overrides: dict[str, Any] = {}

    if cycles is not None:
        exec_overrides["n_cycles"] = cycles
    elif "n_cycles" not in yaml_execution:
        exec_overrides["n_cycles"] = 3  # CLI effective default

    if order is not None:
        exec_overrides["experiment_order"] = order
    elif "experiment_order" not in yaml_execution:
        exec_overrides["experiment_order"] = "shuffle"  # CLI effective default

    if no_gaps:
        exec_overrides["experiment_gap_seconds"] = 0
        exec_overrides["cycle_gap_seconds"] = 0

    # Robustness overrides: circuit breaker, timeout
    if fail_fast:
        exec_overrides["max_consecutive_failures"] = 1
        exec_overrides["circuit_breaker_cooldown_seconds"] = 0
    if no_circuit_breaker:
        exec_overrides["max_consecutive_failures"] = 0
    if timeout is not None:
        exec_overrides["wall_clock_timeout_hours"] = timeout

    # --no-dedup disables library-resolution mechanism sweep dedup (runs every declared config)
    if no_dedup:
        exec_overrides["deduplicate_equivalent"] = False

    study_cli_overrides: dict[str, Any] = {}
    if cli_overrides:
        study_cli_overrides.update(cli_overrides)
    if exec_overrides:
        study_cli_overrides["study_execution"] = exec_overrides
    return study_cli_overrides


def _print_config_summary(
    console: Any,
    study_config: Any,
    is_resume: bool,
    resume_manifest: Any,
    expand_elapsed: float,
) -> None:
    """Print the config-expansion summary lines (valid configs, skipped, resume status)."""
    from llenergymeasure.utils.formatting import format_elapsed as _fmt_elapsed
    from llenergymeasure.utils.formatting import truncate_detail as _trunc_detail

    n_valid = len(study_config.experiments) // max(study_config.study_execution.n_cycles, 1)
    n_skipped = len(study_config.skipped_configs) if study_config.skipped_configs else 0
    _step_n = 1
    _step_total = 1 + (1 if n_skipped else 0) + (1 if is_resume else 0)

    detail_done = _trunc_detail(f"{n_valid} valid configs")
    console.print(
        f"   {f'[{_step_n}/{_step_total}]':>7s}  {'Config':<16s} {detail_done:<34s}"
        f"  [bold green]✓[/]  {_fmt_elapsed(expand_elapsed)}"
    )
    _step_n += 1

    if n_skipped:
        detail_skip = _trunc_detail(f"skipped {n_skipped} invalid config(s)")
        console.print(
            f"   {f'[{_step_n}/{_step_total}]':>7s}  {'Config':<16s} {detail_skip:<34s}"
            f"  [bold green]✓[/]  {_fmt_elapsed(expand_elapsed)}"
        )
        _step_n += 1

    # Resume summary: show status counts from manifest
    if is_resume and resume_manifest is not None:
        m = resume_manifest
        parts = []
        if m.completed > 0:
            parts.append(f"{m.completed} completed")
        if m.failed > 0:
            parts.append(f"{m.failed} failed")
        if m.interrupted > 0:
            parts.append(f"{m.interrupted} interrupted")
        if m.skipped > 0:
            parts.append(f"{m.skipped} skipped")
        n_to_run = m.total_experiments - m.completed
        parts.append(f"{n_to_run} to run")
        resume_detail = _trunc_detail(", ".join(parts))
        console.print(
            f"   {f'[{_step_n}/{_step_total}]':>7s}  {'Resume':<16s} {resume_detail:<34s}"
            f"  [bold green]✓[/]  {_fmt_elapsed(0.0)}"
        )


def _build_historical_rows(resume_manifest: Any) -> list[_CompletedRow]:
    """Build completed/failed experiment rows from a resume manifest for pre-population.

    Uses a sequential display index (1, 2, 3...) for historical rows and derives
    elapsed from started/completed timestamps when the stored value is missing.
    """
    from llenergymeasure.cli._step_display import _CompletedRow

    historical: list[_CompletedRow] = []
    hist_idx = 0
    for entry in resume_manifest.experiments:
        if entry.status not in ("completed", "failed"):
            continue
        hist_idx += 1
        elapsed = entry.elapsed_seconds or 0.0
        if elapsed == 0.0 and entry.started_at and entry.completed_at:
            elapsed = (entry.completed_at - entry.started_at).total_seconds()
        historical.append(
            _CompletedRow(
                idx=hist_idx,
                status="OK" if entry.status == "completed" else "FAIL",
                config=entry.config_summary,
                elapsed=elapsed,
                inference_sec=entry.inference_seconds,
                energy_j=entry.energy_joules,
                adj_energy_j=entry.adj_energy_joules,
                throughput=entry.throughput_tok_s,
                mj_per_tok=entry.mj_per_tok,
            )
        )
    return historical


def _run_study_impl(
    config: Path,
    cli_overrides: dict[str, Any],
    cycles: int | None,
    order: str | None,
    no_gaps: bool,
    quiet: bool,
    verbose: bool,
    skip_preflight: bool = False,
    dry_run: bool = False,
    output: str | None = None,
    resume: bool = False,
    resume_dir: Path | None = None,
    fail_fast: bool = False,
    no_circuit_breaker: bool = False,
    timeout: float | None = None,
    no_lock: bool = False,
    no_dedup: bool = False,
) -> None:
    """Study execution path - separated for clean error handling."""
    import yaml

    from llenergymeasure.api import load_study
    from llenergymeasure.cli._display import print_study_dry_run
    from llenergymeasure.cli._preflight_display import build_preflight_panel
    from llenergymeasure.config.grid import count_sweep_structure

    # Fast-fail: verify resume target exists before expensive grid expansion.
    # For resume, also load the manifest early so we can show a summary and
    # pre-populate the completed experiments table.
    resume_dir, _resume_manifest, is_resume = _resolve_resume_target(resume, resume_dir, output)

    # Check what the YAML execution block specifies (to apply CLI effective defaults)
    raw = yaml.safe_load(config.read_text()) or {}
    yaml_execution = raw.get("study_execution", {}) or {}

    # Merge CLI flags (with CLI-layer effective defaults) into the load_study overrides
    study_cli_overrides = _build_study_cli_overrides(
        cli_overrides,
        cycles,
        order,
        no_gaps,
        fail_fast,
        no_circuit_breaker,
        timeout,
        no_dedup,
        yaml_execution,
    )

    # Load study config with overrides - show step-format spinner during expansion
    from rich.console import Console as _ExpandConsole
    from rich.live import Live as _ExpandLive
    from rich.text import Text as _ExpandText

    from llenergymeasure.utils.formatting import format_elapsed as _fmt_elapsed
    from llenergymeasure.utils.formatting import truncate_detail as _trunc_detail

    _SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    _expand_label = "validating config (resume)..." if is_resume else "expanding study config..."

    _expand_console = _ExpandConsole(stderr=True)
    t0_expand = time.perf_counter()

    def _expand_render() -> _ExpandText:
        elapsed = time.perf_counter() - t0_expand
        frame = _SPINNER[int(elapsed * 8) % len(_SPINNER)]
        detail = _trunc_detail(_expand_label)
        line = _ExpandText()
        line.append(f"   {'[1/2]':>7s}  {'Config':<16s} {detail:<34s}")
        line.append(f"  {frame}", style="yellow")
        line.append(f"  {_fmt_elapsed(elapsed)}")
        return line

    class _ExpandRenderable:
        def __rich_console__(self, console: Any, options: Any) -> Any:
            yield _expand_render()

    with _ExpandLive(
        _ExpandRenderable(),
        console=_expand_console,
        refresh_per_second=8,
        transient=True,
    ):
        study_config = load_study(
            config,
            cli_overrides=study_cli_overrides if study_cli_overrides else None,
        )
    expand_elapsed = time.perf_counter() - t0_expand

    # Print completed lines with green ticks (same format as step display)
    _print_config_summary(
        _expand_console, study_config, is_resume, _resume_manifest, expand_elapsed
    )

    # Count sweep axes vs groups and explicit experiments from raw YAML for panel display
    raw_sweep = raw.get("sweep", {}) or {}
    sweep_axes, sweep_groups = count_sweep_structure(raw_sweep)
    n_explicit = len(raw.get("experiments", []) or [])

    # ---------------------------------------------------------------
    # Resolve runners and compute study dir preview - shared by both
    # dry-run and actual-run so both show the same preflight panel.
    # ---------------------------------------------------------------
    from llenergymeasure.api import probe_energy_sampler, run_study_preflight, study_dir_name
    from llenergymeasure.config.user_config import load_user_config

    user_config = load_user_config()
    preresolved: tuple[dict[str, Any], dict[str, dict[str, str]]] | None = None
    try:
        runner_specs, _system_overrides = run_study_preflight(
            study_config,
            # Dry-run: skip Docker binary checks (just resolve runner modes).
            skip_preflight=skip_preflight or dry_run,
            yaml_runners=study_config.runners,
            user_config=user_config.runners,
            yaml_images=study_config.images,
            user_config_images=user_config.images or None,
        )
        preresolved = (runner_specs, _system_overrides)
    except Exception:
        runner_specs = None  # graceful: Docker unavailable, show YAML runners

    study_dir_preview = Path("results") / study_dir_name(study_config.study_name)

    # --- Dry-run branch ---
    if dry_run:
        print_study_dry_run(
            study_config,
            verbose=verbose,
            runner_specs=runner_specs,
            study_dir=study_dir_preview,
            sweep_axes=sweep_axes,
            sweep_groups=sweep_groups,
            n_explicit=n_explicit,
        )
        return

    effective_mode = _resolve_progress_mode(quiet, verbose)

    # Create live study display before the run so per-experiment progress is shown
    study_display = None
    if effective_mode != "quiet":
        from rich.console import Console as RichConsole

        from llenergymeasure.cli._step_display import StudyStepDisplay

        n_exp = len(study_config.experiments)
        n_cycles = study_config.study_execution.n_cycles
        name = study_config.study_name or "unnamed"

        _stderr_console = RichConsole(stderr=True)
        panel = build_preflight_panel(
            study_config,
            runner_specs=runner_specs,
            study_dir=study_dir_preview,
            probed_energy_sampler=probe_energy_sampler(),
            sweep_axes=sweep_axes,
            sweep_groups=sweep_groups,
            n_explicit=n_explicit,
        )
        _stderr_console.print(panel)

        if study_config.skipped_configs:
            n_skip = len(study_config.skipped_configs)
            _stderr_console.print(
                f"Skipped {n_skip} invalid config(s) - details in skipped_configs.log"
            )

        study_display = StudyStepDisplay(
            total_experiments=n_exp,
            study_name=name,
            n_cycles=n_cycles,
            force_plain=effective_mode == "plain",
        )

        # Pre-populate completed experiments from manifest on resume.
        # Uses sequential index (1, 2, 3...) for historical rows.
        if is_resume and _resume_manifest is not None:
            historical = _build_historical_rows(_resume_manifest)
            if historical:
                study_display.add_historical_rows(historical)

        # Header already printed above; start Live without repeating it
        study_display.start(print_header=False)

    # Track elapsed time around the study run
    _study_start = time.monotonic()

    # Run the study with live progress display.
    # skip_preflight=True because we already ran preflight above.
    from llenergymeasure import run_study

    try:
        result = run_study(
            study_config,
            skip_preflight=True,
            progress=study_display,
            resume=resume,
            resume_dir=resume_dir,
            output_dir=Path(output) if output else None,
            no_lock=no_lock,
            config_path=config.resolve(),
            cli_overrides=cli_overrides or None,
            preresolved=preresolved,
        )
    finally:
        # Safety stop - ensures Rich Live is torn down even on exceptions
        if study_display is not None:
            study_display.stop()

    _study_elapsed = time.monotonic() - _study_start

    # Study completion footer
    if study_display is not None:
        save_path = str(result.result_files[0]) if result.result_files else None
        study_display.finish(save_path=save_path, total_elapsed=_study_elapsed)
