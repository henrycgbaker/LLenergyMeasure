"""CLI output formatting utilities.

All result output goes to stdout (scientific record).
Progress/header output goes to stderr (transient display area).
"""

from __future__ import annotations

import difflib
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

if TYPE_CHECKING:
    from llenergymeasure.config.runner_spec import RunnerSpec

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.domain.experiment import ExperimentResult
from llenergymeasure.utils.exceptions import DockerError, LLEMError
from llenergymeasure.utils.formatting import format_elapsed as _format_duration
from llenergymeasure.utils.formatting import sig3 as _sig3


def print_result_summary(result: ExperimentResult) -> None:
    """Print grouped result summary to stdout.

    Sections: Energy, Performance, Timing, Warnings.
    Strictly raw metrics only - no derived ratios.
    All numeric values formatted to 3 significant figures.
    """
    # Header
    print(f"Result: {result.experiment_id}")
    print()

    # --- Energy ---
    print("Energy")
    print(f"  Total          {_sig3(result.total_energy_j)} J")
    baseline_power_w = (
        result.energy_breakdown.baseline_power_w if result.energy_breakdown is not None else None
    )
    if baseline_power_w is not None:
        print(f"  Baseline       {_sig3(baseline_power_w)} W")
    if result.energy_adjusted_j is not None:
        print(f"  Adjusted       {_sig3(result.energy_adjusted_j)} J")
    # Per-token energy (mJ/tok) - prefer adjusted, fall back to total, no recomputation
    if result.energy_per_token_mj_adjusted is not None:
        print(f"  Per token      {_sig3(result.energy_per_token_mj_adjusted)} mJ/tok (adjusted)")
    elif result.energy_per_token_mj_total is not None:
        print(f"  Per token      {_sig3(result.energy_per_token_mj_total)} mJ/tok")
    print()

    # --- Performance ---
    print("Performance")
    print(f"  Throughput     {_sig3(result.avg_tokens_per_second)} tok/s")

    if result.total_flops > 0:
        flops_val = f"{result.total_flops:.2e}"
        print(f"  FLOPs          {flops_val}")

    if result.latency_stats is not None:
        ls = result.latency_stats
        if ls.ttft_mean_ms is not None:
            print(f"  Latency TTFT   {_sig3(ls.ttft_mean_ms)} ms")
        if ls.itl_mean_ms is not None:
            print(f"  Latency ITL    {_sig3(ls.itl_mean_ms)} ms")
    print()

    # --- Timing ---
    print("Timing")
    print(f"  Meas. window   {_format_duration(result.duration_sec)}")
    if result.warmup_excluded_samples is not None:
        print(f"  Warmup         {result.warmup_excluded_samples} prompts excluded")
    print()

    # --- Warnings ---
    if result.measurement_warnings:
        print("Warnings")
        for warning in result.measurement_warnings:
            print(f"  {warning}")
        print()


def _print_vram_estimate(
    vram: dict[str, float] | None,
    gpu_vram_gb: float | None,
    dtype: str | None,
    *,
    label: str = "VRAM estimate",
) -> None:
    """Print a VRAM-estimate block to stdout.

    Shared by the single-experiment and study dry-run renderers. Prints
    "(unavailable)" when *vram* is None; otherwise the weights/KV/overhead
    breakdown plus a total line annotated with GPU capacity when known.
    """
    print(label)
    if vram is None:
        print("  (unavailable)")
    else:
        print(f"  Weights        {_sig3(vram['weights_gb'])} GB ({dtype or '-'})")
        print(f"  KV cache       {_sig3(vram['kv_cache_gb'])} GB")
        print(f"  Overhead       {_sig3(vram['overhead_gb'])} GB")
        total_line = f"  Total          ~{_sig3(vram['total_gb'])} GB"
        if gpu_vram_gb is not None:
            fits = vram["total_gb"] <= gpu_vram_gb
            status = "OK" if fits else "WARNING: may not fit"
            total_line += f" / {_sig3(gpu_vram_gb)} GB available   {status}"
        print(total_line)
    print()


def print_dry_run(
    config: ExperimentConfig,
    vram: dict[str, float] | None,
    gpu_vram_gb: float | None,
    verbose: bool = False,
    output_dir: str | None = None,
) -> None:
    """Print dry-run output to stdout.

    Shows resolved config and VRAM estimate.
    With verbose=True, adds source annotations.
    """
    # Determine non-default fields for annotations
    defaults = {
        "engine": "transformers",
        "dtype": None,
    }

    def _annotate(field: str, value: object) -> str:
        """Return a "(default)" annotation when value equals the field default."""
        if not verbose:
            return ""
        default = defaults.get(field)
        if value == default:
            return f" ({field} default)" if field not in ("engine", "dtype") else " (default)"
        return ""

    engine_params = config.active_engine_params()
    engine_dtype = getattr(engine_params, "dtype", None)

    print("Config (resolved)")
    print(f"  Model          {config.task.model}")
    print(f"  Engine         {config.engine}{_annotate('engine', config.engine)}")
    dtype_display = engine_dtype or "-"
    print(f"  Dtype          {dtype_display}{_annotate('dtype', engine_dtype)}")

    # Batch size - transformers llem-owned execution knob (llem_execution), if present
    batch_size: int | None = None
    execution = config.active_llem_execution()
    if execution is not None:
        batch_size = execution.batch_size
    if batch_size is not None:
        print(f"  Batch size     {batch_size}")

    # Dataset display
    ds = config.task.dataset
    dataset_str = f"{ds.source} ({ds.n_prompts} prompts)"
    print(f"  Dataset        {dataset_str}")

    output_display = output_dir or "results/ (default)"
    print(f"  Output         {output_display}")
    print()

    _print_vram_estimate(vram, gpu_vram_gb, engine_dtype)

    print("Config valid. Run without --dry-run to start.")


def format_error(error: LLEMError, verbose: bool = False) -> str:
    """Format an LLEMError for stderr output.

    With verbose=True, includes full traceback.
    Otherwise, just the error class name and message.

    For DockerError subclasses, appends fix_suggestion and stderr_snippet
    so the user sees actionable guidance without needing to dig into logs.

    Under verbose, a Docker container failure surfaces the traceback the
    container entrypoint captured (``error.error_payload["traceback"]``) - the
    real engine/CUDA failure inside the container - in preference to the
    host-side traceback, which would only show the DockerRunner's own raise
    site and not the actual cause.
    """
    class_name = type(error).__name__
    message = f"{class_name}: {error}"

    # A Docker container failure carries the real in-container traceback in its
    # error payload (written to ``*_error.json`` by the container entrypoint).
    # Prefer it over the uninformative host-side traceback of the raise site.
    container_tb: str | None = None
    if isinstance(error, DockerError) and error.error_payload:
        container_tb = error.error_payload.get("traceback")

    if verbose:
        if container_tb:
            message = f"In-container traceback (real failure cause):\n{container_tb}\n{message}"
        else:
            tb = traceback.format_exc()
            if tb and tb.strip() != "NoneType: None":
                message = f"{tb}\n{message}"

    # Append Docker-specific details when available
    if isinstance(error, DockerError):
        if error.fix_suggestion:
            message += f"\n\nSuggestion: {error.fix_suggestion}"
        if error.stderr_snippet:
            message += f"\n\nContainer stderr (last 20 lines):\n{error.stderr_snippet}"

    return message


def format_validation_error(e: ValidationError) -> str:
    """Format a Pydantic ValidationError with a friendly header.

    Includes did-you-mean suggestions for literal_error types.
    Does NOT catch or re-wrap the error - only formats it.
    """
    from llenergymeasure.config.ssot import ENGINES

    errors = e.errors()
    n = len(errors)
    header = f"Config validation failed ({n} error{'s' if n > 1 else ''}):"
    lines = [header]

    # Build a set of valid values for did-you-mean suggestions
    valid_engines: list[str] = [str(e) for e in ENGINES]
    valid_dtypes = list({d for descriptor in ENGINES.values() for d in descriptor.dtypes})

    for err in errors:
        loc_parts = [str(part) for part in err.get("loc", [])]
        loc_str = " -> ".join(loc_parts) if loc_parts else "(root)"
        msg = err.get("msg", "")
        lines.append(f"  {loc_str}: {msg}")

        # Did-you-mean for literal errors on known enum fields
        if err.get("type") == "literal_error":
            # Try to extract the bad value from the error input
            bad_value = err.get("input")
            if bad_value is not None and isinstance(bad_value, str):
                # Determine which pool to search based on location
                last_loc = loc_parts[-1] if loc_parts else ""
                if last_loc == "engine":
                    pool = valid_engines
                elif last_loc == "dtype":
                    pool = valid_dtypes
                else:
                    pool = valid_engines + valid_dtypes

                suggestions = difflib.get_close_matches(bad_value, pool, n=3, cutoff=0.6)
                if suggestions:
                    lines.append(f"    Did you mean: {', '.join(suggestions)}?")
                    lines.append(f"    Valid values: {', '.join(pool)}")

    return "\n".join(lines)


def print_study_dry_run(
    study_config: object,
    verbose: bool = False,
    runner_specs: dict[str, RunnerSpec] | None = None,
    study_dir: Path | None = None,
    sweep_axes: int | None = None,
    sweep_groups: int | None = None,
    n_explicit: int = 0,
) -> None:
    """Print dry-run output for a study to stdout.

    Shows grid summary, per-experiment configs, and VRAM estimate for the
    largest model. Mirrors the single-experiment dry-run format.
    """
    from rich.console import Console as RichConsole

    from llenergymeasure.api import probe_energy_sampler
    from llenergymeasure.cli._preflight_display import build_preflight_panel
    from llenergymeasure.cli._vram import estimate_vram, get_gpu_vram_gb
    from llenergymeasure.config.models import StudyConfig
    from llenergymeasure.utils.formatting import format_experiment_header

    assert isinstance(study_config, StudyConfig)

    # Pre-flight panel - same args as actual run so both show identical output
    _stdout_console = RichConsole()
    panel = build_preflight_panel(
        study_config,
        runner_specs=runner_specs,
        study_dir=study_dir,
        probed_energy_sampler=probe_energy_sampler(),
        sweep_axes=sweep_axes,
        sweep_groups=sweep_groups,
        n_explicit=n_explicit,
    )
    _stdout_console.print(panel)

    if study_config.skipped_configs:
        n_skip = len(study_config.skipped_configs)
        _stdout_console.print(
            f"Skipped {n_skip} invalid config(s) - details in skipped_configs.log"
        )
        _stdout_console.print()

    # Per-experiment list using the same header format as the live run
    n = len(study_config.experiments)
    width = len(str(n))
    for i, exp in enumerate(study_config.experiments, 1):
        print(f"  {i:>{width}}  {format_experiment_header(exp)}")
    print()

    # VRAM estimate for the peak model (largest weight estimate).
    # Memoize on (model, dtype) so a sweep of N experiments over one model does
    # not make N identical HuggingFace Hub round-trips - only one per unique key.
    gpu_vram_gb = get_gpu_vram_gb()
    peak_vram: dict[str, float] | None = None
    peak_config: ExperimentConfig | None = None
    vram_cache: dict[tuple[str, str | None], dict[str, float] | None] = {}
    for exp in study_config.experiments:
        exp_engine_params = exp.active_engine_params()
        cache_key = (exp.task.model, getattr(exp_engine_params, "dtype", None))
        if cache_key not in vram_cache:
            vram_cache[cache_key] = estimate_vram(exp)
        vram = vram_cache[cache_key]
        if vram is not None and (peak_vram is None or vram["total_gb"] > peak_vram["total_gb"]):
            peak_vram = vram
            peak_config = exp

    peak_dtype: str | None = None
    if peak_config is not None:
        peak_dtype = getattr(peak_config.active_engine_params(), "dtype", None)
    _print_vram_estimate(peak_vram, gpu_vram_gb, peak_dtype, label="VRAM estimate (peak)")

    print("Config valid. Run without --dry-run to start.")
