"""Result display functions.

This module provides functions for displaying raw and aggregated
experiment results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.markup import escape as rich_escape
from rich.table import Table

from llenergymeasure.cli.display.console import console
from llenergymeasure.cli.display.summaries import show_effective_config

if TYPE_CHECKING:
    from llenergymeasure.domain.experiment import AggregatedResult, RawProcessResult


def show_raw_result(result: RawProcessResult, show_config: bool = True) -> None:
    """Display a raw process result.

    Args:
        result: The raw process result to display.
        show_config: Whether to show the effective config (default True).
    """
    console.print(f"\n[bold cyan]Process {result.process_index}[/bold cyan] (GPU {result.gpu_id})")
    if result.gpu_name:
        console.print(f"GPU: {result.gpu_name}")
    if result.gpu_is_mig:
        console.print(f"MIG Profile: {result.gpu_mig_profile}")
    if result.energy_measurement_warning:
        console.print(f"[yellow]Warning: {result.energy_measurement_warning}[/yellow]")

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
        show_effective_config(result.effective_config, result.cli_overrides)

    # Show config warnings if present
    if show_config and result.config_warnings:
        console.print("\n[yellow]Config warnings at runtime:[/yellow]")
        for warning in result.config_warnings:
            console.print(f"  [dim]{rich_escape(str(warning))}[/dim]")


def show_aggregated_result(result: AggregatedResult) -> None:
    """Display an aggregated result.

    Args:
        result: The aggregated result to display.
    """
    from llenergymeasure.results.aggregation import calculate_efficiency_metrics

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
        _show_latency_stats(result.latency_stats)

    # Show effective config if available
    if result.effective_config:
        show_effective_config(result.effective_config, result.cli_overrides)

    # Show config warnings if present
    if result.config_warnings:
        console.print("\n[yellow]Config warnings at runtime:[/yellow]")
        for warning in result.config_warnings:
            console.print(f"  [dim]{rich_escape(str(warning))}[/dim]")

    # Aggregation metadata
    meta = result.aggregation
    console.print(f"\n[dim]Aggregation method: {meta.method}[/dim]")
    if meta.temporal_overlap_verified:
        console.print("[green]Temporal overlap verified[/green]")
    if meta.gpu_attribution_verified:
        console.print("[green]GPU attribution verified[/green]")
    for warning in meta.warnings:
        console.print(f"[yellow]Warning: {rich_escape(str(warning))}[/yellow]")


def _show_latency_stats(lat: dict[str, float | int | None] | object) -> None:
    """Display streaming latency statistics.

    Args:
        lat: Latency stats (dict from JSON or LatencyStatistics object).
    """
    from llenergymeasure.domain.metrics import LatencyStatistics

    # Handle both dict (from JSON) and object (LatencyStatistics)
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
    elif isinstance(lat, LatencyStatistics):
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
    else:
        console.print("[yellow]Unknown latency stats format[/yellow]")
        return

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


def show_parameter_provenance(
    provenance: dict[str, dict[str, Any]],
    preset_chain: list[str] | None = None,
) -> None:
    """Display parameter provenance information.

    Shows non-default parameters grouped by their source (preset, config file, CLI).

    Args:
        provenance: Parameter provenance dictionary from results.
        preset_chain: List of presets applied in order.
    """
    from collections import defaultdict

    if not provenance:
        return

    # Group parameters by source
    by_source: dict[str, list[tuple[str, Any, str | None]]] = defaultdict(list)

    for path, info in provenance.items():
        if isinstance(info, dict):
            source = info.get("source", "unknown")
            value = info.get("value")
            detail = info.get("source_detail")

            # Skip pydantic defaults - only show non-default
            if source != "pydantic_default":
                by_source[source].append((path, value, detail))

    if not by_source:
        return

    console.print("\n[bold]Parameter Sources[/bold]")

    # Show preset chain if present
    if preset_chain:
        console.print(f"  [dim]Preset chain: {' -> '.join(preset_chain)}[/dim]")

    # Display parameters grouped by source
    source_display = {
        "preset": ("Preset", "cyan"),
        "config_file": ("Config File", "yellow"),
        "cli": ("CLI Override", "green"),
    }

    for source, params in by_source.items():
        if source in source_display:
            label, colour = source_display[source]
            console.print(f"\n  [{colour}]{label}:[/{colour}]")
            for path, value, detail in sorted(params, key=lambda x: x[0]):
                detail_str = f" [dim]({detail})[/dim]" if detail else ""
                console.print(f"    {path}: {value!r}{detail_str}")
