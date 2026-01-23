"""Results inspection and aggregation commands.

This module contains CLI commands for listing, showing, and aggregating
experiment results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from llenergymeasure.cli.display import (
    console,
    show_aggregated_result,
    show_raw_result,
)
from llenergymeasure.exceptions import AggregationError
from llenergymeasure.results.aggregation import aggregate_results, calculate_efficiency_metrics
from llenergymeasure.results.repository import FileSystemRepository

results_app = typer.Typer(help="Results inspection commands", invoke_without_command=True)


@results_app.callback()  # type: ignore[misc]
def results_callback(ctx: typer.Context) -> None:
    """Results inspection commands."""
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


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
            data = [r.model_dump(mode="json") for r in raw_results]
            console.print_json(json.dumps(data, default=str))
        else:
            for i, result in enumerate(raw_results):
                # Show config only for first result to avoid repetition
                show_raw_result(result, show_config=(i == 0))
    else:
        aggregated = repo.load_aggregated(experiment_id)
        if not aggregated:
            console.print(f"[yellow]No aggregated result for:[/yellow] {experiment_id}")
            console.print("Run 'lem aggregate' first, or use --raw")
            raise typer.Exit(1)

        if json_output:
            console.print_json(aggregated.model_dump_json())
        else:
            show_aggregated_result(aggregated)


def aggregate_one(
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
