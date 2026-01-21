"""Batch experiment execution commands."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

from llm_energy_measure.cli.display import console
from llm_energy_measure.config.loader import load_config
from llm_energy_measure.config.models import ExperimentConfig
from llm_energy_measure.exceptions import ConfigurationError


def batch_run_cmd(
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
