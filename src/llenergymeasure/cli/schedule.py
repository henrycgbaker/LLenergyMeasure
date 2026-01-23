"""Scheduled experiment execution commands."""

from __future__ import annotations

import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer

from llenergymeasure.cli.display import console, format_duration
from llenergymeasure.cli.utils import parse_duration
from llenergymeasure.config.loader import load_config
from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.constants import PRESETS


def schedule_experiment_cmd(
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
        lem schedule config.yaml --interval 6h --duration 24h

        # Run daily at 9am for a week
        lem schedule config.yaml --at 09:00 --duration 7d

        # Run at 9am on weekdays only for 2 weeks
        lem schedule config.yaml --at 09:00 --days weekdays --duration 14d

        # Run every 12 hours on weekends
        lem schedule config.yaml --interval 12h --days sat,sun --duration 48h

    Configuration can also be set in YAML:

        schedule_config:
          enabled: true
          interval: "6h"
          at: "09:00"
          days: ["mon", "wed", "fri"]
          total_duration: "7d"
    """
    import schedule as sched

    from llenergymeasure.config.models import DAY_ALIASES, VALID_DAYS

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
        if not model:
            console.print(
                "[red]Error:[/red] --model is required when using --preset without config"
            )
            raise typer.Exit(1)
        preset_config = {**PRESETS[preset], "config_name": f"preset-{preset}", "model_name": model}  # type: ignore[index]
        config = ExperimentConfig(**preset_config)

    # Resolve schedule settings: CLI > config > error
    schedule_cfg = config.schedule
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
        duration_sec = parse_duration(effective_duration)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None

    interval_sec: float | None = None
    if effective_interval:
        try:
            interval_sec = parse_duration(effective_interval)
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
            "lem",
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
                    f"\r[dim]Next run in {format_duration(remaining)} "
                    f"(total: {run_count} runs, {format_duration(end_time - time.time())} remaining)[/dim]",
                    end="",
                )

        time.sleep(10)  # Check every 10 seconds

    # Summary
    console.print("\n\n[bold cyan]━━━ Schedule Complete ━━━[/bold cyan]")
    console.print(f"  [cyan]Total runs:[/cyan] {run_count}")
    console.print(f"  [cyan]Duration:[/cyan] {format_duration(time.time() - start_time)}")

    if stop_requested:
        console.print("  [yellow]Stopped early by user request[/yellow]")

    raise typer.Exit(0)
