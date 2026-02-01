"""Campaign CLI command for multi-config comparison experiments.

Provides the `campaign` command for running multiple experiment configs
across multiple cycles for statistical robustness and fair comparison.
"""

from __future__ import annotations

import glob as glob_module
import os
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

import typer
import yaml

from llenergymeasure.cli.display import console
from llenergymeasure.config.campaign_config import (
    CampaignConfig,
    CampaignExecutionConfig,
)
from llenergymeasure.orchestration.campaign import (
    CampaignExperiment,
    CampaignRunner,
    format_gap_time,
)

if TYPE_CHECKING:
    from llenergymeasure.orchestration.grid import GridExpansionResult
    from llenergymeasure.orchestration.manifest import CampaignManifest


def campaign_cmd(
    config_paths: Annotated[
        list[Path],
        typer.Argument(
            help="Campaign YAML file OR multiple experiment config files (glob patterns supported)"
        ),
    ],
    # Campaign identity
    campaign_name: Annotated[
        str | None,
        typer.Option(
            "--campaign-name",
            help="Campaign name (required when using multiple config files)",
        ),
    ] = None,
    # Prompt source (overrides individual configs)
    dataset: Annotated[
        str | None,
        typer.Option("--dataset", "-d", help="HuggingFace dataset for all experiments"),
    ] = None,
    sample_size: Annotated[
        int | None,
        typer.Option("--sample-size", "-n", help="Number of prompts for all experiments"),
    ] = None,
    # Execution parameters
    cycles: Annotated[
        int | None,
        typer.Option("--cycles", "-c", help="Number of cycles (default: 3)"),
    ] = None,
    structure: Annotated[
        str | None,
        typer.Option(
            "--structure",
            help="Execution order: interleaved, shuffled, or grouped",
        ),
    ] = None,
    # Warmup parameters
    warmup_prompts: Annotated[
        int | None,
        typer.Option("--warmup-prompts", help="Min warmup prompts per config (default: 5)"),
    ] = None,
    warmup_timeout: Annotated[
        float | None,
        typer.Option("--warmup-timeout", help="Max warmup time in seconds (default: 30)"),
    ] = None,
    # Gap parameters
    config_gap: Annotated[
        float | None,
        typer.Option("--config-gap", help="Gap between configs in seconds (default: 60)"),
    ] = None,
    cycle_gap: Annotated[
        float | None,
        typer.Option("--cycle-gap", help="Gap between cycles in seconds (default: 300)"),
    ] = None,
    # Seed for reproducibility
    seed: Annotated[
        int | None,
        typer.Option("--seed", help="Random seed for shuffled structure"),
    ] = None,
    # Output control
    results_dir: Annotated[
        Path | None,
        typer.Option("--results-dir", "-o", help="Results output directory"),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show execution plan without running"),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation prompts"),
    ] = False,
    # Phase 2 options
    resume: Annotated[
        bool,
        typer.Option("--resume/--no-resume", help="Resume from existing manifest"),
    ] = False,
    force_cold_start: Annotated[
        bool | None,
        typer.Option(
            "--force-cold-start/--no-force-cold-start",
            help="Override cold start config",
        ),
    ] = None,
    validate_only: Annotated[
        bool,
        typer.Option("--validate-only", help="Validate grid and exit without running"),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Suppress interactive output (for daemon mode)"),
    ] = False,
) -> None:
    """Run a multi-config campaign for statistical comparison.

    Campaigns run multiple experiment configurations across multiple cycles
    to enable fair comparison with statistical robustness.

    Two modes of operation:

    1. Campaign YAML file (recommended for reproducibility):

       lem campaign campaign.yaml

    2. Multiple experiment configs with CLI options:

       lem campaign configs/*.yaml \\
         --campaign-name "pytorch-vs-vllm" \\
         --cycles 5 --dataset alpaca -n 100

    Execution structures:

    - interleaved: A->B->C, A->B->C, A->B->C (fair comparison, fixed order)
    - shuffled: Random order within each cycle (eliminates ordering bias)
    - grouped: Axn, Bxn, Cxn (all cycles of one config before next)

    Warmup runs actual dataset prompts before measurement to prime:
    - Model weights in GPU memory
    - KV cache patterns
    - CUDA kernels

    Gaps allow GPU thermal recovery between experiments.
    """

    # Validate structure parameter if provided
    valid_structures = ("interleaved", "shuffled", "grouped")
    if structure is not None and structure not in valid_structures:
        console.print(f"[red]Error:[/red] Invalid structure '{structure}'")
        console.print(f"Valid options: {', '.join(valid_structures)}")
        raise typer.Exit(1)

    # Cast to Literal for type safety (validated above)
    structure_typed: Literal["interleaved", "shuffled", "grouped"] | None = structure  # type: ignore[assignment]

    # Expand glob patterns
    expanded_paths: list[Path] = []
    for path in config_paths:
        path_str = str(path)
        if "*" in path_str or "?" in path_str:
            # Glob pattern
            matches = glob_module.glob(path_str)
            expanded_paths.extend(Path(m) for m in sorted(matches))
        else:
            expanded_paths.append(path)

    if not expanded_paths:
        console.print("[red]Error:[/red] No config files found")
        raise typer.Exit(1)

    # Check that all config files exist before proceeding
    missing = [p for p in expanded_paths if not p.exists()]
    if missing:
        for m in missing:
            console.print(f"[red]Error:[/red] Config file not found: {m}")
        raise typer.Exit(1)

    # Determine if this is a campaign YAML or multiple experiment configs
    campaign: CampaignConfig

    if len(expanded_paths) == 1 and _is_campaign_yaml(expanded_paths[0]):
        # Single campaign YAML file
        campaign = _load_campaign_yaml(expanded_paths[0])

        # Apply CLI overrides
        campaign = _apply_cli_overrides(
            campaign,
            campaign_name=campaign_name,
            dataset=dataset,
            num_samples=sample_size,
            cycles=cycles,
            structure=structure_typed,
            warmup_prompts=warmup_prompts,
            warmup_timeout=warmup_timeout,
            config_gap=config_gap,
            cycle_gap=cycle_gap,
        )
    else:
        # Multiple experiment config files
        if not campaign_name:
            console.print(
                "[red]Error:[/red] --campaign-name is required when using multiple config files"
            )
            raise typer.Exit(1)

        # Build campaign config from CLI args
        execution = CampaignExecutionConfig(
            cycles=cycles or 3,
            structure=structure_typed or "interleaved",
            warmup_prompts=warmup_prompts if warmup_prompts is not None else 5,
            warmup_timeout_seconds=warmup_timeout if warmup_timeout is not None else 30.0,
            config_gap_seconds=config_gap if config_gap is not None else 60.0,
            cycle_gap_seconds=cycle_gap if cycle_gap is not None else 300.0,
        )

        campaign = CampaignConfig(
            campaign_name=campaign_name,
            dataset=dataset,
            num_samples=sample_size,
            configs=[str(p) for p in expanded_paths],
            execution=execution,
        )

    # Apply cold start override if provided
    if force_cold_start is not None:
        cold_start_update = campaign.cold_start.model_copy(
            update={"force_cold_start": force_cold_start}
        )
        campaign = campaign.model_copy(update={"cold_start": cold_start_update})

    # --- Early resume check ---
    # Check for existing manifest before doing grid expansion / plan display
    _manifest_path = Path(campaign.io.state_dir) / campaign.io.manifest_filename
    if not resume and _manifest_path.exists() and not dry_run:
        from llenergymeasure.orchestration.manifest import ManifestManager

        _mgr = ManifestManager(_manifest_path)
        _existing = _mgr.load()
        _remaining = _existing.get_remaining() if _existing is not None else []
        if len(_remaining) > 0:
            assert _existing is not None  # guaranteed by _remaining having items
            n_done = _existing.completed_count
            n_remaining = len(_remaining)
            console.print(
                f"\n[yellow]Previous campaign state found:[/yellow] "
                f"{n_done} completed, {n_remaining} remaining"
            )
            if not yes:
                from rich.prompt import Confirm

                resume = Confirm.ask("Resume previous campaign?", default=True)
            else:
                # With --yes, auto-resume
                resume = True
                console.print("[dim]Auto-resuming (--yes flag)[/dim]")

            if not resume:
                # User chose not to resume — delete stale manifest
                _manifest_path.unlink(missing_ok=True)
                console.print("[dim]Cleared previous state, starting fresh[/dim]")
    del _manifest_path  # Clean up temp variable

    # Grid expansion and validation (if grid-based campaign)
    if campaign.grid is not None:
        from llenergymeasure.orchestration.grid import (
            expand_campaign_grid,
            validate_campaign_grid,
        )

        # Build base_config from campaign-level defaults
        base_config: dict[str, Any] = {}
        if campaign.model:
            base_config["model_name"] = campaign.model
        config_dicts = expand_campaign_grid(campaign.grid, base_config=base_config)
        grid_result = validate_campaign_grid(config_dicts)
        if not resume:
            _display_validation_summary(grid_result)

        if validate_only:
            raise typer.Exit(0)

        if not grid_result.valid_configs:
            console.print("[red]Error:[/red] No valid configs after grid expansion")
            raise typer.Exit(1)

    elif validate_only:
        console.print("[yellow]--validate-only requires a grid-based campaign YAML[/yellow]")
        raise typer.Exit(1)

    # Display campaign summary (abbreviated when resuming)
    if resume:
        _display_resume_summary(campaign)
    else:
        _display_campaign_summary(campaign, seed)

    # Create runner
    runner = CampaignRunner(campaign, seed=seed)

    # Use grid-based execution order if grid is defined, else config-list path
    if campaign.grid is not None:
        execution_order = runner.generate_execution_order_from_grid()
    else:
        execution_order = runner.generate_execution_order()

    # --- Manifest setup (before plan display so resume can filter) ---
    from llenergymeasure.orchestration.manifest import ManifestManager

    manifest_path = Path(campaign.io.state_dir) / campaign.io.manifest_filename
    manifest_mgr = ManifestManager(manifest_path)
    manifest: CampaignManifest

    if resume and manifest_mgr.exists():
        _loaded = manifest_mgr.load()
        if _loaded is None:
            console.print("[red]Error:[/red] Failed to load manifest for resume")
            raise typer.Exit(1)
        manifest = _loaded
        _link_manifest_entries(execution_order, manifest)
        execution_order = runner.apply_resume_filter(manifest, execution_order)
        n_completed = manifest.completed_count
        n_failed = manifest.failed_count
        n_remaining = len(execution_order)
        console.print(
            f"\n[cyan]Resuming:[/cyan] {n_remaining} remaining "
            f"({n_completed} completed, {n_failed} failed)"
        )
    else:
        manifest = runner.create_manifest(execution_order)
        manifest_mgr.save(manifest)

    # Display execution plan (only remaining experiments when resuming)
    plan_title = "Remaining Experiments" if resume else "Execution Plan"
    _display_execution_plan(runner, execution_order, title=plan_title)

    if dry_run:
        console.print("\n[cyan]--dry-run: Exiting without running campaign[/cyan]")
        raise typer.Exit(0)

    # Confirm execution (skip when resuming — user already confirmed at early check)
    if not resume and not yes and not quiet:
        from rich.prompt import Confirm

        total_time = _estimate_total_time(campaign, runner)
        console.print(f"\n[dim]Estimated minimum time: {format_gap_time(total_time)}[/dim]")

        if not Confirm.ask("\nStart campaign?", default=True):
            console.print("[dim]Aborted[/dim]")
            raise typer.Abort()

    # --- Daemon scheduling ---
    daemon_cfg = campaign.daemon
    if daemon_cfg is not None and daemon_cfg.enabled:
        # Auto-enable quiet in daemon mode
        quiet = True

        # Wait until scheduled start time if configured
        if daemon_cfg.at:
            _wait_until_time(daemon_cfg.at)

        # Daemon loop: repeat campaign at intervals until total_duration
        daemon_start = time.time()
        completed_daemon_cycles = 0

        while True:
            # Run the campaign (execution code below)
            _run_campaign_loop(
                campaign=campaign,
                runner=runner,
                execution_order=execution_order,
                resume=resume,
                dataset=dataset,
                sample_size=sample_size,
                results_dir=results_dir,
                quiet=quiet,
            )
            completed_daemon_cycles += 1

            # Check total duration
            total_dur = daemon_cfg.total_duration_seconds
            if total_dur is not None:
                elapsed = time.time() - daemon_start
                if elapsed >= total_dur:
                    console.print(
                        f"\n[cyan]Daemon:[/cyan] total duration "
                        f"{daemon_cfg.total_duration} reached after "
                        f"{completed_daemon_cycles} campaign cycles"
                    )
                    break

            # Inter-cycle interval
            interval = daemon_cfg.interval_seconds
            if interval is None:
                break  # No interval = single run in daemon mode
            console.print(
                f"[dim]Daemon: sleeping {daemon_cfg.interval} before next campaign cycle[/dim]"
            )
            time.sleep(interval)

        return  # Daemon mode completes here

    # --- Docker dispatch setup ---
    backends_needed = list({exp.backend for exp in execution_order})
    if _should_use_docker(backends_needed):
        # Ensure .env exists before any Docker compose operation
        from llenergymeasure.config.env_setup import ensure_env_file

        ensure_env_file()

        # Check images are built
        missing_images = _check_docker_images(backends_needed)
        if missing_images:
            _prompt_docker_build(missing_images)

        # Docker dispatch uses `docker compose run --rm` per experiment
        # (containers are task runners, not long-running services)
        console.print(
            f"[dim]Docker dispatch: {', '.join(backends_needed)} "
            f"(docker compose run --rm)[/dim]"
        )

    # Execute campaign
    use_docker_dispatch = _should_use_docker(backends_needed)
    console.print("\n[bold cyan]━━━ Starting Campaign ━━━[/bold cyan]\n")
    console.print(f"Campaign ID: [cyan]{campaign.campaign_id}[/cyan]")
    console.print(f"Campaign Name: [cyan]{campaign.campaign_name}[/cyan]")
    if use_docker_dispatch:
        console.print("Execution: [green]Docker[/green] (per-experiment containers)")
    else:
        console.print("Execution: [blue]Local[/blue] (direct process)")
    console.print()

    failed_experiments: list[tuple[str, int]] = []

    for idx, experiment in enumerate(execution_order):
        cycle_num = experiment.cycle_index + 1
        config_name = experiment.config_name

        console.print(
            f"[bold]Experiment {idx + 1}/{len(execution_order)}:[/bold] "
            f"{config_name} (cycle {cycle_num}/{runner.num_cycles})"
        )

        # Config gap (thermal recovery between experiments)
        if runner.should_wait_config_gap() and idx > 0:
            gap = campaign.execution.config_gap_seconds
            if gap > 0:
                console.print(
                    f"  [dim]Waiting {format_gap_time(gap)} for GPU thermal recovery...[/dim]"
                )
                time.sleep(gap)

        # Update manifest: running
        if experiment.manifest_entry:
            from datetime import datetime

            manifest.update_entry(
                experiment.manifest_entry.exp_id,
                status="running",
                started_at=datetime.now(),
            )
            manifest_mgr.save(manifest)

        # Run the experiment
        runner.record_experiment_start(experiment)
        experiment_id, exit_code = _run_single_experiment(
            experiment=experiment,
            campaign=campaign,
            dataset=dataset,
            sample_size=sample_size,
            results_dir=results_dir,
            use_docker=use_docker_dispatch,
        )

        # Check for interrupt signal (Ctrl+C) - abort entire campaign
        if exit_code == 130:
            console.print("\n[yellow]Campaign interrupted by user (SIGINT)[/yellow]")
            if experiment.manifest_entry:
                from datetime import datetime

                manifest.update_entry(
                    experiment.manifest_entry.exp_id,
                    status="failed",
                    completed_at=datetime.now(),
                    error="Interrupted by user (SIGINT)",
                )
                manifest_mgr.save(manifest)
            raise typer.Exit(130)

        if exit_code == 0:
            console.print(f"  [green]✓[/green] Completed (experiment_id: {experiment_id})")
            runner.record_experiment_complete(experiment, experiment_id)
            if experiment.manifest_entry:
                from datetime import datetime

                result_path = str(
                    Path(campaign.io.results_dir) / "aggregated" / f"{experiment_id}.json"
                )
                manifest.update_entry(
                    experiment.manifest_entry.exp_id,
                    status="completed",
                    completed_at=datetime.now(),
                    result_path=result_path,
                )
                manifest_mgr.save(manifest)
        else:
            console.print(f"  [red]✗[/red] Failed (exit code: {exit_code})")
            failed_experiments.append((config_name, exit_code))
            runner.record_experiment_complete(experiment, experiment_id)
            if experiment.manifest_entry:
                from datetime import datetime

                manifest.update_entry(
                    experiment.manifest_entry.exp_id,
                    status="failed",
                    completed_at=datetime.now(),
                    error=f"Exit code {exit_code}",
                )
                manifest_mgr.save(manifest)

        # Cycle gap (full thermal reset between cycles)
        if runner.is_cycle_complete(idx) and idx < len(execution_order) - 1:
            gap = campaign.execution.cycle_gap_seconds
            console.print(f"\n  [cyan]Cycle {cycle_num} complete.[/cyan]")
            if gap > 0:
                console.print(f"  [dim]Waiting {format_gap_time(gap)} before next cycle...[/dim]\n")
                time.sleep(gap)

    # Campaign summary
    console.print("\n[bold green]━━━ Campaign Complete ━━━[/bold green]")

    if failed_experiments:
        console.print(f"\n[yellow]Warning: {len(failed_experiments)} experiments failed[/yellow]")

    console.print("\nResults by config:")
    for config_name, experiments in runner.get_experiments_by_config().items():
        experiment_ids = runner.get_completed_experiment_ids(config_name)
        console.print(f"  {config_name}: {len(experiments)} experiments")
        console.print(f"    IDs: {', '.join(experiment_ids)}")

    # Bootstrap CI display for multi-cycle campaigns
    if runner.num_cycles > 1:
        _display_campaign_ci_summary(manifest, campaign)
    elif runner.num_cycles == 1:
        console.print("\n[dim]Single cycle: use --cycles 3+ for confidence intervals[/dim]")

    if failed_experiments:
        raise typer.Exit(1)


def _wait_until_time(target_time: str) -> None:
    """Wait until a target time of day (HH:MM format).

    If the target time has already passed today, waits until tomorrow.
    """
    from datetime import datetime, timedelta

    hours, minutes = map(int, target_time.split(":"))
    now = datetime.now()
    target = now.replace(hour=hours, minute=minutes, second=0, microsecond=0)

    if target <= now:
        target += timedelta(days=1)

    delay = (target - now).total_seconds()
    console.print(
        f"[dim]Daemon: waiting until {target_time} to start campaign "
        f"({format_gap_time(delay)} from now)[/dim]"
    )
    time.sleep(delay)


def _run_campaign_loop(
    campaign: CampaignConfig,
    runner: CampaignRunner,
    execution_order: list[CampaignExperiment],
    resume: bool,
    dataset: str | None,
    sample_size: int | None,
    results_dir: Path | None,
    quiet: bool,
) -> None:
    """Execute one full campaign cycle (used by daemon mode).

    Encapsulates manifest setup, container lifecycle, experiment execution,
    health checks, cold start, and CI display.
    """
    from llenergymeasure.orchestration.manifest import ManifestManager

    manifest_path = Path(campaign.io.state_dir) / campaign.io.manifest_filename
    manifest_mgr = ManifestManager(manifest_path)

    if resume and manifest_mgr.exists():
        manifest = manifest_mgr.load()
        if manifest is None:
            console.print("[red]Error:[/red] Corrupt manifest during daemon cycle")
            return
        _link_manifest_entries(execution_order, manifest)
        execution_order = runner.apply_resume_filter(manifest, execution_order)
    else:
        manifest = runner.create_manifest(execution_order)
        manifest_mgr.save(manifest)

    backends_needed = list({exp.backend for exp in execution_order})
    use_docker_dispatch = _should_use_docker(backends_needed)
    if use_docker_dispatch:
        from llenergymeasure.config.env_setup import ensure_env_file

        ensure_env_file()

        missing_images = _check_docker_images(backends_needed)
        if missing_images:
            _prompt_docker_build(missing_images)

    if not quiet:
        console.print("\n[bold cyan]--- Daemon Campaign Cycle ---[/bold cyan]\n")

    failed: list[tuple[str, int]] = []
    for idx, experiment in enumerate(execution_order):
        if experiment.manifest_entry:
            from datetime import datetime

            manifest.update_entry(
                experiment.manifest_entry.exp_id,
                status="running",
                started_at=datetime.now(),
            )
            manifest_mgr.save(manifest)

        runner.record_experiment_start(experiment)
        experiment_id, exit_code = _run_single_experiment(
            experiment=experiment,
            campaign=campaign,
            dataset=dataset,
            sample_size=sample_size,
            results_dir=results_dir,
            use_docker=use_docker_dispatch,
        )

        # Check for interrupt signal - abort daemon cycle
        if exit_code == 130:
            if experiment.manifest_entry:
                from datetime import datetime

                manifest.update_entry(
                    experiment.manifest_entry.exp_id,
                    status="failed",
                    completed_at=datetime.now(),
                    error="Interrupted by user (SIGINT)",
                )
                manifest_mgr.save(manifest)
            return  # Exit daemon cycle gracefully

        if exit_code == 0:
            runner.record_experiment_complete(experiment, experiment_id)
            if experiment.manifest_entry:
                from datetime import datetime

                result_path = str(
                    Path(campaign.io.results_dir) / "aggregated" / f"{experiment_id}.json"
                )
                manifest.update_entry(
                    experiment.manifest_entry.exp_id,
                    status="completed",
                    completed_at=datetime.now(),
                    result_path=result_path,
                )
                manifest_mgr.save(manifest)
        else:
            failed.append((experiment.config_name, exit_code))
            runner.record_experiment_complete(experiment, experiment_id)
            if experiment.manifest_entry:
                from datetime import datetime

                manifest.update_entry(
                    experiment.manifest_entry.exp_id,
                    status="failed",
                    completed_at=datetime.now(),
                    error=f"Exit code {exit_code}",
                )
                manifest_mgr.save(manifest)

        # Cycle gap
        if runner.is_cycle_complete(idx) and idx < len(execution_order) - 1:
            gap = campaign.execution.cycle_gap_seconds
            if gap > 0:
                time.sleep(gap)

    if runner.num_cycles > 1:
        _display_campaign_ci_summary(manifest, campaign)


def _run_single_experiment(
    experiment: CampaignExperiment,
    campaign: CampaignConfig,
    dataset: str | None,
    sample_size: int | None,
    results_dir: Path | None,
    use_docker: bool | None = None,
) -> tuple[str, int]:
    """Run a single experiment via Docker compose run --rm or local subprocess.

    Args:
        experiment: The campaign experiment to run.
        campaign: Parent campaign configuration.
        dataset: Dataset override (None = use config).
        sample_size: Sample size override (None = use config).
        results_dir: Results directory override.
        use_docker: Campaign-level Docker decision. When None, auto-detects per-experiment.

    Returns:
        Tuple of (experiment_id, exit_code).
    """
    import tempfile

    from llenergymeasure.core.distributed import get_persistent_unique_id

    # Generate experiment ID (or use manifest entry ID if available)
    if experiment.manifest_entry:
        experiment_id = experiment.manifest_entry.exp_id
    else:
        experiment_id = get_persistent_unique_id()

    # Build config data — from YAML file or from Pydantic model (grid-based)
    if str(experiment.config_path).startswith("<grid:"):
        config_data = experiment.config.model_dump(mode="json")
    else:
        config_data = yaml.safe_load(experiment.config_path.read_text())

    config_data["_metadata"] = {
        "experiment_id": experiment_id,
        "campaign_name": campaign.campaign_name,
        "campaign_id": campaign.campaign_id,
        "cycle_id": experiment.cycle_index,
        "config_name": experiment.config_name,
    }

    # Detect backend from experiment config
    backend = _detect_backend(config_data)
    console.print(f"  [dim]Backend: {backend}[/dim]")

    # Determine execution mode (campaign-level decision takes precedence)
    if use_docker is None:
        use_docker = _should_use_docker([backend])

    # Write temp config file
    if use_docker:
        # Write to configs/ directory (bind-mounted into container as /app/configs/)
        tmp_dir = Path("configs")
        tmp_dir.mkdir(exist_ok=True)
        tmp_config_path = tmp_dir / f"_campaign_{experiment_id}.yaml"
        with open(tmp_config_path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)
        container_config_path = f"/app/configs/_campaign_{experiment_id}.yaml"
    else:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, prefix="campaign_"
        ) as tmp:
            yaml.dump(config_data, tmp, default_flow_style=False)
            tmp_config_path = Path(tmp.name)
            container_config_path = str(tmp_config_path)

    try:
        console.print(f"  [dim]Running: {experiment.config_name}[/dim]")

        if use_docker:
            cmd = _build_docker_command(
                backend=backend,
                config_path=container_config_path,
                dataset=dataset or campaign.dataset,
                sample_size=sample_size or campaign.num_samples,
                results_dir=results_dir,
            )
        else:
            # Direct execution (inside container or local install)
            extra_args: list[str] = []
            if dataset or campaign.dataset:
                extra_args.extend(["--dataset", str(dataset or campaign.dataset)])
            if sample_size or campaign.num_samples:
                extra_args.extend(["--sample-size", str(sample_size or campaign.num_samples)])
            if results_dir:
                extra_args.extend(["--results-dir", str(results_dir)])
            cmd = ["lem", "experiment", container_config_path, "--yes", "--fresh"]
            cmd.extend(extra_args)

        env = os.environ.copy()
        env["LLM_ENERGY_VERBOSITY"] = os.environ.get("LLM_ENERGY_VERBOSITY", "normal")
        result = subprocess.run(cmd, env=env, check=False)
        return experiment_id, result.returncode

    finally:
        tmp_config_path.unlink(missing_ok=True)


def _link_manifest_entries(
    execution_order: list[CampaignExperiment],
    manifest: CampaignManifest,
) -> None:
    """Link manifest entries to experiments by (config_name, cycle_index) key.

    Robust alternative to index-based linking — works correctly even when
    the execution order is shuffled (no seed) or experiments are filtered.
    """
    entry_lookup: dict[tuple[str, int], Any] = {
        (e.config_name, e.cycle_index): e for e in manifest.experiments
    }
    for exp in execution_order:
        key = (exp.config_name, exp.cycle_index)
        exp.manifest_entry = entry_lookup.get(key)


def _detect_backend(config_data: dict[str, object]) -> str:
    """Detect backend from experiment config.

    Args:
        config_data: Parsed YAML config data.

    Returns:
        Backend name: 'pytorch', 'vllm', or 'tensorrt'.
    """
    # Check explicit backend field
    backend = config_data.get("backend")
    if backend:
        return str(backend).lower()

    # Fallback to environment variable or default
    return os.environ.get("LEM_BACKEND", "pytorch")


def _should_use_docker(backends: list[str] | None = None) -> bool:
    """Determine if campaign should use Docker orchestration.

    Delegates to the docker_detection module for proper detection.
    """
    from llenergymeasure.config.docker_detection import should_use_docker_for_campaign

    if backends is None:
        backends = ["pytorch"]  # Default fallback
    return should_use_docker_for_campaign(backends)


def _check_docker_images(backends: list[str]) -> list[str]:
    """Check which backend Docker images are missing.

    Uses `docker image inspect` on the expected image name (llenergymeasure:<backend>)
    to check if the image has been built.

    Args:
        backends: List of backend service names to check.

    Returns:
        List of backend names whose images are not built.
    """
    missing = []
    for backend in backends:
        image_name = f"llenergymeasure:{backend}"
        result = subprocess.run(
            ["docker", "image", "inspect", image_name],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            missing.append(backend)
    return missing


def _prompt_docker_build(missing: list[str]) -> None:
    """Prompt user to build missing Docker images, or exit.

    Args:
        missing: List of backend names with missing images.

    Raises:
        typer.Exit: If user declines to build.
    """
    services = " ".join(missing)
    console.print(f"\n[yellow]Missing Docker images:[/yellow] {', '.join(missing)}")
    console.print(
        f"\nRun this to build them:\n\n" f"  [bold]docker compose build {services}[/bold]\n"
    )
    console.print("[dim]This is a one-time step. Images are cached for future campaigns.[/dim]")
    raise typer.Exit(1)


def _build_docker_command(
    backend: str,
    config_path: str,
    dataset: str | None,
    sample_size: int | None,
    results_dir: Path | None,
) -> list[str]:
    """Build docker compose command for running experiment.

    Args:
        backend: Backend service name (pytorch, vllm, tensorrt).
        config_path: Path to config file (container path).
        dataset: Dataset override.
        sample_size: Sample size override.
        results_dir: Results directory override.

    Returns:
        Command list for subprocess.run().
    """
    cmd = [
        "docker",
        "compose",
        "run",
        "--rm",
        backend,
        "lem",
        "experiment",
        config_path,
        "--yes",
        "--fresh",
    ]

    if dataset:
        cmd.extend(["--dataset", dataset])
    if sample_size:
        cmd.extend(["--sample-size", str(sample_size)])
    if results_dir:
        # Convert host path to container path
        cmd.extend(["--results-dir", "/app/results"])

    return cmd


def _is_campaign_yaml(path: Path) -> bool:
    """Check if a YAML file is a campaign config (has campaign_name key)."""
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        return isinstance(data, dict) and "campaign_name" in data
    except Exception:
        return False


def _load_campaign_yaml(path: Path) -> CampaignConfig:
    """Load a campaign configuration from YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)

    # Resolve relative config paths
    base_dir = path.parent
    if "configs" in data:
        resolved_configs = []
        for config_path in data["configs"]:
            config_p = Path(config_path)
            if not config_p.is_absolute():
                config_p = base_dir / config_p
            resolved_configs.append(str(config_p))
        data["configs"] = resolved_configs

    return CampaignConfig(**data)


def _apply_cli_overrides(
    campaign: CampaignConfig,
    *,
    campaign_name: str | None,
    dataset: str | None,
    num_samples: int | None,
    cycles: int | None,
    structure: Literal["interleaved", "shuffled", "grouped"] | None,
    warmup_prompts: int | None,
    warmup_timeout: float | None,
    config_gap: float | None,
    cycle_gap: float | None,
) -> CampaignConfig:
    """Apply CLI overrides to campaign config."""

    updates: dict[str, Any] = {}

    if campaign_name is not None:
        updates["campaign_name"] = campaign_name
    if dataset is not None:
        updates["dataset"] = dataset
    if num_samples is not None:
        updates["num_samples"] = num_samples

    # Execution overrides
    exec_updates: dict[str, Any] = {}
    if cycles is not None:
        exec_updates["cycles"] = cycles
    if structure is not None:
        exec_updates["structure"] = structure
    if warmup_prompts is not None:
        exec_updates["warmup_prompts"] = warmup_prompts
    if warmup_timeout is not None:
        exec_updates["warmup_timeout_seconds"] = warmup_timeout
    if config_gap is not None:
        exec_updates["config_gap_seconds"] = config_gap
    if cycle_gap is not None:
        exec_updates["cycle_gap_seconds"] = cycle_gap

    if exec_updates:
        new_execution = campaign.execution.model_copy(update=exec_updates)
        updates["execution"] = new_execution

    if updates:
        return campaign.model_copy(update=updates)
    return campaign


def _display_campaign_summary(campaign: CampaignConfig, seed: int | None) -> None:
    """Display campaign configuration summary."""
    from rich.panel import Panel

    # Describe experiment source
    if campaign.grid is not None:
        backends = campaign.grid.backends
        config_source = f"grid ({', '.join(backends)})"
    elif campaign.configs:
        config_source = f"{len(campaign.configs)} config files"
    else:
        config_source = "explicit experiments"

    lines = [
        f"[cyan]Name:[/cyan] {campaign.campaign_name}",
        f"[cyan]ID:[/cyan] {campaign.campaign_id}",
        f"[cyan]Source:[/cyan] {config_source}",
        f"[cyan]Cycles:[/cyan] {campaign.execution.cycles}",
        f"[cyan]Structure:[/cyan] {campaign.execution.structure}",
    ]

    if campaign.dataset:
        lines.append(f"[cyan]Dataset:[/cyan] {campaign.dataset}")
    if campaign.num_samples:
        lines.append(f"[cyan]Samples:[/cyan] {campaign.num_samples}")

    lines.append("")
    lines.append(
        f"[dim]Warmup: {campaign.execution.warmup_prompts} prompts / "
        f"{campaign.execution.warmup_timeout_seconds}s timeout[/dim]"
    )
    lines.append(
        f"[dim]Gaps: {format_gap_time(campaign.execution.config_gap_seconds)} between configs, "
        f"{format_gap_time(campaign.execution.cycle_gap_seconds)} between cycles[/dim]"
    )

    if seed is not None:
        lines.append(f"[dim]Seed: {seed}[/dim]")

    panel = Panel("\n".join(lines), title="[bold]Campaign Configuration[/bold]")
    console.print(panel)


def _display_resume_summary(campaign: CampaignConfig) -> None:
    """Display abbreviated campaign summary when resuming."""
    from rich.panel import Panel

    lines = [
        f"[cyan]Name:[/cyan] {campaign.campaign_name}",
        f"[cyan]ID:[/cyan] {campaign.campaign_id}",
    ]
    panel = Panel("\n".join(lines), title="[bold]Resuming Campaign[/bold]")
    console.print(panel)


def _display_execution_plan(
    runner: CampaignRunner,
    execution_order: list[CampaignExperiment],
    title: str = "Execution Plan",
) -> None:
    """Display the execution plan as a table."""
    from rich.table import Table

    # Check if we have multiple backends (worth showing)
    backends = {exp.backend for exp in execution_order}
    show_backend = len(backends) > 1

    table = Table(title=title)
    table.add_column("#", style="dim")
    table.add_column("Cycle")
    table.add_column("Config")
    if show_backend:
        table.add_column("Backend", style="cyan")

    for idx, exp in enumerate(execution_order):
        row = [
            str(idx + 1),
            str(exp.cycle_index + 1),
            exp.config_name,
        ]
        if show_backend:
            row.append(exp.backend)
        table.add_row(*row)

    console.print(table)

    if show_backend:
        console.print(
            "\n[dim]Multi-backend campaign: each experiment runs in its own container[/dim]"
        )


def _display_validation_summary(result: GridExpansionResult) -> None:
    """Display grid validation results using Rich console."""
    from rich.panel import Panel

    n_valid = len(result.valid_configs)
    n_filtered = len(result.filtered_configs)
    n_warnings = len(result.warnings)

    lines = [
        f"[cyan]Total generated:[/cyan] {result.total_generated}",
        f"[green]Valid:[/green] {n_valid}",
    ]

    if n_filtered > 0:
        lines.append(f"[red]Filtered (invalid):[/red] {n_filtered}")
        for issue in result.filtered_configs[:5]:
            lines.append(f"  [dim]- {issue.config_desc}: {issue.reason}[/dim]")
        if n_filtered > 5:
            lines.append(f"  [dim]... and {n_filtered - 5} more[/dim]")

    if n_warnings > 0:
        lines.append(f"[yellow]Warnings:[/yellow] {n_warnings}")
        for issue in result.warnings[:5]:
            lines.append(f"  [dim]- {issue.config_desc}: {issue.reason}[/dim]")
        if n_warnings > 5:
            lines.append(f"  [dim]... and {n_warnings - 5} more[/dim]")

    panel = Panel("\n".join(lines), title="[bold]Grid Validation Summary[/bold]")
    console.print(panel)


def _display_campaign_ci_summary(
    manifest: CampaignManifest,
    campaign: CampaignConfig,
) -> None:
    """Display bootstrap CI summary for multi-cycle campaign results.

    Groups completed experiment results by config_name and computes
    bootstrap confidence intervals for key metrics.
    """

    from loguru import logger as _logger
    from rich.table import Table

    from llenergymeasure.results.aggregation import aggregate_campaign_results

    # Build results_by_config from manifest
    results_by_config: dict[str, list[Any]] = {}
    for entry in manifest.experiments:
        if entry.status == "completed" and entry.result_path:
            result_path = Path(entry.result_path)
            if result_path.exists():
                try:
                    from llenergymeasure.domain.experiment import AggregatedResult

                    data = result_path.read_text()
                    result = AggregatedResult.model_validate_json(data)
                    results_by_config.setdefault(entry.config_name, []).append(result)
                except Exception as e:
                    _logger.debug("Failed to load result {}: {}", entry.result_path, e)

    if not results_by_config:
        console.print("\n[dim]No completed results to aggregate for CIs[/dim]")
        return

    try:
        aggregated = aggregate_campaign_results(results_by_config)
    except Exception as e:
        _logger.warning("CI aggregation failed: {}", e)
        console.print(f"\n[yellow]Warning: CI computation failed: {e}[/yellow]")
        return

    table = Table(title="Bootstrap Confidence Intervals (95%)")
    table.add_column("Config", style="cyan")
    table.add_column("Cycles", justify="right")
    table.add_column("Energy (J)", justify="right")
    table.add_column("Throughput (tok/s)", justify="right")
    table.add_column("TTFT (ms)", justify="right")
    table.add_column("ITL (ms)", justify="right")

    for config_name, metrics in aggregated.items():
        n = metrics.get("n_cycles", "?")
        energy = metrics.get("energy_j", {})
        tps = metrics.get("throughput_tps", {})
        ttft = metrics.get("ttft_mean_ms")
        itl = metrics.get("itl_mean_ms")

        energy_str = _format_ci_cell(energy)
        tps_str = _format_ci_cell(tps)
        ttft_str = _format_ci_cell(ttft) if ttft else "-"
        itl_str = _format_ci_cell(itl) if itl else "-"

        table.add_row(config_name, str(n), energy_str, tps_str, ttft_str, itl_str)

    console.print()
    console.print(table)


def _format_ci_cell(ci_data: dict[str, object] | None) -> str:
    """Format a bootstrap CI result as a concise string."""
    if not ci_data or not isinstance(ci_data, dict):
        return "-"
    mean = ci_data.get("mean")
    lower = ci_data.get("ci_lower")
    upper = ci_data.get("ci_upper")
    if mean is None:
        return "-"
    if lower is not None and upper is not None:
        return f"{mean:.1f} [{lower:.1f}, {upper:.1f}]"
    return f"{mean:.1f}"


def _estimate_total_time(campaign: CampaignConfig, runner: CampaignRunner) -> float:
    """Estimate minimum total campaign time (gaps only, not experiments)."""
    num_configs = runner.num_configs
    num_cycles = runner.num_cycles
    total_experiments = num_configs * num_cycles

    # Config gaps: between each experiment (except first)
    config_gap_time = (total_experiments - 1) * campaign.execution.config_gap_seconds

    # Cycle gaps: between each cycle (except last)
    cycle_gap_time = (num_cycles - 1) * campaign.execution.cycle_gap_seconds

    return config_gap_time + cycle_gap_time


__all__ = ["campaign_cmd"]
