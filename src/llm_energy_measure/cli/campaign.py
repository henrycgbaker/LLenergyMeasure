"""Campaign CLI command for multi-config comparison experiments.

Provides the `campaign` command for running multiple experiment configs
across multiple cycles for statistical robustness and fair comparison.
"""

from __future__ import annotations

import glob as glob_module
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated, Literal

import typer
import yaml

from llm_energy_measure.cli.display import console
from llm_energy_measure.config.campaign_config import (
    CampaignConfig,
    CampaignExecutionConfig,
)
from llm_energy_measure.orchestration.campaign import (
    CampaignExperiment,
    CampaignRunner,
    format_gap_time,
)


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
) -> None:
    """Run a multi-config campaign for statistical comparison.

    Campaigns run multiple experiment configurations across multiple cycles
    to enable fair comparison with statistical robustness.

    Two modes of operation:

    1. Campaign YAML file (recommended for reproducibility):

       llm-energy-measure campaign campaign.yaml

    2. Multiple experiment configs with CLI options:

       llm-energy-measure campaign configs/*.yaml \\
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

    # Display campaign summary
    _display_campaign_summary(campaign, seed)

    # Create runner
    runner = CampaignRunner(campaign, seed=seed)
    execution_order = runner.generate_execution_order()

    # Display execution plan
    _display_execution_plan(runner, execution_order)

    if dry_run:
        console.print("\n[cyan]--dry-run: Exiting without running campaign[/cyan]")
        raise typer.Exit(0)

    # Confirm execution
    if not yes:
        from rich.prompt import Confirm

        total_time = _estimate_total_time(campaign, runner)
        console.print(f"\n[dim]Estimated minimum time: {format_gap_time(total_time)}[/dim]")

        if not Confirm.ask("\nStart campaign?", default=True):
            console.print("[dim]Aborted[/dim]")
            raise typer.Abort()

    # Execute campaign
    console.print("\n[bold cyan]━━━ Starting Campaign ━━━[/bold cyan]\n")
    console.print(f"Campaign ID: [cyan]{campaign.campaign_id}[/cyan]")
    console.print(f"Campaign Name: [cyan]{campaign.campaign_name}[/cyan]\n")

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

        # Run the experiment
        runner.record_experiment_start(experiment)
        experiment_id, exit_code = _run_single_experiment(
            experiment=experiment,
            campaign=campaign,
            dataset=dataset,
            sample_size=sample_size,
            results_dir=results_dir,
        )

        if exit_code == 0:
            console.print(f"  [green]✓[/green] Completed (experiment_id: {experiment_id})")
            runner.record_experiment_complete(experiment, experiment_id)
        else:
            console.print(f"  [red]✗[/red] Failed (exit code: {exit_code})")
            failed_experiments.append((config_name, exit_code))
            runner.record_experiment_complete(experiment, experiment_id)

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

    if failed_experiments:
        raise typer.Exit(1)


def _run_single_experiment(
    experiment: CampaignExperiment,
    campaign: CampaignConfig,
    dataset: str | None,
    sample_size: int | None,
    results_dir: Path | None,
) -> tuple[str, int]:
    """Run a single experiment via subprocess.

    Calls the existing experiment infrastructure with campaign metadata.

    Args:
        experiment: The campaign experiment to run.
        campaign: Parent campaign configuration.
        dataset: Dataset override (None = use config).
        sample_size: Sample size override (None = use config).
        results_dir: Results directory override.

    Returns:
        Tuple of (experiment_id, exit_code).
    """
    import tempfile

    from llm_energy_measure.core.distributed import get_persistent_unique_id

    # Generate experiment ID
    experiment_id = get_persistent_unique_id()

    # Load the experiment config and add campaign metadata
    config_data = yaml.safe_load(experiment.config_path.read_text())
    config_data["_metadata"] = {
        "experiment_id": experiment_id,
        "campaign_name": campaign.campaign_name,
        "campaign_id": campaign.campaign_id,
        "cycle_id": experiment.cycle_index,
        "config_name": experiment.config_name,
    }

    # Write temp config with metadata
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="campaign_"
    ) as tmp:
        yaml.dump(config_data, tmp, default_flow_style=False)
        tmp_config_path = tmp.name

    try:
        # Build experiment command
        cmd = [
            sys.executable,
            "-m",
            "llm_energy_measure.cli",
            "experiment",
            tmp_config_path,
            "--yes",  # Skip confirmation (we already confirmed campaign)
        ]

        # Add dataset/sample size overrides
        if dataset or campaign.dataset:
            cmd.extend(["--dataset", dataset or campaign.dataset])  # type: ignore[list-item]
        if sample_size or campaign.num_samples:
            cmd.extend(["--sample-size", str(sample_size or campaign.num_samples)])
        if results_dir:
            cmd.extend(["--results-dir", str(results_dir)])

        # Run experiment
        console.print(f"  [dim]Running: {experiment.config_name}[/dim]")

        # Inherit environment for GPU access and HF token
        env = os.environ.copy()
        env["LLM_ENERGY_VERBOSITY"] = os.environ.get("LLM_ENERGY_VERBOSITY", "normal")

        result = subprocess.run(cmd, env=env, check=False)
        return experiment_id, result.returncode

    finally:
        # Clean up temp file
        Path(tmp_config_path).unlink(missing_ok=True)


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
    from typing import Any

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

    lines = [
        f"[cyan]Name:[/cyan] {campaign.campaign_name}",
        f"[cyan]ID:[/cyan] {campaign.campaign_id}",
        f"[cyan]Configs:[/cyan] {len(campaign.configs)}",
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


def _display_execution_plan(
    runner: CampaignRunner,
    execution_order: list[CampaignExperiment],
) -> None:
    """Display the execution plan as a table."""
    from rich.table import Table

    table = Table(title="Execution Plan")
    table.add_column("#", style="dim")
    table.add_column("Cycle")
    table.add_column("Config")

    for idx, exp in enumerate(execution_order):
        table.add_row(
            str(idx + 1),
            str(exp.cycle_index + 1),
            exp.config_name,
        )

    console.print(table)


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
