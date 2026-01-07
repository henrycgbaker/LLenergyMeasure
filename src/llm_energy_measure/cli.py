"""Command-line interface for llm-energy-measure.

Provides commands for:
- Running experiments
- Aggregating raw results
- Validating configurations
- Listing and inspecting results
"""

import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from llm_energy_measure.config.loader import load_config, validate_config
from llm_energy_measure.config.models import ExperimentConfig, HuggingFacePromptSource
from llm_energy_measure.constants import SCHEMA_VERSION
from llm_energy_measure.core.dataset_loader import (
    list_builtin_datasets,
    load_prompts_from_file,
    load_prompts_from_source,
)
from llm_energy_measure.domain.experiment import AggregatedResult, RawProcessResult
from llm_energy_measure.exceptions import ConfigurationError
from llm_energy_measure.logging import setup_logging
from llm_energy_measure.results.aggregation import aggregate_results, calculate_efficiency_metrics
from llm_energy_measure.results.repository import FileSystemRepository

app = typer.Typer(
    name="llm-energy-measure",
    help="LLM inference efficiency measurement framework",
    add_completion=False,
)
config_app = typer.Typer(help="Configuration management commands")
results_app = typer.Typer(help="Results inspection commands")
app.add_typer(config_app, name="config")
app.add_typer(results_app, name="results")

console = Console()


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
    config_path: Annotated[Path, typer.Argument(help="Path to experiment config file")],
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
) -> None:
    """Run experiment with automatic accelerate handling.

    Reads num_processes from config and spawns accelerate launch automatically.
    This is the recommended way to run experiments.

    Examples:
        llm-energy-measure experiment config.yaml --dataset alpaca -n 100
        llm-energy-measure experiment config.yaml --prompts prompts.txt
    """
    try:
        # Load config to get num_processes
        config = load_config(config_path)
        num_processes = config.num_processes

        console.print(f"[bold]Running experiment: {config.config_name}[/bold]")
        console.print(f"  Model: {config.model_name}")
        console.print(f"  Processes: {num_processes}")

        # Build accelerate command
        cmd = [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            "--num_processes",
            str(num_processes),
            "-m",
            "llm_energy_measure.orchestration.launcher",
            "--config",
            str(config_path.resolve()),
        ]

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

        console.print(f"\n[dim]$ accelerate launch --num_processes {num_processes} ...[/dim]\n")

        # Run accelerate, streaming output
        result = subprocess.run(cmd, check=False)
        raise typer.Exit(result.returncode)

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
        "gsm8k": "Grade school math reasoning",
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
) -> None:
    """Aggregate raw per-process results into final experiment result.

    For multi-GPU experiments, this combines results from each GPU/process
    into a single aggregated result with proper energy summation and
    throughput averaging.
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
            _aggregate_one(repo, exp_id, force)
    elif experiment_id:
        _aggregate_one(repo, experiment_id, force)
    else:
        console.print("[red]Error:[/red] Provide experiment ID or use --all")
        raise typer.Exit(1)


def _aggregate_one(repo: FileSystemRepository, experiment_id: str, force: bool) -> None:
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
        aggregated = aggregate_results(experiment_id, raw_results)
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

    except Exception as e:
        console.print(f"[red]Aggregation failed:[/red] {experiment_id} - {e}")


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
