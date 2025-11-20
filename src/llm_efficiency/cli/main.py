"""
Modern CLI for LLM Efficiency Measurement Tool.

Built with Typer and Rich for beautiful, interactive terminal experience.
"""

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.prompt import Prompt, Confirm

from llm_efficiency.config import (
    ExperimentConfig,
    BatchingConfig,
    QuantizationConfig,
    DecoderConfig,
)
from llm_efficiency.core import (
    setup_accelerator,
    generate_experiment_id,
    load_model_and_tokenizer,
    run_inference_experiment,
)
from llm_efficiency.data import load_prompts_from_dataset, filter_prompts_by_length
from llm_efficiency.metrics import FLOPsCalculator, EnergyTracker, get_gpu_memory_stats
from llm_efficiency.storage import ResultsManager, create_results
from llm_efficiency.utils.logging import setup_logging
from llm_efficiency.__version__ import __version__

app = typer.Typer(
    name="llm-efficiency",
    help="LLM Efficiency Measurement Tool - Comprehensive energy and performance benchmarking",
    add_completion=False,
)
console = Console()


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"[bold]LLM Efficiency Measurement Tool[/bold] version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version and exit",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """LLM Efficiency Measurement Tool."""
    pass


@app.command()
def run(
    model_name: str = typer.Option(..., "--model", "-m", help="Model name from HuggingFace"),
    precision: str = typer.Option("float16", "--precision", "-p", help="Precision (float32/float16/bfloat16)"),
    batch_size: int = typer.Option(8, "--batch-size", "-b", help="Batch size for inference"),
    num_prompts: int = typer.Option(100, "--num-prompts", "-n", help="Number of prompts to process"),
    max_input_tokens: int = typer.Option(512, "--max-input", help="Max input tokens"),
    max_output_tokens: int = typer.Option(128, "--max-output", help="Max output tokens"),
    quantize: bool = typer.Option(False, "--quantize", "-q", help="Enable 4-bit quantization"),
    dataset: str = typer.Option("AIEnergyScore/text_generation", "--dataset", "-d", help="Dataset name"),
    results_dir: Path = typer.Option(Path("results"), "--results-dir", "-r", help="Results directory"),
    track_energy: bool = typer.Option(True, "--energy/--no-energy", help="Enable energy tracking"),
    verbose: bool = typer.Option(False, "--verbose", "-V", help="Verbose logging"),
) -> None:
    """
    Run an efficiency measurement experiment.
    
    Example:
        llm-efficiency run --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --batch-size 16 --num-prompts 100
    """
    # Setup logging
    setup_logging(level="DEBUG" if verbose else "INFO")
    
    console.print(Panel.fit(
        "[bold cyan]LLM Efficiency Measurement Tool[/bold cyan]\n"
        f"Version {__version__}",
        border_style="cyan"
    ))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Create configuration
        progress.add_task("Creating configuration...", total=None)
        
        config = ExperimentConfig(
            model_name=model_name,
            precision=precision,
            num_input_prompts=num_prompts,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            batching=BatchingConfig(batch_size=batch_size),
            quantization=QuantizationConfig(enabled=quantize, bits=4 if quantize else None),
            results_dir=str(results_dir),
            save_outputs=True,
        )
        
        console.print(f"\n[green]✓[/green] Configuration created")
        console.print(f"  Model: [cyan]{model_name}[/cyan]")
        console.print(f"  Precision: [cyan]{precision}[/cyan]")
        console.print(f"  Batch size: [cyan]{batch_size}[/cyan]")
        console.print(f"  Prompts: [cyan]{num_prompts}[/cyan]")
        if quantize:
            console.print(f"  Quantization: [cyan]4-bit[/cyan]")
        
        # Setup distributed
        progress.add_task("Setting up distributed environment...", total=None)
        accelerator = setup_accelerator()
        experiment_id = generate_experiment_id(accelerator)
        
        console.print(f"\n[green]✓[/green] Experiment ID: [bold]{experiment_id}[/bold]")
        
        # Load model
        progress.add_task(f"Loading model {model_name}...", total=None)
        model, tokenizer = load_model_and_tokenizer(config)
        device = next(model.parameters()).device
        
        console.print(f"[green]✓[/green] Model loaded on [cyan]{device}[/cyan]")
        
        # Load data
        progress.add_task(f"Loading prompts from {dataset}...", total=None)
        prompts = load_prompts_from_dataset(
            dataset_name=dataset,
            num_prompts=num_prompts,
        )
        prompts = filter_prompts_by_length(
            prompts, tokenizer, max_tokens=max_input_tokens
        )
        
        console.print(f"[green]✓[/green] Loaded [cyan]{len(prompts)}[/cyan] prompts")
        
        # Run inference
        progress.add_task("Running inference experiment...", total=None)
        
        energy_tracker = None
        if track_energy:
            energy_tracker = EnergyTracker(
                experiment_id=experiment_id,
                output_dir=results_dir / "energy",
            )
            energy_tracker.start()
        
        outputs, inference_metrics = run_inference_experiment(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            config=config,
            accelerator=accelerator,
            warmup=True,
        )
        
        if energy_tracker:
            energy_tracker.stop()
        
        console.print(f"\n[green]✓[/green] Inference complete")
        console.print(f"  Throughput: [cyan]{inference_metrics['tokens_per_second']:.2f}[/cyan] tokens/s")
        console.print(f"  Latency: [cyan]{inference_metrics['avg_latency_per_query']*1000:.2f}[/cyan] ms/query")
        
        # Calculate FLOPs
        progress.add_task("Calculating FLOPs...", total=None)
        calculator = FLOPsCalculator()
        flops = calculator.get_flops(
            model=model,
            model_name=config.model_name,
            sequence_length=config.max_input_tokens,
            device=device,
            is_quantized=config.quantization.enabled,
        )
        
        console.print(f"[green]✓[/green] FLOPs: [cyan]{flops:,}[/cyan]")
        
        # Get memory stats
        import torch
        memory_stats = {}
        if torch.cuda.is_available():
            memory_stats = get_gpu_memory_stats(device)
            console.print(f"[green]✓[/green] GPU Memory: [cyan]{memory_stats.get('gpu_current_memory_allocated_bytes', 0) / 1024**3:.2f}[/cyan] GB")
        
        # Create results
        progress.add_task("Saving results...", total=None)
        
        results = create_results(
            experiment_id=experiment_id,
            config=config.to_dict(),
            model_info={
                "model_name": config.model_name,
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "precision": config.precision,
                "quantization": f"{config.quantization.bits}-bit" if config.quantization.enabled else None,
            },
            inference_metrics=inference_metrics,
            compute_metrics={
                "flops": flops,
                "gpu_memory_allocated_mb": memory_stats.get("gpu_current_memory_allocated_bytes", 0) / 1024**2,
                "gpu_memory_peak_mb": memory_stats.get("gpu_peak_memory_allocated_bytes", 0) / 1024**2,
            },
            energy_metrics=energy_tracker.get_results() if energy_tracker else None,
            outputs=outputs if config.save_outputs else None,
        )
        
        manager = ResultsManager(results_dir=results_dir)
        saved_path = manager.save_experiment(results)
        
        console.print(f"\n[green]✓[/green] Results saved to [cyan]{saved_path}[/cyan]")
        
    # Summary
    console.print("\n" + "="*60)
    console.print("[bold green]Experiment Complete![/bold green]")
    console.print(f"Experiment ID: [bold]{experiment_id}[/bold]")
    console.print(f"Throughput: [cyan]{inference_metrics['tokens_per_second']:.2f}[/cyan] tokens/s")
    
    if energy_tracker:
        energy_results = energy_tracker.get_results()
        console.print(f"Energy: [cyan]{energy_results['energy_consumed_kwh']*1000:.2f}[/cyan] Wh")
        console.print(f"Emissions: [cyan]{energy_results['emissions_kg_co2']*1000:.2f}[/cyan] g CO2")
    
    console.print("="*60 + "\n")


@app.command()
def list(
    results_dir: Path = typer.Option(Path("results"), "--results-dir", "-r", help="Results directory"),
) -> None:
    """List all experiments."""
    manager = ResultsManager(results_dir=results_dir)
    exp_ids = manager.list_experiments()
    
    if not exp_ids:
        console.print("[yellow]No experiments found[/yellow]")
        return
    
    table = Table(title=f"Experiments ({len(exp_ids)} total)", show_header=True, header_style="bold cyan")
    table.add_column("ID", style="cyan")
    table.add_column("Model", style="white")
    table.add_column("Throughput", justify="right")
    table.add_column("Energy", justify="right")
    table.add_column("Timestamp")
    
    for exp_id in exp_ids:
        exp = manager.load_experiment(exp_id)
        if exp:
            model_name = exp.model.model_name if exp.model else "N/A"
            throughput = f"{exp.inference.tokens_per_second:.2f} tok/s" if exp.inference else "N/A"
            energy = f"{exp.energy.total_energy_kwh*1000:.2f} Wh" if exp.energy else "N/A"
            timestamp = exp.timestamp.split("T")[0] if exp.timestamp else "N/A"
            
            table.add_row(exp_id, model_name, throughput, energy, timestamp)
    
    console.print(table)


@app.command()
def show(
    experiment_id: str = typer.Argument(..., help="Experiment ID to show"),
    results_dir: Path = typer.Option(Path("results"), "--results-dir", "-r", help="Results directory"),
) -> None:
    """Show detailed results for an experiment."""
    manager = ResultsManager(results_dir=results_dir)
    results = manager.load_experiment(experiment_id)
    
    if not results:
        console.print(f"[red]Error:[/red] Experiment {experiment_id} not found")
        raise typer.Exit(1)
    
    # Header
    console.print(Panel.fit(
        f"[bold]Experiment {experiment_id}[/bold]\n"
        f"{results.timestamp}",
        border_style="cyan"
    ))
    
    # Model info
    if results.model:
        console.print("\n[bold cyan]Model Information[/bold cyan]")
        console.print(f"  Name: {results.model.model_name}")
        console.print(f"  Parameters: {results.model.total_parameters:,}")
        console.print(f"  Precision: {results.model.precision}")
        if results.model.quantization:
            console.print(f"  Quantization: {results.model.quantization}")
    
    # Inference metrics
    if results.inference:
        console.print("\n[bold cyan]Inference Metrics[/bold cyan]")
        console.print(f"  Throughput: {results.inference.tokens_per_second:.2f} tokens/s")
        console.print(f"  Queries/s: {results.inference.queries_per_second:.2f}")
        console.print(f"  Avg latency: {results.inference.avg_latency_per_query*1000:.2f} ms")
        console.print(f"  Total tokens: {results.inference.total_tokens:,}")
        console.print(f"  Prompts: {results.inference.num_prompts}")
    
    # Compute metrics
    if results.compute:
        console.print("\n[bold cyan]Compute Metrics[/bold cyan]")
        console.print(f"  FLOPs: {results.compute.flops:,}")
        console.print(f"  GPU Memory: {results.compute.gpu_memory_allocated_mb:.2f} MB")
        console.print(f"  GPU Peak: {results.compute.gpu_memory_peak_mb:.2f} MB")
    
    # Energy metrics
    if results.energy:
        console.print("\n[bold cyan]Energy Metrics[/bold cyan]")
        console.print(f"  Duration: {results.energy.duration_seconds:.2f} s")
        console.print(f"  Total energy: {results.energy.total_energy_kwh*1000:.2f} Wh")
        console.print(f"  CPU energy: {results.energy.cpu_energy_kwh*1000:.2f} Wh")
        console.print(f"  GPU energy: {results.energy.gpu_energy_kwh*1000:.2f} Wh")
        console.print(f"  RAM energy: {results.energy.ram_energy_kwh*1000:.2f} Wh")
        console.print(f"  Emissions: {results.energy.emissions_kg_co2*1000:.2f} g CO2")
    
    # Efficiency metrics
    if results.efficiency:
        console.print("\n[bold cyan]Efficiency Metrics[/bold cyan]")
        for key, value in results.efficiency.items():
            console.print(f"  {key}: {value:.4f}")


@app.command()
def export(
    output_file: Path = typer.Argument(..., help="Output CSV file path"),
    results_dir: Path = typer.Option(Path("results"), "--results-dir", "-r", help="Results directory"),
    experiment_ids: Optional[str] = typer.Option(None, "--ids", "-i", help="Comma-separated experiment IDs (default: all)"),
) -> None:
    """Export experiments to CSV."""
    manager = ResultsManager(results_dir=results_dir)
    
    ids = experiment_ids.split(",") if experiment_ids else None
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Exporting to CSV...", total=None)
        manager.export_to_csv(output_file, ids)
    
    console.print(f"[green]✓[/green] Exported to [cyan]{output_file}[/cyan]")


@app.command()
def init(
    output_file: Path = typer.Option(Path("config.json"), "--output", "-o", help="Output config file"),
) -> None:
    """Interactive configuration wizard."""
    console.print(Panel.fit(
        "[bold]LLM Efficiency Configuration Wizard[/bold]\n"
        "Create a configuration file interactively",
        border_style="cyan"
    ))
    
    # Model selection
    model_name = Prompt.ask("\n[cyan]Model name from HuggingFace[/cyan]", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    # Precision
    precision = Prompt.ask(
        "[cyan]Precision[/cyan]",
        choices=["float32", "float16", "bfloat16"],
        default="float16"
    )
    
    # Batch size
    batch_size = int(Prompt.ask("[cyan]Batch size[/cyan]", default="8"))
    
    # Prompts
    num_prompts = int(Prompt.ask("[cyan]Number of prompts[/cyan]", default="100"))
    max_input = int(Prompt.ask("[cyan]Max input tokens[/cyan]", default="512"))
    max_output = int(Prompt.ask("[cyan]Max output tokens[/cyan]", default="128"))
    
    # Quantization
    quantize = Confirm.ask("[cyan]Enable 4-bit quantization?[/cyan]", default=False)
    
    # Create config
    config = ExperimentConfig(
        model_name=model_name,
        precision=precision,
        num_input_prompts=num_prompts,
        max_input_tokens=max_input,
        max_output_tokens=max_output,
        batching=BatchingConfig(batch_size=batch_size),
        quantization=QuantizationConfig(enabled=quantize, bits=4 if quantize else None),
    )
    
    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    
    console.print(f"\n[green]✓[/green] Configuration saved to [cyan]{output_file}[/cyan]")


@app.command()
def summary(
    results_dir: Path = typer.Option(Path("results"), "--results-dir", "-r", help="Results directory"),
    experiment_ids: Optional[str] = typer.Option(None, "--ids", "-i", help="Comma-separated experiment IDs (default: all)"),
) -> None:
    """Generate summary statistics."""
    manager = ResultsManager(results_dir=results_dir)
    
    ids = experiment_ids.split(",") if experiment_ids else None
    summary = manager.generate_summary(ids)
    
    if not summary:
        console.print("[yellow]No experiments found[/yellow]")
        return
    
    console.print(Panel.fit(
        f"[bold]Summary Statistics[/bold]\n"
        f"Total experiments: {summary['total_experiments']}",
        border_style="cyan"
    ))
    
    # Throughput
    console.print("\n[bold cyan]Throughput[/bold cyan]")
    console.print(f"  Mean: {summary['throughput']['mean_tokens_per_second']:.2f} tokens/s")
    console.print(f"  Max: {summary['throughput']['max_tokens_per_second']:.2f} tokens/s")
    console.print(f"  Min: {summary['throughput']['min_tokens_per_second']:.2f} tokens/s")
    
    # Energy
    console.print("\n[bold cyan]Energy[/bold cyan]")
    console.print(f"  Total: {summary['energy']['total_kwh']*1000:.2f} Wh")
    console.print(f"  Mean: {summary['energy']['mean_kwh']*1000:.2f} Wh")


if __name__ == "__main__":
    app()
