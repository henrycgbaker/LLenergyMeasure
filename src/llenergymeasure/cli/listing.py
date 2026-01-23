"""Listing commands for datasets, presets, and GPUs."""

from __future__ import annotations

import typer
from rich.table import Table

from llenergymeasure.cli.display import console
from llenergymeasure.constants import PRESETS
from llenergymeasure.core.dataset_loader import list_builtin_datasets


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
        "gsm8k": "Primary school maths reasoning",
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
    console.print("\n[dim]Usage: lem run config.yaml --dataset alpaca -n 1000[/dim]")


def list_presets_cmd() -> None:
    """List built-in experiment presets.

    Presets provide sensible defaults for common experiment scenarios.
    Use with: lem experiment --preset <name> --model <model>
    """
    table = Table(title="Built-in Presets")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Key Settings")

    preset_descriptions = {
        "quick-test": "Fast validation runs",
        "benchmark": "Formal benchmark measurements",
        "throughput": "Throughput-optimised testing",
    }

    for name, config in PRESETS.items():
        settings = []
        if "max_input_tokens" in config:
            settings.append(f"max_in={config['max_input_tokens']}")
        if "max_output_tokens" in config:
            settings.append(f"max_out={config['max_output_tokens']}")
        if "batching" in config:
            settings.append(f"batch={config['batching'].get('batch_size', 1)}")
        if "fp_precision" in config:
            settings.append(f"precision={config['fp_precision']}")

        table.add_row(
            name,
            preset_descriptions.get(name, ""),
            ", ".join(settings),
        )

    console.print(table)
    console.print(
        "\n[dim]Usage: lem experiment --preset quick-test --model <model> -d alpaca[/dim]"
    )


def list_gpus_cmd() -> None:
    """Show GPU topology including MIG instances.

    Displays all visible CUDA devices with their configuration,
    including Multi-Instance GPU (MIG) partitions if present.
    """
    from llenergymeasure.core.gpu_info import (
        detect_gpu_topology,
        format_gpu_topology,
    )

    topology = detect_gpu_topology()

    if not topology.devices:
        console.print("[yellow]No CUDA devices detected[/yellow]")
        console.print("\nPossible causes:")
        console.print("  - No NVIDIA GPU installed")
        console.print("  - CUDA drivers not installed")
        console.print("  - CUDA_VISIBLE_DEVICES set to empty")
        raise typer.Exit(1)

    console.print(format_gpu_topology(topology))

    # Show usage hint
    if len(topology.devices) > 1:
        indices = ",".join(str(d.index) for d in topology.devices[:2])
        console.print(
            f"\n[dim]Use --gpu-list to select devices: lem experiment config.yaml --gpu-list {indices}[/dim]"
        )
