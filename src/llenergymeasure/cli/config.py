"""Configuration management commands.

This module contains CLI commands for validating, showing, creating,
and generating experiment configurations.
"""

from __future__ import annotations

import copy
import itertools
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import ValidationError
from rich.markup import escape as rich_escape
from rich.prompt import Confirm, Prompt
from rich.table import Table

from llenergymeasure.cli.display import (
    add_section_header,
    console,
    display_config_summary,
    format_field,
)
from llenergymeasure.config import ConfigWarning
from llenergymeasure.config.loader import (
    has_blocking_warnings,
    load_config,
    validate_config,
)
from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.constants import PRESETS
from llenergymeasure.exceptions import ConfigurationError

config_app = typer.Typer(help="Configuration management commands", invoke_without_command=True)


@config_app.command("list")  # type: ignore[misc]
def config_list(
    directory: Annotated[
        Path,
        typer.Option("--directory", "-d", help="Config directory to scan"),
    ] = Path("configs"),
    show_user_config: Annotated[
        bool,
        typer.Option("--show-user-config", "-u", help="Also show .lem-config.yaml settings"),
    ] = False,
) -> None:
    """List available configuration files with metadata.

    Scans the specified directory for YAML configuration files and displays
    a summary table showing the config name, backend, model, and path.
    """
    import yaml as yaml_lib

    from llenergymeasure.config.user_config import load_user_config

    # Find all YAML files recursively
    yaml_files = list(directory.glob("**/*.yaml"))

    if not yaml_files:
        console.print(f"[yellow]No configuration files found in {directory}[/yellow]")
        return

    # Build table
    table = Table(title="Available Configurations")
    table.add_column("Name", style="bold")
    table.add_column("Backend", style="cyan")
    table.add_column("Model")
    table.add_column("Path", style="dim")

    skipped = 0
    for yaml_path in sorted(yaml_files):
        try:
            with open(yaml_path) as f:
                data = yaml_lib.safe_load(f)

            if not isinstance(data, dict):
                skipped += 1
                continue

            # Skip campaign configs (they have campaign_name, not config_name)
            if "campaign_name" in data:
                continue

            # Extract fields
            name = data.get("config_name", yaml_path.stem)
            backend = data.get("backend", "-")
            model = data.get("model_name", "-")

            # Truncate model name if too long
            if isinstance(model, str) and len(model) > 40:
                model = model[:37] + "..."

            # Relative path for display
            try:
                rel_path = yaml_path.relative_to(Path.cwd())
            except ValueError:
                rel_path = yaml_path

            table.add_row(str(name), str(backend), str(model), str(rel_path))

        except yaml_lib.YAMLError:
            skipped += 1
            continue
        except Exception:
            skipped += 1
            continue

    console.print(table)

    if skipped > 0:
        console.print(f"\n[dim]Skipped {skipped} invalid/unreadable files[/dim]")

    # Show user config if requested
    if show_user_config:
        console.print()
        try:
            user_cfg = load_user_config()
            user_table = Table(title="User Configuration (.lem-config.yaml)")
            user_table.add_column("Setting")
            user_table.add_column("Value")

            user_table.add_row("Results directory", user_cfg.results_dir)
            user_table.add_row(
                "Thermal gaps",
                f"{user_cfg.thermal_gaps.between_experiments}s / {user_cfg.thermal_gaps.between_cycles}s",
            )
            user_table.add_row("Docker strategy", user_cfg.docker.strategy)
            webhook = user_cfg.notifications.webhook_url or "[dim]not configured[/dim]"
            user_table.add_row("Webhook URL", webhook)

            console.print(user_table)
        except FileNotFoundError:
            console.print("[dim]No .lem-config.yaml found (using defaults)[/dim]")
        except ValueError as e:
            console.print(f"[yellow]Warning: {e}[/yellow]")


@config_app.callback()  # type: ignore[misc]
def config_callback(ctx: typer.Context) -> None:
    """Configuration management commands."""
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


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
        display_config_summary(config, {})

        # Show warnings with severity
        if warnings:
            console.print()
        for warning in warnings:
            severity_style = "yellow" if warning.severity == "warning" else "dim"
            console.print(f"[{severity_style}]{rich_escape(str(warning))}[/{severity_style}]")

    except ConfigurationError as e:
        console.print(f"[red]Invalid configuration:[/red] {e}")
        raise typer.Exit(1) from None


@config_app.command("show")  # type: ignore[misc]
def config_show(
    config_path: Annotated[Path, typer.Argument(help="Path to config file")],
) -> None:
    """Display resolved configuration with inheritance applied.

    Shows ALL configuration parameters with visual styling:
    - Bold: Section headers
    - Bright/coloured: Explicitly set values
    - Dim: Default values
    """
    try:
        config = load_config(config_path)

        table = Table(title=f"Configuration: {config.config_name}")
        table.add_column("Field")
        table.add_column("Value")

        # =====================================================================
        # Tier 1: Universal Settings
        # =====================================================================
        table.add_row(*format_field("model_name", config.model_name, False))
        table.add_row(*format_field("backend", config.backend, config.backend == "pytorch"))
        table.add_row(*format_field("gpus", config.gpus, config.gpus == [0]))
        table.add_row(
            *format_field(
                "max_input_tokens", config.max_input_tokens, config.max_input_tokens == 512
            )
        )
        table.add_row(
            *format_field(
                "max_output_tokens", config.max_output_tokens, config.max_output_tokens == 128
            )
        )
        table.add_row(
            *format_field(
                "min_output_tokens", config.min_output_tokens, config.min_output_tokens == 0
            )
        )
        table.add_row(
            *format_field(
                "num_input_prompts", config.num_input_prompts, config.num_input_prompts == 1
            )
        )
        table.add_row(
            *format_field("fp_precision", config.fp_precision, config.fp_precision == "float16")
        )
        table.add_row(
            *format_field("save_outputs", config.save_outputs, config.save_outputs is False)
        )
        table.add_row(
            *format_field(
                "decode_token_to_text",
                config.decode_token_to_text,
                config.decode_token_to_text is False,
            )
        )
        table.add_row(*format_field("query_rate", config.query_rate, config.query_rate == 1.0))
        table.add_row(*format_field("random_seed", config.random_seed, config.random_seed is None))

        # Traffic simulation
        add_section_header(table, "traffic_simulation")
        sim = config.traffic_simulation
        table.add_row(*format_field("enabled", sim.enabled, sim.enabled is False, nested=True))
        table.add_row(*format_field("mode", sim.mode, sim.mode == "poisson", nested=True))
        table.add_row(
            *format_field("target_qps", sim.target_qps, sim.target_qps == 1.0, nested=True)
        )
        table.add_row(*format_field("seed", sim.seed, sim.seed is None, nested=True))

        # Schedule config
        add_section_header(table, "schedule")
        sched = config.schedule
        table.add_row(*format_field("enabled", sched.enabled, sched.enabled is False, nested=True))
        table.add_row(
            *format_field("interval", sched.interval, sched.interval is None, nested=True)
        )
        table.add_row(*format_field("at", sched.at, sched.at is None, nested=True))
        days_str = ", ".join(sched.days) if sched.days else None
        table.add_row(*format_field("days", days_str, sched.days is None, nested=True))
        table.add_row(
            *format_field(
                "total_duration", sched.total_duration, sched.total_duration == "24h", nested=True
            )
        )

        # Decoder config (universal sampling params)
        add_section_header(table, "decoder")
        decoder = config.decoder
        table.add_row(*format_field("preset", decoder.preset, decoder.preset is None, nested=True))
        mode = "deterministic (greedy)" if decoder.is_deterministic else "sampling"
        table.add_row(
            *format_field(
                "mode", mode, decoder.temperature == 1.0 and decoder.do_sample, nested=True
            )
        )
        table.add_row(
            *format_field(
                "temperature", decoder.temperature, decoder.temperature == 1.0, nested=True
            )
        )
        table.add_row(
            *format_field("do_sample", decoder.do_sample, decoder.do_sample is True, nested=True)
        )
        table.add_row(*format_field("top_p", decoder.top_p, decoder.top_p == 1.0, nested=True))
        table.add_row(*format_field("top_k", decoder.top_k, decoder.top_k == 50, nested=True))
        table.add_row(
            *format_field(
                "repetition_penalty",
                decoder.repetition_penalty,
                decoder.repetition_penalty == 1.0,
                nested=True,
            )
        )

        # =====================================================================
        # Tier 2: Backend-Specific Settings
        # =====================================================================
        if config.backend == "pytorch" and config.pytorch:
            add_section_header(table, "pytorch (backend)")
            pt = config.pytorch
            table.add_row(
                *format_field("batch_size", pt.batch_size, pt.batch_size == 1, nested=True)
            )
            table.add_row(
                *format_field(
                    "batching_strategy",
                    pt.batching_strategy,
                    pt.batching_strategy == "static",
                    nested=True,
                )
            )
            table.add_row(
                *format_field(
                    "num_processes",
                    pt.num_processes,
                    pt.num_processes == 1,
                    nested=True,
                )
            )
            table.add_row(
                *format_field(
                    "load_in_4bit", pt.load_in_4bit, pt.load_in_4bit is False, nested=True
                )
            )
            table.add_row(
                *format_field(
                    "load_in_8bit", pt.load_in_8bit, pt.load_in_8bit is False, nested=True
                )
            )
            table.add_row(
                *format_field(
                    "torch_compile", pt.torch_compile, pt.torch_compile is False, nested=True
                )
            )
            table.add_row(
                *format_field(
                    "attn_implementation",
                    pt.attn_implementation,
                    pt.attn_implementation == "sdpa",
                    nested=True,
                )
            )
            table.add_row(*format_field("min_p", pt.min_p, pt.min_p == 0.0, nested=True))

        elif config.backend == "vllm" and config.vllm:
            add_section_header(table, "vllm (backend)")
            vl = config.vllm
            table.add_row(
                *format_field("max_num_seqs", vl.max_num_seqs, vl.max_num_seqs == 256, nested=True)
            )
            table.add_row(
                *format_field(
                    "tensor_parallel_size",
                    vl.tensor_parallel_size,
                    vl.tensor_parallel_size == 1,
                    nested=True,
                )
            )
            table.add_row(
                *format_field(
                    "gpu_memory_utilization",
                    vl.gpu_memory_utilization,
                    vl.gpu_memory_utilization == 0.9,
                    nested=True,
                )
            )
            table.add_row(
                *format_field(
                    "enable_prefix_caching",
                    vl.enable_prefix_caching,
                    vl.enable_prefix_caching is False,
                    nested=True,
                )
            )
            table.add_row(
                *format_field("quantization", vl.quantization, vl.quantization is None, nested=True)
            )
            table.add_row(*format_field("min_p", vl.min_p, vl.min_p == 0.0, nested=True))

        elif config.backend == "tensorrt" and config.tensorrt:
            add_section_header(table, "tensorrt (backend)")
            tr = config.tensorrt
            table.add_row(
                *format_field(
                    "max_batch_size", tr.max_batch_size, tr.max_batch_size == 8, nested=True
                )
            )
            table.add_row(*format_field("tp_size", tr.tp_size, tr.tp_size == 1, nested=True))
            table.add_row(
                *format_field(
                    "builder_opt_level",
                    tr.builder_opt_level,
                    tr.builder_opt_level == 3,
                    nested=True,
                )
            )
            table.add_row(
                *format_field(
                    "kv_cache_type", tr.kv_cache_type, tr.kv_cache_type == "paged", nested=True
                )
            )
            table.add_row(
                *format_field(
                    "quantization", tr.quantization, tr.quantization == "none", nested=True
                )
            )

        # Prompt source (if configured)
        if config.prompts is not None:
            add_section_header(table, "prompts")
            ps = config.prompts
            table.add_row(*format_field("type", ps.type, False, nested=True))
            if ps.type == "file":
                table.add_row(*format_field("path", ps.path, False, nested=True))
            else:  # huggingface
                table.add_row(*format_field("dataset", ps.dataset, False, nested=True))
                table.add_row(*format_field("split", ps.split, ps.split == "train", nested=True))
                table.add_row(*format_field("subset", ps.subset, ps.subset is None, nested=True))
                table.add_row(*format_field("column", ps.column, ps.column is None, nested=True))
                table.add_row(
                    *format_field(
                        "sample_size", ps.sample_size, ps.sample_size is None, nested=True
                    )
                )
                table.add_row(
                    *format_field("shuffle", ps.shuffle, ps.shuffle is False, nested=True)
                )
                table.add_row(*format_field("seed", ps.seed, ps.seed == 42, nested=True))

        console.print(table)

    except ConfigurationError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None


@config_app.command("new")  # type: ignore[misc]
def config_new(
    output_path: Annotated[
        Path | None, typer.Option("--output", "-o", help="Output path for config file")
    ] = None,
    base_preset: Annotated[str | None, typer.Option("--preset", help="Start from a preset")] = None,
) -> None:
    """Interactive config builder for creating new experiment configs.

    Guides you through creating a valid configuration file with sensible defaults.
    """
    import yaml

    console.print("[bold]Create new experiment configuration[/bold]\n")

    # Config name
    config_name = Prompt.ask("Config name", default="my-experiment")

    # Model name
    model_name = Prompt.ask(
        "Model name (HuggingFace path)",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    )

    # Start from preset?
    if base_preset is None:
        use_preset = Confirm.ask("Start from a preset?", default=False)
        if use_preset:
            preset_choices = ", ".join(PRESETS.keys())
            base_preset = Prompt.ask(f"Preset ({preset_choices})", default="benchmark")

    # Build config
    config_dict: dict[str, Any] = {}
    if base_preset and base_preset in PRESETS:
        config_dict = copy.deepcopy(PRESETS[base_preset])
        console.print(f"[dim]Starting from preset: {base_preset}[/dim]")

    config_dict["config_name"] = config_name
    config_dict["model_name"] = model_name

    # Backend selection
    backend = Prompt.ask(
        "Backend (pytorch/vllm/tensorrt)",
        default=config_dict.get("backend", "pytorch"),
    )
    config_dict["backend"] = backend

    # GPU configuration
    num_gpus = int(Prompt.ask("Number of GPUs", default="1"))
    if num_gpus > 1:
        gpus = [
            int(g)
            for g in Prompt.ask(
                "GPU indices (comma-separated)",
                default=",".join(str(i) for i in range(num_gpus)),
            ).split(",")
        ]
        config_dict["gpus"] = gpus
    else:
        config_dict["gpus"] = [0]

    # Precision
    precision = Prompt.ask(
        "Precision (float32/float16/bfloat16)",
        default=config_dict.get("fp_precision", "float16"),
    )
    config_dict["fp_precision"] = precision

    # Token limits
    max_input = int(
        Prompt.ask(
            "Max input tokens",
            default=str(config_dict.get("max_input_tokens", 512)),
        )
    )
    max_output = int(
        Prompt.ask(
            "Max output tokens",
            default=str(config_dict.get("max_output_tokens", 128)),
        )
    )
    config_dict["max_input_tokens"] = max_input
    config_dict["max_output_tokens"] = max_output

    # Backend-specific configuration
    if backend == "pytorch":
        batch_size = int(
            Prompt.ask(
                "Batch size",
                default=str(config_dict.get("pytorch", {}).get("batch_size", 1)),
            )
        )
        if "pytorch" not in config_dict:
            config_dict["pytorch"] = {}
        config_dict["pytorch"]["batch_size"] = batch_size

        # Quantization (PyTorch-specific via BitsAndBytes)
        use_quant = Confirm.ask("Enable quantization (BitsAndBytes)?", default=False)
        if use_quant:
            quant_bits = Prompt.ask("Quantization bits (4/8)", default="4")
            config_dict["pytorch"]["load_in_4bit"] = quant_bits == "4"
            config_dict["pytorch"]["load_in_8bit"] = quant_bits == "8"

    elif backend == "vllm":
        max_num_seqs = int(
            Prompt.ask(
                "Max concurrent sequences",
                default=str(config_dict.get("vllm", {}).get("max_num_seqs", 256)),
            )
        )
        if "vllm" not in config_dict:
            config_dict["vllm"] = {}
        config_dict["vllm"]["max_num_seqs"] = max_num_seqs

    elif backend == "tensorrt":
        max_batch_size = int(
            Prompt.ask(
                "Max batch size (compile-time)",
                default=str(config_dict.get("tensorrt", {}).get("max_batch_size", 8)),
            )
        )
        if "tensorrt" not in config_dict:
            config_dict["tensorrt"] = {}
        config_dict["tensorrt"]["max_batch_size"] = max_batch_size

    # Validate config
    try:
        config = ExperimentConfig(**config_dict)
    except Exception as e:
        console.print(f"[red]Invalid configuration:[/red] {e}")
        raise typer.Exit(1) from None

    # Determine output path
    if output_path is None:
        output_path = Path(f"configs/{config_name}.yaml")

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write YAML
    with open(output_path, "w") as f:
        yaml.dump(
            config.model_dump(exclude_defaults=True), f, default_flow_style=False, sort_keys=False
        )

    console.print(f"\n[green]✓[/green] Created: {output_path}")
    console.print(f"\n[dim]Run with: lem experiment {output_path} --dataset alpaca -n 100[/dim]")


@config_app.command("generate-grid")  # type: ignore[misc]
def config_generate_grid(
    base_config: Annotated[Path, typer.Argument(help="Base config file to vary")],
    vary: Annotated[
        list[str] | None,
        typer.Option("--vary", help="Parameter=values to vary (e.g., batch_size=1,2,4,8)"),
    ] = None,
    output_dir: Annotated[
        Path, typer.Option("--output-dir", "-o", help="Output directory for generated configs")
    ] = Path("configs/grid"),
    validate: Annotated[
        bool,
        typer.Option("--validate", help="Only generate valid configs (skip those with errors)"),
    ] = False,
    strict: Annotated[
        bool, typer.Option("--strict", help="Fail if any generated config would be invalid")
    ] = False,
) -> None:
    """Generate a grid of configs from a base config with parameter variations.

    Creates Cartesian product of all --vary parameters.

    Example:
        lem config generate-grid base.yaml \\
            --vary batch_size=1,2,4,8 \\
            --vary fp_precision=float16,float32 \\
            --output-dir ./grid/
    """
    import yaml

    if not vary:
        console.print("[red]Error:[/red] At least one --vary parameter is required")
        raise typer.Exit(1)

    # Load base config
    try:
        base = load_config(base_config)
        base_dict = base.model_dump()
    except ConfigurationError as e:
        console.print(f"[red]Error loading base config:[/red] {e}")
        raise typer.Exit(1) from None

    # Shorthand mappings for common parameters (match CLI flag behavior)
    # Note: batch_size now lives in backend-specific configs
    param_shortcuts: dict[str, str] = {
        "batch_size": "pytorch.batch_size",  # Only works for PyTorch configs
        "temperature": "decoder.temperature",
    }

    # Parse variations
    variations: dict[str, list[Any]] = {}
    for v in vary:
        if "=" not in v:
            console.print(f"[red]Error:[/red] Invalid --vary format: {v}")
            console.print("Expected: parameter=value1,value2,...")
            raise typer.Exit(1)

        param, values_str = v.split("=", 1)
        # Apply shorthand mappings
        param = param_shortcuts.get(param, param)
        values: list[Any] = []
        for val in values_str.split(","):
            val = val.strip()
            # Try to parse as number
            try:
                if "." in val:
                    values.append(float(val))
                else:
                    values.append(int(val))
            except ValueError:
                values.append(val)
        variations[param] = values

    # Generate Cartesian product
    param_names = list(variations.keys())
    param_values = list(variations.values())
    combinations = list(itertools.product(*param_values))

    console.print(f"Generating {len(combinations)} configs from {len(param_names)} parameters...")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate configs with validation
    config_variations: list[tuple[Path, dict[str, Any]]] = []
    base_name = base_config.stem

    for combo in combinations:
        config_dict = copy.deepcopy(base_dict)

        # Build suffix for filename
        suffix_parts = []
        for param, value in zip(param_names, combo, strict=False):
            # Apply the variation
            if "." in param:
                # Nested parameter like batching.batch_size
                parts = param.split(".")
                target = config_dict
                for part in parts[:-1]:
                    if part not in target:
                        target[part] = {}
                    target = target[part]
                target[parts[-1]] = value
            else:
                config_dict[param] = value

            # Clean value for filename
            clean_val = str(value).replace(".", "_")
            suffix_parts.append(f"{param.split('.')[-1]}_{clean_val}")

        # Update config name
        suffix = "_".join(suffix_parts)
        config_dict["config_name"] = f"{base_name}_{suffix}"
        output_path = output_dir / f"{base_name}_{suffix}.yaml"
        config_variations.append((output_path, config_dict))

    # Validate all configs first
    invalid_configs: list[tuple[Path, list[ConfigWarning] | str]] = []
    valid_configs: list[tuple[Path, dict[str, Any]]] = []

    for output_path, config_dict in config_variations:
        try:
            config_obj = ExperimentConfig(**config_dict)
            config_warnings = validate_config(config_obj)

            if has_blocking_warnings(config_warnings):
                invalid_configs.append((output_path, config_warnings))
            else:
                valid_configs.append((output_path, config_dict))

        except ValidationError as e:
            invalid_configs.append((output_path, str(e)))

    # --strict: fail before writing if any invalid
    if strict and invalid_configs:
        console.print(
            f"\n[yellow]⚠ {len(invalid_configs)} config(s) with blocking errors:[/yellow]"
        )
        for path, warnings_or_err in invalid_configs[:5]:
            console.print(f"  {path.name}:")
            if isinstance(warnings_or_err, list):
                for w in warnings_or_err:
                    if w.severity == "error":
                        console.print(f"    [red]{rich_escape(str(w))}[/red]")
            else:
                console.print(f"    [red]{rich_escape(str(warnings_or_err))}[/red]")
        if len(invalid_configs) > 5:
            console.print(f"  ... and {len(invalid_configs) - 5} more")
        console.print("\n[red]Error:[/red] --strict: Some configs have blocking errors")
        console.print("[dim]No configs were written.[/dim]")
        raise typer.Exit(1)

    # Write configs
    generated: list[Path] = []
    skipped_count = 0

    for output_path, config_dict in config_variations:
        # Check if this config was invalid
        is_invalid = any(p == output_path for p, _ in invalid_configs)

        if is_invalid and validate:
            skipped_count += 1
            continue  # Don't write invalid config

        # Write config
        with open(output_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        generated.append(output_path)

    # Report results
    console.print(f"[green]✓[/green] Generated {len(generated)} configs in {output_dir}/")

    if invalid_configs:
        console.print(
            f"\n[yellow]⚠ {len(invalid_configs)} config(s) with blocking errors:[/yellow]"
        )
        for path, warnings_or_err in invalid_configs[:5]:
            console.print(f"  {path.name}:")
            if isinstance(warnings_or_err, list):
                for w in warnings_or_err:
                    if w.severity == "error":
                        console.print(f"    [red]{rich_escape(str(w))}[/red]")
            else:
                console.print(f"    [red]{rich_escape(str(warnings_or_err))}[/red]")
        if len(invalid_configs) > 5:
            console.print(f"  ... and {len(invalid_configs) - 5} more")

    if validate and skipped_count:
        console.print(f"\n[dim]Skipped {skipped_count} invalid configs due to --validate[/dim]")

    if generated:
        console.print(
            f"\n[dim]Run with: lem batch {output_dir}/*.yaml --dataset alpaca -n 100[/dim]"
        )
