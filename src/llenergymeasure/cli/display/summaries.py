"""Config and state summary rendering.

This module provides functions for displaying configuration summaries
and experiment state information.

Updated for backend-native configuration architecture.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from rich.table import Table

from llenergymeasure.cli.display.console import console
from llenergymeasure.cli.display.tables import (
    add_section_header,
    format_dict_field,
    print_value,
)
from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.config.provenance import ResolvedConfig
from llenergymeasure.constants import DEFAULT_STREAMING_WARMUP_REQUESTS

if TYPE_CHECKING:
    # ExperimentState is only used by display_incomplete_experiment(),
    # which is legacy v1.x code (called only from dead cli/experiment.py).
    # Keep under TYPE_CHECKING until that dead code is removed.
    from llenergymeasure.core.state import ExperimentState


def display_config_summary(
    config: ExperimentConfig,
    overrides: dict[str, Any],
    preset_name: str | None = None,
) -> None:
    """Display config summary with override visibility.

    In normal mode: Shows only non-default values (compact display).
    In verbose mode: Shows ALL configuration parameters.

    Visual styling:
    - Bold: Section headers
    - Cyan: Non-default values
    - Dim: Default values (verbose mode only)
    """
    # Determine if we should show defaults based on verbosity
    verbosity = os.environ.get("LLM_ENERGY_VERBOSITY", "normal")
    show_defaults = verbosity == "verbose"

    # In quiet mode, skip config display entirely
    if verbosity == "quiet":
        return

    console.print(f"\n[bold]Experiment: {config.config_name}[/bold]")
    if preset_name:
        console.print(f"  [cyan]Preset: {preset_name}[/cyan]")

    # Core settings - always show model
    console.print("  [bold]core:[/bold]")
    print_value("model", config.model_name, False, indent=4, show_defaults=True)  # Always show
    print_value(
        "adapter", config.adapter, config.adapter is None, indent=4, show_defaults=show_defaults
    )
    print_value(
        "backend",
        config.backend,
        config.backend == "pytorch",
        indent=4,
        show_defaults=show_defaults,
    )
    print_value(
        "precision",
        config.fp_precision,
        config.fp_precision == "float16",
        indent=4,
        show_defaults=show_defaults,
    )
    print_value(
        "gpus",
        str(config.gpus),
        config.gpus == [0],
        indent=4,
        show_defaults=show_defaults,
    )
    print_value(
        "seed",
        config.random_seed,
        config.random_seed is None,
        indent=4,
        show_defaults=show_defaults,
    )

    # Token/prompt settings
    if show_defaults or any(
        [
            config.num_input_prompts != 1,
            config.max_input_tokens != 512,
            config.max_output_tokens != 128,
            config.min_output_tokens != 0,
        ]
    ):
        console.print("  [bold]tokens:[/bold]")
        print_value(
            "num_prompts",
            config.num_input_prompts,
            config.num_input_prompts == 1,
            indent=4,
            show_defaults=show_defaults,
        )
        print_value(
            "max_input",
            config.max_input_tokens,
            config.max_input_tokens == 512,
            indent=4,
            show_defaults=show_defaults,
        )
        print_value(
            "max_output",
            config.max_output_tokens,
            config.max_output_tokens == 128,
            indent=4,
            show_defaults=show_defaults,
        )
        print_value(
            "min_output",
            config.min_output_tokens,
            config.min_output_tokens == 0,
            indent=4,
            show_defaults=show_defaults,
        )

    # Streaming settings
    if (
        show_defaults
        or config.streaming
        or config.streaming_warmup_requests != DEFAULT_STREAMING_WARMUP_REQUESTS
    ):
        console.print("  [bold]streaming:[/bold]")
        print_value(
            "enabled",
            config.streaming,
            config.streaming is False,
            indent=4,
            show_defaults=show_defaults,
        )
        print_value(
            "warmup_requests",
            config.streaming_warmup_requests,
            config.streaming_warmup_requests == DEFAULT_STREAMING_WARMUP_REQUESTS,
            indent=4,
            show_defaults=show_defaults,
        )

    # Backend-specific batching (PyTorch only)
    if config.backend == "pytorch" and config.pytorch is not None:
        pytorch_cfg = config.pytorch
        if show_defaults or any(
            [
                pytorch_cfg.batch_size != 1,
                pytorch_cfg.batching_strategy != "static",
                pytorch_cfg.max_tokens_per_batch is not None,
            ]
        ):
            console.print("  [bold]batching (pytorch):[/bold]")
            print_value(
                "batch_size",
                pytorch_cfg.batch_size,
                pytorch_cfg.batch_size == 1,
                indent=4,
                show_defaults=show_defaults,
            )
            print_value(
                "strategy",
                pytorch_cfg.batching_strategy,
                pytorch_cfg.batching_strategy == "static",
                indent=4,
                show_defaults=show_defaults,
            )
            print_value(
                "max_tokens_per_batch",
                pytorch_cfg.max_tokens_per_batch,
                pytorch_cfg.max_tokens_per_batch is None,
                indent=4,
                show_defaults=show_defaults,
            )

    # Backend-specific parallelism
    if config.backend == "pytorch" and config.pytorch is not None:
        pytorch_cfg = config.pytorch
        if show_defaults or pytorch_cfg.num_processes != 1:
            console.print("  [bold]parallelism (pytorch):[/bold]")
            strategy = "data_parallel" if pytorch_cfg.num_processes > 1 else "none"
            print_value(
                "strategy",
                strategy,
                strategy == "none",
                indent=4,
                show_defaults=show_defaults,
            )
            print_value(
                "num_processes",
                pytorch_cfg.num_processes,
                pytorch_cfg.num_processes == 1,
                indent=4,
                show_defaults=show_defaults,
            )
    elif config.backend == "vllm" and config.vllm is not None:
        vllm_cfg = config.vllm
        if (
            show_defaults
            or vllm_cfg.tensor_parallel_size != 1
            or vllm_cfg.pipeline_parallel_size != 1
        ):
            console.print("  [bold]parallelism (vllm):[/bold]")
            print_value(
                "tensor_parallel_size",
                vllm_cfg.tensor_parallel_size,
                vllm_cfg.tensor_parallel_size == 1,
                indent=4,
                show_defaults=show_defaults,
            )
            print_value(
                "pipeline_parallel_size",
                vllm_cfg.pipeline_parallel_size,
                vllm_cfg.pipeline_parallel_size == 1,
                indent=4,
                show_defaults=show_defaults,
            )
    elif config.backend == "tensorrt" and config.tensorrt is not None:
        trt_cfg = config.tensorrt
        if show_defaults or trt_cfg.tp_size != 1 or trt_cfg.pp_size != 1:
            console.print("  [bold]parallelism (tensorrt):[/bold]")
            print_value(
                "tp_size",
                trt_cfg.tp_size,
                trt_cfg.tp_size == 1,
                indent=4,
                show_defaults=show_defaults,
            )
            print_value(
                "pp_size",
                trt_cfg.pp_size,
                trt_cfg.pp_size == 1,
                indent=4,
                show_defaults=show_defaults,
            )

    # Traffic/pacing settings
    sim = config.traffic_simulation
    if show_defaults or any(
        [
            config.query_rate != 1.0,
            sim.enabled,
            sim.mode != "poisson",
            sim.target_qps != 1.0,
            sim.seed is not None,
        ]
    ):
        console.print("  [bold]traffic:[/bold]")
        print_value(
            "query_rate",
            config.query_rate,
            config.query_rate == 1.0,
            indent=4,
            show_defaults=show_defaults,
        )
        print_value(
            "simulation", sim.enabled, sim.enabled is False, indent=4, show_defaults=show_defaults
        )
        print_value("mode", sim.mode, sim.mode == "poisson", indent=4, show_defaults=show_defaults)
        print_value(
            "target_qps",
            sim.target_qps,
            sim.target_qps == 1.0,
            indent=4,
            show_defaults=show_defaults,
        )
        print_value("sim_seed", sim.seed, sim.seed is None, indent=4, show_defaults=show_defaults)

    # Decoder config (universal params)
    decoder = config.decoder
    if show_defaults or any(
        [
            decoder.preset is not None,
            decoder.temperature != 1.0,
            decoder.do_sample is not True,
            decoder.top_p != 1.0,
            decoder.top_k != 50,
            decoder.repetition_penalty != 1.0,
        ]
    ):
        console.print("  [bold]decoder:[/bold]")
        print_value(
            "preset", decoder.preset, decoder.preset is None, indent=4, show_defaults=show_defaults
        )
        mode = "deterministic (greedy)" if decoder.is_deterministic else "sampling"
        print_value(
            "mode",
            mode,
            decoder.temperature == 1.0 and decoder.do_sample,
            indent=4,
            show_defaults=show_defaults,
        )
        print_value(
            "temperature",
            decoder.temperature,
            decoder.temperature == 1.0,
            indent=4,
            show_defaults=show_defaults,
        )
        print_value(
            "do_sample",
            decoder.do_sample,
            decoder.do_sample is True,
            indent=4,
            show_defaults=show_defaults,
        )
        print_value(
            "top_p", decoder.top_p, decoder.top_p == 1.0, indent=4, show_defaults=show_defaults
        )
        print_value(
            "top_k", decoder.top_k, decoder.top_k == 50, indent=4, show_defaults=show_defaults
        )
        print_value(
            "repetition_penalty",
            decoder.repetition_penalty,
            decoder.repetition_penalty == 1.0,
            indent=4,
            show_defaults=show_defaults,
        )

    # Backend-specific decoder extensions
    if config.backend == "pytorch" and config.pytorch is not None:
        pytorch_cfg = config.pytorch
        if show_defaults or pytorch_cfg.min_p != 0.0 or pytorch_cfg.no_repeat_ngram_size != 0:
            console.print("  [bold]decoder extensions (pytorch):[/bold]")
            print_value(
                "min_p",
                pytorch_cfg.min_p,
                pytorch_cfg.min_p == 0.0,
                indent=4,
                show_defaults=show_defaults,
            )
            print_value(
                "no_repeat_ngram_size",
                pytorch_cfg.no_repeat_ngram_size,
                pytorch_cfg.no_repeat_ngram_size == 0,
                indent=4,
                show_defaults=show_defaults,
            )
    elif config.backend == "vllm" and config.vllm is not None:
        vllm_cfg = config.vllm
        if show_defaults or vllm_cfg.min_p != 0.0:
            console.print("  [bold]decoder extensions (vllm):[/bold]")
            print_value(
                "min_p",
                vllm_cfg.min_p,
                vllm_cfg.min_p == 0.0,
                indent=4,
                show_defaults=show_defaults,
            )

    # Backend-specific quantization
    if config.backend == "pytorch" and config.pytorch is not None:
        pytorch_cfg = config.pytorch
        if show_defaults or any(
            [
                pytorch_cfg.load_in_4bit,
                pytorch_cfg.load_in_8bit,
                pytorch_cfg.bnb_4bit_compute_dtype != "float16",
                pytorch_cfg.bnb_4bit_quant_type != "nf4",
                pytorch_cfg.bnb_4bit_use_double_quant,
            ]
        ):
            console.print("  [bold]quantization (pytorch):[/bold]")
            print_value(
                "load_in_4bit",
                pytorch_cfg.load_in_4bit,
                pytorch_cfg.load_in_4bit is False,
                indent=4,
                show_defaults=show_defaults,
            )
            print_value(
                "load_in_8bit",
                pytorch_cfg.load_in_8bit,
                pytorch_cfg.load_in_8bit is False,
                indent=4,
                show_defaults=show_defaults,
            )
            if pytorch_cfg.load_in_4bit:
                print_value(
                    "bnb_4bit_compute_dtype",
                    pytorch_cfg.bnb_4bit_compute_dtype,
                    pytorch_cfg.bnb_4bit_compute_dtype == "float16",
                    indent=4,
                    show_defaults=show_defaults,
                )
                print_value(
                    "bnb_4bit_quant_type",
                    pytorch_cfg.bnb_4bit_quant_type,
                    pytorch_cfg.bnb_4bit_quant_type == "nf4",
                    indent=4,
                    show_defaults=show_defaults,
                )
                print_value(
                    "bnb_4bit_use_double_quant",
                    pytorch_cfg.bnb_4bit_use_double_quant,
                    pytorch_cfg.bnb_4bit_use_double_quant is False,
                    indent=4,
                    show_defaults=show_defaults,
                )
    elif config.backend == "vllm" and config.vllm is not None:
        vllm_cfg = config.vllm
        if show_defaults or vllm_cfg.quantization is not None:
            console.print("  [bold]quantization (vllm):[/bold]")
            print_value(
                "quantization",
                vllm_cfg.quantization,
                vllm_cfg.quantization is None,
                indent=4,
                show_defaults=show_defaults,
            )
    elif config.backend == "tensorrt" and config.tensorrt is not None:
        trt_cfg = config.tensorrt
        if show_defaults or trt_cfg.quantization != "none":
            console.print("  [bold]quantization (tensorrt):[/bold]")
            print_value(
                "quantization",
                trt_cfg.quantization,
                trt_cfg.quantization == "none",
                indent=4,
                show_defaults=show_defaults,
            )

    # Schedule config
    sched = config.schedule
    if show_defaults or any(
        [
            sched.enabled,
            sched.interval is not None,
            sched.at is not None,
            sched.days is not None,
            sched.total_duration != "24h",
        ]
    ):
        console.print("  [bold]schedule:[/bold]")
        print_value(
            "cron_enabled",
            sched.enabled,
            sched.enabled is False,
            indent=4,
            show_defaults=show_defaults,
        )
        print_value(
            "interval",
            sched.interval,
            sched.interval is None,
            indent=4,
            show_defaults=show_defaults,
        )
        print_value("at", sched.at, sched.at is None, indent=4, show_defaults=show_defaults)
        days_str = ", ".join(sched.days) if sched.days else None
        print_value("days", days_str, sched.days is None, indent=4, show_defaults=show_defaults)
        print_value(
            "total_duration",
            sched.total_duration,
            sched.total_duration == "24h",
            indent=4,
            show_defaults=show_defaults,
        )

    # Prompt source (if configured) - always show when present
    if config.prompts is not None:
        console.print("  [bold]prompts:[/bold]")
        ps = config.prompts
        print_value("type", ps.type, False, indent=4, show_defaults=True)
        if ps.type == "file":
            print_value("path", ps.path, False, indent=4, show_defaults=True)
        else:  # huggingface
            print_value("dataset", ps.dataset, False, indent=4, show_defaults=True)
            print_value(
                "split", ps.split, ps.split == "train", indent=4, show_defaults=show_defaults
            )
            print_value(
                "subset", ps.subset, ps.subset is None, indent=4, show_defaults=show_defaults
            )
            print_value(
                "column", ps.column, ps.column is None, indent=4, show_defaults=show_defaults
            )
            print_value(
                "sample_size",
                ps.sample_size,
                ps.sample_size is None,
                indent=4,
                show_defaults=show_defaults,
            )
            print_value(
                "shuffle", ps.shuffle, ps.shuffle is False, indent=4, show_defaults=show_defaults
            )
            print_value("seed", ps.seed, ps.seed == 42, indent=4, show_defaults=show_defaults)

    # Show overrides
    if overrides:
        console.print("\n[dim]CLI overrides:[/dim]")
        for key, info in overrides.items():
            original = info.get("original")
            new = info.get("new")
            if original is not None and original != new:
                console.print(f"  {key}: {new} [dim](was: {original})[/dim]")
            else:
                console.print(f"  {key}: {new}")


def display_incomplete_experiment(state: ExperimentState) -> None:
    """Display information about an incomplete experiment."""
    console.print("\n[yellow]Incomplete experiment detected:[/yellow]")
    console.print(f"  Experiment ID: [cyan]{state.experiment_id}[/cyan]")
    if state.config_path:
        console.print(f"  Config: {state.config_path}")
    if state.model_name:
        console.print(f"  Model: {state.model_name}")
    if state.prompt_args:
        dataset = state.prompt_args.get("dataset", "")
        sample_size = state.prompt_args.get("sample_size", "")
        if dataset:
            console.print(f"  Dataset: {dataset}" + (f" (n={sample_size})" if sample_size else ""))
    if state.started_at:
        console.print(f"  Started: {state.started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    console.print(f"  Status: {state.status.value}")
    console.print(
        f"  Progress: {state.processes_completed}/{state.num_processes} processes completed"
    )


def show_effective_config(
    config: dict[str, Any], cli_overrides: dict[str, Any] | None = None
) -> None:
    """Display effective configuration from results.

    Shows ALL configuration parameters with visual styling:
    - Bold: Section headers
    - Bright/coloured: Non-default values
    - Dim: Default values
    """
    table = Table(title="Experiment Configuration")
    table.add_column("Field")
    table.add_column("Value")

    # Core settings
    add_section_header(table, "core")
    table.add_row(*format_dict_field("model", config.get("model_name", "N/A"), None, nested=True))
    table.add_row(*format_dict_field("adapter", config.get("adapter"), None, nested=True))
    table.add_row(
        *format_dict_field("backend", config.get("backend", "pytorch"), "pytorch", nested=True)
    )
    table.add_row(
        *format_dict_field(
            "precision", config.get("fp_precision", "float16"), "float16", nested=True
        )
    )
    gpus = config.get("gpus", [0])
    table.add_row(*format_dict_field("gpus", str(gpus), "[0]", nested=True))
    table.add_row(*format_dict_field("seed", config.get("random_seed"), None, nested=True))

    # Token settings
    add_section_header(table, "tokens")
    table.add_row(
        *format_dict_field("num_prompts", config.get("num_input_prompts", 1), 1, nested=True)
    )
    table.add_row(
        *format_dict_field("max_input", config.get("max_input_tokens", 512), 512, nested=True)
    )
    table.add_row(
        *format_dict_field("max_output", config.get("max_output_tokens", 128), 128, nested=True)
    )
    table.add_row(
        *format_dict_field("min_output", config.get("min_output_tokens", 0), 0, nested=True)
    )

    # Streaming settings
    add_section_header(table, "streaming")
    table.add_row(*format_dict_field("enabled", config.get("streaming", False), False, nested=True))
    table.add_row(
        *format_dict_field(
            "warmup_requests",
            config.get("streaming_warmup_requests", DEFAULT_STREAMING_WARMUP_REQUESTS),
            DEFAULT_STREAMING_WARMUP_REQUESTS,
            nested=True,
        )
    )

    # Backend-specific configs (handle dict from results)
    backend = config.get("backend", "pytorch")

    # Batching (PyTorch)
    if backend == "pytorch":
        pytorch_cfg = config.get("pytorch", {})
        if pytorch_cfg:
            add_section_header(table, "pytorch")
            table.add_row(
                *format_dict_field("batch_size", pytorch_cfg.get("batch_size", 1), 1, nested=True)
            )
            table.add_row(
                *format_dict_field(
                    "batching_strategy",
                    pytorch_cfg.get("batching_strategy", "static"),
                    "static",
                    nested=True,
                )
            )
            table.add_row(
                *format_dict_field(
                    "num_processes",
                    pytorch_cfg.get("num_processes", 1),
                    1,
                    nested=True,
                )
            )
            table.add_row(
                *format_dict_field(
                    "load_in_4bit", pytorch_cfg.get("load_in_4bit", False), False, nested=True
                )
            )
            table.add_row(
                *format_dict_field(
                    "load_in_8bit", pytorch_cfg.get("load_in_8bit", False), False, nested=True
                )
            )
    elif backend == "vllm":
        vllm_cfg = config.get("vllm", {})
        if vllm_cfg:
            add_section_header(table, "vllm")
            table.add_row(
                *format_dict_field(
                    "max_num_seqs", vllm_cfg.get("max_num_seqs", 256), 256, nested=True
                )
            )
            table.add_row(
                *format_dict_field(
                    "tensor_parallel_size", vllm_cfg.get("tensor_parallel_size", 1), 1, nested=True
                )
            )
            table.add_row(
                *format_dict_field("quantization", vllm_cfg.get("quantization"), None, nested=True)
            )
    elif backend == "tensorrt":
        trt_cfg = config.get("tensorrt", {})
        if trt_cfg:
            add_section_header(table, "tensorrt")
            table.add_row(
                *format_dict_field(
                    "max_batch_size", trt_cfg.get("max_batch_size", 8), 8, nested=True
                )
            )
            table.add_row(*format_dict_field("tp_size", trt_cfg.get("tp_size", 1), 1, nested=True))
            table.add_row(
                *format_dict_field(
                    "quantization", trt_cfg.get("quantization", "none"), "none", nested=True
                )
            )

    # Traffic/pacing settings
    add_section_header(table, "traffic")
    table.add_row(*format_dict_field("query_rate", config.get("query_rate", 1.0), 1.0, nested=True))
    traffic = config.get("traffic_simulation", {})
    table.add_row(
        *format_dict_field("simulation", traffic.get("enabled", False), False, nested=True)
    )
    table.add_row(
        *format_dict_field("mode", traffic.get("mode", "poisson"), "poisson", nested=True)
    )
    table.add_row(
        *format_dict_field("target_qps", traffic.get("target_qps", 1.0), 1.0, nested=True)
    )
    table.add_row(*format_dict_field("sim_seed", traffic.get("seed"), None, nested=True))

    # Schedule config
    add_section_header(table, "schedule")
    schedule = config.get("schedule", {})
    table.add_row(
        *format_dict_field("cron_enabled", schedule.get("enabled", False), False, nested=True)
    )
    table.add_row(*format_dict_field("interval", schedule.get("interval"), None, nested=True))
    table.add_row(*format_dict_field("at", schedule.get("at"), None, nested=True))
    days = schedule.get("days")
    days_str = ", ".join(days) if days else None
    table.add_row(*format_dict_field("days", days_str, None, nested=True))
    table.add_row(
        *format_dict_field(
            "total_duration", schedule.get("total_duration", "24h"), "24h", nested=True
        )
    )

    # Decoder config (universal)
    add_section_header(table, "decoder")
    decoder = config.get("decoder", {})
    table.add_row(*format_dict_field("preset", decoder.get("preset"), None, nested=True))
    is_deterministic = decoder.get("temperature", 1.0) == 0.0 or not decoder.get("do_sample", True)
    mode = "deterministic (greedy)" if is_deterministic else "sampling"
    default_mode = decoder.get("temperature", 1.0) == 1.0 and decoder.get("do_sample", True)
    table.add_row(
        *format_dict_field("mode", mode, "sampling" if default_mode else mode, nested=True)
    )
    table.add_row(
        *format_dict_field("temperature", decoder.get("temperature", 1.0), 1.0, nested=True)
    )
    table.add_row(
        *format_dict_field("do_sample", decoder.get("do_sample", True), True, nested=True)
    )
    table.add_row(*format_dict_field("top_p", decoder.get("top_p", 1.0), 1.0, nested=True))
    table.add_row(*format_dict_field("top_k", decoder.get("top_k", 50), 50, nested=True))
    table.add_row(
        *format_dict_field(
            "repetition_penalty", decoder.get("repetition_penalty", 1.0), 1.0, nested=True
        )
    )

    # Prompt source (if in results)
    ps = config.get("prompts")
    if ps:
        add_section_header(table, "prompts")
        table.add_row(*format_dict_field("type", ps.get("type", "file"), None, nested=True))
        if ps.get("type") == "file":
            table.add_row(*format_dict_field("path", ps.get("path"), None, nested=True))
        else:  # huggingface
            table.add_row(*format_dict_field("dataset", ps.get("dataset"), None, nested=True))
            table.add_row(
                *format_dict_field("split", ps.get("split", "train"), "train", nested=True)
            )
            table.add_row(*format_dict_field("subset", ps.get("subset"), None, nested=True))
            table.add_row(*format_dict_field("column", ps.get("column"), None, nested=True))
            table.add_row(
                *format_dict_field("sample_size", ps.get("sample_size"), None, nested=True)
            )
            table.add_row(
                *format_dict_field("shuffle", ps.get("shuffle", False), False, nested=True)
            )
            table.add_row(*format_dict_field("seed", ps.get("seed", 42), 42, nested=True))

    console.print(table)

    # Show CLI overrides if any
    if cli_overrides:
        console.print("\n[dim]CLI overrides:[/dim]")
        for key, override in cli_overrides.items():
            if isinstance(override, dict):
                console.print(f"  {key}: {override.get('original')} -> {override.get('new')}")
            else:
                console.print(f"  {key}: {override}")


def display_non_default_summary(resolved: ResolvedConfig) -> None:
    """Display a compact summary of non-default parameters before running.

    Shows all parameters that differ from Pydantic defaults, grouped by
    section. Useful for quick verification of experiment configuration.

    Args:
        resolved: ResolvedConfig with provenance information.
    """
    from collections import defaultdict

    from llenergymeasure.config.provenance import ParameterSource

    non_defaults = resolved.get_non_default_parameters()

    if not non_defaults:
        return

    console.print("\n[bold]Configuration summary (non-default values):[/bold]")

    # Group by top-level section
    by_section: dict[str, list[tuple[str, Any, str]]] = defaultdict(list)
    for prov in non_defaults:
        parts = prov.path.split(".")
        section = parts[0] if parts else "other"
        source_label = {
            ParameterSource.PRESET: "preset",
            ParameterSource.CONFIG_FILE: "config",
            ParameterSource.CLI: "CLI",
        }.get(prov.source, "")

        by_section[section].append((prov.path, prov.value, source_label))

    # Display grouped by section
    for section in sorted(by_section.keys()):
        params = by_section[section]
        console.print(f"  [bold]{section}:[/bold]")
        for path, value, source in sorted(params, key=lambda x: x[0]):
            # Show short path (without section prefix for cleaner display)
            short_path = path.split(".", 1)[-1] if "." in path else path
            source_str = f" [dim]({source})[/dim]" if source else ""
            console.print(f"    {short_path}: [cyan]{value!r}[/cyan]{source_str}")
