"""Project initialization wizard command."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Any

import questionary
import typer
import yaml

from llenergymeasure.cli.display import console
from llenergymeasure.config.user_config import (
    DockerConfig,
    NotificationsConfig,
    ThermalGapConfig,
    UserConfig,
    load_user_config,
)

CONFIG_FILE = Path(".lem-config.yaml")


def _detect_environment() -> dict[str, Any]:
    """Detect environment info for init wizard display.

    Returns dict with:
    - python_version: str
    - in_venv: bool
    - venv_name: str | None
    - cuda_available: bool
    - gpu_count: int
    - gpu_names: list[str]
    - docker_available: bool
    - docker_running: bool
    - backends: dict[str, bool]  # backend_name -> is_available
    """
    env: dict[str, Any] = {}

    # Python version
    env["python_version"] = sys.version.split()[0]

    # Virtual environment
    if sys.prefix != sys.base_prefix:
        env["in_venv"] = True
        env["venv_name"] = Path(sys.prefix).name
    else:
        # Check for conda
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            env["in_venv"] = True
            env["venv_name"] = os.environ.get("CONDA_DEFAULT_ENV", Path(conda_prefix).name)
        else:
            env["in_venv"] = False
            env["venv_name"] = None

    # CUDA / GPU detection
    env["cuda_available"] = False
    env["gpu_count"] = 0
    env["gpu_names"] = []
    try:
        import torch

        if torch.cuda.is_available():
            env["cuda_available"] = True
            env["gpu_count"] = torch.cuda.device_count()
            env["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(env["gpu_count"])]
    except ImportError:
        pass
    except Exception:
        # Any other error - leave defaults
        pass

    # Docker detection
    env["docker_available"] = shutil.which("docker") is not None
    env["docker_running"] = False
    if env["docker_available"]:
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=5,
                check=False,
            )
            env["docker_running"] = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
            pass

    # Backend detection
    env["backends"] = {}
    try:
        from llenergymeasure.config.backend_detection import KNOWN_BACKENDS, is_backend_available

        for backend in KNOWN_BACKENDS:
            env["backends"][backend] = is_backend_available(backend)
    except ImportError:
        pass

    return env


def _display_environment(env: dict[str, Any]) -> None:
    """Display detected environment information."""
    console.print("\n[dim]Environment detected:[/dim]")

    # Python
    console.print(f"  Python: {env['python_version']}")

    # GPU
    if env["cuda_available"]:
        if env["gpu_count"] == 1:
            console.print(f"  GPU:    [green]✓[/green] 1 x {env['gpu_names'][0]}")
        else:
            console.print(f"  GPU:    [green]✓[/green] {env['gpu_count']} GPUs")
    else:
        console.print("  GPU:    [dim]not detected[/dim]")

    # Docker
    if env["docker_running"]:
        console.print("  Docker: [green]✓[/green] running")
    elif env["docker_available"]:
        console.print("  Docker: [yellow]![/yellow] installed but not running")
    else:
        console.print("  Docker: [dim]not installed[/dim]")

    # Backends with Docker requirements
    backends = env.get("backends", {})
    local_backends = []
    docker_backends = []
    for backend, available in backends.items():
        if backend == "pytorch":
            local_backends.append(("pytorch", available))
        else:
            docker_backends.append((backend, available))

    if local_backends or docker_backends:
        console.print("\n[dim]Backends:[/dim]")
        for backend, available in local_backends:
            status = "[green]✓[/green]" if available else "[dim]✗[/dim]"
            console.print(f"  {status} {backend} [dim](local)[/dim]")
        for backend, available in docker_backends:
            status = "[green]✓[/green]" if available else "[dim]✗[/dim]"
            console.print(f"  {status} {backend} [dim](requires Docker)[/dim]")

    console.print()


def _validate_url(url: str) -> bool | str:
    """Validate webhook URL format."""
    if not url:
        return True  # Empty is allowed
    if url.startswith(("http://", "https://")):
        return True
    return "Must start with http:// or https://"


def init_cmd(
    non_interactive: Annotated[
        bool, typer.Option("--non-interactive", help="Use defaults without prompts")
    ] = False,
    results_dir: Annotated[
        str | None, typer.Option("--results-dir", help="Results directory")
    ] = None,
    webhook_url: Annotated[
        str | None, typer.Option("--webhook-url", help="Webhook URL for notifications")
    ] = None,
) -> None:
    """Initialize project with guided configuration wizard."""
    console.print("\n[bold]LLenergyMeasure Configuration Wizard[/bold]")

    # Load existing config if present
    existing: UserConfig | None = None
    if CONFIG_FILE.exists():
        if non_interactive:
            # In non-interactive mode, just load and update
            existing = load_user_config(CONFIG_FILE)
        else:
            update = typer.confirm("Config already exists. Update it?", default=False)
            if not update:
                console.print("Cancelled.")
                return
            existing = load_user_config(CONFIG_FILE)

    # Detect environment
    env = _detect_environment()

    if non_interactive:
        # Non-interactive mode: use CLI args or existing values or defaults
        config = UserConfig(
            verbosity=existing.verbosity if existing else "normal",
            results_dir=results_dir or (existing.results_dir if existing else "results"),
            thermal_gaps=existing.thermal_gaps if existing else ThermalGapConfig(),
            docker=existing.docker if existing else DockerConfig(),
            notifications=NotificationsConfig(
                webhook_url=webhook_url
                or (existing.notifications.webhook_url if existing else None)
            ),
        )
    else:
        # Interactive mode
        _display_environment(env)

        # Get defaults from existing config or use model defaults
        defaults = existing or UserConfig()

        # Question 1: Verbosity level
        console.print(
            "[dim]Controls how much output you see during experiments and campaigns.[/dim]"
        )
        verbosity_choices = [
            questionary.Choice(
                title="normal (recommended) — Standard info messages with progress",
                value="normal",
            ),
            questionary.Choice(
                title="quiet — Warnings only, minimal output",
                value="quiet",
            ),
            questionary.Choice(
                title="verbose — Debug output, all backend logs",
                value="verbose",
            ),
        ]
        # Reorder choices to put default first
        default_idx = next(
            (i for i, c in enumerate(verbosity_choices) if c.value == defaults.verbosity),
            0,
        )
        verbosity_choices = (
            [verbosity_choices[default_idx]]
            + verbosity_choices[:default_idx]
            + verbosity_choices[default_idx + 1 :]
        )
        verbosity_answer = questionary.select(
            "Verbosity level:",
            choices=verbosity_choices,
            default=defaults.verbosity,
        ).ask()
        if verbosity_answer is None:
            raise typer.Abort()
        console.print()

        # Question 2: Results directory
        console.print(
            "[dim]Path relative to project root where experiment results are saved.[/dim]"
        )
        results_answer = questionary.text(
            "Results directory:",
            default=defaults.results_dir,
        ).ask()
        if results_answer is None:
            raise typer.Abort()
        # Show resolved path for clarity
        resolved_path = Path(results_answer).resolve()
        console.print(f"[dim]  → {resolved_path}[/dim]\n")

        # Question 2: Thermal gap between experiments
        thermal_answer = questionary.text(
            "Thermal gap between experiments (seconds):",
            default=str(int(defaults.thermal_gaps.between_experiments)),
        ).ask()
        if thermal_answer is None:
            raise typer.Abort()
        try:
            thermal_gap = float(thermal_answer)
        except ValueError:
            thermal_gap = defaults.thermal_gaps.between_experiments

        # Question 2b: Thermal gap between cycles
        thermal_cycles_answer = questionary.text(
            "Thermal gap between cycles (seconds):",
            default=str(int(defaults.thermal_gaps.between_cycles)),
        ).ask()
        if thermal_cycles_answer is None:
            raise typer.Abort()
        try:
            thermal_gap_cycles = float(thermal_cycles_answer)
        except ValueError:
            thermal_gap_cycles = defaults.thermal_gaps.between_cycles

        # Question 3: Docker strategy (only if Docker available)
        docker_strategy = defaults.docker.strategy
        if env["docker_available"]:
            console.print(
                "[dim]Container lifecycle for multi-backend experiments (vLLM, TensorRT).[/dim]"
            )
            strategy_choices = [
                questionary.Choice(
                    title="ephemeral (recommended) — Fresh container per experiment. Reproducible.",
                    value="ephemeral",
                ),
                questionary.Choice(
                    title="persistent — Keep container running. Faster, but may have state carryover.",
                    value="persistent",
                ),
            ]
            strategy_answer = questionary.select(
                "Docker container strategy:",
                choices=strategy_choices,
                default="ephemeral" if defaults.docker.strategy == "ephemeral" else "persistent",
            ).ask()
            if strategy_answer is None:
                raise typer.Abort()
            docker_strategy = strategy_answer
            console.print()

        # Question 4: Webhook URL
        console.print(
            "[dim]Receive HTTP POST notifications when experiments complete or fail.[/dim]"
        )
        console.print("[dim]Leave blank to disable. Example: https://hooks.slack.com/...[/dim]")
        webhook_answer = questionary.text(
            "Webhook URL (optional):",
            default=defaults.notifications.webhook_url or "",
            validate=_validate_url,
        ).ask()
        if webhook_answer is None:
            raise typer.Abort()

        # Question 4b/4c: Webhook toggles (only if webhook URL provided)
        on_complete = defaults.notifications.on_complete
        on_failure = defaults.notifications.on_failure
        if webhook_answer:
            on_complete = questionary.confirm(
                "Send notification on completion?",
                default=defaults.notifications.on_complete,
            ).ask()
            if on_complete is None:
                raise typer.Abort()

            on_failure = questionary.confirm(
                "Send notification on failure?",
                default=defaults.notifications.on_failure,
            ).ask()
            if on_failure is None:
                raise typer.Abort()

        console.print()

        # Build config
        config = UserConfig(
            verbosity=verbosity_answer,
            results_dir=results_answer,
            thermal_gaps=ThermalGapConfig(
                between_experiments=thermal_gap,
                between_cycles=thermal_gap_cycles,
            ),
            docker=DockerConfig(
                strategy=docker_strategy,
                warmup_delay=defaults.docker.warmup_delay,
                auto_teardown=defaults.docker.auto_teardown,
            ),
            notifications=NotificationsConfig(
                webhook_url=webhook_answer if webhook_answer else None,
                on_complete=on_complete,
                on_failure=on_failure,
            ),
        )

    # Write config
    config_data = config.model_dump(exclude_none=True)
    # Clean up empty nested dicts for cleaner YAML
    if config_data.get("notifications", {}).get("webhook_url") is None:
        config_data.get("notifications", {}).pop("webhook_url", None)

    with open(CONFIG_FILE, "w") as f:
        yaml.safe_dump(config_data, f, default_flow_style=False, sort_keys=False)

    console.print(f"[green]✓[/green] Config written to {CONFIG_FILE}")

    # Multi-backend info panel (always shown)
    console.print("\n[bold]Multi-Backend Setup[/bold]")
    console.print("[dim]─────────────────────────────────────────────────────────────[/dim]")
    console.print("PyTorch works locally with your current installation.")
    console.print("")
    console.print(
        "[yellow]vLLM and TensorRT require Docker[/yellow] due to conflicting CUDA dependencies."
    )
    if env["docker_running"]:
        console.print("Docker detected. To enable multi-backend campaigns:")
        console.print("  [cyan]docker compose build vllm tensorrt[/cyan]")
        console.print(
            "  [cyan]lem campaign config.yaml[/cyan]  [dim]# auto-dispatches to containers[/dim]"
        )
    elif env["docker_available"]:
        console.print("Docker installed but not running. Start Docker, then:")
        console.print("  [cyan]docker compose build vllm tensorrt[/cyan]")
    else:
        console.print("To use vLLM/TensorRT backends:")
        console.print("  1. Install Docker: https://docs.docker.com/get-docker/")
        console.print("  2. [cyan]docker compose build vllm tensorrt[/cyan]")
    console.print("[dim]─────────────────────────────────────────────────────────────[/dim]")

    # Run doctor
    console.print("\n[dim]Running diagnostics...[/dim]")
    from llenergymeasure.cli.doctor import doctor_cmd

    doctor_cmd()

    # Next steps
    console.print("\nRun `lem --help` to get started.")
