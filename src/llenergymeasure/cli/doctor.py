"""Diagnostic command for system environment status."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import typer

from llenergymeasure.cli.display import console

app = typer.Typer()


@app.command("doctor")  # type: ignore[misc]
def doctor_cmd() -> None:
    """Run diagnostic checks and print system environment status.

    Displays:
    - Python version and virtual environment status
    - CUDA/GPU availability
    - Installed backends
    - Docker status
    - Project files (.env, docker-compose.yml)
    """
    console.print("\n[bold]LLenergyMeasure Doctor[/bold]")
    console.print("═" * 50 + "\n")

    # 1. Python Environment
    _print_python_info()

    # 2. CUDA / GPU
    _print_cuda_info()

    # 3. Installed Backends
    _print_backends_info()

    # 4. Docker
    _print_docker_info()

    # 5. Project Files
    _print_project_files()

    console.print()


def _print_python_info() -> None:
    """Print Python environment information."""
    console.print("[bold cyan]Python[/bold cyan]")

    # Version
    python_version = sys.version.split()[0]
    console.print(f"  Version:     {python_version}")

    # Platform
    platform_str = platform.platform()
    console.print(f"  Platform:    {platform_str}")

    # Virtual environment
    venv_info = _get_venv_info()
    console.print(f"  Environment: {venv_info}\n")


def _get_venv_info() -> str:
    """Detect virtual environment type and name."""
    # Check if in virtual environment
    if sys.prefix != sys.base_prefix:
        # In a venv/virtualenv
        venv_path = Path(sys.prefix)
        return f"venv ({venv_path.name})"

    # Check for conda
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        conda_name = os.environ.get("CONDA_DEFAULT_ENV", Path(conda_prefix).name)
        return f"conda ({conda_name})"

    # Check for VIRTUAL_ENV
    virtual_env = os.environ.get("VIRTUAL_ENV")
    if virtual_env:
        return f"venv ({Path(virtual_env).name})"

    return "system (no virtual env)"


def _print_cuda_info() -> None:
    """Print CUDA and GPU information."""
    console.print("[bold cyan]CUDA / GPU[/bold cyan]")

    try:
        import torch

        if torch.cuda.is_available():
            # CUDA available
            cuda_version = torch.version.cuda or "unknown"
            console.print(f"  CUDA:        {cuda_version}")

            # GPU count and names
            gpu_count = torch.cuda.device_count()
            gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]

            if gpu_count == 1:
                console.print(f"  GPUs:        1 x {gpu_names[0]}")
            else:
                console.print(f"  GPUs:        {gpu_count} GPUs")
                for i, name in enumerate(gpu_names):
                    console.print(f"               [{i}] {name}")
        else:
            console.print("  CUDA:        not available")
    except ImportError:
        console.print("  CUDA:        torch not installed")
    except Exception as e:
        console.print(f"  CUDA:        error checking ({type(e).__name__})")

    console.print()


def _print_backends_info() -> None:
    """Print installed backend information."""
    # Import inside function to avoid forcing imports at module load
    from llenergymeasure.config.backend_detection import (
        KNOWN_BACKENDS,
        get_backend_install_hint,
        is_backend_available,
    )

    console.print("[bold cyan]Backends[/bold cyan]")

    for backend in KNOWN_BACKENDS:
        if is_backend_available(backend):
            console.print(f"  [green]✓[/green] {backend:<12} (installed)")
        else:
            hint = get_backend_install_hint(backend)
            # Escape hint to prevent Rich interpreting [backend] as markup tags
            escaped_hint = hint.replace("[", "\\[")
            console.print(f"  [red]✗[/red] {backend:<12} (not installed — {escaped_hint})")

    console.print()


def _print_docker_info() -> None:
    """Print Docker status information."""
    # Import inside function
    from llenergymeasure.config.docker_detection import is_inside_docker

    console.print("[bold cyan]Docker[/bold cyan]")

    # Check if inside Docker
    if is_inside_docker():
        console.print("  [dim]Running inside Docker container[/dim]\n")
        return

    # Outside Docker - check Docker availability
    # Binary
    docker_binary = shutil.which("docker")
    if docker_binary:
        console.print(f"  Binary:      [green]✓[/green] {docker_binary}")
    else:
        console.print("  Binary:      [red]✗[/red] not found")

    # Daemon running
    daemon_running = _check_docker_daemon()
    if daemon_running:
        console.print("  Daemon:      [green]✓[/green] running")
    else:
        console.print("  Daemon:      [red]✗[/red] not running")

    # GPU passthrough
    if docker_binary and daemon_running:
        gpu_access = _check_docker_gpu()
        if gpu_access:
            console.print("  GPU access:  [green]✓[/green] nvidia runtime configured")
        else:
            console.print("  GPU access:  [red]✗[/red] not available")
    else:
        console.print("  GPU access:  [dim]—[/dim] (docker not available)")

    console.print()


def _check_docker_daemon() -> bool:
    """Check if Docker daemon is running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
            check=False,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
        return False


def _check_docker_gpu() -> bool:
    """Check if NVIDIA Container Toolkit is configured for Docker GPU access.

    Checks `docker info` output for the nvidia runtime rather than pulling
    and running a CUDA image (which may not be cached and times out).
    """
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.Runtimes}}"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0:
            return False
        return "nvidia" in result.stdout.lower()
    except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
        return False


def _print_project_files() -> None:
    """Print project file status."""
    console.print("[bold cyan]Project[/bold cyan]")

    # .env file
    env_file = Path(".env")
    if env_file.exists():
        console.print("  .env:              [green]✓[/green] exists")
    else:
        console.print("  .env:              [red]✗[/red] not found")

    # docker-compose.yml
    compose_file = Path("docker-compose.yml")
    if compose_file.exists():
        console.print("  docker-compose.yml: [green]✓[/green] exists")
    else:
        console.print("  docker-compose.yml: [red]✗[/red] not found")
