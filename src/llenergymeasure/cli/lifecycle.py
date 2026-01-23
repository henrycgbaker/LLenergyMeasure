"""Experiment lifecycle management utilities.

Provides subprocess orchestration with graceful signal handling,
process group management, and state transitions.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import typer

from llenergymeasure.cli.display import console
from llenergymeasure.constants import GRACEFUL_SHUTDOWN_TIMEOUT_SEC
from llenergymeasure.state.experiment_state import (
    ExperimentState,
    ExperimentStatus,
    StateManager,
)

if TYPE_CHECKING:
    from types import FrameType


@dataclass
class SubprocessRunner:
    """Manages subprocess execution with graceful signal handling.

    Handles:
    - Process group management for clean subprocess termination
    - SIGINT/SIGTERM signal handling with graceful shutdown
    - State transitions on interrupt/completion
    - Timeout-based force kill for unresponsive processes

    Usage:
        runner = SubprocessRunner(state_manager)
        exit_code = runner.run(
            cmd=["python", "-m", "some_module"],
            env=my_env,
            state=experiment_state,
        )
    """

    state_manager: StateManager
    shutdown_timeout_sec: int = GRACEFUL_SHUTDOWN_TIMEOUT_SEC

    # Internal state
    _subprocess: subprocess.Popen[bytes] | None = field(default=None, init=False)
    _current_state: ExperimentState | None = field(default=None, init=False)
    _interrupt_in_progress: bool = field(default=False, init=False)
    # Signal handlers - using Any for compatibility with signal.signal() return type
    _original_sigint: Any = field(default=None, init=False)
    _original_sigterm: Any = field(default=None, init=False)

    def run(
        self,
        cmd: list[str],
        env: dict[str, str] | None = None,
        state: ExperimentState | None = None,
    ) -> int:
        """Run subprocess with signal handling and process group management.

        Args:
            cmd: Command and arguments to execute.
            env: Environment variables for subprocess.
            state: Experiment state to update on interrupt.

        Returns:
            Exit code from subprocess.

        Raises:
            typer.Exit: On interrupt (exit code 130).
        """
        self._current_state = state
        self._interrupt_in_progress = False

        # Register signal handlers (save originals for restoration)
        self._original_sigint = signal.signal(signal.SIGINT, self._handle_interrupt)
        self._original_sigterm = signal.signal(signal.SIGTERM, self._handle_interrupt)

        try:
            # Start subprocess with its own process group for clean termination
            self._subprocess = subprocess.Popen(
                cmd,
                env=env,
                start_new_session=True,  # Creates new process group
            )

            # Wait for completion
            return self._subprocess.wait()

        finally:
            # Restore original signal handlers
            if self._original_sigint:
                signal.signal(signal.SIGINT, self._original_sigint)
            if self._original_sigterm:
                signal.signal(signal.SIGTERM, self._original_sigterm)
            self._subprocess = None

    def _handle_interrupt(self, signum: int, frame: FrameType | None) -> None:
        """Handle SIGINT/SIGTERM gracefully.

        Uses process groups to ensure all child processes (accelerate workers)
        are terminated, not just the parent process.
        """
        # Prevent re-entry from multiple Ctrl+C presses
        if self._interrupt_in_progress:
            console.print("[dim]Already shutting down, please wait...[/dim]")
            return
        self._interrupt_in_progress = True

        console.print("\n[yellow]Interrupt received, shutting down...[/yellow]")

        if self._subprocess and self._subprocess.poll() is None:
            # Get the process group ID (same as PID when start_new_session=True)
            pgid = os.getpgid(self._subprocess.pid)

            # Send SIGTERM to entire process group first
            console.print(
                f"[dim]Waiting up to {self.shutdown_timeout_sec}s for subprocess group...[/dim]"
            )
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGTERM)

            # Wait in 1-second increments for responsiveness
            for i in range(self.shutdown_timeout_sec):
                try:
                    self._subprocess.wait(timeout=1)
                    break  # Subprocess exited
                except subprocess.TimeoutExpired:
                    if i < self.shutdown_timeout_sec - 1:
                        console.print(f"[dim]...{self.shutdown_timeout_sec - i - 1}s[/dim]")
            else:
                # Timeout expired, force kill entire process group
                console.print("[red]Timeout, sending SIGKILL to process group...[/red]")
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(pgid, signal.SIGKILL)

                # Brief wait to let kernel reap the process
                with contextlib.suppress(subprocess.TimeoutExpired):
                    self._subprocess.wait(timeout=2)

        # Update state to INTERRUPTED
        if self._current_state:
            self._current_state.transition_to(
                ExperimentStatus.INTERRUPTED,
                error_message="Interrupted by user (SIGINT/SIGTERM)",
            )
            self.state_manager.save(self._current_state)
            console.print("\n[yellow]Experiment interrupted. Resume with:[/yellow]")
            console.print(f"  lem experiment --resume {self._current_state.experiment_id}")

        raise typer.Exit(130)  # Standard exit code for SIGINT


def build_pytorch_launch_cmd(
    num_processes: int,
    mixed_precision: str,
    dynamo_backend: str,
    module: str = "llenergymeasure.orchestration.launcher",
) -> list[str]:
    """Build accelerate launch command for PyTorch backend.

    Args:
        num_processes: Number of parallel processes.
        mixed_precision: One of 'fp16', 'bf16', 'no'.
        dynamo_backend: Torch dynamo backend ('inductor' or 'no').
        module: Python module to launch.

    Returns:
        Command list ready for subprocess.
    """
    import sys

    return [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        str(num_processes),
        "--num_machines",
        "1",
        "--mixed_precision",
        mixed_precision,
        "--dynamo_backend",
        dynamo_backend,
        "-m",
        module,
    ]


def build_vllm_launch_cmd(
    module: str = "llenergymeasure.orchestration.launcher",
) -> list[str]:
    """Build direct launch command for vLLM backend.

    vLLM manages its own distribution internally, so no accelerate needed.

    Args:
        module: Python module to launch.

    Returns:
        Command list ready for subprocess.
    """
    import sys

    return [
        sys.executable,
        "-m",
        module,
    ]


def build_subprocess_env(
    backend: str,
    gpus: list[int],
    verbosity: str | None = None,
) -> dict[str, str]:
    """Build environment for experiment subprocess.

    Args:
        backend: Inference backend ('pytorch', 'vllm', 'tensorrt').
        gpus: List of GPU indices to use.
        verbosity: Verbosity level ('quiet', 'normal', 'verbose').

    Returns:
        Environment dict for subprocess.
    """
    env = os.environ.copy()

    # Pass verbosity
    if verbosity:
        env["LLM_ENERGY_VERBOSITY"] = verbosity
    else:
        env["LLM_ENERGY_VERBOSITY"] = os.environ.get("LLM_ENERGY_VERBOSITY", "normal")

    # Propagate HuggingFace token for gated models
    if hf_token := os.environ.get("HF_TOKEN"):
        env["HF_TOKEN"] = hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token

    # Backend-specific settings
    if backend == "vllm":
        # vLLM v1 multiprocessing issues with CUDA initialization
        env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        # Disable torch.compile - requires C compiler
        env["TORCH_COMPILE_DISABLE"] = "1"
        # Set CUDA_VISIBLE_DEVICES for vLLM
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)

    return env


__all__ = [
    "SubprocessRunner",
    "build_pytorch_launch_cmd",
    "build_subprocess_env",
    "build_vllm_launch_cmd",
]
