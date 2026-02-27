"""Subprocess lifecycle management for llenergymeasure.

Provides signal handling, process group management, and graceful shutdown
for experiment subprocesses. Extracted from cli/lifecycle.py (v1.x) and
updated for v2.0: no Rich, no Typer, no loguru dependencies.
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import subprocess
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from llenergymeasure.core.state import ExperimentState, StateManager

if TYPE_CHECKING:
    from types import FrameType

logger = logging.getLogger(__name__)

# Default grace period before SIGKILL is sent to an unresponsive process group.
_DEFAULT_SHUTDOWN_TIMEOUT_SEC = 2


@dataclass
class SubprocessRunner:
    """Manages subprocess execution with graceful signal handling.

    Handles:
    - Process group management for clean subprocess termination
    - SIGINT/SIGTERM signal handling with graceful shutdown
    - State updates on interrupt (mark_failed rather than INTERRUPTED transition)
    - Timeout-based force kill for unresponsive processes

    Args:
        state_manager: StateManager used to persist state changes on interrupt.
        shutdown_timeout_sec: Seconds to wait for graceful shutdown before SIGKILL.

    Usage:
        runner = SubprocessRunner(state_manager)
        exit_code = runner.run(
            cmd=["python", "-m", "some_module"],
            env=my_env,
            state=experiment_state,
        )
    """

    state_manager: StateManager
    shutdown_timeout_sec: int = _DEFAULT_SHUTDOWN_TIMEOUT_SEC

    # Internal state (not init params)
    _subprocess: subprocess.Popen[bytes] | None = field(default=None, init=False)
    _current_state: ExperimentState | None = field(default=None, init=False)
    _interrupt_in_progress: bool = field(default=False, init=False)
    _original_sigint: Any = field(default=None, init=False)
    _original_sigterm: Any = field(default=None, init=False)

    def run(
        self,
        cmd: list[str],
        env: dict[str, str] | None = None,
        state: ExperimentState | None = None,
    ) -> int:
        """Run a subprocess with signal handling and process group management.

        Args:
            cmd: Command and arguments to execute.
            env: Environment variables for the subprocess. Defaults to inherited env.
            state: Experiment state to update on interrupt (optional).

        Returns:
            Exit code from the subprocess.

        Raises:
            SystemExit: With code 130 on SIGINT/SIGTERM (standard shell convention).
        """
        self._current_state = state
        self._interrupt_in_progress = False

        # Register signal handlers, saving originals for restoration.
        self._original_sigint = signal.signal(signal.SIGINT, self._handle_interrupt)
        self._original_sigterm = signal.signal(signal.SIGTERM, self._handle_interrupt)

        try:
            # start_new_session=True creates a new process group, enabling
            # os.killpg() to terminate all child processes in one call.
            self._subprocess = subprocess.Popen(
                cmd,
                env=env,
                start_new_session=True,
            )
            return self._subprocess.wait()

        finally:
            # Always restore original signal handlers.
            if self._original_sigint is not None:
                signal.signal(signal.SIGINT, self._original_sigint)
            if self._original_sigterm is not None:
                signal.signal(signal.SIGTERM, self._original_sigterm)
            self._subprocess = None

    def _handle_interrupt(self, signum: int, frame: FrameType | None) -> None:
        """Handle SIGINT or SIGTERM with graceful shutdown.

        Sends SIGTERM to the entire process group, waits up to
        ``shutdown_timeout_sec`` seconds, then sends SIGKILL if needed.
        Updates experiment state to failed before exiting.
        """
        # Prevent re-entry from multiple Ctrl+C presses.
        if self._interrupt_in_progress:
            print("Already shutting down, please wait...")
            return
        self._interrupt_in_progress = True

        print("\nInterrupt received, shutting down...")

        if self._subprocess is not None and self._subprocess.poll() is None:
            # Get the process group ID (same as PID when start_new_session=True).
            try:
                pgid = os.getpgid(self._subprocess.pid)
            except ProcessLookupError:
                pgid = None

            if pgid is not None:
                print(f"Waiting up to {self.shutdown_timeout_sec}s for subprocess group...")
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(pgid, signal.SIGTERM)

                # Wait in 1-second increments to stay responsive to further signals.
                for i in range(self.shutdown_timeout_sec):
                    try:
                        self._subprocess.wait(timeout=1)
                        break
                    except subprocess.TimeoutExpired:
                        remaining = self.shutdown_timeout_sec - i - 1
                        if remaining > 0:
                            print(f"...{remaining}s")
                else:
                    # Timeout expired — force kill the entire process group.
                    print("Timeout expired, sending SIGKILL to process group...")
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(pgid, signal.SIGKILL)
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        self._subprocess.wait(timeout=2)

        # Update experiment state so the manager knows it was interrupted.
        if self._current_state is not None:
            self._current_state.mark_failed("Interrupted by user (SIGINT/SIGTERM)")
            try:
                self.state_manager.save(self._current_state)
                print("\nExperiment interrupted. Resume with:")
                print(f"  llem run --resume {self._current_state.experiment_id}")
            except Exception as exc:
                logger.warning("Could not persist interrupted state: %s", exc)

        raise SystemExit(130)  # Standard exit code for SIGINT


def build_subprocess_env(
    backend: str,
    gpus: list[int],
    verbosity: str | None = None,
) -> dict[str, str]:
    """Build an environment dict for an experiment subprocess.

    Inherits the current environment and applies backend-specific overrides.

    Args:
        backend: Inference backend name ('pytorch', 'vllm', 'tensorrt').
        gpus: GPU device indices to expose to the subprocess.
        verbosity: Verbosity level ('quiet', 'normal', 'verbose'). Inherits
            from ``LLM_ENERGY_VERBOSITY`` env var if not specified.

    Returns:
        A copy of the current environment with backend-specific additions.
    """
    env = os.environ.copy()

    # Propagate verbosity setting.
    env["LLM_ENERGY_VERBOSITY"] = verbosity or os.environ.get("LLM_ENERGY_VERBOSITY", "normal")

    # Propagate HuggingFace token for gated models.
    if hf_token := os.environ.get("HF_TOKEN"):
        env["HF_TOKEN"] = hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token

    # Backend-specific overrides.
    if backend == "vllm":
        env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        env["TORCH_COMPILE_DISABLE"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)

    return env


__all__ = [
    "SubprocessRunner",
    "build_subprocess_env",
]
