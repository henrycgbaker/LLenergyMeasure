"""Subprocess-worker surface for experiment isolation.

Each experiment runs in a freshly spawned subprocess with a clean CUDA context.
This module holds the child-process entry point (``_run_experiment_worker``),
the parent-side result collector (``_collect_result``), and the small helpers
they share: process-group signalling, exit-reason derivation, and the failure
classifier constants.

Key design decisions (locked in .product/decisions/experiment-isolation.md):
- spawn context: CUDA-safe; fork causes silent CUDA corruption (CP-1)
- daemon=False: clean CUDA teardown if parent exits unexpectedly (CP-4)
- Pipe-only IPC: ExperimentResult fits in Pipe buffer for typical experiment sizes
- SIGKILL on timeout: SIGTERM may be ignored by hung CUDA operations
- Process group kill: worker calls os.setpgrp() to become group leader so all
  descendant processes (vLLM workers, MPI ranks, etc.) are killed together
"""

from __future__ import annotations

import contextlib
import os
import signal
import time
import traceback
from typing import TYPE_CHECKING, Any

from llenergymeasure.study._progress import _QueueProgressCallback

if TYPE_CHECKING:
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.domain.environment import EnvironmentSnapshot

# Failure classifiers produced by ``_collect_result`` and consumed by
# parent-side sentinel handling. Lifted so the producer and consumer don't
# drift on bare strings.
COLLECT_RESULT_PROCESS_CRASH = "ProcessCrash"
COLLECT_RESULT_TIMEOUT = "TimeoutError"

# Sentinel object used to distinguish "no payload provided" from a None payload.
_UNSET = object()


def _derive_exit_reason(exitcode: int | None) -> str | None:
    """Map a negative subprocess exit code to its POSIX signal name.

    Negative exit codes are signals (POSIX convention). Returns the signal
    name (e.g. ``SIGKILL``, ``SIGSEGV``, ``SIGTERM``) for any recognised
    signal, ``None`` otherwise. The raw code is preserved elsewhere.
    """
    if exitcode is None or exitcode >= 0:
        return None
    try:
        return signal.Signals(-exitcode).name
    except ValueError:
        return None


def _kill_process_group(pid: int, sig: int) -> None:
    """Send signal to the entire process group rooted at pid.

    Uses os.killpg so that all descendant processes (vLLM workers, MPI ranks, etc.)
    receive the signal, not just the parent. Errors are suppressed because the
    process group may already be dead by the time this is called.
    """
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, sig)


def _run_experiment_worker(
    config: ExperimentConfig,
    conn: Any,  # multiprocessing.Connection (child end)
    progress_queue: Any,  # multiprocessing.Queue
    snapshot: EnvironmentSnapshot | None = None,
    output_dir: str | None = None,
    save_timeseries: bool = True,
    baseline: Any = None,  # BaselineCache | None (avoids import at module level)
    *,
    study_dir: str,
    study_run_id: str,
    cycle: int,
    config_hash: str,
) -> None:
    """Entry point for the child process. Runs one experiment and returns result via Pipe.

    Signal handling:
        Installs SIGINT → SIG_IGN so the child ignores Ctrl+C.
        The parent handles SIGINT and decides whether to kill the child.

    IPC protocol:
        On success: sends ExperimentResult (or result dict) via conn.
        On failure: sends {"type": ..., "message": ..., "traceback": ...} via conn.
        Progress events are put to progress_queue for the consumer thread.

    Args:
        output_dir: Directory for timeseries parquet output. Passed through to harness.
        save_timeseries: Whether to persist GPU timeseries. Passed through to harness.
        baseline: Pre-measured baseline power from parent process (study-level cache).
        study_dir: Study output directory. Runtime observations append to
            ``{study_dir}/runtime_observations.jsonl``.
        study_run_id: UUID identifying this invocation of ``StudyRunner.run()``.
        cycle: 1-based cycle counter for this config within the study.
        config_hash: ``compute_declared_config_hash(config)``. The parent is
            the single SSOT for this value.
    """
    # Become process group leader so all descendants (vLLM workers, MPI ranks, etc.)
    # share this PGID. The parent can then kill the whole group via os.killpg().
    os.setpgrp()

    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Wrap the worker body BEFORE run_preflight() / get_engine() so engine
    # import-time warnings under ``spawn`` are captured.
    from llenergymeasure.study.runtime_observations import capture_runtime_observations

    obs_ctx = capture_runtime_observations(
        config,
        study_dir=study_dir,
        study_run_id=study_run_id,
        cycle=cycle,
        config_hash=config_hash,
    )

    try:
        with obs_ctx:
            progress_queue.put({"event": "started", "config_hash": config_hash})

            # Create progress callback that serialises step events to queue
            progress_cb = _QueueProgressCallback(progress_queue)

            # Run the actual experiment in-process (within the spawned subprocess)
            from llenergymeasure.engines import get_engine
            from llenergymeasure.harness import MeasurementHarness
            from llenergymeasure.harness.preflight import run_preflight

            # Pre-flight inside subprocess: CUDA availability must be checked in the
            # process that will use the GPU.
            progress_cb.on_step_start("container_preflight", "Checking", "CUDA, model access")
            t0_pf = time.perf_counter()
            run_preflight(config)
            progress_cb.on_step_done("container_preflight", time.perf_counter() - t0_pf)

            engine = get_engine(config.engine)
            harness = MeasurementHarness()
            from llenergymeasure.device.gpu_info import _resolve_gpu_indices

            gpu_indices = _resolve_gpu_indices(config)
            result = harness.run(
                engine,
                config,
                snapshot=snapshot,
                gpu_indices=gpu_indices,
                progress=progress_cb,
                output_dir=output_dir,
                save_timeseries=save_timeseries,
                baseline=baseline,
            )

            # Send result back to parent via Pipe
            conn.send(result)
            progress_queue.put({"event": "completed", "config_hash": config_hash})

    except Exception as exc:
        error_payload = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        with contextlib.suppress(Exception):
            # Pipe may be broken (e.g. parent killed). Best-effort only.
            conn.send(error_payload)

        with contextlib.suppress(Exception):
            progress_queue.put({"event": "failed", "error": str(exc)})

        raise

    finally:
        conn.close()


def _as_failure_payload(payload: Any, config_hash: str) -> dict[str, Any] | None:
    """Stamp and return ``payload`` if it is a worker error dict, else None.

    A worker error dict carries both ``type`` and ``message`` keys; the
    ``config_hash`` is added so the parent can attribute the failure.
    """
    if isinstance(payload, dict) and "type" in payload and "message" in payload:
        payload["config_hash"] = config_hash
        return payload
    return None


def _collect_result(
    p: Any,  # multiprocessing.Process
    parent_conn: Any,  # multiprocessing.Connection (parent end)
    config: ExperimentConfig,
    timeout: float,
    pipe_payload: Any = _UNSET,
) -> Any:
    """Inspect process outcome and return either a result or a failure dict.

    Called after the pipe has been drained and p.join() has returned.

    Args:
        p: The child process.
        parent_conn: Parent end of the Pipe (read-only).
        config: Experiment configuration.
        timeout: Timeout used for the experiment (for error messages).
        pipe_payload: Pre-drained pipe value from the recv-before-join pattern
            (H5 deadlock fix). When provided, skips calling recv() again.
            Pass _UNSET (default) to fall back to reading from the pipe directly.

    Returns:
        ExperimentResult on success, dict with keys (type, message) on failure.
    """
    from llenergymeasure.domain.experiment import compute_declared_config_hash

    config_hash = compute_declared_config_hash(config)

    if p.is_alive():
        # Timed out - kill with SIGKILL
        # SIGKILL: SIGTERM may be ignored by hung CUDA operations
        _kill_process_group(p.pid, signal.SIGKILL)
        p.join()
        return {
            "type": COLLECT_RESULT_TIMEOUT,
            "message": f"Experiment exceeded timeout of {timeout}s and was killed.",
            "config_hash": config_hash,
        }

    if p.exitcode != 0:
        # Non-zero exit - try to read error payload from pipe
        # Use pre-drained payload if available; otherwise poll/recv
        if pipe_payload is not _UNSET:
            failure = _as_failure_payload(pipe_payload, config_hash)
            if failure is not None:
                return failure
        elif parent_conn.poll():
            try:
                failure = _as_failure_payload(parent_conn.recv(), config_hash)
                if failure is not None:
                    return failure
            except Exception:
                pass

        return {
            "type": COLLECT_RESULT_PROCESS_CRASH,
            "message": f"Subprocess exited with code {p.exitcode} and no error data in Pipe.",
            "config_hash": config_hash,
        }

    # Success path - use pre-drained payload if available
    if pipe_payload is not _UNSET:
        # If payload is an error dict (exception in worker), treat as failure
        return _as_failure_payload(pipe_payload, config_hash) or pipe_payload

    # Fallback: read from pipe directly (no pre-drained payload)
    if parent_conn.poll():
        try:
            payload = parent_conn.recv()
            # If payload is an error dict (exception in worker), treat as failure
            return _as_failure_payload(payload, config_hash) or payload
        except Exception as exc:
            return {
                "type": "PipeError",
                "message": f"Failed to receive result from subprocess: {exc}",
                "config_hash": config_hash,
            }

    return {
        "type": COLLECT_RESULT_PROCESS_CRASH,
        "message": "Subprocess exited 0 but sent no data through Pipe.",
        "config_hash": config_hash,
    }
