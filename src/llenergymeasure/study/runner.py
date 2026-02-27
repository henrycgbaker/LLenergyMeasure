"""StudyRunner — subprocess dispatch core for experiment isolation.

Each experiment in a study runs in a freshly spawned subprocess with a clean CUDA
context. Results travel parent←child via multiprocessing.Pipe. The parent survives
experiment failures, timeouts, and SIGINT without data corruption.

Key design decisions (locked in .product/decisions/experiment-isolation.md):
- spawn context: CUDA-safe; fork causes silent CUDA corruption (CP-1)
- daemon=False: clean CUDA teardown if parent exits unexpectedly (CP-4)
- Pipe-only IPC: ExperimentResult fits in Pipe buffer for M2 experiment sizes
- SIGKILL on timeout: SIGTERM may be ignored by hung CUDA operations
"""

from __future__ import annotations

import multiprocessing
import signal
import sys
import threading
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llenergymeasure.study.gaps import run_gap

if TYPE_CHECKING:
    from llenergymeasure.config.models import ExperimentConfig, StudyConfig
    from llenergymeasure.study.manifest import ManifestWriter

__all__ = ["StudyRunner", "_calculate_timeout", "_run_experiment_worker"]


# =============================================================================
# Module-level helpers
# =============================================================================


def _calculate_timeout(config: ExperimentConfig) -> int:
    """Generous timeout heuristic: 2 seconds per prompt, minimum 10 minutes.

    No model-size scaling — keep it simple. The escape hatch is
    execution.experiment_timeout_seconds in the study YAML.
    """
    return max(config.n * 2, 600)


# =============================================================================
# Worker function (runs inside child process)
# =============================================================================


def _run_experiment_worker(
    config: ExperimentConfig,
    conn: Any,  # multiprocessing.Connection (child end)
    progress_queue: Any,  # multiprocessing.Queue
) -> None:
    """Entry point for the child process. Runs one experiment and returns result via Pipe.

    Signal handling:
        Installs SIGINT → SIG_IGN so the child ignores Ctrl+C.
        # parent owns SIGINT; child ignores it
        The parent handles SIGINT and decides whether to kill the child.

    IPC protocol:
        On success: sends ExperimentResult (or result dict) via conn.
        On failure: sends {"type": ..., "message": ..., "traceback": ...} via conn.
        Progress events are put to progress_queue for the consumer thread.
    """
    # parent owns SIGINT; child ignores it
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        from llenergymeasure.domain.experiment import compute_measurement_config_hash

        config_hash = compute_measurement_config_hash(config)
        progress_queue.put({"event": "started", "config_hash": config_hash})

        # Run the actual experiment in-process (within the spawned subprocess)
        from llenergymeasure.core.backends import get_backend
        from llenergymeasure.orchestration.preflight import run_preflight

        # Pre-flight inside subprocess: CUDA availability must be checked in the
        # process that will use the GPU.
        run_preflight(config)

        backend = get_backend(config.backend)
        result = backend.run(config)

        # Send result back to parent via Pipe
        conn.send(result)
        progress_queue.put({"event": "completed", "config_hash": config_hash})

    except Exception as exc:
        error_payload = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        try:
            conn.send(error_payload)
        except Exception:
            # Pipe may be broken (e.g. parent killed). Best-effort only.
            pass

        try:
            progress_queue.put({"event": "failed", "error": str(exc)})
        except Exception:
            pass

        raise

    finally:
        conn.close()


# =============================================================================
# Progress consumer (daemon thread in parent)
# =============================================================================


def _consume_progress_events(q: Any) -> None:
    """Consume and discard progress events from the queue until None sentinel.

    Runs as a daemon thread. Display wiring is Phase 12 — this stub just
    drains the queue so the child never blocks on a full Queue.
    """
    while True:
        event = q.get()
        if event is None:
            break
        # Phase 12: forward to Rich display layer here


# =============================================================================
# Result collection (parent, after p.join)
# =============================================================================


def _collect_result(
    p: Any,  # multiprocessing.Process
    parent_conn: Any,  # multiprocessing.Connection (parent end)
    config: ExperimentConfig,
    timeout: int,
) -> Any:
    """Inspect process outcome and return either a result or a failure dict.

    Called after p.join(timeout=...) has returned.

    Returns:
        ExperimentResult on success, dict with keys (type, message) on failure.
    """
    from llenergymeasure.domain.experiment import compute_measurement_config_hash

    config_hash = compute_measurement_config_hash(config)

    if p.is_alive():
        # Timed out — kill with SIGKILL
        # SIGKILL: SIGTERM may be ignored by hung CUDA operations
        p.kill()
        p.join()
        return {
            "type": "TimeoutError",
            "message": f"Experiment exceeded timeout of {timeout}s and was killed.",
            "config_hash": config_hash,
        }

    if p.exitcode != 0:
        # Non-zero exit — try to read error payload from pipe
        if parent_conn.poll():
            try:
                payload = parent_conn.recv()
                if isinstance(payload, dict) and "type" in payload:
                    payload["config_hash"] = config_hash
                    return payload
            except Exception:
                pass

        return {
            "type": "ProcessCrash",
            "message": f"Subprocess exited with code {p.exitcode} and no error data in Pipe.",
            "config_hash": config_hash,
        }

    # Success path — read result from pipe
    if parent_conn.poll():
        try:
            payload = parent_conn.recv()
            # If payload is an error dict (exception in worker), treat as failure
            if isinstance(payload, dict) and "type" in payload and "traceback" in payload:
                payload["config_hash"] = config_hash
                return payload
            return payload
        except Exception as exc:
            return {
                "type": "PipeError",
                "message": f"Failed to receive result from subprocess: {exc}",
                "config_hash": config_hash,
            }

    return {
        "type": "ProcessCrash",
        "message": "Subprocess exited 0 but sent no data through Pipe.",
        "config_hash": config_hash,
    }


# =============================================================================
# StudyRunner
# =============================================================================


class StudyRunner:
    """Dispatcher: runs each experiment in a freshly spawned subprocess.

    Uses multiprocessing.get_context('spawn') — never fork.
    Results travel via Pipe. Failures are structured and non-fatal.
    Handles SIGINT (Ctrl+C) with two-stage escalation: SIGTERM → 2s grace → SIGKILL.
    """

    def __init__(
        self, study: StudyConfig, manifest_writer: ManifestWriter, study_dir: Path
    ) -> None:
        self.study = study
        self.manifest = manifest_writer
        self.study_dir = study_dir
        self.result_files: list[str] = []
        # SIGINT state — initialised here, set live in run()
        self._interrupt_event: threading.Event = threading.Event()
        self._active_process: Any = None  # multiprocessing.Process | None
        self._interrupt_count: int = 0

    def run(self) -> list[Any]:
        """Run all experiments in order; return list of results or failure dicts.

        Installs a SIGINT handler for the duration of the run. First Ctrl+C sends
        SIGTERM to the active subprocess and sets interrupt_event. Second Ctrl+C (or
        grace period expiry) sends SIGKILL. After the loop exits, if interrupted,
        calls manifest.mark_interrupted() and sys.exit(130).
        """
        from llenergymeasure.study.grid import CycleOrder, apply_cycles

        ordered = apply_cycles(
            self.study.experiments,
            self.study.execution.n_cycles,
            CycleOrder(self.study.execution.cycle_order),
            self.study.study_design_hash or "",
            self.study.execution.shuffle_seed,
        )

        # spawn: CUDA-safe; fork causes silent CUDA corruption (CP-1)
        mp_ctx = multiprocessing.get_context("spawn")

        # Reset interrupt state for this run
        self._interrupt_event.clear()
        self._interrupt_count = 0
        self._active_process = None

        def _sigint_handler(signum: int, frame: Any) -> None:
            self._interrupt_count += 1
            self._interrupt_event.set()
            if self._interrupt_count == 1:
                print(
                    "\nInterrupt received. Waiting for experiment to finish cleanly "
                    "(Ctrl+C again to force)..."
                )
                if self._active_process is not None and self._active_process.is_alive():
                    self._active_process.terminate()  # SIGTERM — gentle first attempt
            else:
                print("\nForce-killing experiment subprocess...")
                if self._active_process is not None and self._active_process.is_alive():
                    self._active_process.kill()  # SIGKILL

        original_sigint = signal.signal(signal.SIGINT, _sigint_handler)

        try:
            results: list[Any] = []
            n_unique = len(self.study.experiments)

            for i, config in enumerate(ordered):
                if self._interrupt_event.is_set():
                    break

                # Config gap: between every consecutive experiment pair
                if i > 0:
                    gap_secs = float(self.study.execution.experiment_gap_seconds or 0)
                    if gap_secs > 0:
                        run_gap(gap_secs, "Experiment gap", self._interrupt_event)
                        if self._interrupt_event.is_set():
                            break

                # Cycle gap: after every complete round of N unique configs
                if n_unique > 0 and i > 0 and i % n_unique == 0:
                    cycle_gap_secs = float(self.study.execution.cycle_gap_seconds or 0)
                    if cycle_gap_secs > 0:
                        run_gap(cycle_gap_secs, "Cycle gap", self._interrupt_event)
                        if self._interrupt_event.is_set():
                            break

                result = self._run_one(config, mp_ctx)
                results.append(result)

        finally:
            signal.signal(signal.SIGINT, original_sigint)

        if self._interrupt_event.is_set():
            completed = sum(1 for r in results if not isinstance(r, dict))
            total = len(ordered)
            print(
                f"\n{completed}/{total} experiments completed. "
                "Results in study directory. Manifest: interrupted."
            )
            self.manifest.mark_interrupted()
            sys.exit(130)

        return results

    def _run_one(self, config: ExperimentConfig, mp_ctx: Any) -> Any:
        """Spawn a subprocess for one experiment; collect result or failure dict.

        If interrupt_event is set after join, attempts graceful SIGTERM → 2s grace →
        SIGKILL before collecting whatever result is available.
        """
        from llenergymeasure.domain.experiment import compute_measurement_config_hash

        config_hash = compute_measurement_config_hash(config)
        cycle = 1  # cycle tracking deferred to Phase 12 wiring

        # Use user-supplied timeout if set, otherwise fall back to heuristic
        user_timeout = getattr(self.study.execution, "experiment_timeout_seconds", None)
        timeout = int(user_timeout) if user_timeout is not None else _calculate_timeout(config)

        child_conn, parent_conn = mp_ctx.Pipe(duplex=False)
        progress_queue = mp_ctx.Queue()

        p = mp_ctx.Process(
            target=_run_experiment_worker,
            args=(config, child_conn, progress_queue),
            daemon=False,  # daemon=False: clean CUDA teardown if parent exits unexpectedly
        )

        consumer = threading.Thread(
            target=_consume_progress_events,
            args=(progress_queue,),
            daemon=True,
        )
        consumer.start()

        self.manifest.mark_running(config_hash, cycle)
        self._active_process = p

        p.start()
        child_conn.close()
        p.join(timeout=timeout)

        # SIGINT was received during join: SIGTERM was already sent by handler.
        # Give child 2s grace for clean CUDA teardown, then SIGKILL.
        if self._interrupt_event.is_set() and p.is_alive():
            p.join(timeout=2)  # 2s grace after SIGTERM
            if p.is_alive():
                p.kill()
                p.join()

        self._active_process = None

        # Sentinel stops consumer thread — covers SIGKILL path too
        progress_queue.put(None)
        consumer.join()

        result = _collect_result(p, parent_conn, config, timeout)

        # Update manifest based on outcome
        if isinstance(result, dict) and "type" in result:
            error_type = result.get("type", "UnknownError")
            error_message = result.get("message", "")
            self.manifest.mark_failed(config_hash, cycle, error_type, error_message)
        else:
            # Save result to study directory and track path (RES-15)
            try:
                from llenergymeasure.results.persistence import save_result

                result_path = save_result(result, self.study_dir)
                self.result_files.append(str(result_path))
                rel_path = str(result_path.relative_to(self.study_dir))
                self.manifest.mark_completed(config_hash, cycle, rel_path)
            except Exception:
                # Save failure does not abort the study
                self.manifest.mark_completed(config_hash, cycle, result_file="")

        return result
