# Experiment Process Isolation Design

**Last updated**: 2026-02-25
**Source decisions**: [../decisions/experiment-isolation.md](../decisions/experiment-isolation.md)
**Target version**: v2.0 (local + Docker multi-backend as later milestone)
**Status**: Confirmed

---

## Why: GPU State Is Process-Bound

PyTorch's CUDA allocator does not fully release GPU memory across `del model; torch.cuda.empty_cache()`.
Memory fragmentation, cached allocations, and CUDA context state persist within a process.
A fresh `multiprocessing.Process` with `spawn` start method guarantees a clean CUDA context.
There is no other reliable mechanism.

`llem run experiment.yaml` (single experiment): runs `ExperimentOrchestrator` in-process — no
subprocess needed (clean state at start is guaranteed for a single run).

`llem run study.yaml` (multi-experiment): each experiment runs in a fresh
`multiprocessing.Process` via `StudyRunner`. Clean GPU state between experiments is a hard
correctness requirement.

**Reference**: Optimum-benchmark (HuggingFace) makes this a core design principle for the same
reason. See [../decisions/experiment-isolation.md](../decisions/experiment-isolation.md) for
peer comparison table.

---

## Start Method: `spawn` (Not `fork`)

```python
mp_ctx = multiprocessing.get_context("spawn")
```

Linux default is `fork`. CUDA requires `spawn`. From PyTorch docs:
> CUDA runtime does not support `fork`. Use `spawn` or `forkserver`.

`get_context("spawn")` is scoped to `StudyRunner` — it does not affect other libraries via a
global `set_start_method()` call.

---

## StudyRunner: Full Implementation Pattern

```python
# src/llenergymeasure/study/runner.py

import multiprocessing
import threading
import time
import traceback
from multiprocessing.connection import Connection
from queue import Queue

from llenergymeasure.config.models import ExperimentConfig, StudyConfig
from llenergymeasure.domain.results import ExperimentResult, StudyFailed, StudyResult
from llenergymeasure.orchestration.orchestrator import ExperimentOrchestrator


# Timeout: generous estimate — model loading alone can take minutes
def _calculate_timeout(config: ExperimentConfig) -> int:
    return max(config.n * 2, 600)   # minimum 10 minutes; 2s per prompt estimate


class StudyRunner:
    def __init__(self, study: StudyConfig, display) -> None:
        self.study = study
        self.display = display   # Rich display instance

    def run(self) -> StudyResult:
        from llenergymeasure.study.grid import expand_grid
        experiments = expand_grid(self.study)

        study_result = StudyResult.create(self.study)
        mp_ctx = multiprocessing.get_context("spawn")

        for i, config in enumerate(experiments):
            # Thermal gap between experiments
            gap = self.study.execution.config_gap_seconds
            if i > 0 and gap > 0:
                self.display.show_thermal_gap(gap)
                time.sleep(gap)

            result_or_error = self._run_one(config, mp_ctx)
            study_result.add(result_or_error)
            study_result.write_manifest()   # checkpoint after each experiment

        return study_result

    def _run_one(
        self,
        config: ExperimentConfig,
        mp_ctx,
    ) -> ExperimentResult | StudyFailed:
        timeout = _calculate_timeout(config)

        child_conn, parent_conn = mp_ctx.Pipe(duplex=False)
        progress_queue = mp_ctx.Queue()

        p = mp_ctx.Process(
            target=_run_experiment_worker,
            args=(config, child_conn, progress_queue),
            daemon=False,     # daemon=False: clean CUDA teardown if parent exits
        )

        # Start progress consumer BEFORE p.start()
        # Worker may send events immediately on start
        consumer = threading.Thread(
            target=_consume_progress_events,
            args=(progress_queue, self.display),
            daemon=True,
        )
        consumer.start()

        p.start()
        child_conn.close()   # parent does not write; close its copy
        p.join(timeout=timeout)

        # Sentinel to stop consumer — covers both normal and SIGKILL paths
        progress_queue.put(None)
        consumer.join()

        return _collect_result(p, parent_conn, config, timeout)


def _run_experiment_worker(
    config: ExperimentConfig,
    conn: Connection,
    progress_queue,
) -> None:
    """Runs inside child process. Sends result or error dict via Pipe."""
    try:
        progress_queue.put({"event": "started", "config_hash": config.config_hash})
        result = ExperimentOrchestrator(config).run()
        progress_queue.put({"event": "completed", "result": result.summary_dict()})
        _send_result(conn, result)
    except Exception as e:
        error = {
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        progress_queue.put({"event": "failed", "error": error})
        try:
            conn.send(error)
        except Exception:
            pass  # pipe may be broken if parent already timed out
        raise   # re-raise → non-zero exit code for parent to detect
    finally:
        conn.close()


def _send_result(conn: Connection, result: ExperimentResult) -> None:
    """Send result via Pipe; fall back to temp file if result is large."""
    import json, tempfile
    from pathlib import Path

    FILE_BASED_THRESHOLD = 1_000_000   # 1MB — matches optimum-benchmark pattern

    serialised = result.model_dump_json()
    if len(serialised) > FILE_BASED_THRESHOLD:
        tmp = Path(tempfile.mkstemp(suffix=".json")[1])
        tmp.write_text(serialised)
        conn.send({"__file__": str(tmp)})
    else:
        conn.send(result)


def _collect_result(
    p,
    parent_conn: Connection,
    config: ExperimentConfig,
    timeout: int,
) -> ExperimentResult | StudyFailed:
    """Interprets process exit state and reads result from Pipe."""
    import json
    from pathlib import Path

    if p.is_alive():
        # Timeout: SIGKILL (SIGTERM insufficient for hung CUDA calls)
        p.kill()
        p.join()
        return StudyFailed(
            config=config.model_dump(),
            exception_type="TimeoutError",
            error_message=f"Experiment exceeded timeout ({timeout}s)",
        )

    if p.exitcode != 0:
        exc_info = parent_conn.recv() if parent_conn.poll() else None
        return StudyFailed(
            config=config.model_dump(),
            exception_type=exc_info["type"] if exc_info else "ProcessCrash",
            error_message=exc_info["message"] if exc_info else f"Exit code {p.exitcode}",
        )

    # Success — read result
    data = parent_conn.recv()
    if isinstance(data, dict) and "__file__" in data:
        # File-based IPC path (large result)
        tmp = Path(data["__file__"])
        result = ExperimentResult.model_validate_json(tmp.read_text())
        tmp.unlink()
        return result
    return data   # ExperimentResult directly


def _consume_progress_events(queue, display) -> None:
    """Daemon thread in parent — drains progress events from worker queue."""
    while True:
        event = queue.get()     # blocks until an event arrives
        if event is None:       # sentinel: worker done OR parent sent after SIGKILL
            break
        display.update(event)
```

---

## `daemon=False` Rationale

`daemon=False` (the default) means the child is NOT automatically killed when the parent exits.

With `daemon=True`: `Ctrl+C` kills the daemon child immediately. CUDA contexts don't teardown
cleanly — TRT-LLM in particular holds device memory reservations that become orphaned.

With `daemon=False`: if the parent exits while a child is running, the child becomes an orphan
but continues to completion, allowing CUDA to teardown cleanly. `StudyRunner` always calls
`p.join()` before exiting — orphaned processes should not occur in normal operation.

---

## `p.kill()` Rationale (SIGKILL, Not SIGTERM)

For hung CUDA processes, SIGTERM (`p.terminate()`) may be ignored. A GPU kernel stuck in a
CUDA operation may deadlock during signal handling (CUDA signal handling is not re-entrant).
SIGKILL (`p.kill()`) is guaranteed to terminate immediately.

Optional gentler pattern (not used in v2.0):
```python
p.terminate()
p.join(timeout=2)
if p.is_alive():
    p.kill()
```

---

## Progress Display During Study

Each experiment runs in a subprocess — the parent must actively report progress.
A `multiprocessing.Queue` carries progress events; a daemon thread in the parent drains them.

**Why a background thread**: the parent's main thread is blocked at `p.join()`. Without the
consumer thread, no progress updates appear until the experiment completes.

**SIGKILL + Queue interaction**: events queued by the worker but not yet flushed may be lost
on `p.kill()`. The parent always sends `None` sentinel AFTER `p.kill()` — this covers both
the normal path (worker sends `None` itself) and the SIGKILL path (parent sends on worker's
behalf). Consumer thread always exits cleanly.

See [observability.md](observability.md) for the Rich progress display spec.

---

## Pipe Buffer Edge Case

Python's `Pipe()` uses an OS pipe (~64KB buffer on Linux). If the data sent before the parent
reads it exceeds this buffer, the sender blocks — deadlock with parent at `p.join()`.

**In practice**: exception tracebacks and `ExperimentResult` in v2.0 are well under 64KB.
The 1MB file-based IPC threshold handles the rare large-result case (`_send_result()` above).

**If this becomes a concern**: use a reader thread to drain the pipe concurrently with `p.join()`.
Not implemented in v2.0 — add if a future ExperimentResult field causes buffer issues.

---

## Key Differences from Current Codebase

Current `CampaignRunner` uses `subprocess.run(["lem", "experiment", ...])` — CLI re-entry.

| Concern | Current (CLI re-entry) | v2.0 (multiprocessing.Process) |
|---|---|---|
| CUDA isolation | ✓ Clean (fresh `execve`) | ✓ Clean (`spawn` start method) |
| Timeout | ✗ None — can block forever | ✓ `p.join(timeout=...)` |
| Error IPC | ✗ Exit code only | ✓ Structured exception dict via Pipe |
| Result IPC | ✗ Filesystem (result files) | ✓ In-memory Pipe (fast, typed) |
| Overhead | Higher (CLI startup per experiment) | Lower (no CLI re-entry cost) |

The subprocess isolation requirement does NOT apply to single-experiment invocations
(`llem run experiment.yaml`) because there is no state bleed concern for a single run.

---

## Related

- [../decisions/experiment-isolation.md](../decisions/experiment-isolation.md): Decision rationale
- [architecture.md](architecture.md): Call graph, module placement
- [docker-execution.md](docker-execution.md): Docker-specific isolation (v2.0 Docker milestone)
- [observability.md](observability.md): Progress display that consumes queue events
