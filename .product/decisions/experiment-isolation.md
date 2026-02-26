# Experiment Process Isolation

**Status:** Accepted
**Date decided:** 2026-02-19
**Last updated:** 2026-02-25
**Research:** [../research/13-execution-isolation-patterns.md](../research/13-execution-isolation-patterns.md)

## Decision

`multiprocessing.Process` per experiment with `spawn` start method (CUDA-safe). IPC via `multiprocessing.Pipe` (file-based fallback for >1MB results). Timeout via `p.join(timeout=...)`. Failed experiments return structured error dict, don't crash the study. Queue used for progress events only.

---

## Context

Running multiple experiments in sequence (a study) requires isolation between them. GPU state,
CUDA context, and memory allocation all persist within a process. Without isolation, GPU state
from one experiment contaminates the next — invalidating energy measurements and throughput
figures. This decision covers how experiments are isolated, which IPC mechanism transfers
results, how the start method is chosen, and how failures and timeouts are handled.

The current v1.x codebase uses `subprocess.run(["lem", "experiment", ...])` (CLI re-entry per
experiment). The v2.0 redesign moves to `multiprocessing.Process` + Pipe, which is faster and
supports structured IPC. See Codebase Audit section for the comparison.

---

## Considered Options

### Primary isolation mechanism

| Option | Pros | Cons |
|--------|------|------|
| **`multiprocessing.Process` per experiment (chosen)** | Clean CUDA context per experiment; structured result IPC via Pipe; timeout via `p.join(timeout=...)`; lower overhead than CLI re-entry; matches optimum-benchmark | Requires `spawn` start method on Linux (cannot use `fork` with CUDA); slightly more complex setup |
| `subprocess.run` / CLI re-entry (current v1.x) | Works; CUDA isolation guaranteed via `execve`; simple | No timeout mechanism; IPC only via filesystem or exit code; CLI startup overhead per experiment |
| Thread-based isolation | Fast | Does not isolate GPU memory; CUDA allocator state shared across threads; not viable for measurement rigour |
| In-process per experiment | Fastest | GPU memory fragmentation accumulates; CUDA context state bleeds; unacceptable for research-grade measurement |
| Docker-per-experiment | Strongest isolation | Requires Docker; v2.0 is local-only; v2.2 concern |

### IPC mechanism

| Option | Pros | Cons |
|--------|------|------|
| **`multiprocessing.Pipe` (chosen)** | In-memory, typed, fast; structured exception dict; result object transfer | 64KB OS pipe buffer limit (mitigated by 1MB file-based threshold) |
| Filesystem (temp file per experiment) | No buffer limit | Higher latency; cleanup required; current v1.x pattern |
| `multiprocessing.Queue` for results | Fan-out friendly | Overkill for parent–child result passing; Queue used for progress events only |

### `multiprocessing` start method

| Option | Pros | Cons |
|--------|------|------|
| **`get_context('spawn')` (chosen)** | CUDA-safe; scoped to StudyRunner only; macOS/Windows already use spawn | Slower process startup than fork; child must re-import all modules |
| `fork` (Linux default) | Fast | **Unsafe with CUDA**: forked child inherits CUDA driver state → `RuntimeError: Cannot re-initialize CUDA in forked subprocess` or silent incorrect measurements |
| `set_start_method('spawn')` globally | Simpler | Affects all multiprocessing in the process; breaks libraries expecting fork |
| `forkserver` | CUDA-safe | More complex setup; `spawn` is simpler and equally correct |

### Daemon mode for child processes

| Option | Pros | Cons |
|--------|------|------|
| **`daemon=False` (chosen)** | Child process allowed to complete CUDA teardown if parent exits unexpectedly; TRT-LLM device memory reservations released cleanly | Child can become orphan if parent crashes without calling `p.join()` |
| `daemon=True` | Child killed immediately on parent exit | CUDA context not torn down cleanly; GPU left in dirty state, especially with TRT-LLM |

### Timeout enforcement

| Option | Pros | Cons |
|--------|------|------|
| **`p.join(timeout=...)` + SIGKILL (chosen)** | Guarantees termination; no blocked study run; matches optimum-benchmark approach | SIGKILL is abrupt; any events queued by the worker but not yet flushed may be lost |
| `p.join(timeout=...)` + SIGTERM + grace period + SIGKILL | Gentler; gives child chance to cleanup | For hung CUDA calls, SIGTERM may be ignored or deadlock during signal handling |
| No timeout | Simple | Hung CUDA call blocks study indefinitely — hard requirement violation |

---

## Decision

Every experiment runs in a fresh, isolated process. Hard requirement.

This applies to both execution paths:
- **Local (v2.0):** `multiprocessing.Process` per experiment, with `spawn` start method
- **Docker (v2.2):** ephemeral `docker run` per experiment (see `docker-execution.md`)

Both are the same isolation primitive — a fresh OS process — implemented differently.

Rationale: PyTorch's CUDA allocator does not fully release GPU memory across
`del model; torch.cuda.empty_cache()`. Memory fragmentation, cached allocations, and residual
CUDA context state all persist within a process. A fresh process guarantees a clean CUDA context.
There is no other reliable mechanism.

**optimum-benchmark (HuggingFace) makes this a core design principle** for the same reason:
every benchmark gets a new `multiprocessing.Process`, regardless of whether it is the same
backend. Clean state is non-negotiable for measurement rigour.

### Consequences

Positive: Provably clean GPU state per experiment; structured IPC for results and errors;
timeout prevents hung studies; matches industry best practice.
Negative / Trade-offs: `spawn` start method means child must re-import all modules (slower
startup than `fork`). Complexity added vs in-process approach. Pipe buffer limit requires
file-based fallback for large results (>1MB, inherited from optimum-benchmark).
Neutral: `llem run` (single experiment) runs in-process — no subprocess needed. The isolation
requirement applies to `StudyRunner` only (invoked when `llem run` receives a study YAML).

---

## Codebase Audit: Current State (as of Phase 4.5)

The current `CampaignRunner` (to be replaced by `StudyRunner` in Phase 5) uses
`subprocess.run(["lem", "experiment", ...])` — CLI re-entry per experiment.

| Concern | Current (CLI re-entry) | Proposed (multiprocessing.Process + Pipe) |
|---------|------------------------|-------------------------------------------|
| CUDA isolation | Clean — fresh `execve` process | Clean — spawn start method |
| Timeout | None — can block indefinitely | `p.join(timeout=...)` |
| Error IPC | Exit code only | Structured exception dict via Pipe |
| Result IPC | Filesystem (result files) | In-memory Pipe (fast, typed) |
| Overhead | Higher — CLI startup per experiment | Lower — no CLI re-entry cost |

The current code works but has no timeout (a hard requirement for correctness — a hung CUDA
call can block forever) and no structured error forwarding. Phase 5 must replace it.

---

## Local Execution Model (v2.0)

```python
# StudyRunner — pseudocode
mp_ctx = multiprocessing.get_context("spawn")   # See: spawn vs fork decision above

for experiment_config in study.experiments:
    time.sleep(config_gap_seconds)              # thermal gap (host-managed)

    child_conn, parent_conn = mp_ctx.Pipe()
    progress_queue = mp_ctx.Queue()

    p = mp_ctx.Process(
        target=_run_experiment_worker,
        args=(experiment_config, child_conn, progress_queue),
        daemon=False,                           # See: daemon=False rationale above
    )

    # Start progress consumer BEFORE p.start() (worker may send events immediately)
    consumer = threading.Thread(
        target=_consume_progress_events,
        args=(progress_queue, rich_display),
        daemon=True,
    )
    consumer.start()

    p.start()
    p.join(timeout=experiment_timeout_seconds)  # See: timeout rationale above

    # Signal consumer to stop (covers both normal completion and SIGKILL)
    progress_queue.put(None)
    consumer.join()

    result: ExperimentResult = _collect_result(p, parent_conn, experiment_config)
    results.append(result)
```

```python
# _run_experiment_worker — runs inside the child process
def _run_experiment_worker(
    config: ExperimentConfig,
    conn: Connection,
    progress_queue: Queue,
) -> None:
    try:
        progress_queue.put({"event": "started", "config": config.model_dump()})
        result = ExperimentOrchestrator(config).run()   # full experiment in child
        progress_queue.put({"event": "completed"})
        conn.send(result)
    except Exception as e:
        # Send exception info before exit — parent gets structured failure data
        conn.send({"type": type(e).__name__, "message": str(e), "traceback": traceback.format_exc()})
        raise   # re-raise → subprocess exits with non-zero code
    finally:
        conn.close()
```

**Large results:** If `ExperimentResult` exceeds ~1MB (e.g. time-series data), write to a
temp file and send the path via Pipe instead. This mirrors optimum-benchmark's pattern:

```python
# optimum-benchmark: FILE_BASED_COMM_THRESHOLD = 1_000_000 bytes
if len(str(result_dict)) > threshold:
    tmp_path = write_to_tmp(result)
    conn.send(tmp_path)   # parent reads from tmp file, then deletes it
else:
    conn.send(result)
```

---

## Subprocess Error Handling

Subprocess failures must not crash the parent `StudyRunner`. Full error handling pattern:

```python
# StudyRunner — full pattern including error handling
for experiment_config in study.experiments:
    time.sleep(config_gap_seconds)

    child_conn, parent_conn = mp_ctx.Pipe()
    p = mp_ctx.Process(
        target=_run_experiment_worker,
        args=(experiment_config, child_conn, progress_queue),
        daemon=False,
    )
    p.start()
    p.join(timeout=experiment_timeout_seconds)   # must set a timeout — never block forever

    if p.is_alive():
        # Timeout: subprocess hung (e.g. deadlocked CUDA call)
        p.kill()
        p.join()
        # Send synthetic failed event — dead worker cannot send it itself
        progress_queue.put({"event": "failed", "reason": "timeout"})
        study_result.add_failed(StudyFailed(
            config=experiment_config.model_dump(),
            exception_type="TimeoutError",
            error_message=f"Experiment exceeded timeout ({experiment_timeout_seconds}s)",
        ))
        continue

    if p.exitcode != 0:
        # Subprocess crashed — OOM, CUDA error, unhandled exception, OS kill
        # The pipe will be empty (worker never sent result)
        exc_info = parent_conn.recv() if parent_conn.poll() else None
        study_result.add_failed(StudyFailed(
            config=experiment_config.model_dump(),
            exception_type=exc_info["type"] if exc_info else "ProcessCrash",
            error_message=exc_info["message"] if exc_info else f"Exit code {p.exitcode}",
        ))
        continue

    # Success — read result from pipe
    result = parent_conn.recv()
    study_result.add_result(result)
```

### Timeout — How It's Set

`experiment_timeout_seconds` is an internal constant, not user-facing config in v2.0:

```python
EXPERIMENT_TIMEOUT_SECONDS = max(
    config.n * 2,       # generous estimate: 2s per prompt
    600,                # minimum 10 minutes for model loading
)
```

Set conservatively. A missed timeout (too generous) wastes one experiment slot.
A false timeout (too tight) discards valid results. When in doubt, be generous.

### `p.kill()` vs `p.terminate()`

SIGKILL (`p.kill()`) is chosen over SIGTERM (`p.terminate()`) for hung CUDA processes.
A GPU kernel stuck in a CUDA operation may ignore SIGTERM or deadlock during signal handling
(CUDA's signal handling is not re-entrant). SIGKILL is the only mechanism guaranteed to
terminate the process immediately.

Alternative considered: `p.terminate() → time.sleep(2) → p.kill() if p.is_alive()`. This
gives the process a chance to cleanup, but for measurement rigour (prevent GPU state bleed
from partial teardown), immediate SIGKILL is defensible. Chosen for v2.0.

---

## Single Experiment (`llem run` with experiment YAML)

`llem run experiment.yaml` runs a single experiment. In-process is acceptable here — there
is nothing before or after it to contaminate. No subprocess needed for single-experiment
invocations.

The subprocess isolation requirement applies to `StudyRunner` only (invoked when `llem run`
receives a study YAML with multiple experiments), where multiple experiments run in sequence
and GPU state bleed between them matters.

---

## Implications for ExperimentOrchestrator

```
run_experiment(config)            run_study(config)
      │                                  │
      ▼                                  ▼
ExperimentOrchestrator          StudyRunner
(in-process, single shot)       │
                                 │  for each experiment:
                                 │    mp_ctx.Process(ExperimentOrchestrator)
                                 │    result ← Pipe
                                 ▼
                             StudyResult

CLI: llem run experiment.yaml → run_experiment() → ExperimentOrchestrator (in-process)
CLI: llem run study.yaml      → run_study()      → StudyRunner → subprocess per experiment
```

Phase 5 must:

1. Keep `ExperimentOrchestrator` as-is — it runs inside the child process unchanged.
2. Add `StudyRunner` which manages the subprocess lifecycle around it.
3. `run_experiment()` (library API, single experiment) calls `ExperimentOrchestrator` directly
   in-process — no subprocess.
4. `run_study()` (library API, multi-experiment) uses `StudyRunner` which wraps each
   `ExperimentOrchestrator` invocation in a `multiprocessing.Process`.
5. `llem run` (unified CLI) detects YAML type and dispatches to the appropriate library function.

---

## Progress Output During Study Execution

Each experiment runs in a subprocess, so the parent process must actively report progress.
The mechanism: a `multiprocessing.Queue` for progress events, consumed by the parent's
display loop running in a background thread.

A background thread is required because the parent's main thread is blocked at `p.join()`.
Without it, no progress updates appear until the experiment completes.

When `p.kill()` is issued, any events queued but not yet flushed may be lost. To guarantee
the consumer thread exits: the parent always sends a `None` sentinel to the queue after
`p.kill()`, covering both normal completion and SIGKILL paths.

**Display approach:** Rich `Progress` with a live-updating task bar:

```
llem run batch-size-sweep.yaml

  Study: batch-size-sweep (12 experiments × 3 cycles = 36 runs)

  ████████████░░░░░░░░  8/12 experiments  [00:43:21]

  ✓  llama-3.1-8b pytorch bf16 batch=1    42.3 J  847 tok/s  [00:03:12]
  ✓  llama-3.1-8b pytorch bf16 batch=8    38.1 J  2140 tok/s [00:03:45]
  ✓  llama-3.1-8b pytorch bf16 batch=16   36.9 J  3820 tok/s [00:04:01]
  ▶  llama-3.1-8b pytorch bf16 batch=32   running... [00:01:23]

  Thermal gap: cooling 60s ░░░░░░░░░░░░░░░░░░░░ 47s remaining
```

Key UX decisions:
- Progress bar is at experiment level, not prompt level (subprocess hides per-prompt detail)
- Thermal gaps shown with countdown so user knows the tool is not stalled
- Each completed experiment shows key metrics inline (energy, throughput) — immediate feedback
- Failed experiments shown in red with exception type — user sees failures as they happen
- No verbose subprocess stdout by default; `--verbose` flag pipes subprocess stdout to parent

---

## Pipe Buffer: Edge Case (Implementation Note)

Python's `multiprocessing.Pipe()` uses an OS pipe with a buffer of ~64KB on Linux. If the
data sent through the pipe exceeds this buffer before the parent reads it, the sender blocks
indefinitely (deadlock with parent at `p.join()`).

In practice: exception tracebacks are well under 64KB. `ExperimentResult` in v2.0 has no
time-series data and is unlikely to approach this limit. The 1MB file-based IPC threshold
handles the rare large-result case.

If this becomes a concern: use a reader thread to drain the pipe concurrently with `p.join()`.
Not implemented in v2.0 — document if a future ExperimentResult field causes buffer issues.

---

## Peer Codebase Reference

| Tool | Local isolation | Docker isolation |
|------|-----------------|-----------------|
| **optimum-benchmark** | `multiprocessing.Process` per experiment | Per backend family (CI only) |
| **AIEnergyScore** | N/A | Ephemeral `docker run` per experiment |
| **vLLM bench sweep** | Long-running `subprocess.Popen` server (throughput tool — different objective) | N/A |
| **MLPerf** | Threaded in-process SUT | One container per backend run |

LLenergyMeasure follows optimum-benchmark for local and AIEnergyScore for Docker.

---

## Related

- [docker-execution.md](docker-execution.md): Docker-specific decisions (ephemeral containers, no HTTP API)
- [architecture.md](architecture.md): Library-first architecture (ExperimentOrchestrator is internal)
- Research: [../research/13-execution-isolation-patterns.md](../research/13-execution-isolation-patterns.md)
