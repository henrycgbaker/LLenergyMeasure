# Architecture Research

**Domain:** Multi-experiment study/sweep execution integration for an LLM inference efficiency measurement tool
**Researched:** 2026-02-27
**Confidence:** HIGH (subprocess and IPC patterns verified against PyTorch official docs and optimum-benchmark source; manifest and display patterns verified from existing confirmed design docs)

---

## Scope

This file answers: **How does study/sweep execution integrate with the existing single-experiment architecture?**

Specifically:
- Where does `StudyRunner` live and how does it dispatch to `ExperimentOrchestrator`?
- How do subprocess-spawned experiments access the orchestrator?
- How do results flow back from child to parent process?
- How does `ManifestWriter` checkpoint state incrementally?
- How does the CLI display multi-experiment progress?

All architectural decisions are already confirmed in `.product/designs/`. This file documents the integration pattern, component boundaries, data flow, and build order for the milestone implementation. Note: the existing ARCHITECTURE.md was an earlier decision audit — this file supersedes it with integration-focused architecture for study/sweep.

---

## System Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  CLI Layer   cli/commands/run.py                                              │
│                                                                               │
│   llem run study.yaml                                                         │
│        │                                                                      │
│        ├── detect: study (has sweep: or experiments: list)                   │
│        ├── load_study_config()  -> StudyConfig                               │
│        ├── run pre-flight (study-level: backends, energy, models)            │
│        └── call run_study(StudyConfig)  -> StudyResult                      │
└─────────────────────────────────┬────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼────────────────────────────────────────────┐
│  Public API   _api.py                                                         │
│                                                                               │
│   run_study(StudyConfig)                                                      │
│        └── _run(StudyConfig)  -> StudyResult      <- internal always          │
└─────────────────────────────────┬────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼────────────────────────────────────────────┐
│  Study Layer   study/                                                         │
│                                                                               │
│   ┌─────────────────────┐     ┌──────────────────────┐                       │
│   │    StudyRunner      │────>│    grid.expand()     │  sweep -> list[Cfg]   │
│   │   (runner.py)       │     └──────────────────────┘                       │
│   │                     │     ┌──────────────────────┐                       │
│   │ for cfg in configs: │────>│   ManifestWriter     │  checkpoint per step  │
│   │   _run_one(cfg)     │     │   (manifest.py)      │                       │
│   └─────────┬───────────┘     └──────────────────────┘                       │
│             │                                                                  │
│             │  mp_ctx.Process(target=_run_experiment_worker, ...)             │
│             │                                                                  │
│  ┌──────────▼──────────────────────────────────────┐                         │
│  │  Child Process  (isolated, spawn start method)  │                         │
│  │                                                  │                         │
│  │  _run_experiment_worker(config, conn, queue)     │                         │
│  │       └── ExperimentOrchestrator(config).run()  │                         │
│  │             └── ExperimentResult                │                         │
│  │                      │                          │                         │
│  │               conn.send(result)                 │                         │
│  └──────────────────────────────────────────────── ┘                         │
│                         │                                                      │
│   Parent reads Pipe ────> ExperimentResult | StudyFailed                     │
└─────────────────────────────────┬────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼────────────────────────────────────────────┐
│  Existing Layers  (unmodified by this milestone)                             │
│                                                                               │
│  orchestration/orchestrator.py  <- ExperimentOrchestrator (single expt)      │
│  core/backends/                 <- PyTorchBackend (+ future vllm, tensorrt)  │
│  core/energy/                   <- NVMLBackend / ZeusBackend / CodeCarbon    │
│  domain/results.py              <- ExperimentResult, StudyResult, StudyFailed│
│  results/persistence.py         <- to_json(), to_parquet(), from_json()      │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Responsibilities

| Component | Responsibility | Module | New or Modified |
|-----------|---------------|--------|-----------------|
| `StudyRunner` | Orchestrate sequential subprocess execution; coordinate manifest, gaps, cycles | `study/runner.py` | **NEW** |
| `grid.expand()` | Resolve sweep grammar -> `list[ExperimentConfig]` with n_cycles and cycle_order applied | `study/grid.py` | **NEW** |
| `ManifestWriter` | Write/update `study_manifest.json` at each state transition | `study/manifest.py` | **NEW** |
| `_run_experiment_worker()` | Module-level function; runs in child process; calls `ExperimentOrchestrator.run()` | `study/runner.py` | **NEW** |
| `_collect_result()` | Interprets process exit state; reads result from Pipe | `study/runner.py` | **NEW** |
| `_consume_progress_events()` | Daemon thread in parent; drains Queue; calls `display.update()` | `study/runner.py` | **NEW** |
| `run_study()` | Public API wrapper; thin; delegates to `_run()` | `_api.py` | **MODIFIED** (add) |
| `_run()` | Internal dispatcher; always takes `StudyConfig`; single-experiment is degenerate study | `_api.py` | **MODIFIED** (add) |
| CLI `run.py` | Detect study vs experiment from YAML; route to `run_study()` or `run_experiment()` | `cli/commands/run.py` | **MODIFIED** |
| `display.py` | Add study-level display: outer progress bar, per-experiment status lines, thermal gap countdown | `cli/display.py` | **MODIFIED** |
| `ExperimentOrchestrator` | Unchanged — still handles single experiment lifecycle in child process | `orchestration/orchestrator.py` | **UNMODIFIED** |
| `StudyResult` | `add(result_or_error)` method needs implementing for accumulation | `domain/results.py` | **MODIFIED** (minor) |

---

## Recommended Module Structure

The study module is already defined in the architecture design. This confirms the new files:

```
src/llenergymeasure/
  study/
    __init__.py
    runner.py           <- StudyRunner, _run_experiment_worker, _collect_result,
                        <-   _consume_progress_events, _calculate_timeout, _send_result
    grid.py             <- expand_grid(StudyConfig) -> list[ExperimentConfig]
                        <-   handles n_cycles, cycle_order (sequential/interleaved/shuffled)
    manifest.py         <- ManifestWriter, StudyManifest, ExperimentManifestEntry
```

Three new files. All other changes are modifications to existing modules.

---

## Architectural Patterns

### Pattern 1: Subprocess-per-Experiment with `spawn` Start Method

**What:** Each experiment in a study runs in a fresh `multiprocessing.Process`. The parent `StudyRunner` blocks at `p.join(timeout=...)` while the child runs. Results return via `multiprocessing.Pipe`.

**Why:** PyTorch's CUDA allocator does not fully release GPU memory across `del model; torch.cuda.empty_cache()`. Memory fragmentation and CUDA context state persist within a process. A fresh process with `spawn` start method guarantees a clean CUDA context. This is a correctness requirement, not an optimisation choice.

**Confidence:** HIGH. PyTorch official docs explicitly state CUDA requires `spawn` or `forkserver`, not `fork`. Optimum-benchmark (HuggingFace) independently arrived at the identical pattern for the same reason, using it as a core design principle.

```python
# study/runner.py
mp_ctx = multiprocessing.get_context("spawn")   # not fork -- CUDA requirement

child_conn, parent_conn = mp_ctx.Pipe(duplex=False)
progress_queue = mp_ctx.Queue()

p = mp_ctx.Process(
    target=_run_experiment_worker,
    args=(config, child_conn, progress_queue),
    daemon=False,    # NOT daemon: CUDA teardown needs to complete cleanly
)
p.start()
child_conn.close()   # parent does not write; close parent's copy
p.join(timeout=timeout)
```

**Single vs multi distinction:** `run_experiment()` (single experiment) runs `ExperimentOrchestrator` in-process — no subprocess needed because clean GPU state at start is guaranteed for a single run. Only `run_study()` uses subprocess isolation.

**Trade-offs:**
- Higher per-experiment overhead (process start + CUDA init, ~1-2s on Linux with spawn)
- Clean GPU state is guaranteed — no contamination between experiments
- `daemon=False` means CUDA teardown completes even if parent exits abnormally (optimum-benchmark uses same setting)

### Pattern 2: Dual-Channel IPC — Pipe for Result, Queue for Progress Events

**What:** Two IPC channels per experiment subprocess: a `Pipe` (one-way, child to parent) for the final result or error, and a `Queue` (multi-item, child to parent) for incremental progress events during execution.

**Why separate channels:**
- `Pipe` for result: single send at experiment end; typed; low overhead; direct
- `Queue` for progress: multiple events during experiment (started, warmup progress, measurement progress, completed/failed); needs buffering; never blocks worker on slow parent

**Optimum-benchmark precedent (HIGH confidence):** Verified from source code. Uses identical dual-channel pattern — `Pipe` for result return with 1MB file-based fallback for large payloads. Same `FILE_BASED_COMM_THRESHOLD = 1_000_000` value. Uses a separate mechanism for process synchronisation events.

```python
# study/runner.py -- result transmission with size-based fallback
def _send_result(conn: Connection, result: ExperimentResult) -> None:
    serialised = result.model_dump_json()
    if len(serialised) > 1_000_000:   # 1MB threshold -- matches optimum-benchmark
        tmp = Path(tempfile.mkstemp(suffix=".json")[1])
        tmp.write_text(serialised)
        conn.send({"__file__": str(tmp)})   # parent reads from file
    else:
        conn.send(result)   # direct in-memory Pipe transmission
```

**Pipe buffer edge case:** OS pipe buffer on Linux is ~64KB. If the child sends data before the parent reads and it exceeds this, the child blocks — deadlock with parent at `p.join()`. The 1MB file-based fallback handles large results. In practice, v2.0 `ExperimentResult` is well under 64KB.

**Pipe vs Queue choice:** `Pipe` is ~3x faster than `Queue` (Queue is built on Pipe plus a feeder thread). For the single final result, `Pipe` is preferred. For streaming progress events, `Queue` provides buffering so the worker never blocks waiting for the parent to drain.

### Pattern 3: Progress Queue Consumer as Daemon Thread

**What:** A daemon thread in the parent process drains the progress queue while the parent's main thread is blocked at `p.join()`. Without this thread, no progress updates appear until the experiment completes.

**Why a daemon thread (not async, not polling):**
- Parent main thread is synchronously blocked at `p.join()` — it cannot drain the queue
- Daemon thread exits automatically when parent exits — no cleanup required
- Thread is cheap for this I/O-bound draining work; cheaper than another process

**SIGKILL sentinel handling:** Events queued by the worker but not yet consumed may be lost when `p.kill()` is called. The parent sends `None` (sentinel) AFTER `p.kill()` or `p.join()`, covering both the normal path (worker sends sentinel itself) and the SIGKILL path (parent sends on worker's behalf). Consumer thread always exits cleanly.

```python
# study/runner.py -- consumer thread starts BEFORE p.start()
consumer = threading.Thread(
    target=_consume_progress_events,
    args=(progress_queue, display),
    daemon=True,
)
consumer.start()
p.start()
# ... p.join() ...
progress_queue.put(None)   # sentinel -- always, regardless of how p exited
consumer.join()
```

### Pattern 4: Manifest Checkpoint Written After Each State Transition

**What:** `ManifestWriter` writes `study_manifest.json` after every status change: `mark_running()` writes, `mark_completed()` writes, `mark_failed()` writes. The manifest is always consistent with actual state.

**Why write-on-every-change (not write-at-end):**
- Studies run for hours; power failures and OOM kills happen
- Resume logic requires knowing which experiments completed before interruption
- Cost of one small JSON write per experiment is negligible vs study duration

**Ray Tune precedent (MEDIUM confidence):** Ray Tune maintains `experiment/trial_runner.json` checkpoint with trial statuses, enabling study resume after interruption. Same pattern — incremental checkpoint, not end-only write.

**Two distinct objects:**
- `StudyManifest` = in-progress checkpoint; written incrementally; survives interruption; file: `study_manifest.json`
- `StudyResult` = final return value of `run_study()`; written once at completion; file: `study_summary.json`

```python
# study/manifest.py -- write-on-every-transition
class ManifestWriter:
    def mark_running(self, config_hash: str, cycle: int) -> None:
        entry = self._find(config_hash, cycle)
        entry.status = "running"
        entry.started_at = datetime.utcnow()
        self._write()   # write immediately after every state change

    def mark_completed(self, config_hash: str, cycle: int, result_file: str) -> None:
        entry = self._find(config_hash, cycle)
        entry.status = "completed"
        entry.result_file = result_file
        entry.completed_at = datetime.utcnow()
        self._write()   # write immediately
```

### Pattern 5: Grid Expansion Before Subprocess Dispatch

**What:** `grid.expand(study_config)` resolves the complete `list[ExperimentConfig]` (including n_cycles and cycle_order) before any subprocess is launched. The runner iterates a plain list.

**Why expand first:**
- Pre-flight validation (invalid config skipping via Pydantic) happens at expansion time — no GPU time wasted on invalid combos
- Manifest is populated with all pending entries at study start
- `cycle_order` (interleaved/shuffled/sequential) is applied at expansion time across the full experiment set
- Simpler runner: iterates a known, fully-resolved list

**lm-eval precedent (MEDIUM confidence):** lm-eval loads all tasks upfront via `task_manager.load(tasks)` before dispatching any evaluation. Same "resolve everything first, iterate second" principle.

**Hydra multirun precedent (MEDIUM confidence):** Hydra's `BasicLauncher` computes the full set of parameter combinations ("sweeps") before dispatching any job sequentially. Hydra docs state: "Hydra composes configs lazily at job launching time" — the full sweep is known upfront.

```python
# study/grid.py -- pure function
def expand_grid(study: StudyConfig) -> list[ExperimentConfig]:
    """
    Returns complete ordered list of ExperimentConfig to run.
    Applies: sweep Cartesian product, explicit experiments, n_cycles, cycle_order.
    Invalid combinations (Pydantic ValidationError) are excluded with a warning logged.
    """
    # 1. Universal sweep params x backend-scoped sweep params (per-backend Cartesian)
    # 2. Append explicit experiments list
    # 3. Apply n_cycles (repeat the list n times)
    # 4. Apply cycle_order: sequential=as-is, interleaved=round-robin, shuffled=random
    ...
```

---

## Data Flow

### Study Execution Flow

```
llem run study.yaml
        |
        v
load_study_config(path)      <- config/loader.py
        |
        v
run_study(StudyConfig)       <- _api.py (public API)
        |
        v
_run(StudyConfig)            <- _api.py (internal)
        |
        v
StudyRunner(study, display)
        |
        +---> expand_grid(study)   -> list[ExperimentConfig]  (grid.py)
        |         |
        |         +-- for each raw config:
        |               try: ExperimentConfig(**raw)   <- Pydantic L1 validation
        |               except ValidationError: log skip, exclude
        |
        +---> ManifestWriter.create(study, results_dir)  (manifest.py)
        |         +-- writes study_manifest.json with all entries as "pending"
        |
        +---> for i, config in enumerate(experiments):
                  |
                  +-- if i > 0 and gap > 0: sleep(gap)  [thermal gap + countdown]
                  |
                  +-- manifest.mark_running(config.config_hash, cycle)
                  |
                  +-- _run_one(config, mp_ctx)
                  |       |
                  |       +-- mp_ctx.Pipe()  [child_conn, parent_conn]
                  |       +-- mp_ctx.Queue() [progress_queue]
                  |       +-- consumer thread.start()
                  |       +-- mp_ctx.Process(target=_run_experiment_worker).start()
                  |       +-- p.join(timeout=...)
                  |       +-- progress_queue.put(None)  [sentinel]
                  |       +-- consumer.join()
                  |       +-- _collect_result(p, parent_conn, config, timeout)
                  |               -> ExperimentResult | StudyFailed
                  |
                  +-- study_result.add(result_or_error)
                  |
                  +-- manifest.mark_completed | mark_failed (writes to disk)

        +---> return StudyResult
```

### Child Process Internal Flow

```
_run_experiment_worker(config, conn, progress_queue)
        |
        +-- progress_queue.put({"event": "started", "config_hash": config.config_hash})
        |
        +-- ExperimentOrchestrator(config).run()    <- EXISTING, UNMODIFIED
        |       |
        |       +-- pre-flight (within single experiment)
        |       +-- load model
        |       +-- warmup
        |       +-- measure (NVML/Zeus polling)
        |       +-- aggregate -> ExperimentResult
        |
        +-- progress_queue.put({"event": "completed", "result": result.summary_dict()})
        |
        +-- _send_result(conn, result)
              |
              +-- if len(serialised) <= 1MB: conn.send(result)  [direct Pipe]
              +-- else: write to tempfile; conn.send({"__file__": path})
```

### Result Collection in Parent

```
_collect_result(p, parent_conn, config, timeout)
        |
        +-- if p.is_alive():
        |       p.kill()  [SIGKILL -- SIGTERM insufficient for hung CUDA calls]
        |       p.join()
        |       return StudyFailed(exception_type="TimeoutError")
        |
        +-- if p.exitcode != 0:
        |       exc_info = parent_conn.recv() if parent_conn.poll() else None
        |       return StudyFailed(
        |           exception_type=exc_info["type"] if exc_info else "ProcessCrash",
        |           error_message=exc_info["message"] if exc_info else f"Exit {p.exitcode}"
        |       )
        |
        +-- data = parent_conn.recv()
              +-- if isinstance(data, dict) and "__file__" in data:
              |       result = ExperimentResult.model_validate_json(tmp.read_text())
              |       tmp.unlink()
              |       return result
              +-- else: return data  [ExperimentResult directly]
```

---

## Integration Points

### New vs Existing Module Boundaries

| Boundary | Communication | Notes |
|----------|--------------|-------|
| `StudyRunner` -> `grid.expand()` | Direct function call | `grid.py` is pure: takes `StudyConfig`, returns `list[ExperimentConfig]` — no side effects |
| `StudyRunner` -> `ManifestWriter` | Instance method calls | Single `ManifestWriter` instance per study run; it owns the manifest file |
| `StudyRunner` -> child process | `multiprocessing.Process` + `Pipe` + `Queue` | spawn context; parent blocks at `p.join()` |
| `_run_experiment_worker` -> `ExperimentOrchestrator` | Direct instantiation in child process | `ExperimentOrchestrator(config).run()` — no IPC needed within the child |
| `_api.py` -> `StudyRunner` | Direct instantiation | `run_study()` creates `StudyRunner(study, display)`, calls `.run()` |
| CLI `run.py` -> `_api.py` | `run_study()` or `run_experiment()` call | CLI detects study vs experiment from YAML structure; routes accordingly |
| `StudyRunner` -> `display` | `display.update(event_dict)` via consumer thread | Display object passed to `StudyRunner`; events sourced from progress queue |
| `StudyResult` -> `results/persistence.py` | `to_json()` call at study completion | Individual `ExperimentResult` files written inside child process |

### CLI Detection: Experiment vs Study

```python
# cli/commands/run.py -- routing logic
def _detect_yaml_type(path: Path) -> Literal["experiment", "study"]:
    """
    Study if YAML has 'sweep:' key or 'experiments:' list. Otherwise experiment.
    """
    raw = yaml.safe_load(path.read_text())
    if "sweep" in raw or ("experiments" in raw and isinstance(raw["experiments"], list)):
        return "study"
    return "experiment"
```

### `_api.py` Routing

```python
# _api.py -- both paths go through _run(StudyConfig)
def run_experiment(config: str | Path | ExperimentConfig | None, **kwargs) -> ExperimentResult:
    study_config = _to_study_config(config, **kwargs)
    study_result = _run(study_config)
    return study_result.experiments[0]   # unwrap single-experiment study

def run_study(config: str | Path | StudyConfig) -> StudyResult:
    study_config = _to_study_config(config)
    return _run(study_config)

def _run(study: StudyConfig) -> StudyResult:
    display = _make_display(study)
    return StudyRunner(study, display).run()
```

### Progress Display Integration

The display layer receives events from the progress queue consumer thread. Study mode requires two display changes:

1. **Outer progress bar** (tqdm): tracks `N/total_experiments` across the study
2. **Per-experiment status lines** (print to stderr): symbols `+` completed, `>` running, `.` queued, `!` failed
3. **Thermal gap countdown** (print to stderr): `waiting thermal gap (55s remaining)` — silence during a 5-minute gap would make users think the tool crashed

```python
# cli/display.py -- study event handler
def update(self, event: dict) -> None:
    event_type = event.get("event")
    if event_type == "started":
        tqdm.write(f"  > [{self._current}/{self._total}]  {event['config_summary']}  running...", file=sys.stderr)
    elif event_type == "completed":
        tqdm.write(f"  + [{self._current}/{self._total}]  {event['config_summary']}  -> ...", file=sys.stderr)
        self._pbar.update(1)
    elif event_type == "failed":
        tqdm.write(f"  ! [{self._current}/{self._total}]  failed: {event['error']['message']}", file=sys.stderr)
        self._pbar.update(1)
```

**Critical:** Use `tqdm.write()` for all print calls during study — not plain `print()`. `tqdm.write()` coordinates with the progress bar to avoid overwriting the bar's output line.

---

## Anti-Patterns

### Anti-Pattern 1: CLI Re-entry (current v1.x `CampaignRunner` pattern)

**What people do:** Spawn a subprocess via `subprocess.run(["llem", "experiment", ...])` to run each experiment. The current `CampaignRunner` does exactly this.

**Why it is wrong:**
- No structured error IPC — only exit code; exception type and message lost
- No timeout — a hung CUDA process blocks forever
- Result IPC via filesystem only — no in-memory path; relies on result file conventions
- CLI startup overhead per experiment (Python import + Typer init, ~0.5–1s per run)

**Do this instead:** `multiprocessing.Process` with `spawn` start method and `Pipe` for result IPC. `ExperimentOrchestrator` is importable as a library object — no CLI re-entry needed.

### Anti-Pattern 2: Sharing CUDA State Across Experiments In-Process

**What people do:** Run multiple experiments sequentially in the same Python process, calling `del model; torch.cuda.empty_cache()` between them.

**Why it is wrong:**
- PyTorch's CUDA allocator does not fully release memory on `empty_cache()`
- CUDA context state (caching allocator pools, memory fragmentation, pinned memory) persists
- Experiment N+1 starts with contaminated CUDA state from experiment N
- Energy measurements are inflated and irreproducible

**Do this instead:** Fresh `multiprocessing.Process` with `spawn` start method per experiment in study mode. Single-experiment (`run_experiment()`) runs in-process because clean state at start is guaranteed.

### Anti-Pattern 3: Using `fork` Start Method with CUDA

**What people do:** Use Linux's default `fork` start method.

**Why it is wrong:** PyTorch explicitly states CUDA does not support `fork`. The CUDA runtime is not fork-safe — forked child processes inherit CUDA context state from the parent, causing incorrect measurements, deadlocks, or crashes.

**Do this instead:** `multiprocessing.get_context("spawn")` — scoped to `StudyRunner`, never via global `set_start_method()` which would affect other libraries.

### Anti-Pattern 4: Generating Configs On-the-Fly During Study Execution

**What people do:** Generate the next `ExperimentConfig` inside the study loop, after the previous experiment completes.

**Why it is wrong:**
- Cannot pre-validate all configs upfront (invalid combos discovered mid-study after wasting warmup time and GPU allocation)
- Cannot populate manifest with all pending entries at study start (resume shows incomplete picture)
- Cannot apply `cycle_order` across the full experiment set before starting

**Do this instead:** `expand_grid(study)` resolves the complete ordered `list[ExperimentConfig]` before any subprocess launches. Invalid configs are caught by Pydantic at expansion time and logged as skipped. The runner iterates a plain, fully-known list.

### Anti-Pattern 5: Plain `print()` During Study Progress Display

**What people do:** Use `print(..., file=sys.stderr)` for experiment status lines while tqdm is showing an outer progress bar.

**Why it is wrong:** `print()` writes newlines that leave the tqdm progress bar on a stale line. The bar then re-draws on the next line, leaving visual artifacts. tqdm assumes exclusive control over its output line.

**Do this instead:** `tqdm.write(message, file=sys.stderr)` for all study progress messages. `tqdm.write()` coordinates with tqdm's output to print cleanly above the active bar.

---

## Build Order

The study/sweep integration has clear dependencies. This order allows incremental testing at each step.

### Step 1: Grid Expansion (`study/grid.py`)

Build and test in isolation. Pure function — no side effects, no subprocess needed.
- Input: `StudyConfig`. Output: `list[ExperimentConfig]`.
- Cartesian product of universal sweep dims
- Backend-scoped sweep dims (dotted notation: `pytorch.batch_size`)
- Independent grids per backend (multi-backend study)
- `n_cycles` repetition with `cycle_order`: sequential, interleaved, shuffled
- Invalid config exclusion (Pydantic `ValidationError` catch + log)

**Test:** Unit tests with mock `StudyConfig`, assert output list shape and ordering for each cycle_order mode.

**Dependency:** Only `config/models.py` (existing). No new dependencies introduced.

### Step 2: Manifest Schema and Writer (`study/manifest.py`)

Build and test in isolation. No subprocess needed.
- `StudyManifest` and `ExperimentManifestEntry` Pydantic models
- `ManifestWriter` lifecycle: `__init__` creates file with all entries as "pending" -> `mark_running` -> `mark_completed` / `mark_failed`
- File write on every state transition via `_write()`

**Test:** Create manifest for 3-experiment study; call mark_running then mark_completed; re-read file; assert status transitions are persisted correctly.

**Dependency:** Only `domain/results.py` (existing `StudyFailed`). No new dependencies.

### Step 3: Subprocess Worker Functions (`study/runner.py` — worker functions only)

Implement `_run_experiment_worker`, `_send_result`, `_collect_result` as standalone functions. Test by manually spawning a subprocess without using `StudyRunner`.
- `spawn` context confirmed working
- `Pipe` result transmission: both direct and file-based (>1MB) paths
- Error propagation: child exception -> `StudyFailed` with type and message
- Timeout: `p.join(timeout=...)` -> `p.kill()` -> `StudyFailed("TimeoutError")`

**Test:** Minimal child function that either raises or returns a dummy `ExperimentResult`; verify parent `_collect_result` handles all cases (success, non-zero exit, timeout).

**Dependency:** `orchestration/orchestrator.py` (existing). The worker function imports and calls `ExperimentOrchestrator` directly.

### Step 4: Progress Event Queue and Display Consumer

Implement progress `Queue` + `_consume_progress_events` daemon thread + display integration.
- Queue send from child worker at key lifecycle events: started, completed, failed
- Consumer thread draining queue, calling `display.update(event)`
- Sentinel `None` sent after `p.join()` (covers both normal and SIGKILL paths)
- `display.py` study-mode: tqdm outer progress bar, per-experiment status lines with `tqdm.write()`, thermal gap countdown

**Test:** Mock display; verify events arrive in correct order and sentinel causes clean thread exit; verify no deadlock on SIGKILL path.

**Dependency:** Steps 3 + display.py modifications.

### Step 5: `StudyRunner` Integration

Wire steps 1–4 together in `StudyRunner.run()`.
- Sequential experiment dispatch with thermal gap (`time.sleep`)
- `manifest.mark_running()` / `mark_completed()` / `mark_failed()` calls around each `_run_one()`
- `StudyResult.add(result_or_error)` accumulation
- Cycle gap (`cycle_gap_seconds`) between complete cycles

**Test:** Integration test with mock `ExperimentOrchestrator` (monkeypatched via `unittest.mock` inside child, or replaced with a fast stub); 3-experiment study; assert all manifest entries written, `StudyResult` contains expected results.

**Dependency:** Steps 1–4. Requires `StudyResult.add()` to be implemented in `domain/results.py`.

### Step 6: `_api.py` and CLI Integration

Implement `run_study()`, `_run()`, `_to_study_config()`. Modify CLI `run.py` for study detection and routing. Finesse `display.py` for full study output.
- `run_study()` public signature matching confirmed library API design
- `run_experiment()` unchanged externally; internally routes via degenerate `StudyConfig`
- CLI YAML type detection heuristic
- Study pre-flight (study-level backend checks before first experiment launches)

**Test:** End-to-end with real PyTorch backend; 2-experiment study; verify `StudyResult` and two `ExperimentResult` files on disk; verify manifest at study completion.

---

## Scaling Considerations

| Study Size | Architectural Impact |
|------------|---------------------|
| 1–10 experiments (typical single-sweep) | No changes needed; sequential subprocess with Pipe is sufficient |
| 10–100 experiments (multi-backend sweep) | No architecture changes; overnight study; manifest checkpointing becomes critical for recovery |
| 100+ experiments (publication-quality, multi-cycle) | No architecture changes; study resume (`llem run --resume`) becomes essential (later milestone). Hundreds of `ExperimentResult` JSON files are fine — studies are sequential not parallel. |

Parallelism across experiments is not a goal. GPU ownership prevents within-machine parallelism. Cross-machine parallelism is a future concern not in v2.0 scope.

---

## Key Peer Evidence

| Tool | Pattern | Relevance | Confidence |
|------|---------|-----------|------------|
| **optimum-benchmark** | `ProcessLauncher`: `mp_ctx.Process + Pipe + daemon=False + 1MB file fallback` | Direct precedent for our exact subprocess + IPC pattern | HIGH — source verified |
| **optimum-benchmark** | Hydra `--multirun` for sequential config sweep dispatch | Sequential sweep -> process per run is the industry standard | HIGH — docs verified |
| **PyTorch docs** | `spawn` required for CUDA; `fork` explicitly unsupported | Confirms start method choice is a correctness requirement, not style | HIGH — official docs |
| **lm-eval-harness** | All tasks loaded upfront via `task_manager.load()`; then sequential evaluation | Confirms "expand first, iterate second" pattern | MEDIUM — source verified |
| **Ray Tune** | `experiment/trial_runner.json` checkpoint for study resume | Confirms manifest checkpoint pattern for long-running study recovery | MEDIUM — docs verified |
| **tqdm-multiprocess** | Queue-based progress events from worker processes to tqdm in parent | Confirms Queue + consumer thread for cross-process progress display | MEDIUM — docs verified |
| **Hydra BasicLauncher** | Full sweep computed before first job dispatched; jobs run serially by default | Confirms expand-first pattern | MEDIUM — docs verified |

---

## Sources

- PyTorch Multiprocessing with CUDA: https://docs.pytorch.org/docs/stable/notes/multiprocessing.html
- Optimum-benchmark ProcessLauncher (source verified): https://github.com/huggingface/optimum-benchmark/blob/main/optimum_benchmark/launchers/process/launcher.py
- Optimum-benchmark README (Hydra multirun + process isolation): https://github.com/huggingface/optimum-benchmark
- lm-eval-harness evaluator.py (task loading pattern): https://github.com/EleutherAI/lm-evaluation-harness
- Ray Tune checkpoints and resume: https://docs.ray.io/en/latest/tune/tutorials/tune-trial-checkpoints.html
- tqdm-multiprocess (queue-based cross-process progress): https://github.com/EleutherAI/tqdm-multiprocess
- Hydra multirun documentation: https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/
- `.product/designs/experiment-isolation.md`: Subprocess isolation pattern (confirmed design, source of truth)
- `.product/designs/architecture.md`: Module layout, call graph, StudyRunner pseudocode
- `.product/designs/study-resume.md`: ManifestWriter and StudyManifest schema
- `.product/designs/observability.md`: Progress display spec for study mode

---

*Architecture research for: LLM inference efficiency measurement tool — study/sweep execution integration*
*Researched: 2026-02-27*
