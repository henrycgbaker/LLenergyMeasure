# N-X03: Progress Reporting Architecture

**Module**: `src/llenergymeasure/progress.py`
**Risk Level**: MEDIUM
**Decision**: Keep — v2.0 (tqdm for experiment-level progress; Rich for study-level as per observability design)
**Planning Gap**: `designs/observability.md` explicitly specifies Rich for progress display. The current code uses tqdm. The document acknowledges this mismatch explicitly (line: "Experiment warmup/measure: tqdm (existing) — Retained at v2.0; migrate to Rich later") — so this is not a contradiction, but it means the existing `ProgressTracker` class may be partially superseded by the new Rich-based display layer rather than fully carried forward.

---

## What Exists in the Code

**Primary file(s)**: `src/llenergymeasure/progress.py`
**Key classes/functions**:
- `VerbosityLevel` (line 22) — Enum with three values: `QUIET = auto()`, `NORMAL = auto()`, `VERBOSE = auto()`
- `get_verbosity_from_env()` (line 30) — reads `LLM_ENERGY_VERBOSITY` env var; maps "quiet"/"normal"/"verbose" to `VerbosityLevel`; defaults to `NORMAL`
- `ProgressTracker` (line 50) — tqdm-based wrapper with: `is_main_process: bool` guard (only shows on process 0 in multi-process runs), `should_show` property (line 99), verbosity-aware display, `update(n, latency_ms=None)` with running average latency in postfix, `warning()` using `tqdm.write()` to preserve bar position, `info()` for verbose-only messages
- `batch_progress()` (line 185) — context manager; wraps `ProgressTracker` with `unit="batch"`, `position` for nested bars
- `prompt_progress()` (line 219) — context manager; wraps `ProgressTracker` with `unit="prompt"`

The `ProgressTracker` uses a custom `bar_format` (line 114): `"{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"` — compact format showing elapsed and estimated remaining time.

Multi-process awareness: `is_main_process=True` means only process 0 (the main accelerate process) shows the progress bar. Worker processes remain silent. This is essential for multi-GPU PyTorch runs where multiple processes would otherwise each render a progress bar.

Total: ~251 lines including docstrings.

## Why It Matters

The `VerbosityLevel` enum and `get_verbosity_from_env()` are the system-wide verbosity authority — they are used in `summaries.py` to gate config display. `ProgressTracker` provides the only user-facing progress signal during inference (without it, a 100-prompt experiment runs silently). The `is_main_process` guard is essential for distributed correctness — removing it would cause garbled output in multi-GPU runs.

## Planning Gap Details

`designs/observability.md` (lines 148–149):
> "Experiment warmup/measure: tqdm (existing) — Retained at v2.0; migrate to Rich later"

This explicitly confirms tqdm is retained at v2.0, meaning `ProgressTracker` survives the rebuild. However, the observability design adds study-level progress using Rich `Progress` with `Live` — a layer that does not exist yet. The two must coexist correctly: Rich's `Live` context manager redirects stdout/stderr, which can break tqdm output if they are run in the same terminal session simultaneously.

The critical rule from `observability.md` (line 154):
> "All output inside a Rich Progress context MUST use `progress.console.print()`. Rich redirects stdout/stderr inside a Progress context — print() without this breaks layout."

This means the tqdm `ProgressTracker` (which writes to `sys.stderr` directly) cannot be used inside a Rich `Progress` context. The Phase 5 orchestrator must ensure tqdm bars are only active when the study-level Rich `Live` display is NOT active, or migrate warmup/measurement tracking to Rich entirely.

## Recommendation for Phase 5

Carry `VerbosityLevel`, `get_verbosity_from_env()`, and `ProgressTracker` forward into `progress.py`. These are immediately needed and working.

At the v2.0 layer boundary:
- `llem run` (no study-level Rich `Live` active): use `ProgressTracker` with tqdm as-is for warmup and measurement phases — this is the confirmed plan
- `llem study` (study-level Rich `Live` active): do NOT use tqdm `ProgressTracker` inside the subprocess event loop. Instead, route progress events via the multiprocessing queue to the parent process, where the Rich display can render them as `[dim]` indented lines under the experiment row (as shown in `observability.md`)

Keep `batch_progress()` and `prompt_progress()` context managers — they are used by the inference backends and should remain available.

One cleanup: `DEFAULT_STREAMING_WARMUP_REQUESTS` is imported in `summaries.py` from `constants.py` (not `progress.py`) — this is correct, no action needed.
