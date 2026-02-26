# Live Observability (Progress & Verbose Output)

**Status:** Accepted
**Date decided:** 2026-02-19
**Last updated:** 2026-02-25
**Research:** N/A

## Decision

Rich `Progress` + `Live` for study-level display; tqdm retained at experiment level. Three verbosity levels: standard / `--quiet` / `--verbose`. Progress → stderr; result JSON → stdout. Final summary always prints. Live power display deferred.

---

## Context

Researchers running overnight studies need moment-to-moment awareness of what the tool is
doing. Long experiments (hours of runtime, dozens of configurations) risk silent failures
or ambiguous hangs — the user cannot distinguish "still working" from "stalled". The tool
also needs to support scripted/pipeline use where human-readable output must be suppressed.
Three distinct use patterns must be satisfied: interactive debugging, standard monitoring,
and machine-readable pipeline output.

The existing codebase already uses Rich for some output. tqdm is used for experiment-level
progress. The design must integrate with both, avoid the `LiveError: Only one live display
may be active at once` constraint in Rich, and route output correctly between stdout/stderr
for Unix composability.

---

## Design Principle

**Maximally explicit.** The tool should show everything that is happening — in standard mode,
all experiment-level events; in verbose mode, all subprocess-level events too. Researchers
running overnight studies need to know what the tool is doing at a glance. Opacity is a bug.

This principle applies to both **config/input** (no hidden defaults, no silent coercions) and
**output** (no silent phases, no "please wait" with no indication of what is being waited for).

---

## Considered Options

### Sub-decision 1: Study-level progress display library

| Option | Pros | Cons |
|--------|------|------|
| **Rich `Progress` + `Live` — Chosen** | Multiple simultaneous task tracks; clean mixing of logs and progress; terminal-aware (`NO_COLOR`, `TERM=dumb`); already used in codebase | Requires careful single-instance management to avoid `LiveError` |
| tqdm for everything | Already in codebase; simple | Cannot cleanly interleave log lines with progress bars; no multi-task display |
| Custom ANSI cursor manipulation | Full control | High maintenance; fragile across terminals; reinvents the wheel |

### Sub-decision 2: Experiment-level (warmup/measurement) progress

| Option | Pros | Cons |
|--------|------|------|
| **tqdm retained at v2.0 — Chosen** | Working; sufficient; consistent with existing implementation | Inconsistency with Rich at study level; may need migration later |
| Migrate to Rich immediately | Unified library | Unnecessary scope increase for v2.0; risk of regression in working code |

### Sub-decision 3: Verbosity levels

| Option | Pros | Cons |
|--------|------|------|
| **Three levels: standard / `--quiet` / `--verbose` — Chosen** | Covers scripted use (quiet), interactive use (standard), debugging use (verbose); peers (lm-eval, pytest) use this model | Slightly more implementation surface than two levels |
| Two levels: quiet / verbose | Simpler | No clean "default interactive" middle ground; too chatty for standard use if verbose is the default |
| Single level | Minimal | Cannot serve both scripted and debugging workflows |

### Sub-decision 4: Final summary suppression

| Option | Pros | Cons |
|--------|------|------|
| **Final summary always prints regardless of verbosity — Chosen** | Final summary is a scientific record; must always appear | Slightly surprises users who expect `--quiet` to silence everything |
| `--quiet` suppresses everything including summary | Maximum silence | Loses the scientific record in scripted runs; user must redirect stdout manually |

### Sub-decision 5: stdout vs stderr routing

| Option | Pros | Cons |
|--------|------|------|
| **Progress → stderr; result JSON/summary → stdout — Chosen** | Standard Unix convention; enables `llem run > results.json 2>/dev/null` | Requires explicit routing discipline in implementation |
| All output to stdout | Simple | Cannot separate human-readable noise from machine-readable results |
| All output to stderr | Results always visible | Results not capturable via stdout redirection |

### Sub-decision 6: Live power display in v2.0

| Option | Pros | Cons |
|--------|------|------|
| **Defer live power display to v2.1 — Chosen** | Requires Zeus streaming, not yet in scope; placeholder in verbose output sufficient | Users cannot monitor power consumption in real time at v2.0 |
| Implement live power display in v2.0 | More informative | Depends on Zeus streaming integration not yet planned |

---

## Decision

We will use Rich `Progress` + `Live` for study-level display, retain tqdm for
experiment-level batch/prompt progress at v2.0, and implement three verbosity levels
(standard, `--quiet`, `--verbose`). Progress output routes to stderr; result JSON/summary
routes to stdout. The final summary always prints regardless of verbosity. Live power display
is deferred to v2.1.

Rationale: Rich handles multiple simultaneous task tracks cleanly and is already present in
the codebase. tqdm is retained at experiment level to avoid regression risk at v2.0. Three
verbosity levels match the established pattern in peer tools (lm-eval, pytest). The
stdout/stderr split is a Unix convention that enables pipeline composability. The final
summary is a scientific record and must not be suppressible.

---

## Consequences

Positive:
- Researchers get live signal during long studies (inline results as each experiment finishes).
- Scripted use supported via `--quiet` and stdout/stderr separation.
- Verbose mode exposes full subprocess events for debugging, matching Docker BuildKit's
  "child lines under parent" pattern.
- Thermal gap countdowns prevent ambiguity between "waiting" and "stalled".

Negative / Trade-offs:
- Two progress libraries in the codebase (Rich at study level, tqdm at experiment level)
  creates minor inconsistency until tqdm is migrated in a later version.
- `transient=True` and single-Progress-instance discipline must be enforced — incorrect
  nesting raises `LiveError`.

Neutral / Follow-up decisions triggered:
- Zeus streaming integration needed for live power display (v2.1 concern).
- Per-backend container log streaming for Docker studies deferred to v2.2.
- TUI (full terminal UI) explicitly rejected — the Rich Live approach is sufficient.

---

## Output Levels

### `llem run` (single experiment)

**Standard (default):**
```
Experiment: llama-3.1-8b / pytorch / bf16 / batch=8
  Pre-flight     ✓  GPU, backend, model, energy backend
  Warmup         ████████░░░░░░░░  8/20 prompts  CV=0.087 → target <0.050
  Measuring      ██████████████░░  70/100 prompts

Result saved: results/llama-3.1-8b_pytorch_2026-02-19T14-30.json

  Energy         312.4 J  (3.12 J/request)
  Throughput     847 tok/s
  Latency        TTFT 142ms  ITL 28ms
  Duration       4m 32s
```

**`--verbose` additionally shows:**
```
  [Loading model]  meta-llama/Llama-3.1-8B  bf16  →  14.2 GB VRAM
  [Energy backend]  NVML direct (Zeus not installed)
  [Warmup]  prompt 8: latency=148ms  CV=0.087 (running avg of last 5)
  [Measuring]  prompt 70: latency=141ms  tokens=87
```

**`--quiet`:**
```
Result saved: results/llama-3.1-8b_pytorch_2026-02-19T14-30.json

  Energy         312.4 J  (3.12 J/request)
  Throughput     847 tok/s
  Latency        TTFT 142ms  ITL 28ms
  Duration       4m 32s
```
(No warmup/measurement progress; final summary still prints.)

---

### `llem run study.yaml` (multi-experiment study)

**Standard (default):**
```
Study: batch-size-sweep-2026-02                  cycle 1/3   00:45:32   ETA ~2:15:00
  [================================================] 3/12 experiments

  ✓ [1/12]  pytorch / batch=1 / bf16    →   87.3 J   1,243 tok/s   (2m 14s)
  ✓ [2/12]  pytorch / batch=4 / bf16    →  142.1 J     847 tok/s   (3m 02s)
  ▶ [3/12]  pytorch / batch=8 / bf16    →  measuring...  (00:01:23 elapsed)
  · [4/12]  pytorch / batch=16 / bf16   →  (queued — thermal gap 60s)
  · [5/12]  pytorch / batch=32 / bf16   →  (queued)
  ...
```

Symbols: `✓` completed, `▶` running, `·` queued, `✗` failed.

Completed results appear inline as each experiment finishes — researchers get live signal
during long studies rather than waiting for the full study summary.

**`--verbose` additionally shows subprocess events as indented sub-lines:**
```
  ▶ [3/12]  pytorch / batch=8 / bf16    →  measuring...
      Loading model: meta-llama/Llama-3.1-8B  (14.2 GB, bf16)
      Energy backend: NVML direct
      Warmup: 12/20 prompts  CV=0.062 → target <0.050
      Measuring: 44/100 prompts
```

This is the "subprocess output as child lines" pattern — analogous to Docker BuildKit
showing per-layer progress under the overall build progress.

**`--quiet`:** No live display. On completion:
```
Study complete: 12/12 ran, 0 failed, 0 skipped
Results: results/batch-size-sweep-2026-02/
```

---

## Implementation

**Library stack:**

| Layer | Library | Notes |
|-------|---------|-------|
| Study-level progress header | Rich `Live` + `Progress` | Persistent, updates in place |
| Experiment status table | Rich `Console.print()` inside Live | Scrolls cleanly past progress bar |
| Experiment-level warmup/measure | tqdm (existing) | Retained at v2.0; may migrate to Rich later |
| Verbose subprocess logs | Rich `console.print()` with `[dim]` markup | Indented, visually subordinate |
| Final summary | Rich `Panel` + `Table` | Always printed, regardless of verbosity |

**Critical implementation rules:**
- All output inside a Rich `Progress` context must use `progress.console.print()`. Rich also
  redirects `stdout`/`stderr` by default inside a Progress context, so subprocess output
  naturally routes correctly — but `progress.console.print()` is the explicit, safe pattern.
- `transient=True` on experiment-level inner tasks (warmup, measurement progress): they
  disappear on completion, leaving only the completed result line printed above the bars.
  This produces the clean "completed items scroll up, current item shows progress" pattern.
- Do NOT nest `rich.progress.track()` calls — this raises `LiveError: Only one live display
  may be active at once`. Use a single `Progress` instance with multiple `add_task()` calls.

**Pattern for mixing subprocess logs with study progress:**
```python
# Inside StudyRunner, each experiment subprocess emits events over a queue.
# The study display layer subscribes and routes to the correct output level.

with Progress(console=stderr_console) as progress:
    study_task = progress.add_task("Study", total=n_experiments)

    for event in experiment_events:
        if event.type == "log" and verbose:
            stderr_console.print(f"    [dim]{event.message}[/dim]")
        elif event.type == "result":
            stderr_console.print(format_inline_result(event.result))
            progress.update(study_task, advance=1)
```

**Environment variables respected:**
- `NO_COLOR` — disables all colour output (Rich respects natively)
- `TERM=dumb` — falls back to ASCII-only progress (Rich detects)
- `LLM_ENERGY_JSON_OUTPUT=true` — existing env var; suppresses all human-readable output,
  emits machine-readable JSON on stdout only (for pipeline use)

## Thermal Gap Display

During `config_gap_seconds` and `cycle_gap_seconds` pauses, display a countdown rather
than silence — researchers need to know the tool is alive and waiting intentionally:

```
  ▶ [4/12]  pytorch / batch=16 / bf16   →  waiting thermal gap  (55s remaining)
```

---

## Deferred

- Live power draw display during measurement (requires Zeus streaming — v2.1)
- Per-backend container log streaming for Docker studies (v2.2)
- TUI (full terminal UI) — not warranted; the Rich Live approach is sufficient

---

## Related

- [../designs/observability.md](../designs/observability.md): Implementation design for the display layer
- [cli-ux.md](cli-ux.md): Verbosity flags (`--quiet`, `--verbose`) as part of the CLI contract
- [future-versions.md](future-versions.md): Zeus streaming for live power display (v2.1)
