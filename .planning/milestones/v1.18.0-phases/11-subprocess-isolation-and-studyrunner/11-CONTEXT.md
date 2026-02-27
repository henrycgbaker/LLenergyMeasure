# Phase 11: Subprocess Isolation and StudyRunner - Context

**Gathered:** 2026-02-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Each experiment in a study runs in a freshly spawned subprocess with clean CUDA state. Results return via `Pipe`, progress events flow via `Queue`, and the study survives experiment failures, timeouts, and SIGINT without data corruption. The subprocess isolation pattern, IPC mechanism (`spawn` + `Pipe` + `Queue`), daemon mode (`daemon=False`), and kill strategy (`SIGKILL` for timeouts) are locked decisions from `.product/decisions/experiment-isolation.md` and `.product/designs/experiment-isolation.md`.

This phase implements `StudyRunner` on top of the `ManifestWriter` from Phase 10.

</domain>

<decisions>
## Implementation Decisions

### SIGINT / Ctrl+C Behaviour
- First Ctrl+C: SIGTERM to active subprocess, wait 2-3s grace period for clean result
- Second Ctrl+C (or grace period expired): escalate to SIGKILL
- Manifest status set to `"interrupted"` (not `"failed"` — it's a user action, not an error)
- Exit code 130 (standard SIGINT convention)
- Ctrl+C during a thermal/cycle gap = full interrupt (stop the study immediately, no "skip gap" interpretation)
- Print brief summary before exit: "3/12 experiments completed. Results in ./results/study_name/. Manifest: interrupted."
- Children install `signal.signal(signal.SIGINT, signal.SIG_IGN)` — parent owns the signal
- Fully scriptable: no interactive prompts, signals are standard POSIX behaviour

### Gap Countdown UX
- Inline countdown that overwrites in-place: `Config gap: 47s remaining (Enter to skip)`
- Same display format for both gap types, distinguished by label: `Config gap:` vs `Cycle gap:`
- Enter keypress skips remaining gap time (interactive convenience; non-interactive environments ignore this)
- Auto-format duration: under 120s shows `47s remaining`, over 120s shows `4m 32s remaining`
- Fully scriptable: gaps configured via YAML `execution:` block or `--no-gaps` CLI flag

### Timeout Calculation
- Internal heuristic by default: `max(n * 2, 600)` — 2s per prompt estimate, minimum 10 minutes
- Keep heuristic simple (no model-size scaling) — generous enough for all practical cases
- Escape hatch: optional `experiment_timeout_seconds` field in `execution:` block (YAML only, no CLI flag)
- On timeout: display shows elapsed time and limit: `✗ llama-3.1-70b pytorch bf16 — TIMEOUT after 600s (limit: 600s)`

### Cycle Ordering Logic
- Three modes: `sequential`, `interleaved`, `shuffled`
- **Sequential** (A,A,A,B,B,B,C,C,C): all cycles of one experiment, then next. `cycle_gap` between cycles of the same experiment. `config_gap` between different experiments.
- **Interleaved** (A,B,C,A,B,C,A,B,C): round-robin — each round visits all experiments once. Deterministic order. Reduces thermal autocorrelation.
- **Shuffled**: random permutation of all runs. Seeded with `random_seed` from StudyConfig for reproducibility. Strongest protection against order effects.
- Gap semantics uniform across all modes: `config_gap` between every experiment pair, `cycle_gap` after every N runs (one full round, where N = number of unique configs)

### Claude's Discretion
- Exact SIGTERM grace period duration (2-3s range)
- Rich display integration details for countdown
- Internal implementation of Enter-to-skip (threading, select, etc.)
- How the seeded shuffle integrates with existing `random_seed` field

</decisions>

<specifics>
## Specific Ideas

- SIGINT handling pattern: SIG_IGN in child workers, parent owns the signal via custom handler that sets a threading.Event — standard Python multiprocessing best practice
- Gap skip with Enter should degrade gracefully in non-TTY environments (no crash, just runs full gap)
- The "interrupted" manifest status is distinct from "failed" — enables future study resume (Phase deferred)

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 11-subprocess-isolation-and-studyrunner*
*Context gathered: 2026-02-27*
