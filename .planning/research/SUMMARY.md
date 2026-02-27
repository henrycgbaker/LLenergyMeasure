# Project Research Summary -- M2 Study/Sweep Execution

**Project:** LLenergyMeasure v2.0 -- M2 (Study/Sweep Execution)
**Domain:** Multi-experiment study execution for LLM inference efficiency measurement
**Researched:** 2026-02-27
**Confidence:** HIGH

> **Supersedes:** The prior SUMMARY.md (2026-02-25) covered M1 decisions audit and
> harmonisation. This file covers M2-specific research only. M1 decisions referenced
> here are already confirmed and implemented.

---

## Executive Summary

M2 adds multi-experiment study execution to the working single-experiment foundation from M1. All four research files converge on a single architectural approach: sequential subprocess dispatch using `multiprocessing.get_context("spawn")` with dual-channel IPC (`Pipe` for results, `Queue` for progress events). This is not speculative -- optimum-benchmark (HuggingFace) uses the identical pattern (subprocess + Pipe + 1MB file fallback + `daemon=False`) for the identical reason: PyTorch's CUDA allocator does not fully release GPU memory across `del model; torch.cuda.empty_cache()`, making subprocess isolation a correctness requirement, not an optimisation choice.

**No new dependencies are required for M2.** All subprocess isolation, grid expansion (`itertools.product`), manifest checkpointing (`os.replace` atomic rename), and thermal gap management (`time.sleep`) use Python stdlib plus existing project dependencies (Pydantic 2.12.5, Rich 14.2.0, Loguru 0.7.3). The implementation produces three new files in a `study/` package (`runner.py`, `grid.py`, `manifest.py`) plus modifications to four existing modules (`_api.py`, `cli/commands/run.py`, `cli/display.py`, `domain/results.py`).

The primary risks are concentrated in the subprocess/IPC layer. Seven critical pitfalls are identified, all with documented prevention strategies. The most dangerous is `fork` vs `spawn` (CP-1): using Linux's default `fork` start method causes silent CUDA corruption where measurements appear valid but are not. The most complex is SIGINT handling (CP-3): Ctrl+C during a study touches signal handling, multiprocessing, CUDA teardown, manifest writes, and terminal display simultaneously. The most common is sweep combinatorial explosion (CP-5): a four-dimensional sweep with 5 values each generates 625 experiments -- 31+ hours with thermal gaps -- and the tool must show this before starting. All pitfalls have clear verification steps and phase mappings.

---

## Key Findings

### Recommended Stack

**No new `pyproject.toml` entries required for M2.** Every capability uses Python stdlib or an existing dependency.

**Core technologies (all stdlib):**
- `multiprocessing.get_context("spawn")` -- subprocess isolation per experiment; PyTorch mandates `spawn` for CUDA; scoped context avoids breaking other libraries
- `multiprocessing.Pipe(duplex=False)` -- result IPC from child to parent; 3x faster than Queue for single payloads; 1MB file fallback for large results
- `multiprocessing.Queue` -- progress events from child; buffered; never blocks worker on slow parent
- `itertools.product` -- Cartesian grid expansion; standard across all ML sweep tools (Hydra, W&B, Ray Tune, Optuna, lm-eval)
- `os.replace()` -- atomic manifest write via POSIX `rename(2)`; prevents partial JSON on interruption
- `threading.Thread(daemon=True)` -- drains progress Queue in parent while main thread blocks at `p.join()`

**Existing dependencies used (no version changes needed):**
- Pydantic 2.12.5 -- `model_dump_json()` for manifest and result serialisation
- Rich 14.2.0 -- study-level progress display, thermal gap countdown
- Loguru 0.7.3 -- study lifecycle logging

### Expected Features

**Must have (table stakes for M2 launch):**
- Grid sweep grammar with backend-scoped dotted notation
- Explicit experiment list (`experiments:` key)
- Subprocess isolation per experiment (spawn context, fresh CUDA state)
- Progress display during study (tqdm bar + per-experiment status lines)
- Thermal gap countdown (visible countdown, not silence)
- Skip-and-continue on experiment failure
- StudyManifest checkpoint (always-on, written at every state transition)
- n_cycles repetition with cycle_order (sequential/interleaved/shuffled)
- Dry-run grid preview

**Should have (differentiators -- no peer tool has these):**
- Backend-scoped sweep dimensions (independent grids per backend)
- Study design hash excluding execution block
- Interleaved cycle ordering for thermal fairness
- Structured per-experiment failure IPC (type + message + traceback)
- Multi-level pre-flight validation (L1 Pydantic upfront, L3 runtime graceful)

**Defer (M4+):**
- `--resume` from interrupted study (manifest written in M2; resume logic M4)
- `cold_start: true` (changes measurement semantics)
- Bootstrap CIs on study-level aggregation
- Optimisation/HPO features (metric targets, early termination, pruning)
- Parallel experiment execution (GPU energy contamination)

### Architecture Approach

Three new files, four modified files. The child process imports and calls `ExperimentOrchestrator` directly -- no CLI re-entry, no serialisation boundary beyond the Pipe. Single-experiment `run_experiment()` runs in-process (no subprocess needed when clean GPU state at start is guaranteed). Only `run_study()` uses subprocess isolation.

**Major components:**
1. `StudyRunner` (`study/runner.py`) -- orchestrates sequential subprocess dispatch; coordinates manifest, thermal gaps, cycle ordering; owns the study execution loop
2. `grid.expand()` (`study/grid.py`) -- pure function; resolves StudyConfig into ordered `list[ExperimentConfig]`; handles n_cycles, cycle_order, invalid combo exclusion at expansion time
3. `ManifestWriter` (`study/manifest.py`) -- writes `study_manifest.json` at every state transition via atomic rename; `mark_running()`, `mark_completed()`, `mark_failed()`
4. `_run_experiment_worker()` (`study/runner.py`) -- module-level function in child process; calls `ExperimentOrchestrator(config).run()`; sends result via Pipe, progress via Queue
5. `_collect_result()` (`study/runner.py`) -- parent-side; interprets process exit state; handles success/crash/timeout with three distinct code paths

### Critical Pitfalls

1. **Fork instead of spawn (CP-1)** -- use `get_context("spawn")`, never `set_start_method()` globally; silent CUDA corruption where measurements appear valid but are not; the first experiment often works by accident
2. **Pipe buffer deadlock (CP-2)** -- OS pipe buffer is ~64KB; check payload size before `conn.send()`; file-based IPC fallback for >1MB; cap traceback strings to prevent runaway error payloads
3. **SIGINT/Ctrl+C dirty state (CP-3)** -- catch `KeyboardInterrupt` in parent; `p.kill()` the child (not `p.terminate()` -- SIGTERM may be ignored by CUDA); send sentinel to progress queue; mark manifest as `"failed"` with `UserInterrupted`; this is the hardest path to get right
4. **Sweep combinatorial explosion (CP-5)** -- always show pre-flight summary with experiment count and estimated wall time; add configurable hard cap requiring `--yes` to proceed
5. **Manifest corruption from non-atomic writes (CP-6)** -- use `write-to-temp + os.replace()` from day one; never `path.write_text()` for checkpoint files
6. **Timeout miscalibration for large models (CP-7)** -- 70B models need 30+ minutes for loading alone; make timeout formula model-size-aware or configurable via `execution.experiment_timeout_seconds`

---

## Implications for Roadmap

M2 splits into 4 sequential phases following a strict dependency chain. Each phase produces a testable artifact before the next begins.

### Phase 1: Grid Expansion and StudyConfig

**Rationale:** Pure function, zero side effects, no subprocess or IPC needed. All downstream phases depend on having a resolved `list[ExperimentConfig]`.
**Delivers:** `study/grid.py` with `expand_grid()`; `StudyConfig` Pydantic model with `sweep:`, `experiments:`, `execution:` blocks; Pydantic L1 pre-flight validation; pre-flight experiment count display.
**Addresses:** Grid sweep grammar (CFG-11 to CFG-14), explicit experiment list, n_cycles, cycle_order, backend-scoped sweep dimensions.
**Avoids:** CP-5 (sweep explosion) -- pre-flight count display is implemented alongside the expander, not deferred.
**Test strategy:** Unit tests with mock StudyConfig; assert output list shape, ordering for each cycle_order mode, and invalid combo exclusion.

### Phase 2: Manifest Writer

**Rationale:** Pure I/O module, no subprocess needed. Must exist before StudyRunner can checkpoint state.
**Delivers:** `study/manifest.py` with ManifestWriter, StudyManifest, ExperimentManifestEntry models; atomic write via `os.replace()`.
**Addresses:** StudyManifest checkpoint (STU-08, STU-09), structured failure recording.
**Avoids:** CP-6 (manifest corruption) -- atomic rename from day one.
**Test strategy:** Unit tests; create manifest, run state transitions, verify file integrity after simulated interruption.

### Phase 3: Subprocess Isolation and IPC

**Rationale:** Core runtime complexity. Depends on M1 `ExperimentOrchestrator` being importable. Progress display depends on this.
**Delivers:** `_run_experiment_worker()`, `_send_result()`, `_collect_result()`, progress Queue + consumer daemon thread, Pipe with 1MB file fallback, SIGINT handling.
**Addresses:** Subprocess isolation (STU-01 to STU-04), structured failure IPC, progress display, thermal gap countdown, skip-and-continue.
**Avoids:** CP-1 (fork vs spawn), CP-2 (Pipe deadlock), CP-3 (SIGINT), CP-4 (daemon=True).
**Test strategy:** Integration tests with mock ExperimentOrchestrator; test success/crash/timeout/SIGKILL paths; test large payload fallback; manual Ctrl+C test on GPU hardware.

### Phase 4: StudyRunner Integration, API, and CLI

**Rationale:** Wires phases 1-3 together. Thinnest layer -- just wiring and polishing.
**Delivers:** `StudyRunner.run()`, `run_study()` in `_api.py`, CLI study detection and routing, study-mode display (tqdm + per-experiment lines + thermal countdown), StudyResult assembly, `--dry-run` for studies, study CLI flags (`--cycles`, `--no-gaps`, `--order`), multi-backend hard error (CM-10).
**Addresses:** All remaining table-stakes and differentiator features; study design hash; run_study() library function (LA-02, LA-05).
**Avoids:** CP-7 (timeout miscalibration) -- model-aware timeout formula.
**Test strategy:** End-to-end with real PyTorch backend; 2-3 experiment study; verify manifest, result files, display output.

### Phase Ordering Rationale

- Grid expansion must come first because all downstream phases consume its output
- Manifest must precede subprocess work because StudyRunner calls `manifest.mark_running()` before spawning each child
- Subprocess isolation is the highest-risk phase and benefits from having grid and manifest already tested
- Integration is last because it is mechanical wiring of tested components
- Each phase is independently testable -- no phase requires another phase's code to be written simultaneously

### Research Flags

**Needs deeper research during planning:**
- **Phase 3 (Subprocess Isolation):** SIGINT/Ctrl+C interaction with CUDA teardown and multiprocessing is complex and not fully unit-testable. The `_collect_result()` three-path logic (success/crash/timeout) has edge cases around partial Pipe reads. Manual verification on GPU hardware is required.

**Standard patterns (skip research):**
- **Phase 1 (Grid Expansion):** Well-documented `itertools.product` pattern; sweep grammar fully designed in `.product/designs/study-yaml.md`
- **Phase 2 (Manifest Writer):** Simple Pydantic models + atomic file write; two lines of code
- **Phase 4 (Integration):** All components exist; wiring is mechanical

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | No new deps; all stdlib + existing; versions verified locally on Python 3.12.12; optimum-benchmark uses identical pattern |
| Features | HIGH | All features pre-decided in `.product/decisions/` and `.product/designs/`; peer comparison verified against source code of 4 tools |
| Architecture | HIGH | Pattern directly validated by optimum-benchmark source code; PyTorch docs mandate the approach; build order follows natural dependency chain |
| Pitfalls | HIGH | 7 critical pitfalls identified with prevention strategies and verification steps; SIGINT path is the most complex and least unit-testable |

**Overall confidence:** HIGH

### Gaps to Address

- **SIGINT manual testing:** The Ctrl+C path touches too many systems for pure unit testing. Must be manually verified on GPU hardware during Phase 3 implementation.
- **Timeout formula calibration:** Model-size heuristic (`"70b" in model.lower()`) is a rough proxy. Consider making `experiment_timeout_seconds` configurable in `execution:` block.
- **tqdm vs Rich conflict:** Progress display uses both `tqdm.write()` and Rich `Progress`. If they conflict on stderr, may need to choose one framework. Rich's own `Progress` class may be sufficient without tqdm.
- **Multi-backend hard error wording:** The error for `backend: [pytorch, vllm]` in M2 must clearly direct users to the Docker runner (M3), not just say "not supported."
- **Large manifest scaling:** `model_dump_json()` serialises the entire manifest on every write. Acceptable for v2.0 (<500 experiments). For v3.0+, consider append-only format.

---

## Cross-Research Consensus

All four research files agree on these points without exception:

1. **`spawn` not `fork`** -- STACK, FEATURES, ARCHITECTURE, and PITFALLS all reference this independently
2. **No new dependencies** -- all four confirm stdlib + existing deps are sufficient
3. **Sequential execution only** -- no parallel experiments (GPU energy contamination)
4. **Expand grid before dispatch** -- all four reference the "resolve everything first, iterate second" pattern
5. **Manifest is always-on** -- not opt-in; negligible cost; enables future resume
6. **optimum-benchmark is the closest peer** -- referenced in all four files as the primary precedent
7. **ExperimentOrchestrator is unmodified** -- M2 wraps it; does not change it

No conflicts or tensions were identified between the four files.

---

## Sources

### Primary (HIGH confidence)
- PyTorch Multiprocessing with CUDA -- `spawn` mandate, `fork` incompatibility
- optimum-benchmark ProcessLauncher source (GitHub) -- identical subprocess + Pipe + 1MB fallback + daemon=False
- Python multiprocessing docs -- `get_context()`, Pipe buffer, Queue semantics
- `.product/designs/experiment-isolation.md` -- StudyRunner pattern (project source of truth)
- `.product/designs/study-yaml.md` -- sweep grammar, dotted key notation, cycle_order
- `.product/designs/study-resume.md` -- StudyManifest schema, ManifestWriter API
- `.product/designs/observability.md` -- progress display spec for study mode
- Local verification -- all stdlib functions tested on Python 3.12.12

### Secondary (MEDIUM confidence)
- lm-eval harness -- "expand first, iterate second" task loading pattern
- Ray Tune -- manifest checkpoint pattern (trial_runner.json)
- Hydra BasicLauncher -- full sweep computed before first job dispatched
- vLLM bench sweep -- param_sweep module, per-config status display

---
*Research completed: 2026-02-27*
*Ready for roadmap: yes*
