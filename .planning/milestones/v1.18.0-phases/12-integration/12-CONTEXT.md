# Phase 12: Integration - Context

**Gathered:** 2026-02-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Wire `llem run study.yaml` end-to-end: CLI study detection, `StudyRunner.run()` delegation, `run_study()` public API implementation, `_run()` dispatcher body (single vs multi), CLI study flags (`--cycles`, `--order`, `--no-gaps`), study progress display, `StudyResult` assembly with full schema, and multi-backend pre-flight hard error. This phase connects Phases 9-11 components into a working study pipeline.

</domain>

<decisions>
## Implementation Decisions

### Study detection & routing
- Top-level key check: if YAML has `sweep:` or `experiments:` keys, it's a study; otherwise single experiment
- CLI always routes through `_run(StudyConfig)` internally — study YAML goes through `load_study_config()` then `_run()`, experiment YAML goes through `load_experiment_config()` then wraps in `StudyConfig` then `_run()`
- `_run()` dispatch: if `len(experiments) == 1` and `n_cycles == 1`, run in-process (no subprocess overhead); otherwise delegate to `StudyRunner`

### CLI flag interaction with study YAML
- CLI always wins (Hydra-style): `--model X` narrows the sweep grid to just that model
- Warning to stderr when CLI flags narrow a sweep: `"Warning: --model gpt2 narrows sweep from 6 to 4 experiments"`
- Warning also propagated into `StudyResult.summary.warnings` list for traceability in saved results
- Document this behaviour in user-facing docs (Phase 13)

### CLI study flag override semantics
- Standard merge: CLI flags override matching `execution:` block fields
- `--cycles 5` overrides `n_cycles`; `--order shuffled` overrides `cycle_order`; `--no-gaps` sets both `experiment_gap_seconds` and `cycle_gap_seconds` to 0
- Unspecified flags keep YAML values
- CLI effective defaults (when neither YAML nor CLI specifies): `n_cycles=3`, `cycle_order="shuffled"`
- Pydantic defaults remain conservative (`n_cycles=1`, `cycle_order="sequential"`) — CLI layer applies effective defaults

### Rename: config_gap_seconds to experiment_gap_seconds
- `config_gap_seconds` in `ExecutionConfig` renamed to `experiment_gap_seconds` for consistency with naming elsewhere
- Applies to: ExecutionConfig model (Phase 9), YAML field, CLI display, manifest, StudyResult measurement_protocol
- Phase 12 propagates this rename through all integration points

### Progress display
- Compact status line per experiment, scrolling up as completed:
  `[3/12] <icon> model backend precision -- elapsed (energy)`
- Icons: completed, failed, running spinner
- Gap countdown inline below active experiment: `Config gap: 47s remaining (Enter to skip)`
- `--quiet` suppresses progress lines and countdowns but still shows final summary table

### Study summary display
- Rich table showing all experiments with columns: Config, Cycle, Time, Energy, tok/s
- Failed experiments shown inline with error type instead of metrics
- Footer with totals: experiment count, cycle count, wall time, total energy
- Output path displayed at end

### StudyResult assembly (RES-13)
- `measurement_protocol`: flat dict from `ExecutionConfig` — `{n_cycles, cycle_order, experiment_gap_seconds, cycle_gap_seconds, shuffle_seed, experiment_timeout_seconds}`
- `study_design_hash`: carried from `StudyConfig`
- `result_files`: list of paths to per-experiment result files (RES-15)
- `summary`: `StudySummary` with `total_experiments`, `completed`, `failed`, `total_wall_time_s`, `total_energy_j`, `warnings: list[str]`
- Warnings list captures: CLI narrowing messages, failed experiment counts, any other runtime warnings
- `experiments` list contains all `ExperimentResult` objects (successful have full metrics, failed have error info)

### Multi-backend pre-flight
- Study YAML with multiple backends (e.g., `backend: [pytorch, vllm]`) raises `PreFlightError` at pre-flight
- Message directs user to Docker runner (M3): "Multi-backend studies require Docker isolation (available in M3). Use a single backend for now."
- Exit code 1

### Claude's Discretion
- Exact implementation of sweep narrowing logic
- How `load_study_config()` integrates with existing loader patterns
- Rich table formatting details and colour choices
- How failed experiment results are structured (error-only ExperimentResult vs separate failure model)
- Integration test structure

</decisions>

<specifics>
## Specific Ideas

- Progress display inspired by the compact status line mockup: completed lines scroll up, active line updates in-place, gap countdown appears below
- Summary table similar to the single-experiment result display but scaled to multiple rows with cycle column
- Warning propagation pattern: warnings appear both in stderr (interactive) and in StudyResult.summary.warnings (persisted) — ensures traceability whether results are viewed live or from files
- The `_run()` dispatcher should be the only place that decides in-process vs subprocess — clean separation of concerns

</specifics>

<deferred>
## Deferred Ideas

- Document CLI flag + study YAML interaction in user-facing docs -- Phase 13
- `--resume` flag for interrupted studies -- M4 (manifest always-on in M2 enables this)

</deferred>

---

*Phase: 12-integration*
*Context gathered: 2026-02-27*
