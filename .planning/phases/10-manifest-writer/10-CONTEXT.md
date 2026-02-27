# Phase 10: Manifest Writer - Context

**Gathered:** 2026-02-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Atomic, corruption-proof checkpoint file (`manifest.json`) that records the state of every experiment during a study run. Written after each state transition so an interrupted study leaves a readable manifest. Includes `StudyManifest` Pydantic model, `ManifestWriter` with atomic writes, and study output directory layout.

Resume logic (`--resume`) is a separate phase (M4). This phase builds the manifest infrastructure that resume will later consume.

</domain>

<decisions>
## Implementation Decisions

### Atomic write mechanics
- Use temp file + `os.replace()` pattern (POSIX atomic rename semantics)
- No `fsync` — `os.replace()` is sufficient; no peer tool uses fsync either
- Reuse the existing `_atomic_write()` utility from `persistence.py` — one atomic write implementation for the whole codebase
- Write after **every state transition**: `mark_running()`, `mark_completed()`, `mark_failed()` — maximum checkpoint fidelity
- On manifest write failure: **log warning and continue the study** (matches all peer tools — CodeCarbon, Zeus, lm-eval all log-and-continue). The manifest is secondary to experiment result files.
- On study output directory creation failure: **fail fast with `StudyError`** — this is a pre-flight failure; no results can be produced without the directory.

### Output directory layout
- Study directory: `{study_name}_{timestamp}/` (as specified in design doc)
- Per-experiment results: **flat files** in the study directory (no per-experiment subdirectories)
- Experiment result file naming: `{model}_{backend}_{precision}_{hash[:8]}.json` — human-scannable with short hash suffix for uniqueness
- Timeseries parquet files: same naming pattern with `.parquet` extension, alongside the JSON
- `manifest.json` sits at study directory root (no dot-prefix — visible in `ls`, easy to find)

### Manifest content and readability
- Pretty-printed JSON with `indent=2` (file size ~5-50KB for typical studies, indentation overhead negligible)
- `config_summary` auto-generated from sweep dimensions: `"{backend} / {key_sweep_params} / {precision}"` — highlights what varies across the study
- Include **top-level aggregate counters**: `total_experiments`, `completed`, `failed`, `pending` — enables quick progress check without parsing every entry
- Record `study_design_hash` only (not `study_yaml_hash`) — the design hash is the semantically meaningful identity; raw file hash dropped to reduce confusion
- Record `llenergymeasure_version` at manifest creation — enables version mismatch detection for future resume logic
- `schema_version: "2.0"` on the manifest model

### Error and edge cases
- `mark_failed()` captures `error_type` (exception class name) and `error_message` (exception str) — no full tracebacks (keeps manifest compact; matches or exceeds all peer tools)
- Directory creation failure: raise `StudyError` immediately (pre-flight failure)
- Manifest write failure: log warning, continue study (mid-study resilience)

### Claude's Discretion
- State transition validation (pending->running->completed|failed) — whether to enforce strict state machine or overwrite silently
- Internal `_find()` implementation details
- `_build_entries()` implementation for populating initial manifest from `StudyConfig`
- How `config_summary` selects which sweep params to display

</decisions>

<specifics>
## Specific Ideas

- Reuse `_atomic_write()` from `persistence.py` rather than reimplementing
- The `StudyManifest` and `ManifestWriter` schemas in `.product/designs/study-resume.md` are the upstream spec — implementation should follow that shape closely
- No peer tool has a study-level manifest — this is a novel contribution; closest analogue is MLflow's run tracking (database-backed, not JSON checkpoint)
- config_summary should surface the dimensions that *vary* in the sweep, not repeat fields that are constant across all experiments

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 10-manifest-writer*
*Context gathered: 2026-02-27*
