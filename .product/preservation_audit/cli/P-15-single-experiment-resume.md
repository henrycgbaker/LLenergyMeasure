# P-15: Single-Experiment Resume

**Module**: `src/llenergymeasure/cli/resume.py`, `src/llenergymeasure/state/experiment_state.py`
**Risk Level**: MEDIUM
**Decision**: Pending — the state machine is needed for v2.0 (it tracks experiment lifecycle); the `resume_cmd` CLI command is a different question
**Planning Gap**: `decisions/cli-ux.md` "What Was Cut" table lists `llem resume` as cut ("Becomes `llem campaign resume` subcommand at v2.2"). However, the underlying `ExperimentState` state machine and `StateManager` are not cut — they are required infrastructure. The current `resume.py` is campaign-oriented, not single-experiment-oriented, which creates a mismatch with the planning note.

---

## What Exists in the Code

**Primary file(s)**: `src/llenergymeasure/cli/resume.py`, `src/llenergymeasure/state/experiment_state.py`
**Key classes/functions from resume.py**:
- `resume_cmd()` (line 21) — CLI command (`lem resume`); scans `.state/` for `campaign_manifest.json` files, filters incomplete ones, presents an interactive `questionary` menu, then shows the resume command to the user (it does NOT re-run the campaign itself — it guides the user)
- Uses `CampaignManifest` and `ManifestManager` from `orchestration/manifest.py`

**Key classes/functions from experiment_state.py** (see also N-X10):
- `ExperimentStatus` enum (line 19): `INITIALISED`, `RUNNING`, `COMPLETED`, `AGGREGATED`, `FAILED`, `INTERRUPTED` — 6 states
- `ExperimentState` model (line 70): `config_hash: str | None`, `subprocess_pid: int | None`, `process_progress: dict[int, ProcessProgress]`, `is_subprocess_running()` method (line 164) using `os.kill(pid, 0)` for stale detection
- `StateManager` (line 267): `find_by_config_hash()` (line 393) — finds incomplete experiment by config hash; `cleanup_stale()` (line 407) — marks RUNNING states as INTERRUPTED if their subprocess PID is no longer alive
- `compute_config_hash()` (line 253): SHA-256 of config dict (excluding volatile fields), first 16 hex chars

The resume flow uses config hash matching: when `llem run` is run, it computes the hash of the current config, then calls `StateManager.find_by_config_hash()` to check if there is an incomplete experiment with the same config. If found and in INTERRUPTED status, it offers to resume rather than starting fresh.

## Why It Matters

The state machine (`ExperimentState`, `StateManager`, `compute_config_hash`) is not optional — it is the mechanism by which the orchestrator tracks whether an experiment is in-flight, completed, or failed. Even if the `llem resume` CLI command is cut in v2.0, the state machine must exist for the orchestrator to detect stale/interrupted experiments and avoid re-running completed work. The hash-based config matching is specifically valuable for study runs where dozens of experiments may be running sequentially — if the study runner crashes at experiment 7 of 20, the hash match allows it to skip 1–6 on restart.

## Planning Gap Details

- `decisions/cli-ux.md` (line: "`llem resume` → Becomes `llem campaign resume` subcommand at v2.2"): this refers to the campaign-level resume. The note does not address single-experiment resume or the `ExperimentState` state machine at all.
- `designs/architecture.md` (line 177–191): confirms "3-state target" (INITIALISING → MEASURING → DONE) vs current 6-state machine. This is a significant planned change — see N-X10 for full detail.
- Neither doc specifies what happens to `StateManager`, `compute_config_hash()`, or the stale-subprocess detection on the 3-state reduction.

The current `resume.py` is campaign-oriented (uses `CampaignManifest`, not `ExperimentState`). The single-experiment `--resume <exp_id>` pattern referenced in `lifecycle.py` (line 156: `lem experiment --resume {experiment_id}`) exists in the UI but the actual --resume flag implementation in the experiment command is not in scope here.

## Recommendation for Phase 5

**State machine** (`ExperimentState`, `StateManager`, `compute_config_hash`): carry forward but reduce to 3 states as planned in `designs/architecture.md`. The transition from 6 to 3 states is:
- `INITIALISED` + `WARMING_UP` → `INITIALISING`
- `MEASURING` → `MEASURING` (unchanged)
- `COMPLETED` + `AGGREGATING` + `SAVING` → `DONE`
- `FAILED` and `INTERRUPTED` → collapse into `DONE` (with `failed: bool` field)

Verify no external code depends on the current state value strings (they are stored in JSON state files — migration needed if strings change).

**`resume.py` CLI command**: this is a campaign-level helper and should be deferred to v2.2 alongside `llem study resume`. Remove from v2.0 CLI surface. Ensure `compute_config_hash()` and `StateManager.find_by_config_hash()` are retained as they are used by the orchestrator's own startup logic.
