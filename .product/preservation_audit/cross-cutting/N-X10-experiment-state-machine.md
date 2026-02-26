# N-X10: Experiment State Machine

**Module**: `src/llenergymeasure/state/experiment_state.py`
**Risk Level**: HIGH
**Decision**: Keep — v2.0 (with deliberate simplification from 6 states to 3 states per `designs/architecture.md`)
**Planning Gap**: `designs/architecture.md` specifies 3 target states but does not detail the migration path from 6 states, the fate of `StateManager` and `compute_config_hash()`, or how state files written in the 6-state schema are handled by the 3-state reader.

---

## What Exists in the Code

**Primary file(s)**: `src/llenergymeasure/state/experiment_state.py`
**Key classes/functions**:

- `ExperimentStatus` (line 19) — Enum with 6 values: `INITIALISED`, `RUNNING`, `COMPLETED`, `AGGREGATED`, `FAILED`, `INTERRUPTED`

- `EXPERIMENT_VALID_TRANSITIONS` (line 32) — transition table:
  ```
  INITIALISED → {RUNNING, FAILED, INTERRUPTED}
  RUNNING     → {COMPLETED, FAILED, INTERRUPTED}
  COMPLETED   → {AGGREGATED, FAILED}
  AGGREGATED  → {} (terminal)
  FAILED      → {RUNNING} (retry)
  INTERRUPTED → {RUNNING} (resume)
  ```

- `ProcessProgress` (line 59) — Pydantic model: `process_index: int`, `status: ProcessStatus`, `gpu_id: int | None`, `started_at/completed_at: datetime | None`, `error_message: str | None`

- `ExperimentState` (line 70) — Pydantic model with:
  - `experiment_id: str`, `status: ExperimentStatus`, `cycle_id: int`
  - `config_path: str | None`, `config_hash: str | None`, `model_name: str | None`, `prompt_args: dict`
  - `num_processes: int`, `subprocess_pid: int | None`, `process_progress: dict[int, ProcessProgress]`
  - `started_at: datetime | None`, `last_updated: datetime`
  - `error_message: str | None`
  - `completed_runs/failed_runs: dict[str, str]`, `total_runs: int` (legacy batch tracking fields)
  - `transition_to(new_status, error_message, validate)` (line 206) — enforces `EXPERIMENT_VALID_TRANSITIONS`; raises `InvalidStateTransitionError` on invalid transition; timestamps `last_updated`
  - `can_transition_to(new_status)` (line 241) — boolean check without side effects
  - `is_subprocess_running()` (line 164) — `os.kill(self.subprocess_pid, 0)` — signal 0 checks process existence without sending signal; returns `False` on `ProcessLookupError` or `PermissionError`
  - `can_aggregate()` (line 155) — checks all `ProcessProgress` entries are `COMPLETED`

- `compute_config_hash(config_dict)` (line 253) — SHA-256 of `json.dumps(stable, sort_keys=True, default=str)` where stable excludes `experiment_id` and `_metadata`; returns first 16 hex chars

- `StateManager` (line 267) — manages JSON state files in `DEFAULT_STATE_DIR` (`.state/`):
  - `save(state)` (line 306) — atomic write via temp-file-then-rename
  - `load(experiment_id)` (line 284) — reads and validates JSON; uses `is_safe_path` before reading
  - `find_by_config_hash(config_hash)` (line 393) — scans all incomplete experiments for matching hash
  - `find_incomplete()` (line 376) — all states where `status != AGGREGATED`
  - `cleanup_stale()` (line 407) — transitions RUNNING states to INTERRUPTED if `is_subprocess_running()` returns False

## Why It Matters

The state machine is the experiment resumability mechanism. Without it, a crashed study runner has no way to determine which experiments completed successfully and which need to be re-run. The `compute_config_hash()` function is the key: it enables config-based matching so a re-run of `llem study study.yaml` can skip experiments that already have a state file in `COMPLETED` or `AGGREGATED` status. The atomic write pattern (`save()`) prevents corrupt state files if the process crashes during a write. `cleanup_stale()` handles the common case where the orchestrator process was killed (rather than gracefully interrupted) — it detects orphaned RUNNING states using PID existence checks.

## Planning Gap Details

`designs/architecture.md` (lines 177–191):
```
Current codebase has 6 states. v2.0 target: 3 states.
class ExperimentState(str, Enum):
    INITIALISING = "initialising"
    MEASURING    = "measuring"
    DONE         = "done"
```

The document also includes a `<!-- TODO: Review current 6-state machine to confirm the 3-state target is sufficient -->` comment, explicitly acknowledging this is unresolved.

What is missing:
1. How `FAILED` and `INTERRUPTED` map to `DONE` (are they both just `DONE` + a separate `failed: bool` field?)
2. Whether state files written by v1.x (6-state) can be read by v2.0 (3-state) — a migration concern
3. Whether `StateManager`, `compute_config_hash()`, and `find_by_config_hash()` are retained in the 3-state model
4. Whether `subprocess_pid`-based stale detection is still relevant when the execution model changes to `multiprocessing.Process` (PIDs still exist — so yes, but verify)
5. Whether `process_progress: dict[int, ProcessProgress]` is still needed in a single-process-per-experiment model

## Recommendation for Phase 5

The 3-state reduction is correct for v2.0 (INITIALISING → MEASURING → DONE). Implementation:

```python
class ExperimentState(str, Enum):
    INITIALISING = "initialising"
    MEASURING    = "measuring"
    DONE         = "done"
```

Retain as a separate field on `ExperimentState`:
```python
failed: bool = False
error_message: str | None = None
```

This avoids needing a separate `FAILED` state while preserving failure information. `DONE + failed=True` is equivalent to the current `FAILED` or `INTERRUPTED`. Resume logic checks `DONE + failed=True` and offers retry.

Retain `StateManager`, `compute_config_hash()`, `find_by_config_hash()`, `cleanup_stale()`, and atomic save — these are load-bearing infrastructure. Remove `completed_runs`/`failed_runs`/`total_runs` legacy fields (they were for batch/campaign tracking, not single experiments).

Add a note to `designs/architecture.md` resolving the TODO comment about state reduction.
