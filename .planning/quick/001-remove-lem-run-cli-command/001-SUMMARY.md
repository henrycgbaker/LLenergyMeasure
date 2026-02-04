---
type: quick
plan: 001
subsystem: cli
tags: [cli, cleanup, refactor]
requires: []
provides:
  - Single clear experiment entry point (lem experiment only)
affects: []
tech-stack:
  added: []
  patterns: []
key-files:
  created: []
  modified:
    - src/llenergymeasure/cli/__init__.py
    - src/llenergymeasure/cli/experiment.py
    - src/llenergymeasure/cli/listing.py
    - docs/cli.md
    - .planning/codebase/ARCHITECTURE.md
decisions: []
metrics:
  duration: 2 min
  completed: 2026-02-04
---

# Quick Task 001: Remove lem run CLI Command Summary

**One-liner:** Removed non-functional `lem run` command stub, leaving `lem experiment` as sole entry point.

## What Was Done

### Task 1: Remove run_cmd and all references

**Completed in:** 1 commit (316fc36)

1. **Removed command registration** from `src/llenergymeasure/cli/__init__.py`
   - Deleted line 100: `app.command("run")(experiment.run_cmd)`

2. **Deleted run_cmd function** from `src/llenergymeasure/cli/experiment.py`
   - Removed entire function (lines 155-236)
   - Function was non-functional stub that only printed "requires accelerate launch" message
   - Removed `"run_cmd"` from `__all__` export list

3. **Updated usage hints** in `src/llenergymeasure/cli/listing.py`
   - Changed `lem run config.yaml` to `lem experiment config.yaml` in dataset listing command

4. **Removed documentation** from `docs/cli.md`
   - Deleted entire `### run` section (lines 213-232)
   - Removed legacy command description and options table

5. **Updated architecture docs** in `.planning/codebase/ARCHITECTURE.md`
   - Removed `run_cmd` reference from Experiment Command entry point description
   - Changed "lem experiment or lem run" to "lem experiment"

## Results

- `lem run` command no longer exists - returns "No such command" error
- `lem experiment` remains the sole working entry point for experiments
- No stale references to `run_cmd` or `lem run` in codebase
- Cleaner CLI interface with one clear path for users

## Verification

All verification checks passed:
- ✓ `lem --help` shows no `run` command
- ✓ `lem experiment --help` still works
- ✓ No `run_cmd` references in src/llenergymeasure/cli/ (except batch_run_cmd which is unrelated)
- ✓ No `lem run` references in docs/ or src/ (except docker compose run which is unrelated)

## Deviations from Plan

None - plan executed exactly as written.

## Impact

**User-facing:**
- Users attempting `lem run` will get clear "No such command" error
- Documentation now shows only `lem experiment` for consistency

**Developer:**
- 112 lines of dead code removed
- Simpler CLI interface to maintain
- No more confusion between `run` and `experiment` commands

## Next Phase Readiness

This was a standalone cleanup task. No blockers or dependencies for future work.

## Commits

| Commit | Description | Files Changed |
|--------|-------------|---------------|
| 316fc36 | Remove deprecated lem run command | 5 files (-112 lines) |
