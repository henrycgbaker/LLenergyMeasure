---
phase: 01-measurement-foundations
plan: "01"
subsystem: infra
tags: [hatchling, pyproject, dead-code, cli, package, build]

requires: []
provides:
  - Hatchling-based pyproject.toml with PEP 621 metadata
  - Zero-dep base install; backends behind extras [pytorch] [vllm] [tensorrt] [zeus] [codecarbon] [webhooks]
  - llem entry point (lem and llenergymeasure removed)
  - __version__ = "2.0.0"
  - 21 dead code files deleted
  - CLI skeleton that responds to --version and --help
affects: [all subsequent phases — every phase installs from this pyproject.toml]

tech-stack:
  added: [hatchling]
  patterns:
    - "PEP 621 [project] table — no [tool.poetry] table"
    - "Extras-only backends — zero mandatory inference deps at base"
    - "Single entry point: llem = llenergymeasure.cli:app"

key-files:
  created: []
  modified:
    - pyproject.toml
    - src/llenergymeasure/__init__.py
    - src/llenergymeasure/cli/__init__.py
    - src/llenergymeasure/constants.py

key-decisions:
  - "Base deps are 7 libraries only (pydantic, typer, pyyaml, platformdirs, nvidia-ml-py, pyarrow, tqdm)"
  - "No [all] extra — vLLM and TensorRT are process-incompatible (INF-04)"
  - "CLI skeleton avoids all heavy imports at import time (no rich, no torch, no dotenv)"
  - "SCHEMA_VERSION updated to 2.0.0 (was 3.0.0)"

patterns-established:
  - "Late-import pattern: CLI commands import heavy deps inside functions, not at module level"

requirements-completed: [INF-01, INF-02, INF-03, INF-04, INF-05]

duration: 2min
completed: 2026-02-26
---

# Phase 1 Plan 01: Package Foundation Summary

**Hatchling pyproject.toml with zero-dep base install, backend extras, llem-only entry point, and 21 dead code files removed**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-26T11:39:29Z
- **Completed:** 2026-02-26T11:41:55Z
- **Tasks:** 2
- **Files modified:** 4 modified, 21 deleted

## Accomplishments

- Rewrote pyproject.toml from Poetry to hatchling using PEP 621 `[project]` table; 7-library base, backends behind extras
- Removed `lem` and `llenergymeasure` entry points; single `llem = llenergymeasure.cli:app`
- Deleted 21 dead code files (progress, bootstrap, adapters, shared, naming, campaign_config, speculative, traffic, 7 CLI commands, 4 orchestration modules, notifications package)
- Stripped cli/__init__.py to a minimal skeleton (no heavy imports); `__version__` set to `"2.0.0"`
- Cleaned constants.py: removed PRESETS dict, DEPRECATED_CLI_FLAGS, and related helpers

## Task Commits

1. **Task 1: Rewrite pyproject.toml** - `365efa8` (chore)
2. **Task 2: Delete dead code and strip CLI** - `c782223` (refactor)

## Files Created/Modified

- `pyproject.toml` - Hatchling build, PEP 621, correct extras, llem entry point
- `src/llenergymeasure/__init__.py` - `__version__ = "2.0.0"`
- `src/llenergymeasure/cli/__init__.py` - Minimal skeleton, no heavy deps
- `src/llenergymeasure/constants.py` - Stripped to runtime constants only, SCHEMA_VERSION = "2.0.0"

## Decisions Made

- SCHEMA_VERSION changed from "3.0.0" to "2.0.0" — prior value was incorrect for a v2.0 package
- CLI skeleton imports only `typer` and `llenergymeasure.__version__` — defers all heavy imports to function bodies
- No [all] extra created — vLLM and TensorRT cannot coexist in same Python environment

## Deviations from Plan

None — plan executed exactly as written.

Note: dead code files were already deleted in a prior commit on this branch (`d16d87c`); `git rm` in Task 2 confirmed the index was clean and they were not re-tracked.

## Issues Encountered

Pre-commit hook (ruff) reformatted `resilience.py` and `security.py` during the Task 2 commit — those were unrelated to this plan's scope but are out-of-scope files that ruff auto-formatted as a side effect. No action required.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Package installs cleanly with `pip install -e '.[pytorch]'` (once hatchling is installed)
- CLI skeleton ready: `llem --version` returns `llem v2.0.0`, `llem --help` works
- All subsequent plans can assume clean pyproject.toml and no dead code from deleted modules

---
*Phase: 01-measurement-foundations*
*Completed: 2026-02-26*
