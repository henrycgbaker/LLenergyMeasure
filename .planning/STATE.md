# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-29)

**Core value:** Accurate, comprehensive measurement of the true cost of LLM inference — energy, compute, and quality tradeoffs — with research-grade rigour.
**Current focus:** Phase 3 planning (Parameter Completeness)

## Current Position

Phase: 2.4 of 7 (CLI Polish & Testing)
Plan: 5 of 6 (remaining: 04)
Status: In progress
Last activity: 2026-02-04 — Completed 02.4-06-PLAN.md (Docker Lifecycle Output)

Progress: [██████████] 100% Phase 1 (6/6)
          [██████████] 100% Phase 2 (8/8)
          [██████████] 100% Phase 2.1 (6/6)
          [██████████] 100% Phase 2.2 (4/4) ✓
          [██████████] 100% Phase 2.3 (4/4) ✓
          [████████░░] 83% Phase 2.4 (5/6)

## Performance Metrics

**Velocity:**
- Total plans completed: 29
- Average duration: 5.2 min
- Total execution time: 150 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-measurement-foundations | 6/6 | 48 min | 7 min |
| 02-campaign-orchestrator | 8/8 | 31 min | 3.9 min |
| 02.1-zero-config-install | 6/6 | 27 min | 4.5 min |
| 02.2-campaign-execution-model | 4/4 | 18 min | 4.5 min |
| 02.3-campaign-state-resume | 4/4 | 26 min | 6.5 min |

**Recent Trend:**
- 01-01: 7 min (2 tasks, domain models + config extensions)
- 01-03: 4 min (1 task, warmup convergence module)
- 01-04: 4 min (2 tasks, CSV export + timeseries)
- 01-02: 11 min (2 tasks, NVML measurement primitives)
- 01-05: 7 min (2 tasks, orchestrator integration)
- 01-06: 15 min (2 tasks, unit tests + UAT checkpoint)
- 02-01: 4 min (2 tasks, CampaignConfig models + dependency)
- 02-02: 6 min (2 tasks, ContainerManager Docker lifecycle)
- 02-04: 2 min (1 task, campaign manifest persistence)
- 02-05: 3 min (1 task, grid expansion and validation)
- 02-07: 2 min (1 task, SSOT campaign config introspection)
- 02-06: 7 min (4 tasks, campaign orchestrator integration)
- 02-08: 5 min (2 tasks, unit tests + UAT checkpoint)
- 02.1-01: 1 min (3 tasks, detection modules)
- 02.1-02: 4 min (2 tasks, doctor diagnostic command)
- 02.1-06: 2 min (2 tasks, campaign refactor)
- 02.1-04: 4 min (2 tasks, documentation refresh)
- 02.1-03: 12 min (3 tasks, packaging update)
- 02.2-01: 3 min (3 tasks, campaign context propagation)
- 02.2-02: 5 min (2 tasks, user config + container routing)
- 02.2-03: 6 min (2 tasks, ContainerManager + dual container strategy)
- 02.2-04: 4 min (4 tasks, unit tests + manual verification)
- 02.3-01: 7 min (2 tasks, webhook notifications + resume command)
- 02.3-02: 7 min (2 tasks, lem init wizard)
- 02.3-03: 7 min (3 tasks, unit tests for Phase 2.3)
- 02.3-04: 5 min (1 task, manual verification checkpoint)
- 02.4-02: 2 min (2 tasks, schema version + example configs)
- 02.4-01: 5 min (2 tasks, config list + campaign grouping)
- 02.4-05: 4 min (3 tasks, backend noise filtering + JSON output)
- 02.4-03: 5 min (2 tasks, smoke test warning capture + issues.yaml)
- 02.4-06: 5 min (4 tasks, Docker lifecycle output)
- Trend: Consistent ~4-7 min; Phase 2.4 5/6 complete (remaining: 04)

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Split v2 into 4 sub-releases (v1.19-v1.22): Each independently shippable, reduces risk, enables early UAT
- Measurement foundations before parameter work: Fixes systematic 15-30% energy bias; accurate data before more features
- Campaign orchestrator as own release: Highest-risk item (Docker exec model), shouldn't block measurement improvements
- Long-running containers with `docker compose exec`: Avoids per-experiment container startup overhead; containers stay warm, shared volumes for results
- Warmup convergence (CV-based) over fixed count: Scientifically more robust; existing CycleStatistics already tracks CV
- [01-01] Removed `from __future__ import annotations` from domain models: Pydantic v2 incompatibility with deferred annotations in nested models
- [01-01] Runtime imports for new domain types in experiment.py: Pydantic needs resolved types at class definition time
- [01-04] Renamed CSV column `total_energy_j` to `energy_raw_j`: Grouped-prefix convention for CSV readability; Pydantic model field unchanged
- [01-02] Used `typing.Any` for pynvml module/handle params: mypy cannot type-check dynamically imported modules; `from __future__ import annotations` causes ruff to remove type imports as unused
- [01-06 UAT] Warmup skips gracefully for backend-managed models (BackendModelLoaderAdapter returns None)
- [01-06 UAT] Energy breakdown uses experiment timestamps when CodeCarbon reports duration_sec=0.0
- [02-02] Lazy import with _is_docker_exception() type-check pattern: avoids mypy errors with dynamic exception catching while keeping python-on-whales optional
- [02-02] ContainerManager not added to orchestration __init__.py: optional dependency would break imports for non-campaign users
- [02-05] Used Sequence[str | None] for models axis to satisfy mypy list invariance
- [02-06] Synthetic `<grid:config_name>` config_path distinguishes grid-based from file-based experiments
- [02-06] Daemon mode runs in foreground process (users use nohup/screen for background)
- [02-06] ContainerManager lazy fallback to docker compose run --rm if python-on-whales not installed
- [02-06] Extracted _run_campaign_loop for daemon reuse of full execution flow
- [02-08] Docker dispatch UAT deferred to Phase 2.1 (_should_use_docker() bug + missing .env)
- [02-08] Added file existence check before campaign YAML detection (fixes misleading error)
- [02-08] Added execution mode display (Docker exec / Docker run / Local) to campaign CLI
- [02.1-01] Lazy import of backend_detection in docker_detection to avoid circular dependencies
- [02.1-01] Dual-method Docker detection (/.dockerenv + /proc/1/cgroup) for robustness
- [02.1-01] .env generation is idempotent (never overwrites existing file)
- [02.1-02] All diagnostic checks wrapped in try/except to ensure lem doctor never crashes
- [02.1-02] Lazy imports of detection modules inside functions to avoid forcing dependencies at module load
- [02.1-02] Rich console for coloured output (green ✓, red ✗) with NO_COLOR support
- [02.1-02] Fixed backend_detection.py to catch OSError, not just ImportError (TensorRT import errors)
- [02.1-06] Delegate _should_use_docker() to should_use_docker_for_campaign() for consistent detection logic
- [02.1-06] Pass backend list to all _should_use_docker() call sites for proper dispatch decisions
- [02.1-06] Call ensure_env_file() before ContainerManager creation in both campaign entry points
- [02.1-03] PyTorch backend moved to core dependencies (no [pytorch] extra needed)
- [02.1-03] ONNX Runtime as separate [onnxrt] extra (PyTorch-specific optimization, not core requirement)
- [02.1-03] setup.sh deleted, Makefile provides setup/docker-setup/dev targets
- [02.1-03] Package validated as PyPI-publishable (poetry build, wheel install in isolated venv)
- [02.1-04] Documentation shows two-path install: local-first (pip install -e .), Docker for multi-backend campaigns
- [02.1-04] All docs reference lem doctor for diagnostics, removed all setup.sh references
- [02.1-05] Rich markup escape for install hints (brackets interpreted as tags)
- [02.1-05] Docker GPU check uses `docker info` runtime detection, not image pull (avoids timeout)
- [02.1-05] Two-tier backend model: PyTorch=core (pip), vLLM/TensorRT=Docker recommended
- [post-2.1] Container strategy changed from `docker compose up + exec` (ContainerManager) to `docker compose run --rm` per-experiment. Research confirmed ephemeral containers are correct for this workload — no model persistence needed between experiments, simpler lifecycle, no health-check complexity. ContainerManager deleted as dead code. See `.planning/debug/container-strategy-research.md`.
- [02.2-01] Campaign context via environment variables (LEM_CAMPAIGN_ID, LEM_CAMPAIGN_NAME, LEM_CYCLE, LEM_TOTAL_CYCLES)
- [02.2-01] Removed --cycles from experiment CLI (experiments are atomic, campaigns own repetition)
- [02.2-01] Experiment CLI shows "Part of campaign X (cycle Y/Z)" when LEM_CAMPAIGN_ID is set
- [02.2-02] User config file is .lem-config.yaml with graceful degradation (missing file = defaults)
- [02.2-02] CLI flags always override user config defaults (standard precedence)
- [02.2-02] Lazy import of load_user_config inside campaign_cmd() to avoid forcing dependency at module load
- [02.2-03] ContainerManager re-added for persistent container strategy (optional, not default)
- [02.2-03] --container-strategy flag: ephemeral (default, run --rm) or persistent (up + exec)
- [02.2-03] Strategy precedence: CLI > user config (.lem-config.yaml) > default
- [02.2-03] Persistent mode shows warning and requires confirmation (unless --yes)
- [02.3-01] Removed default_backend from UserConfig (backend must be explicit in config)
- [02.3-01] Fail-fast validation on invalid .lem-config.yaml (raises ValueError vs silent defaults)
- [02.3-01] Webhooks lazily imported at call sites in campaign.py
- [02.3-01] Resume command guides to lem campaign --resume rather than direct resumption
- [02.3-02] Environment detection reuses doctor.py patterns for consistency
- [02.3-02] lem init runs doctor automatically after config creation for immediate feedback
- [Quick-003] Webhook toggles conditional in init wizard (only shown when webhook_url provided)
- [Quick-003] Advanced Docker options (warmup_delay, auto_teardown) remain config-file-only
- [02.4-01] Campaign YAML files filtered from `lem config list` (identified by `campaign_name` key)
- [02.4-01] Campaign grouping via `--group-by` supports dot-notation for nested fields (e.g., `pytorch.batch_size`)
- [02.4-05] Backend filtering uses stdlib logging.getLogger().setLevel() to suppress noisy libraries
- [02.4-05] Log files written to results/<exp_id>/logs/ with DEBUG level for full capture
- [02.4-05] JSON output mode via --json global flag stored in LLM_ENERGY_JSON_OUTPUT env var
- [02.4-03] Capture-first philosophy: capture ALL warnings/errors, filter known issues via issues.yaml
- [02.4-03] Issue sources auto-detected from message content (vllm, torch, transformers, etc.)
- [02.4-03] Status values for known issues: wip, wontfix, external
- [02.4-06] _check_docker_images() returns tuple (existing, missing) for better status display
- [02.4-06] ContainerManager uses StatusCallback type alias for status_callback parameter
- [02.4-06] _handle_missing_images() offers interactive build prompt with progress

### Pending Todos

- **[CRITICAL] Context vs Plan Alignment Audit** — CONTEXT.md files for Phases 01, 02, 02.2, 02.3 were misplaced (in `.planning/phases/` instead of phase subdirectories). Plans may not reflect user decisions. Audit ALL completed phases for discrepancies between CONTEXT.md decisions and implemented plans. If gaps found, will require re-planning and re-implementation. Phases to audit: 01-measurement-foundations, 02-campaign-orchestrator, 02.2-campaign-execution-model, 02.3-campaign-state-resume.

**Completed todos (moved to Phase 2.4 plans):**
- ~~[Phase 2.1-06] Docker lifecycle CLI output~~ → Phase 2.4 plans
- ~~[cli] Review CLI args vs options pattern~~ → Audited, found correct ✓
- ~~[cli] Improve CLI output readability for vLLM/TensorRT~~ → Phase 2.4 plans (backend noise filtering)
- ~~[cli] Add `lem config list` command~~ → Phase 2.4 Plan 01
- ~~[docs] Update example configs~~ → Phase 2.4 Plan 02
### Quick Tasks Completed

- [Quick 001] Removed deprecated `lem run` command (2026-02-04) - Non-functional stub deleted, `lem experiment` is sole entry point
- [Quick 002] Documented container strategies in deployment guide (2026-02-04) - Added Container Strategies section explaining ephemeral vs persistent modes with configuration examples
- [Quick 003] Fixed lem init wizard (2026-02-04) - Added prompts for between_cycles thermal gap and webhook notification toggles (on_complete/on_failure)

### Roadmap Evolution

- Phase 2.1 inserted after Phase 2: Zero-Config Install Experience (URGENT) — pip install must work out-of-box with auto Docker detection, auto .env generation, PyPI-ready packaging. Discovered during Phase 2 UAT: _should_use_docker() misidentified conda installs, .env required manual setup.sh.
- Phase 2.1 REVISED after ecosystem research: Dropped `lem docker setup/build` wizard (no ML tool does this), merged `lem docker status` + `lem backend list` into `lem doctor`, simplified from 3 install paths to 2. Docker setup uses standard `docker compose` commands.
- [2026-02-03] Phase 2.2 + 2.3 inserted after Phase 2.1: Architecture fixes and campaign resume discovered during UAT. Phase 2.2 = Campaign Execution Model (container routing, cycle context, dual strategy). Phase 2.3 = Campaign State & Resume (state persistence, `lem resume`, user preferences). See `.planning/ARCHITECTURE-DISCUSSION.md` for full decisions.

### Blockers/Concerns

None. Phase 1 validated end-to-end on A100 GPU. Phase 2 local UAT passed.

## Session Continuity

Last session: 2026-02-04
Stopped at: Completed 02.4-06-PLAN.md (Docker Lifecycle Output)
Resume file: None
Next action: Continue Phase 2.4 (02.4-04-PLAN.md remaining)
