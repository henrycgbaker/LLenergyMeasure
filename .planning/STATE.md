# Project State


TODO: this is now degraded - it needs updating based on new product decisions and desings (see phase 4.5)

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-17)

**Core value:** Accurate, comprehensive measurement of the true cost of LLM inference — energy, compute, and quality tradeoffs — with research-grade rigour.
**Current focus:** Phase 5 (Clean Foundation — v2.0)

## Current Position

Phase: 5 of 10 (Clean Foundation — v2.0) — NOT STARTED
Plan: 0 of ?
Status: Ready to plan
Last activity: 2026-02-17 — Completed Phase 4.5 (Strategic Reset) — vision settled, GSD docs updated

Progress: [██████████] 100% Phase 1 (6/6) ✓
          [██████████] 100% Phase 2 (8/8) ✓
          [██████████] 100% Phase 2.1 (6/6) ✓
          [██████████] 100% Phase 2.2 (4/4) ✓
          [██████████] 100% Phase 2.3 (4/4) ✓
          [██████████] 100% Phase 2.4 (6/6) ✓
          [██████████] 100% Phase 3 (3/3) ✓
          [██████████] 100% Phase 4 (6/6) ✓
          [██████████] 100% Phase 4.5 (strategic reset) ✓
          [░░░░░░░░░░] 0% Phase 5 (0/?) — Clean Foundation (v2.0)
          [░░░░░░░░░░] 0% Phase 6 (0/?) — Measurement Depth (v2.1)
          [░░░░░░░░░░] 0% Phase 7 (0/?) — Scale (v2.2)
          [░░░░░░░░░░] 0% Phase 8 (0/?) — Parameter Completeness (v2.3)
          [░░░░░░░░░░] 0% Phase 9 (0/?) — Shareability (v2.4)
          [░░░░░░░░░░] 0% Phase 10 (0/?) — Quality + Efficiency (v3.0)

## Performance Metrics

**Velocity:**
- Total plans completed: 46
- Average duration: 7.0 min
- Total execution time: 320 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-measurement-foundations | 6/6 | 48 min | 7 min |
| 02-campaign-orchestrator | 8/8 | 31 min | 3.9 min |
| 02.1-zero-config-install | 6/6 | 27 min | 4.5 min |
| 02.2-campaign-execution-model | 4/4 | 18 min | 4.5 min |
| 02.3-campaign-state-resume | 4/4 | 26 min | 6.5 min |
| 02.4-cli-polish-testing | 6/6 | 46 min | 7.7 min |
| 03-gpu-routing-fix | 3/3 | 17 min | 5.7 min |
| 04-codebase-audit | 6/6 | 78 min | 13.0 min ✓

**Recent Trend:**
- 04-01: 7 min (automated analysis — vulture, deadcode, ruff, industry comparison)
- 04-02: 6 min (CLI surface + config system audit)
- 04-03: 6 min (core engine + backend completeness audit)
- 04-04: 5 min (orchestration, campaign, results, state audit)
- 04-05: 45 min (infrastructure, tests, planning cross-reference)
- 04-06: 8 min (report assembly + checkpoint)
- Trend: Phase 4 complete — audit plans averaged 13 min (longer due to read-heavy analysis)

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
- [03-01] NVIDIA_VISIBLE_DEVICES controls which GPUs are mounted by container runtime
- [03-01] CUDA_VISIBLE_DEVICES inside container uses remapped indices (0,1,2,...)
- [03-01] Container context detected via NVIDIA_VISIBLE_DEVICES presence (not empty and not "all")
- [03-02] Parallelism validation runs before backend-specific validation for fail-fast behaviour
- [03-02] Validation returns severity='error' warnings that block execution unless --force
- [03-02] Suggestion hints displayed below each config warning for immediate remediation guidance
- [03-03] config.gpus is declarative (specifies expected GPU count, not hardware probe)
- [03-03] Validation is arithmetic (checks tp_size <= len(config.gpus))
- [03-03] Container-side adapts to actual NVIDIA_VISIBLE_DEVICES from SLURM/runtime
- [04-04] Campaign system identified as over-engineered: 3150 lines for functionality industry doesn't provide
- [04-04] Campaign simplification approach: Extract grid to config command, minimal resume-only campaign (3150 → 500 lines, 84% reduction)
- [04-04] Dead code identified: resilience.py (97), progress.py (250), bootstrap.py (118) — 465 lines total with zero imports
- [04-04] State machine over-engineering: 6 states for fire-and-forget execution, 3 sufficient (422 → 150 lines, 64% reduction)
- [04-04] Late aggregation pattern in results pipeline justified: Avoids "average of averages" bias, essential for multi-process accuracy
- [04-05] Detection modules kept separate: docker_detection, backend_detection, env_setup are orthogonal concerns, no unification needed
- [04-05] Docker hybrid model maintained: Local-first with Docker fallback serves both casual and serious users, Docker-only increases barrier to entry
- [04-05] Planning alignment 100%: All 50 success criteria from Phases 1-3 implemented, no features lost in translation
- [04-05] Test quality strong: 873 tests with 96.3% having assertions, 32 exception tests without assertions are intentional
- [04.5] Library-first architecture — `import llenergymeasure` as primary interface; CLI is a thin wrapper. Industry norm (lm-eval, MLflow, CodeCarbon). Adopted at v2.0, not deferred.
- [04.5] llem rename — `lem` → `llem` at v2.0, clean break, no compatibility shim or alias
- [04.5] No default backend — `pip install llenergymeasure` installs no GPU backend; each is explicit (`[pytorch]`, `[vllm]`, `[tensorrt]`). Matches lm-eval pattern.
- [04.5] Zeus as v2.1 energy backend — EnergyBackend Protocol extended, `llenergymeasure[zeus]` extra, ~5% accuracy vs CodeCarbon ~10-15%
- [04.5] lm-eval integration as v3.0 — quality-alongside-efficiency, unique differentiator, deferred until solid CLI foundation
- [04.5] Campaign as separate module — not cut; 1000+ config parameter sweep runs are genuine use case; extracted from CLI monolith to importable `campaign` module at v2.2
- [04.5] Positioning: deployment efficiency (Option A primary) — "which deployment config?" not "which model?" — the gap nobody else fills
- [04.5] Library as secondary offering (Option B) — `import llenergymeasure` as composable tool for other benchmarking infrastructure
- [04.5] v4.0 web platform: static JSON MVP first, then dynamic API, then live features — three-stage rollout
- [04.5] Web platform: separate product, separate repo — shares library API but not server; different tech stack, different lifecycle
- [04.5] Versioning: v2.0 Clean Foundation → v2.1 Measurement Depth → v2.2 Scale → v2.3 Parameter Completeness → v2.4 Shareability → v3.0 Quality → v4.0 Web Platform

### Pending Todos

1. **Singularity/Apptainer HPC support** (infrastructure) — Container-runtime abstraction for HPC environments. Post-v2.0.
2. **Backend parameter completeness audit** (testing) — (1) verify all backend params are in config models, (2) verify all are wired through to backend API calls, (3) write robust integration tests per backend to validate wiring, (4) write full parameter reference doc per backend. Target: Phase 6.
3. **Revisit peak_memory_mb measurement semantics** (measurement) — Inference-only vs whole-process vs both. Decision deferred to implementation phase.
4. **Consider study-level cross-config aggregation** (measurement) — Cut from v2.0 (P-02). Revisit post-v2.0 whether built-in grouping/CIs justify scope expansion.

### Resolved Todos

- **[RESOLVED 2026-02-04] Context vs Plan Alignment Audit** — Audit completed via `/gsd:audit-milestone`. Result: 95% alignment. The `lem docker setup/build/status` commands were intentionally deferred (not a gap) — industry research showed ML tools use standard `docker compose` commands. CONTEXT.md files updated to document this decision. See `.planning/v2.0.0-MILESTONE-AUDIT.md`.

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
- [2026-02-04] Phases 3-5 added, existing phases renumbered:
  - Phase 3: GPU Routing Fix — Fix CUDA_VISIBLE_DEVICES propagation to containers, fail-fast config validation
  - Phase 4: Codebase Audit — Identify stubs, dead code, unimplemented features, verify plans vs implementation
  - Phase 5: Refactor & Simplify — Remove dead code, simplify workflows, rename lem→llem
  - Old Phase 3 (Parameter Completeness) → Phase 6
  - Old Phase 4 (Polish + UAT) → Phase 7
- [2026-02-17] Phase 4.5 Strategic Reset complete — full versioning roadmap and architecture confirmed. Major restructuring:
  - Roadmap replaces old v1.19–v1.22 / Phase 5–7 structure with v2.0–v4.0 milestone series
  - Phase 5 renamed "Clean Foundation" (was "Refactor & Simplify") — same scope, clearer framing
  - Phases 6–10 added (Measurement Depth, Scale, Parameter Completeness, Shareability, Quality)
  - v4.0 Web Platform confirmed as separate product, separate repo
  - Decision records created in .planning/decisions/ directory
  - Design specs created in .planning/design/ directory
  - See ROADMAP.md and PROJECT.md for full updated structure

### Blockers/Concerns

**Resolved:** GPU routing bug fully fixed. All three plans complete:
- 03-01: GPU env var propagation to Docker containers
- 03-02: Fail-fast parallelism validation
- 03-03: Unit tests (34 tests) + manual verification confirmed HPC compatibility

## Session Continuity

Last session: 2026-02-19
Stopped at: Session resumed, proceeding to plan Phase 5
Resume file: (none)
Next action: Plan Phase 5 (Clean Foundation) — `/gsd:plan-phase 5`
