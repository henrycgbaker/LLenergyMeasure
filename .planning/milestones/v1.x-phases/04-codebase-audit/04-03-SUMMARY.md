---
phase: 04-codebase-audit
plan: 03
subsystem: core
tags: [inference, backends, pytorch, vllm, tensorrt, energy, measurement, codecarbon]

# Dependency graph
requires:
  - phase: 03-gpu-routing-fix
    provides: GPU environment propagation and validation
provides:
  - Complete core engine audit identifying functional gaps, dead code, and backend-native divergence
  - Quantified abstraction costs (10x LOC vs lm-eval-harness)
  - Critical bug identification (PyTorch model_kwargs passthrough)
  - Measurement chain validation (config → energy → results)
affects: [05-refactor-simplify, Phase 6 (parameter completeness)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Backend protocol pattern (InferenceBackend) for pluggable inference engines"
    - "Lazy backend loading via registry (prevents import errors for optional deps)"
    - "Fallback chain pattern (FLOPs estimation, energy backends)"

key-files:
  created:
    - .planning/phases/04-codebase-audit/04-03-REPORT-core-engine.md
  modified: []

key-decisions:
  - "Identified PyTorch model_kwargs bug (L375) - breaks attn_implementation, low_cpu_mem_usage"
  - "vLLM missing native stream=True - using estimation instead of true per-token capture"
  - "TensorRT backend unverified - no evidence of successful runs"
  - "adapters.py (209 lines) dead code - zero imports"
  - "Backend abstraction costs 10x LOC vs lm-eval-harness patterns"
  - "CodeCarbon is only energy backend - 8-line base.py abstraction questionable"

patterns-established:
  - "Audit pattern: Backend-native comparison (upstream does X, we do Y)"
  - "Audit pattern: Completeness matrix (functional/stub/broken)"
  - "Audit pattern: Import chain validation (detect orphaned modules)"

# Metrics
duration: 6min
completed: 2026-02-05
---

# Phase 04 Plan 03: Core Engine Audit Summary

**Comprehensive audit of inference backends, measurement primitives, and core utilities identified 1 critical bug, 1 major gap, 359 lines dead code, and 10x abstraction cost vs industry patterns**

## Performance

- **Duration:** 6 min
- **Started:** 2026-02-05T01:05:40Z
- **Completed:** 2026-02-05T01:11:54Z
- **Tasks:** 2
- **Files modified:** 1 (report created)

## Accomplishments

- **Audited all three inference backends** (PyTorch 1155L, vLLM 1006L, TensorRT 1171L) for functional completeness, backend-native alignment, and Docker execution paths
- **Identified critical PyTorch bug** (L375) - model_kwargs built but never passed to loader, breaking attn_implementation, low_cpu_mem_usage, max_memory configs
- **Documented vLLM streaming gap** - not using native `stream=True`, ITL measurements are estimates vs true per-token capture
- **Quantified abstraction costs** - backends are 10x LOC vs lm-eval-harness (450 lines abstraction overhead per backend)
- **Found 359 lines dead code** - adapters.py (209L, zero imports), unused shared.py utilities (150L)
- **Validated measurement chain** - traced energy/power/thermal/warmup/extended metrics from config → orchestrator → results output
- **Assessed all core utilities** - 2700 lines mostly necessary, no orphaned modules found

## Task Commits

Each task was committed atomically:

1. **Task 1: Inference backend audit** - `55e8801` (audit)
   - Per-backend completeness tables (PyTorch functional, vLLM near-complete, TensorRT unverified)
   - Backend-native comparison (HuggingFace/vLLM/TensorRT-LLM upstream patterns vs our wrappers)
   - Docker execution path analysis (all 3 backends broken - Phase 4 constraint)
   - Dead code identification (adapters.py, unused shared.py, deprecated BetterTransformer)
   - Code volume: 3480 lines across backends

2. **Task 2: Measurement and core utility audit** - `487ac3d` (audit)
   - Energy backend assessment (CodeCarbon only, 8-line abstraction questioned)
   - Power/thermal sampling traced (100ms polling, pynvml thread safety OK)
   - Warmup system verified (CV-based convergence, orchestrator integration)
   - Extended metrics validated (null handling correct, stable schema)
   - Core utilities necessity matrix (all used, gpu_info.py 482 lines large)
   - Measurement chain: config → CodeCarbon → pynvml → EnergyMetrics → results ✓

**Plan metadata:** _(no metadata commit for audit reports)_

## Files Created/Modified

- `.planning/phases/04-codebase-audit/04-03-REPORT-core-engine.md` (1157 lines) - Full audit report with backend completeness tables, backend-native comparison, measurement system assessment, core utility necessity matrix

## Decisions Made

**Backend Assessment:**
- PyTorch backend most complete (1155 lines) - full streaming, batch inference, warmup
- vLLM backend near-complete (1006 lines) - missing native streaming (`stream=True` not used)
- TensorRT backend unverified (1171 lines) - logic present but no evidence of successful runs

**Critical Findings:**
- PyTorch Bug L375: `_build_model_kwargs()` returns dict but never passed to loader - breaks attn_implementation, low_cpu_mem_usage, max_memory
- vLLM streaming gap: Using proportional estimation instead of native `llm.generate(..., stream=True)` - ITL values estimated, not measured per-token
- Docker failures: All 3 backends broken (PyTorch hangs, vLLM workers crash, TensorRT wrong container route)

**Dead Code:**
- `adapters.py` (209 lines) - zero imports anywhere, orphaned
- `shared.py` utilities (150 lines) - 5/7 functions never imported by backends
- PyTorch `_apply_bettertransformer()` (20 lines) - deprecated, warning at L1105

**Industry Comparison:**
- Our backends: 10x LOC vs lm-eval-harness (450 lines abstraction overhead per backend)
- Our GPU info: 48x LOC vs MLPerf nvidia-smi parsing (482 lines vs 10 lines)
- Our warmup: More rigorous (CV-based vs fixed iterations)
- Protocol abstraction heavier than needed (vLLM-style direct engine usage would be simpler)

**Measurement System:**
- CodeCarbon only energy backend → 8-line base.py abstraction questionable
- Power/thermal sampling functional (100ms polling, pynvml thread safety OK)
- Warmup system verified (CV-based convergence, orchestrator integration)
- Extended metrics null handling correct (stable schema, graceful degradation)
- All measurement primitives traced to results output

**Core Utilities:**
- Total 2700 lines, mostly necessary (no orphaned modules)
- gpu_info.py largest (482 lines) - justify vs nvidia-smi CLI parsing
- FLOPs integration needs verification (unclear if used in orchestrator)
- inference.py minimal (36 lines) - verify usage, consider inlining

## Deviations from Plan

None - audit report created exactly as specified.

## Issues Encountered

None - plan execution straightforward.

## Next Phase Readiness

**Ready for Phase 5 (Refactor & Simplify):**
- Dead code identified: 359 lines ready for deletion
- Critical bug documented: PyTorch L375 fix required
- Simplification opportunities quantified: 450 lines abstraction per backend
- Docker execution paths documented (failures known)

**Blockers/Concerns:**
- TensorRT backend unverified - recommend end-to-end test before Phase 6
- vLLM streaming gap impacts research accuracy (ITL estimation vs measurement)
- Docker failures (all 3 backends) - root cause needs investigation before UAT

**Recommendations for Phase 5:**
1. Fix PyTorch model_kwargs bug (critical)
2. Add vLLM native streaming (research accuracy)
3. Delete dead code (adapters.py, unused shared.py utilities)
4. Add Docker pre-flight checks (CUDA availability, shm-size)
5. Consider simplifying backend abstractions (10x overhead vs industry)
6. Verify FLOPs integration in orchestrator
7. Run TensorRT end-to-end test

---
*Phase: 04-codebase-audit*
*Completed: 2026-02-05*
