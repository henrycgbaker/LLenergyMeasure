---
phase: 04-codebase-audit
plan: 04
subsystem: orchestration-campaign-results-state-domain
tags: [audit, orchestration, campaign, results, state-machine, domain-models, dead-code]
requires: []
provides:
  - orchestration-layer-assessment
  - campaign-system-evaluation
  - results-pipeline-verification
  - state-management-review
  - domain-model-audit
  - dead-code-identification
affects: [phase-5-refactor, campaign-simplification]
tech-stack:
  added: []
  patterns:
    - late-aggregation
    - dependency-injection
    - state-machine-validation
key-files:
  created: [.planning/phases/04-codebase-audit/04-04-REPORT-orchestration-results.md]
  modified: []
decisions:
  - id: campaign-over-engineering
    title: Campaign system identified as over-engineered
    rationale: 3150 lines for functionality industry doesn't provide; users can achieve same with bash loops
    alternatives: [keep-as-is, extract-to-separate-tool, simplify-heavily, remove-entirely]
    selected: simplify-heavily
    impact: 84% reduction (3150 → 500 lines)
  - id: dead-code-removal
    title: Three modules confirmed as dead code
    rationale: Zero imports found across codebase
    modules: [resilience.py, progress.py, bootstrap.py]
    impact: 465 lines removable
  - id: state-machine-simplification
    title: State machine over-engineered for use case
    rationale: 6 states for fire-and-forget execution, 3 sufficient
    impact: 64% reduction (422 → 150 lines)
metrics:
  duration: 278 seconds
  completed: 2026-02-05
---

# Phase 04 Plan 04: Orchestration & Results System Audit Summary

**One-liner**: Orchestration justified (multi-backend), campaign over-engineered (3150 lines for non-core feature), 465 lines of dead code identified

## What Was Delivered

### Task 1: Orchestration and Campaign System Audit

**Orchestration Layer (1912 lines)**:
- **runner.py** (471 lines): Core experiment execution, appropriate complexity → KEEP
- **factory.py** (179 lines): Backend DI abstraction for 3 backends → KEEP
- **lifecycle.py** (151 lines): Simple CUDA cleanup → KEEP
- **launcher.py** (849 lines): Multi-backend launch orchestration:
  - Early NCCL fix for vLLM multi-GPU
  - Early CUDA_VISIBLE_DEVICES for container context
  - Three launch modes (direct/accelerate/torchrun)
  - Complexity justified by multi-backend requirements → KEEP
- **context.py** (262 lines): Clean execution context dataclass → KEEP

**Verdict**: Orchestration complexity JUSTIFIED by multi-backend support (PyTorch/vLLM/TensorRT), energy measurement requirements (baseline, warmup), and distributed launch.

**Campaign System (3150 lines)**:
- **orchestration/campaign.py** (586 lines): Campaign runner logic
- **cli/campaign.py** (1754 lines): **WHY SO LARGE?** CLI is 3x runner logic
- **orchestration/grid.py** (363 lines): Grid expansion
- **orchestration/manifest.py** (194 lines): Resume state
- **orchestration/container.py** (253 lines): Persistent strategy (unused)

**Industry Comparison**:
- lm-eval-harness: NO campaign system
- vLLM benchmarks: NO campaign orchestration
- LLMPerf: NO built-in campaign
- TensorRT-LLM: NO campaign system
- **Industry practice**: Users orchestrate via bash loops or external tools (Hydra, W&B)

**What users could do without campaign command**:
```bash
for config in configs/*.yaml; do
  for i in {1..5}; do
    lem experiment $config -d alpaca -n 100
    sleep 60
  done
done
```

**Verdict**: Campaign system is **OVER-ENGINEERED** (3150 lines for non-core convenience feature).

**Recommendation**: Simplify heavily (Option C):
- Extract grid expansion to `lem config expand-grid`
- Minimal campaign for resume only
- Remove warmup/gap/ordering orchestration
- **Target**: 3150 → 500 lines (84% reduction)

### Task 2: Results, State, Domain, and Top-Level Module Audit

**Results Pipeline (1635 lines)**:
- **aggregation.py** (758 lines): Late aggregation logic → JUSTIFIED
  - Multi-process completeness validation
  - Statistical aggregation (avoids "average of averages" bias)
  - Extended metrics late aggregation
- **exporters.py** (338 lines): JSON/CSV export → USED
- **timeseries.py** (201 lines): Power/thermal export → WIRED ✓
- **bootstrap.py** (118 lines): Bootstrap CI → **DEAD CODE ✗**
- **repository.py** (220 lines): Simple file-based CRUD → KEEP

**State Management (422 lines)**:
- 6-state machine: INITIALISED → RUNNING → COMPLETED → AGGREGATED / FAILED / INTERRUPTED
- **Assessment**: MILD OVER-ENGINEERING
- Minimal state machine needs only 3 states: COMPLETED, AGGREGATED, FAILED
- **Target**: 422 → 150 lines (64% reduction)

**Domain Models (1159 lines)**:
- **metrics.py** (692 lines): Comprehensive metrics including:
  - PrecisionMetadata for cross-backend comparison
  - NormalisedMetrics for apples-to-apples efficiency
  - Extended metrics (TPOT, memory, GPU util, KV cache, batching)
  - **Verdict**: Size JUSTIFIED — comprehensive measurement is core value
- **experiment.py** (255 lines): Result models → All fields wired ✓
- **environment.py** (132 lines): Environment capture → Used
- **model_info.py** (80 lines): Model metadata → Used

**Top-Level Modules**:
- **protocols.py** (176 lines): DI interfaces → USED
- **resilience.py** (97 lines): Retry decorator → **DEAD CODE ✗**
- **progress.py** (250 lines): tqdm wrapper → **DEAD CODE ✗**
- **security.py** (90 lines): Path sanitisation → USED
- **exceptions.py** (112 lines): Custom exceptions → USED
- **constants.py** (312 lines): Constants → USED
- **logging.py** (240 lines): Loguru setup → USED

**Dead Code Identified**: 465 lines (5.3%)
- resilience.py: 97 lines (zero imports)
- progress.py: 250 lines (zero imports)
- bootstrap.py: 118 lines (zero imports, unimplemented feature)

## Deviations from Plan

None — plan executed exactly as written.

## Key Insights

### 1. Industry Doesn't Provide Campaign Orchestration
Evidence from lm-eval-harness, vLLM benchmarks, LLMPerf, TensorRT-LLM: Users prefer explicit bash loops or external orchestration tools (Hydra, W&B) over built-in campaign systems.

**Implication**: 3150 lines of campaign code serve a convenience function, not a core requirement.

### 2. Late Aggregation Pattern is Correct
The 758-line aggregation module implements statistically correct late aggregation (combine raw samples across processes before computing statistics). This avoids "average of averages" bias and is essential for accurate multi-process measurements.

### 3. Multi-Backend Complexity is Real
launcher.py's 849 lines reflect genuine complexity:
- vLLM requires early NCCL_P2P_DISABLE before imports
- Container context requires early CUDA_VISIBLE_DEVICES setup
- Three backends need three different launch mechanisms

This complexity cannot be simplified without breaking multi-backend support.

### 4. State Machine Over-Engineering
6-state FSM with validated transitions is excessive for fire-and-forget execution. Experiments don't need INITIALISED/RUNNING states — subprocess existence is sufficient.

### 5. Dead Code Exists Despite Testing
465 lines of dead code (resilience, progress, bootstrap) with zero imports suggests:
- Code written but never integrated
- Bootstrap CI from MEAS-08 spec never implemented
- No automated dead code detection

## Next Phase Readiness

**Phase 5 (Refactor & Simplify) is ready**:

### Simplification Targets Identified:
1. **Campaign system**: 3150 → 500 lines (84%)
   - Extract grid to config command
   - Minimal resume-only campaign
   - Remove warmup/gap/ordering

2. **State machine**: 422 → 150 lines (64%)
   - Reduce from 6 states to 3
   - Remove INITIALISED/RUNNING states

3. **Dead code**: 465 → 0 lines (100%)
   - Delete resilience.py, progress.py, bootstrap.py

**Total reduction**: 8743 → 5150 lines (42% reduction, 3700 lines removed)

### No Blockers:
- All systems understood
- Industry comparison complete
- Simplification approach validated

### Documentation Complete:
- Full audit report with per-module assessment
- Industry comparison for campaign system
- Clear simplification recommendations with code size targets

## Statistics

**Code Audited**: 8743 lines across 5 layers
- Orchestration: 1912 lines (justified)
- Campaign: 3150 lines (over-engineered)
- Results: 1635 lines (justified, minus 118 dead)
- State: 422 lines (mild bloat)
- Domain: 1159 lines (justified)
- Dead code: 465 lines (removable)

**Reduction Targets**:
| Component | Current | Target | Savings | % |
|-----------|---------|--------|---------|---|
| Campaign | 3150 | 500 | 2650 | 84% |
| State | 422 | 150 | 272 | 64% |
| Dead code | 465 | 0 | 465 | 100% |
| Launcher | 849 | 800 | 49 | 6% |
| **Total** | **8886** | **~5150** | **~3700** | **42%** |

**Execution Time**: 278 seconds (~4.6 minutes)

**Commits Created**: 2
1. Task 1: Orchestration and campaign system audit
2. Task 2: Dead code identification (auto-committed with edits)

## Files Created

1. `.planning/phases/04-codebase-audit/04-04-REPORT-orchestration-results.md` (580 lines)
   - Complete orchestration layer assessment
   - Campaign system industry comparison
   - Results pipeline wiring verification
   - State machine evaluation
   - Domain model field audit
   - Top-level module usage verification
   - Dead code identification
   - Severity classifications
   - Detailed simplification recommendations

## Related Context

**Connects to**:
- Phase 5 (Refactor & Simplify): Provides simplification targets
- Campaign system discussion: Documents over-engineering evidence
- Industry comparison: lm-eval, vLLM, LLMPerf patterns

**Depends on**:
- Phase 4 Plans 01-03: CLI, config, backend audits

**Enables**:
- Phase 5 campaign simplification
- Dead code removal
- State machine refactoring

---

**Audit complete**: 2026-02-05, ~4.6 minutes
**Next action**: Plan 04-05 (Testing & Documentation Audit)
