---
phase: 04-codebase-audit
plan: 02
subsystem: cli-config
tags: [cli, configuration, ssot, audit, simplification]
requires: [04-RESEARCH]
provides:
  - CLI command surface audit with keep/remove recommendations
  - Config field wiring audit (140+ fields traced)
  - SSOT introspection evaluation
  - Unwired field/module identification
affects: [05-refactor]
tech-stack:
  added: []
  patterns:
    - SSOT introspection (Pydantic model derivation)
    - Multi-stage config merging (preset → file → CLI → metadata)
    - Provenance tracking
key-files:
  created:
    - .planning/phases/04-codebase-audit/04-02-REPORT-cli-config.md
  modified: []
decisions:
  - id: cli-surface-3x-industry
    desc: "CLI has 15 commands vs industry norm of 2-5"
    rationale: "Batch, schedule, resume, config new/list/generate-grid are thin wrappers or discovery-only"
    alternatives: ["Keep all commands", "Remove more aggressively"]
    chosen: "Remove 6 commands (40% reduction)"
  - id: campaign-subprocess-nesting
    desc: "Campaign calls 'lem experiment' as subprocess instead of library function"
    rationale: "Unusual pattern causing config loading duplication and coupling"
    alternatives: ["Keep nested calls", "Refactor to direct orchestration layer calls"]
    chosen: "Refactor in Phase 5 to call orchestration directly"
  - id: ssot-introspection-keep
    desc: "SSOT introspection system (851 lines) is 59% genuine, 41% hand-maintained"
    rationale: "Provides research-grade correctness, prevents config drift (per CONTEXT.md)"
    alternatives: ["Remove introspection", "Fully hand-maintain constraints", "Improve SSOT coverage"]
    chosen: "Keep but improve SSOT coverage (move constraints to Pydantic validators)"
  - id: unwired-config-fields
    desc: "5 unwired fields identified (query_rate, traffic_simulation.*, notifications.on_start)"
    rationale: "Fields defined in Pydantic models but never used in execution code"
    alternatives: ["Keep for future", "Implement features", "Remove"]
    chosen: "Remove in Phase 5 (dead/stub code)"
  - id: naming-module-complexity
    desc: "naming.py is 304 lines for experiment name generation"
    rationale: "Excessive complexity for string formatting task"
    alternatives: ["Keep as-is", "Simplify name format", "Remove naming module"]
    chosen: "Simplify to ~150 lines (50% reduction)"
metrics:
  duration: "6 minutes"
  completed: "2026-02-05"
---

# Phase 4 Plan 2: CLI & Configuration System Audit Summary

**One-liner:** Comprehensive CLI and config audit revealing 3x command surface vs industry, 6% unwired config, and 29% code reduction opportunity

## What Was Delivered

### CLI Surface Audit
- **Catalogued 15 unique commands** across 13 modules (5,648 lines)
- **Traced execution paths** for experiment and campaign workflows
- **Industry comparison** against lm-eval-harness, vLLM, nanoGPT
- **Per-command assessment** with keep/simplify/remove recommendations
- **Identified command overlap:** Campaign executes `lem experiment` as subprocess (nested CLI invocation)

### Configuration System Audit
- **Field wiring audit:** 140+ fields across 4 config models (UniversalConfig, backend configs, campaign, user)
- **Wiring status:** 94% fully wired, 4% unwired, 2% partially wired
- **Unwired fields identified:** 5 fields (query_rate, traffic_simulation.*, notifications.on_start)
- **Unwired module identified:** speculative.py (105 lines of stub code)
- **Loader complexity:** 488 lines (4.9x lm-eval-harness)
- **SSOT introspection:** 851 lines, 59% genuinely derived from Pydantic, 41% hand-maintained
- **Supplementary modules:** 8 modules assessed (1,779 lines total)

### Key Findings

**CLI Complexity:**
1. **15 commands vs industry norm of 2-5** (3x more complex)
2. **6 commands are candidates for removal:**
   - `batch` (133 lines) - thin subprocess wrapper
   - `schedule` (298 lines) - niche use case, cron is standard
   - `resume` (178 lines) - discovery-only, doesn't actually resume
   - `config new` (~150 lines) - users copy examples instead
   - `config list` (~70 lines) - thin wrapper over ls/find
   - `config generate-grid` (~200 lines) - should be external script
3. **Campaign.py is largest module** (1,754 lines) due to:
   - Embedded grid expansion logic
   - Nested subprocess calls to `lem experiment`
   - Container management mixed with business logic

**Configuration Wiring:**
1. **132/140 fields fully wired** (94%)
2. **5 unwired fields:**
   - `query_rate` - defined but never used
   - `traffic_simulation.enabled/mode/target_qps` - stub feature, no implementation
   - `notifications.on_start` - field exists, not used in webhook code
3. **1 unwired module:** speculative.py (105 lines, model exists but no execution code)
4. **Loader is 4.9x more complex** than lm-eval-harness (488 vs ~100 lines)
5. **SSOT introspection provides value** but 41% is hand-maintained constraints

**Major Concerns:**
- Nested CLI invocations (campaign → experiment subprocess)
- Grid generation logic in both CLI and orchestration (duplication)
- naming.py is 304 lines for name string generation (excessive)

## Recommendations

### CLI Simplification (40% command reduction)

| Action | Commands Affected | Lines Saved | Justification |
|--------|------------------|-------------|---------------|
| Remove | batch, schedule | 431 | Thin wrappers, users can script |
| Simplify | resume → merge into campaign | 178 | Discovery-only, not execution |
| Remove | config new, config list | ~220 | Rarely used, users have alternatives |
| Extract | config generate-grid | ~200 | Decouple from execution |
| **Total** | **6 commands** | **~1,029** | **Industry norm: 2-5 commands** |

**Result:** 15 → 9 commands (40% reduction)

### Configuration System Simplification (13% code reduction)

| Component | Current LOC | Target LOC | Reduction | Action |
|-----------|-------------|------------|-----------|--------|
| loader.py | 488 | 400 | 88 | Simplify provenance tracking |
| introspection.py | 851 | 650 | 201 | Move constraints to Pydantic validators |
| naming.py | 304 | 150 | 154 | Simplify name generation logic |
| speculative.py | 105 | 0 | 105 | Remove (unwired stub) |
| **Total** | **1,748** | **1,200** | **548** | **31% reduction in these modules** |

**Unwired field/module removal:**
- Remove 5 unwired fields (query_rate, traffic_simulation.*)
- Remove speculative.py (105 lines)
- Fix notifications.on_start (implement or remove)

### Overall Impact

| Metric | Current | Target | Reduction |
|--------|---------|--------|-----------|
| CLI commands | 15 | 9 | 40% |
| CLI code (LOC) | 5,648 | ~3,500 | 38% |
| Config code (LOC) | 3,401 | 2,953 | 13% |
| **Total** | **9,049** | **6,453** | **29%** |

**Maintained capabilities:**
- ✓ Provenance tracking (research-grade feature)
- ✓ SSOT introspection (prevents config drift)
- ✓ Multi-backend support
- ✓ Campaign orchestration (simplified, not removed)
- ✓ Validation and discovery commands

## Deviations from Plan

None - plan executed exactly as written. Both Task 1 (CLI audit) and Task 2 (config audit) completed with all required deliverables:

**Task 1 deliverables:**
- ✓ Complete command catalogue (15 commands)
- ✓ Execution path analysis (experiment and campaign traced)
- ✓ Industry comparison (lm-eval, vLLM, nanoGPT)
- ✓ Per-command keep/simplify/remove recommendations

**Task 2 deliverables:**
- ✓ Config field wiring table (140+ fields)
- ✓ Unwired fields identified (5 fields, 1 module)
- ✓ Loader complexity assessment (488 lines, 4.9x industry)
- ✓ SSOT introspection evaluation (851 lines, 59% genuine)
- ✓ Supplementary module assessment (8 modules)

## Next Phase Readiness

**Phase 5 (Refactor & Simplify) can proceed with:**

### Immediate Actions (High Priority)
1. **Remove 6 CLI commands** (batch, schedule, resume, config new/list/generate-grid)
   - Estimated effort: 2-3 hours
   - Risk: Low (thin wrappers, minimal dependencies)
2. **Remove unwired code** (5 fields, speculative.py)
   - Estimated effort: 1 hour
   - Risk: Low (confirmed unused via code search)
3. **Simplify campaign.py** (remove subprocess nesting)
   - Estimated effort: 4-6 hours
   - Risk: Medium (significant refactor, needs testing)

### Follow-up Actions (Medium Priority)
4. **Simplify naming.py** (304 → 150 lines)
   - Estimated effort: 2 hours
   - Risk: Low (string formatting, no business logic)
5. **Simplify loader.py provenance** (488 → 400 lines)
   - Estimated effort: 3 hours
   - Risk: Low (reduce implementation, keep functionality)
6. **Move introspection constraints to Pydantic** (851 → 650 lines)
   - Estimated effort: 4-5 hours
   - Risk: Medium (changes model layer, needs validation)

**Blockers:** None

**Concerns:**
- Campaign subprocess nesting refactor is largest task (1,754 → ~800 lines)
- Need to maintain provenance tracking and SSOT features during simplification
- Test coverage should be verified before removing commands

## Files Changed

**Created:**
- `.planning/phases/04-codebase-audit/04-02-REPORT-cli-config.md` (1,140 lines)

**Modified:**
- None (read-only audit)

## Commits

- `3f5f4a7` - docs(04-02): CLI and config system audit

---

**Audit complete.** Report provides actionable recommendations with industry evidence for Phase 5 refactor.
