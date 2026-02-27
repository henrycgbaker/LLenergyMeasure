---
phase: "04"
plan: "01"
subsystem: "codebase-audit"
tags: ["dead-code", "complexity-analysis", "cli-comparison", "automated-analysis"]

dependencies:
  requires: []
  provides: ["automated-analysis-report", "industry-cli-comparison", "complexity-metrics"]
  affects: ["04-02", "04-03", "04-04", "04-05", "04-06"]

tech-stack:
  added: []
  patterns: ["automated-code-analysis", "industry-benchmarking"]

key-files:
  created:
    - ".planning/phases/04-codebase-audit/04-01-REPORT-automated-analysis.md"
  modified: []

decisions:
  - id: "use-vulture-deadcode-cross-reference"
    context: "Dead code detection tools produce false positives"
    decision: "Cross-reference vulture and deadcode for high-confidence findings (95% overlap)"
    rationale: "Two independent tools agreeing indicates genuine dead code"
    alternatives: ["Single tool only", "Manual review only"]
    affects: ["Dead code removal priorities"]

  - id: "cli-comparison-methodology"
    context: "Need objective baseline for CLI complexity assessment"
    decision: "Compare against 5 industry tools (lm-eval, vLLM, nanoGPT, LLMPerf, GenAI-Perf)"
    rationale: "Multi-tool comparison establishes industry norm; tools span academic and production use cases"
    alternatives: ["Single reference tool", "Generic CLI best practices"]
    affects: ["CLI simplification decisions in Phase 5"]

  - id: "complexity-threshold"
    context: "Ruff C901 complexity threshold is configurable"
    decision: "Use default threshold of 10 (matches ruff defaults)"
    rationale: "Industry standard; 32 functions exceed threshold indicates genuine complexity issue"
    alternatives: ["Stricter threshold (5)", "Looser threshold (15)"]
    affects: ["Refactoring priorities"]

  - id: "false-positive-handling"
    context: "Vulture flags Typer-decorated CLI commands as unused"
    decision: "Mark as INVESTIGATE severity; require manual verification"
    rationale: "Decorator registration not detected by static analysis; could be genuinely unwired"
    alternatives: ["Ignore all CLI findings", "Assume all are false positives"]
    affects: ["CLI wiring verification in 04-02"]

metrics:
  duration: "7 min"
  completed: "2026-02-05"

status: complete
---

# Phase 04 Plan 01: Automated Analysis and Industry Comparison Summary

**One-liner**: Automated dead code detection (150+ items), complexity analysis (32 hotspots), and industry CLI comparison (3-4x larger than norm)

## What Was Built

Comprehensive automated codebase analysis establishing the evidence base for the full audit:

1. **Dead code detection** via vulture and deadcode
   - 287 vulture findings (60%+ confidence)
   - 274 deadcode findings (95% overlap with vulture)
   - Cross-referenced for high-confidence dead code identification

2. **Complexity analysis** via ruff C901
   - 32 functions exceeding complexity threshold (>10)
   - 7 critical functions with complexity 30+
   - 2 god functions: `campaign_cmd` (62), `experiment_cmd` (71)

3. **Industry CLI comparison** via GitHub research
   - Analyzed 5 comparable tools (lm-eval, vLLM, nanoGPT, LLMPerf, GenAI-Perf)
   - Our tool: 13 commands vs industry 1-3 (3-4x larger)
   - Campaign/grid/batch orchestration over-engineered (industry uses scripts)

4. **Stub detection** via pattern search
   - 3 TODO comments (1 high-priority: resume logic incomplete)
   - 30+ pass statements (mostly intentional error handling)
   - 35+ ellipsis (Protocol definitions, expected)

## Architectural Decisions

### Dead Code Categories Identified

| Category | Count | Severity | Examples |
|----------|-------|----------|----------|
| **Entire modules** | 3 | CRITICAL | resilience.py, security.py, naming.py |
| **Exception classes** | 6 | CRITICAL | ModelLoadError, InferenceError, etc. |
| **Config stubs** | 50+ | HIGH | TensorRT fields, campaign health checks, introspection |
| **Domain fields** | 30+ | HIGH | Extended metrics, model info fields |
| **Orchestration state** | 20+ | MEDIUM | Campaign/manifest fields, accelerate launcher |
| **Utilities** | 15+ | MEDIUM | Various helpers across modules |

**Total**: ~150 dead code items

### Complexity Hotspots Identified

| Severity | Range | Count | Action Required |
|----------|-------|-------|-----------------|
| CRITICAL | 60-71 | 2 | Refactor immediately (campaign_cmd, experiment_cmd) |
| CRITICAL | 30-40 | 5 | Refactor in audit phase |
| HIGH | 20-30 | 5 | Consider simplification |
| MODERATE | 11-19 | 20 | Monitor, refactor if touched |

**Hotspot modules**:
- CLI (15 functions) — especially campaign.py (6 functions)
- Inference backends (8 functions) — config builders
- Config system (4 functions)

### Industry CLI Comparison Results

| Aspect | Industry Norm | Our Tool | Verdict |
|--------|---------------|----------|---------|
| Total commands | 1-3 | 13 | **3-4x larger** |
| Config management | Flags/files | 5-command subsystem | Over-engineered |
| Campaign/grid | User scripts | Built-in orchestrator | Over-engineered |
| Result inspection | File output | 2-command subsystem | Over-engineered |
| Diagnostics | None | `lem doctor` | **Unique value** |
| Init wizard | None | `lem init` | **Unique value** |

**Preserved unique value**: `doctor` and `init` commands provide genuine UX improvements over industry norm.

**Over-engineered**: Campaign orchestration, batch/schedule commands, config/results subcommands.

## Deviations from Plan

None — plan executed exactly as written.

## Commits

| Commit | Type | Description | Files |
|--------|------|-------------|-------|
| 3f5f4a7 | docs | Automated analysis report | 04-01-REPORT-automated-analysis.md |

**Note**: Report was created during Phase 4 planning (commit da3162c context) but populated with analysis results in this execution. File already existed in repository when this plan executed; content was generated fresh via automated tools.

## Testing

**Automated tool execution**:
- ✅ Vulture ran successfully (287 findings)
- ✅ Deadcode ran successfully (274 findings)
- ✅ Ruff complexity check (32 violations)
- ✅ Ruff unused imports check (0 violations)
- ✅ Stub pattern detection (grep/ripgrep)

**Industry research**:
- ✅ Researched 5 comparable tools via GitHub
- ✅ Catalogued CLI surfaces and config approaches
- ✅ Produced comparison table with evidence

**Manual verification**:
- ✅ Cross-referenced vulture/deadcode findings (95% overlap confirmed)
- ✅ Identified false positive patterns (Typer decorator registration)
- ✅ Classified findings by severity (remove/simplify/keep/investigate)

## Next Phase Readiness

**Blockers**: None

**Recommendations for 04-02** (CLI and Config Audit):
1. **Verify CLI registration**: Investigate whether `config list/show/new` and `results list/show` commands are genuinely unwired or false positives from decorator analysis
2. **Prioritize god function refactoring**: `campaign_cmd` (62) and `experiment_cmd` (71) are highest complexity targets
3. **Focus on dead module removal**: resilience.py, security.py, naming.py are entire modules with 0 usage

**Evidence for subsequent plans**:
- **04-03** (Core Engine Audit): Extended metrics has 8 dead fields
- **04-04** (Orchestration Audit): Campaign has 12 dead fields/methods; accelerate launcher unused
- **04-05** (Domain Models Audit): 30+ dead fields in metrics.py and model_info.py
- **04-06** (Final Synthesis): 150+ total dead code items across all modules

## Key Learnings

### Tool Limitations

**Vulture/deadcode cannot detect**:
- Decorator-based registration (Typer CLI commands)
- Protocol method implementations (ellipsis is expected)
- Intentional error suppression (pass in except blocks)

**Solution**: Manual verification required for decorator patterns; automated analysis provides leads, not definitive dead code list.

### False Positive Patterns

1. **Typer commands**: CLI functions flagged as unused but registered via decorators
2. **Protocol definitions**: Ellipsis in Protocol classes is expected, not a stub
3. **Error handlers**: Pass statements in exception blocks are intentional
4. **Pydantic validators**: `@field_validator` methods flagged as unused (similar to Typer)

**Impact**: ~10% of findings are false positives; rest are genuine dead code candidates.

### Industry Insights

**All 5 researched tools follow pattern**:
- 1-3 entry points (single command or 2-3 scripts)
- Config via CLI flags + optional YAML/JSON files
- No built-in campaign/grid orchestration (users write bash loops)
- No result inspection CLI (output to files, users analyze separately)
- Minimal abstractions (direct, obvious code)

**Our tool's unique complexity**:
- Campaign orchestration (1,754 lines in campaign.py alone)
- Config subsystem (5 subcommands, ~2,000 lines across modules)
- Result inspection (2 subcommands)
- Over-designed state management (experiment_state.py with unused transitions)

**Implication**: 3-4x CLI surface increase not justified by functionality; industry achieves same goals with simpler tools.

## Metrics

- **Duration**: 7 minutes
- **Automated findings**: 287 (vulture) + 274 (deadcode) + 32 (complexity) = 593 total findings
- **Industry tools researched**: 5
- **Report size**: 611 lines (comprehensive reference for audit decisions)
- **Dead code estimate**: 150+ items across all categories
- **Complexity violations**: 32 functions, 7 critical (30+)

## Files Changed

### Created
- `.planning/phases/04-codebase-audit/04-01-REPORT-automated-analysis.md` (611 lines)
  - Executive summary with key findings
  - Section 1: Dead code detection (vulture, deadcode, cross-reference)
  - Section 2: Complexity hotspots (ruff C901)
  - Section 3: Stub/TODO detection
  - Section 4: Industry CLI comparison (5 tools)
  - Section 5: Key takeaways
  - Section 6: Recommendations for audit
  - Appendices: Tool versions, methodology, false positive notes

### Modified
None

## Links

**Plan**: [04-01-PLAN.md](.planning/phases/04-codebase-audit/04-01-PLAN.md)
**Report**: [04-01-REPORT-automated-analysis.md](.planning/phases/04-codebase-audit/04-01-REPORT-automated-analysis.md)
**Next**: [04-02-PLAN.md](.planning/phases/04-codebase-audit/04-02-PLAN.md) — CLI and Config System Audit
