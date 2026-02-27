---
milestone: M1
name: "Core Single-Experiment"
audited: 2026-02-27T14:00:00Z
status: passed
resolved: 2026-02-27
resolved_by: [Phase 8.1, Phase 8.2]
scores:
  requirements: 102/102
  phases: 9/9
  integration: 7/7
  flows: 7/7
gaps:
  requirements:
    - id: "RES-06"
      status: "resolved"
      resolved_by: "Phase 8.1"
      evidence: "PyTorchBackend._build_result() now extracts baseline_power_w and energy_adjusted_j from EnergyBreakdown. 08.1-VERIFICATION.md confirms field wiring."
    - id: "RES-16"
      status: "resolved"
      resolved_by: "Phase 8.1"
      evidence: "effective_config populated, timeseries co-located with result.json via CLI save_result() wiring. 08.1-VERIFICATION.md confirms."
    - id: "CM-16"
      status: "resolved"
      resolved_by: "Phase 8.1"
      evidence: "Field name mismatch fixed (timeseries_path → timeseries). extra='forbid' on ExperimentResult catches future mismatches. 08.1-VERIFICATION.md confirms."
    - id: "CFG-01 through CFG-10, CFG-18–CFG-26"
      status: "resolved"
      resolved_by: "Phase 8.2"
      evidence: "02-VERIFICATION.md created with status: passed, 11/11 truths, 19/19 requirements SATISFIED. 08.2-VERIFICATION.md confirms."
  integration:
    - from: "PyTorchBackend._build_result()"
      to: "ExperimentResult"
      issue: "timeseries_path= kwarg silently ignored (field is named timeseries). ExperimentResult.model_config missing extra='forbid' allows silent field name mistakes."
      affected_reqs: ["RES-16", "RES-17", "CM-16"]
    - from: "PyTorchBackend.run()"
      to: "results/persistence.py"
      issue: "Timeseries written to output_dir/ flat but result saved in output_dir/{subdir}/ — sidecar unreachable via load_result()"
      affected_reqs: ["RES-16", "RES-17"]
    - from: "PyTorchBackend._build_result()"
      to: "ExperimentResult.effective_config"
      issue: "effective_config never set — persistence names all dirs unknown_pytorch_{ts}"
      affected_reqs: ["RES-16"]
    - from: "PyTorchBackend._build_result()"
      to: "ExperimentResult top-level fields"
      issue: "baseline_power_w and energy_adjusted_j never set; data exists in energy_breakdown but not propagated"
      affected_reqs: ["RES-06"]
  flows:
    - name: "Timeseries round-trip"
      breaks_at: "ExperimentResult constructor — field name mismatch"
    - name: "Output directory naming"
      breaks_at: "persistence._experiment_dir_name() — effective_config empty"
    - name: "Baseline display"
      breaks_at: "cli._display.print_result_summary() — top-level fields always None"
    - name: "Timeseries sidecar discovery"
      breaks_at: "load_result() looks in result.json parent dir, but timeseries written to different path"
tech_debt:
  - phase: "01-measurement-foundations"
    items:
      - "cli/experiment.py imports from deleted state.experiment_state — ImportError if directly imported (not reachable via cli/__init__.py)"
      - "cli/utils.py same broken import — only reachable via function-body lazy import"
      - "cli/CLAUDE.md references deleted v1.x commands"
  - phase: "02-config-system"
    items:
      - "Missing formal VERIFICATION.md (has UAT only)"
  - phase: "04-pytorch-backend-pre-flight"
    items:
      - "_prepare_prompts() is M1 placeholder generating synthetic 'Hello, ' strings — functional but not real dataset loading"
  - phase: "05-energy-measurement"
    items:
      - "REQUIREMENTS.md status drift: CM-15, CM-16, CM-17, CM-18, CM-19, CM-20, CM-25 remain 'Pending' despite being implemented"
      - "_prepare_prompts() comment says 'Phase 5 replaces this' but Phase 5 plans contain no such task"
  - phase: "06-results-schema-and-persistence"
    items:
      - "ExperimentResult.model_config missing extra='forbid' — allows silent field name mistakes (root cause of timeseries bug)"
  - phase: "cross-phase"
    items:
      - "aggregate_results() in results/aggregation.py defined but no caller in v2.0 _run() pipeline"
      - "export_aggregated_to_csv() in results/exporters.py — no CLI command or internal caller"
      - "FlopsEstimator legacy class in core/flops.py — superseded by estimate_flops_palm()"
      - "StateManager (core/state.py) not wired into v2.0 _run() pipeline — only used by broken v1.x cli/experiment.py"
      - "SubprocessRunner (infra/subprocess.py) not wired into v2.0 pipeline — created in Phase 1, unused"
      - "UserConfig.output.results_dir never applied as default output_dir in CLI"
      - "Dataset field parsed in ExperimentConfig but _prepare_prompts() ignores config.dataset entirely"
---

# M1 — Core Single-Experiment: Milestone Audit Report

**Milestone:** M1 — Core Single-Experiment (Phases 1–8)
**Shipped:** 2026-02-27
**Audited:** 2026-02-27
**Status:** passed (gaps resolved by Phases 8.1 + 8.2)

---

## Executive Summary

M1 delivered a functional single-experiment measurement pipeline: `llem run --model gpt2 --backend pytorch` executes end-to-end with energy measurement, timeseries, pre-flight checks, and CLI display. 95 of 102 M1 requirements are satisfied in code. However, **4 cross-phase integration bugs** in the PyTorch backend → ExperimentResult wiring mean that timeseries linkage, output directory naming, and baseline display do not work correctly at runtime. Additionally, **Phase 2 (Config System)** is missing a formal VERIFICATION.md document (7 requirements have no formal verification evidence beyond UAT).

---

## Phase Verification Summary

| Phase | VERIFICATION.md | Status | Score | Notes |
|-------|----------------|--------|-------|-------|
| 1. Package Foundation | Yes | passed | 11/11 | Re-verified after gap closure |
| 2. Config System | **No** | **unverified** | 11/11 UAT | UAT exists, SUMMARYs complete — no formal VERIFICATION.md |
| 3. Library API | Yes | passed | 5/5 | |
| 4. PyTorch Backend | Yes | passed | 12/12 | |
| 4.1. Parameter Audit | Yes | passed | 5/5 | No formal REQ-IDs (quality audit) |
| 5. Energy Measurement | Yes | gaps_found | 15/18 | Doc drift + scope clarification (code is correct) |
| 6. Results Schema | Yes | passed | 18/18 | |
| 7. CLI | Yes | passed | 13/14 | CLI-03 partial — study detection deferred to M2 |
| 8. Testing | Yes | passed | 5/5 | |

---

## Requirements Coverage (3-Source Cross-Reference)

### Summary

| Source | Coverage |
|--------|----------|
| VERIFICATION.md | 83/102 requirements verified (19 missing — all Phase 2) |
| SUMMARY frontmatter | 61/102 requirements listed in `requirements-completed` |
| Integration check | 4 requirements downgraded to PARTIAL by wiring bugs |

### Requirements with Integration Issues (3)

| Requirement | Description | Issue | Severity |
|-------------|-------------|-------|----------|
| RES-06 | `baseline_power_w`, `energy_adjusted_j` top-level fields | Fields exist but never populated by PyTorchBackend — always None | HIGH |
| RES-16 | Output in `{name}_{timestamp}/result.json` + `timeseries.parquet` | Timeseries in wrong dir; effective_config empty → dir named `unknown_*` | HIGH |
| CM-16 | Timeseries 1 Hz sidecar | File written correctly but `ExperimentResult.timeseries` always None (field name mismatch) | HIGH |

### Requirements with Verification Gap (19)

Phase 2 has no VERIFICATION.md. These requirements are covered by UAT (11/11 passed) and SUMMARY frontmatter but lack formal verification:

CFG-01 through CFG-10, CFG-18, CFG-19, CFG-20, CFG-21, CFG-22, CFG-23, CFG-24, CFG-25, CFG-26

**Mitigating evidence:** All downstream phases (3, 4, 7, 8) consume ExperimentConfig successfully. UAT confirmed all 11 acceptance tests pass. All 4 plan SUMMARYs list requirements-completed.

### Fully Satisfied Requirements (80)

All remaining M1 requirements are satisfied across all 3 sources (VERIFICATION.md passed, SUMMARY listed or downstream consumed, integration check wired).

---

## Integration Check Results

### Complete Flows (3/7)

1. **`llem run --model X` happy path** — CLI → load_experiment_config → run_experiment → _run → preflight → PyTorchBackend.run → ExperimentResult → print_result_summary ✓
2. **PyTorch measurement pipeline** — environment snapshot → baseline → model load → warmup → energy start → measurement → CUDA sync → energy stop → FLOPs → build result ✓
3. **`llem config`** — probe GPU → probe backends → load user config → display ✓

### Broken Flows (4/7)

1. **Timeseries round-trip** — Parquet file written but `ExperimentResult.timeseries` always None (field name `timeseries_path=` vs `timeseries=`)
2. **Output directory naming** — `effective_config` never set → all dirs named `unknown_pytorch_{ts}`
3. **Baseline display** — `baseline_power_w` and `energy_adjusted_j` always None in CLI display
4. **Timeseries sidecar discovery** — `load_result()` looks in `result.json` parent but timeseries written to different path

### Root Cause

All 4 broken flows trace to `PyTorchBackend._build_result()` not setting 3 fields:
- `timeseries=` (passes wrong kwarg name `timeseries_path=`)
- `effective_config=` (never set at all)
- `baseline_power_w=` / `energy_adjusted_j=` (data in `energy_breakdown` but not extracted to top-level)

Additionally, `ExperimentResult.model_config` missing `extra="forbid"` means Pydantic silently discards unrecognised kwargs instead of raising a validation error.

---

## Orphaned Exports

| Export | Location | Status |
|--------|----------|--------|
| `aggregate_results()` | `results/aggregation.py` | No caller in v2.0 `_run()` pipeline |
| `export_aggregated_to_csv()` | `results/exporters.py` | No CLI command or internal caller |
| `FlopsEstimator` (legacy) | `core/flops.py` | Superseded by `estimate_flops_palm()` |
| `StateManager` | `core/state.py` | Not wired into v2.0 `_run()` pipeline |
| `SubprocessRunner` | `infra/subprocess.py` | Created in Phase 1, unused in v2.0 |

---

## Tech Debt Summary

**Total: 12 items across 6 categories**

See YAML frontmatter `tech_debt:` section for full itemised list.

Key items:
- 5 orphaned exports (carry-forward code not wired into v2.0 pipeline)
- 3 v1.x import chain breakages (cli/experiment.py, cli/utils.py — not reachable via entry point)
- 1 missing Phase 2 VERIFICATION.md
- 1 dataset loading placeholder (`_prepare_prompts()` generates synthetic prompts, ignores `config.dataset`)
- 1 REQUIREMENTS.md status drift (7 Phase 5 requirements show Pending)
- 1 missing `extra="forbid"` on ExperimentResult (silent field name errors)

---

## Human Verification Required

Aggregated from phase VERIFICATION.md files:

1. **End-to-end GPU experiment** — Run `llem run --model gpt2 --backend pytorch` on GPU hardware (Phases 5, 7, 8)
2. **CUDA sync timing** — Verify `torch.cuda.synchronize()` completes before energy stop at wall-clock level (Phase 5)
3. **tqdm spinner display** — Verify indeterminate spinner renders in real TTY (Phase 7)
4. **Non-TTY suppression** — Pipe output to file, verify spinner suppressed (Phase 7)
5. **SIGINT handling** — Ctrl-C during measurement → exit code 130 (Phase 7)
6. **GPU CI workflow** — Trigger `gpu-ci.yml` on self-hosted runner (Phase 8)

---

## Open Items from .product/REQUIREMENTS.md

| Item | Status |
|------|--------|
| Create `aienergyscore.jsonl` built-in dataset file | TODO — carried to M2 |
| Confirm `peak_memory_mb` measurement semantics | TODO — carried to M2 |
| Fix PyTorch P0 bug (model_kwargs L375) | DONE — fixed in Phase 4 |

---

_Audited: 2026-02-27_
_Auditor: Claude (gsd audit-milestone)_
