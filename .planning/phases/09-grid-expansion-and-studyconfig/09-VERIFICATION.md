---
phase: 09-grid-expansion-and-studyconfig
verified: 2026-02-27T12:00:00Z
status: passed
score: 10/10 must-haves verified
re_verification: false
---

# Phase 9: Grid Expansion and StudyConfig Verification Report

**Phase Goal:** Researchers can express a sweep configuration in YAML and have it resolve to a complete, ordered list of ExperimentConfig objects before any subprocess is spawned — with a pre-flight count display that prevents combinatorial surprises.
**Verified:** 2026-02-27
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

Source: `09-01-PLAN.md` must_haves + `09-02-PLAN.md` must_haves (combined)

| #  | Truth | Status | Evidence |
|----|-------|--------|----------|
| 1  | ExecutionConfig validates n_cycles >= 1, cycle_order literal, gap fields, shuffle_seed | VERIFIED | `models.py` L435-483; 9 unit tests in TestExecutionConfig all pass |
| 2  | StudyConfig holds list[ExperimentConfig] + ExecutionConfig + study_design_hash + skipped_configs | VERIFIED | `models.py` L486-520; 4 unit tests in TestStudyConfig all pass |
| 3  | expand_grid() resolves sweep: block into Cartesian product of ExperimentConfig objects | VERIFIED | `grid.py` L76-151; TestExpandGridSweep 3 tests pass; 4 configs from 2x2 universal sweep confirmed |
| 4  | Dotted notation pytorch.batch_size routes to backend section dict, not top-level | VERIFIED | `grid.py` L303-344 (`_expand_sweep`); test_backend_scoped_sweep_routes_to_section passes |
| 5  | Three modes work: grid sweep only, explicit experiments: list only, combined sweep + explicit | VERIFIED | `grid.py` L100-107; 3 test classes (TestExpandGridSweep, TestExpandGridExplicit, TestExpandGridCombined) all pass |
| 6  | apply_cycles() produces correct ordering for sequential, interleaved, and shuffled | VERIFIED | `grid.py` L164-191; 7 tests in TestApplyCycles all pass including determinism tests |
| 7  | compute_study_design_hash() returns stable 16-char hex, excludes execution block | VERIFIED | `grid.py` L154-161; 5 tests in TestComputeStudyDesignHash pass; test_load_study_config_hash_excludes_execution confirms execution exclusion |
| 8  | Invalid ExperimentConfig combinations are collected as SkippedConfig, not raised | VERIFIED | `grid.py` L120-128; TestExpandGridInvalidHandling::test_invalid_configs_collected_as_skipped passes |
| 9  | base: file loaded relative to study YAML path, study-only keys stripped | VERIFIED | `grid.py` L256-279 (`_load_base`); TestExpandGridBase 3 tests pass |
| 10 | All configs invalid raises ConfigError; >50% skip rate triggers warning | VERIFIED | `grid.py` L132-149; test_all_invalid_raises_config_error and test_high_skip_rate_warning both pass |

**Score:** 10/10 truths verified

### Plan 02 Additional Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 11 | load_study_config() takes a file path and returns a resolved StudyConfig | VERIFIED | `loader.py` L107-199; 10 integration tests all pass |
| 12 | Sweep resolution at YAML parse time before Pydantic validation | VERIFIED | `loader.py` L158 calls `expand_grid(raw, study_yaml_path=path)` before any individual Pydantic construction |
| 13 | Pre-flight summary shows hash, config count, cycle count, total runs, order mode | VERIFIED | `grid.py` L194-243; TestFormatPreflightSummary 8 tests pass |
| 14 | Skipped configs shown with per-skip reason; >50% skip rate shows WARNING | VERIFIED | `grid.py` L227-241; test_with_skipped_shows_skip_line and test_high_skip_rate_warning pass |
| 15 | CLI overrides on execution block merge correctly | VERIFIED | `loader.py` L143-145; test_load_study_config_cli_overrides passes |

---

## Required Artifacts

### Plan 01 Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/llenergymeasure/config/models.py` | ExecutionConfig model, expanded StudyConfig model | VERIFIED | L435-520: both classes present, fully populated with all fields |
| `src/llenergymeasure/study/__init__.py` | study package init | VERIFIED | File exists with docstring: "Study module — sweep expansion, cycle ordering, manifest, runner." |
| `src/llenergymeasure/study/grid.py` | expand_grid, apply_cycles, compute_study_design_hash, SkippedConfig, CycleOrder | VERIFIED | All 5 exports present and fully implemented (350 lines); no NotImplementedError stubs |
| `tests/unit/test_study_grid.py` | Unit tests covering all 9 test groups | VERIFIED | 46 tests across 8 test classes; all pass |

### Plan 02 Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/llenergymeasure/config/loader.py` | load_study_config() public function | VERIFIED | L107-199; in `__all__` at L31 |
| `src/llenergymeasure/study/grid.py` (extended) | format_preflight_summary() display function | VERIFIED | L194-243; imported and tested |
| `tests/unit/test_config_loader.py` | Tests for load_study_config() | VERIFIED | 10 load_study_config tests appended to existing file, all pass |
| `tests/unit/test_study_grid.py` (extended) | Tests for format_preflight_summary() | VERIFIED | TestFormatPreflightSummary class with 8 tests, all pass |

---

## Key Link Verification

### Plan 01 Key Links

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `study/grid.py` | `config/models.py` | `from llenergymeasure.config.models import ExperimentConfig` | WIRED | Line 18 of grid.py — exact pattern match |
| `study/grid.py` | `exceptions.py` | `from llenergymeasure.exceptions import ConfigError` | WIRED | Line 19 of grid.py — exact pattern match |

### Plan 02 Key Links

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `config/loader.py` | `study/grid.py` | `from llenergymeasure.study.grid import` (expand_grid, apply_cycles, compute_study_design_hash, CycleOrder) | WIRED | Lines 24-29 of loader.py; all 4 symbols imported |
| `config/loader.py` | `config/models.py` | `from llenergymeasure.config.models import ExecutionConfig, ExperimentConfig, StudyConfig` | WIRED | Line 22 of loader.py; all 3 symbols imported and used |

---

## Requirements Coverage

Requirements from PLAN frontmatter: CFG-11, CFG-12, CFG-13, CFG-14, CFG-15, CFG-16 (all 6 from REQUIREMENTS.md Phase 9 traceability table).

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| CFG-11 | 09-01 | `StudyConfig` = resolved container: `list[ExperimentConfig]` + `ExecutionConfig` | SATISFIED | `StudyConfig` at models.py L486: holds `experiments`, `execution`, `study_design_hash`, `skipped_configs`; `ExecutionConfig` at L435 |
| CFG-12 | 09-02 | Sweep resolution at YAML parse time, before Pydantic validation | SATISFIED | `loader.py` L158: `expand_grid(raw, ...)` called before individual `ExperimentConfig(**raw_config)` construction; test_load_study_config_hash_excludes_execution confirms hash is independent of execution |
| CFG-13 | 09-01 | Dotted notation: `pytorch.batch_size: [1, 8]` — backend-scoped grid | SATISFIED | `_expand_sweep` at grid.py L303 splits on first `.`; scoped param goes to `config_dict.setdefault(backend, {})[dim_key] = value`; test passes with batch_sizes {1, 8} in pytorch.batch_size |
| CFG-14 | 09-01 | Three modes: grid sweep (Cartesian), explicit `experiments:` list, combined | SATISFIED | grid.py L100-107: sweep_raw_configs + explicit_raw_configs combined; all three test classes pass |
| CFG-15 | 09-01 | `ExecutionConfig`: n_cycles (Pydantic default=1, CLI effective default=3), cycle_order, config_gap_seconds, cycle_gap_seconds | SATISFIED | `ExecutionConfig` at models.py L435-483: all 5 fields with correct defaults and validators |
| CFG-16 | 09-01 | `study_design_hash` = SHA-256[:16] of sweep+experiments only (execution excluded) | SATISFIED | `compute_study_design_hash` at grid.py L154-161: `json.dumps([exp.model_dump() for exp in experiments], sort_keys=True)` — execution block never serialised; test_load_study_config_hash_excludes_execution confirms |

**No orphaned requirements.** REQUIREMENTS.md maps exactly CFG-11 through CFG-16 to Phase 9, all 6 are covered by plans.

---

## Anti-Patterns Found

Scanned: `src/llenergymeasure/study/grid.py`, `src/llenergymeasure/config/models.py`, `src/llenergymeasure/config/loader.py`, `tests/unit/test_study_grid.py`, `tests/unit/test_config_loader.py`

| File | Pattern | Severity | Impact |
|------|---------|----------|--------|
| `grid.py` L263, L294 | `return {}` and `return []` | INFO | Legitimate guard-branch returns in `_load_base` (no base file) and `_expand_sweep` (empty sweep without model) — not stubs |

No blockers or warnings found. All `return []` / `return {}` instances are in implemented guard branches with corresponding tests.

---

## Human Verification Required

None. All behaviours are programmatically verifiable:
- YAML-to-StudyConfig resolution: covered by integration tests using `tmp_path`
- Cartesian product correctness: counted in unit tests
- Cycle ordering: verified by index-checking in unit tests
- Hash stability: asserted across two separate calls with identical inputs
- Pre-flight display format: string-matched in unit tests

---

## Test Results Summary

| Test File | Tests | Result |
|-----------|-------|--------|
| `tests/unit/test_study_grid.py` | 46 | All pass |
| `tests/unit/test_config_loader.py` | 31 (including 10 new load_study_config tests) | All pass |
| Full `tests/unit/` suite | 461 | All pass — no regressions |

---

## Gaps Summary

No gaps. All must-haves verified at all three levels (exists, substantive, wired). All 6 requirements fully satisfied. 461 unit tests pass with no regressions.

The phase goal is achieved: a researcher can express a sweep YAML with dotted backend-scoped dimensions, explicit experiments, or both; `load_study_config()` resolves the Cartesian product into a flat ordered list of validated `ExperimentConfig` objects before any subprocess is spawned; and `format_preflight_summary()` renders the count display that prevents combinatorial surprises.

---

_Verified: 2026-02-27_
_Verifier: Claude (gsd-verifier)_
