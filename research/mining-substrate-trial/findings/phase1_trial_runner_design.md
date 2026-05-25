# Phase 1 Day 5 - Trial runner + scoring harness design

**Status:** Phase 1 stubs complete; contracts wired up; Phase 2 fills implementation bodies.
**Authored:** 2026-05-25.
**Cross-refs:**
- `.planning/mining-substrate-empirical-trial.md` § Experimental design / Tooling (shared).
- `research/mining-substrate-trial/findings/trial_epistemic_framing.md` § Capture failure modes explicitly, § Adjacent observations get a place to land.
- `research/mining-substrate-trial/scripts/trial_scoring.py`, `research/mining-substrate-trial/scripts/trial_runner.py`, `research/mining-substrate-trial/scripts/trial_aggregate.py`, `research/mining-substrate-trial/scripts/test_trial_scoring.py`.
- `scripts/validate_invariants.py` + `scripts/_invariant_validation_common.py` (the runtime-validation harness; intentionally NOT reused - see § Reuse decision).

---

## 1. Data flow

```
                 (trial_runner.py - orchestrator)
                              |
                              v
                +-------------+--------------+
                |    resolve_cell_config     |
                |  (venv path, prompt ref,   |
                |   hybrid pattern, etc)     |
                +-------------+--------------+
                              |
                              v
                +-------------+--------------+
                | measure_energy_during(...) |
                |    wraps strategy run      |
                |    captures wall + Wh      |
                +-------------+--------------+
                              |
              +---------------+---------------+
              |               |               |
              v               v               v
        run_strategy_a   run_strategy_b   run_strategy_d
        (existing        (Ollama OSS LLM, (hybrid pattern;
         producers;       prompt template, dispatcher to
         emits to         JSON-mode+retry, individual
         engine_versions/) emits to         pattern fns)
                          research/mining-substrate-trial/findings/
                          trial_runs/b/)
                              |
                              v
                +-------------+--------------+
                |  trial_scoring.score_cell  |
                |  (7-metric rubric +        |
                |   failure-mode tags)       |
                +-------------+--------------+
                              |
                              v
                +-------------+--------------+
                |  _write_score_and_diffs    |
                |  emits .json + 3 sidecar   |
                |  YAMLs per cell            |
                +-------------+--------------+
                              |
                              v
              research/mining-substrate-trial/findings/trial_scores/<cell>.json
              research/mining-substrate-trial/findings/trial_runs/<strategy>/<engine>/<vslug>/
                  recall_misses.yaml
                  precision_spurious.yaml
                  type_mismatches.yaml

                 (trial_aggregate.py - runs separately)
                              |
                              v
              research/mining-substrate-trial/findings/trial_matrix.md
              research/mining-substrate-trial/findings/trial_matrix.csv
```

The flow is per-cell sequential; the registry holds `(strategy, engine, version_slug) -> CellSpec`. The CLI supports a `--cell-spec` filter for re-runs and `--all` for sweeps. `--dry-run` short-circuits at the planning stage.

---

## 2. Per-strategy output dir layout

The trial keeps strategy outputs SEPARATE from the canonical `engine_versions/` tree (the product surface). Justification: the trial is an excursion; mixing trial artefacts into product paths leaks experimental work into ship-able code.

```
research/mining-substrate-trial/findings/
+- trial_runs/
|   +- a/                               # strategy (a) "pure mining"
|   |   +- transformers/
|   |       +- v4_57_3/
|   |           +- recall_misses.yaml
|   |           +- precision_spurious.yaml
|   |           +- type_mismatches.yaml
|   |       (the schema + invariants OUTPUTS for (a) live in the canonical
|   |        engine_versions/transformers/v4_57_3/outputs/ - NOT here. The
|   |        cell only emits sidecar diffs alongside.)
|   +- b/                               # strategy (b) "OSS LLM"
|   |   +- transformers/
|   |       +- v4_57_3/
|   |           +- schema.json          # candidate schema (parallel artefact)
|   |           +- proposed.yaml        # candidate invariants
|   |           +- llm_transcript/      # raw LLM input/output
|   |           |   +- schema_call.txt
|   |           |   +- invariants_call.txt
|   |           +- recall_misses.yaml
|   |           +- precision_spurious.yaml
|   |           +- type_mismatches.yaml
|   +- c/                               # strategy (c) "Claude API"
|   |   ... same shape as b/
|   +- d-ab/                            # strategy (d) "hybrid mining + OSS LLM"
|   |   +- transformers/
|   |       +- v4_57_3/
|   |           +- schema.json
|   |           +- proposed.yaml
|   |           +- llm_transcript/
|   |           +- hybrid_log.md        # orchestration trace
|   |           +- recall_misses.yaml
|   |           +- ...
|   +- d-ac/                            # strategy (d) "hybrid mining + Claude"
|       ... same shape as d-ab/
+- trial_scores/                        # per-cell records (JSON; aggregator input)
|   +- a__transformers__v4_57_3.json
|   +- b__transformers__v4_57_3.json
|   +- ... etc
+- trial_matrix.md                      # aggregator output (human)
+- trial_matrix.csv                     # aggregator output (machine)
```

Score files (`<strategy>__<engine>__<version_slug>.json`) live ABOVE the per-strategy run dirs so the aggregator can glob across strategies in one pass.

---

## 3. Per-cell record shape (CellScore dataclass)

Format version `"1.0.0"`. JSON-serialisable. Field list (10 numeric + 1 tag + observations + raw counts + metadata):

| Field | Type | Source |
|---|---|---|
| `strategy` | `str` | Cell identity |
| `engine` | `str` | Cell identity |
| `version_slug` | `str` | Cell identity |
| `bump_distance` | `str` | Cell identity ("v-2" \| "v-1" \| "active" \| "v+1" \| "v+major") |
| `schema_recall` | `float` | 7-metric rubric |
| `schema_precision` | `float` | 7-metric rubric |
| `schema_type_accuracy` | `float` | 7-metric rubric |
| `invariant_recall` | `float` | 7-metric rubric |
| `invariant_precision` | `float` | 7-metric rubric |
| `invariant_severity_accuracy` | `float` | 7-metric rubric |
| `wall_clock_sec` | `float` | Runner measurement |
| `energy_wh` | `float` | Runner measurement (NVML via `llenergymeasure.energy.select_energy_sampler`) |
| `schema_failure_mode` | `str` | Failure-mode tag for the schema artefact |
| `invariant_failure_mode` | `str` | Failure-mode tag for the invariants artefact |
| `failure_modes` | `list[str]` | Aggregate, deduplicated |
| `brittleness_pass_through_rate` | `float \| None` | Aggregator-filled (compares against active-version cell) |
| `brittleness_silent_fail_count` | `int \| None` | Aggregator-filled |
| `brittleness_detectable_fail_count` | `int \| None` | Aggregator-filled |
| `brittleness_patch_cost_loc` | `int \| None` | Aggregator-filled (None = "needs human estimate") |
| `schema_reference_count` | `int` | Raw count |
| `schema_cell_count` | `int` | Raw count |
| `schema_intersection_count` | `int` | Raw count |
| `invariant_reference_count` | `int` | Raw count |
| `invariant_cell_count` | `int` | Raw count |
| `invariant_intersection_count` | `int` | Raw count |
| `observations` | `list[str]` | Adjacent-observations rollup feed |
| `scoring_format_version` | `str` | Always "1.0.0" |
| `scored_at` | `str` | ISO-8601 timestamp |
| `reference_path` | `str` | Path to reference catalogue (audit trail) |
| `cell_schema_path` | `str` | Path to cell schema output |
| `cell_invariants_path` | `str` | Path to cell invariants output |

### Failure-mode taxonomy

`FailureMode` enum (wire values are stable strings; downstream CSV depends):

- `none` - cell completed; output parsable; metrics computed.
- `crash` - strategy raised before producing output (timeout, exception, no file).
- `detectable` - output present but with visible defects (parse error, empty arrays, marker `error: ...`).
- `silent` - parsable output, materially incorrect. Recall < `silent_threshold` (default 0.20) AND parsable AND no detectable signals. The dangerous case.
- `partial` - one of two artefacts present (schema parsed, invariants didn't, or vice versa).

### Canonical item-matching

- **Schema field identity**: `(namespace, name)` - `namespace` is `"engine_params"`, `"sampling_params"`, or `"$defs.<class>"`.
- **Invariant identity**: `(namespace, native_field, predicate_kind)` - predicate_kind is one of `exact | not_in | not_equal | gt | lt | ge | le | type_is_not | present | unknown`. Severity is deliberately NOT part of identity (severity accuracy is a separate metric).

NO fuzzy matching. The trial explicitly forbids it for the comparison axis to stay clean. The `ScoringConfig.fuzzy_field_match` knob exists but is locked to `False` in v1.0.

---

## 4. Reuse vs new-build

The directive was "prefer REUSE over rewrite. Don't reimplement; import + compose."

### Pieces inspected from `scripts/validate_invariants.py` + `_invariant_validation_common.py`

| Symbol | Considered for reuse | Decision |
|---|---|---|
| `CaseResult` | Per-invariant runtime outcome (raised / warned / dormant) | NOT reused. CaseResult answers "did the library behave as the corpus claimed?". Scoring answers "did the cell's mined catalogue match the reference?". Different question, different data shape. |
| `Divergence` | Expected-vs-observed mismatch tuple | NOT reused. Same reasoning - `Divergence` is runtime-behaviour mismatch; `ItemDiff` is catalogue mismatch. Confusing them would force a single struct to carry two semantic loads. |
| `compare_expected_vs_observed` | Field-by-field comparison loop | NOT reused. Compares declared-vs-observed for one INVARIANT'S runtime trace. Scoring compares two CATALOGUES of invariants. The semantics don't overlap. |
| `run_case` / `CaptureBuffers` | Runtime execution capture | NOT reused at scoring time. WILL be useful in Phase 3 when the runner OPTIONALLY runs the cell's emitted `kwargs_positive`/`kwargs_negative` to tie-break "plausible but wrong" invariants. Phase 1 does not import this; Phase 3 will import via `from scripts._invariant_validation_common import ...`. |
| `classify_outcome` / `classify_emission_channel` | Runtime classification | NOT reused; same reasoning as `CaseResult`. |
| `message_template_to_substring` | Static fragment extraction for message comparison | NOT reused; not relevant to the scoring rubric. |

### Why NO reuse in Phase 1 scoring

The runtime-validation harness and the catalogue-scoring harness operate on different inputs (live library calls vs static YAML/JSON files), produce different outputs (runtime outcome classification vs catalogue-set comparison), and serve different questions. Trying to fit one onto the other would create a forced abstraction. The trial's scoring is fundamentally a structured set-comparison + per-item attribute-comparison; that maps cleanly onto Python's set operations + dataclass diffs.

### Where reuse WILL happen (later phases)

| Phase | What gets imported | Purpose |
|---|---|---|
| Phase 3 | `from scripts._invariant_validation_common import run_case, classify_outcome` | Optional tie-break: a cell that emits "plausible" invariants gets its `kwargs_positive`/`kwargs_negative` actually executed through the library to spot silent-failure cases that pure catalogue comparison misses. |
| Phase 3 | Per-engine producer entry points (`scripts.engine_producers.*`) | Strategy (a) execution dispatches to existing producer scripts; no rewriting. |
| Phase 5 | `validate_invariants.validate_engine` | The chosen-strategy curation pipeline re-validates emitted invariants against the runtime, end-to-end. |

### What IS reused right now

- `llenergymeasure.energy.select_energy_sampler` - the project's own energy API. Imported lazily inside `measure_energy_during` to keep import cost light when energy measurement isn't required (dry-run, test).
- `yaml`, `json` - stdlib + PyYAML.
- `dataclasses`, `enum`, `pathlib.Path`, `time.perf_counter`, `datetime` - stdlib.

No new third-party deps introduced.

---

## 5. CLI surface

### `trial_runner.py`

```
python -m _spike.scripts.trial_runner --strategy a --engine transformers --version-slug v4_57_3
python -m _spike.scripts.trial_runner --all                       # full sweep
python -m _spike.scripts.trial_runner --all --dry-run             # plan only
python -m _spike.scripts.trial_runner --cell-spec a/vllm/v0_7_3    # re-run filter
```

### `trial_aggregate.py`

```
python -m _spike.scripts.trial_aggregate                          # read default paths
python -m _spike.scripts.trial_aggregate --scores-dir /custom \
    --md-out /tmp/matrix.md --csv-out /tmp/matrix.csv
```

### `trial_scoring.py`

Library only; no CLI. Imported by `trial_runner.py` and `test_trial_scoring.py`.

---

## 6. Phase 1 contract - what's stubbed, what's implemented

### Implemented (Phase 1, today)

- All dataclass shapes (`CellSpec`, `CellScore`, `ItemDiff`, `ScoringConfig`, `StrategyAggregate`, `EngineAggregate`, `BumpAggregate`, `TrialMatrix`).
- `FailureMode` enum + wire values.
- Identity extraction:
  - `schema_field_identities()` - fully implemented.
  - `schema_type_at()` - fully implemented (handles primitive types + anyOf).
  - `invariant_identity()` / `invariant_identities()` / `invariants_by_identity()` - fully implemented.
  - `_predicate_kind_for()` - fully implemented; covers all documented predicate kinds.
- `normalise_severity()` - fully implemented.
- `write_diff_artefact()` - fully implemented (stable YAML, sortable).
- `CellScore.to_json()` - fully implemented.
- CLI scaffolding (argparse, dispatch) for both runner + aggregator.
- `CELL_REGISTRY` seeded with the 3 active-version cells per the matrix table.
- `rollup_observations()` in the aggregator - fully implemented (dedup by exact match).
- `load_scores()` in the aggregator - fully implemented.

### Stubbed (Phase 2+ fills in)

- `score_cell()` / `score_schema()` / `score_invariants()` - the core scoring bodies.
- `compute_brittleness()` - Phase 3 (needs active-cell scores for cross-cell comparison).
- `resolve_cell_config()` - Phase 2 (Day 3 venvs + Day 4 prompt templates need to exist first).
- `measure_energy_during()` - Phase 2 (small; just needs the project venv has GPU access path validated).
- `run_strategy_a/b/c/d()` - Phase 2 (the actual cell execution).
- `run_cell()` - Phase 2 dispatcher.
- `_emit_crash_record()`, `_write_score_and_diffs()` - Phase 2 (small adapters around the implemented pieces).
- `cells_matching()` - Phase 2 (trivial filter; stubbed because the registry is still being built out for v-bumped cells).
- `aggregate_strategies()` / `aggregate_engines()` / `aggregate_bumps()` / `build_matrix()` - Phase 3.
- `emit_markdown()` / `emit_csv()` - Phase 3.

---

## 7. Open questions for Phase 2/3

### Q1: How are reference catalogues bootstrapped for non-active cells?

The plan calls for "bootstrap from union-of-strategies during Phase 1 Day 4; human review to remove spurious." This means the reference catalogue for, say, `(transformers, v+1)` is the UNION of every strategy's output for that cell, human-curated. But that union depends on running the strategies FIRST. Resolution options:

- **(A) Two-pass**: Phase 2 runs strategies without scoring (writes outputs only); Day 4 builds references from outputs; Phase 2 re-runs with scoring. Adds a pass but isolates concerns.
- **(B) Lazy scoring**: Strategies write outputs; references built incrementally as cells complete; aggregator re-scores any cell whose reference was last updated AFTER the cell's score. Self-healing but harder to reason about.
- **(C) Defer non-active scoring entirely**: Phase 2 + 3 only score against the active version for each engine; brittleness analysis (Phase 4) uses cross-strategy comparison on the bumped cells directly (no fixed reference). Cheaper; risk of conflating strategy-quality and reference-quality.

Recommendation (open): start with (C) and defer ground-truth references for non-active cells unless the trial signal demands it. This keeps Phase 1 Day 4's human-review budget bounded (~5 hours per engine for the active version only).

### Q2: How aggressive should silent-failure detection be?

`ScoringConfig.silent_threshold` defaults to 0.20 - anything below 20% recall in a parsable envelope is tagged silent. But Bake-off B's actual transformers schema result was 52% recall with parsable output and clear failure modes (long-signature truncation, BitsAndBytesConfig under-walking). That's NOT "silent" in the trial's intended sense - the failure modes are visible. Resolution options:

- **(A) Keep threshold low (0.20)**: silent tag fires only on near-empty parsable output.
- **(B) Raise threshold to ~0.50**: silent tag fires on "looks complete but isn't" cases like Bake-off D's sampling-param type-strip.
- **(C) Use a different signal entirely**: silent = "high precision + low recall + no missing-section markers." More semantic but trickier to compute.

Recommendation (open): go with (A) for now; phase 3 may add (C) once enough cells are scored to calibrate.

### Q3: Should `kwargs_positive` / `kwargs_negative` validation feed back into scoring?

Strategy (b/c) emits invariants with `kwargs_positive`/`kwargs_negative`. The runtime-validation harness (`scripts/validate_invariants.py`) can execute these against the live library. Should scoring penalise cells whose emitted invariants fail runtime-validation, beyond the precision miss?

Recommendation (open): yes, but as a separate metric (`runtime_validated_count` / `runtime_failed_count`), NOT folded into precision. The matrix synthesis benefits from knowing both axes separately.

### Q4: Hybrid pattern variants - registry shape

Strategy (d) is exploratory; the plan calls for ~5-10 distinct hybrid patterns. The runner currently has one `run_strategy_d()` dispatcher; Phase 2 builds per-pattern functions under `research/mining-substrate-trial/scripts/hybrid/<pattern>.py`. Question: should the registry carry `(strategy, pattern, engine, version_slug)` as identity (lengthens keys) or should each pattern be its own pseudo-strategy name (`d-deterministic-validates`, `d-llm-extends`, ...)?

Recommendation (open): use pseudo-strategy names. Keeps the 3-tuple registry key consistent; pattern variants are just additional strategy IDs. Trade-off: the aggregator's per-strategy rollup will have more rows; but each pattern is meant to be compared independently anyway (per the epistemic framing).

### Q5: Parallelisation

The runner is currently sequential. Strategy (a) cells run in seconds; (b)/(c) cells run in 30-90s each per the tactical context. With 75 cells max, sequential = ~75 minutes for (b/c)-heavy runs. Worth parallelising?

Recommendation (open): yes for (b/c) on the same engine + version (one inference at a time per GPU model). Phase 3 can introduce a `--parallel N` flag if needed. Sequential default in Phase 2 keeps the implementation simple.

---

## 8. Concrete next-steps for Phase 2

In priority order:

1. **Day 3 dependency**: confirm `/tmp/trial_<engine>_<version_slug>_venv/` paths exist for every cell the runner will execute. Implement `resolve_cell_config()` with fail-fast on missing venv.
2. **Wire `measure_energy_during()`**: small (~15 lines). Test on the project venv; verify NVML samples on a real GPU-bearing host.
3. **Wire `score_schema()` + `score_invariants()`**: the core scoring loop. Self-test target: scoring the reference against itself yields recall=precision=1.0 on both axes. The test file has stubs for this assertion (currently asserts `NotImplementedError`); Phase 2 inverts these.
4. **Wire `run_strategy_a()`**: invokes the existing producers via subprocess (no in-process import - keeps the trial sandbox-friendly).
5. **Build prompt-template scaffolding for (b)**: under `research/mining-substrate-trial/scripts/prompts/<engine>/{schema,invariants}.txt`. Reuses the bake-off B prompts as the v0; trial Phase 2 calibration iterates on these.
6. **Wire `run_strategy_b()`**: Ollama call with retry-on-parse-error + structured-output + code-fence stripping (Bake-off B failure mode lesson).
7. **Wire `cells_matching()` + `run_cell()` orchestration**: small adapters.
8. **Sanity-run on transformers v4_57_3 (active cell)**: confirm strategy (a) self-scores 1.0/1.0, strategy (b) scores ~0.5/0.9 (Bake-off B baseline).
9. **Wire aggregator** (`build_matrix()`, `emit_markdown()`, `emit_csv()`): straightforward; the dataclass shapes are already in place.
10. **Add v-bumped cells to the registry**: depends on Phase 1 Day 3's PyPI version lock; appended to `_register()` calls.

The scoring harness contract is locked. Phase 2's job is to fill the bodies; the signatures and dataclasses don't change.

---

## 9. Test status (Phase 1 smoke)

`uv run python -m pytest research/mining-substrate-trial/scripts/test_trial_scoring.py -v` - **13/13 PASSED** in 0.18s.

Coverage:
- Identity extraction (schema + invariants) against real reference data: 4 tests.
- Predicate-kind classifier: 1 test (11 cases).
- Severity normalisation: 1 test (loose + exact mode).
- CellScore dataclass + serialisation + failure-mode enum: 2 tests.
- Stub contract (NotImplementedError) for 4 deferred functions: 4 tests.
- Sidecar writer determinism + sort stability: 1 test.

Phase 2 inverts the 4 stub tests to assert real behaviour once `score_cell()` is implemented.
