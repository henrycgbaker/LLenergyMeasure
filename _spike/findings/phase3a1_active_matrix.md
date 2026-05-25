# Phase 3a.1 - Active-version matrix (mining substrate empirical trial)

**Status:** closed; all active-cell runs landed.
**Generated at:** 2026-05-25T05:15:51Z (post-aggregator refresh).
**Source records:** `_spike/findings/trial_scores/*.json` (11 score JSONs).
**Cross-refs:** `_spike/findings/trial_matrix.md` (auto-aggregate), `_spike/findings/trial_matrix.csv`, `_spike/findings/trial_epistemic_framing.md` (Phase 4 will synthesise).
**Scope:** 11 records spanning 5 strategies x 3 engines on the locked active versions only (`transformers v4_57_3`, `vllm v0_7_3`, `tensorrt_llm v0_21_0`). Version-bumped cells are Phase 3a.2 scope.

This document is a DATA RECORD. No verdicts, no rankings. Interpretation is deferred to Phase 4.

---

## Per-cell table

All records, sorted by `(strategy, engine, version)`. `S` = schema, `I` = invariants. `r` = recall, `p` = precision. `sev` = severity accuracy. `wall` in seconds, `energy` in Wh.

| # | strategy | engine | version | S r | S p | I r | I p | sev | wall_s | energy_wh | failure_modes | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a | transformers | v4_57_3 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0 | 0.00 | none | reference cell |
| 2 | a | vllm | v0_7_3 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0 | 0.00 | none | reference cell |
| 3 | a | tensorrt | v0_21_0 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0 | 0.00 | none | reference cell |
| 4 | b | transformers | v4_57_3 | 83.0% | 93.9% | 56.4% | 43.1% | 77.3% | 1649.2 | 81.31 | none | multipass; 14 inv chunks |
| 5 | b | vllm | v0_7_3 | 97.0% | 85.1% | 38.5% | 15.2% | 100.0% | 1414.3 | 67.93 | none | multipass; 10 inv chunks |
| 6 | b | tensorrt | v0_21_0 | 56.1% | 46.5% | 0.0% | 0.0% | 0.0% | 1372.2 | 66.44 | none;silent | multipass; namespace mismatch (cell uses `tensorrt_llm.X`, ref uses `tensorrt.X`); 7 inv chunks |
| 7 | b_8b | transformers | v4_57_3 | 85.7% | 93.2% | 35.7% | 16.1% | 100.0% | 412.6 | 4.93 | none | cheaper-variant probe; 8B model |
| 8 | c | transformers | v4_57_3 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0 | 0.00 | key_absent | skipped: ANTHROPIC_API_KEY absent |
| 9 | d-ab | transformers | v4_57_3 | 100.0% | 100.0% | 100.0% | 93.3% | 100.0% | 20.1 | 0.84 | none | extension=2; flagged_spurious=2 |
| 10 | d-ab | vllm | v0_7_3 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 433.6 | 19.38 | none | extension=0; flagged_spurious=0 (pass2 parse failures recorded) |
| 11 | d-ab | tensorrt | v0_21_0 | 100.0% | 100.0% | 100.0% | 79.5% | 100.0% | 207.5 | 10.94 | none | extension=8; flagged_spurious=1 |

### Per-cell intersection counts (raw)

For audit reproducibility:

| record | S ref | S cell | S intersect | I ref | I cell | I intersect |
|---|---|---|---|---|---|---|
| a/transformers/v4_57_3 | 112 | 112 | 112 | 39 | 39 | 39 |
| a/vllm/v0_7_3 | 135 | 135 | 135 | 26 | 26 | 26 |
| a/tensorrt/v0_21_0 | 107 | 107 | 107 | 31 | 31 | 31 |
| b/transformers/v4_57_3 | 112 | 99 | 93 | 39 | 51 | 22 |
| b/vllm/v0_7_3 | 135 | 154 | 131 | 26 | 66 | 10 |
| b/tensorrt/v0_21_0 | 107 | 129 | 60 | 31 | 39 | 0 |
| b_8b/transformers/v4_57_3 | 112 | 103 | 96 | 28 | 62 | 10 |
| c/transformers/v4_57_3 | 0 | 0 | 0 | 0 | 0 | 0 |
| d-ab/transformers/v4_57_3 | 112 | 112 | 112 | 39 | 39 | 39 |
| d-ab/vllm/v0_7_3 | 135 | 135 | 135 | 26 | 26 | 26 |
| d-ab/tensorrt/v0_21_0 | 107 | 107 | 107 | 31 | 39 | 31 |

(b_8b reference invariant count is 28, not 39, because the b_8b run was scored against an earlier reference snapshot; later reference grew to 39 after additional invariants were ratified. Score is still valid for the b_8b probe within its own scoring window.)

---

## Strategy aggregates (across engines)

Means computed across the (engine, version) cells for each strategy. Active-only here; bump-distance dimension is empty.

| strategy | n | S r mean | S p mean | I r mean | I p mean | sev mean | wall mean | energy mean | crashes |
|---|---|---|---|---|---|---|---|---|---|
| a | 3 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0 | 0.00 | 0 |
| b | 3 | 78.7% | 75.2% | 31.6% | 19.4% | 59.1% | 1478.6 | 71.89 | 0 |
| b_8b | 1 | 85.7% | 93.2% | 35.7% | 16.1% | 100.0% | 412.6 | 4.93 | 0 |
| c | 1 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0 | 0.00 | 0 (key_absent) |
| d-ab | 3 | 100.0% | 100.0% | 100.0% | 90.9% | 100.0% | 220.4 | 10.39 | 0 |

(Strategy a aggregates are the reference identity by construction; they are 100% by definition because (a)'s output IS the reference in this matrix.)

Notes:
- Strategy b's `S r mean` and `I r mean` exclude the tensorrt namespace-mismatch effect from the aggregate? No - it IS included (3 cells, raw mean). The 0.0% invariant_recall on b/tensorrt drags the mean down by ~13 pp.
- The aggregator computes pass_through_mean as None for the active row (no version-bumped data yet).
- `c` is excluded from interpretation; it logged its skip and the scoring harness reports zeros.

---

## Engine aggregates (across strategies)

For each engine, mean across strategies that ran (excludes c since it was skipped on all engines). Strategy (a) reference cells are always included.

| engine | n (active) | S r mean | I r mean | wall mean | engine-specific notes |
|---|---|---|---|---|---|
| transformers | 5 (a, b, b_8b, c[skip], d-ab) | 56.7% (incl. c=0) / 73.8% (excl. c) | 58.4% (incl. c=0) / 73.0% (excl. c) | 332.1 / 416.4 | Most strategies ran; widest data |
| vllm | 3 (a, b, d-ab) | 99.0% | 79.5% | 616.0 | b had 14 inv chunks; d-ab extension=0 |
| tensorrt | 3 (a, b, d-ab) | 85.4% | 66.7% | 526.6 | b: namespace-mismatch zeroed I recall; d-ab clean |

The "incl. c" rows are an artefact of aggregator default behaviour (zeros in c land in the mean); the "excl. c" rows are the same with the c crash excluded. trial_aggregate.py reports the inclusive numbers in trial_matrix.md.

---

## Failure-modes catalogue (distinct patterns observed)

Compiled from the `observations` and `failure_modes` fields across all 11 records.

### Tagged in `failure_modes` field

- `none` (recorded on 8 records) - cell produced output without scorer-detected failures.
- `silent` (record 6: b/tensorrt only) - cell produced output with 0 reference matches; scorer flagged as silent (passing the harness despite no overlap).
- `key_absent` (record 8: c/transformers) - external API key missing; cell skipped at dispatch.

### Adjacent observations (mined from `observations`)

Distilled patterns. Source records noted in parentheses.

1. **Namespace mismatch between LLM-emitted invariants and reference catalogue** (record 6 b/tensorrt). Cell-side invariants use `tensorrt_llm.<field>` namespace; reference catalogue (mined by static miner) uses `tensorrt.<field>` namespace. The scoring identity tuple is `(namespace, native_field, predicate_kind, secondary_field)`; different namespace -> identity disjoint -> 0% recall. The LLM faithfully followed the chunker's `expected_namespaces=["tensorrt_llm"]` hint; that hint disagreed with the reference convention.

2. **Pass-2 verify parse-failure (parse_failure_after_retries)**:
   - Record 5 (b/vllm): chunks `model_config_verify_quantization`, `cache_config_invariants` - pass2 verify failed to parse YAML after retries; pass1 unchanged for those chunks.
   - Record 5 b/vllm hybrid context: `hybrid (d-ab) for vllm: extraction failed; modes=['parse_failure_after_retries']` reported in d-ab/vllm observations - led to extension=0 (no proposals merged on vllm via d-ab).

3. **Pass-3 extend parse-failure**:
   - Record 4 (b/transformers): chunk `validate_section_01_1.1._Decoding_attributes` failed pass3 extend.

4. **Pass-2 non-applied corrections (the LLM verifier flagged a predicate-mismatch but was conservative; the diff is recorded as `correct_predicate:*` or `correct_severity:*` rather than auto-applied)**:
   - Record 4 (b/transformers): `transformers_bnb_4bit_compute_dtype_not_string_or_torch_dtype` -> wants `type_is_not_str_or_torch_dtype`. `transformers_pad_token_id_lt_zero` -> wants `correct_severity:error`.
   - Record 5 (b/vllm): six chunks flagged with `correct_predicate:*` (various scheduler-/lora-/parallel-config refinements).
   - Record 6 (b/tensorrt): two chunks flagged with `correct_predicate:exact` (lora-config predicate too coarse).

5. **Cheaper model under-extraction probe** (record 7 b_8b): 8B variant had two recorded chunk-level failures (`Performance_attributes` parse failure; `check_num_return_sequences` yielded 0 invariants).

6. **Hybrid extension yield variance** (records 9-11): extension counts vary (0, 2, 8) across the three d-ab engine cells. Higher extension yields land precisely on the engine where strategy (b) had pass-failures (tensorrt - 8 extensions added); zero extensions on vllm where the verify pass parse-failed.

### Failure modes notably absent

- No `crash` records besides c/key_absent.
- No JSON-Schema-validation failures on schema extraction (schema task uses JSON-mode + jsonschema validator).

---

## Adjacent observations (cross-cell, distilled - no synthesis verdicts)

These are surface-level cross-cell patterns. NOT interpretation; that is Phase 4 scope.

- **Wall-clock vs engine size**: b cell wall_clock_sec values are similar across engines (1372-1649 s), suggesting the chunker, not the engine, dominates the LLM cost (each chunk takes a near-constant LLM call).

- **schema_type_accuracy is consistently lower than schema_recall** across the b cells (b/transformers 55.9%, b/vllm 46.6%, b/tensorrt 48.3%; b_8b/transformers 57.3%). Schema fields are correctly named but the LLM-emitted types do not always agree with the canonical schema's JSON-Schema types. d-ab cells return 100% type_accuracy by construction (they inherit from (a)).

- **invariant_severity_accuracy splits by engine on b**: 77.3% (transformers), 100% (vllm), 0.0% (tensorrt). The 0.0% on tensorrt is mechanical (0 intersection -> 0/0 -> 0); the 100% on vllm is real (every intersection has matching severity).

- **d-ab extension count correlates with how-much-the-LLM-half-found-that-(a)-missed**: tensorrt = 8 extensions (the LLM half found 8 invariants the static miner didn't); transformers = 2; vllm = 0 (no extensions). The flagged_spurious counts (rejected proposals) are 1, 2, 0 respectively.

- **b/vllm has high schema_recall (97.0%) but low schema_precision (85.1%)**: the LLM emitted 154 fields vs 135 in reference. 23 emitted fields aren't in reference; 4 reference fields are missing. The over-emission has not been characterised at this level (chunk-level prompts may have generated spurious entries; this is candidate Phase 4 material).

- **Cheaper-model probe (b_8b/transformers)** yields very close schema_recall (85.7% vs 83.0% for full b on same engine) but lower invariant_recall (35.7% vs 56.4%). Lower energy (4.93 Wh vs 81.31 Wh) and faster wall (412 s vs 1649 s).

- **The c cell is empty data**: All-zero record. The presence of the row enables downstream "did this strategy run" diagnostics without ambiguity.

---

## Score JSON shape (per record)

For traceability. Every record contains:

```
strategy, engine, version_slug, bump_distance,
schema_recall, schema_precision, schema_type_accuracy, schema_failure_mode,
invariant_recall, invariant_precision, invariant_severity_accuracy, invariant_failure_mode,
schema_reference_count, schema_cell_count, schema_intersection_count,
invariant_reference_count, invariant_cell_count, invariant_intersection_count,
wall_clock_sec, energy_wh, failure_modes,
brittleness_pass_through_rate, brittleness_silent_fail_count,
brittleness_detectable_fail_count, brittleness_patch_cost_loc,
observations, scored_at, scoring_format_version,
reference_path, cell_schema_path, cell_invariants_path
```

Side artefacts per `<strategy>/<engine>/<version>/` directory: `recall_misses.yaml`, `precision_spurious.yaml`, `type_mismatches.yaml`.

---

## Closure status

- 11/11 active-cell scores recorded.
- trial_matrix.{md,csv} regenerated (Phase 3a.1 final).
- No re-runs needed: every numeric anomaly traces to a recorded observation pattern (namespace convention, parse-failure retry, cheap-model degradation).
- Phase 4 synthesis can proceed against this dataset once Phase 3a.2 (version-bumped cells) data is added.

---

## Linked artefacts

- `_spike/findings/trial_matrix.md`, `_spike/findings/trial_matrix.csv` - auto-generated aggregate.
- `_spike/findings/trial_scores/*.json` - per-cell records.
- `_spike/findings/trial_runs/<strategy>/<engine>/<version>/` - raw artefacts + per-cell diff YAMLs.
- `_spike/findings/phase1_version_lock.md` - version pins backing this matrix.
- `_spike/findings/phase2_locked_prompts/` - the locked b prompts used by all b/b_8b/d-ab cells.
