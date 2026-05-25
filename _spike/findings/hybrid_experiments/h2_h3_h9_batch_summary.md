# Phase 3b cheap-patterns batch: H2 + H3 + H9

**Pattern bundle:** Three patterns that play to the LLM diagnostic
strength established by H4: each LLM call READS rather than SYNTHESISES.
H2 validates (a); H3 verifies (b) via runtime / schema gate; H9
diagnoses (a)'s gaps without proposing patches.

**Backend:** container Ollama @ port 11435, model `llama3.1:70b`
(q4_K_M), num_ctx=32768, temperature=0.

**Cells:** 3 engines x 3 patterns. H2 + H9 use one LLM call per cell
(6 calls total); H3 uses zero LLM calls (it reuses existing (b)
extraction and applies a deterministic gate).

**Wall-clock:** H2 (132.6 + 101.9 + 112.4) + H9 (29.7 + 55.4 + 37.9)
+ H3 (2.2 + 0.0 + 0.0) = 472.1 s LLM time total. Well under the
45-90 min budget projected.

## Per-pattern aggregate results

### H2 (LLM validates (a))

| Engine | Entries | CONFIRM | UNCERTAIN | SUSPECT-SPURIOUS | Drop rate | Recall vs ref |
|---|---:|---:|---:|---:|---:|---:|
| transformers v4.57.3 | 41 | 24 | 17 | 0 | 0% | 1.000 |
| vllm v0.7.3 | 26 | 23 | 0 | 3 | 11.5% | 0.885 |
| tensorrt v0.21.0 | 35 | 27 | 8 | 0 | 0% | 1.000 |

**Key finding**: the LLM was conservative on dropping - it kept the
"prefer UNCERTAIN over SUSPECT-SPURIOUS" guidance and only dropped
when it had a specific reason. Drop rate was 0 on transformers +
tensorrt and 11.5% on vllm.

The 3 vllm drops were ALL FALSE-DROPS. The LLM's reasoning for each:

| Dropped ID | LLM reason | Source check |
|---|---|---|
| vllm_samplingparams_dormant_seed_eq_neg1 | "seed is not set to -1 in the source" | WRONG: `sampling_params.py` has `if self.seed == -1: self.seed = None` |
| vllm_loraconfig_dormant_max_cpu_loras_unset | "max_cpu_loras is not set to None in the source" | WRONG: `config.py` has `if self.max_cpu_loras is None: self.max_cpu_loras = ...` |
| vllm_promptadapterconfig_dormant_max_cpu_prompt_adapters_unset | Same pattern | WRONG: same |

All three are dormant-normalisation-pattern entries the (a) walker is
already capable of surfacing (severity=dormant), and the LLM
misclassified them as spurious. Pattern: LLM struggles with dormant
patterns where "is the value X" predicates feel less concrete than
"raises when X" predicates.

**Precision lift from H2**: NONE for transformers + tensorrt (drops 0).
For vllm, the 3 drops were false-drops so precision goes from 1.0 to
1.0 (within rounding; trial recall metric measures recall against (a)
itself, so drops that are correct also reduce recall by the same
amount).

H2 verdict: LLM-as-validator is OPERATIONAL but not RIGOROUS at 70B-q4.
Conservative prompt phrasing successfully prevented mass-drops, but
the few drops emitted were errors. The validator approach would need
a stronger model OR a deterministic second-opinion stage to be safe
for production.

### H3 (LLM proposes -> (a) runtime/schema verifies)

| Engine | Gate | (b) emitted | Verified | Dropped | Recall (b -> verified) | Precision (b -> verified) | Precision lift |
|---|---|---:|---:|---:|---|---|---:|
| transformers v4.57.3 | runtime | 51 | 39 | 12 | 0.564 -> 0.487 | 0.431 -> 0.487 | **+5.6%** |
| vllm v0.7.3 | schema-existence | 66 | 62 | 4 | 0.385 -> 0.385 | 0.152 -> 0.161 | +1.5% |
| tensorrt v0.21.0 | schema-existence | 39 | 39 | 0 | 0.258 -> 0.258 | 0.205 -> 0.205 | 0% |

**Key finding**: the verification gate's strength is asymmetric across
engines.

For transformers (runtime gate via `runtime_validate_invariants`):
- 12 entries dropped (23.5% of (b) entries).
- Precision recovers from 0.431 to 0.487 (+5.6%).
- Recall drops from 0.564 to 0.487 (-7.7%) - 3 of the 12 dropped
  entries WERE in the reference (their positive case didn't trigger at
  runtime - which is the gate doing its job, but at the cost of
  reference-confirmed entries).
- This is the substantive gate. It IS effective at catching
  hallucinations but also catches real-but-runtime-asymmetric cases.

For vllm + tensorrt (schema-existence fallback via AST class-body +
__init__-param scan):
- vllm: 4 entries dropped (6% of (b)). One was a true-positive drop
  (entries with wrong field names); 3 of the dropped are non-reference
  entries. Precision lift is marginal (+1.5%).
- tensorrt: 0 entries dropped. All (b)-emitted fields exist in the
  declared schema. No precision lift.

**Why the asymmetry?** The schema-existence gate only fails when an
LLM completely fabricates a field name that doesn't exist in the
class. (b)'s LLM is good enough at staying within the engine's
declared field set that schema-existence is too weak a gate to catch
the actual hallucination patterns (which are more about wrong
predicates and wrong severities on REAL fields).

**Tensorrt v0.21.0 hallucination pattern**: NOT observed at the active
version. The H4 summary mentioned "tensorrt b's HF GenerationConfig
hallucinations" as a known failure mode; this batch did not reproduce
that at v0.21.0 (b)'s output - all native_types are tensorrt-local. The
hallucination is a bumped-version artefact, not an active-version one.

H3 verdict: runtime gate IS the right shape for verification, but
transformers-only currently. Schema-existence is a useful sanity check
(catches blatant fabrications) but not a precision-lifter for active
cells. **Recommendation for Phase 4**: extend runtime validation to
vllm + tensorrt via their per-engine containers (Phase 2.5 deferred
work).

### H9 (LLM diagnoses, no output mutation)

| Engine | Diagnoses | Already-known | New-here | Wall |
|---|---:|---:|---:|---:|
| transformers v4.57.3 | 1 | 1 | 0 | 29.7s |
| vllm v0.7.3 | 4 | 3 | 1 | 55.4s |
| tensorrt v0.21.0 | 3 | 2 | 1 | 37.9s |
| **Aggregate** | **8** | **6** | **2** | **123s** |

**Key finding**: 6/8 diagnoses re-confirm the `post_trial_a_gap_closure.md`
inventory's structural categories with correct example_field citations.
2/8 diagnoses are genuinely new (one operationally useful - SamplingParams
branch-descent in vllm - and one minor - tensorrt model_config
arbitrary_types_allowed).

**Zero false-positives**: no fabricated gap categories; no
example_field claims that don't exist in the engine.

**Cross-engine correlation with H4**: H9's diagnoses match H4's
diagnoses on the 6 overlapping inventory entries. H9 surfaces 2
additional entries H4 did not (because H4's prompt was tighter to
inventory-listed gaps; H9 had a broader category palette).

H9 verdict: cheapest-effective pattern of the three. Per-engine cost
~50s LLM wall. Output is structured-categorical-with-example-fields,
which is ready input for spike-refactor backlog filing.

## H4 vs H9 correlation

| Inventory gap | H4 diagnosed? | H9 diagnosed? | H9 marked as | Match? |
|---|:-:|:-:|---|:-:|
| G-trf-1 (defensive imports) | yes | yes | yes-already-known | yes |
| G-vllm-1 (normalisation-only) | yes | yes | yes-already-known | yes |
| G-vllm-2 (local-var-alias) | yes | yes | yes-already-known | yes |
| G-vllm-3 (branch-descent) | yes | yes | yes-already-known | yes |
| G-trt-1 (type-blindness) | yes | yes | yes-already-known | yes |
| G-trt-3 (nested-config) | yes | yes | yes-already-known | yes |

All 6 inventory gaps surfaced by H4 are independently re-surfaced by
H9. H9 adds 2 new gap instances (N1: vllm SamplingParams branch-descent;
N2: tensorrt Pydantic config arbitrary_types_allowed branch-descent).

## What this batch establishes

1. **LLM diagnostic intelligence is robust and reproducible**. H4 and
   H9 independently surface the same gap categories with the same
   example fields. The 70B-q4 model is reliable for STRUCTURED-CATEGORICAL
   output when the prompt locks the category vocabulary.

2. **LLM validation intelligence is OPERATIONAL but not RIGOROUS**.
   H2 produced 3 false-DROPS on vllm (the entries it dropped were
   actually present in the source). Conservative prompting helped but
   didn't eliminate the failure mode. Recommendation: H2-style
   subtractive validation needs a stronger model (claude-opus or
   larger) OR a second-opinion deterministic check.

3. **Runtime-gate is the right shape; schema-existence is too weak**.
   H3 transformers achieved +5.6% precision lift with a real runtime
   gate. H3 vllm/tensorrt with schema-existence achieved <2% lift.
   Phase 4 should extend runtime validation per-engine.

4. **Tensorrt v0.21.0 (b) has NO active-version hallucinations**. The
   hallucination failure mode reported by H4 is a bumped-version
   artefact, not present at the active row. Active-row schema is
   too small to leverage the schema-existence gate meaningfully.

5. **Cheap patterns are CHEAP**. The full 9-cell batch ran in ~8
   minutes wall-clock (LLM time). Diagnoses-only (H9) is the
   cheapest-per-bit-of-information pattern explored so far.

## Implications for Phase 4 synthesis

H2 + H3 + H9 + H4 together establish a clear LLM-role split:

| LLM role | Pattern | Quality | Cost | Phase 4 viability |
|---|---|---|---|---|
| diagnose-only | H4 (text) + H9 (categorical) | excellent | low | strong - production-ready diagnostic accelerator |
| validate / subtract | H2 | inconsistent at q4 | medium | weak - needs stronger model or deterministic second-opinion |
| extend / propose | (b) + H3 verify | mixed | high (per-cell) | mixed - depends on runtime-gate availability |
| modify-miner / synthesise patches | H4 (patches) | poor (anchors broken, helpers undefined) | low | weak - useful as diagnosis spawner, not patches |

The natural Phase 4 ship-with-defaults shape: (a) deterministic +
LLM-diagnose-on-top + runtime-verify-where-available. Validation
(H2-style subtractive) is NOT viable for production with 70B-q4.

## Recommendation for next Phase 3b pattern

Given this batch's findings, the highest-value Phase 3b investments
are:

1. **H7 (agentic loop with tool use)** - the cheap-batch confirms
   single-shot patterns hit a ceiling. Iterative tool-use gives the
   LLM the closed-loop feedback H4 lacked. May convert poor patch-
   synthesis into competent patch-synthesis.

2. **Tier 2 chunking ablations (H5 + H6)** - directly test whether
   chunking is the (b) recall ceiling. If yes-per-validator beats
   per-class, Phase 4 should change (b)'s chunking strategy. Cheap to
   run; high-info.

3. **H8 (parallel reconcile)** - LLM reads BOTH (a) and (b) outputs +
   source; produces reconciled union. Combines diagnostic + validation
   pressure in one prompt.

H7 + H5/H6 + H8 (4 patterns, ~12 cells, ~6-8 hours LLM serial) would
materially close the Phase 3b exploration.

The 4 patterns NOT to invest more in: H1 (already covered by d-ab),
H2 (model limitation; needs stronger), H4 (patch-synthesis weak), H9
(harvested; further engines = diminishing returns since gap categories
recur).

## Artefacts

Per-cell:
- H2: `hybrid_experiments/h2_validate/<engine>/{prompt.md,raw_response.txt,classifications.json,filtered_proposed.yaml,score.json}`
- H3: `hybrid_experiments/h3_propose_verify/<engine>/{verified_invariants.yaml,dropped_entries.yaml,score.json,observations.md}`
- H9: `hybrid_experiments/h9_diagnose/<engine>/{prompt.md,raw_response.txt,diagnoses.yaml,observations.md}`

Cross-pattern: this file.

Aggregate machine-readable:
`hybrid_experiments/h2_h3_h9_aggregate.json` (list of per-cell score
dicts; one entry per pattern x engine).

H9 cross-engine: `hybrid_experiments/h9_diagnose/h9_cross_engine_summary.md`.

Runner: `_spike/scripts/strategies/run_h2_h3_h9.py`.
