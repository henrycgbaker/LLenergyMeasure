# Bake-off A: Refactor analysis of the current handwritten machinery

**Date:** 2026-05-24
**Scope:** Read schema introspector + 2 invariant miners for transformers v4_57_3, plus shared layer (`_base.py`, `_common.py`); identify whether the "ballooned" feel is genuine engine-surface complexity or accidental complexity that a refactor could collapse.
**Verdict (TL;DR):** Mostly **accidental** complexity. A focused refactor could plausibly take the per-engine miner footprint from ~3800 LoC to ~1500 LoC without losing fidelity. The cap on collapse is the runtime probing logic (which IS doing something the source can't tell you alone) — about 800 LoC of essential complexity. The other ~3000 LoC is duplication, premature-narrow abstractions, and bespoke inference logic that a shared layer could carry.

---

## What's there today

| File | LoC | Role |
|---|---|---|
| `scripts/engine_producers/_base.py` | 665 | Shared AST primitives + 5 detector classes (`ConditionalRaiseDetector`, `ConditionalSelfAssignDetector`, `ConditionalWarningsWarnDetector`, `ConditionalLoggerWarningDetector`, …) + `InvariantCandidate` dataclass. |
| `scripts/engine_producers/_common.py` | 1449 | Schema envelope helpers (canonical JSON Schema 2020-12, $defs propagation, dataclass walking, sphinx-kwargs parsing). NOT used by invariant miners. |
| `scripts/engine_producers/_pydantic_lift.py` | 271 | Pydantic-class → JSON Schema lift. |
| `scripts/engine_producers/_msgspec_lift.py` | 296 | msgspec-class → JSON Schema lift. |
| `scripts/engine_producers/_dataclass_lift.py` | 157 | stdlib-dataclass → JSON Schema lift. |
| `engine_versions/transformers/v4_57_3/producers/schema_introspector.py` | 294 | Calls into `_common`. Per-engine LANDMARKS + call orchestration. Largely thin. |
| `engine_versions/transformers/v4_57_3/producers/static_invariant_miner.py` | **1599** | AST walks `GenerationConfig.validate()`. Defines its own `FieldPredicate`, `DetectedBody`, `_detect_raise`, `_detect_self_assign`, etc. — **parallel** to `_base.py` types. |
| `engine_versions/transformers/v4_57_3/producers/dynamic_invariant_miner.py` | **1880** | Combinatorial probing of `GenerationConfig(**kwargs).validate(strict=True)` + 4 separate predicate-inference passes (divisibility, comparison, threshold, type allowlist). |

**Total per-engine machinery (transformers v4_57_3): ~3800 LoC.** Shared layer: ~2800 LoC (some of which is for schema discovery, not invariants).

For comparison: the **ground-truth output** is `invariants.proposed.yaml` at 1303 lines (mostly data, not logic).

---

## Why it ballooned — three concrete sources of accidental complexity

### 1. Parallel detector hierarchy in static miner (~400 LoC saveable)

`_base.py` defines `DetectedPattern`:

```python
@dataclass
class DetectedPattern:
    severity: Severity                # error / warn / dormant
    emission_channel: EmissionChannel
    affected_field: str | None
    message_template: str | None
    detail: str
```

Plus 5 detector classes that each produce one `DetectedPattern` per matching AST node.

The static invariant miner **does not use these**. It defines a parallel `DetectedBody` type (richer: also carries `_detect_raise`-class info + ExtractedCondition) and 6 parallel `_detect_*` functions (`_detect_raise`, `_detect_assert`, `_detect_warnings_warn`, `_detect_logger_warning`, `_detect_minor_issues`, `_detect_self_assign`).

The comment in `static_invariant_miner.py:82-94` explains: "the base detectors emit `DetectedPattern` which carries severity / channel / affected_field but not the structured `FieldPredicate` data we need for cross-field corpus invariants … Extending the base classes would either change their public `DetectedPattern` shape (breaking the introspection extractor that currently consumes it) or require lossy adapter shims at every emission site. With one miner live today, the cheaper choice is per-miner detectors."

This is a self-aware accidental-complexity admission. The fix is straightforward:

- Replace `DetectedPattern` with a discriminated-union shape (Pydantic / dataclass with `kind: Literal[...]` + per-kind payload field). Detector classes return the richer payload natively.
- The "introspection extractor" mentioned as a consumer is one place; updating it is the cost of the refactor.

**Savings: ~400 LoC** (the parallel detector functions in static miner collapse to base-class subclass overrides).

### 2. Predicate-extraction logic is bespoke per miner, but predicates are generic (~600 LoC saveable)

Static miner has `_extract_compare`, `_extract_call_predicate`, `_isinstance_type_names`, `_extract_unary_not`, `extract_predicates`, `negate_predicates` (lines 321-611). All of these inspect AST condition nodes and produce `FieldPredicate` records.

Dynamic miner has `_infer_divisibility`, `_infer_cross_field_comparison`, `_infer_single_field_threshold`, `_infer_type_allowlist` (lines 747-948). All of these inspect probe-result tables and produce predicates.

**Both produce the same output type (predicates)** but from different inputs (AST vs probe-result tables). Today they're entirely separate.

The right abstraction: a `Predicate` type (kind + operands + canonical-form) that both miners construct, plus a shared `PredicateEvaluator` that can:
- Render predicate → AST (for static comparison against extracted conditions)
- Render predicate → probe gate function (for dynamic combinatorial checks)
- Render predicate → YAML match block (for corpus emission)

With this in place:
- Static miner's "extract from condition" reduces to "pattern-match the AST against the registered predicate kinds"
- Dynamic miner's "infer from probe table" reduces to "for each predicate kind, ask the evaluator: does this predicate explain the error rows?"

**Savings: ~600 LoC.** The four `_infer_*` functions in dynamic + the four `_extract_*` functions in static become ~200 LoC of predicate registry instead of ~800 LoC of bespoke logic.

### 3. Dynamic miner's combinatorial-probe scaffolding is engine-shaped (~200 LoC saveable; the rest is essential)

Dynamic miner has `_Cluster` (manually defined groups of related kwargs: beam-search, watermarking, compile-config, …), `_synthesise_probe_value`, hardcoded probe values per cluster (`_watermarking_probe_values`, `_compile_config_probe_values`), `_run_cluster_probes`, `_run_cartesian` (~500 LoC for this scaffolding alone).

The clusters and probe values ARE engine-specific knowledge (you can't know that `num_beams + num_beam_groups + diversity_penalty` belong together by inspecting source alone — or can you? An LLM could). But the CARTESIAN-PRODUCT MACHINERY is generic. Today it's inlined in this one file.

**Savings: ~200 LoC.** Move `_run_cartesian` + `_split_error_rows` + `_check_predicate_explains_errors` to `_base.py`. Clusters stay per-engine (they're knowledge).

### Net potential refactor

| | Current | Post-refactor | Saving |
|---|---|---|---|
| `_base.py` | 665 | ~900 (absorbed shared logic) | -235 |
| `static_invariant_miner.py` (transformers) | 1599 | ~700 | +900 |
| `dynamic_invariant_miner.py` (transformers) | 1880 | ~1100 | +780 |
| Per-engine total (transformers) | 3480 | 1800 | **+1680 LoC saved** |

Critically, when a second engine (vllm, tensorrt) lands invariant miners, the savings multiply — most of the predicate / cartesian / detector logic is shared, so the second engine's miners would be much thinner.

---

## What WON'T shrink no matter how clever the refactor

About 800 LoC of essential complexity that has to live somewhere:

1. **Probe-value synthesis** for non-trivial types (CompileConfig, watermarking config, etc.). Generating "an instance that triggers this clause" requires engine knowledge.
2. **Message-template extraction** with substitution markers (`{X}` for `self.X`, etc.). Required for the validation CI to substring-match emitted messages against live library messages.
3. **Landmark / probe / fail-loud machinery**. The `MinerLandmarkMissingError` + probe-fingerprint + per-version vendoring is load-bearing for Renovate-driven version bumps. ~200 LoC of necessary infrastructure.
4. **Conflict detection** between static and dynamic outputs. Both can emit the "same" invariant from different angles; deduplication is real work.

---

## How LLM-driven extraction would look against the same target

For invariant mining specifically (which is where the bulk of LoC lives):

A single LLM call given:
- `inspect.getsource(GenerationConfig.validate)` (≤ 2k tokens for transformers v4.57.3)
- The canonical invariant-YAML output schema (one entry shape)
- A few-shot prompt with 2-3 examples from `invariants.proposed.yaml`

would produce a candidate list of invariants directly. No predicate-AST-pattern-matching, no combinatorial probing, no inference passes.

**Code footprint** for this approach: ~150-200 LoC orchestration (fetch source, build prompt, call API/model, parse YAML, validate shape). Plus the runtime verification harness that re-checks each emitted invariant by actually invoking `validate(strict=True)` — which we'd want anyway as a quality gate. That's another ~200 LoC.

**Total LLM-based**: ~400 LoC.
**Current handwritten**: ~3800 LoC.
**Refactored handwritten** (per analysis above): ~1800 LoC.

So the comparison the user is asking us to do is:
- Refactored handwritten (1800 LoC) vs LLM (400 LoC) — if quality is comparable, LLM wins on maintenance.
- Refactored handwritten (1800 LoC) vs LLM (400 LoC) — if LLM quality is much worse, handwritten still wins.
- **Current** handwritten (3800 LoC) vs LLM (400 LoC) — the gap is so big that even mediocre LLM quality probably wins on net.

---

## What the bake-off (B + C) needs to prove

For LLM-driven to be a real pivot rather than a research aspiration, the bake-off needs to show:

1. **Coverage**: >80% of ground-truth invariants present in LLM output (with field-level matching, not surface-string matching).
2. **Precision**: <20% spurious / hallucinated invariants (where "spurious" = LLM emits a predicate that doesn't actually fire in live `validate(strict=True)`).
3. **Type-structure fidelity**: LLM produces the right `severity` (error/warn/dormant), the right `affected_field`, the right `expected_outcome.outcome`. These map directly to llem-side gate behaviour, so getting them wrong is functional regression.
4. **Version-bump robustness**: re-run on transformers v5.x (where handwritten walkers may break on rename / refactor); LLM should degrade gracefully.

If (1) + (2) hit, the refactor case becomes "skip the handwritten refactor; pivot to LLM-driven now."

If only (1) hits (high recall, high false-positive rate), LLM is a coverage backstop layered ON the refactored machinery, not a replacement.

If neither hits, the refactor case stands on its own merits regardless of LLM availability.

---

## Recommendation to synthesis layer

**Don't invest engineering time in the handwritten-refactor (1800 LoC target) until the bake-off settles**:

- If LLM passes the bar → the refactor is wasted work; the LLM replaces the machinery wholesale.
- If LLM fails → the refactor is straightforward and worth doing (the analysis above shows the savings are real).

The bake-off cost (a few hours of Sonnet + LLM tokens) is much smaller than the refactor cost (multi-day work on 3800 LoC of intricate code). The cheap experiment runs first.

**Single recommendation post-bake-off**: pick ONE of the three paths (current / refactored / LLM) and commit. Don't split investment.

---

## Auxiliary observations (not core to the decision)

- `_common.py` (1449 LoC) is doing a lot for schema discovery; the recent #671 work made it dataclass-aware. It's a tractable size for its scope.
- The per-engine schema introspector (294 LoC for transformers) is the leanest part of the system. Most of its lines are LANDMARK declarations + the new #671 nested-dataclass walker — both are engine knowledge, not boilerplate. Whatever decision the bake-off drives, the schema introspector is NOT a top refactor target.
- The `_pydantic_lift` / `_msgspec_lift` / `_dataclass_lift` trio (724 LoC combined) exists to teach the schema walker about three "introspectable surface" flavours. This is a real domain abstraction — engines model their config differently. An LLM walker would skip all of this (it just reads the source). Counts as another 700 LoC of potential savings in the LLM pivot scenario.
