# Mining strategy bake-off — open design questions

**Status:** In flight (Bake-offs A done; B + D running; C deferred).
**Opened:** 2026-05-24 (end-of-session-3, after #671 landed).
**Cross-ref:** `research/mining-substrate-trial/DECISIONS_LOG.md` (chronological log), `.product/designs/engine-knowledge-as-data.md` § Open Questions 10 + 11, `research/mining-substrate-trial/findings/bakeoff_A_refactor_analysis.md` (Bake-off A artefact).

This document holds the OPEN design questions about engine knowledge mining — how it should be implemented, what the right substrate is, and how we'd know. Decisions are NOT made here; this is the framing + the experiment design that the bake-off results will resolve.

---

## The question that prompted this

After Phase 3 landed (introspection.py rewrite + engine_configs.py deletion + #671 nested-dataclass walker), an honest re-read of the producer machinery surfaced:

> "I thought the introspectors were elegant and small recursive walkers, but they seem to have ballooned?"

Numbers:
- Schema introspector (transformers v4_57_3): 294 LoC. Tractable.
- Static invariant miner: **1599 LoC**.
- Dynamic invariant miner: **1880 LoC**.
- Shared layer: `_base.py` (665) + `_common.py` (1449) + 3 framework lifts (724 combined) = **2838 LoC**.
- **Total per-engine machinery: ~3800 LoC; shared: ~2800 LoC.** Output: a 1303-line YAML + a 629-line JSON.

The ratio of mining-machinery code to mining-output data is ~3:1, where the OUTPUT is essentially "what GenerationConfig.validate() does." A senior engineer reading that source could write the output directly.

This triggered a deeper question: **are we building infrastructure to laboriously re-derive something an LLM (or human) could read off the source in one pass?**

---

## Sub-questions to settle empirically

### Q1: Is the current machinery's size genuine complexity or accidental?

If accidental, the right move is refactor it down. If genuine, the size is the cost of correctness and we just keep it.

**Bake-off A** (`research/mining-substrate-trial/findings/bakeoff_A_refactor_analysis.md`) answers this. Headline verdict: **mostly accidental**.

- ~400 LoC of parallel detector functions in static miner duplicating `_base.py` (the miner self-admittedly forked rather than extending — known accidental complexity).
- ~600 LoC of bespoke predicate logic split between static AST extraction and dynamic probe inference, where both produce the same `Predicate` type from different inputs (needs an abstraction).
- ~200 LoC of cartesian-probe scaffolding in dynamic miner that should live in `_base.py`.

Net: refactor target ~1800 LoC (down from 3800). Roughly halves the per-engine cost.

**Remaining essential complexity (~800 LoC)**: probe-value synthesis for non-trivial types, message-template extraction, landmark / probe machinery, conflict detection between static and dynamic outputs.

### Q2: Can an OSS LLM produce equivalent mining output from source?

If yes at sufficient quality, the deterministic machinery may be replaceable. If no, the machinery's size is the price of correctness.

**Bake-off B** is testing this with Ollama / llama3.1:70b against the transformers v4_57_3 ground truth.

What "sufficient quality" means here:
- Coverage: ≥80% of ground-truth fields / invariants present in LLM output.
- Precision: <20% spurious / hallucinated entries.
- Type-structure fidelity: right severity, right affected_field, right expected_outcome.
- Wall-clock + cost per run.

Result pending. Document outcome at `research/mining-substrate-trial/findings/bakeoff_B_local_llm.md` when sonnet B reports.

### Q3: Does the current machinery survive engine version bumps?

If it transfers cleanly across bumps, it's a stable foundation worth keeping. If every bump requires per-version code changes, the maintenance cost is high enough that LLM-extraction's "regenerate per bump" approach might be cheaper.

**Bake-off D** is testing this by running the v4_57_3 producers against a newer transformers version (5.x if available) without modifying the producer code.

Measurements:
- Landmark resolution rate (does the entry-point class / method still exist?).
- Output volume vs ground truth (field count, invariant count delta).
- Categorical failure modes: silent miss vs noisy break. Noisy break is GOOD (we detect it); silent miss is BAD (degradation we don't see).
- Counterfactual patch size: if we'd had to update the producer for this bump, how many LoC?

Result pending. Document outcome at `research/mining-substrate-trial/findings/bakeoff_D_version_robustness.md` when sonnet D reports.

### Q4 (deferred): Does Claude API beat OSS LLM at the same task?

Test C is the same as B but with Claude as the model. Decision-relevant if (B) underperforms but cost is acceptable; less relevant if (B) is already good enough.

Deferred pending `ANTHROPIC_API_KEY` setup. Will be picked up if B + D's combined verdict makes it worth running.

---

## The 4-quadrant decision matrix

Given the answers to Q1-Q3, the synthesis question becomes which architecture to invest in. Four mutually-exclusive outcomes:

```
                        Q3 (version robust?)
                       Low                   High
              +-----------------+-------------------+
       Low    | Refactor          | Refactor +        |
              | (~1800 LoC) +     | nothing else;     |
   Q2 (LLM    | bespoke each bump | machinery is fine |
  good?)      |                   | (just refactor)   |
              +-----------------+-------------------+
       High   | LLM replaces      | Belt + braces:    |
              | (machinery        | deterministic     |
              | obsolete; OSS LLM | floor + LLM       |
              | is the substrate) | verify/extend     |
              +-----------------+-------------------+
```

| Cell | Architecture | LoC estimate |
|---|---|---|
| Low-Low | Refactor + per-bump maintenance | 1800 + N×(bump-patch) per engine |
| Low-High | Refactor only | 1800 per engine |
| High-Low | LLM replaces | ~400 per engine + LLM cost-per-run |
| **High-High** | **Belt + braces (the user's synthesis)** | 1800 + ~400 LLM verifier + LLM cost-per-run |

### Why the High-High quadrant is special

If both (Q2 = LLM good) AND (Q3 = deterministic robust to bumps), the natural architecture is NOT to pick one. It's:

1. **Deterministic floor** runs first per engine (refactored, ~1800 LoC per engine, with high reuse across versions per Q3).
2. **LLM verifier pass** reads the deterministic output AND the source; flags any mismatch as "deterministic miss" candidate. Also reads source independently and proposes new entries the deterministic miner didn't surface.
3. **Maintainer review** for LLM-only additions before promotion to `validated.yaml`. LLM-only never lands without human sign-off (consistent with current overlay-completion model).

Belt-and-braces properties:
- **Stability when deterministic agrees**: most common case; LLM provides confidence signal but doesn't add work.
- **Coverage when deterministic misses**: LLM acts as backstop for edge cases (idioms not in walker grammar, fields documented in prose, version-specific semantics walker missed).
- **Drift detection on version bumps**: when LLM finds things deterministic missed, AND those things are in the new upstream version, that signals a walker gap to patch — surfaces the maintenance cost as a concrete actionable diff.
- **Cost**: LLM call per (engine, version, miner-output) — bounded; cacheable by content hash of source.

The trade-off is integration complexity (running both, reconciling outputs) vs the single-substrate options. Worth it ONLY if both Q2 and Q3 land "good".

### What each quadrant means for the project trajectory

- **High-High (belt + braces)**: invest in the refactor AND the LLM pipeline. Best-quality outcome; highest integration complexity.
- **Low-High (refactor only)**: invest in the refactor; LLM is wasted effort. Risk: we miss coverage that LLM could provide.
- **High-Low (LLM replaces)**: scrap the current machinery; build LLM-only pipeline. Lowest LoC; highest dependency on LLM quality + availability.
- **Low-Low (worst case)**: refactor doesn't materially help AND LLM doesn't either. Means deterministic miners are version-specific snowflakes. Implies the engine surface is fundamentally too irregular for either substrate; pure judgement / curation may be the only path.

---

## What's NOT in scope of the bake-off

- Schema introspection (vs invariant mining). Schema discovery is the leaner part of the machinery (~294 LoC for transformers's introspector); refactor savings would be modest. The bake-off targets BOTH but the heavier weight goes to invariants because that's where the LoC + accidental-complexity sits.
- Other engines (vllm, tensorrt). One-engine, two-knowledge-category bake-off. Generalisation is a downstream decision per the per-engine maturity of each.
- Production hardening of any LLM pipeline. The bake-off measures viability, not the engineering it would take to make a pipeline production-grade (rate limiting, retry, output validation, prompt versioning, etc.).
- Cost optimisation (prompt caching, model selection beyond the two we're testing). Out of scope for the v0 experiment.

---

## What the synthesis will look like

When B + D report, I (or the human reviewer) will:

1. Read both reports for raw numbers.
2. Slot the result into the 4-quadrant matrix.
3. Recommend an architecture path with explicit defence: "(B) hit X / (D) hit Y, so quadrant Z, so we pick architecture W."
4. If C is decision-relevant (boundary case where Claude-vs-llama matters), revive C.
5. Update this doc with the synthesis section and update the design doc's Open Question 11 status.

---

## Where adjacent decisions sit

- **Open Question 10** (`.product/designs/engine-knowledge-as-data.md` line ~505): mining-as-SSOT vs mining-as-evidence-for-curation. The architectural framing this bake-off operationalises. If the bake-off lands in the High-High or High-Low quadrant, that's evidence for "mining-as-evidence" — the LLM (or refactored deterministic + LLM) becomes the curator layer.
- **Future enhancements § LLM-driven introspection** (design doc line ~538): already captured as a direction with reconsider-trigger. The bake-off is the trigger.
- **#540 ($defs propagation)**: implementation complete (commit `06af5fa2`). Belongs to the deterministic-walker-quality story. Doesn't change the bake-off framing.
- **#671 (transformers CompileConfig walker)**: implementation complete (commit `15f34240`). Specific micro-evidence that walker enhancement work has cost but tractable. The bake-off question is whether the AGGREGATE cost is worth it.
- **HarnessConfig vs engine-config separation**: the 7 STAYS_ALLOWLISTED entries (audit) live in HarnessConfig now. Not a bake-off concern; separately settled.

---

## Synthesis (preliminary, single-engine, single-version)

**Bake-off A** (done): refactor target ~1800 LoC (down from 3800). Mostly accidental complexity. Artefact: `bakeoff_A_refactor_analysis.md`.

**Bake-off B** (done; verdict in artefact's report is too pessimistic — see correction below): single-engine, single-version POC against transformers v4_57_3.
- **Schema**: 52.3% recall, 91.8% precision, 50% Jaccard. 35min wall-clock for one prompt call. Misses dominated by deeply-nested fields (BitsAndBytesConfig `bnb_4bit_*`, `llm_int8_*`) and long-signature truncation casualties (`dtype`, `device_map`, `attn_implementation`, `tp_plan`). Spurious rate low (8.2%, mostly internal HF plumbing leaking through).
- **Invariants**: report claims 0% recall but the LLM **actually produced ~20 invariants** in its raw output — the harness failed to parse because the model wrapped the YAML in markdown code fences (```yml ... ```). Once the wrapping is stripped, the extracted invariants look quality-real (e.g. `early_stopping_enum_violation`, `max_new_tokens_range_violation`, `cache_implementation_enum_violation`, `compile_config_type_mismatch`). 24-min call. With prompt iteration (output-format constraint, structured-output mode), realistic recall estimate is 40-60%.
- **Cost per run**: 35min wall-clock total, ~145Wh energy.

**Bake-off D** (done): v4_57_3 producers run against transformers 5.9.0 unchanged. Artefact: `bakeoff_D_version_robustness.md`.
- Schema introspector: 88-96% field overlap (high), BUT 66/67 sampling params lost type spec — silent failure due to `__init__` refactor to `**kwargs`-pop pattern. Field NAMES preserved.
- Static invariant miner: 92% overlap, +16 NEW invariants auto-found in 5.9.0 with zero code changes. Most robust.
- Dynamic invariant miner: 75% overlap, 4 silent misses (probe-value synthesis assumptions invalidated by HF adding type gates).
- Counterfactual patch: ~80-90 LoC across all three producers.
- **Key asymmetry**: static miner failures are detectable (CI re-validates); dynamic miner + schema failures can be silent.

### 4-quadrant placement (preliminary)

| | Q3 (version robust?) Low | Q3 High |
|---|---|---|
| Q2 (LLM good?) Low | (a) Refactor + per-bump bespoke | (b) Refactor only |
| Q2 High | (c) LLM replaces | **(d) Belt + braces** |

Current single-engine POC suggests Q2 = MEDIUM (B's report-claimed "low" overstates the negative — see invariants correction above) and Q3 = MEDIUM-HIGH (D shows tractable per-bump maintenance with detectable+silent failure mix). That places us at the **(b)-(d) boundary**. The user's belt-and-braces synthesis would be optimal IF Q2 lifts to "high" with proper LLM setup.

### What changed in the framing

The single-engine, single-version POCs were sufficient to establish:

1. The current handwritten machinery is mostly refactorable (Q1 / Bake-off A).
2. A single-prompt OSS LLM call with minimal infrastructure hits ~50% recall on both tasks (Q2 / Bake-off B with corrections). **Not zero, not great, room to grow.**
3. The current machinery transfers across version bumps at ~80-95% coverage with visible silent-failure modes (Q3 / Bake-off D).

These results push toward **a larger empirical trial** rather than a binary decision. The user's framing post-bake-off:

> Full empirical trial across all 3 engines, across various version bumps major and minor; 4 strategies (pure mining / pure OSS LLM / pure Claude API / hybrid).

The full-trial roadmap lives in a sibling doc: `research/mining-substrate-trial/findings/empirical_trial_roadmap.md`.

### Decision deferred

No single-architecture decision is made from this preliminary bake-off. Instead: invest in the empirical trial to get decision-quality data across the full matrix.
