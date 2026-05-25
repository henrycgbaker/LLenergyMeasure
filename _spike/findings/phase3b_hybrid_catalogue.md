# Phase 3b hybrid space catalogue

**Status:** Pattern specs for Phase 3b exploratory hybrid experiments.
**Discipline:** Each pattern gets one subagent + 2-3 cells. Log everything
(prompts, outputs, scores, observations). Negative findings welcome.

The catalogue spans four dimensions:

1. **Information flow direction** - (a)->LLM | LLM->(a) | (a)||LLM
2. **LLM read scope** - per-class | per-validator | whole-file | (a)+producer code
3. **Iteration depth** - single-shot | multi-pass | agentic-loop
4. **LLM role** - extend | validate | curate | modify-miner | orchestrate

Patterns selected for diversity across dimensions. Tier 1 = essential.
Tier 2 = high-info. Tier 3 = stretch.

---

## TIER 1: essential (run these first; H4 prioritised first per cross-pollination value)

### Pattern H1: (a) -> LLM extends (current d-ab baseline)

- **Flow**: (a) runs first. LLM reads (a)'s output + engine source via chunker. Proposes additions.
- **Scope**: per-class chunking (existing chunkers).
- **Iteration**: single-shot.
- **Role**: extend.
- **Cells**: transformers v4_57_3 + vllm v0_7_3 + tensorrt v0_21_0 (active row).
- **Files**: `_spike/findings/hybrid_experiments/h1_extend/{prompt, outputs, scores, observations}.md`.
- **Calibration anchor**: already partly run as `d-ab__*__active` cells in Phase 3a.1. Extend with brittleness cell (1 bumped per engine).
- **What we learn**: baseline hybrid signal.

### Pattern H2: (a) -> LLM validates (drops spurious + flags low-confidence)

- **Flow**: (a) runs first. LLM reads (a)'s output + source. For each (a) entry, classifies: confirm / flag-as-likely-spurious / flag-as-uncertain. Output = (a) minus dropped.
- **Scope**: per-class chunking.
- **Iteration**: single-shot.
- **Role**: validate (subtractive).
- **Cells**: transformers v4_57_3 + vllm v0_7_3 + transformers 5.9.0 (bumped).
- **Why bumped**: at bump, (a) over-emits or includes stale invariants. LLM-validates may clean this up better than human review.
- **Files**: `_spike/findings/hybrid_experiments/h2_validate/`.
- **What we learn**: can LLM reliably distinguish (a)'s true positives from spurious / stale?

### Pattern H3: LLM proposes -> (a) runtime-verifies (LLM-first with deterministic gate)

- **Flow**: pure (b) extraction. Each emitted invariant is runtime-validated using `_spike/scripts/trial_scoring.runtime_validate_invariants` (Phase 2.5 harness; transformers-only currently). Accepted = runtime-fired-as-predicted. Rejected = (b) entries that don't validate.
- **Scope**: per-class chunking.
- **Iteration**: single-shot extract + runtime-verify pass.
- **Role**: extend (with deterministic gate).
- **Cells**: transformers v4_57_3 (active) + transformers 4.56.2 (bumped). Tensorrt + vllm need runtime validation extended; this pattern is transformers-only initially.
- **Files**: `_spike/findings/hybrid_experiments/h3_propose_verify/`.
- **What we learn**: the validated subset of (b)'s output - this IS the per-cell partial "validated union" the user described. If pattern works well, this is a candidate Phase 4.0 mechanism.

### Pattern H4: LLM modifies the miner (LLM-as-maintenance-engineer) [HIGH PRIORITY]

- **Flow**: LLM reads (a)'s output + (a)'s producer source code (the AST walker module). Diagnoses gap patterns. Proposes concrete patch to the walker (e.g. "add _verify_cache_dtype to LANDMARKS; extend detector to handle if/elif/else: raise"). Patches applied; (a) re-run.
- **Scope**: per-producer (the AST walker module itself + the gap evidence from (a) on a specific cell).
- **Iteration**: single-pass (LLM proposes, human or scripted accepts).
- **Role**: modify-miner (META).
- **Cells**: transformers v4_57_3 producer (already mature; little to patch) + vllm v0_7_3 producer (gaps surfaced in Phase 1 Day 1: EngineArgs structurally unminable; can LLM see the gap and patch the walker?) + tensorrt v0_21_0 producer (also has known gaps).
- **Files**: `_spike/findings/hybrid_experiments/h4_modify_miner/`.
- **What we learn**: can LLM raise (a)'s ceiling by improving (a)'s code? Highest-impact meta-pattern but riskiest (could break (a)).
- **Constraint**: patches stay in the trial's local engine_versions/ trees; do NOT touch src/. Patches re-applied to a copy of the producer; (a)-with-patch re-run.
- **DUAL VALUE (raised priority)**: outputs feed spike branch's vllm + tensorrt mining refactor (Bake-off A's ~1800 LoC target). Walker patches + gap diagnoses are mergeable into the spike-refactor PR regardless of trial-internal score. Even if H4 doesn't "win" as a strategy, the artefacts are post-trial-useful.
- **Output artefacts (dual-purpose)**:
  - `proposed_patches/<engine>__<producer>.diff` - unified diffs against the original walker; reviewable + cherrypickable into spike.
  - `diagnoses.md` - structured gap analysis per engine; design input for the refactor.

---

## TIER 2: high info (run after Tier 1 lands)

### Pattern H5: (b) per-validator-method chunking (chunking ablation)

- **Flow**: pure (b) extraction. Chunking changes from per-class to per-validator-method (each `_verify_*`, each `__post_init__`, each Pydantic validator gets its own chunk).
- **Scope**: per-validator (finer than per-class).
- **Iteration**: single-shot (no multi-pass; isolate the chunking effect).
- **Role**: extend (pure (b)).
- **Cells**: transformers v4_57_3 + vllm v0_7_3 + tensorrt v0_21_0 (so we can compare per-class vs per-validator per engine).
- **Files**: `_spike/findings/hybrid_experiments/h5_per_validator_chunk/`.
- **What we learn**: does finer chunking lift invariant recall? Bake-off B hypothesis was yes; never directly tested.

### Pattern H6: (b) whole-file no-chunking (chunking ablation, upper bound)

- **Flow**: pure (b) extraction. Single prompt per engine, whole-engine-source (where it fits in 32k context).
- **Scope**: whole-file (no chunking).
- **Iteration**: single-shot.
- **Role**: extend (pure (b)).
- **Cells**: transformers v4_57_3 only (vllm + tensorrt source likely > 32k). May need to skip if context exceeded.
- **Files**: `_spike/findings/hybrid_experiments/h6_no_chunk/`.
- **What we learn**: is chunking helping or hurting? If whole-file > chunked, chunking is leaving recall on the table.

### Pattern H7: agentic loop with tool use

- **Flow**: LLM has access to tools: `read_file(path)`, `run_miner(engine, version)`, `score_against(reference)`, `list_validators(class)`. LLM decides next action each step. Budget: max 30 tool calls or 30 min wall-clock.
- **Scope**: LLM-decides (no fixed read scope; uses tools).
- **Iteration**: agentic-loop.
- **Role**: orchestrate.
- **Cells**: transformers v4_57_3 + vllm v0_7_3.
- **Files**: `_spike/findings/hybrid_experiments/h7_agentic/`.
- **What we learn**: does tool-use + adaptive iteration beat single-shot? Most novel pattern; high variance.
- **Implementation note**: use the project's own model-runner; tools are Python functions.

### Pattern H8: (a) || (b) parallel + LLM reconciles

- **Flow**: (a) and (b) both run independently. LLM reads both outputs + source; produces reconciled output: union with conflicts resolved + dropped duplicates merged.
- **Scope**: per-class chunking on (b); existing on (a); LLM gets both outputs + source per class.
- **Iteration**: single-shot reconciliation.
- **Role**: curate.
- **Cells**: transformers v4_57_3 + vllm v0_7_3.
- **Files**: `_spike/findings/hybrid_experiments/h8_parallel_reconcile/`.
- **What we learn**: how often do (a) and (b) disagree? When they agree, are both right? When they disagree, who wins?

---

## TIER 3: stretch (if time)

### Pattern H9: (a) -> LLM diagnoses gaps (analytic; no output mutation)

- **Flow**: LLM reads (a)'s output + source. Categorises gaps by reason ("normalisation pattern", "local-variable compare", "nested companion class", etc.). Outputs structured diagnosis only.
- **Scope**: per-engine.
- **Iteration**: single-shot.
- **Role**: diagnose (analytic).
- **Cells**: transformers + vllm + tensorrt (active).
- **Files**: `_spike/findings/hybrid_experiments/h9_diagnose/`.
- **What we learn**: shape of (a)'s blindspots across engines. Cross-engine pattern detection.

### Pattern H10: hierarchical chunking (class-level + nested method-level)

- **Flow**: pure (b) extraction. Chunking = class-level overview first, then nested per-method drill-down for classes that look invariant-heavy.
- **Scope**: hierarchical.
- **Iteration**: 2-pass per class (overview + drill).
- **Role**: extend.
- **Cells**: transformers + vllm (active).
- **Files**: `_spike/findings/hybrid_experiments/h10_hierarchical/`.
- **What we learn**: does structured decomposition help vs flat per-class?

### Pattern H11: cross-engine transfer (few-shot from transformers applied to vllm/tensorrt)

- **Flow**: LLM gets the transformers v4_57_3 invariant catalogue as canonical few-shot examples. Then extracts vllm + tensorrt invariants using transformers' shape patterns as priors.
- **Scope**: per-class (vllm/tensorrt source).
- **Iteration**: single-shot.
- **Role**: extend.
- **Cells**: vllm + tensorrt active.
- **Files**: `_spike/findings/hybrid_experiments/h11_cross_engine/`.
- **What we learn**: does transformers' richer surface help vllm/tensorrt extraction? Tests universal-invariant-shape hypothesis.

### Pattern H12: Ollama-F16 quantisation ablation

- **Flow**: pure (b) extraction (same as Phase 3a.1 (b)) but using Ollama-F16 70B instead of Ollama-q4 70B.
- **Scope**: per-class (existing chunking).
- **Iteration**: multi-pass (same as Phase 3a.1).
- **Role**: extend (pure (b)).
- **Cells**: transformers v4_57_3 (single-cell ablation).
- **Files**: `_spike/findings/hybrid_experiments/h12_fp16/`.
- **Setup**: pull `llama3.1:70b-instruct-fp16` GGUF (~140 GB; verify Ollama tag); confirm 4-GPU tensor split works.
- **What we learn**: how much of (b)'s recall ceiling is q4 quantisation vs substrate ceiling?

---

## Allocation summary

12 patterns total. Realistic Phase 3b plan:
- Tier 1 (4 patterns): MUST run.
- Tier 2 (4 patterns): SHOULD run.
- Tier 3 (4 patterns): COULD run.

Cell budget:
- Tier 1: ~10 cells (some bumped).
- Tier 2: ~8 cells.
- Tier 3: ~6 cells (+ Ollama-F16 download cost).
- Total: ~24 cell-runs.

LLM time: ~24 cells x 25 min = 10 hrs of LLM serialised. Hybrid agents can spawn in parallel-but-LLM-bottlenecked sequence.

Each pattern's subagent receives this catalogue entry + its assigned 2-3 cells + the trial-resume-prompt.md context.

---

## Cross-cutting tooling needs

Phase 3b may surface needs that should be built once and reused:

- **Validated-union builder**: takes all per-cell outputs + runtime-validates each unique entry; emits per-cell `validated_union.yaml`. Phase 4.0 work. H3 + H8 both produce inputs. Two-layer validation:
  - **Invariants**: existing `runtime_validate_invariants` (Phase 2.5, transformers-only) + extension to vllm + tensorrt via their containers. Uses kwargs_positive / kwargs_negative.
  - **Schema (NEW)**: `runtime_validate_schema(schema, engine)` mirror. Layer A: `Config(**{field: plausible_value})` doesn't raise + `field in Model.__fields__`. Handles extra=allow models by using the declared `__fields__` filter (declared fields stay enumerable even with extra=allow). The hard case (undeclared field accepted via extra=allow with runtime effect) needs Layer B (vary + measure via llem itself); out of scope for the trial.
- **(a)-with-patch runner**: H4 needs an isolated copy of the producer + a patch-apply + re-run mechanism. Doesn't touch src/ or canonical engine_versions/<e>/v*/ trees.
- **Agentic tool harness**: H7 needs `read_file`, `run_miner`, `score_against`, `list_validators` exposed as LLM-callable tools. Should be reusable across other agentic patterns.

These are infrastructure builds; first subagent that needs each one builds it + checks in via `_spike/scripts/`.
