# Post-trial (a) gap closure commitment

**Status:** authoritative; whichever substrate wins Phase 4, these gaps in
strategy (a) deterministic mining MUST be closed before the substrate
decision becomes the production state.
**Audience:** Phase 5 curation pipeline implementer; spike-branch refactor
maintainer; any future agent picking up post-trial.

---

## Why this doc exists

The mining-substrate empirical trial deliberately preserved known gaps in
strategy (a) as part of the realistic baseline. The epistemic framing
(`research/mining-substrate-trial/findings/trial_epistemic_framing.md`) calls this out: the trial
measures (a) at the as-is state; mid-trial optimisation toward what looks
promising would corrupt the data.

But "preserved during the trial" is NOT "accepted forever." Regardless of
which substrate Phase 4 recommends, these gaps must be plugged:

- If (a) wins -> patch and ship.
- If (b)/(c) wins -> the deterministic floor still needs to function
  correctly where retained.
- If (d) hybrid wins -> (a) is the lower layer; gaps propagate.
- If inconclusive -> ship-with-defaults defaults to (a); gaps still hurt.

H4 (LLM-modifies-miner) is the pattern designed to produce these patches
automatically. If H4 succeeds, most of the work is its output. If H4
fails or partially succeeds, the residual is human + LLM-assisted work
on the spike branch.

---

## Gap inventory

Each gap has: location, structural reason, patch path, estimated effort,
status, acceptance criteria.

### vllm v0_7_3

**G-vllm-1: EngineArgs.__post_init__ validators uncaptured.**

- **Location:** `vllm/engine/arg_utils.py::EngineArgs.__post_init__`
- **Reason:** vllm 0.7.3's EngineArgs.__post_init__ has ZERO raises. All
  validation is normalisation (`if x is None: x = default`). The current
  AST walker only emits invariants from `if X: raise` patterns.
- **Plan target:** 10-15 invariants. Actual: 0.
- **Patch path:**
  - Option A (low confidence): extend walker to detect normalisation
    patterns and emit them as severity=`warn`/`info` (different shape
    than error invariants). Risk: emits low-value notices, not real
    constraints.
  - Option B (recommended): re-frame these as "defaults" in the schema,
    not invariants. Move normalisation logic to schema_introspector
    output as default-value annotations.
  - Option C (LLM-mediated): H4 reads source + diagnoses + proposes a
    schema-vs-invariant boundary refactor. Most flexible.
- **Effort:** 80-150 LoC depending on option.
- **Status:** deferred during trial.
- **Acceptance criteria:** EngineArgs's actual validation surface is
  surfaced somewhere (either invariants or schema defaults); no field
  silently dropped.

**G-vllm-2: ModelConfig validators uncaptured.**

- **Location:** `vllm/config.py::ModelConfig._verify_quantization`,
  `_verify_tokenizer_mode`, `_verify_cuda_graph`, `_verify_bnb_config`
- **Reason:** ModelConfig has no `__post_init__`; uses `__init__`. The
  `_verify_*` helpers compare LOCAL variables (e.g. `if quantization
  not in QUANTIZATION_METHODS:`) not `self.X`. The AST walker can't
  tie the local-variable comparison to a field invariant without
  understanding the call graph.
- **Plan target:** 5-8 invariants. Actual: 0.
- **Patch path:**
  - Option A: extend walker with light call-graph analysis: when a
    method is called from `__init__` with `self.X` as argument and
    the method body has `if arg not in ALLOWED: raise`, tie the
    invariant to field `X`.
  - Option B (LLM-mediated): H4 reads the verify methods + their call
    sites + proposes invariant entries.
- **Effort:** 150-250 LoC for Option A; possibly less for Option B.
- **Status:** deferred during trial.
- **Acceptance criteria:** each `_verify_*` method's constraints
  surface as invariants on the appropriate field.

**G-vllm-3: CacheConfig._verify_cache_dtype if/elif/else pattern.**

- **Location:** `vllm/config.py::CacheConfig._verify_cache_dtype`
- **Reason:** Uses `if X: raise / elif Y: raise / else: raise` chain.
  Walker only handles top-level `if X: raise` not the elif/else branches.
- **Plan target:** 3-5 invariants. Actual: partial.
- **Patch path:** extend walker to traverse all branches of an
  if/elif/else chain. Emit one invariant per terminal raise.
- **Effort:** 30-50 LoC walker extension.
- **Status:** deferred during trial.
- **Acceptance criteria:** all raise statements in if/elif/else chains
  surface as invariants.

### tensorrt v0_21_0

**G-trt-1: Type-blind probe-value synthesis.**

- **Location:** `scripts/engine_producers/_common.py::_value_satisfying`
  (or wherever probe synthesis lives).
- **Reason:** `_value_satisfying("present", True)` returns `"x"` even
  for int-typed fields. Runtime validation fails with type errors,
  masking 11/35 tensorrt invariants as "broken" when the invariants
  themselves are correct.
- **Patch path:** make probe synthesis type-aware. Inspect the field's
  declared type; emit `0`, `True`, `"x"`, or `[]` depending. Already
  scoped at ~30 LoC by Phase 1 Day 2 audit.
- **Effort:** ~30 LoC. Lives in src/.
- **Status:** deferred during trial (would have been "mid-trial
  optimisation"). Phase 4 + Phase 5 must apply this regardless of
  substrate winner.
- **Acceptance criteria:** runtime validation of tensorrt invariants
  drops infrastructure-error count from 11/35 to <3/35.

**G-trt-2: DeprecationWarning poisoning of negative-case capture.**

- **Location:** `scripts/_invariant_validation_common.py` or wherever
  `_run_tensorrt` captures emissions.
- **Reason:** tensorrt_llm emits a generic `DeprecationWarning` on
  validation paths that contaminates the negative-case capture. 18/35
  tensorrt invariants are positive-confirmed but negative-tripped by
  this warning (mis-classified as "fires when it shouldn't").
- **Patch path:** add a strip-list pattern matching the vLLM
  `_VLLM_BOOTSTRAP_NOISE` convention. Filter the DeprecationWarning
  signature out of the captured emissions.
- **Effort:** ~10 LoC.
- **Status:** deferred during trial.
- **Acceptance criteria:** tensorrt invariants validation pass rate
  lifts from 11% both-confirmed to ~80%+ (both-confirmed AND positive-
  only collapsed into both-confirmed once DeprecationWarning stops
  poisoning).

**G-trt-3: Nested-config dispatch (SchedulerConfig, QuantConfig, KvCacheConfig).**

- **Location:** `engine_versions/tensorrt/v0_21_0/producers/static_invariant_miner.py`
- **Reason:** Pydantic validators on nested config classes (e.g.
  `SchedulerConfig.capacity_scheduler_policy`, `QuantConfig.quant_algo`)
  don't get walked because the walker doesn't follow class-level type
  references into companion config classes.
- **Status during trial:** BuildCache + CalibConfig D1/D3 deferrals
  LIFTED in Phase 1; SchedulerConfig + remaining nested-config dispatch
  stayed deferred.
- **Patch path:** extend walker to recurse into Pydantic field types
  that are themselves config classes. Mirror the transformers nested-
  dataclass walker from spike commit `15f34240`.
- **Effort:** 200-400 LoC.
- **Status:** deferred during trial.
- **Acceptance criteria:** invariants from SchedulerConfig, QuantConfig,
  KvCacheConfig (at minimum) surface; cross-engine evidence of nested-
  config gap (also exists in transformers BNB and vllm CacheConfig
  companions) gets a unified abstraction.

### transformers v4_57_3

Mostly complete. Mature 41-invariant baseline. Per Phase 1 Day 1, no
significant gaps that affect (a)'s ceiling on the active version.

The ONE known gap is at version bumps:
**G-trf-1: tokenizers + huggingface_hub API rename brittleness.**

- **Location:** producer-side imports that crash on transformers v-2
  (4.55.4) and v+major (5.9.0).
- **Reason:** `from tokenizers import ...` or `from huggingface_hub
  import ...` patterns broke across these versions; the producer
  module fails at import time on bumped versions (per Phase 3a.2
  brittleness data).
- **Patch path:** make the producer's imports defensive (`try ...
  except ImportError`). Or use the AST file-reading pattern (which
  is what (b) does to sidestep imports).
- **Effort:** 50-100 LoC defensive import wrapping.
- **Status:** deferred during trial (the crashes ARE the brittleness
  signal; (b) recovers where (a) crashes per Phase 3a.2 transformers
  data).
- **Acceptance criteria:** (a) producers don't crash on v-2 or v+major
  source; they may emit zero invariants if the API truly changed (real
  brittleness) but the failure mode is "no-op" not "ImportError".

---

## Closure path mapping

Two natural closure mechanisms:

### A. H4 LLM-modifies-miner outputs (trial-internal)

Phase 3b H4 produces:
- `research/mining-substrate-trial/findings/hybrid_experiments/h4_modify_miner/proposed_patches/<engine>__<producer>.diff`
- `research/mining-substrate-trial/findings/hybrid_experiments/h4_modify_miner/diagnoses.md`

H4 success: each proposed patch maps to one of the gaps above. Review
+ apply. Closure work reduces to "review LLM-proposed patches" which
is cheap.

H4 partial success: some gaps closed by LLM patches; residual to (B).

H4 failure: full residual to (B).

### B. Spike-branch refactor (post-trial human/Claude)

Whichever gaps H4 doesn't close get assigned to the spike branch's
existing mining refactor work (Bake-off A's ~1800 LoC target). Each
gap above is a single PR-scope task with clear acceptance criteria.

Estimated total effort for all gaps without H4 help: ~500-1000 LoC.

---

## Commitment

When Phase 4 + Phase 5 conclude, this doc gets converted into either:

1. **A backlog of GH issues** (one per gap; pointing at this doc for
   context) for the spike-branch maintainer.
2. **A subset of the trial PR-extraction** if H4's patches make the
   gaps trivially closable, the diffs themselves PR-extract.

In neither case do these gaps become "accepted forever." The trial's
discipline preserved them as research data; the discipline of "ship
correct production code" closes them after.

---

## Cross-references

- Inventory came from: `research/mining-substrate-trial/findings/phase1_vllm_miner_lift.md`,
  `research/mining-substrate-trial/findings/phase1_tensorrt_miner_lift.md`,
  `research/mining-substrate-trial/findings/phase3a1_active_matrix.md`.
- H4 catalogue entry: `research/mining-substrate-trial/findings/phase3b_hybrid_catalogue.md`
  Tier 1 H4 (LLM-modifies-miner).
- Trial epistemic framing: `research/mining-substrate-trial/findings/trial_epistemic_framing.md`.
- Spike refactor scope: `research/mining-substrate-trial/findings/bakeoff_A_refactor_analysis.md`
  (the ~1800 LoC target).
