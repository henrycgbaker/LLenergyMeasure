# Phase 1 Day 1 - vllm v0_7_3 static invariant miner lift

**Status:** ready (with caveats below).
**Authored:** 2026-05-25.
**Trial cross-ref:** `.planning/mining-substrate-empirical-trial.md` Phase 1 Day 1; epistemic framing `_spike/findings/trial_epistemic_framing.md`.

---

## Summary

| Metric | Starting | Ending |
|---|---|---|
| Invariants in `invariants.proposed.yaml` | 10 | 26 |
| Validation pass rate (positive + negative confirmed) | 1/10 | 26/26 |
| Divergences in `invariants.validated.yaml` envelope | n/a | 0 |
| Plan target | 25-30 | hit (26) |

Re-validation in the canonical container (`vllm/vllm-openai:v0.7.3`) reproduces the on-disk envelope byte-for-byte (timestamp aside). Re-running the static miner via the dispatcher emits 61 raw candidates of which 26 survive curation into `invariants.proposed.yaml` (see Source-acquisition + curation flow below).

---

## Per-surface audit

Plan targets vs. actual coverage (counts are invariants per validation surface in the curated proposed.yaml):

| Plan target surface | Plan count | Actual | Verdict |
|---|---|---|---|
| `vllm/engine/arg_utils.py::EngineArgs.__post_init__` | 10-15 | 0 | unmet at this surface; redirected to LoRA/PromptAdapter/TokenizerPool (see "Why" below) |
| `vllm/config.py::ModelConfig.__post_init__` / `_verify_*` | 5-8 | 0 | unmet at this surface; same reason |
| `vllm/config.py::CacheConfig.__post_init__` / `_verify_*` | 3-5 | 1 | under (only `_verify_args` yielded; `_verify_cache_dtype` / `_verify_prefix_caching` patterns not extractable) |
| `vllm/config.py::SchedulerConfig.__post_init__` / `_verify_args` | 3-5 | 5 | met |
| `vllm/sampling_params.py::SamplingParams.__post_init__` (deeper walk) | 5-8 on top | 12 total (7 new vs starting baseline of 5) | met |
| **Total (plan)** | **25-30** | **26** | **headline target met** |

Bonus surfaces filled to make up the headline count:

| Bonus surface | Count | Notes |
|---|---|---|
| `vllm.config.ParallelConfig._verify_args` | 1 | `distributed_executor_backend not_in {ray, mp, uni, external_launcher, None}` |
| `vllm.config.LoRAConfig.__post_init__` | 3 | `max_loras < 1`, `max_cpu_loras < max_loras`, `max_cpu_loras absent` (dormant_silent) |
| `vllm.config.PromptAdapterConfig.__post_init__` | 3 | `max_prompt_adapters < 1`, `max_prompt_adapter_token == 0`, `max_cpu_prompt_adapters absent` (dormant_silent) |
| `vllm.config.TokenizerPoolConfig.__post_init__` | 1 | `extra_config not type dict` |

### Why the under-coverage on EngineArgs / ModelConfig / CacheConfig is structural

The static miner's predicate translator (`_extract_predicates` in `engine_versions/vllm/v0_7_3/producers/static_invariant_miner.py`) requires `self.<field> <op> <literal>` or `self.<field> <op> self.<other_field>`. Most non-trivial validation in EngineArgs and ModelConfig violates these constraints:

1. **`EngineArgs.__post_init__` (vllm 0.7.3, lines 213-236) is trivial** - 100% normalisation assignments (`self.tokenizer = self.model`, `self.enable_prefix_caching = bool(envs.VLLM_USE_V1)`, etc.), no `raise` / `warn`. **0 mineable invariants by design at this surface in 0.7.3.** The plan's "10-15" assumed validation density that isn't there.

2. **`EngineArgs.create_engine_config` (lines 1093-1396)** is where the engine-side raises live, but they use:
   - compound predicates with module-level helpers (`current_platform.is_cuda()`) -> miner can't extract,
   - compound predicates with local variables (`speculative_config is not None` where `speculative_config` is a local) -> miner can't extract,
   - `Or` boolean compositions (`self.quantization == "bitsandbytes" or self.qlora_adapter_name_or_path is not None`) -> miner rejects `BoolOp(Or)` by design.

3. **`ModelConfig.__post_init__` does not exist** in 0.7.3. The class uses `__init__`, not a dataclass `__post_init__`. The miner registry already targets the four `_verify_*` methods (`_verify_quantization`, `_verify_tokenizer_mode`, `_verify_cuda_graph`, `_verify_bnb_config`), but these compare to *local* variables computed from `self.X.lower()` / `getattr(...)`, not to `self.X` directly:
   ```python
   def _verify_tokenizer_mode(self) -> None:
       tokenizer_mode = self.tokenizer_mode.lower()
       if tokenizer_mode not in ["auto", "slow", "mistral", "custom"]:
           raise ValueError(...)
   ```
   The miner expects `self.tokenizer_mode not in [...]` but sees `tokenizer_mode not in [...]` (local) and skips.

4. **`CacheConfig._verify_cache_dtype` / `_verify_prefix_caching`** have `if/elif/else: raise` shapes and early-return gates that the walker doesn't handle (see "Miner shape gaps" below).

Closing these gaps requires either (a) extending the predicate translator (significant: tracking local-variable bindings to self-attribute roots, propagating through `if/elif/else: raise`, handling early-return gates as implicit negated predicates) or (b) hand-writing invariants in `proposed.yaml` (drifts further from "miner output is the proposed corpus"). Neither is in scope for Phase 1 Day 1.

The agent's choice to fill the headline count from LoRA/PromptAdapter/TokenizerPool surfaces - where simple `self.X op LITERAL` patterns dominate - is the correct triage under the miner's current limits. The plan's surface-by-surface target overshoots what the static substrate can actually deliver on 0.7.3.

---

## Extension diff summary

**File:** `engine_versions/vllm/v0_7_3/producers/static_invariant_miner.py` (+27 lines).

Three additions:

1. **Three new namespaces** (`NS_LORA`, `NS_PROMPT_ADAPTER`, `NS_TOKENIZER`) for match-field paths.
2. **Five new LANDMARKS** (drift-tool contract): `PromptAdapterConfig`, `PromptAdapterConfig.__post_init__`, `TokenizerPoolConfig`, `TokenizerPoolConfig.__post_init__`, `ParallelConfig._verify_args`.
3. **Two new `_CLASS_TARGETS`** entries:
   - `_ASTTarget(module_attr="config.PromptAdapterConfig", method="__post_init__", ...)`
   - `_ASTTarget(module_attr="config.TokenizerPoolConfig", method="__post_init__", ...)`

The detector classes (`_detect_raise`, `_detect_self_assign`, `_detect_logger_warning`, `_detect_warnings_warn`), predicate translator, kwargs synthesiser, and emission logic were NOT modified - they already handle these surfaces' patterns.

`ParallelConfig._verify_args` was already in `_CLASS_TARGETS` (existing); the LANDMARK addition is for drift-tool coverage.

**No changes to** `_DETECTORS`, `_extract_predicates`, `_synthesise_kwargs`, `_build_rule`, `_make_invariant_id`, or any other miner machinery. The miner is unchanged; the registry was extended.

### Curation drift (transparency note)

Re-running the miner via `python -m scripts.engine_producers.vllm_static_invariant_miner --out /tmp/x.yaml` inside the canonical container emits **61 raw candidates**, of which **26 survive into `invariants.proposed.yaml`**. The 35 dropped candidates are:

- noise from the `confidence_penalty=1` fall-through (`self.X` alone in an `if`-condition emits a `present True` predicate) - mostly `dormant_<field>_set_true` / `dormant_<field>_unset_true` patterns whose semantics aren't meaningful invariants,
- duplicate `__2`-suffix entries when two raises in the same `if`-body share the same predicate frame,
- entries on `SpeculativeConfig._verify_args` whose `kwargs_positive` can't construct a valid object (the class requires `draft_model_config` + `draft_parallel_config`).

The 26 surviving entries were also **hand-edited** in places to make `kwargs_positive` / `kwargs_negative` construct-able:
- `ParallelConfig` invariants got `pipeline_parallel_size=1, tensor_parallel_size=1` added (without these, `ParallelConfig(...)` fails),
- the `SchedulerConfig` chunked-prefill invariant got `chunked_prefill_enabled` (internal, `init=False`) remapped to `enable_chunked_prefill` (public init field),
- IDs were renamed in a few places to drop the trailing `_true` slug from `_unset` / `_present` predicates.

**Consequence:** the proposed.yaml is NOT regeneratable by `make refresh-invariants` alone. A re-run produces the 61-entry raw output; getting back to the curated 26 requires the agent's selection + edits, which aren't captured in code. This is acceptable as Phase 1 baseline (the trial measures *the substrate's output quality*, not pipeline mechanics), but should be flagged for Phase 3 matrix execution: if the matrix runner expects "regenerate proposed.yaml from miner", it needs to use `_staging/vllm_static_invariant_miner.yaml` (raw) rather than the curated `outputs/invariants.proposed.yaml`.

---

## Validation results

Validation was run via `scripts/validate_invariants.py` inside `vllm/vllm-openai:v0.7.3` (canonical container), with `--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all` for GPU access.

| Case | Count |
|---|---|
| `positive_confirmed: true` | 26/26 |
| `negative_confirmed: true` | 26/26 |
| `positive_confirmed: false` | 0 |
| `negative_confirmed: false` | 0 |
| Envelope `divergences: []` | confirmed |

Per-invariant outcomes:
- 23 raises with concrete `observed_exception.message` matching the declared `message_template` after field-substitution.
- 2 `dormant_silent` cases (`samplingparams.seed == -1`, `loraconfig.max_cpu_loras absent`) with observed `silent_normalisation` records.
- 1 dormant case (`promptadapterconfig.max_cpu_prompt_adapters absent`) with silent normalisation.

No "fired but message diverged" cases. No "did not fire" cases. No outcome-class mismatches.

The fresh re-run I did during this audit (`diff /tmp/validated_check.yaml engine_versions/vllm/v0_7_3/outputs/invariants.validated.yaml`) showed a single-line diff on `validated_at` timestamp - the envelope is reproducible byte-for-byte modulo timestamp.

---

## Observations (for Phase 2 / Phase 3 signal)

### vllm-specific miner shape gaps

These are structural limits of the current static miner that surfaced during the audit. Logged here for the synthesis (and as candidate "what LLM substrate would catch that we don't"):

1. **`if/elif/else: raise` not walked.** `CacheConfig._verify_cache_dtype` has `if cache_dtype == "auto": pass / elif cache_dtype in (...): logger.info(...) / else: raise`. The walker's `_handle_if` traverses `if_node.orelse` only as nested `if`s; the bare `raise` in the final `else` is invisible. Phase 2 LLM substrate would naturally pick this up because it reads the whole method body.

2. **Local-variable binding not propagated.** `_verify_tokenizer_mode` does `tokenizer_mode = self.tokenizer_mode.lower(); if tokenizer_mode not in [...]: raise`. Static miner sees only the second statement and rejects (no `self.X`). A tiny constant-propagation pass (track `local <- self.attr.lower()` aliases) would unlock this and ~3-5 other ModelConfig _verify_ patterns.

3. **Early-return gates not treated as implicit predicates.** `_verify_prefix_caching` reads `if not self.enable_prefix_caching: return / if self.sliding_window is not None: raise`. The second raise is contingent on `enable_prefix_caching == True`, but the walker emits an invariant predicated only on `sliding_window present True`, which would over-fire. Tracking early-return as a frame-level conjoined predicate is feasible.

4. **`Or` boolean composition silently dropped.** `_extract_predicates` returns `[]` on `BoolOp(Or)`. Several EngineArgs raises use `or` to chain alternative trigger conditions; the miner sees nothing. Decomposing `or` into two separate invariants (one per branch) would be a non-trivial but tractable extension.

5. **Module-level / property-chain `self.X.attr` predicates.** `_self_attr` only matches `self.<name>`, not `self.<name>.<attr>`. `_verify_cuda_graph` reads `self.hf_config.model_type in [...]`; not extractable.

These are SAME failure modes the static substrate would face across other engines (transformers, tensorrt). Worth grouping as a single "static AST predicate translator's limit set" finding in Phase 4 synthesis.

### Validation surface density per class

The mineable invariant count per class is heavily skewed:

- **`SamplingParams` (sampling_params.py)**: 12 invariants. High density: stdlib-style validation, `_verify_args` is a single method with ~10 sequential `if self.X op LITERAL: raise`. This is the static miner's sweet spot.
- **`SchedulerConfig` / `LoRAConfig` / `PromptAdapterConfig` / `TokenizerPoolConfig`**: 5-3 invariants each. Same pattern as SamplingParams but smaller methods.
- **`ModelConfig` / `EngineArgs`**: 0 invariants despite having extensive validation. The validation is just shaped differently (helper functions, locals, compound conditions, `Or` chains).

**Implication for Phase 2 (LLM substrate):** the LLM should outperform here precisely on `ModelConfig` / `EngineArgs` / `CacheConfig._verify_cache_dtype`, where the static substrate's translator is fundamentally blind. The bake-off measures the right gap.

**Implication for Phase 4 (synthesis):** when comparing pure-mining vs LLM on vllm v0_7_3, expect the LLM to extract ~10-15 ModelConfig/EngineArgs invariants that pure-mining misses. That's the headline gap, not the SamplingParams overlap.

### Curation cost

Hand-curation from 61 raw candidates to 26 clean entries took ~3 hours of agent time (per the diff timestamps and the killed-agent's work-state). For a 3-engine x 5-version matrix, naively scaling = 45 cells x 3 hours = ~135 hours of curation. **This is not sustainable.** Two mitigation paths:

1. **Improve the raw miner to filter its own noise** (drop `present True` confidence-penalty entries by default; drop kwargs_positive that fail construction; collapse `__2` duplicates intelligently). ~200-300 LoC of miner work; pays for itself across the matrix.
2. **Accept the raw 61-entry output as Strategy (a)'s baseline** and let the LLM substrate (Strategy b/c) do the curation in Phase 2/3. This treats the deterministic stack as "evidence, not SSOT" - the curator pattern from Open Question 10.

Both paths matter for Phase 4 synthesis. Option 1 is the deterministic-stack refactor argument; option 2 is the mining-as-evidence-for-curation argument.

---

## Source-acquisition path used

- **vllm 0.7.3 wheel source:** unpacked at `/tmp/vllm-unpacked/` (host-side, already present at session start). Container's site-packages provides the runtime import path during validation.
- **Static AST walks:** performed inside the vllm container, sourcing from `vllm/__file__` (resolves to the container's `/usr/local/lib/python3.10/site-packages/vllm/`).
- **Validation:** `vllm/vllm-openai:v0.7.3` image (already pulled, 16.4GB).

No PyPI re-download / isolated venv required; the canonical container is the validation environment.

---

## Status

**Ready.** Headline target met (26 invariants, 100% validation pass rate, zero divergences). The work is acceptable as the Strategy (a) baseline for Phase 3 matrix execution on the `(vllm, v0_7_3)` cell.

**Caveats for the coordinator / Phase 3 runner:**

1. The 26 invariants in `engine_versions/vllm/v0_7_3/outputs/invariants.proposed.yaml` are NOT bit-for-bit regeneratable by `make refresh-invariants ENGINE=vllm`. A re-run emits 61 raw candidates; getting to the curated 26 requires the hand-editing done in this session. If Phase 3's runner expects mechanical regen, it needs to call the miner directly and accept the noisy 61-entry output, OR ingest the curated proposed.yaml as a stored artefact.
2. Per-surface coverage is asymmetric vs. the plan: EngineArgs / ModelConfig / CacheConfig.verify_cache_dtype / CacheConfig.verify_prefix_caching are 0 in proposed because the miner's predicate translator can't extract their patterns. The gap is structural, not effort-related, and is itself decision-relevant evidence for Phase 4 (where Strategy b/c LLM should outperform Strategy a).
3. Tensorrt v0_21_0 Phase 1 Day 2 work is also in the diff (3 -> 38 invariants in `engine_versions/tensorrt/v0_21_0/outputs/invariants.proposed.yaml`). Not audited in this report; out of scope for Day 1.

No follow-up needed on vllm v0_7_3 to unblock Phase 3 matrix execution. Coordinator can commit the diff as Phase 1 Day 1 deliverable.
