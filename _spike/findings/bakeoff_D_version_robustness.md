# Bakeoff D - Version Robustness: v4.57.3 Producers vs Transformers 5.9.0

## Verdict

**Medium-fidelity transfer.** All three v4.57.3 producers ran to completion against transformers 5.9.0 with zero crashes. Field discovery transferred at high volume (88-96% Jaccard for schema, 75-92% for invariants by count). The degradations are real but non-catastrophic: schema loses type information for 66/67 sampling params (silent miss - the miner records fields correctly but emits `null` type specs instead of `bool`/`int`/`number`); the static miner gains 16 new invariants from new validate() checks, misses 1 due to predicate structure change; the dynamic miner silently misses 4 dormancy invariants due to a probe-value/type-check incompatibility introduced in 5.9.0. No false positives were observed. The failure mode split is roughly: schema = silent degradation (type loss), invariant miners = mix of noisy-good (new invariants found) and silent-bad (probe bypasses new type-gate). Maintenance cost estimate: ~30-50 LoC of targeted patches per bump for the invariant miners, plus a schema path refactor for type recovery (~100 LoC one-time).

---

## Env Setup

- Target version: **transformers 5.9.0** (latest stable on PyPI as of 2026-05-24)
- Isolated venv: `uv venv /tmp/bakeoff_d_venv --python 3.11`
- Install: `uv pip install transformers==5.9.0 torch==2.7.0+cpu pyyaml`
- No dependency conflicts. CPU torch required for `AutoModelForCausalLM` landmark (otherwise it resolves to a `DummyObject` that raises on `from_pretrained` access).
- Producers run: `v4_57_3/producers/{schema_introspector, static_invariant_miner, dynamic_invariant_miner}.py`
- **Producer files were not modified.** All observations reflect unpatched v4.57.3 code against the 5.9.0 library.

---

## Per-Producer Table

| Producer | Ran to completion | Landmark resolution | Output volume vs GT | Overlap with GT | Time | Failure mode |
|---|---|---|---|---|---|---|
| schema_introspector | Yes | 7/7 OK | EP: 40 vs 39 (+1); SP: 69 vs 68 (+1) | EP 88% Jaccard; SP 96% Jaccard | 12s | Silent: 66/67 SP fields lose type info |
| static_invariant_miner | Yes | 4/4 OK | 28 vs 13 GT-static (+15 net) | 92% (12/13 GT recovered) | 8s | 1 lost (predicate structure change); 16 new (5.9.0 validate() additions) |
| dynamic_invariant_miner | Yes | 4/4 OK | 28 vs 28 GT-dynamic (same count) | 75% Jaccard (24/28 GT-dynamic) | 6s | Silent: 4 dormancy invariants missed due to probe-type incompatibility |

---

## Schema Introspector - Detail

**engine_params (39 -> 40):**
- Lost: `from_flax`, `from_tf` - removed from `from_pretrained` signature in 5.x
- Added: `disable_mmap`, `experts_implementation`, `fusion_config` - new in 5.x
- Type specs: all 37 overlapping EP fields retained correct type info (Sphinx-kwargs walker still works)

**sampling_params (68 -> 69):**
- Lost: `return_legacy_cache` - deprecated, removed in 5.x
- Added: `continuous_batching_config`, `top_h` - new in 5.x
- **Type degradation on 66/67 overlapping SP fields**: In 4.x, `GenerationConfig.__init__` had explicit typed parameters; `inspect.signature` returned real type annotations. In 5.9.0, `__init__` accepts `**kwargs` only (all fields set via `self.X = kwargs.pop("X", None)`). Every field defaults to `None`, `runtime_value_to_spec(None)` emits `{"description": "...", "default": null}` instead of `{"type": "boolean"}` etc. The miner correctly records the limitation ("GenerationConfig has no type annotations; None defaults yield empty schemas") - it's noisy-visible, not hidden. The field names are all correct; the type info is gone.

---

## Static Invariant Miner - Detail

**Lost (1):** `transformers_raises_num_beams_eq_1` - in 4.x the predicate was `num_return_sequences != 1 AND num_beams == 1`. In 5.9.0 a third field was added: `num_return_sequences != 1 AND num_beams == 1 AND NOT do_sample`. The AST walker emitted the 2-field form; 5.9.0 changed this to a 3-field conjunction. The invariant itself is real and still fires; the walker emitted a partial version of it under a different ID (`transformers_num_return_sequences_exceeds_num_beams` - the `> num_beams` sub-branch which was also in 4.x but now the greedy/no-do_sample branch is a separate predicate). Net: the constraint is still captured, just split.

**Added (16 new invariants):** All appear to reflect genuine new validation checks in 5.9.0's `validate()`:
- 8 new dormancy invariants (`transformers_dormant_*`) - 5.9.0 added dormancy warnings for beam-only flags (`top_h`, temperature, top_p, etc.) with explicit non-None checks
- `transformers_raises_early_stopping_not_in_set` - new strict type check (`early_stopping` must be in `{None, True, False, "never"}`)
- `transformers_raises_max_new_tokens_le_0` - new check `max_new_tokens > 0`
- `transformers_raises_cache_implementation_set_true`, `transformers_raises_context_width_lt_1`, `transformers_raises_do_sample_unset_true`, `transformers_raises_seeding_scheme_not_in_set` - new validation checks
- `transformers_raises_bnb_4bit_compute_dtype_not_type_dtype` - new BNB field added in 5.x

These are genuine discoveries. The static miner is performing correctly - it found new library constraints.

---

## Dynamic Invariant Miner - Detail

**Lost (4, all silent misses):**

1. `transformers_single_beam_strips_early_stopping`: In 5.9.0 `early_stopping` defaults to `None` (not `False`). The dynamic miner synthesises probe value `0.5` for `None`-default fields. In 4.x this was accepted by construction; in 5.9.0 a new type-gate in `validate()` rejects `early_stopping=0.5` with `ValueError: early_stopping must be a boolean or 'never'`. Construction crashes, miner silently skips the field. Root cause: `_synthesise_probe_value(None)` returns `0.5` (float guess), which is now invalid for this field.

2-4. `transformers_no_return_dict_strips_output_{attentions,hidden_states,scores}`: In 5.9.0 the dormancy check uses identity (`if getattr(self, extra_output_flag) is True`), not truthiness. The miner probes with `0.5` (synthesised from `None` default); `0.5 is True` is False, so the check silently passes. The invariants are real and still exist (verified: probing with `True` correctly raises), but the miner's probe value bypasses the gate.

**Added (4):**
- `transformers_greedy_strips_top_h` - `top_h` is a new field in 5.x; miner correctly found its dormancy behaviour
- `transformers_early_stopping_type_num_beams_eq_1` - new raise in 5.9.0 for `num_beams=1` + `early_stopping` combination
- `transformers_output_token_ids_pad_token_id_lt_zero` - now surfaces via `validate(strict=True)` raise path (was minor_issues only in 4.x)
- `transformers_watermarking_type_watermarking_config_exceeds_zero` - new check

---

## Categorical Analysis

| Change category | Impact | Example |
|---|---|---|
| `GenerationConfig.__init__` signature refactor (explicit params -> `**kwargs`) | Schema type loss (silent, 66/67 SP fields) | All `sampling_params` fields lose `type` key |
| New field added to library | Miner picks it up correctly | `top_h`, `continuous_batching_config`, `disable_mmap` |
| Field removed from library | Correctly absent from output | `from_flax`, `from_tf`, `return_legacy_cache` |
| New type-gate in `validate()` (construct-time rejection) | Dynamic miner probe crashes, silently skips | `early_stopping=0.5` rejected; dormancy miss |
| `is True` identity check (not `== True`) | Dynamic probe with `0.5` bypasses check; silent miss | `output_scores`, `output_attentions`, `output_hidden_states` |
| New `validate()` conditional added | Static miner finds it correctly (AST walk) | 16 new invariants |
| Predicate gained a new field (2-field -> 3-field conjunction) | Miner emits the simpler form; partial capture | `num_beams_eq_1` invariant split |
| `minor_issues` path vs `raise` path swap | Dynamic miner switches from finding/missing based on `strict=True` raise pattern | `pad_token_id_lt_zero` now raises instead of warning |

The dominant pattern: **structural changes to how HF constructs/validates configs cause probe-value incompatibilities** in the dynamic miner. The static miner is more robust because it reads AST, not probes. The schema miner's type degradation is architectural - it relied on `inspect.signature` typed params that no longer exist.

---

## Counterfactual: Patches Needed for 5.9.0

If the v4.57.3 producers needed updating for 5.9.0, the minimum patches are:

**schema_introspector.py (~60 LoC):**
The `runtime_value_to_spec` path is now useless for GenerationConfig. The fix is to scrape type information from the class docstring (which still has `int`, `bool`, `float` annotations in the Args section). The `parse_sphinx_kwargs` walker already handles this for `from_pretrained` kwargs; extending it to cover the class-level docstring is ~40-60 LoC. Alternatively, fall back to the type hints from the `# Args:` block in the class docstring.

**static_invariant_miner.py (~5 LoC):**
The lost invariant (`num_beams_eq_1` -> 3-field check) requires the AST walker to handle `elif` branches correctly when a prior `if` branch includes a `not do_sample` sub-check. The walker already drops sub-clauses; the fix is a 3-5 LoC adjustment to the `_detect_conditional_raise` path to emit both the 2-field and 3-field forms. The 16 new invariants were found automatically with no changes required.

**dynamic_invariant_miner.py (~20 LoC):**
Two fixes:
1. `_synthesise_probe_value(None)` for typed fields: where the field has a new type-gate, `0.5` is invalid. Fix: synthesise `True` for bool-like fields when the field is known to be bool (e.g., by checking if the validate() source contains `not in {None, True, False}`). Or: synthesise `True` for ALL `None`-default fields as the primary probe (since the dormancy checks tend to be `field is True` or `field is not False`). ~10 LoC change to `_synthesise_probe_value`.
2. Identity check bypass: same fix covers this - `True` would pass the `is True` gate correctly.

**Total estimate: ~80-90 LoC of targeted changes.** The static miner needs the least (~5 LoC), the dynamic miner needs probe-value logic fixes (~20 LoC), the schema introspector needs a docstring-scraping fallback path (~60 LoC).

---

## Decision Impact

The producers transferred at medium-to-high fidelity for the two invariant miners and medium fidelity for the schema introspector. Based on this experiment:

**Deterministic floor: worth keeping, with per-bump patching.** The miners successfully discovered all new validate() checks in 5.9.0 automatically (static miner: 16 new invariants found, no code change needed; dynamic miner: 4 new invariants found). The failures are in finding 4 existing invariants that changed mechanically. The patch cost per bump is ~80-90 LoC, concentrated on two narrow failure modes (probe-value synthesis and schema type recovery).

The critical observation is failure mode asymmetry by producer:
- **Static miner** failures are mostly **noisy-good** (new invariants surfaced) or detectable (predicate structure changes produce incorrect but present invariants that CI validates).
- **Dynamic miner** failures are **silent misses** - the miner runs cleanly, emits fewer invariants, and you don't know what you've lost without a ground-truth comparison. This is the more dangerous failure mode for the "LLM is verification layer" architecture.
- **Schema miner** failure is **loudly declared** in `discovery_limitations` - the miner knows it lost type information and says so. Not silent.

If the architecture treats deterministic miners as primary and LLM as secondary (current bet), the silent misses in the dynamic miner are the main risk. For the 4 missed invariants in this bump, the LLM layer would need to catch `output_scores` dormancy under `return_dict_in_generate=False`, `early_stopping` dormancy under `num_beams=1`, etc. These are semantically non-obvious enough that LLM miss probability is non-trivial.

**If medium transfer**: the current evidence supports treating deterministic miners as a stable-but-degrading floor. The per-bump patching cost is bounded (~80-90 LoC) and the patch patterns repeat (probe-value synthesis, new-field addition, new validate() check). The LLM as verification/supplement layer is appropriate - it catches what the dynamic miner misses by silent bypass.

**If low transfer** had been the verdict (it wasn't): LLM as primary would be right. But the static miner's performance (12/13 GT invariants preserved, 16 new ones found, zero crashes, no code changes) argues strongly against treating deterministic miners as version-specific snowflakes.
