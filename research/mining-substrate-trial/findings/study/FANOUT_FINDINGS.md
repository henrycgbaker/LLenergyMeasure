# Fan-out wave 1 findings - cross-engine schema + second invariant cell

> **RE-BASE NOTE (post-review, constraint-grain identity).** Two adversarial
> reviews + a focused identity review found the original headline numbers were
> computed on an over-collapsed identity `(leaf, coarse_bucket)` that merged
> genuinely-distinct per-class/per-bound constraints and let one member's pass
> "confirm" a whole group. Identity is now `(leaf, coarse_bucket,
> canonical_predicate_value)` (count + confirm at the CONSTRAINT grain;
> tolerant key retained only for cross-source recall), and the field-attribution
> guard now covers all leniently-confirmed entries. Corrected tensorrt-1.2.1
> headline: **union 212 constraints** (was 144 tolerant; 53/144 tolerant keys
> held >1 constraint), **60 confirmed** (was an inflated 46), **GT-growth +23 vs
> PoC** (was 8). Substantive domain review separately verified the confirmed
> entries are 100% real/correctly-captured (~22-entry sample), and the
> attribution tightening dropped 0 confirmations. **The deterministic-ceiling and
> lever-1 sections below have now been RECOMPUTED at the constraint grain
> (denominator 212 union / 60 confirmed). Headline change: the old tolerant-grain
> "33% confirm / 74% surfaced" collapses to 25% confirm / 25% surfaced - the
> apparent probe-synthesis gap was an identity-collapse artefact (the old 74%
> reproduces as the tolerant-key reach, 34/46); the entire deterministic deficit
> is mining-scope. Lever 1 (PluginConfig walk) lifts recall 25% -> 46.7% AND grows
> the GT by +13 constraints outside the frozen union.**


Builds on the tensorrt-1.2.1 invariant pilot (see
`ground_truth/tensorrt/v1_2_1/invariants/PILOT_REPORT.md`). This wave proves both
gates GENERALISE, using only already-available infra (no new container builds,
no new producers), and surfaces what full fan-out actually requires.

## Track S (schema gate) across 3 engines

Ran `scripts/validate_schema.py` against each engine's SHIPPED
`schema.discovered.json`, in the matching container (link 1->2: is the
discovered schema true of the live engine):

| engine | version | fields | confirmed | divergences | enum probes |
|---|---|---|---|---|---|
| transformers | 4.57.3 | 107 | 107 | 0 | 0/0 (no enum fields flagged) |
| vllm | 0.7.3 | 135 | 116 | 19 | 0/0 |
| tensorrt | 0.21.0 | 107 | 92 | 15 | 2/2 confirmed |

Artefacts: `schema_validation/<engine>_shipped_validated.json`.

**The gate runs across all 3 engines** (each via its own SSOT introspector;
reflection-diff is engine-agnostic, construct-probe currently tensorrt-only).
transformers is clean. The vllm/tensorrt divergences are GENUINE but are
staleness of the 3-week-old shipped schemas vs the refactored introspectors,
NOT engine drift: nested configs the old introspector flattened to `dict`
(`lora_config`, `quant_config`, `kv_cache_config`) now resolve as proper nested
types, and enums the old one missed (`config_format` -> `Literal[...]`,
`tokenizer_mode` -> `Literal[...]`) are now extracted. Semantic type
normalisation (added this wave) already absorbed the pure-representation cases
(enum vs Literal, string vs anyOf[string, path]); what remains is real
shipped-vs-current-producer drift. Implication: the shipped schemas should be
regenerated; for the study's FRESH per-cell discovery this drift does not arise
(discovery and gate use the same current introspector).

Caveat on semantics: re-running the introspector measures
"stored == current-producer output", which couples engine truth to
introspector-code version. For fresh study cells they coincide; for old shipped
artefacts, introspector improvements appear as divergences (as seen here).

## Track I (invariant gate) second cell: tensorrt 0.21.0

Ran the union+gate pilot on tensorrt 0.21.0 (container + mechanical d-ab output
+ PoC GT all already present; NO Opus passes for this cell).

- Sources: mech (56 candidates) + poc (75). Union 89 tolerant identities.
- Gate-confirmed: **3** identities (all via synthesis), 8 failed, 7 infra-error
  (BaseLlmArgs cross-field validators - build_config/lora consistency - that
  need multiple co-fields, same hardness class as 1.2.1), 38 skipped.

**The pipeline generalises** (ran end-to-end in the 0.21 container, synthesis
active), but the GT is THIN without Opus passes: compare tensorrt-1.2.1 (with 2
Opus passes) = 46 confirmed identities vs 0.21 (no Opus passes) = 3. This
quantifies that the Opus passes are LOAD-BEARING for GT depth - they supply the
`native_type` + probe coverage that makes most entries gateable; mechanical
synthesis alone clears only the simplest single-field constraints. A clean
bump-delta-recovery across the 0.21->1.2 boundary therefore needs Opus passes on
0.21 (and ideally the intervening 1.0/1.1 cells) too.

## Deterministic ceiling decomposition (tensorrt 1.2.1) - the core cost-frontier datapoint

CONSTRAINT-GRAIN recompute (supersedes the tolerant-grain 33%/74% this section
previously carried). Ablation: mechanical-only (improved-det-v2 + gate
probe-synthesis, ZERO Opus) vs the FROZEN gate-validated union GT (**212 union
constraints, 60 gate-confirmed**). The 60 gate-confirmed constraints are the
denominator - a constraint counts as "real" only once the runtime gate confirms
it. Non-tautological BECAUSE GT is the Opus+runtime union, not the mechanical
output: the 45 missed below were surfaced ONLY by the Opus passes / PoC; a
mech-only GT would be blind to them and falsely report ~100%.

| split | constraints | % of confirmed-60 |
|---|---|---|
| confirmed GT (denominator) | 60 | 100% |
| mech CONFIRMS today | 15 | 25.0% |
| mech SURFACED (strict constraint ckey) | 15 | 25.0% |
| mech MISSED entirely | 45 | 75.0% |
| surfaced-but-unconfirmed gap | **0** | 0% |

**Headline correction: at constraint grain the probe-synthesis gap VANISHES.**
Everything mech surfaces (15) it also confirms (15) - synthesis is reliable on
what the miner reaches. The old "33% confirm / 74% surfaced" were TOLERANT-grain
(leaf, coarse_bucket), and that 74% reproduces EXACTLY as mech's tolerant-key
reach here (34/46 = 73.9%): the tolerant key counted mech as "surfacing" a
confirmed constraint whenever it touched the same field+bucket, even when mech
asserted a DIFFERENT predicate value. So the old 33->74 "probe-synthesis gap" was
an ARTEFACT of identity over-collapse, not a real deterministic-engineering
opportunity. The entire deterministic deficit is a MINING-SCOPE gap: mech reaches
25% of the real constraints; the other 75% it never surfaces.

The 45 missed leaf fields are dominated by (a) PluginConfig literal fields
(gemm_plugin, bert_attention_plugin, gemm_swiglu_plugin, low_latency_gemm_*,
dtype, ...) the miner does not walk; (b) sampling/serving params (temperature,
top_k, top_p, best_of, max_batch_size, max_draft_len, free_gpu_memory_fraction,
...); and (c) cross-field / abstract-config constraints (allowed_backends,
per_worker_gpu_share, capacity_scheduler_policy) surfaced only by Opus. (a) is the
cheap mining-scope lever (lever 1 below, already built); (c) is the structural LLM
tail.

Deterministic levers, re-ranked at constraint grain:
1. **Miner walk-surface widening (BUILT, lever 1 below):** the highest-yield and
   ONLY mech lever still live - the probe-synthesis lever is retired (0 gap).
   Surfaces the (a) PluginConfig family. Measured below.
2. **Further walk-surface targets:** sampling/serving config classes (b) the
   miner still does not walk. Same mechanism as lever 1, next batch.
3. **Residual (cross-field / abstract / context-dependent configs (c)):** the
   irreducible LLM-mining tail - not cheaply surfaceable by static walk.

**Answer to "can deterministic reach ~1.0 for invariants?":** NO - and the
constraint-grain number is lower and more honest than the old tolerant read.
Bare mechanical reaches 25%; one cheap walk-surface lever lifts confirmed recall
to ~47% (below), with a structural tail (cross-field / abstract / context) that
resists cheap determinism. Contrast schema, which IS ~1.0 deterministic
(reflection is engine truth). So the product architecture (heavy-deterministic
mine + cheap runtime-gate verify) holds for schema outright; for invariants the
cheap-deterministic plateau sits well below 1.0 and a meaningful minority needs
LLM mining. Crucially the VERIFY half is cheap, deterministic, AND reliable (0
probe gap - the gate confirms what the miner surfaces); the open question is
purely how far deterministic MINING SCOPE can push surfacing.

## Round 0b lever 1 BUILT + measured (tensorrt 1.2.1): miner walk-surface widening

Implemented the highest-yield deterministic lever in the production static miner
(`engine_versions/tensorrt/v1_2_1/producers/static_invariant_miner.py`):
- walk `plugin/plugin.py::PluginConfig` (previously unwalked);
- `_literal_args` now unwraps `Optional[Literal[...]]` and resolves module-level
  `Literal` aliases (`DefaultPluginDtype = Literal[...]`);
- literal/strenum rules emit `{not_in: ...}` (firing condition) so they bucket as
  `membership`, matching the GT/gt_adapter convention (was misbucketing `presence`).

Re-measured at CONSTRAINT grain on the FROZEN denominator (212 union / 60
confirmed), separating the THREE effects the old "15->42" conflated. Combined
deterministic set = improved-det-v2 (mech, 110 candidates) + widened production
miner (prod, 44 candidates), gated together:

| effect (vs frozen 60-confirmed) | mech-only | + prod widening |
|---|---|---|
| SURFACES (strict ckey, of 60) | 15 (25%) | **30 (50%)** |
| CONFIRMS (recall of frozen GT, of 60) | 15 (25%) | **28 (46.7%)** |
| GT-GROWTH (confirmed OUTSIDE the 212 union) | 0 | **+13** |
| promoted union tail (unverified -> confirmed) | 0 | +1 |

Read separately (NOT summed into one recall number, which is the error the old
"42" made):
- **Recall of the frozen GT rises 25% -> 46.7%** (+13 newly-confirmed constraints
  the prod widening added on top of mech). Surfacing of the frozen GT doubles
  25% -> 50%. The combined det leaves a small 2-constraint surfaced-but-unconfirmed
  residual (30 surfaced, 28 confirmed) - prod's structural cases.
- **GT-GROWTH is separate and additive: +13 plugin-literal constraints
  gate-confirmed that the ENTIRE frozen 212-union lacked** (all from the prod
  walk; 19 PluginConfig literals surfaced, 13 confirmed outside the union, +1
  promoted from the unverified tail). This is the non-tautology mechanism working:
  a new method surfaces gate-confirmed invariants the union missed, so the GT
  itself should grow and be re-gated to 73 confirmed / 225 union in the next
  refresh. It is NOT counted as recall of the frozen denominator (doing so would
  re-introduce the tautology the re-base removed).

So the honest decomposition of the old "15 -> 42 confirmed": 28 recall of the
frozen GT + 13 GT-growth + 1 promoted tail = 42 total gate-confirmed deterministic
constraints, but only 28/60 = 46.7% is RECALL. Caveats: (a) prod hit 9
infra_errors (PluginConfig fields that don't construct standalone) - a ceiling on
the walk lever until multi-field construction lands; (b) the production per-version
miner is still much weaker than trial improved-det-v2 (44 vs 110 candidates) - it
has not absorbed the trial R&D primitives (deferred to milestone end).

## Fan-out readiness (what the full 15-cell x 2-track matrix needs)

- **Per-version producers exist for only ~6 versions** (transformers v4_57_3 /
  v5_3_0 / v5_6_2; vllm v0_7_3 / v0_16_0 / v0_18_1 / v0_19_1; tensorrt v0_21_0 /
  v1_2_0 / v1_2_1). The locked window (transformers 5.6-5.10, vllm 0.18-0.22,
  tensorrt 0.20/0.21/1.0/1.1/1.2) needs ~9 NEW per-version producers
  (schema_introspector + static_invariant_miner). This is the main blocker and
  overlaps the in-flight engine-knowledge-as-data refactor.
- **Containers**: tensorrt 0.20/1.0/1.1 are NGC pulls; vllm 0.18/0.20/0.21/0.22
  are Docker Hub pulls; transformers 5.7-5.10 are builds. Cheaper than feared.
- **Opus-pass GT** is the per-cell cost driver and (per above) is load-bearing,
  so it cannot be skipped for a quality GT.

Recommended sequencing: build/port the missing per-version producers first
(unblocks both tracks per cell), then per cell run schema-gate (cheap) +
Opus-passes + union+gate (invariants). The tensorrt 0.21->1.0 major boundary is
the headline bump-robustness datapoint and should be prioritised once 1.0 has a
producer + container + Opus passes.
