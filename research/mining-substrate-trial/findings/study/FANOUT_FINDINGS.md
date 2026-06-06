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
(`lora_config`, `quant_config`, `kv_cache_config`) now reflect as a typed
reference (`$ref` to the nested model); the gate compares that reference by
target identity (`ref:LoraConfig`), so flatten-vs-nested drift is flagged rather
than silently swallowed - though the gate checks ref identity, not the
dereferenced nested fields (those reflect as their own sections). Enums the old
one missed (`config_format` -> `Literal[...]`,
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

## GT re-gate: folding the lever-1 growth back in (tensorrt 1.2.1)

The production widened miner is now a committed per-cell union source
(`prod_static_miner.yaml`, discovered by `study_gt_pilot.configure`), and the
1.2.1 GT has been RE-GATED with it included. New denominator: **74 confirmed /
228 union** (was 60 / 212) - the +14 confirmed = the 13 plugin-literal
constraints confirmed outside the old union + 1 promoted from the unverified
tail. GT-growth vs the PoC N=1 GT rose 23 -> 37.

Circularity caveat (the reason the deterministic-ceiling section above keeps the
pre-re-gate FROZEN 60/212 denominator and its dedicated per-method gate runs,
rather than reading recall off this re-gated GT): with `prod` and `mech` both in
the union, **only 45 of the 74 confirmed constraints have an independent
(Opus/PoC) contributor** - the other 29 rest on deterministic sources alone.
Measuring "deterministic recall" against a denominator that now includes the
methods under evaluation would be circular. So the re-gated 74/228 is the
best-knowledge GT for downstream use (e.g. cross-version comparison), while the
ceiling numbers stay anchored to the method-independent frozen denominator. The
GT-level source tally is also noisy as a recall proxy: at constraint grain a
deterministic method's predicate-value variant of a real constraint lands in a
SEPARATE ckey from the Opus-confirmed one, so contributing-source counts
understate true agreement (another reason the dedicated mech-only / combined-det
gate runs are the correct ceiling instrument).

## tensorrt 0.21.0 -> 1.0.0 -> 1.1.0 -> 1.2.1: the bump-robustness gradient

Established GT for the full tensorrt window (0.21.0, 1.0.0, 1.1.0; 1.2.1 already
done) via the same union+gate pipeline, each new cell with 2 Opus passes
(entry-point call-graph + class-hierarchy/MRO walk) as GT contributors, then an
INDEPENDENT adversarial source-review of every confirmed entry per cell (each
reviewer opens the cited source line and tries to refute the predicate / catch a
gate false-confirm / find fabrication). The 4-point window spans ONE major
boundary (0.21 -> 1.0) and two minor bumps (1.0 -> 1.1 -> 1.2.1).

### Per-cell GT (runtime-gated)

| cell | union | confirmed | GT contributors (constraints) | adversarial review |
|---|---|---|---|---|
| 0.21.0 | 164 | 18 | passA 74, passB 62, mech(d-ab) 55, poc 73 | 17/18 REAL, 0 false-confirm, 0 fab, 1 mis-stated (redundant) |
| 1.0.0 | 123 | 21 | passA 75, passB 63, mech 43 (no PoC) | 21/21 REAL, 0 false-confirm, 0 fab |
| 1.1.0 | 84 | 24 | passA 77, passB 70 (no mech, no PoC) | 24/24 REAL, 0 false-confirm, 0 fab |
| 1.2.1 | 228 | 74 | re-gated (mech 107, passA 97, passB 98, poc 90, prod 42) | prior: 100% on ~22-entry sample |

0.21.0 was previously thin (128 / 3, NO Opus passes); the 2 passes lift it to
164 / 18, re-confirming Opus passes are load-bearing for GT depth. Union sizes
are NOT comparable across cells (they depend on which extra sources -
mech / poc - happen to exist per cell; 1.1.0 has only the two Opus passes). For
cross-version comparison use the OPUS basis below (passA+passB, present in every
cell).

Reviews confirmed the gate's ANTI-FALSE-CONFIRM machinery works: the
attribution-hardening (the gate forces `positive_confirmed=False` unless the
raised message names the leaf field) closes the "CUDA-incidental-error" hole that
the unconditional `torch.cuda.get_device_properties(0)` probe in `validate_dtype`
would otherwise open - so args-model Literals confirm for the RIGHT reason even
though construction touches CUDA (gate runs `--gpus all`). Caveats both reviews
surfaced (IDENTITY layer, NOT correctness): constraint grain UNDER-merges
cross-source re-encodings (the same source rule encoded by two sources with
divergent predicate values does not merge), so confirmed ENTRY counts run ~30%
over distinct RULES (~12 behind 0.21's 18, ~17-19 behind 1.0's 21); and one
mis-stated-but-redundant 0.21 `lora_ckpt_source` numeric encoding (real rule,
correctly captured by its passB twin). A citation-keyed canonicalisation pass
would tighten the first; both are filed open items.

### Bump-robustness gradient (matched by tolerant key = leaf field + coarse bucket)

OPUS basis (passA+passB only - the apples-to-apples set present in every cell,
free of the per-cell mech/poc source-availability confound). This is the headline
instrument:

| bump | persist | added | dropped | rebounded (of persist) |
|---|---|---|---|---|
| 0.21->1.0 (**MAJOR**) | 43/81 = **53%** | 21 | 38 | 18 (42%) |
| 1.0->1.1 (minor) | 60/64 = **94%** | 11 | 4 | 12 (20%) |
| 1.1->1.2.1 (minor) | 65/71 = **92%** | 28 | 6 | 9 (14%) |

CONFIRMED basis (runtime-verified subset; small-N, corroborates direction):
0.21->1.0 = 10/14 (71%); 1.0->1.1 = 15/17 (88%); 1.1->1.2.1 = 23/23 (100%).

Reading:
- **The gradient cleanly ISOLATES the major-boundary spike.** On the
  apples-to-apples Opus basis, the 0.21 -> 1.0 MAJOR bump drops or changes ~47%
  of mined knobs (persist 53%, 38 dropped), while BOTH minor bumps within 1.x
  persist ~92-94% (4-6 dropped). The major boundary churns roughly 8x as many
  knobs as a minor bump. (The raw UNION basis muddies this - e.g. 1.0 -> 1.1
  looks like 64% only because 1.0 carried a mech source that 1.1 lacks; the Opus
  basis removes that artefact.)
- **RE-BOUNDED knobs (same field+bucket, CHANGED valid-set/bound) are the
  silent-staleness cases**: a knob still present but with a different constraint,
  where stale mined knowledge would be WRONG, not merely incomplete. Even among
  knobs that PERSIST, the major bump re-bounds 42% (18/43) vs 14-20% for the
  minor bumps. These are exactly what the runtime gate catches by re-validating
  each carried-over constraint against the live engine.
- **1.1.0 sits with 1.0, not 1.2.1.** Independent confirmation from both 1.1
  Opus passes: at 1.1 PluginConfig is still pre-pydantic (assert-based property
  setters), SamplingParams has no top_p/top_k/temperature range checks,
  CacheTransceiver has no timeout Fields, and `validate_build_config` RAISES
  (1.0-style) rather than warn/clamp (1.2.1-style). The big 1.2.x feature
  additions (PluginConfig pydantic-isation, SamplingParams ranges, Nvfp4/Ray/
  sparse-attn families) all land between 1.1 and 1.2.x - the 1.1 -> 1.2.1 delta
  is addition-heavy (28 Opus / many union new knobs) on a stable existing base.

Caveats: confirmed-basis percentages are small-N; the OPUS basis (64-81 knobs)
is the robust signal. Field renames (e.g. `speculative_model` ->
`speculative_model_dir`) show as drop+add, not rebound, so true semantic-rebound
is slightly understated. Raw: /tmp/cross_major_delta.json.

Implication for the north star: a major bump reorganises the config-validation
surface enough that ~half of mined knobs vanish or change and ~40% of the
SURVIVORS silently re-bound, whereas minor bumps leave ~92% of knowledge intact
and are addition-dominated. So a cheap runtime GATE that re-validates every
carried-over constraint against the live engine is necessary to catch the
dangerous (rebounded) cases on majors - and it is already cheap. The mining half
mostly needs to MINE-NEW on minors (cheap deterministic walk-surface widening for
the mechanical tail; LLM mining for the structural residual) and additionally
RE-MINE the dropped/rebounded surface on a major.

## Cross-engine generalisation: vllm 0.18.1 -> 0.19.1

Carried the methodology to a SECOND engine (vllm) to test whether the bump
findings are tensorrt-specific. The study window has no non-tensorrt MAJOR
boundary, so this tests minor-bump behaviour. Two cells, each 2 Opus passes
(entry-point + class-hierarchy), runtime-gated in the vllm container (CPU-only
dispatch), then full adversarial source-review.

### Per-cell GT

| cell | union | confirmed | sources | adversarial review |
|---|---|---|---|---|
| vllm 0.18.1 | 145 | 94 | passA 76, passB 69 (no mech, no PoC) | 91/94 REAL, 0 false-confirm, 0 fab, 3 mis-stated predicate_value |
| vllm 0.19.1 | 249 | 90 | passA 82, passB 64, mech 105, poc 79 | 90/90 REAL, 0 false-confirm, 0 fab |

The gate generalises cleanly to a different engine: vllm config objects construct
CPU-only, so the CPU-only vllm dispatch confirms many invariants (94 / 90, the
deepest cells in the study). The 0.19.1 reviewer additionally RE-RAN 50+ entries
end-to-end in-container to verify each fires for the right reason. Cross-engine GT
integrity: **243/247 confirmed entries verified REAL** across all six cells
(tensorrt 62/63 + vllm 181/184). (0.19.1 had 63 infra_errors = configs needing a
real model dir like ModelConfig; a recall ceiling, not a confirmed-correctness
issue. 0.18.1 has no PoC/mech, so all 94 confirmed are Opus-only.)

### vllm minor-bump delta (Opus basis, tolerant-key match)

| basis | persist | added | dropped | rebounded (of persist) |
|---|---|---|---|---|
| OPUS (passA+passB) | 87/111 = **78%** | 24 | 24 | 31/87 = **36%** |
| confirmed | 57/75 = 76% | 9 | 18 | 19/57 |

Reading (with the load-bearing caveat):
- **vllm 0.18->0.19 churns MORE than a tensorrt 1.x minor (92-94% persist) but
  LESS than the tensorrt major (53%).** The caveat is decisive: vllm uses 0.x
  versioning, where the MINOR digit is the breaking-change position - a vllm
  0.18->0.19 bump is effectively a feature release, NOT semantically a "minor"
  like tensorrt 1.0->1.1. So vllm's minor sitting between tensorrt's minor and
  major is exactly what the versioning conventions predict; it is NOT evidence
  that vllm churns more "for the same kind of bump."
- **The survivor RE-BOUND rate generalises and is the robust cross-engine
  signal:** 36% of vllm knobs that persist 0.18->0.19 changed their bound/
  allowlist - close to the tensorrt MAJOR's 42% and well above tensorrt minors'
  14-20%. Silent re-bounding of surviving knobs is an engine-independent hazard,
  reinforcing that the runtime gate (re-validate each carried-over constraint
  against the live engine) is necessary across engines, not a tensorrt quirk.

### Systemic finding: what "gate-confirmed" does and does not guarantee

The 0.18.1 review surfaced a real bound on the gate's guarantee (3 mis-stated
entries): the gate verifies BINARY fire/pass BEHAVIOUR (a bad value fires, a good
value passes) but does NOT cross-check the recorded `predicate_value` against the
cited source. So an entry can gate-confirm correctly while its recorded allowlist/
bound is slightly off, whenever the synthesised/authored positive+negative kwargs
happen to straddle the TRUE boundary (e.g. a 6-value allowlist recorded for an
8-value Literal still confirms, because the probed out-of-set value fires and the
in-set value passes). Implication: gate-confirmation establishes the constraint
EXISTS and is roughly located, not that its recorded boundary is exact. For the
product this is acceptable (existence + rough location is the high-value signal),
but a precise-boundary guarantee would need the gate to additionally probe AT the
recorded boundary edges. Filed as a known gate-scope limitation.

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
