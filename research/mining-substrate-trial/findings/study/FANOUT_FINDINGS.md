# Fan-out wave 1 findings - cross-engine schema + second invariant cell

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

Ablation: mechanical-only (improved-det-v2 + gate probe-synthesis, ZERO Opus)
vs the full gate-validated union GT (46 confirmed identities). This measurement
is only meaningful BECAUSE GT is the Opus+runtime union, not the mechanical
output - else mechanical recall would be 1.0 tautologically. The 12 "missed"
identities below were surfaced ONLY by Opus; a mechanical-only GT would be blind
to them and would falsely report 100%. So the thorough (non-tautological) GT is
what makes the deterministic ceiling measurable at all.

| split | identities | % of GT |
|---|---|---|
| confirmed GT (denominator) | 46 | 100% |
| mech CONFIRMS today | 15 | 33% |
| mech SURFACED (det-reachable if probing were perfect) | 34 | 74% |
| mech MISSED entirely (12/12 surfaced by Opus) | 12 | 26% |

The 33% is a FLOOR, not a deterministic ceiling. The gap decomposes into two
separately-addressable, mostly-deterministic levers:
- **Probe-synthesis gap (33% -> 74%, ~19 identities):** the miner already
  SURFACED these; synthesis just can't probe them yet (cross-field, dispatch,
  presence, type). Pure deterministic engineering - multi-field probes,
  type-probes, dispatch-aware construction. No LLM.
- **Mining-scope gap (74% -> ~100%, 12 identities):** the miner never surfaced
  these. Inspection shows most are PluginConfig literal fields
  (gemm_plugin, bert_attention_plugin, gemm_swiglu_plugin, low_latency_gemm_*,
  allowed_backends, dtype, lora_ckpt_source) - i.e. a WALK-SURFACE gap (the
  miner doesn't walk tensorrt_llm.plugin), also deterministically fixable by
  extending the miner's module surface. Only a residual is expected to need LLM
  mining.

Implication for the production CI architecture (heavy-deterministic mine +
cheap-LLM verify): looks ACHIEVABLE. The verify/validate half is ALREADY cheap
and deterministic (the runtime gate). The open empirical question is the exact
deterministic mining+probing plateau once both levers are pushed (Round 0b) -
the residual above plateau is the irreducible per-bump LLM-mining need. Opus
passes remain required as GT CONTRIBUTORS (denominator), even when the
production miner is the cheap method under evaluation.

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
