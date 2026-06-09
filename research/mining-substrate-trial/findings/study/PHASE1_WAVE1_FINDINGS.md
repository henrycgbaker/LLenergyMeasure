# Phase 1, Wave 1 - findings (minimal LLM-extend integration probe)

Status: COMPLETE. Pre-registration: `PHASE1_WAVE1_PREREG.md` (locked design point).
Artifacts: `phase1_wave1/results/*.json` (per cell x rung), `phase1_wave1/llm_proposed/*.yaml`
(the gated LLM output). Runner: `scripts/phase1/wave1.py`.

## Integration result: SUCCESS

The PoC LLM-extend harness now runs end-to-end against the NEW runtime-gated GT
and the REAL production gate (`validate_invariants.py` in-container via
`study_gt_pilot` load+gate), replacing the PoC hallucination proxy. Both rungs
ran: OSS (`gemma3:12b`, Ollama, chunked) and Opus (Agent tool, whole validator
source in one call/cell). The gate returns real per-invariant verdicts
(confirmed/failed/skipped/infra). Port fixed (11434); tensorrt 1.2.1 cell added.

## Results (4 cells x rungs)

| cell | rung | raw | net-new (post floor-dedup) | confirmed | failed | skipped | infra | lift over floor |
|---|---|---|---|---|---|---|---|---|
| vllm 0.19.1 | gemma3:12b | 114 | 32 | 0 | 11 | 18 | 3 | 0 |
| vllm 0.19.1 | opus | 95 | 64 | 0 | 5 | 54 | 5 | 0 |
| tensorrt 1.2.1 | gemma3:12b | 30 | 9 | 0 | 2 | 5 | 2 | 0 |
| tensorrt 1.2.1 | opus | 33 | 20 | 0 | 0 | 12 | 8 | 0 |

Floor (improved-det-v2) tolerant recall vs GT: vllm 44/74, tensorrt 35/60.

## Headline finding

**det-then-llm-extend, with the current auto-synthesizing gate, produced ZERO
gate-confirmed lift over the deterministic floor - on BOTH cells and BOTH rungs
(cheap gemma3:12b AND the Opus ceiling).** The LLM proposes many net-new
invariants; none auto-confirm.

## Why (diagnosis, validated offline + by spot-test)

After floor-dedup removes the single-field bounds the deterministic floor already
covers, the LLM's net-new contributions are dominated by forms the gate's
single-field probe-synthesizer cannot turn into a probe. For vllm 0.19.1 Opus
(64 net-new), the breakdown:

- 23 cross-field (multi-key guards: `world_size == tp*pp*cp`, `min_tokens <=
  max_tokens`, `max_cpu_loras >= max_loras`, ...)
- 19 presence-only (`{present: true}` - no violating value to synthesise)
- 8 bool/exact-value conditional (`enable_eplb: true` + dependency)
- 4 type / other (`type_is_not`, empty `not_in`)
- 10 single-field SYNTHESISABLE - and of these 10, **0 confirmed** (5 failed, 5
  infra): the floor already covers the confirmable single-field surface, so the
  synthesisable residual is redundant-or-wrong.

The 54 "skipped" = `skipped_unsynthesizable`. A spot-test hand-authoring
`kwargs_positive/negative` for one cross-field invariant (EPLB-requires-expert-
parallel) did NOT confirm either - it hit `infra_error` at construction (the same
entangled-required-args / construction issue seen in the self-confirm fan-out).

## Honest conclusion (do not overclaim in either direction)

The auto-gate **cannot currently adjudicate the LLM's cross-field / conditional /
presence tail at all** - blocked by BOTH (a) single-field probe-synthesis scope
and (b) config-construction infra for many classes. So:

- This is NOT "the LLM adds nothing": Opus surfaced ~50 net-new vllm invariants
  (cross-field relations, feature-gate guards, dormant/deprecation warnings) that
  are plausibly real and that the deterministic miner does not reach - the
  genuine LLM tail.
- It is also NOT "validated GT-growth": the gate confirmed none of them. Their
  validity is UNRESOLVED by this assembly + gate.
- The GT's own cross-field constraints were confirmed in Round 0 via
  hand-authored kwargs; the `wg_extend` prompt emits only `match.fields`, so the
  auto-synthesis path cannot re-confirm them.

The bottleneck is the VALIDATION path, not LLM recall.

## Cost (wave-1, lean OSS-first)

- gemma3:12b (GPU-energy proxy = wall): vllm 595s, tensorrt 168s (chunked,
  14 + 4 chunks). Single A100, single-tenant.
- Opus (token-$): ~142k (vllm) + ~64k (tensorrt) subagent tokens for the
  canonical write-to-file runs (whole-source single call/cell). No GPU.

## Caveats

- N=2 cells, single-shot, one OSS rung - DIRECTIONAL, not a frontier point.
- `0 confirmed` = `0 auto-gateable`, NOT `0 valid` (see conclusion).
- Floor-dedup is by tolerant (leaf,bucket); it removes LLM re-emissions of
  floor-covered fields before scoring, so "net-new" is genuinely-new leaves.

## Deviations from pre-registration (logged)

- Opus rung run over the WHOLE validator source in one call/cell (its 200k
  context allows it), vs the OSS rung's chunking (16k ctx). Intended: the Opus
  ceiling anchor sees the full source; the chunk/call-shape difference is a
  known assembly detail, not a confound for the integration-probe goal.
- Two Opus preview calls preceded the canonical write-to-file calls (no
  SendMessage to persist the first outputs); the committed `llm_proposed/*.yaml`
  are the canonical (write-to-file) runs.

## What wave 1 sets up (the wave-2 questions)

1. **Validation-path upgrade is the lever, not model tier.** Test whether
   (a) prompting the LLM to emit `kwargs_positive/negative` for cross-field
   invariants + (b) a construction-robust, cross-field-aware gate unlocks the
   tail. If the cross-field tail confirms with kwargs, the LLM's value is real
   and large; if not, the auto-gate ceiling is the binding constraint.
2. The deterministic floor already saturates the auto-gateable single-field
   surface (lift 0 from both rungs) - so LLM value REQUIRES richer validation;
   whether it clears the cost axis is the open production question.
3. Only then is a tier sweep (32B/70B) or other assemblies/roles worth the spend.
