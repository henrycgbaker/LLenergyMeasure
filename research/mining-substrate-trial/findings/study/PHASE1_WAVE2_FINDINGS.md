# Phase 1, Wave 2 - findings (validation-path lever; Arm A)

Status: Arm A COMPLETE. Arm B (construction-robust gate) DEFERRED - justified
below. Pre-registration: `PHASE1_WAVE2_PREREG.md`. Artifacts:
`phase1_wave2/results/*` (per cell x rung + the confirmed-invariant lists),
`phase1_wave2/llm_proposed/*` (Opus kwargs-bearing proposals). Runner:
`scripts/phase1/wave1.py` (`--rung oss --prompt-file` added for the wave-2
prompt).

## The lever WORKS

Wave 1 left the LLM's net-new cross-field/conditional tail at 0 gate-confirmed
(the single-field auto-synthesis gate could not probe it). Wave 2 had the LLM
ALSO emit constructible `kwargs_positive/negative`. Result, Opus rung:

| cell | wave-1 (match-only) | wave-2 Arm A (Opus + kwargs) |
|---|---|---|
| vllm 0.19.1 | 0 confirmed | 8 confirmed (7 REAL cross-field + 1 spurious) |
| tensorrt 1.2.1 | 0 confirmed | 1 confirmed (REAL) |

kwargs-emission turns the cross-field tail gate-confirmable. The bottleneck was
the validation path, exactly as wave 1 localized.

## Confirms adversarially verified (mandatory - cross-field bypasses attribution)

The gate's confirm-attribution is single-field; cross-field confirms bypass it,
so every one was source-verified by an independent Opus reviewer. Verdict:
**8 REAL / 1 SPURIOUS / 0 unverifiable of 9.**

| id | rule (file:line) | verdict |
|---|---|---|
| repetitiondetectionparams min_pattern_size<=max | sampling_params.py:128-138 | REAL |
| structuredoutputsparams exactly-one-constraint | sampling_params.py:55-76 | REAL |
| samplingparams min_tokens<=max_tokens | sampling_params.py:485-489 | REAL |
| structuredoutputsconfig disable_any_whitespace backend-gate | config/structured_outputs.py:64-68 | REAL |
| structuredoutputsconfig disable_additional_properties backend-gate | config/structured_outputs.py:69-73 | REAL |
| eplbconfig log_balancedness_interval>0 | config/parallel.py:90-91 | REAL |
| loraconfig max_cpu_loras>=max_loras | config/lora.py:103-107 | REAL |
| nvfp4gemmconfig allowed_backends non-empty (tensorrt) | llm_args.py:499-500 | REAL |
| eplbconfig async-requires-default-policy | config/parallel.py:88-89 | **SPURIOUS** |

The spurious confirm is structurally forced and characterises a NEW gate
soundness gap: when a cross-field invariant's `kwargs_positive` violates the
relation by setting a constrained Literal/enum field OUT of its allowed set, the
field-level `literal_error` fires BEFORE the cross-field model-validator - the
gate sees a raise and confirms, but for the wrong reason. (`EPLBPolicyOption =
Literal["default"]` is single-valued, so the labelled "async requires default
policy" rule is logically unreachable via valid construction.) FIX = extend the
wave-1 attribution hardening to cross-field: match the raised error's locus to
the labelled rule, and treat relations over single-valued Literals as
a-priori-unreachable. DONE (commit `fix(gate): attribute cross-field confirms
by error locus`): cross-field confirms are now rejected when the positive raised
a field-level pydantic error; re-gate drops the eplb spurious confirm (vllm
8 -> 7) and keeps the 7 source-verified rules.

## Cost-vs-coverage: small-OSS vs Opus (the cost story)

| rung | vllm conf | tensorrt conf | vllm failed | cost |
|---|---|---|---|---|
| deterministic floor | (covers single-field surface; 44/74 vllm tolerant) | | | ~0 (CPU-sec) |
| gemma3:12b + kwargs | 2 (UNVERIFIED) | 0 | 17 of 25 | ~500s/cell A100 |
| Opus + kwargs | 8 (8 REAL) | 1 (REAL) | 3 | ~142k+64k tok |

**The cheap rung does NOT clear the cross-field tail.** gemma3:12b failed 17 of
25 deduped proposals - it cannot reliably produce constructible probes that fire
the labelled rule, and its 2 confirms are unverified (low precision regardless).
The cross-field/conditional extraction + correct-kwargs generation needs
Opus-tier capability. So for this assembly (det-then-llm-extend + kwargs), the
LLM's value (the cross-field tail) is REAL but reachable only at Opus cost;
deterministic + small-OSS own the cheaper single-field surface and do not bridge
to the cross-field tail.

## Arm B (construction-robust gate): DEFERRED

The Opus residual is 5 vllm `infra_error` (+ 27 skipped null-kwargs the LLM could
not make constructible) - the entangled config classes (ModelConfig / VllmConfig
/ ParallelConfig / SchedulerConfig) that need a live model / distributed init to
construct. Making those generically gateable is exactly the construction-
robustness Arm B would build - but that cost would be paid by the production gate
on EVERY bump, and per the north-star (CI-affordable) that complexity is likely
its own no-go. The lever is already demonstrated (8 real confirms without Arm B,
on the constructible-in-isolation classes). So Arm B is deferred: build it only
if a concrete frontier point needs those entangled classes AND the construction
cost proves cheap. Not pursued speculatively.

## Decision-gate outcome (per the pre-registration)

UNLOCK. The LLM cross-field/conditional tail is REAL and gate-confirmable - 8
verified-real constraints the deterministic floor structurally cannot reach.
Two consequences:
1. Production design for capturing the tail = kwargs-emitting LLM + the existing
   gate (+ the bounded cross-field attribution fix). The tail costs Opus; the
   cheap rung does not reach it.
2. Next question (if cost-optimising the tail): a MID-tier sweep (32B/70B) to
   find the cheapest rung that bridges the gemma->Opus gap. Needs provisioning
   (not pulled) + spend. Alternatively, FOLD the 8 verified-real cross-field
   confirms as GT-growth (they are real, gate-confirmed, source-verified).

## Consolidation (done 2026-06-09)

Per the decision-gate "consolidate" path: (1) the cross-field error-locus
attribution fix landed (above); (2) the 8 verified-real confirms were FOLDED into
the GT via the fixed gate (the eplb spurious one auto-excluded): vllm 0.19.1
n_confirmed 98 -> 105 (+7 cross-field), tensorrt 1.2.1 74 -> 75 (+1). Verified
additive (0 lost, metadata preserved); entries carry `source: llm`, the full
multi-field `match` (re-gateable), and a `foldins` audit batch. These are the
first CROSS-FIELD constraints in the GT - a class the deterministic floor
structurally cannot reach, now banked as runtime-confirmed knowledge.

The tier sweep (32B/70B) is NOT run here: it belongs in the later systematic
LLM-pattern phase (tiers x assemblies x call-shapes), which wants a real agentic-
orchestration framework (LangGraph/LangChain) rather than this minimal single-
call harness. Waves 1-2 established what that phase needs: the harness<->gate
integration, the validation-path-is-the-bottleneck finding, the kwargs-emission
lever, and the det/OSS/Opus cost-comparison method.

## Caveats

- N=2 cells, single-shot, one OSS rung. Directional.
- gemma's 2 confirms are UNVERIFIED (the conclusion - cheap insufficient - holds
  regardless, given its 17/25 failure rate).
- The 8 REAL confirms are cross-field; folding them needs the cross-field
  attribution fix in place first so future re-gates do not re-admit the spurious
  class.
