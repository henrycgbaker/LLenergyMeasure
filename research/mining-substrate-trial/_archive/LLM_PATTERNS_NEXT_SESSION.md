# Systematic LLM-pattern phase - next-session bootstrap

Start a NEW context from this file. It is self-contained; the authoritative
detail is in the docs it points to. Branch `spike/engine-knowledge-as-data`,
HEAD `454a25f4` (verify with `git log -1`; `git fetch origin` first if anything
looks off).

## Goal of this phase

Phase 1's WIDE NET, done systematically: empirically test patterns of directed
LLM use for engine-config invariant mining, across the STUDY_DESIGN Section 8
design space, scored by the real runtime gate. The headline deliverable is the
MODEL-SIZE x WORKFLOW-SHAPE matrix (see "CORE DELIVERABLE" below): map what each
model size (small/mid/large OSS + Opus) achieves at each workflow shape
(all assemblies x call-shapes, INCLUDING pure-LLM). Waves 1-2 ran only the
narrowest slice (det-then-llm-extend, single call-shape, small-OSS + Opus); this
phase fills the grid. Use a real agentic-orchestration framework (LangGraph /
LangChain) for the patterns that need it (closed-loop, self-consistency,
ensemble-vote, agentic call-shape) - NOT the minimal single-call harness.

North star (unchanged): a cheap, CI-affordable workflow that keeps well-validated
engine-config knowledge (schema + invariants) current across version bumps; the
engine owns its SSOT; a runtime gate validates mined knowledge in-container
("observe, don't re-encode"). Cost is understood ORDINALLY via the det/OSS/Opus
comparison (deterministic ~free < small-OSS < Opus) - the research question is
"does the cheap rung suffice, or do you need the big model?". Do NOT build a
plotted per-bump-$ Pareto frontier (user-confirmed); the tier x pattern
comparison IS the cost story.

## The design space (STUDY_DESIGN Section 8)

- Role: {extract, extend-residual, gate, diagnose, diff-review, curate}
- Assembly: {det-only, llm-only, det-then-llm-extend, llm-then-det-gate,
  closed-loop, ensemble-vote, self-consistency, det-then-llm-patches-det}
- Call-shape: {single, k-vote, chunked, chained, agentic}
- Tier (cost gradient): {OSS-small 7-14B, OSS-mid ~32B, OSS-high ~70B, Opus}
The full grid x 15 cells is infeasible: use FRACTIONAL SAMPLING + the
anti-local-optimum guards (a protected deepen-before-prune quota for late-payoff
assemblies like closed-loop/self-consistency; a planned assembly x bump-shape
block - the cliff is that interaction). Per-phase PRE-REGISTRATION is mandatory
(locked prompts, pinned model+container digests, deviation log, no mid-wave
changes) - see PHASE1_WAVE1_PREREG.md / PHASE1_WAVE2_PREREG.md for the format.

## What waves 1-2 BANKED (the scaffolding this phase builds on)

Read PHASE1_WAVE1_FINDINGS.md + PHASE1_WAVE2_FINDINGS.md. Key results:
1. Harness<->gate integration WORKS: LLM proposals (floor-schema YAML) gate
   through the REAL production gate (`scripts/validate_invariants.py` via
   `study_gt_pilot` load+gate), scored vs the runtime-gated GT.
2. The VALIDATION PATH is the bottleneck, not LLM recall: the det floor owns the
   gate-synthesizable single-field surface; the LLM's value is the cross-field /
   conditional tail the single-field auto-synth gate cannot probe.
3. The kwargs-emission LEVER unlocks that tail: have the LLM emit constructible
   kwargs_positive/negative (locked prompt
   `phase1_wave2/wg_extend_kwargs_prompt.md`). Opus: 0 -> 8 verified-real
   cross-field confirms; gemma3:12b FAILS it (17/25 failed, 2 unverified). So the
   cross-field tail is real but reachable only at OPUS cost - the central open
   question this phase widens (does a mid 32B/70B bridge the gemma->Opus gap?).
4. SOUNDNESS: cross-field confirms bypass single-field attribution; the gate now
   rejects a cross-field confirm whose positive raised a FIELD-level pydantic
   error (the `_is_cross_field` + `exception_locs` check in validate_invariants).
   EVERY cross-field confirm still needs adversarial source-verification before
   counting/folding (mandatory; it caught 1 spurious in wave 2).
5. The 8 verified-real cross-field constraints are FOLDED into the GT (vllm
   0.19.1 = 105 confirmed, tensorrt 1.2.1 = 75) with `source: llm` + foldins.

## FIRST STEPS for this phase

1. SET UP THE FRAMEWORK: check if LangGraph/LangChain is installed (likely NOT;
   the current harness is direct ollama/Agent). Decide the orchestration
   substrate. The OSS rung talks to Ollama (`http://localhost:11434`); the Opus
   rung is the Agent tool (Anthropic-side). The gate is the scoring oracle
   (reuse `scripts/phase1/wave1.py` gate+score, or the `study_gt_pilot` path).
2. PROVISION TIERS: only `gemma3:12b` is pulled. Pull a ~32B (e.g.
   qwen2.5-coder:32b) + ~70B (e.g. llama3.1:70b) for the gradient. ~60GB disk.
3. PRE-REGISTER the first systematic wave: pick a SCOPED subset of
   assemblies x call-shapes x tiers x roles (fractional), the cells, the metrics
   (recall vs GT, gate-confirmed precision, cross-field-verified GT-growth, cost
   = tier comparison + wall/tokens), and the mandatory adversarial-verify gate.
   Get user sign-off BEFORE running (the wave-1/2 discipline).
4. RUN -> verify (mandatory cross-field check) -> record findings -> fold
   verified-real growth -> decide next wave (deepen vs prune per the stopping
   rule).

## CORE DELIVERABLE: the model-size x workflow-shape matrix (user-confirmed)

The headline of this phase is a MATRIX, not one-axis sweeps: cross
{tier: small-OSS (gemma3:12b, done) x mid-OSS ~32B x large-OSS ~70B x Opus} with
{workflow shape: ALL the assemblies x call-shapes, INCLUDING pure-LLM
(llm-only, no deterministic floor)}. Goal = get a sense of WHAT IS ACHIEVABLE at
each (model size, workflow shape) cell - recall/coverage vs the GT + cost
(ordinal tier comparison). So:

- Provision mid (~32B) + large (~70B) OSS up front (not optional) - they are
  tested across EVERY shape, same as Opus.
- Include PURE-LLM (llm-only assembly): the LLM extracts the full catalogue from
  source with NO det floor. Waves 1-2 only did det-then-llm-extend; pure-LLM is
  the untested "what can the model do alone" datapoint and is required for the
  size x shape map. (The PoC has a `pure_b_prompt.md` locked prompt; the
  kwargs-emission requirement from wave 2 should carry into every shape so the
  gate can probe cross-field.)
- Shapes to cover (fractional sampling, but spanning the space): pure-LLM,
  det-then-llm-extend (+kwargs, done at small/Opus), llm-then-det-gate,
  self-consistency (k-vote), ensemble-vote, closed-loop; call-shapes single vs
  k-vote vs chunked. Roles: extract primary; diagnose/diff-review as the
  self-update-adjacent roles (scored on caught-a-silent-collapse, NOT recall).

Pre-register the matrix in tranches (it is large x 15 cells - sample cells +
shapes per the discipline), but the deliverable is the filled size x shape grid:
"this size at this shape achieves this coverage at this cost." Mandatory
adversarial cross-field verification on every confirm, every cell.

## STRONGEST un-run north-star item (parallel option, per the 2026-06-09 audit)

The self-update / degradation-signal binary on the tensorrt 0.21->1.0 MAJOR bump:
"does carried-over knowledge auto-re-validate across the bump with NO human edit,
and does gate-acceptance-rate FIRE on the churn?" This is the actual product
property and nothing has tested it. It is more north-star-relevant than another
coverage wave and could run in parallel with / instead of the pattern sweep.
STUDY_DESIGN Section 5 "dedicated cells".

## Infra pointers (so the new session doesn't rediscover)

- Driver venv: `/tmp/round0b_venv` (tree-sitter etc.). If wiped: `uv venv
  /tmp/round0b_venv && uv pip install --python /tmp/round0b_venv/bin/python
  tree-sitter tree-sitter-python pyyaml`.
- Ollama: `http://localhost:11434` (the PoC harness hardcodes :11435 - wave1.py
  overrides). 4xA100-40GB visible on host. Opus via Agent tool (no GPU).
- Gate dispatch (`trial_scoring.runtime_validate_invariants_dispatch`): tensorrt
  = docker --gpus all (image nvcr.io/nvidia/tensorrt-llm/release:<v>; auto-maps
  only v1_2_1); vllm = docker CPU (vllm/vllm-openai:v<v>); transformers =
  IN-PROCESS from the per-version tfvenv (which needs tree-sitter added for
  mining). Sequence GPU-heavy ops (gemma serving + tensorrt gate both want GPU).
- Runner: `scripts/phase1/wave1.py` (det-then-llm-extend; --rung oss/opus,
  --prompt-file, --proposed). Reusable bits in `scripts/wave2_llm_cells.py`
  (ollama_generate, parse_invariants, floor_invariants, render_prompt),
  `scripts/wave2_llm_source.py` (source chunking). Floor = improved-det-v2
  (`findings/trial_runs/wave2/w2-a-improved-det-v2/<engine>/<vslug>/`).
- Sources (EPHEMERAL /tmp): vllm `/tmp/vllm-<v>/vllm`; tensorrt
  `/tmp/trt-llm-<v>/tensorrt_llm` (+ `/tmp/trial_tensorrt_v1_2_1_venv/src/...`);
  transformers `/tmp/tfvenv-<v>/lib/python3.12/site-packages/transformers`. If
  wiped, re-extract from docker images / recreate tfvenvs via uv.
- GT denominator per cell:
  `findings/study/ground_truth/<engine>/<vslug>/invariants/PILOT_GT.yaml`.
- 15 cells: tensorrt v0_20_0/v0_21_0/v1_0_0/v1_1_0/v1_2_1; vllm
  v0_18_1/v0_19_1/v0_20_0/v0_21_0/v0_22_0; transformers
  v5_6_2/v5_7_0/v5_8_1/v5_9_0/v5_10_2.
- `findings/trial_runs/` artefacts are reproducible mining intermediates;
  discard, do not commit. Memory `project_mining_study_pilot.md` has full state.

## Gotchas

- Use Opus 4.8 for ALL subagents. ASCII only (no em/en-dashes), no
  "Co-Authored-By" footer, no "CLAUDE" mentions in committed files (hooks reject).
- After committing, `git log -1` to confirm HEAD is yours (the sync hook can
  replay a stale snapshot).
- Background python: run directly with run_in_background (do NOT nohup-&).
- ruff RUF059 (unused var) is non-blocking; commits still land.
