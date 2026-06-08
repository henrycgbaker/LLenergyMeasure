# Full 15-cell matrix: results and the corrected gradient

Authoritative result for the complete 5-version x 3-engine window. Supersedes the
partial bump-robustness numbers in FANOUT_FINDINGS (tensorrt-only gradient + vllm
0.18->0.19 pair) AND CORRECTS the survivor-rebound claims there (see Section 3).

## 1. The matrix (runtime-gated GT, union / confirmed constraints)

| version slot | tensorrt | vllm | transformers |
|---|---|---|---|
| v1 | 0.20: 104/14 | 0.18.1: 145/94 | 5.6.2: 247/83 |
| v2 | 0.21: 164/18 | 0.19.1: 249/90 | 5.7.0: 104/74 |
| v3 | 1.0: 123/21 | 0.20: 189/119 | 5.8.1: 118/84 |
| v4 | 1.1: 84/24 | 0.21: 213/132 | 5.9.0: 160/116 |
| v5 | 1.2.1: 228/74 | 0.22: 204/118 | 5.10.2: 120/85 |

Every cell: 2 Opus passes (entry-point + class-hierarchy) as GT contributors,
runtime-gated (tensorrt GPU container; vllm CPU container; transformers in-process
in a per-version uv venv, no torch, NO container build). tensorrt confirm counts
are low because most args-model validators are CUDA/model-dir gated (dormant);
vllm/transformers configs construct CPU-only so confirm rates are high.

## 2. Per-engine bump gradient - PERSIST (field-level, tolerant key) is the robust signal

Matched consecutive cells by tolerant key (leaf field + coarse bucket) on the OPUS
basis (passA+passB, present in every cell). PERSIST = fraction of the earlier
cell's mined knobs whose field+bucket still exists in the next cell.

| engine | consecutive bumps (persist %) |
|---|---|
| tensorrt | 0.20->0.21 89.3, **0.21->1.0 (MAJOR) 53.1**, 1.0->1.1 93.8, 1.1->1.2.1 91.5 |
| vllm | 0.18->0.19 78.4, 0.19->0.20 91.9, 0.20->0.21 86.2, 0.21->0.22 85.1 |
| transformers | 5.7->5.8 75.9, 5.8->5.9 100.0, 5.9->5.10 98.9 (5.6->5.7 61.7 - contaminated*) |

**Headline (well-powered across 3 engines, 14 bumps):** the lone MAJOR boundary
(tensorrt 0.21->1.0) persists only **53%** of mined config-knobs - roughly HALF
churns - versus **76-100%** across the eleven clean minor bumps. A major version
bump reorganises the config-validation surface about 2x as much as a minor one;
minor bumps are largely additive on a stable base. This is the core empirical
support for a cheap runtime gate that re-validates carried-over knowledge against
the live engine, especially on majors.

*Contaminated comparison: transformers 5.6.2 folded a PoC GT that uses a different
field-naming convention (`transformers.sampling.X` vs the passes' `GenerationConfig.X`),
so the 5.6.2->5.7.0 delta is dominated by a naming artefact, not version change
(the agents confirmed the 5.6-5.7 source is near-identical). Discount it; the clean
pure-net_new transformers minors (5.7->5.8->5.9->5.10) are the reliable ones.

## 3. CORRECTION: the "survivor re-bound rate" metric is unreliable

Earlier findings (FANOUT_FINDINGS cross-major + cross-engine) headlined a
"survivor re-bound rate" (tensorrt major 42%, vllm 36%) as evidence of silent
staleness. **That metric is confounded and those specific numbers are retracted.**

Re-bound requires the SAME `canonical_predicate_value` across two cells, but
different agents/sources encode the same bound differently (`{ge=1}` vs
`{ge=1,when_not_none=True}` vs `1`), so the metric conflates real constraint
changes with predicate-ENCODING variance (the same root cause as the identity
under-merge documented in STUDY_RESULTS Section 7). Proof from the matrix:
transformers 5.8.1->5.9.0, whose source is byte-identical per the mining agents,
shows **0% rebound**, while 5.6.2->5.7.0 (PoC-folded vs pure-pass encoding) shows
**80%** - the difference is encoding style, not the engine. Across the matrix the
rebound rate tracks "were these two cells built by the same agent style" far more
than "did the constraint change".

Only PERSIST (field+bucket, encoding-agnostic) is trustworthy. The runtime-gate
argument therefore rests on FIELD-LEVEL churn on a major (38 tensorrt knobs vanish
or rename across 0.21->1.0, ~47% of the surface) plus the standing fact that the
gate re-validates every carried-over constraint - NOT on a quantified rebound rate.

## 4. GT integrity (review status)

- REVIEWED (independent adversarial source-review, earlier): tensorrt 0.21/1.0/1.1
  + vllm 0.18.1/0.19.1 = **243/247 confirmed entries verified REAL** (62/63 tensorrt
  + 181/184 vllm), zero false-confirms, zero fabrications; non-real were a redundant
  mis-encoding + 3 imprecise predicate_values (all real rules).
- DEFERRED: the 9 new cells (tensorrt 0.20; vllm 0.20/0.21/0.22; transformers
  5.6.2-5.10.2) were queued for adversarial review but the run hit the weekly
  subagent limit (resets 5pm Europe/Berlin, 2026-06-08) before doing any work. To
  resume after reset: rerun `/tmp/review_instructions.md` per cell (ENGINE/VERSION/
  SOURCE/GTDIR), writing each verdict to the cell's `ADVERSARIAL_REVIEW.md`. The
  mining agents self-verified by replaying their kwargs in-venv (transformers/vllm),
  which is corroborating but not an independent adversarial check.

## 5. Reproduce

- Matrix gate metrics: `<cell>/invariants/pilot_metrics.json` per cell.
- Gradient: `/tmp/full_gradient.py` (reads all PILOT_GTs, prints per-engine deltas).
- transformers venvs: `/tmp/tfvenv-<ver>/` (uv, transformers + pyyaml + pydantic, no
  torch); gate run with that venv's python (in-process). tensorrt/vllm sources at
  `/tmp/trt-llm-<ver>/` and `/tmp/vllm-<ver>/`.
