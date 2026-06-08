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

Confirmed counts here are raw gate output; Section 4 reports the adversarially
review-validated counts (one transformers 5.9.0 entry is reclassified as a
false-confirm, leaving 115 review-validated there).

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

## 4. GT integrity (independent adversarial source-review)

All cells have now been adversarially source-reviewed (refute-first: each confirmed
entry is assumed wrong until the cited source line proves predicate + outcome +
field + bound/allowlist match exactly, and the positive probe is checked to fire the
CLAIMED rule, not an incidental error). 14 of 15 cells carry a written
`invariants/ADVERSARIAL_REVIEW.md`; tensorrt 1.2.1 was validated as the original
pilot (in-venv replay), which is corroborating rather than an independent check.

| cell(s) | confirmed | reviewed | REAL | non-real (class) |
|---|---|---|---|---|
| tensorrt 0.20 | 14 | 14 (full) | 14 | - |
| tensorrt 0.21/1.0/1.1 | 63 | 63 (full) | 62 | 1 redundant mis-encoding (real rule) |
| vllm 0.18.1/0.19.1 | 184 | 184 (full) | 181 | 3 imprecise predicate_value (real rules) |
| vllm 0.20 | 119 | 68 (sample 57%) | 68 | - |
| vllm 0.21 | 132 | 74 (sample 56%) | 74 | - |
| vllm 0.22 | 118 | 68 (sample 58%) | 68 | - |
| transformers 5.6.2 | 83 | 83 (full; 62 PoC-folded all checked) | 83 | - |
| transformers 5.7.0 | 74 | 74 (full) | 74 | - |
| transformers 5.8.1 | 84 | 84 (full) | 84 | - |
| transformers 5.9.0 | 116 | 116 (full) | 115 | 1 false-confirm (excluded from validated GT) |
| transformers 5.10.2 | 85 | 85 (full) | 85 | - |

**Totals: 913 entries reviewed, 908 verified REAL (99.5%), 5 non-real - 0
fabrications, 1 false-confirm, 4 mis-stated/imprecise (each still points at a real
source rule).** The three vllm 0.20/0.21/0.22 cells were sampled (>=50 spanning
every native_type, every predicate_kind, every non-`invalid` outcome, plus all
synthesised entries); the sample surfaced no non-real entry so no full-expansion
override fired. Every other cell was verified in full.

The single false-confirm (transformers 5.9.0,
`transformers_watermarking_type_watermarking_config_type_not_in_WatermarkingConfig`,
mech-source): its positive probe (`watermarking_config: 42`) raised an incidental
`AttributeError` from `GenerationConfig.validate()` calling `.validate()` on an int,
NOT the claimed construction-time type guard - which does not exist in 5.9.0 source.
It is kept here as a measured gate-quality result (false-confirm rate ~0.1% of
reviewed entries) and excluded from the review-validated GT count for that cell
(gate-confirmed 116 -> review-validated 115); the raw gate artifacts
(`PILOT_GT.yaml`, `pilot_metrics.json`) retain the as-run 116, so Section 1 counts
are unchanged. Re-gating would reproduce the artifact, so the fix is the documented
exclusion, not a re-run. Per-entry reasoning lives in each cell's
`invariants/ADVERSARIAL_REVIEW.md`.

Systemic, non-defect: every non-real entry across the whole study is mech-source or
a cross-grain restatement; the OPUS basis (passA+passB) that powers the Section 2/3
gradient carried zero non-real entries, so the bump-robustness signal is unaffected.
mech-source entries are the sole locus of the lone false-confirm and the 4 earlier
imprecisions - direct support for the milestone-end item to port operator-ful
canonical encoding into the production per-version miners.

## 5. Reproduce

- Matrix gate metrics: `<cell>/invariants/pilot_metrics.json` per cell.
- Gradient: `/tmp/full_gradient.py` (reads all PILOT_GTs, prints per-engine deltas).
- transformers venvs: `/tmp/tfvenv-<ver>/` (uv, transformers + pyyaml + pydantic, no
  torch); gate run with that venv's python (in-process). tensorrt/vllm sources at
  `/tmp/trt-llm-<ver>/` and `/tmp/vllm-<ver>/`.
