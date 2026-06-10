# mining-substrate-trial

Research spike on keeping **engine-config knowledge** (schema + invariants) current
across upstream version bumps of the inference engines (vLLM, TensorRT-LLM,
transformers), cheaply enough to run on every bump in CI.

An "invariant" is a rule the engine enforces at config construction (`if <pred>:
raise/warn`, a `Field(ge/gt/le/Literal)` constraint, a cross-field relation). The
north star: a cheap, general CI workflow that re-mines + re-validates this knowledge
per bump. The engine owns the SSOT; a **runtime gate** validates mined knowledge in
the engine's own container ("observe, don't re-encode"); cost is understood
ordinally (deterministic ~free < small-OSS < mid-OSS < Opus).

## Start here

**[`findings/study/CONSOLIDATED_FINDINGS.md`](findings/study/CONSOLIDATED_FINDINGS.md)**
is the single, self-contained writeup of the whole study: experimental design,
the predecessor PoC bake-off, the deterministic-baseline / bump-robustness layer,
the LLM-pattern waves (1-4), and the cross-bump degradation work (wave 5). It folds
every load-bearing result into one narrative and is the canonical entry point.

## The study, in two layers

- **Layer 1 - deterministic baseline + bump-robustness.** A 15-cell matrix
  (5 versions x 3 engines), runtime-gated and adversarially source-reviewed
  (908/913 entries verified REAL). Establishes the major-vs-minor bump gradient
  (a major bump churns ~half the config surface vs 76-100% persistence on minors)
  and the silent vllm refactor cliff that a deterministic Primitive-8 recovers ~44%
  of with no LLM.
- **Layer 2 - LLM-pattern waves (1-4).** How far the OSS rung gets on a frozen cell:
  the validation-path bottleneck, the kwargs-emission lever, the scale-vs-code-tuning
  tier story, and construction-grounding as the OSS infra-wall lever. Plus the
  cross-bump degradation signal + LLM diff-diagnose (wave 5) - the actual product
  property.

## Live canonical docs

| doc | role |
|---|---|
| [`findings/study/CONSOLIDATED_FINDINGS.md`](findings/study/CONSOLIDATED_FINDINGS.md) | the consolidated writeup (entry point) |
| [`STUDY_DESIGN.md`](STUDY_DESIGN.md) | the locked pre-registered program spec |
| [`findings/study/FULL_MATRIX.md`](findings/study/FULL_MATRIX.md) | the authoritative 15-cell bump-robustness matrix + GT integrity |
| [`findings/wave2_bump_survivability.md`](findings/wave2_bump_survivability.md) | the bump cliff + Primitive-8 recovery |
| [`findings/study/PHASE1_WAVE{1,2,3,4}_FINDINGS.md`](findings/study/) | per-wave detail |
| [`findings/study/WAVE4_RECONCILIATION_MAP.md`](findings/study/WAVE4_RECONCILIATION_MAP.md) | maps the waves onto the PoC taxonomy |
| [`findings/study/REVIEW_northstar_strategy.md`](findings/study/REVIEW_northstar_strategy.md) | adversarial strategic review (the "wrong axis" correction) |

`_archive/` holds the superseded prose: the predecessor PoC bake-off
(`RESEARCH_WRITEUP.md`, `DECISIONS_LOG.md`), the Wave-2 PoC planning corpus
(`wave2_poc/`), the per-wave pre-registrations (`prereg/`), and earlier synthesis
drafts. It is lineage, not the hot path; the consolidated doc supersedes it.

## Status

Phase 1 waves 1-4 + cross-bump wave 5 complete, on `spike/engine-knowledge-as-data`
(not on main). The `scripts/` tree is live - `scripts/strategies/wave2/` holds the
`improved-det-v2` deterministic floor (the per-bump re-mine generator), consumed by
`scripts/round0b/` and `scripts/wave2_runner.py`.

## Reproducibility

Scripts live under `scripts/`. The directory name contains hyphens, so scripts are
not importable as a package; invoke by file path, and each script prepends its
scripts dir to `sys.path` for sibling imports.

- **LLM-pattern waves:** `scripts/phase1/wave*.py` (pure-LLM, construction-grounding,
  multistage/hybrid langchain). The langchain cells need a venv with
  `langchain-core` + `langchain-ollama` (`/tmp/round0b_venv` in this environment).
- **Cross-bump:** `scripts/phase1/wave5_gate_acceptance.py` (degradation signal),
  `scripts/phase1/wave5_bump_diagnose.py` (LLM diff-diagnose). Note: these resolve
  source from `/tmp/trial_<engine>_<vslug>_venv/src` - if `/tmp` is reaped, re-point
  to the surviving source (the chunked inputs are not committed; see the consolidated
  doc's methodology note).
- **Local Ollama:** a containerized Ollama on port 11435 (`--gpus all`) escapes the
  shared-host cgroup cap; the wave runners point `OLLAMA` at it.
- **Runtime gate:** `scripts/validate_invariants.py` + `scripts/study_gt_pilot.py`
  construct violating configs in the engine's own container and observe the raise.

### Tests

```bash
uv run python -m pytest research/mining-substrate-trial/scripts/test_trial_scoring.py -v
```
