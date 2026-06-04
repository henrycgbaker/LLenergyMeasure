# Wave 2 workflow shapes

**Status:** Reference doc 2026-06-05. The 5-6 production workflow shapes Wave 2 collects evidence about. Each workflow is a different way to compose the substrate + LLM primitives in `WAVE2_PRIMITIVES.md` axes 1-3.

Wave 2 does NOT pick a workflow. It produces evidence about each so a downstream engineering session can design the production CI flow.

---

## The decomposition

Each candidate workflow decomposes into 3 steps (validation step is fixed):

- **Detect**: when an upstream bump arrives, what tells us the situation in the new version?
- **Extract**: what produces the new catalogue (schema + invariants)?
- **Validate (fixed)**: runtime gate via `scripts/validate_invariants.py` against the engine's own container. SSOT. No false positives reach vendored output.

| ID | Detect | Extract | Sees additions? | Per-bump cost | Self-update profile |
|---|---|---|---|---|---|
| D1 | Landmark probe (status quo: hand-declared LANDMARKS tuple) | - | No (tautological) | Free | Fragile |
| D2 | AST / tree-sitter diff vs prior version | - | Yes (structural) | Cheap | Robust to renames |
| D3 | LLM reads both versions, narrates diff | - | Yes (incl. semantic) | Medium | Robust |
| D4 | Behavioural diff: run engine on synthetic configs | - | Some | Medium-high | Catches behavioural changes only |
| D5 | Skip. Always re-extract everything. | - | n/a | Zero | Work moves to extraction |
| E1 | - | Hand-cut producer (status quo) | If hand-edited | Per-bump human PR | Brittle |
| E2 | - | Universal substrate (tree-sitter / framework reflection / pyright stubs) | Yes (structural) | Near-zero | Robust |
| E3 | - | LLM reads source, emits catalogue | Yes | Per-bump LLM cost | Robust at LLM scale |
| E4 | - | LLM patches old producer to work on new version | If LLM patches correctly | Per-bump LLM + retry | Mixed |
| E5 | - | Maintainer reviews candidates produced by other extractors | Always (with hint) | Per-bump human review | Robust but expensive |

## The 5-6 workflow candidates

| Workflow | Detect | Extract | One-line pitch | Wave 1 evidence |
|---|---|---|---|---|
| **W-A: status quo** | D1 | E1 | Today's pipeline. Drift tool gates; human authors new vendored producer when drift fires. | Currently in use |
| **W-B: pure universal substrate** | D5 | E2 | No detection step. Re-extract via reflection / tree-sitter / stubs each bump. Self-updating by construction. Quality bounded by substrate. | Tree-sitter probe: 98.5% vllm schema vs old ref; ~50% invariants |
| **W-C: pure LLM** | D5 | E3 | No detection. LLM re-extracts each bump from source. Quality bounded by LLM. Cost = LLM-per-bump. | (b): ~50% recall ceiling at 70B-q4 |
| **W-D: LLM patches producer** | D2 or D3 | E4 | AST-diff or LLM-diff identifies what changed; LLM patches producer; runtime gate validates re-run. Producer stays as catalogue source but maintains itself. | H4: 0/3 patches lifted recall single-shot |
| **W-E: universal floor + LLM extend** | D5 | E2 + E3 | Universal substrate produces baseline; LLM looks at diff vs prior catalogue and extends. d-ab pattern with universal substrate as floor. | d-ab in Wave 1: by-construction 100% + 0-8 extensions per bump |
| **W-F: LLM diagnoses + maintainer authorises** | D3 | E5 | LLM reads new source + emits structured "things to consider"; maintainer authors. Curation primacy. Lowest silent-breakage risk; highest human cost. | H9: 0 fabrications across 8 diagnoses |

## Special W-G candidate: improved-det + LLM extend (added post batch-1 GT)

After batch 1 ground truth revealed that ~70-80% of baseline misses are mechanically catchable with better deterministic primitives (see `findings/wave2_improved_det_primitives.md`), a 6th candidate becomes attractive:

**W-G: improved-det floor + LLM extend (the most promising hybrid)**
- Detect: D5 (skip; re-extract every bump)
- Extract: improved-det primitive set (the 7 new tree-sitter-based primitives) + small-LLM extends the genuinely-hard residual
- Pitch: covers 70-80% mechanically; LLM only handles dynamic registries / C++ pybind / semantic resolution; lowest LLM cost per bump while reaching highest catalogue completeness

This is the strongest a priori production candidate. Wave 2 cells should specifically measure it.

## Self-update dimension

Per user direction 2026-06-05: self-updating means dynamic-change-robust, not snapshot-optimised. The workflow must respond to underlying engine changes (renames, removals, additions, new validator surfaces) without per-bump human intervention. Each workflow above gets scored on a **self-update success binary** per bump-pair:

- Did the workflow produce a usable updated artefact (catalogue or producer code) without human intervention?
- If "auto-PR for human review" is the success criterion, did the PR's content actually need substantive review or was it a rubber-stamp?

Workflows that require landmark-list updates (W-A) fail this test by construction. Workflows with universal substrates (W-B, W-E, W-G) likely score well. LLM-based workflows (W-C, W-D) depend on LLM stability across bumps.

## What Wave 2 measures per workflow

For each of the 6 workflow candidates above, run cells across:

- 3 engines (transformers / vllm / tensorrt-llm)
- 2 bump-pairs (v_old → v_active and v_active → v_new) — real recent bumps with significant source-shape change
- 2 tasks (schema / invariants)

Each cell records the per-cell-level fields documented in `WAVE2_PRIMITIVES.md`. Per-workflow aggregates land in `findings/wave2_workflow_<id>.md`.

## What Wave 2 does NOT decide

- Which workflow ships in production.
- How to phase the rollout.
- Cost vs accuracy tradeoffs at deployment scale.

Those are the downstream engineering session's questions. Wave 2 supplies the evidence base.

## Cross-references

- `WAVE2_SCOPE.md` - framing
- `WAVE2_PRIMITIVES.md` - axes
- `WAVE2_PROTOCOL.md` - experimental protocol
- `WAVE2_EXPERIMENT_QUEUE.md` - concrete cell queue
- `findings/wave2_improved_det_primitives.md` - the new substrate candidate for W-G
- `DECISIONS_LOG.md` - 2026-06-05 workflow-first reframe entry
