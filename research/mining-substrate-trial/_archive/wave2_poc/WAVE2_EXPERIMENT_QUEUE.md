# Wave 2 experiment queue

**Status:** Priority-ordered. Authored 2026-06-05. Drive top-to-bottom; parallelise where indicated.

The queue covers Wave 2.0 (foundation) → 2.1 (substrate characterisation) → 2.2 (assembly characterisation) → 2.3 (LLM-as-non-extractor roles) → 2.4 (self-update) → 2.5 (synthesis). All cells recorded under `findings/trial_scores/wave2/`.

---

## Wave 2.0 - Foundation (3am CET 2026-06-06 kickoff)

| # | Cell | Estimated wall | Outputs |
|---|---|---|---|
| 0.1 | Launch batch 2 ground-truth agents (3 parallel Opus) per `findings/wave2_batch2_prompts.md` | ~25 min wall (parallel) | `findings/ground_truth/<engine>/v_new/{schema,invariants,methodology,delta,version_delta}.{json,yaml,md}` |
| 0.2 | Synthesise batch 2 cross-engine results + per-bump-pair deltas; append to DECISIONS_LOG | ~10 min | DECISIONS_LOG entry |
| 0.3 | Implement `scripts/strategies/wave2/a_improved_det.py` per `findings/wave2_improved_det_primitives.md` (~600-1000 LoC) | ~3-4 hours dev | Module + smoke test |
| 0.4 | Re-score Wave 1 cells against full GT (all 6 versions in batch 1+2) | ~30 min compute | `findings/wave1_rescored_against_gt.md` |
| 0.5 | Update `WAVE2_PRIMITIVES.md` Axis 1 to include `improved-det` as a level | ~5 min | Doc edit |

**Wave 2.0 exit condition:** GT is complete across 3 engines x 2 versions; improved-det module exists + has been smoke-tested; Wave 1 has been re-scored.

---

## Wave 2.1 - Substrate primitive characterisation

For each substrate primitive on Axis 1, run per-task per-engine per-version cells against GT. Engines x versions = 3 x 2 = 6 cells per substrate per task. Tasks = 2 (schema + invariants). So ~12 cells per substrate.

| # | Substrate primitive | Cells | Wall (rough) |
|---|---|---|---|
| 1.1 | `hand-walker` (existing baseline, captured as control) | 12 | Free (read existing artefacts) |
| 1.2 | `tree-sitter-universal` (already implemented; recall full coverage) | 12 | <1 hr (sub-second per cell) |
| 1.3 | `framework-reflection` (`a_pydantic_native` extended to all 3 engines) | 12 | ~1 hr (needs engine importable) |
| 1.4 | `runtime-trace` (`a_runtime_trace` on transformers + vllm + tensorrt) | 12 | ~3-4 hr (perturbation loops) |
| 1.5 | `behavioural-fuzz` (`a_fuzz` pilot on transformers; expand only if recall > 30%) | 4 then maybe 8 more | ~2-4 hr |
| 1.6 | **`improved-det`** (the new substrate from `findings/wave2_improved_det_primitives.md`) | 12 | <1 hr |
| 1.7 | `pyright-stubs` (single-cell benchmark transformers active) | 1 | ~30 min |
| 1.8 | `sphinx-xml` (single-cell benchmark transformers active) | 1 | ~1 hr (Sphinx build) |
| 1.9 | `rag-over-source` (single-cell benchmark vllm active) | 1 | ~2 hr (indexing + query) |
| 1.10 | `llm-reads-raw` (the (b) strategy; small LLM only at this stage) | 12 | ~3-6 hr |

**Wave 2.1 exit condition:** per-substrate per-task recall + precision tables populated against GT. Substrate complementarity matrix computed (which pairs catch disjoint entries).

---

## Wave 2.2 - Assembly shape characterisation

Compose substrates per Axis 3. Cells = assembly x engine x version x task. Pick the strongest substrate primitive from 2.1 as the deterministic component; small-LLM (best from Wave 2f below) as the LLM component.

| # | Assembly | Cells | Wall (rough) |
|---|---|---|---|
| 2.1 | `det-only` (best 2.1 substrate, no LLM) | 12 | Already covered |
| 2.2 | `det-then-llm-extends` (W-E / W-G shape; d-ab pattern) | 12 | ~3-6 hr |
| 2.3 | `llm-then-det-validates` (b + runtime gate; H3 pattern) | 12 | ~3-6 hr |
| 2.4 | `closed-loop-feedback` (h15: extract -> gate -> failure list -> re-extract) | 12 | ~6-12 hr |
| 2.5 | `llm-self-consistency` (k=3 vote on small LLM; h11) | 12 | ~12-18 hr (3x LLM cost) |
| 2.6 | `det-then-llm-patches-det` (W-D shape; the H4 idea retooled with diff context) | 6 (transformers + vllm only) | ~2-4 hr |

**Wave 2.2 exit condition:** assembly-shape comparison table. Cost-per-recall-pp ladder.

---

## Wave 2.3 - LLM-as-non-extractor roles

LLM in roles beyond proposal. Per Axis 2.

| # | LLM role | Cells | Wall (rough) |
|---|---|---|---|
| 3.1 | LLM-as-diagnose (categorical gaps; H9 pattern) | 6 (3 engines x 2 versions) | ~1-2 hr |
| 3.2 | LLM-as-gate (yes/no on candidates produced by improved-det) | 12 (apply gate to each Wave 2.1 candidate) | ~2-4 hr |
| 3.3 | LLM-as-decide (which assembly applies; meta-orchestration) | 6 | ~2-4 hr |
| 3.4 | LLM-as-curate (long-running maintenance simulation across multiple bumps) | 3 (one per engine across multiple version-pair sequences) | ~3-6 hr |

**Wave 2.3 exit condition:** LLM-role table populated. Quality-per-role per model-scale recorded.

---

## Wave 2.4 - Model-scale sweep (Wave 2f from the original protocol)

| # | Model | Cells (best assembly x 3 engines x active) | Wall (rough) |
|---|---|---|---|
| 4.1 | Qwen2.5-Coder-7B fp16 | 3 | ~30-45 min |
| 4.2 | DeepSeek-Coder-V2-Lite 16B q4 | 3 | ~30-60 min |
| 4.3 | Phi-4-14B fp16 | 3 | ~30-60 min |
| 4.4 | Llama-3.1-8B fp16 | 3 | ~30-45 min |
| 4.5 | Llama-3.3-70B fp16 (benchmark ceiling) | 1 (transformers active only) | ~60-90 min |
| 4.6 | Qwen2.5-Coder-32B fp16 (benchmark) | 1 (transformers active only) | ~30-60 min |

**Wave 2.4 exit condition:** cost-quality curve per model scale. Smallest viable model identified.

---

## Wave 2.5 - Self-update characterisation (the key research question)

For each candidate workflow (W-A through W-G), run bump-update cells across both bump-pairs per engine = 12 cells per workflow.

| # | Workflow | Cells | Wall (rough) | Key measurement |
|---|---|---|---|---|
| 5.1 | W-A status quo (re-run existing pipeline on v_new) | 12 | minimal | Manual PR cost baseline |
| 5.2 | W-B pure universal | 12 | Already covered | Self-update success binary |
| 5.3 | W-C pure LLM | 12 | Already covered | Self-update success binary |
| 5.4 | W-D LLM patches producer | 6 (transformers + vllm) | ~6-12 hr | Patch acceptance rate + manual touch-up cost |
| 5.5 | W-E universal floor + LLM extend | 12 | Already covered | Self-update success binary |
| 5.6 | W-F LLM diagnoses + maintainer authorises | 6 | ~2-4 hr | Diagnosis quality + maintainer time per bump |
| 5.7 | **W-G improved-det floor + LLM extend** | 12 | Already covered | Self-update success binary |

**Wave 2.5 exit condition:** per-workflow self-update success rate + per-bump cost decomposition.

---

## Wave 2.6 - Synthesis deliverables

Per `WAVE2_PRIMITIVES.md` "what gets characterised per axis":

| # | Deliverable | Effort |
|---|---|---|
| 6.1 | `findings/wave2_substrate_frontier.md` - cost-recall frontier per task per substrate | ~1 hr |
| 6.2 | `findings/wave2_assembly_ladder.md` - assembly-shape cost-per-recall-pp ladder | ~1 hr |
| 6.3 | `findings/wave2_model_scale_curve.md` - per-model-size cost-quality curve | ~30 min |
| 6.4 | `findings/wave2_llm_role_matrix.md` - LLM role quality per model-scale | ~1 hr |
| 6.5 | `findings/wave2_workflow_comparison.md` - 6 workflows compared on self-update + per-bump cost | ~2 hr |
| 6.6 | `findings/wave2_substrate_complementarity.md` - which substrates union to what | ~1 hr |
| 6.7 | `findings/wave2_failure_mode_catalogue.md` - per-cell failure modes + interactions | ~1 hr |
| 6.8 | **`WAVE2_RESEARCH_OUTCOMES.md`** - consolidated research output for the downstream engineering session | ~3-4 hr |
| 6.9 | DECISIONS_LOG.md "Wave 2 closed" entry | ~30 min |

**Wave 2 exit condition:** WAVE2_RESEARCH_OUTCOMES.md exists, gives the engineering session everything it needs to design the two workflows without re-running any experiments.

---

## Estimated total wall-clock

- Wave 2.0: ~4-5 hours (parallel with batch 2 wait)
- Wave 2.1: ~12-18 hours sequential (parallel where possible)
- Wave 2.2: ~30-50 hours sequential
- Wave 2.3: ~10-20 hours
- Wave 2.4: ~5-8 hours
- Wave 2.5: ~12-20 hours
- Wave 2.6: ~10-15 hours

**Total: ~85-135 hours of compute + dev work** for a complete Wave 2. Highly parallelisable where independent. User's instruction is "one long session to completion"; pace accordingly.

## Priority discipline

If something blocks (hardware contention, model pull failures, runtime errors), skip and continue. The synthesis (2.6) gracefully tolerates partial cell coverage; flag what's missing and why. Do NOT loop forever on a stuck cell.

## Concurrency suggestions

- Wave 2.0 step 0.1 (batch 2 ground truth) parallelises with 0.3 (improved-det implementation) — different agent threads.
- Wave 2.1 substrate cells parallelise across substrates if hardware allows.
- LLM cells in 2.1, 2.2, 2.4 serialise per GPU (Ollama is single-tenant by default).
- Use the `Workflow` tool for orchestrated parallelism where it helps.
