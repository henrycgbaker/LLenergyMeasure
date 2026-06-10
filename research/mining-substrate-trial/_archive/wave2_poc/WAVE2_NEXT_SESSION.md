# Wave 2 next-session entry point

**Read this first.** This doc is the bootstrap for a fresh Claude session driving Wave 2 to completion.

**Authored 2026-06-05 00:58 CEST** by the prior session before /clear. All context lives on disk; nothing in conversation memory carries over.

---

## What Wave 2 is

LLEM mines schema + invariants from upstream LLM-inference engine source (transformers / vllm / tensorrt-llm), then validates each invariant at runtime via the engine's own container. The existing producer pipeline is brittle and undercovers reality by ~70%.

Wave 2 is RESEARCH. Its job is to generate empirical evidence about the decision landscape for two CI workflows that will be designed in a downstream engineering exercise:

- **Workflow 1**: schema discovery (enumerate engine config fields)
- **Workflow 2**: invariant mining (extract validation predicates; invariants subsume invalid-config-set boundary)

Wave 2 does NOT design the final workflow. It produces the evidence base that a future engineering session uses.

## Read these in order

1. `WAVE2_SCOPE.md` - the framing: info-generation, production constraints, what's in and out of scope.
2. `WAVE2_PRIMITIVES.md` - the 8-axis decision landscape Wave 2 characterises (substrate, LLM role, assembly, model scale, call shape, task, engine, version situation).
3. `WAVE2_WORKFLOWS.md` - the 5-6 workflow shapes (W-A through W-F) under consideration as production candidates.
4. `WAVE2_PROTOCOL.md` - the experimental discipline: pre-registration, success criteria, what each cell records.
5. `WAVE2_EXPERIMENT_QUEUE.md` - concrete priority-ordered list of cells to run. Drive this end-to-end.
6. `DECISIONS_LOG.md` - chronological narrative. Read the entries marked 2026-06-04 onward (Wave 2 starts there).
7. `findings/wave2_treesitter_probe.md` - first experimental cell completed (tree-sitter substrate, both tasks, transformers + vllm).
8. `findings/wave2_improved_det_primitives.md` - empirically-grounded proposal for a new substrate (~70-80% coverage at near-zero cost, dwarfs the current baseline).
9. `findings/wave2_batch2_prompts.md` - 3 ground-truth Opus agent prompts ready to launch at 3am CET (separately scheduled via cron).
10. `findings/ground_truth/<engine>/<version>/` - per-engine ground-truth catalogues. Batch 1 (v_old) landed; batch 2 (v_new) launches via the cron.

## What has landed already

- Ground truth (batch 1): vllm v0.7.3 + transformers v4.57.3 + tensorrt v0.21.0 catalogued at depth. Existing baselines covered ~26-30% of reality; env vars (169 total across 3 engines) had 0% coverage.
- Tree-sitter probe: schema task is essentially solved cheaply on vllm (98.5% recall vs old reference; will drop against the richer GT). Invariants task is bounded at ~50% syntax-only.
- 9-strategy scaffolding under `scripts/strategies/wave2/`. The treesitter walker is implemented; pydantic-native + runtime-trace + h15 closed-loop are also implemented; others are stubs.
- `scripts/wave2_runner.py` dispatcher that scores cells through `trial_scoring.score_cell`.
- `WAVE2_INFRA_SETUP.md` documents the venv + Ollama + container prerequisites.

## What you do next, in order

### Step 1: launch batch 2 ground truth (foundational)

Per `findings/wave2_batch2_prompts.md`. Three parallel Opus subagents at xhigh effort. They take ~15-25 minutes wall-clock each (batch 1 took ~15-22 min). Wait for all 3 to complete and synthesise the cross-engine pattern (mirror the batch-1 synthesis in DECISIONS_LOG).

### Step 2: implement `scripts/strategies/wave2/a_improved_det.py`

Per `findings/wave2_improved_det_primitives.md` - the 7-primitive deterministic substrate empirically proposed from the batch-1 delta patterns. ~600-1000 LoC. Estimated to close 60-80% of the baseline coverage gap CHEAPLY without LLM calls. This becomes the new Tier A floor.

### Step 3: re-score Wave 1 cells against ground truth

`trial_scoring.score_cell` already supports this; just point at the new GT reference path. Aggregate to `findings/wave1_rescored_against_gt.md`. Keep old (validated-union) scores side-by-side for cross-Wave comparison.

### Step 4: run the experiment queue

`WAVE2_EXPERIMENT_QUEUE.md` lists cells in priority order. Drive them with `/goal` orchestrating subagents where parallelism helps. Each cell records: per-task recall + precision against GT, wall-sec + energy + estimated $, failure mode tag, hallucination rate (LLM cells), self-update success binary, observations.

### Step 5: synthesis deliverables (after all cells run)

Per `WAVE2_PRIMITIVES.md` "what gets characterised per axis":
- Per-task cost-recall frontier with each substrate primitive's curve
- Marginal cost of LLM in each assembly shape
- Substrate complementarity matrix
- Bump-survivability per primitive
- Self-update success rate per workflow shape

Write to `findings/wave2_synthesis_<topic>.md`. These deliverables are the actual research output that the downstream engineering session consumes.

## Discipline

- Append every decision + finding to `DECISIONS_LOG.md` as you go. Don't batch.
- Cite source line / file / qualname for every catalogue entry.
- ASCII only. No em-dashes. No emojis. No `Co-Authored-By: Claude` footer in commits (project hook rejects).
- Treat ground truth as a MINIMUM SET, not a ceiling. Add to GT files when in-the-wild encounters surface new entries.
- Sync-hook gotcha: `~/.local/bin/llem-sync-full` was patched 2026-06-05 to unset `GIT_DIR` before its git operations. If you see sync commits landing on your branch with author `llem-sync-full <sync@local>`, the patch regressed - check `head -22` of the script.

## Hard constraints / out of scope

- No Claude / GPT API runs (Wave 3 candidate, deferred until ANTHROPIC_API_KEY available).
- No SGLang / LMDeploy vendoring (Wave 3).
- No LangGraph dep (build minimal state-machine harness inline if needed).
- No statistical inference (record point estimates only; bootstrap CIs are Wave 3).
- 4xA100-40G is the hardware budget. Models that don't fit don't get run.

## Token / compute budget

User stated: "One long session: run all experiments to completion." Budget is not a per-experiment cap but a total-session cap. Spend whatever it takes; stop when the synthesis deliverables are complete and the next-session-not-needed.

## When you're done

Write `WAVE2_RESEARCH_OUTCOMES.md` - the consolidated findings document that the downstream engineering session reads. Map each finding to: which axis it characterises, which primitives it favours, what the cost-recall picture is, what's unknown / would benefit from Wave 3.

Then update `DECISIONS_LOG.md` with a "Wave 2 closed" entry.
