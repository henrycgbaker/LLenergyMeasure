# Wave 2 protocol (info-generation framing)

**Status:** Rewritten 2026-06-05 around the info-generation reframe. Supersedes the earlier cost-frontier "find-the-winner" draft (which is preserved in DECISIONS_LOG.md 2026-06-04 entries for chronological record).

Wave 2 is a research round. Its deliverable is comprehensive characterisation of the decision landscape for two CI workflows (schema discovery + invariant mining). The protocol below is the experimental discipline.

---

## 1. What Wave 2 measures

For each cell `(workflow OR substrate primitive, engine, version situation, task)`:

- Per-task recall + precision against ground truth (the new SSOT) AND against validated-union (for cross-Wave comparison).
- Wall-clock + GPU energy (Wh) + estimated $.
- Failure mode tag from a closed vocabulary: `silent / detectable / crash / hallucinated-from-empty / under-emit / over-emit / gate-rejected-most / self-update-failed`.
- Self-update binary: did the cell produce a usable updated artefact without human intervention?
- Hallucination rate (LLM cells): gate-rejected entries / total proposed.
- Source-citation rate (LLM cells): proportion of entries with verifiable source citation.
- Observations free-text: anything noticed in passing worth carrying forward.

Per-axis aggregates (computed after cells complete):

- Cost-recall frontier per task per substrate.
- Marginal cost of LLM in each assembly shape.
- Substrate complementarity matrix.
- Bump-survivability per primitive.
- Self-update success rate per workflow shape.
- Failure-mode interactions per assembly.

## 2. Discipline rules

The Wave 1 epistemic_framing rules carry over with workflow-aware additions:

**Pre-registration.** This protocol + `WAVE2_PRIMITIVES.md` + `WAVE2_WORKFLOWS.md` + `WAVE2_EXPERIMENT_QUEUE.md` lock the experimental design. Mid-wave changes log to `findings/wave2_deviations.md` with rationale.

**Locked prompts.** All LLM-extraction cells use the prompts that landed at Wave 1 Phase 2 closure (`findings/phase2_locked_prompts/`). New LLM roles (gate, decide, curate) lock new prompts before any cell runs them; store under `findings/wave2_locked_prompts/`.

**Pinned model + container digests.** Per `findings/wave2_model_digests.toml`. The Wave 2 runner verifies served digest before any LLM cell runs.

**Complete every cell.** No early-exit on data that looks bad. The bad-cells failure modes are decision-relevant.

**Synthesis only at Wave 2.6.** Observations during execution go to `observations` arrays and `DECISIONS_LOG.md`. No mid-wave architectural changes; new candidates log to `WAVE2_DEFERRED.md` for Wave 3.

**Don't fix gaps mid-wave.** If the existing handwritten producer has a 2-line walkable gap, don't patch it. The gap is research data.

**Ground truth is a minimum set.** As cells run, in-the-wild encounters can grow ground-truth files. Append; never reset. Log each growth in DECISIONS_LOG.

**Validation is given.** The runtime gate at `scripts/validate_invariants.py` is the SSOT for what counts as a valid invariant. Wave 2 does not modify or measure it; it consumes it.

## 3. Out of scope (Wave 3 candidates)

- Claude / GPT API runs.
- Statistical inference (bootstrap CIs, seed-variance).
- Layer B (behavioural) validation as a 4th gate.
- LangGraph as a multi-step harness (build minimal inline if needed).
- SGLang / LMDeploy vendoring.
- Property-based test generation; SMT / Z3 targets.
- Hardware beyond 4xA100-40G.
- Frontier-API model benchmarking beyond what's already deferred.

These go to `WAVE2_DEFERRED.md` if surfaced during Wave 2; not retrofitted.

## 4. Acceptance criteria (no "winner", instead deliverables)

Wave 2 closes when ALL of these exist on disk:

1. Ground truth complete: `findings/ground_truth/<engine>/<version>/` populated for all 3 engines x 2 versions, each with schema + invariants + methodology + delta + version_delta.
2. Wave 1 cells re-scored against GT: `findings/wave1_rescored_against_gt.md`.
3. Per-axis characterisation deliverables: 8 files under `findings/wave2_<topic>.md` per `WAVE2_EXPERIMENT_QUEUE.md` section 2.6.
4. `WAVE2_RESEARCH_OUTCOMES.md` consolidated findings doc.
5. `DECISIONS_LOG.md` updated with "Wave 2 closed" entry summarising the headline findings.

Wave 2 explicitly does NOT close with a recommended workflow. The downstream engineering session picks the workflow after consuming the deliverables.

## 5. Threats to validity

**T1. Model pin drift.** Ollama models occasionally bump weights without tag change. Mitigation: digest-pinning per Discipline F. Each Wave 2 cell record contains the model digest it actually saw.

**T2. Hardware contention.** 4xA100-40G shared with other tenants. Mitigation: sequential per-GPU; concurrency only for 1xA100-fitting small models.

**T3. Ground truth itself incomplete.** GT is established by Opus subagent reads + docs cross-reference; agent could miss things. Mitigation: GT-as-minimum policy lets later cells grow GT when they catch new entries. Cell records flag "candidate adds to GT".

**T4. Runtime validation infra missing for some engine versions.** If the engine's container doesn't exist for v_new, the gate silently fails. Mitigation: every cell records gate-availability binary; cells without gate get aggregated separately.

**T5. Self-update binary is subjective at boundaries.** "Did the workflow produce a usable artefact" can be debated. Mitigation: usable means the catalogue passes the runtime gate at >= 95% acceptance AND covers >= a threshold of GT.

**T6. Tree-sitter Python grammar lag.** Recent Python syntax may not parse. Mitigation: tree-sitter-python==0.23.6 pinned; AST fallback documented.

## 6. Concretely what each axis-level expands to in cells

See `WAVE2_EXPERIMENT_QUEUE.md` for the priority-ordered cell list. Cells are addressed as `<axis-level> x <engine> x <version-situation> x <task>` tuples; outputs land at `findings/trial_runs/wave2/<strategy_id>/<engine>/<version_slug>/` and scores at `findings/trial_scores/wave2/<strategy_id>__<engine>__<version_slug>.json` (matches the existing Wave 1 layout for cross-Wave aggregation).

## 7. Cross-references

- `WAVE2_SCOPE.md` - framing + production constraints.
- `WAVE2_PRIMITIVES.md` - 8-axis decision landscape.
- `WAVE2_WORKFLOWS.md` - 6 workflow shapes under consideration.
- `WAVE2_EXPERIMENT_QUEUE.md` - priority-ordered cells to run.
- `WAVE2_NEXT_SESSION.md` - entry doc for the fresh session that drives execution.
- `WAVE2_INFRA_SETUP.md` - infra prerequisites per cell type.
- `DECISIONS_LOG.md` - chronological narrative.
- `findings/wave2_treesitter_probe.md` - first cell completed.
- `findings/wave2_improved_det_primitives.md` - empirically-derived new substrate proposal.
- `findings/wave2_batch2_prompts.md` - ground-truth batch 2 prompts (fire at 3am CET).
- `findings/ground_truth/<engine>/<version>/` - per-engine GT artefacts.

---

*Protocol locked 2026-06-05. Deviations logged at `findings/wave2_deviations.md`.*
