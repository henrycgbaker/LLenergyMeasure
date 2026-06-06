# Engine-Config Mining: Optimization Study (5-version window)

**Status: EXECUTING (rev 5 locked; revisions in Section 15.2). Round 0 GT
established and runtime-gated across the full tensorrt window
(0.21->1.0->1.1->1.2.1); both headline findings in hand (deterministic ceiling +
bump-robustness gradient). RESULTS SYNTHESIS: see `STUDY_RESULTS.md`. Execution
log + design revisions: Section 15. Chronological detail:
`findings/study/FANOUT_FINDINGS.md`.**

A rigorous, systematised re-run of the engine-config mining problem, constrained to
a tight recent-version window, informed by the Wave 1/2/2.5 proof-of-concept (now
treated as exploratory scaffolding). This study is the clean, pre-registered program
whose artefacts land on main via PR.

> **Cross-reference note.** This design builds on the PoC corpus so we reuse what we
> learned and do not repeat mistakes. Referenced artefacts (`DECISIONS_LOG.md`,
> `WAVE2_*.md`, `findings/*`, `scripts/*`) currently live on branch
> `trial/mining-substrate-bakeoff` under `research/mining-substrate-trial/`; they
> co-locate beside this doc when the corpus is PR'd to main. A consolidated map is
> in Section 14.

---

## 1. Objective

A **constrained optimization** run as a multi-phase study. For each of two
independent tasks (schema, invariants), produce **two co-equal headline
deliverables**:

- **(A) Recall-cost Pareto** - the cheapest workflow that holds high catalogue
  recall vs GT on a fixed version (snapshot completeness).
- **(B) Bump-delta-recovery + GT-growth** - per bump-pair, the fraction of entries
  that CHANGED (added / removed / renamed / relocated) that the workflow tracks
  WITHOUT a human editing its code or landmark list, plus how many runtime-confirmed
  entries it surfaces that GT lacked. This is the direct measure of the
  non-tautological + self-updating property (Section 1.5).

The cheapest-at-recall-plateau design and the cheapest-bump-robust design **may not
be the same**; the production choice needs both frontiers.

Commitments: **Pareto, not naive recall-max** (cost penalises expensive methods -
Section 9); **start wide / max-diversity** (include expensive rungs early to SEE if
they buy recall, then prune those that do not justify their cost); **plateau-
terminated, defined mechanically** (Section 9).

This reframes, but does not contradict, the PoC's cost-frontier framing
(`findings/wave2_substrate_frontier.md`, `findings/wave2_assembly_ladder.md`).

## 1.5 Why: the workflow properties this study informs

Informs LLEM's production CI workflow for keeping engine-config catalogues current
as engines bump. Target properties (`WAVE2_SCOPE.md` "Production target" +
`WAVE2_WORKFLOWS.md` self-update dimension + `DECISIONS_LOG.md` 2026-06-05 late):
**non-tautological**, **robust to dynamic version change / self-updating**, **cheap
where possible**, **comprehensive at proposal**, **clean deterministic gate**, **LLMs
in multiple roles**, **composable**.

Metric map: (A) covers cheap + comprehensive; (B) covers non-tautological +
bump-robust + self-updating. The PoC measured only (A)-like snapshot recall, which is
why its bump conclusions were engine-specific noise
(`findings/wave2_bump_survivability.md`); this study instruments (B) directly.

## 2. Scope: the 5-version window

Last 5 minor/major release lines per engine (latest patch of each locked at
execution). Earlier versions out of scope.

| Engine | In-window lines | Bump-pairs | Bump character |
|---|---|---|---|
| transformers | 5.6, 5.7, 5.8, 5.9, 5.10 | 4 | minor-only (all v5) |
| vllm | 0.18, 0.19, 0.20, 0.21, 0.22 | 4 | post-`config/`-subpackage minor churn |
| tensorrt-llm | 0.20, 0.21, 1.0, 1.1, 1.2 | 4 | spans the 0.x -> 1.x MAJOR |

**15 cells PER TASK; 12 bump-pairs per task. ALL cells established FRESH** to the
Section-6 GT standard. The PoC GT (transformers 5.6, vllm 0.19, tensorrt 0.21 + 1.2;
under `findings/ground_truth/`) is **folded in as one contributing source and
re-validated by the gate** (reuse the valid finds; do not discard found work) - NOT
the privileged reference it accidentally became in the PoC (N=1 single-Opus,
source-walk-only, no runtime verification, hand-scoped to a subjective "LLEM-scope
subset" per its `methodology.md` files).

**Window caveat (external validity).** Mostly minor / post-refactor churn; only
**tensorrt 0.21 -> 1.0 is a true major boundary**. The PoC's headline bump
phenomenon (the imperative -> declarative recall cliff,
`findings/wave2_bump_survivability.md`) was a MAJOR-bump effect. A mostly-minor
window can show high stable recall trivially; the **bump-robustness conclusion is
carried by the tensorrt major pair**.

## 3. Priors from the PoC (hypotheses; each cites its source)

- **P1 - Schema solved deterministically, invariants not** (schema 0.37-0.97 vs
  invariants 0.15-0.51). Source: `findings/wave2_substrate_frontier.md`,
  `findings/wave2_substrate_matrix.json`. -> two independent tasks.
- **P2 - Four convergent bump patterns** (default-indirection; imperative->declarative
  `Field`; nesting/subpackage; opaque->pydantic over majors). Source: `DECISIONS_LOG.md`
  2026-06-05 "Step 0.2", and the per-engine `findings/ground_truth/<e>/<v>/version_delta.md`.
- **P3 - Declarative-constraint parsing is decisive** (Primitive 8: vllm 0.147->0.309,
  tensorrt lifts; built + measured). Source: `findings/wave2_primitive8_results.json`,
  `DECISIONS_LOG.md` Wave 2.5, `scripts/strategies/wave2/a_improved_det_v2.py`.
- **P4 - Small OSS LLMs are weak extractors** (+0.02 extend; pure-extract 4-30x below
  floor; scale knee ~8B). Source: `findings/wave2_assembly_ladder.md`,
  `findings/wave2_model_scale_curve.md`, `findings/wave2_llm_cells.json`.
- **P5 - Cross-catalogue identity is convention-fragile** (~16x strict/tolerant gap).
  Source: `findings/wave2_failure_mode_catalogue.md`, `findings/wave2_deviations.md`,
  `scripts/gt_scoring.py`. -> tolerant scoring; GT validity load-bearing.
- framework-reflection is a HYPOTHESIS ("likely complementary + bump-immune"), NOT a
  result - it was never measured (crashed on import). Source:
  `findings/wave2_substrate_complementarity.md`, `findings/wave2_failure_mode_catalogue.md`.

## 4. Two independent tasks

**Track S (schema)** and **Track I (invariants)**: **separate GT**, **separate
frontiers**. Shared: harness machinery, the window, the compute pool. The design
space (substrate primitives, LLM roles, assemblies, call-shapes) is catalogued in
`WAVE2_PRIMITIVES.md`; the workflow candidates (W-A..W-G) in `WAVE2_WORKFLOWS.md`.

**Cross-track dependency (composability).** Track I mines invariants **over Track S's
enumerated field set + their validators**, not raw source (you cannot mine a
constraint on a field you have not enumerated; the Primitive-8 story is schema-side
`Field` enumeration becoming invariant recovery). Track I's INPUT is Track S's
OUTPUT; GT and frontiers stay separate.

**Env vars: scoped OUT of both tracks (settled).** LLEM forwards only `LLEM_*` +
`HF_HOME`/`HF_TOKEN` (`src/llenergymeasure/infra/docker_runner.py:855-857`) and
sweeps ZERO engine-native env vars; the few behaviour knobs (`env_config.py`) wrap
to config-object fields already in Track S; both consumer audits
(`findings/phase3_audit_consumers.md`, `findings/phase3_audit_llem_fields.md`) show
zero env-var consumption. Forward-pointer: a strong FUTURE product surface, but
building it now serves a consumer that does not exist.

## 5. Methodology: the phase ladder (per task)

- **Round 0 - Ground truth** (Section 6).
- **Round 0b - Deterministic baseline** (Section 7): start from improved-det-v2; add
  four named primitives; ensemble by MEASURED complementarity; iterate to plateau.
- **Phase 1 - LLMs, wide net** (Section 8): tiers x roles x assemblies x call-shapes;
  map BOTH frontiers.
- **Phases 2..n - prune + deepen** (Section 9 stopping rule; deepen-before-prune
  quota; planned assembly x bump-shape block).
- **Dedicated cells (promoted from PoC "deferred" - they measure the production
  property, so first-class):** bump-UPDATE / auto-propose (true self-update binary -
  does a proposed patch pass the gate with NO human edit?); degradation-signal (does
  gate-acceptance-rate OR an LLM diff-reviewer FIRE on a vllm-style silent-collapse?);
  GT-refresh-cost-per-bump (GT upkeep must cost less than the workflow it validates);
  diagnose/diff-review role scored on caught-a-silent-collapse rate, NOT recall (PoC:
  0 fabrications across 8 diagnoses, `findings/wave2_llm_role_matrix.md`).
- **Final:** both frontiers per task; recommended design(s) with per-bump cost +
  CI-affordability check; bump-survivability.

## 6. Ground truth (Round 0): the pooled-max, runtime-anchored denominator

**GT is the maximum-recall union across all available discovery methods, with every
entry reviewed for validity by the deterministic gate - NOT any one method's
authority.** No method is presumed best. In the PoC, Opus AUTHORED the GT in a single
pass (N=1), so every other method was scored against Opus's output - which made Opus
look definitionally complete (a CIRCULAR artefact, not a demonstrated recall win) and
let Opus's own misses pass unnoticed (it hand-scoped a "minimum set", deliberately
excluding a long tail, per its `methodology.md` files; mechanical enumerators can
catch fields a source-walk misses). This study removes that privilege: Opus is one
high-coverage CONTRIBUTOR among several, pooled with model-independent methods and
gate-validated. Any method that surfaces a gate-confirmed entry the others missed
GROWS GT. So "recall -> 1.0" means: how close does a CHEAP workflow get to the best
ALL methods can collectively, verifiably find.

Per cell, per task, GT = the gate-validated, adjudicated union of:
1. **N=2 independent Opus passes** with DIFFERENT decomposition strategies
   (entry-point/call-graph walk vs class-hierarchy/type-tree walk) so traversal blind
   spots differ. (A high-coverage contributor; NOT presumed authoritative.)
2. **Deterministic AST enumeration** (improved-det-v2,
   `scripts/strategies/wave2/a_improved_det_v2.py`) - a MODEL-INDEPENDENT contributor
   that breaks the LLM-measures-LLM circularity the review flagged.
3. **Runtime reflection** (framework-reflection-in-container) - the resolved model's
   own field/validator set, model-independent.
4. **Existing PoC GT** (`findings/ground_truth/<engine>/<version>/`) - folded in as a
   contributing source (reuse the valid finds; do not discard found work). Carries no
   privilege; every entry is re-validated by the gate like all others.

**Runtime verification is the validity anchor, and in BOTH tasks the engine library
owns its own SSOT** ("observe, don't re-encode" -
`.product/designs/config-deduplication-dormancy/runtime-config-validation.md`):

- **INVARIANTS: `scripts/validate_invariants.py` - LLEM's CANONICAL production gate**
  (not a trial-parallel one). It replays each invariant's kwargs_positive/negative
  through the live engine in its container, compares declared vs observed, and FAILS
  CI on divergence (exit 0/1/2). Operates on the product corpus
  `src/llenergymeasure/engines/{engine}/invariants.proposed.yaml` ->
  `invariants.validated.yaml`. We USE this, not a copy.
- **SCHEMA: `scripts/validate_schema.py` (sibling gate, to be developed)** - same
  engine-owns-SSOT principle. For each proposed field it REFLECTS the live engine's
  resolved config object (`pydantic model_fields` / `dataclasses.fields` /
  `inspect.signature`) for {exists, type, default}, and for caller-touchable fields
  CONSTRUCTS the config with a type-valid probe (must be ACCEPTED) and a type-invalid
  probe (must be REJECTED). Divergence (absent field / type mismatch / default
  mismatch / wrong accept-reject) FAILS, mirroring the invariant gate. Operates on
  `schema.discovered.json` -> `schema.validated.json`. With reflection + probe this is
  a STRONG gate (the engine owns its field/type/default truth); only a
  name-existence-only shortcut would be weak.

Entries failing the gate are kept but labelled unverified and excluded from the strict
denominator.

**The gate IS the validity review, applied to every candidate from every source** -
N=2 Opus, mechanical AST, runtime reflection, AND the existing PoC GT alike. Pass =
confirmed-valid GT; fail = held as unverified. No entry is trusted because of who
proposed it; this is the "review each GT entry for validity = the deterministic gate"
principle.

**Labels** per entry {source: opus|mechanical|reflection (which found it), runtime:
confirmed|unverified|contradicted}. **GT-growth report per cell** (the non-tautology
instrument): runtime-confirmed entries a mining cell surfaces that GT lacked - GT then
grows; growth is a first-class scored outcome.

**Adjudication checkpoint (locked):** GT CONSTRUCTION accepts LLM (Opus) adjudication
of one-pass disagreements - NO human checkpoint - BUT the runtime-verified gate then
validates every entry it can (full container coverage, Section 10). Validity rests on
RUNTIME confirmation, not LLM agreement; LLM-adjudication only assembles the candidate
union. The residual LLM-adjudicated limit applies ONLY to entries runtime cannot reach
(labelled unverified, Section 12).

Cost: this GT establishment is a real Opus + container bill before any mining cell;
budgeted (Section 10) and measured as the GT-refresh data point.

## 7. Deterministic baseline (Round 0b)

**Start from improved-det-v2** (improved-det's 7 primitives + Primitive 8; built +
measured, `scripts/strategies/wave2/a_improved_det_v2.py`,
`findings/wave2_primitive8_results.json`). Do NOT re-derive it. Equally, avoidd path dependence: there may be other methods we haven't considered yet that might be worthwhile.

**Add the four primitives the PoC residual-miss data names** (sources: P2 above +
`findings/wave2_improved_det_primitives.md` + the `version_delta.md` files):
1. **Default-indirection resolver** (follow defaults out of `__init__` into lazy
   funcs / class-attr refs / flagged-C++).
2. **Per-platform `check_and_update_config` walker** (vllm; a whole missed surface).
3. **Validator-body predicate extractor** (Primitive 6 finds validators but not their
   body predicate - "necessary but not sufficient").
4. **Generalised subpackage glob** (generalise Primitive 8's `config/*.py`; never pin
   paths - they go stale across bumps).

**Ensemble by MEASURED complementarity.** "improved-det subsumes tree-sitter" is a
RESULT (`findings/wave2_substrate_complementarity.md`, union gain +0.03-0.07) -> do
not union them. "framework-reflection complementary" is an untested HYPOTHESIS ->
measure; union only on disjoint runtime-confirmed entries. Iterate to plateau. Lock
the deterministic baseline + its bump-delta-recovery curve.

## 8. LLM design space (Phase 1+)

**The LLM is invoked as a fresh subagent / served model per cell; tiers form the cost
gradient:**
- **OSS-small (7-14B), OSS-mid (~32B), OSS-high (~70B)** served via Ollama on the
  **4xA100 pool** - cost = GPU-energy, RISING with size (so mid/high are penalised on
  the cost axis - Section 9).
- **Opus subagent** (via the Agent tool; runs Anthropic-side, NOT on the GPUs) - cost
  = TOKEN-$, the frontier-quality and expensive rung.

The expensive rungs (OSS-high, Opus) are **included EARLY for max diversity** - to SEE
whether they buy materially greater recall - and PRUNED later if they do not justify
their cost. P4 killed OSS-small as an extractor, but "the cheapest LLM rung that clears
the deterministic floor" is the open production question this gradient answers; Opus
answers "what is reachable at any cost." The 14B->70B band the PoC left UNMEASURED
(`findings/wave2_model_scale_curve.md`) is now covered by OSS-mid/high.

**Axes** (`WAVE2_PRIMITIVES.md`): Role {extract, extend-residual, gate, diagnose,
diff-review, curate}; Assembly {det-only, llm-only, det-then-llm-extend,
llm-then-det-gate, closed-loop, ensemble-vote, self-consistency,
det-then-llm-patches-det}; Call-shape {single, k-vote, chunked, chained, agentic}.

**Anti-local-optimum guards:** fractional sampling EXCEPT (i) a protected
deepen-before-prune quota for late-payoff assemblies (closed-loop, self-consistency) -
not pruned on a shallow probe; (ii) assembly x bump-shape as a planned full cell-block
(the cliff is that interaction).

## 9. Measurement, cost model, stopping rule

- **Recall:** tolerant identity (headline) + strict (lower bound), vs the
  runtime/mechanically-confirmed GT core (`scripts/gt_scoring.py`).
- **Anti-gaming guard:** GT is a MINIMUM set + PoC hallucination proxy 0.87-1.0, so
  "recall + gate-acceptance" alone rewards spray. Add a **tolerant-precision FLOOR /
  emit-budget** (disqualify below it) and report **cost-per-true-positive**.
- **Delta-recovery + GT-growth** (deliverable B) per bump-pair; **degradation-signal-
  fires** binary per cliff-bump.
- **Cost axis (Pareto x) = recurring per-bump cost, one $ scale:** det ~ 0 (CPU-sec);
  OSS = GPU-energy-$ (rising with model size); Opus = token-$. One-time dev amortised
  separately. Plus a per-bump-$ + CI-affordability check for any recommended
  LLM-bearing design. **Cost penalises the expensive methods** - they must earn recall
  to stay on the frontier.
- **Stopping rule (mechanical, up front):** stop a task only after **K consecutive
  phases each move both frontiers by < epsilon AND no surviving design improved any
  per-bump-pair delta-recovery score**; **epsilon tied to GT measurement noise**
  (adjudication disagreement + strict/tolerant gap), not a guessed constant.
- **Discipline** (carried from `WAVE2_PROTOCOL.md`): per-phase pre-registration;
  locked + versioned prompts; pinned model + container digests; deviation log; no
  mid-phase architectural changes.

## 10. Compute and prerequisites

- **4xA100 available** - for serving OSS-mid/high (32B/70B) AND multi-GPU
  runtime-verification containers. (Resolves the PoC's single-GPU cap, `DECISIONS_LOG.md`
  GPU resolution.) **Opus needs no GPU** (Agent tool, Anthropic-side).
- **Engine containers for ALL 15 (engine, version) cells** (LOCKED - runtime-verified
  gate for every cell). Some exist from the PoC; the rest (~11; tensorrt heaviest at
  ~50 GB) are BUILT upfront as the first execution task. Entries that are
  runtime-UNverifiable by nature (predicate not constructible, C++-resolved default)
  stay labelled unverified - the residual LLM-adjudicated limit (Section 12).
- **OSS models** via Ollama (provisioned in the PoC); **Opus** via the Agent tool.
- **GT-establishment budget** (N=2 Opus + container bill) explicit + measured.

## 11. Deliverables and PR packaging

Clean artefacts -> orthogonal PRs onto main (snapshot off main): **Code** (GT harness,
deterministic ensemble, tiered LLM-assembly harness, runner, scorer); **GT corpus**
(15-cell, per-task, provenance + confidence labelled); **Study writeup** (this design
+ per-phase pre-registrations + per-phase findings-with-reasoning + both frontiers per
task + recommended workflows + bump-survivability). The wide-net -> prune/deepen ->
narrow narrative is itself a deliverable.

## 12. Threats to validity

- **LLM-measures-LLM circularity** - mitigated by the mechanical + runtime denominator
  sources (6.2/6.3); residual where only Opus passes find an entry.
- **GT-as-subset ceiling illusion** - 0.95 of a scoped GT may miss half the real
  surface; the GT-growth instrument (Section 6) is the defence and is load-bearing.
- **Window-selection bias toward easy (minor) bumps** - tensorrt major pair carries the
  bump-robustness conclusion.
- **Runtime-coverage bias** toward containerisable engines.
- **LLM-adjudicated GT** - no-human-checkpoint limit, named; optional human spot-check.

## 13. Parameters - LOCKED at approval (2026-06-06)

1. **Container coverage: ALL 15 cells.** Build/obtain a runtime container per cell;
   the runtime-verified gate validates everywhere. Container builds are the first
   execution task (Section 10).
2. **GT adjudication: LLM-adjudicated union + runtime gate validates all-possible.**
   No human checkpoint; validity rests on runtime confirmation; residual
   runtime-unreachable entries kept labelled unverified.
3. **OSS model rungs:** PoC small set (7-14B) + a ~32B + a ~70B.
4. **GT cross-validation N = 2 + adjudication.**
5. **Det-baseline refine cap:** iterate to plateau, soft max ~3 rounds.
6. **Precision floor / emit-budget + stopping K + epsilon: calibrated from the first
   waves** (epsilon tied to GT noise; values set once the frontier shape is visible).

## 14. Cross-reference map (PoC corpus, on `trial/mining-substrate-bakeoff`)

- Narrative + decisions: `DECISIONS_LOG.md` (2026-06-05/06 entries).
- Consolidated PoC outcomes: `WAVE2_RESEARCH_OUTCOMES.md`.
- Framing + objectives + design space + workflows: `WAVE2_SCOPE.md`,
  `WAVE2_WORKFLOWS.md`, `WAVE2_PRIMITIVES.md`, `WAVE2_PROTOCOL.md`.
- Synthesis findings: `findings/wave2_substrate_frontier.md`, `..._complementarity.md`,
  `..._bump_survivability.md`, `..._assembly_ladder.md`, `..._model_scale_curve.md`,
  `..._llm_role_matrix.md`, `..._workflow_comparison.md`, `..._failure_mode_catalogue.md`.
- Hard numbers: `findings/wave2_substrate_matrix.json`, `..._primitive8_results.json`,
  `..._llm_cells.json`, `..._substrate_analysis.json`.
- Code to reuse/extend: `scripts/strategies/wave2/a_improved_det.py` (+ `_v2.py`),
  `scripts/gt_scoring.py`, `scripts/gt_adapter.py`, `scripts/wave2_runner.py`,
  `scripts/trial_scoring.py`, `scripts/validate_invariants.py` (the runtime gate).
- GT corpus (cross-check input): `findings/ground_truth/<engine>/<version>/`.
- Env-var verdict evidence: `findings/phase3_audit_consumers.md`,
  `findings/phase3_audit_llem_fields.md`; LLEM `src/.../infra/docker_runner.py`,
  `src/.../utils/env_config.py`.
- This study's review: `STUDY_DESIGN_REVIEW.md`.

---

## 15. Execution log + design revisions (post-rev-5, 2026-06-06)

What has actually been built and run since rev 5 was locked, plus the design
parameters that CHANGED as a result. Findings detail in
`findings/study/FANOUT_FINDINGS.md`; per-cell GT in
`findings/study/ground_truth/<engine>/<v>/invariants/`.

### 15.1 Built

- **Invariant gate is now dynamic.** `scripts/validate_invariants.py` (the
  CANONICAL product gate) gained: (a) TRT-LLM `native_type` resolution across the
  engine's export modules + model-placeholder injection for any `*LlmArgs`;
  (b) **probe synthesis** - `synthesize_probe_kwargs` derives positive/negative
  kwargs from a declared predicate (`predicate_kind`+`predicate_value` or a
  single-field `match.fields` operator) so predicate-only mined entries are
  gateable WITHOUT hand-authored kwargs. Safe by construction (mis-synthesis ->
  unverified, never false-confirm) + field-attribution guard.
- **Schema gate built (Track S, link 1->2).** `scripts/validate_schema.py`
  re-runs the engine's own introspector in-container and diffs the discovered
  schema vs live for {exists, type, default} (semantic type comparison) +
  construct-probes enum fields. Distinct from
  `check_pydantic_matches_discovered.py` (link 2->3, discovered-vs-packaged).
- **Union+gate driver.** `research/mining-substrate-trial/scripts/study_gt_pilot.py`
  - cell-parameterised (`--engine/--version-slug/--image/--sources`),
  auto-discovers GT sources (Opus passes / mechanical / PoC), runtime-gates,
  writes `PILOT_GT.yaml` + `PILOT_REPORT.md` + `pilot_metrics.json`.
- **Production miner widened (lever 1, tensorrt 1.2.1 only).** PluginConfig walk
  + `Optional[Literal]` unwrap + module-level `Literal` alias resolution +
  `not_in` membership encoding. NOTE: full porting of the trial improved-det-v2
  primitives into the production per-version miners is DEFERRED to milestone end;
  only this targeted widening was done as a research probe.

### 15.2 Design revisions (supersede the locked rev-5 parameters)

- **Identity (CHANGED).** The rev-5 tolerant key `(leaf_native_field,
  coarse_predicate_bucket)` OVER-COLLAPSES distinct per-class / per-bound
  constraints (e.g. `max_draft_len` across Eagle/NGram/DraftTarget/TorchLlmArgs),
  and the "group confirmed if ANY member confirms" rule then false-confirmed
  them. A defining-class/MRO fix was considered and REJECTED (the headline case
  is one inherited field with per-subclass constraints, which MRO would merge
  wrongly; `native_type` is inconsistent across sources -> drift). **New scheme:**
  count + confirm at a CONSTRAINT identity
  `(leaf, coarse_bucket, canonical_predicate_value)` - keyed on what the
  invariant ASSERTS (drift-stable), not where it is declared. The tolerant
  `(leaf, coarse_bucket)` key is RETAINED only as the cross-source recall-match
  axis. `gt_adapter.canonical_predicate_value` is the new helper.
- **Confirmation (CHANGED).** Now PER CONSTRAINT (a tolerant group can hold
  several constraints; an easy sibling no longer confirms a hard one). The
  field-attribution guard applies to ALL leniently-confirmed entries (no
  `expected_outcome`), not just synthesised ones.
- **GT entries** persist the `match` block so every entry is independently
  re-gateable.

### 15.3 Findings to date (tensorrt only)

- **tensorrt 1.2.1 invariant GT (constraint grain):** RE-GATED to union **228
  constraints**, **74 gate-confirmed**, GT-growth **+37 vs PoC** - after folding
  the widened production miner in as a committed union source
  (`prod_static_miner.yaml`); the +14 vs the prior 60 = 13 plugin-literal
  constraints confirmed outside the old union + 1 promoted tail. (Pre-re-gate:
  212 / 60; old tolerant-grain 144 / 46 - both superseded.) Circularity caveat:
  only 45 of the 74 confirmed have an independent Opus/PoC contributor, so the
  deterministic-ceiling measurement deliberately keeps the frozen 60/212
  denominator (see FANOUT_FINDINGS GT-re-gate note). Two adversarial reviews + a
  domain review: GT content **100% substantively correct** on a ~22-entry
  sample; pipeline + anti-tautology design sound; attribution tightening dropped
  0 confirmations.
- **tensorrt 0.21.0:** RE-DONE with 2 Opus passes -> **164 union / 18 confirmed**
  (was 128 / 3 without Opus passes; the lift re-confirms Opus passes are
  load-bearing). Adversarial source-review: 17/18 REAL, 0 false-confirm, 0
  fabrication, 1 mis-stated redundant encoding.
- **tensorrt 1.0.0 (NEW, major boundary):** **123 union / 21 confirmed**, 2 Opus
  passes + mech, no PoC. Adversarial source-review: 21/21 REAL, 0 false-confirm,
  0 fabrication.
- **tensorrt 1.1.0 (NEW):** **84 union / 24 confirmed**, 2 Opus passes only (no
  mech/PoC). Adversarial source-review: 24/24 REAL, 0 false-confirm, 0
  fabrication.
- **Bump-robustness gradient (0.21->1.0->1.1->1.2.1):** on the apples-to-apples
  OPUS basis (passA+passB, present every cell), the MAJOR bump churns ~8x a minor
  one: persistence **53% (0.21->1.0 major)** vs **94% (1.0->1.1)** and **92%
  (1.1->1.2.1)**. Among surviving knobs the major bump RE-BOUNDS 42% (changed
  bound/allowlist = silent-staleness the runtime gate catches) vs 14-20% on
  minors. 1.1 sits with 1.0 (pre-pydantic PluginConfig, no SamplingParams ranges,
  validate_build_config raises); the 1.2.x feature additions land after 1.1. The
  raw UNION basis is source-confounded across cells (1.1 has no mech) - use the
  OPUS basis. Detail in FANOUT_FINDINGS; raw /tmp/cross_major_delta.json.
- **Cross-engine (vllm 0.18.1->0.19.1):** 145/94 and 249/90 confirmed; Opus-basis
  persistence **78%** with **36% survivor re-bound**. Sits between tensorrt's minor
  (92-94%) and major (53%) - but vllm 0.x minors are semver-BREAKING (feature
  releases), so not directly comparable to tensorrt 1.x minors. The robust
  cross-engine signal is survivor RE-BOUNDING (36%, near the tensorrt major's 42%):
  silent re-bounding is engine-independent, so the runtime gate's necessity
  generalises. Gate-scope finding: confirmation verifies fire/pass BEHAVIOUR, not
  exact recorded predicate_value (3 vllm 0.18.1 entries had imprecise allowlists yet
  confirmed). GT integrity across all six cells: **243/247 confirmed entries REAL**
  on source-review (tensorrt 62/63 + vllm 181/184), zero false-confirms/fabrications.
- **Cross-engine schema gate** (shipped schemas vs live): transformers 4.57.3
  107/107 clean; vllm 0.7.3 116/135; tensorrt 0.21.0 92/107 (divergences are
  shipped-schema staleness vs refactored introspectors, not engine drift).
- **Deterministic ceiling (RECOMPUTED at constraint grain, denominator 60
  confirmed / 212 union):** bare mech-only (improved-det-v2) confirms **15/60 =
  25%** and surfaces exactly the same 15 - the **surfaced-but-unconfirmed gap is
  0**: at constraint grain the probe-synthesis gap VANISHES (the old "74% surfaced"
  was a tolerant-collapse artefact, reproduced here as the tolerant-key reach
  34/46 = 73.9%). The entire deficit is mining-scope. Lever 1 (production
  PluginConfig walk) lifts recall to **28/60 = 46.7%** (surfacing 30/60 = 50%) AND
  grows the GT by **+13 plugin-literal constraints outside the frozen union** (the
  old "15->42" conflated these three effects). Schema remains ~1.0 deterministic;
  invariants plateau well below 1.0 with a structural tail (cross-field /
  abstract-config / context-dependent) needing LLM mining. Detail in
  FANOUT_FINDINGS.md; raw recompute in /tmp/ceiling_recompute.json +
  /tmp/lever1_recompute.json.

### 15.4 Open items (next)

1. DONE - **deterministic ceiling recomputed at constraint grain** (mech-only 25%;
   lever-1 46.7% recall + GT-growth +13; see 15.3 and FANOUT_FINDINGS.md).
2. DONE - **lever 1 re-measured on the frozen denominator** (surfacing / recall /
   GT-growth reported separately); **GT re-gated to 74 confirmed / 228 union**
   (prod folded in); schema gate now **resolves `$ref`** to target type; **vLLM
   noise-filter regression test** added.
3. DONE - **full tensorrt window 0.21->1.0->1.1->1.2.1 established + reviewed**
   (1.0 + 1.1 containers pulled, both Opus passes per cell, union+gated,
   adversarial source-review per cell, 4-point bump-robustness gradient on the
   Opus basis; see 15.3 and FANOUT_FINDINGS). The gradient isolates the major
   boundary (53% persist) from the two minor bumps (92-94%). Next carry-targets:
   minor-bump deltas for vllm (0.18->0.22) and transformers (5.6->5.10) to test
   whether the minor-bump stability generalises across engines (the study window
   has no non-tensorrt major boundary).
4. cross-engine minor-bump deltas: **vllm 0.18->0.19 DONE** (see 15.3); remaining
   transformers 5.6->5.10. (The window has no non-tensorrt major boundary.)
5. Per-version producers exist for only ~6 versions; the window needs ~9 more
   (overlaps the engine-knowledge-as-data refactor - now the same trunk).
6. DEFERRED to milestone end: port trial improved-det-v2 primitives into the
   production per-version miners. INCLUDE the identity-encoding fix here: make the
   mechanical miner emit OPERATOR-FUL canonical values (`{gt=0}`, not the lossy
   bare `0`) so its encodings match the Opus passes and stop spuriously splitting
   the same constraint across sources. This is the real root cause of the
   confirmed-count over-count (measured below) and is a PRODUCER-side fix.
7. RESOLVED (measured, not fixed in-place): the identity UNDER-merge is a
   count-precision caveat, not a validity threat - no wrong invariant, neither
   headline finding affected. Measured behind 0.21's 18 confirmed: ~5 genuine
   duplicates (-> ~13 distinct), dominated by the mech miner's lossy bare-value
   encoding vs the Opus operator form (see item 6). A citation-keyed identity
   merge was REJECTED: distinct constraints often share one source function (3
   Lookahead fields under one `validate_positive_values`; several SamplingParams
   rules under one `_validate`), so keying on file+qualname would re-introduce the
   over-collapse the re-base fixed. The over-split is the correct fail-safe; no
   safe identity-layer change exists. Detail: STUDY_RESULTS Section 7.

### 15.5 Repo state

All study + refactor work is unified on ONE trunk: `study/5version-window`
== `spike/engine-knowledge-as-data` (fast-forwarded; stale producer branches
retired). Worktree: `~/workspace/llenergymeasure-trial`. tensorrt source on disk:
1.2.1 at `/tmp/trial_tensorrt_v1_2_1_venv/src/tensorrt_llm`, 0.21.0 at
`/tmp/trt-llm-0.21.0/tensorrt_llm`, 1.0.0 at `/tmp/trt-llm-1.0.0/tensorrt_llm`,
1.1.0 at `/tmp/trt-llm-1.1.0/tensorrt_llm`; vllm source at `/tmp/vllm-0.18.1/vllm`
and `/tmp/vllm-0.19.1/vllm` (all extracted from the release containers).
Containers present: tensorrt 0.21.0 + 1.0.0 + 1.1.0 + 1.2.1, vllm 0.7.3 + 0.18.1 +
0.19.1, transformers 4.57.3.
