# Wave 2 research outcomes

**Status: COMPLETE (autonomous run 2026-06-05). Partial-coverage wave - the
deferred cells (framework-reflection, runtime-trace, large/frontier LLMs, live
runtime gate for vllm/tensorrt) are itemised in Section 8; everything reached is
final.**

Consolidated findings from Wave 2 of the LLEM mining-substrate trial. This is the
document the downstream ENGINEERING + DESIGN session consumes to design the
production CI workflow(s) for keeping LLEM's engine-config catalogues current as
upstream engines bump. Per WAVE2_SCOPE, Wave 2 characterises the decision
landscape; it does NOT pick the production workflow.

Each finding is tagged with the axis it characterises (WAVE2_PRIMITIVES), the
primitives/workflows it favours, and the cost-recall picture. All recall is vs
the Opus-established ground truth (GT), tolerant identity (the strict lower bound
and the convention-drift gap are in `wave2_deviations.md`).

---

## 0. What Wave 2 produced

- **Ground truth for 3 engines x 2 versions** (the new SSOT): transformers
  v4.57.3 + v5.6.2, vllm v0.7.3 + v0.19.1, tensorrt-llm v0.21.0 + v1.2.1. Each
  with schema + invariants + methodology + delta + version_delta. GT is a
  MINIMUM set (grows when cells encounter new entries).
- **A new deterministic substrate** (`w2-a-improved-det`, 1418 LoC, 7 primitives)
  that roughly doubles baseline invariant recall vs GT.
- **A GT-scoring harness** (`gt_adapter` + `gt_scoring`) with strict + tolerant
  matching, because GT shape != the locked scorer's envelope.
- **Wave 1 re-scored vs GT** (`wave1_rescored_against_gt.md`).
- **A static-substrate matrix vs GT** (10/18 cells) + complementarity analysis.
- **LLM-extend (W-G) + pure-b + a 7b/8b/14b model-scale sweep** on a single A100.
- 8 synthesis deliverables (this file + 7 `wave2_*.md`).

## 1. Ground truth: the surface is 3-4x larger than the baseline producers see

Counts (GT): transformers 118 invariants / 38 env; vllm 79 inv / 238 env; tensorrt
92 inv / 55 env. The pre-Wave-1 baseline producers covered ~26-30% of invariants
and **0% of env vars** (169+ env vars across engines were entirely uncatalogued).

Implication: any production workflow MUST treat env vars as a first-class category
(no existing LLEM pipeline does), and must expect the true config surface to be
3-4x the hand-cut producer's view.

## 2. Cross-engine convergent bump patterns (the deepest finding)

All three v_old->v_new bumps are major/high-churn, and they share FOUR structural
patterns that directly drive substrate design (full detail: DECISIONS_LOG 0.2
synthesis + `wave2_bump_survivability.md`):

1. **Authoritative default moved OUT of the constructor signature** (all 3):
   transformers lazy `_get_default_generation_params()`, vllm class-attribute
   references, tensorrt C++/subconfig resolution. A substrate reading `__init__`
   defaults reads `None`/references. Default-mining must follow the indirection.
2. **Imperative `raise` -> declarative pydantic `Field(ge/gt/le/Literal)`** (vllm
   + tensorrt strongly). A grep/AST-for-`raise` substrate misses a growing
   fraction of constraints. This is the single most actionable substrate-design
   gap (see Primitive 8 recommendation).
3. **Config nesting + subpackage growth** (all 3): top-level-only walks miss
   nested knobs; source-line/landmark pins go 100% stale across the vllm bump.
4. **The hardest targets got EASIER** (tensorrt metaclass->pydantic; C++->Python
   validators): the static-substrate ceiling RISES over time on these engines.

Net: upstream is (unevenly) migrating surface from opaque (metaclass/C++/env)
toward declarative-Python. Bets on pydantic/dataclass-reflection + a
declarative-constraint primitive get STRONGER across bumps; bets on grep-for-raise
get weaker.

## 3. Axis 1 (substrate): improved-det is the dominant cheap floor

(`wave2_substrate_frontier.md`, `wave2_substrate_complementarity.md`)

- improved-det beats tree-sitter on invariant recall on EVERY shared cell
  (transformers ~0.40 vs ~0.20; vllm-v0.7.3 0.51 vs 0.43) and schema recall
  (vllm 0.97 vs 0.62), at the same near-zero cost (sub-second, no GPU).
- improved-det SUBSUMES tree-sitter: union adds only +3-7% recall. Do NOT union
  the two static substrates in production; run improved-det alone.
- The cheap-end ceiling is ~0.40-0.51 invariant recall vs GT on stable versions.
  The residual ~0.5-0.6 is not mechanically catchable by the current 7 primitives.
- **Schema is far easier than invariants** (improved-det schema 0.37-0.97 vs
  invariants 0.15-0.51). Treat the two tasks asymmetrically: a cheap det substrate
  can largely OWN schema; invariants need the LLM tail.
- DEFERRED substrates (infra-bound, not measured GPU-free): framework-reflection,
  runtime-trace, behavioural-fuzz, pyright-stubs, sphinx-xml, rag-over-source.
  framework-reflection is the highest-value deferred cell - it reads the resolved
  pydantic model, so it should be both COMPLEMENTARY to source-walkers and IMMUNE
  to the imperative->declarative shift that sinks them (see 2.2).

## 4. Axis 8 (version-situation): static floor is NOT bump-robust alone

(`wave2_bump_survivability.md`)

Same fixed substrate, recall vs each version's own GT:
- vllm 0.51 -> **0.15** (COLLAPSE: invariants moved to declarative Field).
- tensorrt 0.27 -> 0.40 (RISE: surface became more static-visible).
- transformers 0.40 -> 0.42 (flat).

Three decision-relevant consequences:
- The substrate does NOT signal its own collapse (silent under-emit). Self-update
  needs an EXTERNAL signal: runtime-gate acceptance rate, or an LLM diff-reviewer.
- Landmark/citation pinning (W-A status quo) fails ALL three bumps by construction.
- Pattern-matched primitives degrade gracefully (partial recall) rather than
  crashing - a real advantage over the status quo even when recall drops.

## 5. Axes 2/3/4 (LLM role, assembly, model scale): the small LLM is a weak extender

(`wave2_assembly_ladder.md`, `wave2_model_scale_curve.md`, `wave2_llm_role_matrix.md`.)
Measured on qwen2.5-coder-7b (+ llama-8b, phi4-14b) on a single A100; the
registered LLM strategies were "dispatch deferred" stubs, so a minimal Ollama
dispatch was wired for these cells.

- **W-G extend (improved-det floor + 7b LLM proposes the residual):** mean recall
  lift +0.020 (range +0.00 to +0.04), and precision DROPPED on EVERY cell (e.g.
  transformers 0.630 -> 0.464). ~2 precision points lost per recall point. At OSS
  scale the LLM-extend rung is net-negative once the precision loss + the gate
  work to clean it are counted.
- **Pure-b (7b LLM only):** 0.05-0.12 recall vs GT, 4x-30x BELOW the deterministic
  floor. The Wave 1 ~50% pure-extract ceiling (at 70B-q4) does NOT survive the
  drop to 7B.
- **Model scale (vllm v0.7.3):** floor 0.513 / 7b 0.513 / 8b 0.566 / 14b 0.566.
  The knee is ~8B and shallow; 7B adds zero, 14B nothing over 8B. There is no
  gradient to climb in the 7-14B band. The interesting 14B->70B region is
  UNMEASURED (single-GPU 40GB cap).
- **Hallucination proxy 0.87-1.0** (over-counts - GT is a minimum set - but the
  direction is clear): small models emit mostly-unverifiable entries. Any
  LLM-touching path needs the runtime gate. The transformers in-process gate is
  functional but the token-economical prompts omitted the per-entry
  kwargs_positive/negative replay fields, so the live gate infra-errored on floor +
  W-G + pure-b alike (NOT an LLM-specific failure); vllm/tensorrt gates need
  containers (deferred).
- **The recall ceiling lives in the SUBSTRATE, not the small LLM.** At <=14B OSS
  scale the LLM's value is in JUDGMENT roles (gate, diagnose, diff-review - the
  roles Wave 1 found it good at), NOT extraction.

## 6. Workflow-candidate implications (W-A .. W-G)

(`wave2_workflow_comparison.md` for the full table.) Wave 2 does NOT choose; the
evidence so far says:
- **W-A (status quo, landmark+hand producer):** fails self-update on every bump
  (Section 4). The baseline to beat.
- **W-B (pure universal substrate):** inherits the vllm bump cliff with no
  recovery path. Quality-bounded by the substrate; not bump-robust alone.
- **W-G (improved-det floor + LLM extend):** the evidence SPLITS the a-priori
  claim. The floor half is confirmed strongest (improved-det is the best cheap
  floor, owns schema + ~0.4-0.5 of invariants). But the LLM-extend half is NOT
  supported at OSS scale (Section 5: +0.02 recall, precision drops, ~0.9
  hallucination proxy) - a small LLM does not close the residual and does not
  recover the vllm bump cliff. So the supported shape is "improved-det floor as
  primary + LLM in gate/diagnose roles + frontier-LLM extend deferred", NOT W-G as
  originally framed. See `wave2_workflow_comparison.md`.
- The fixed runtime-validate gate is doing real work regardless of workflow: the
  static substrates over-emit (precision 0.23-0.63), and the gate is what keeps
  false positives out of vendored output.

## 7. Concrete recommendations to the engineering session

These are evidence-backed DESIGN CONSTRAINTS, not a workflow choice:
1. **Run improved-det as the deterministic floor**, not tree-sitter, and not a
   union of static substrates.
2. **Split the two tasks:** let the cheap det floor own schema; budget the LLM
   for invariants.
3. **Add a declarative-`Field` constraint primitive ("Primitive 8")** to the det
   floor. This is the mechanical fix for the vllm bump cliff and the
   imperative->declarative trend; likely the highest-ROI engineering item.
4. **Make the diff/gate convention-tolerant.** Exact-identity catalogue diffing
   reports false "lost invariant" regressions on pure naming drift (the strict vs
   tolerant 16x gap). Use leaf-field + coarse-predicate matching.
5. **Treat env vars as first-class** (0% baseline coverage; 169+ entries).
6. **Self-update needs an external degradation signal**, not substrate trust:
   gate-acceptance-rate or an LLM diff-reviewer (W-F diagnose role) to catch
   silent recall collapse on a bump.
7. **Evaluate framework-reflection** in a per-version container before finalising
   the floor - it may be both complementary and bump-immune.
8. **Do NOT use a small (<=14B OSS) LLM as an extractor or trusted extender** -
   it is net-negative on recall/precision (Section 5). Give the LLM GATE +
   DIAGNOSE/DIFF-REVIEW roles only, behind the runtime gate. Defer the
   LLM-as-extractor question to a frontier-scale (32B+/70B+/API) re-test before
   committing the LLM to any producing role.

## 8. What is unknown / Wave 3

- framework-reflection, runtime-trace, behavioural-fuzz recall vs GT (need
  per-version GPU containers).
- The declarative-`Field` Primitive 8's actual recovery of the vllm cliff
  (hypothesised, not built).
- Large/frontier models (32B+/70B+, Claude/GPT API): single-GPU 40GB cap +
  no-API constraint this run. The 2xA100/80GB path (DS01 `container deploy`) is
  reachable for 32B/70B-q4 but tty-gated; deferred.
- True hallucination/gate-rejection rates for vllm/tensorrt LLM cells (need the
  per-engine runtime-validate containers).
- bump-UPDATE cells (auto-propose a producer/catalogue patch that passes the gate
  without human edit) - the true self-update binary.

## 9. Artefact index

- GT: `findings/ground_truth/<engine>/<version>/` (+ canonical: `..._canonical/`)
- Substrate: `scripts/strategies/wave2/a_improved_det.py`; matrix
  `findings/wave2_substrate_matrix.json`; analysis `..._analysis.json`
- Scoring harness: `scripts/gt_adapter.py`, `scripts/gt_scoring.py`,
  `scripts/run_substrate_matrix.py`, `scripts/compute_substrate_analysis.py`
- Rescore: `findings/wave1_rescored_against_gt.md`
- Deviations / matching tables: `findings/wave2_deviations.md`
- Synthesis: `findings/wave2_substrate_frontier.md`, `..._complementarity.md`,
  `..._bump_survivability.md`, `..._failure_mode_catalogue.md`,
  `..._assembly_ladder.md`, `..._model_scale_curve.md`,
  `..._llm_role_matrix.md`, `..._workflow_comparison.md`
- LLM cells: `findings/wave2_llm_cells.json`, `..._llm_cells_findings.md`,
  `findings/wave2_locked_prompts/`, scripts `scripts/wave2_llm_*.py`
- Narrative: `DECISIONS_LOG.md` (2026-06-05 entries)
