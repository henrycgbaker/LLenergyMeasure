# Wave 2 scope (info-generation framing)

**Status:** Locked 2026-06-05. Supersedes the implicit "find-the-winner" framing of WAVE2_PROTOCOL.md drafts.

## What this is

Wave 2 is RESEARCH, not engineering. It generates the empirical information that a subsequent ENGINEERING + DESIGN exercise will consume to build the production CI workflow(s) for keeping LLEM's engine-config catalogues current as upstream engines bump.

The deliverable of Wave 2 is a comprehensive **characterisation of the decision landscape**: for each axis of design choice (substrate, LLM role, assembly, model scale, etc.), what works, what doesn't, what the cost-recall tradeoffs are, and where the interaction effects lie. The eventual workflow's shape is NOT chosen during Wave 2; that's the downstream engineering exercise.

## What this is not

- Not a "pick the best substrate" exercise. Wave 1 already converged on Architecture II + V; Wave 2 widens the evidence base, doesn't relitigate that conclusion.
- Not a finished product. No CI hook gets shipped from Wave 2.
- Not a single-number leaderboard. Each experiment yields multi-dimensional evidence (recall + precision + cost + failure modes + interactions).

## Production target the research informs

The eventual CI workflow that Wave 2's evidence supports designing should have these properties (per user statement 2026-06-05):

1. **Cheap where possible** - the deterministic baseline carries as much as it can.
2. **Comprehensive recall in the proposal stage** - looser tolerances OK at proposal.
3. **Clean, guaranteed deterministic validation gate at the end** - no false positives reach vendored output. This gate is the runtime-validate-against-engine pattern that already exists on main (`scripts/validate_invariants.py`).
4. **Self-updating** - the workflow degrades gracefully as new engine versions land; it doesn't require a human PR per bump.
5. **LLMs in multiple roles** - not only proposers. Can be gates, decision-makers, curators, diff reviewers.
6. **Composable** - multiple deterministic tools + multiple LLM steps unioned where complementary. Not committed to a single pipeline shape.

These properties are not assumptions to be tested; they are constraints the eventual design satisfies. Wave 2's job is producing the evidence that lets the designer choose how to satisfy them.

## Tasks Wave 2 covers

Per the per-engine taxonomy in DECISIONS_LOG (2026-06-05), three task types matter:

- **Task 1: schema discovery** - enumerate all configurable fields per engine.
- **Task 2: invariant mining** - extract validation rules / cross-field predicates.
- **Task 3 (provisional): invalid-config mining** - enumerate known-invalid config tuples as a regression corpus. Defaulting to "same axis as Task 2" until evidence shows it benefits from its own pipeline.

Each axis below should be characterised PER TASK independently. A primitive that wins on schema may lose on invariants.

## Open questions carried forward (set sensibly, can revise on evidence)

1. **Task 2 vs Task 3 split**: defaulting to single task until evidence shows splitting helps.
2. **Self-updating definition**: experiments measure BOTH auto-PR readiness and workflow-knowledge-update behaviour; let evidence decide which matters more.
3. **Re-scoring against ground truth**: once GT lands, re-score every Wave 1 cell against GT AND keep validated-union score for comparison.
4. **SGLang + LMDeploy**: out of scope for Wave 2; revisit once the core 3 are characterised.
5. **Agentic / tool-use at higher model scales**: Wave 1 found collapse at 70B-q4; the question of whether Claude or unquantised 70B+ recovers it is deferred (Wave 3).

## Cross-references

- `WAVE2_PROTOCOL.md` - the experimental protocol (to be rewritten around this scope).
- `WAVE2_PRIMITIVES.md` - the axes inventory.
- `WAVE2_INFRA_SETUP.md` - what infrastructure each experiment needs.
- `DECISIONS_LOG.md` - chronological narrative; the authoritative record.
- `findings/wave2_treesitter_probe.md` - first empirical finding (tree-sitter on both tasks).
- `findings/ground_truth/<engine>/v<v>/` - per-engine ground-truth artefacts (in progress).
