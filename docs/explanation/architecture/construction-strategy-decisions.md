---
title: Engine-knowledge construction strategy decisions
description: Why the engine-knowledge construction pipeline mixes a deterministic floor, a targeted LLM proposer, and an engine-as-adjudicator gate, with the empirical basis for each disposition.
---

# Engine-knowledge construction strategy decisions

This is a decision record. It explains WHY the engine-knowledge construction
pipeline (the machinery that builds each engine's typed schema and validated
invariant corpus on every version bump) uses the deterministic-versus-LLM
strategy mix it does, and why several plausible alternatives are deliberately
absent from the shipped code.

A contributor reading the repository sees four families of approach present
(deterministic introspection and AST walkers, a deterministic constraint
walker, a gated LLM bump-diagnoser, and an in-container construction gate) and
several plausible alternatives absent (a pure-LLM extractor, an agentic
tool-loop, multi-stage LLM chains). The dispositions below are not arbitrary:
each is the outcome of an internal mining-substrate evaluation that measured
roughly 150 strategy-by-setup-by-model-by-engine cells across the three
supported engines (transformers, vLLM, TensorRT-LLM) and several real
consecutive version bumps. Where a number appears below, it is a measured
result from that evaluation, expressed as recall (fraction of the
ground-truth invariant set recovered), precision (fraction of proposals that
were genuine), or a verified-real count (gate-confirmed and then independently
audited as a true constraint).

For pipeline mechanics see
[Engine introspection pipelines](engine-introspection-pipelines.md) and the
[Auto-refresh pipeline](auto-refresh-pipeline.md). For the runtime side of the
corpus see [Parameter discovery](parameter-discovery.md).

## The principle: a layered pipeline with one adjudicator

The pipeline is three layers, and the engine itself decides what is true.

```
LAYER 0  Cheap deterministic floor       owns the bulk; runs on EVERY bump; CPU-affordable
   |     (introspection + AST walkers)
   v
LAYER 1  Targeted LLM enhancement        fills only the tail the floor structurally
   |     (gated proposer, >=32B code)     cannot reach; conditional, diff-scoped
   v
GATE     Engine-as-adjudicator           the engine's own in-container construction
         (in-container construction)      confirms EVERY proposal; nothing lands unconfirmed
```

Three rules make this work:

1. The engine is the single source of truth. No proposal, deterministic or
   LLM-sourced, lands in a shipped artefact until the engine's own
   construction or validation confirms it in-container. The gate, not the
   miner, decides what is true.
2. Proposers optimise recall; the gate optimises precision. This is encoded
   in `scripts/engine_producers/build_corpus.py`: "the merger optimises for
   recall; validation-gate is the single architectural gate." Both the floor
   and the LLM emit liberally; the gate quarantines false positives.
3. Cheap owns the bulk, expensive owns the tail. The deterministic floor
   recovers near-complete structural schema coverage and a substantial
   fraction of invariant recall at near-zero marginal cost per bump. The LLM
   is reserved for exactly the surface the floor structurally cannot see.

Because divergent proposals quarantine rather than ship, a proposer can be
recall-greedy without endangering correctness: a hallucinated invariant dies
at the gate. This is what licenses an aggressive LLM role at all.

## Strategy dispositions

Each row gives a disposition and the measured reason for it. The two products
(schema and invariants) share the floor machinery and the LLM harness; they
differ only in gate depth (schema confirms field presence, invariants run a
full positive and negative construction probe).

| Approach | Disposition | Empirical reason |
|---|---|---|
| Deterministic introspection + AST / declarative walkers | KEPT (the workhorse) | Owns the bulk of recall at near-zero cost per bump and is never beaten by any pure-LLM tier (floor 46.6% recall versus pure-LLM 42.3% at the same model). Schema discovery is essentially deterministic (per-field agreement near 1.0). Implemented as the per-engine introspectors and miners plus the shared `_source_walker` declarative walk and the pydantic/msgspec lifts. |
| Construction-grounding: deterministic constraint walker | KEPT | Highest-return deterministic result measured. When vLLM migrated value constraints into declarative `Field(...)` metadata, the single-field imperative floor fell off a cliff (0.513 to 0.147 tolerant invariant recall); the declarative subpackage-glob walker recovered roughly 44% of that cliff (0.147 to 0.309, +21 true positives) with no LLM, and lifted TensorRT-LLM by +0.100. Shipped as `walk_declarative_constraints` and the lift helpers. |
| Construction-grounding: LLM constructor-signature extraction | ADOPTED (productionizing; not yet shipped) | The only measured method that breaks the TensorRT-LLM construction-validation wall, where the Python constructor accepts structurally invalid configs and real constraints fire later in native code. Pure-LLM with no constructor grounding confirmed 0 of 61 there (100% infra-blocked); injecting the AST constructor signature took a code-tuned 32B model to 23 confirmed / 20 verified-real, and recovered the cross-field tail at vLLM (5 verified-real cross-field relations where the floor found 0). Status: the deterministic half is shipped; the constructor-signature injection (`_class_signature` / `format_sig_block` feeding the diagnose prompts) is planned and not yet present in the tree. |
| Gated LLM bump-diagnose | KEPT (shipped) | On a real version bump a code-tuned 32B model reached precision/recall 1.00 on classification, 0 fabrications, gate-confirmed 4 of 4, in 73 seconds (matching a 70B general model at roughly 3.8x less wall time and half the VRAM). The negative control matters: an unscoped diff-reviewer scored only about 6.7% precision, which is exactly why the role is diff-scoped and gate-disposed rather than standalone. Shipped as `src/llenergymeasure/api/diagnose.py` (carried-residual triage plus gap-diagnose). |
| Engine-construction gate / soundness checks | KEPT (the backbone) | The sole adjudicator. The kwargs-emission lever plus a cross-field attribution fix moved match-only extension from 0 gate-confirmed lift to 8 verified-real cross-field constraints; the carried re-gate acceptance rate (measured trending 95.0% down to 92.4% across consecutive vLLM bumps) is the clean cross-bump decay signal. Shipped as `scripts/validate_rules.py` (`validate_engine`, `compute_gate_soundness_divergences`). |
| Pure-LLM extraction (no deterministic floor) | REJECTED | Measured below the floor at every tier: 42.3% recall versus the 46.6% floor at the same model, collapsing to 25.0% at a smaller model, and 0 of 61 confirmed on the TensorRT-LLM wall. Removing source chunking (a whole-source single shot) HALVED recall (to 17.9%), so the deficit is not a chunking artefact. It cannot be the primary extractor; it is value-add only as a gated topping. |
| Agentic / tool-loop extraction | REJECTED (for this use) | Robust 0 finalised across every executed cell: a 30-call tool loop hit its cap with an empty result on both cells tested, and a fixed-harness re-run including an agentic-tuned model still produced 0 gate-confirmed, driven by tool-call flakiness and no incremental emit. Cost greatly exceeds the (zero) recall; deterministic exploration is strictly better here. (The rejection is for the models evaluated; it is not a proof that no tool-caller could ever work.) |
| Multi-stage LLM chains | REJECTED / DEFERRED | At best parity, never superiority: after a setup fix a two-stage chain reached recall 20 versus 21 for the grounded single shot, and a hybrid chain trailed (49 versus 55). The cell is explicitly under-explored and setup-confounded (no inter-stage validation or state threading), so it is deferred rather than ruled out, worth revisiting only if floor-plus-diagnose recall plateaus below target. |

## Constraints learned empirically

These are the conditions under which the measured LLM lift actually holds.
Getting any of them wrong collapses the result to or below the floor.

- The local code-model floor is a dense, code-tuned model of at least 32B
  parameters. The lever is family-specific and capability-gated:
  - Below 14B emits type-malformed kwargs that the gate drops.
  - At 7B the result is a trap, not a help: it gate-confirmed 4 of 4 with
    0 of 7 correct, because it echoes the carried probes rather than
    reasoning about them. A high confirm rate does not imply good output.
  - A 30B mixture-of-experts code model and a 16B code model do not
    generalise (they fail to break the wall or regress on it).
  - Code-tuning beats raw scale: a 32B code model beats a 70B general model
    (recall 21 versus 15) at roughly a third of the wall time, and a 72B
    general model collapsed entirely (degenerate under-emission).
  The default model is pinned to the qwen2.5-coder line accordingly
  (`DEFAULT_MODEL` in `src/llenergymeasure/api/diagnose.py`).
- LLM proposals must carry constructible positive and negative probes, or the
  gate cannot confirm them. Match-only extension (propose a field, ask the
  gate to check it) produced ZERO gate-confirmed lift at every tier, including
  the strongest model tested: the bottleneck was the validation path, not the
  model. Only proposals carrying constructible `kwargs_positive` /
  `kwargs_negative` against a real constructor signature become gateable
  (this is what moved 0 to 8 verified-real). The planned constructor-signature
  injection exists precisely to make those probes constructible.

## Why the engine is the source of truth

The gate is what makes every other decision safe. `validate_rules.py` runs
each candidate in-container against the engine's own construction path:
construct with the positive kwargs (which fires the engine's real validators),
capture every channel (exception type and message, warnings, logger output,
post-construction state), construct with the negative kwargs to confirm the
rule does not fire on a valid value, classify the outcome, and compare
declared against observed. Any mismatch is a divergence; divergent candidates
quarantine to a failed-validation file and never reach the corpus. CI always
runs with divergence treated as fatal.

This single adjudicator is why proposers can afford to be recall-greedy, why a
deterministic floor and an LLM proposer can feed the same corpus on equal
footing, and why an absent strategy is absent on measured grounds rather than
on caution: any approach that could not produce more gate-confirmed, audited
constraints than the deterministic floor has no role to play, because the
floor already runs for free on every bump and the gate already guarantees
correctness.
