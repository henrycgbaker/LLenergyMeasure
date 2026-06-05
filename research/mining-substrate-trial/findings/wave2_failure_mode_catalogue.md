# Wave 2.6 deliverable: failure-mode catalogue + interactions

Per-cell failure modes observed in Wave 2, drawn from the closed vocabulary in
WAVE2_PROTOCOL (silent / detectable / crash / under-emit / over-emit /
gate-rejected-most / self-update-failed) plus the infra-bound modes surfaced this
session. Source: `wave2_substrate_matrix.json`, runner status records under
`findings/trial_scores/wave2/`, agent reports.

## Substrate (deterministic) failure modes

| mode | where observed | signal | decision impact |
|---|---|---|---|
| **over-emit** (low precision) | all static cells: tolerant inv precision 0.23-0.63 | visible (precision measurable vs GT) | proposal-stage OK; the runtime-validate gate must filter before vendoring. This is the designed division of labour. |
| **silent under-emit on bump** | vllm v0.7.3->v0.19.1 (recall 0.51->0.15) | INVISIBLE in substrate output alone | the dangerous one - see bump-survivability. Needs external signal (gate acceptance, LLM diff). |
| **unsupported_engine** | tree-sitter on tensorrt (registry) | detectable (registry declares it) | coverage gap, not a quality failure; pick a substrate that supports the engine. |
| **crash** | pydantic-native on vllm v0.7.3 (engine not importable in project venv) | detectable (exception record) | framework-reflection needs the engine importable = per-version GPU container. |
| **deferred** | pydantic-native vllm v0.19.1; all registered LLM strategies (LLM dispatch stubbed) | detectable (NotImplementedError record) | infra-bound; not a quality result. |

## Cross-cutting failure mode: cross-catalogue identity drift

Surfaced building the GT scorer. The SAME invariant is emitted by GT and by a
mining cell under different (namespace, predicate-kind) labels, producing
DISJOINT strict identities -> the strict scorer reports ~0.025-0.14 invariant
recall where the tolerant (label-insensitive) recall is 0.40-0.51 (a ~16x gap on
transformers). This is a failure mode of any production gate that compares
catalogues by exact identity: it will report false regressions on pure
convention drift. Mitigation adopted: tolerant matching on
(leaf_field, coarse_predicate_bucket); logged in `wave2_deviations.md`. The
production workflow's diff/gate step must be convention-tolerant or it will
generate spurious "lost invariant" alarms on every refactor bump.

## Failure-mode interactions (per assembly)

- **det-only** masks the over-emit cost into the downstream gate, and masks the
  bump under-emit entirely (no internal signal). So det-only's failure modes are
  "pushed downstream": precision -> gate, recall-collapse -> undetected.
- **det + LLM-extend (W-G)** is expected to convert silent under-emit into
  recoverable residual (the LLM targets exactly the entries the det floor
  missed), at the cost of introducing a NEW failure mode: LLM hallucination
  (proposed invariants with no source basis). The hallucination rate + whether
  the runtime gate catches it is measured in `wave2_assembly_ladder.md`.
- **pure-LLM (b)** trades the det modes for hallucination + non-determinism; its
  recall ceiling and hallucination proxy are in `wave2_assembly_ladder.md` /
  `wave2_model_scale_curve.md`.

## Self-update failure (the W-A status-quo mode)

Not run as a live cell (no v_new producer authored), but established by the GT
deltas: a landmark/citation-pinned producer (W-A) hits `self-update-failed` on
every one of the 3 bumps by construction (stale citations, hand-curated class
lists miss new classes). This is the baseline the self-updating candidates must
beat; documented in `wave2_bump_survivability.md`.

## Coverage-honesty ledger (what did NOT run and why)

- Static substrate matrix: 10/18 cells scored. 8 unscored = tree-sitter/tensorrt
  (unsupported) + pydantic-native (crash/deferred/unsupported).
- LLM cells: registered strategies were stubs; a minimal Ollama dispatch was
  wired for the W-G + pure-b measurement (`wave2_assembly_ladder.md`). Large
  models (32B+/70B+) deferred (single-GPU 40GB cap; see DECISIONS_LOG GPU
  resolution).
- runtime-trace, behavioural-fuzz, pyright-stubs, sphinx-xml, rag-over-source:
  not run (GPU/import-bound or out of time). Deferred to a container run.
