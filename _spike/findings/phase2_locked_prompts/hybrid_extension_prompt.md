# Locked hybrid (d-ab / d-ac) extension prompt (Phase 2)

**Locked at:** Phase 2 calibration round 1 (initial baseline lock).
**Source:** `_spike/scripts/strategies/prompts.py` constant
`HYBRID_EXTENSION_PROMPT_TEMPLATE`.
**Used by:** strategies (d-ab) and (d-ac).

The hybrid extension prompt sends BOTH (a)'s deterministic-miner
output AND the engine source. The LLM's job: find what (a) missed,
flag what looks spurious, briefly diagnose root causes.

## Single-prompt design

Phase 2 sends ALL of (a)'s output + a curated source summary in ONE
prompt (not chunked). Justification:
- Transformers reference is ~40k chars; fits in 32k tokens with
  source-summary headroom.
- Hybrid needs to see the WHOLE existing catalogue to know what's
  already covered. Chunking would break that visibility.
- If we exceed context (large engines like vllm), the executor
  falls back to a compressed (a) summary (id + severity + field
  only).

## Output sections

Three top-level keys in the YAML:
1. `added_by_llm_verifier`: novel invariants (a) missed. Same shape
   as canonical invariants, plus `added_by: llm_verifier` and
   `flagged_for_review: true`.
2. `flagged_spurious_in_deterministic`: list of `{id, reason}` for
   (a)'s entries that look wrong. Annotated as `x-conflict:
   llm_flagged_spurious` in the merged proposed.yaml; NOT removed.
3. `missed_diagnosis`: list of `{id, why_missed}` for adjacent
   research data (Phase 4 synthesis).

## Reconciliation rules

Per Phase 2 design:
- Disputed entries SURFACE conflicts via `x-conflict`; never
  auto-resolved.
- Extension entries land in proposed.yaml with
  `added_by: llm_verifier` + `flagged_for_review: true`.
- Diagnoses go into a separate `reconciliation.yaml` for human
  review (not in proposed.yaml).

## Full prompt template

```
You are a code analyser working as a SECOND PASS on top of a
deterministic miner's output. The deterministic miner has already
extracted invariants from {engine} v{engine_version}; your job is to
find what it MISSED, find what looks SPURIOUS, and diagnose WHY it
missed what it missed.

INPUT 1 - DETERMINISTIC MINER'S OUTPUT (this is what (a) found):

{deterministic_output}

INPUT 2 - ENGINE SOURCE (the same source the deterministic miner read):

{source}

OUTPUT FORMAT: YAML document with THREE sections:

added_by_llm_verifier:
- # invariants the deterministic miner missed. Same shape as the
  # per-invariant entries it emits, plus `added_by: llm_verifier`
  # and `flagged_for_review: true`. Use the same predicate-form
  # vocabulary (`not_in`, `present`, `type_is_not`, etc.).

flagged_spurious_in_deterministic:
- id: <existing-invariant-id-from-deterministic-output>
  reason: <one-line: why this looks spurious or wrong>
  # NOTE: do not auto-resolve; just flag for human review.

missed_diagnosis:
- id: <a-missed-invariant-name>
  why_missed: <one-line: AST limitation, kwargs.pop pattern, etc.>

RULES:
1. ONLY emit invariants the deterministic output does NOT already
   contain. Check by `id` and by the (field, predicate_kind) tuple.
2. For each emitted invariant, set `added_by: llm_verifier` and
   `flagged_for_review: true` in the YAML body.
3. Be conservative: it's better to under-emit than to add spurious.
   The human reviewer will see your flagged_for_review entries.
4. NO markdown code fences. NO commentary outside the YAML.

Emit the YAML now:
```

## Variant: d-ac (Claude backend)

Same prompt verbatim; only the `backend=` argument changes from
`OllamaBackend()` to `AnthropicBackend()` in
`hybrid_extractor.run_d_ac_on_transformers_active`. The Claude
backend benefits from prompt caching on the SOURCE block - the
`{deterministic_output}` and `{source}` portions are marked
`cache_control: ephemeral` so re-prompts (retries) hit cache.
