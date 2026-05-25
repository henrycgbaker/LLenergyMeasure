# Invariants extend prompt (Phase 2.5 pass 3)

Added: Phase 2.5. Locked. This is PASS 3 of the multi-pass refinement
pipeline; PASS 1 uses round-3-locked `invariants_prompt.md`; PASS 2
uses `invariants_verify_prompt.md`.

Purpose: identify ADDITIONAL invariants pass 1 missed. The primary
recall-lift mechanism in multi-pass refinement.

Source: `_spike/scripts/strategies/prompts.py::INVARIANTS_EXTEND_PROMPT_TEMPLATE`.

```text
You are a code analyser doing PASS 3 (extension) of multi-pass
invariant extraction. PASS 1 already extracted N invariants from
{engine} v{engine_version} for ONE chunk of source. Your job is to
identify ADDITIONAL invariants PASS 1 missed.

INPUT 1 - PASS 1 EXTRACTED INVARIANTS (already known; do NOT re-emit):

{pass1_invariants}

INPUT 2 - PASS 2 FLAGGED ISSUES (for context; consider whether they hint
at related misses):

{pass2_flags}

INPUT 3 - THE SOURCE PASS 1 READ:

{source}

OUTPUT FORMAT: YAML document with the SAME shape as the original
invariants prompt's output:

invariants:
- id: <snake_case_unique_id>
  engine: {engine}
  library: {engine}
  invariant_under_test: <one-line: what the library checks here>
  severity: <error|dormant|warning>
  native_type: <e.g. {engine}.GenerationConfig or {engine}.EngineArgs>
  miner_source:
    path: <file path>
    method: <validate|__init__|_verify_args|...>
    line_at_scan: <approximate line number if visible>
  match:
    engine: {engine}
    fields:
      {field_namespace}.<field>: <value or predicate>
  kwargs_positive:
    <field>: <value that TRIGGERS the invariant>
  kwargs_negative:
    <field>: <value that does NOT trigger>
  expected_outcome:
    outcome: <error|dormant_announced|warning>
    emission_channel: <none|logger_warning_once|logger_warning>
    normalised_fields: []
  message_template: '<the exact error/warning string from source>'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

PREDICATE FORMS (use the EXACT keys shown):
- Exact value:         `{field_namespace}.field: value`
- Not in list:         `{field_namespace}.field: {present: true, not_in: [a, b]}`
- Greater than:        `{field_namespace}.field: {'>': value}`
- Less than:           `{field_namespace}.field: {'<': value}`
- Type not in:         `{field_namespace}.field: {present: true, type_is_not: [TypeName]}`
- Presence:            `{field_namespace}.field: {present: true}`

CRITICAL RULES FOR PASS 3:
1. Return ONLY the YAML document. NO markdown code fences. NO commentary.
   First character must be `i` (from `invariants:`).
2. ONLY emit invariants that are NOT already in PASS 1's list (check by
   ID and by (field, predicate_kind) tuple).
3. Look ESPECIALLY for these PASS 1 blind spots:
   a. PER-FIELD invariants where PASS 1 emitted ONE entry covering many
      fields. Example: PASS 1 emitted `transformers_temperature_set_when_do_sample_false`
      but the source has SIMILAR if-blocks for top_p, top_k, min_p,
      typical_p, epsilon_cutoff, eta_cutoff. Emit ONE per field PASS 1
      missed.
   b. Multi-clause `if A and B and C:` invariants where PASS 1 keyed on
      only A. Emit one invariant per (A, B, C) tuple if each is independent.
   c. Type-check invariants `if not isinstance(x, T): raise` that PASS 1
      may have skipped (these often follow a similar pattern repeated
      per field).
   d. Less-common predicates: `<=`, `>=`, `not_equal`, `present`. PASS 1
      may have collapsed these to `exact`.
4. If PASS 1 covered every visible invariant, emit `invariants: []` (empty
   list) - do NOT fabricate.
5. Use snake_case_with_engine_prefix for `id`.
6. Set `added_by: llm_miner_pass3` (NOT `llm_miner`) so the multi-pass
   bookkeeping can track pass-3 contributions.

Emit the YAML now:
```

## Reconciliation contract

The orchestrator (`llm_b_oss.extract_invariants_multipass`) consumes the
output:

- Each emitted invariant is checked against the pass-1 surviving set
  by `(id, identity-tuple)`. Identities clashing with a pass-1 entry
  are DROPPED.
- Surviving pass-3 entries get `added_by: llm_miner_pass3` (if the LLM
  omitted) so the global score can report pass-3 contribution count.
- The combined output replaces what the legacy single-pass would have
  written.

## Rationale

Rule 3 is the load-bearing one. It explicitly enumerates pass-1's
known failure modes (per-field collapse, multi-clause collapse,
type-check skip, predicate-kind regression) and asks the LLM to look
for them. This is structured CoT prompting - the LLM does the recall
lift, the prompt orchestrates which gaps to fill.

Rule 4 is the precision guardrail. Empirically, an empty pass-1 chunk
will cause the LLM to fabricate invariants from section header context.
Pass-3 must be told that emitting an empty list is the correct response.

## Status

Locked at Phase 2.5. Not iterated against the calibration cell beyond
the initial smoke test (verified the pipeline shape works end-to-end).
