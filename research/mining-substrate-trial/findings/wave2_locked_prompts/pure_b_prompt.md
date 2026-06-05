# Wave 2 pure-b prompt (LLM-only extract, from scratch)

**Locked:** 2026-06-05 (Wave 2 LLM cells session).
**Role:** Axis-2 `extract` in the Axis-3 `llm-only` assembly (no floor).
**Reuses:** Wave 1 locked `phase2_locked_prompts/invariants_prompt.md`
(`INVARIANTS_PROMPT_TEMPLATE`), generalised so the field namespace is
parameterised per engine rather than hard-wired to transformers. This is the
"LLM ceiling at 7B" data point. The chunking + few-shot structure are
preserved verbatim from the Wave 1 lock.

```text
You are a code analyser extracting validation invariants from {engine}
library v{engine_version}. An "invariant" is a rule the library checks at
runtime - typically `if <predicate>: raise ValueError(...)` or
`if <predicate>: minor_issues[...] = ...` (which surfaces as a warning) or
`if <predicate>: warnings.warn(...)`.

You will be shown ONE CHUNK of validation source. Extract every invariant
visible in this chunk.

OUTPUT FORMAT: a YAML document matching EXACTLY this shape. Field namespace
MUST be `{field_namespace}` for engine params and `{sampling_namespace}` for
sampling/generation params:

invariants:
- id: <snake_case_unique_id_with_engine_prefix>
  engine: {engine}
  library: {engine}
  invariant_under_test: <one-line: what the library checks here>
  severity: <error|warning|dormant>
  native_type: <e.g. {engine}.GenerationConfig or {engine}.EngineArgs>
  miner_source:
    path: <file path as shown in the chunk header>
    method: <validate|__init__|__post_init__|_verify_args|...>
    line_at_scan: <approximate line number if visible>
  match:
    engine: {engine}
    fields:
      <namespace>.<field>: <value or predicate>
  added_by: llm_pure_b

INVARIANT TYPES TO EXTRACT (one per `if ... :` block typically):
1. ERROR (raises ValueError/TypeError at construction or validate()):
   - value not in allowed enum -> severity: error, predicate: not_in
   - type mismatch (`not isinstance(x, T)`) -> severity: error, type_is_not
   - value out of range (`<= 0`) -> severity: error, gt/lt
   - cross-field invalid combo -> severity: error
2. DORMANT (logs warning, parameter silently ignored / normalised):
   - sampling-only param set when sampling disabled -> dormant
   - beam-only param set when not beam search -> dormant
3. WARNING (logs, execution continues with user value).

PREDICATE FORMS for match.fields (use the EXACT keys shown):
- Exact value:        `<ns>.field: value`
- Not in list:        `<ns>.field: {present: true, not_in: [a, b]}`
- Not equal:          `<ns>.field: {present: true, not_equal: value}`
- Greater than:       `<ns>.field: {'>': value}`
- Less than:          `<ns>.field: {'<': value}`
- Greater or equal:   `<ns>.field: {'>=': value}`
- Less or equal:      `<ns>.field: {'<=': value}`
- Type not in:        `<ns>.field: {present: true, type_is_not: [TypeName]}`
- Presence (any val): `<ns>.field: {present: true}`

CRITICAL RULES:
1. Return ONLY the YAML document. NO markdown code fences. NO commentary.
   The first character must be `i` (from `invariants:`).
2. Extract ONLY invariants VISIBLE in the source below. Do NOT invent.
3. Use snake_case_with_engine_prefix for `id`.
4. Each `if <cond>: raise / warn / minor_issues[...] =` block = ONE invariant.
   If a guard checks several independent fields, emit one per field.
5. severity: error when source has `raise`; warning when `warnings.warn`/
   `logger.warning`; dormant when the param is silently ignored/normalised.

VALIDATOR SOURCE (this chunk only):

{source}

Emit the YAML now:
```

## Merge / score

Across-chunk outputs are concatenated and deduplicated by
`(leaf_field, coarse_predicate_bucket)` before scoring vs GT (same tolerant
key the scorer uses). No floor is included - this is the LLM-only catalogue.
