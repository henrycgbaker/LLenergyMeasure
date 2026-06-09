# Phase 1 Wave 4 - pure-LLM extract + kwargs-emission prompt (NO floor)

**Locked:** 2026-06-09 (Phase 1 wave 4). Derived from the wave-2
`wg_extend_kwargs_prompt.md` (sha 7cd74960eab09e19): REMOVES the floor input +
the floor-dedup rule (this is the llm-only assembly - no deterministic floor),
and reframes "extend the miner" as "extract the FULL catalogue from scratch".
KEEPS the kwargs_positive/negative probe-emission requirement so the gate can
probe cross-field/conditional invariants directly. Output is floor-schema-compatible.

```text
You are a code analyser extracting the COMPLETE validation-invariant catalogue
for {engine} v{engine_version} directly from source. There is NO prior
catalogue - read the SOURCE in INPUT 1 and extract EVERY invariant visible in
this chunk, AND for each emit a runnable probe pair the validation gate can
execute.

An "invariant" is a rule the library enforces at config/params construction or
validation time: `if <predicate>: raise ValueError/TypeError(...)`,
`if <predicate>: warnings.warn(...)`, `if <predicate>: logger.warning(...)`,
or a silent normalisation / dormant override.

INPUT 1 - VALIDATOR SOURCE (this chunk only):

{source}

OUTPUT FORMAT: a YAML document with EXACTLY this shape (one list item per
invariant). Field namespace MUST be `{field_namespace}` for engine params and
`{sampling_namespace}` for sampling/generation params:

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
  kwargs_positive:
    <ctor_arg>: <value>   # constructs the native_type AND TRIGGERS the rule
  kwargs_negative:
    <ctor_arg>: <value>   # constructs the native_type AND PASSES (no fire)
  added_by: llm_pure_kwargs

THE PROBE PAIR (kwargs_positive / kwargs_negative) IS LOAD-BEARING:
- Both must be valid CONSTRUCTOR keyword args for `native_type` - use the
  PLAIN constructor field names (NOT the namespaced match keys).
- Include EVERY field the constructor REQUIRES to build without error (read the
  class definition / dataclass fields in the source); omitting a required field
  makes the probe un-runnable. Use minimal valid placeholders for required
  fields you are not testing.
- kwargs_positive must set the field(s) so the invariant FIRES (raises / warns /
  normalises). kwargs_negative must differ ONLY in what is needed to make it
  PASS. For a CROSS-FIELD rule, set ALL the fields in the relation (e.g. both
  sides of `A <= B`); positive violates the relation, negative satisfies it.
- For a PRESENCE / conditional rule, positive sets the field to the value that
  activates the guarded branch; negative leaves it at its safe default.

PREDICATE FORMS for match.fields (use the EXACT keys shown):
- Exact value: `<ns>.field: value`   - Not in list: `<ns>.field: {present: true, not_in: [a, b]}`
- Not equal: `<ns>.field: {present: true, not_equal: value}`
- Greater than: `<ns>.field: {'>': value}`   - Less than: `<ns>.field: {'<': value}`
- Greater or equal: `<ns>.field: {'>=': value}`   - Less or equal: `<ns>.field: {'<=': value}`
- Type not in: `<ns>.field: {present: true, type_is_not: [TypeName]}`
- Presence (any val): `<ns>.field: {present: true}`

CRITICAL RULES:
1. Return ONLY the YAML document. NO markdown code fences. NO commentary.
   The first two characters must be `in` (from `invariants:`).
2. Extract ONLY invariants VISIBLE in the SOURCE chunk. Do NOT invent. Do NOT
   fabricate kwargs values not grounded in the source's types/defaults.
3. Emit ONE invariant per `if <cond>:` guard. For `if A or B or C` emit one per
   independent field.
4. Cover the FULL surface visible in the chunk: single-field bounds/types/enums
   AND cross-field relations, conditional/feature-gate guards, presence checks.
5. severity: error when source has `raise`; warning when `warnings.warn` /
   `logger.warning`; dormant when the param is silently ignored / normalised.
6. If a class genuinely cannot be constructed in isolation (needs a live model /
   GPU / file), still emit the invariant + match, set kwargs_positive and
   kwargs_negative to `null`, and note it in invariant_under_test.

Emit the YAML now:
```

## Notes for the runner / scorer

- llm-only assembly: NO floor is injected or deducted. The full LLM catalogue is
  deduped INTERNALLY by tolerant (leaf, bucket) key, then gated. Scored for
  RECALL vs the runtime-gated GT (not lift-over-floor).
- The gate honours hand-authored kwargs directly (no gate code change).
- SOUNDNESS (load-bearing): cross-field confirms BYPASS single-field attribution;
  every cross-field confirm MUST be adversarially source-verified before counting.
  Internals-guard applies (drop private/underscore fields, type-trivia,
  observability, launch-state from the candidate-config tally).
```
