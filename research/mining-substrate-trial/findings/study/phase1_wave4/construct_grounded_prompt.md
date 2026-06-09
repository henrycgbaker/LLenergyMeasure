# Phase 1 Wave 4 - construction-grounded extract + kwargs prompt (NO floor)

**Locked:** 2026-06-10. Derived from `pure_llm_kwargs_prompt.md`: ADDS an
INPUT 1 = CONSTRUCTOR SIGNATURES block (AST-extracted required/optional fields
per native_type) so the model emits CONSTRUCTIBLE kwargs and stops hitting the
infra_error wall. Strategy: deterministic AST does the construction-context
discovery; the LLM does the synthesis. Still llm-only for the invariant content
(no floor of invariants injected).

```text
You are extracting the COMPLETE validation-invariant catalogue for {engine}
v{engine_version} directly from source, AND emitting a runnable probe pair per
invariant that the validation gate can execute.

An "invariant" is a rule the library enforces at config/params construction:
`if <predicate>: raise ValueError/TypeError(...)`, `warnings.warn(...)`, or a
silent normalisation / dormant override.

INPUT 1 - CONSTRUCTOR SIGNATURES (the native_types you may probe, with their
constructor fields). REQUIRED fields have NO default - you MUST pass every
REQUIRED field (use a minimal valid value) for the probe to construct:

{class_signatures}

INPUT 2 - VALIDATOR SOURCE (this chunk only):

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
  native_type: <one of the classes in INPUT 1>
  miner_source:
    path: <file path as shown in the chunk header>
    method: <validate|__init__|__post_init__|_verify_args|...>
    line_at_scan: <approximate line number if visible>
  match:
    engine: {engine}
    fields:
      <namespace>.<field>: <value or predicate>
  kwargs_positive:
    <ctor_arg>: <value>   # constructs native_type AND TRIGGERS the rule
  kwargs_negative:
    <ctor_arg>: <value>   # constructs native_type AND PASSES (no fire)
  added_by: llm_construct_grounded

THE PROBE PAIR IS LOAD-BEARING - and INPUT 1 is how you make it construct:
- Use the EXACT constructor field names from INPUT 1's signature for native_type.
- Include EVERY field marked REQUIRED in that signature (minimal valid value),
  even ones you are not testing - otherwise the constructor raises for the wrong
  reason and the probe is wasted.
- kwargs_positive sets the tested field(s) so the invariant FIRES; kwargs_negative
  differs ONLY in what makes it PASS. For a CROSS-FIELD rule set all fields in the
  relation. For a conditional/presence rule, positive activates the guarded branch.

PREDICATE FORMS for match.fields (use the EXACT keys shown):
- Exact: `<ns>.field: value`   - Not in list: `<ns>.field: {present: true, not_in: [a, b]}`
- Not equal: `<ns>.field: {present: true, not_equal: value}`
- Greater than: `<ns>.field: {'>': value}`   - Less than: `<ns>.field: {'<': value}`
- Greater or equal: `<ns>.field: {'>=': value}`   - Less or equal: `<ns>.field: {'<=': value}`
- Type not in: `<ns>.field: {present: true, type_is_not: [TypeName]}`
- Presence: `<ns>.field: {present: true}`

CRITICAL RULES:
1. Return ONLY the YAML document. NO code fences. NO commentary. First two chars `in`.
2. Extract ONLY invariants VISIBLE in INPUT 2. Do NOT invent. Ground kwargs in
   INPUT 1's signatures + INPUT 2's types/defaults.
3. ONE invariant per `if <cond>:` guard; one per independent field for `if A or B`.
4. Cover single-field bounds/types/enums AND cross-field relations, conditional
   guards, presence checks.
5. If a native_type is NOT in INPUT 1 (cannot be constructed in isolation), still
   emit the invariant + match but set kwargs_positive/kwargs_negative to null.

Emit the YAML now:
```

## Notes for the runner / scorer

- Same gate + scoring as wave4_pure (recall vs GT, internal dedup, no floor).
- INPUT 1 is built per-chunk by the AST signature extractor in
  `scripts/phase1/wave4_construct.py` (classes whose names appear in the chunk).
- Internals-guard + mandatory cross-field adversarial verification still apply.
```
