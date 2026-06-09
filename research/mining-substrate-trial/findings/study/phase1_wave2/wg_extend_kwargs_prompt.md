# Phase 1 Wave 2 - W-G extend + kwargs-emission prompt

**Locked:** 2026-06-09 (Phase 1 wave 2). Derived from the wave-1
`wg_extend_prompt.md` (sha 18251cb6979bde77); ADDS a requirement to emit
constructible `kwargs_positive` / `kwargs_negative` so the gate can probe
cross-field / conditional / presence invariants directly instead of relying on
single-field auto-synthesis (the wave-1 bottleneck). Output remains
floor-schema-compatible.

```text
You are a code analyser extending a deterministically-mined invariant
catalogue for {engine} v{engine_version}. A deterministic miner already
extracted the invariants in INPUT 1 from this engine's source. Your job is
to read the SOURCE in INPUT 2 and propose ADDITIONAL invariants the
deterministic miner MISSED - AND, for each, emit a runnable probe pair the
validation gate can execute.

An "invariant" is a rule the library enforces at config/params construction
or validation time: `if <predicate>: raise ValueError/TypeError(...)`,
`if <predicate>: warnings.warn(...)`, `if <predicate>: logger.warning(...)`,
or a silent normalisation / dormant override.

INPUT 1 - FLOOR INVARIANTS ALREADY MINED (do NOT re-emit these):

{floor_invariants}

INPUT 2 - VALIDATOR SOURCE (this chunk only):

{source}

OUTPUT FORMAT: a YAML document with EXACTLY this shape (one list item per
NEW invariant). Field namespace MUST be `{field_namespace}` for engine
params and `{sampling_namespace}` for sampling/generation params:

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
  added_by: llm_wg_extend_kwargs

THE PROBE PAIR (kwargs_positive / kwargs_negative) IS THE POINT OF THIS TASK:
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
   The first character must be `i` (from `invariants:`).
2. ONLY emit invariants NOT already in INPUT 1 (check by leaf field name AND
   predicate form). Do not duplicate the floor.
3. Extract ONLY invariants VISIBLE in the SOURCE chunk. Do NOT invent. Do NOT
   fabricate kwargs values not grounded in the source's types/defaults.
4. Emit ONE invariant per `if <cond>:` guard. For `if A or B or C` emit one per
   independent field.
5. Prioritise the deterministic miner's blind spots: cross-field relations,
   conditional/feature-gate guards (`enable_X` + dependency), presence checks,
   type checks, and less-common predicates (`<=`, `>=`, `not_equal`, `present`).
6. If a class genuinely cannot be constructed in isolation (needs a live model /
   GPU / file), still emit the invariant + match, set kwargs_positive and
   kwargs_negative to `null`, and note it in invariant_under_test.
7. If the floor already covered every visible invariant, emit `invariants: []`.

Emit the YAML now:
```

## Notes for the runner / scorer

- The gate honours hand-authored `kwargs_positive`/`kwargs_negative` directly
  (validate_invariants.py: "Use hand-authored probes when present"), so this
  prompt needs NO gate code change for the primary arm.
- SOUNDNESS CAVEAT (load-bearing for wave 2): the gate's positive-confirm
  ATTRIBUTION hardening is single-field (`_leaf_field` returns None for a
  multi-field `match`), so CROSS-FIELD confirms BYPASS attribution - a
  cross-field positive could fire for the wrong reason and confirm. Every
  cross-field confirm in wave 2 MUST be adversarially verified against source
  before being counted as real (mandatory, not optional).
