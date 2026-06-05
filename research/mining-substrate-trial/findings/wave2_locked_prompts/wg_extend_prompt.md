# Wave 2 W-G extend prompt (det-floor + small-LLM extend)

**Locked:** 2026-06-05 (Wave 2 LLM cells session).
**Role:** Axis-2 `extract` in the Axis-3 `det-then-llm-extends` assembly.
**Used by:** the W-G hybrid cells (improved-det floor + qwen/llama/phi extend).

Derived from the Wave 1 locked `invariants_extend_prompt.md` (PASS 3),
re-pointed so the "already known" input is the improved-det FLOOR catalogue
(not a prior LLM pass), and the source is a chunk of validator-bearing engine
code. Output shape is byte-compatible with the floor's
`invariants.proposed.yaml` so floor + extensions merge and score directly.

```text
You are a code analyser extending a deterministically-mined invariant
catalogue for {engine} v{engine_version}. A deterministic miner already
extracted the invariants in INPUT 1 from this engine's source. Your job is
to read the SOURCE in INPUT 2 and propose ADDITIONAL invariants the
deterministic miner MISSED.

An "invariant" is a rule the library enforces at config/params construction
or validation time: `if <predicate>: raise ValueError/TypeError(...)`,
`if <predicate>: warnings.warn(...)`, `if <predicate>: logger.warning(...)`,
or `if <predicate>: minor_issues[...] = ...` (silently normalised / dormant).

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
  added_by: llm_wg_extend

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
2. ONLY emit invariants NOT already in INPUT 1 (check by leaf field name AND
   predicate form). The floor already has many; do not duplicate them.
3. Extract ONLY invariants VISIBLE in the SOURCE chunk. Do NOT invent.
4. Emit ONE invariant per `if <cond>:` guard. If a guard checks several
   fields independently (`if A or B or C`), emit one per independent field.
5. Look ESPECIALLY for the deterministic miner's blind spots:
   a. PER-FIELD guards where many sibling fields share the same check
      (e.g. a loop or repeated `if x < 0: raise` across several fields).
   b. Multi-clause `if A and B:` guards keyed on a cross-field relation.
   c. Type checks `if not isinstance(x, T): raise` repeated per field.
   d. Less-common predicates: `<=`, `>=`, `not_equal`, `present`.
6. If the floor already covered every visible invariant, emit `invariants: []`
   (empty list) - do NOT fabricate to look productive.
7. Use snake_case_with_engine_prefix for `id`.

Emit the YAML now:
```

## Reconciliation

Merge step (in `wave2_llm_cells.py`): floor entries are kept verbatim; LLM
entries are appended UNLESS their `(leaf_field, coarse_predicate_bucket)`
tuple already exists in the floor (those are dropped as duplicates before
scoring, so the +LLM recall/precision reflect genuinely-new contributions).
`added_by: llm_wg_extend` tags every surviving LLM entry for counting.
