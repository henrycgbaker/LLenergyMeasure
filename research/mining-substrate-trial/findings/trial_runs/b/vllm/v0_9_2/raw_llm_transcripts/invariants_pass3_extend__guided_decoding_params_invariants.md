# invariants_pass3_extend extraction transcript: guided_decoding_params_invariants

- chunk_description: GuidedDecodingParams.__post_init__ mutual-exclusion check
- expected_namespaces: ['vllm.sampling']
- attempts: 1
- elapsed_sec: 37.05
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser doing PASS 3 (extension) of multi-pass
invariant extraction. PASS 1 already extracted N invariants from
vllm v0.9.2 for ONE chunk of source. Your job is to
identify ADDITIONAL invariants PASS 1 missed.

INPUT 1 - PASS 1 EXTRACTED INVARIANTS (already known; do NOT re-emit):

invariants:
- id: vllm_guided_decoding_params_mutual_exclusion
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.json:
        present: true
      vllm.sampling.regex:
        present: true
      vllm.sampling.choice:
        present: true
      vllm.sampling.grammar:
        present: true
      vllm.sampling.json_object:
        present: true
  invariant_under_test: GuidedDecodingParams.__post_init__ flags mutual exclusion
    of guide fields


INPUT 2 - PASS 2 FLAGGED ISSUES (for context; consider whether they hint
at related misses):

(no flags)

INPUT 3 - THE SOURCE PASS 1 READ:

=== CONTEXT ===
GuidedDecodingParams enforces mutual exclusion of its guide fields (only one of json/regex/choice/grammar/json_object may be set). Use namespace `vllm.sampling` for the field key.

=== SOURCE: GuidedDecodingParams.__post_init__ ===
    def __post_init__(self):
        """Validate that some fields are mutually exclusive."""
        guide_count = sum([
            self.json is not None, self.regex is not None, self.choice
            is not None, self.grammar is not None, self.json_object is not None
        ])
        if guide_count > 1:
            raise ValueError(
                "You can only use one kind of guided decoding but multiple are "
                f"specified: {self.__dict__}")

        if self.backend is not None and ":" in self.backend:
            self._extract_backend_options()




OUTPUT FORMAT: YAML document with the SAME shape as the original
invariants prompt's output:

invariants:
- id: <snake_case_unique_id>
  engine: vllm
  library: vllm
  invariant_under_test: <one-line: what the library checks here>
  severity: <error|dormant|warning>
  native_type: <e.g. vllm.GenerationConfig or vllm.EngineArgs>
  miner_source:
    path: <file path>
    method: <validate|__init__|_verify_args|...>
    line_at_scan: <approximate line number if visible>
  match:
    engine: vllm
    fields:
      vllm.sampling.<field>: <value or predicate>
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
- Exact value:         `vllm.sampling.field: value`
- Not in list:         `vllm.sampling.field: {present: true, not_in: [a, b]}`
- Greater than:        `vllm.sampling.field: {'>': value}`
- Less than:           `vllm.sampling.field: {'<': value}`
- Type not in:         `vllm.sampling.field: {present: true, type_is_not: [TypeName]}`
- Presence:            `vllm.sampling.field: {present: true}`

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

### Response

```
invariants:
  - id: vllm_backend_options_extraction_mutual_exclusion
    engine: vllm
    library: vllm
    invariant_under_test: GuidedDecodingParams extracts backend options when colon is present in backend string
    severity: error
    native_type: vllm.GuidedDecodingParams
    miner_source:
      path: GuidedDecodingParams.__post_init__
      method: __post_init__
      line_at_scan: 10
    match:
      engine: vllm
      fields:
        vllm.backend: {present: true, '>': ''}
    kwargs_positive:
      backend: 'option:value'
    kwargs_negative:
      backend: 'no_option'
    expected_outcome:
      outcome: error
      emission_channel: logger_warning_once
      normalised_fields: []
    message_template: ''
    added_by: llm_miner_pass3
    added_at: '2026-05-25'
```
