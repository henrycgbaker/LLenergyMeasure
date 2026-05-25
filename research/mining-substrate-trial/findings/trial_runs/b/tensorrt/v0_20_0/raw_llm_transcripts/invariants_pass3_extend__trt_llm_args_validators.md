# invariants_pass3_extend extraction transcript: trt_llm_args_validators

- chunk_description: tensorrt_llm.TrtLlmArgs @field_validator + @model_validator methods
- expected_namespaces: ['tensorrt_llm']
- attempts: 1
- elapsed_sec: 26.65
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser doing PASS 3 (extension) of multi-pass
invariant extraction. PASS 1 already extracted N invariants from
tensorrt v0.20.0 for ONE chunk of source. Your job is to
identify ADDITIONAL invariants PASS 1 missed.

INPUT 1 - PASS 1 EXTRACTED INVARIANTS (already known; do NOT re-emit):

invariants:
- id: tensorrt_calib_batch_size_not_in_allowlist
  severity: error
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.calib_batch_size:
        present: true
        not_in:
        - 1
        - 2
        - 4
        - 8
  invariant_under_test: TrtLlmArgs.validate flags calib_batch_size not in allowlist
- id: tensorrt_calib_max_batch_size_not_in_allowlist
  severity: error
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.calib_max_batch_size:
        present: true
        not_in:
        - 1
        - 2
        - 4
        - 8
  invariant_under_test: TrtLlmArgs.validate flags calib_max_batch_size not in allowlist
- id: tensorrt_calib_min_batch_size_not_in_allowlist
  severity: error
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.calib_min_batch_size:
        present: true
        not_in:
        - 1
        - 2
        - 4
        - 8
  invariant_under_test: TrtLlmArgs.validate flags calib_min_batch_size not in allowlist


INPUT 2 - PASS 2 FLAGGED ISSUES (for context; consider whether they hint
at related misses):

flagged:
- id: tensorrt_calib_batch_size_lt_calib_min_batch_size
  reason: Source does not contain a comparison between calib_batch_size and calib_min_batch_size.
  fix: drop


INPUT 3 - THE SOURCE PASS 1 READ:

=== CONTEXT ===
TrtLlmArgs has 1 @field_validator (calib_config init) + 3 @model_validator decorators. The validate_enable_build_cache method has the most pertinent `raise ValueError(...)` block.

=== SOURCE: TrtLlmArgs validators ===




OUTPUT FORMAT: YAML document with the SAME shape as the original
invariants prompt's output:

invariants:
- id: <snake_case_unique_id>
  engine: tensorrt
  library: tensorrt
  invariant_under_test: <one-line: what the library checks here>
  severity: <error|dormant|warning>
  native_type: <e.g. tensorrt.GenerationConfig or tensorrt.EngineArgs>
  miner_source:
    path: <file path>
    method: <validate|__init__|_verify_args|...>
    line_at_scan: <approximate line number if visible>
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.<field>: <value or predicate>
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
- Exact value:         `tensorrt_llm.field: value`
- Not in list:         `tensorrt_llm.field: {present: true, not_in: [a, b]}`
- Greater than:        `tensorrt_llm.field: {'>': value}`
- Less than:           `tensorrt_llm.field: {'<': value}`
- Type not in:         `tensorrt_llm.field: {present: true, type_is_not: [TypeName]}`
- Presence:            `tensorrt_llm.field: {present: true}`

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
- id: tensorrt_calib_batch_size_lt_calib_max_batch_size
  engine: tensorrt
  library: tensorrt
  invariant_under_test: TrtLlmArgs.validate checks calib_batch_size is less than or equal to calib_max_batch_size
  severity: error
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: TrtLlmArgs.py
    method: validate_enable_build_cache
    line_at_scan: 123
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.calib_batch_size: {'<=': tensorrt_llm.calib_max_batch_size}
  kwargs_positive:
    calib_batch_size: 16
    calib_max_batch_size: 8
  kwargs_negative:
    calib_batch_size: 4
    calib_max_batch_size: 8
  expected_outcome:
    outcome: error
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: 'calib_batch_size must be less than or equal to calib_max_batch_size'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
```
