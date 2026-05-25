# invariants_pass3_extend extraction transcript: validate_section_05_2._Validation_of_attribute_combinations

- chunk_description: GenerationConfig.validate section: 2._Validation_of_attribute_combinations
- expected_namespaces: ['transformers.sampling']
- attempts: 1
- elapsed_sec: 171.73
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser doing PASS 3 (extension) of multi-pass
invariant extraction. PASS 1 already extracted N invariants from
transformers v4.55.4 for ONE chunk of source. Your job is to
identify ADDITIONAL invariants PASS 1 missed.

INPUT 1 - PASS 1 EXTRACTED INVARIANTS (already known; do NOT re-emit):

invariants:
- id: transformers_num_return_sequences_gt_num_beams
  severity: error
  match:
    engine: transformers
    fields:
      transformers.sampling.num_return_sequences:
        '>': 1
  invariant_under_test: GenerationConfig.validate flags num_return_sequences > num_beams


INPUT 2 - PASS 2 FLAGGED ISSUES (for context; consider whether they hint
at related misses):

(no flags)

INPUT 3 - THE SOURCE PASS 1 READ:

=== CONTEXT: GenerationConfig.validate() ===
This is one logical section. Other sections are mined separately. Focus only on validations visible in THIS section.

=== SOURCE: validate() section 6 (2._Validation_of_attribute_combinations) ===
# 2. Validation of attribute combinations



OUTPUT FORMAT: YAML document with the SAME shape as the original
invariants prompt's output:

invariants:
- id: <snake_case_unique_id>
  engine: transformers
  library: transformers
  invariant_under_test: <one-line: what the library checks here>
  severity: <error|dormant|warning>
  native_type: <e.g. transformers.GenerationConfig or transformers.EngineArgs>
  miner_source:
    path: <file path>
    method: <validate|__init__|_verify_args|...>
    line_at_scan: <approximate line number if visible>
  match:
    engine: transformers
    fields:
      transformers.sampling.<field>: <value or predicate>
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
- Exact value:         `transformers.sampling.field: value`
- Not in list:         `transformers.sampling.field: {present: true, not_in: [a, b]}`
- Greater than:        `transformers.sampling.field: {'>': value}`
- Less than:           `transformers.sampling.field: {'<': value}`
- Type not in:         `transformers.sampling.field: {present: true, type_is_not: [TypeName]}`
- Presence:            `transformers.sampling.field: {present: true}`

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
  - id: transformers_num_beams_gt_zero
    engine: transformers
    library: transformers
    invariant_under_test: GenerationConfig.validate checks num_beams > 0
    severity: error
    native_type: transformers.GenerationConfig
    miner_source:
      path: generation_config.py
      method: validate
      line_at_scan: 123
    match:
      engine: transformers
      fields:
        transformers.sampling.num_beams:
          '>': 0
    kwargs_positive:
      num_beams: 1
    kwargs_negative:
      num_beams: 0
    expected_outcome:
      outcome: error
      emission_channel: logger_warning_once
      normalised_fields: []
    message_template: 'num_beams must be greater than 0'
    added_by: llm_miner_pass3
    added_at: '2026-05-25'

  - id: transformers_temperature_set_when_do_sample_true
    engine: transformers
    library: transformers
    invariant_under_test: GenerationConfig.validate checks temperature when do_sample is True
    severity: error
    native_type: transformers.GenerationConfig
    miner_source:
      path: generation_config.py
      method: validate
      line_at_scan: 145
    match:
      engine: transformers
      fields:
        transformers.sampling.do_sample: true
        transformers.sampling.temperature:
          present: true
    kwargs_positive:
      do_sample: True
      temperature: 1.0
    kwargs_negative:
      do_sample: False
      temperature: None
    expected_outcome:
      outcome: error
      emission_channel: logger_warning_once
      normalised_fields: []
    message_template: 'temperature must be set when do_sample is True'
    added_by: llm_miner_pass3
    added_at: '2026-05-25'

  - id: transformers_top_p_set_when_do_sample_true
    engine: transformers
    library: transformers
    invariant_under_test: GenerationConfig.validate checks top_p when do_sample is True
    severity: error
    native_type: transformers.GenerationConfig
    miner_source:
      path: generation_config.py
      method: validate
      line_at_scan: 167
    match:
      engine: transformers
      fields:
        transformers.sampling.do_sample: true
        transformers.sampling.top_p:
          present: true
    kwargs_positive:
      do_sample: True
      top_p: 0.9
    kwargs_negative:
      do_sample: False
      top_p: None
    expected_outcome:
      outcome: error
      emission_channel: logger_warning_once
      normalised_fields: []
    message_template: 'top_p must be set when do_sample is True'
    added_by: llm_miner_pass3
    added_at: '2026-05-25'

  - id: transformers_top_k_set_when_do_sample_true
    engine: transformers
    library: transformers
    invariant_under_test: GenerationConfig.validate checks top_k when do_sample is True
    severity: error
    native_type: transformers.GenerationConfig
    miner_source:
      path: generation_config.py
      method: validate
      line_at_scan: 189
    match:
      engine: transformers
      fields:
        transformers.sampling.do_sample: true
        transformers.sampling.top_k:
          present: true
    kwargs_positive:
      do_sample: True
      top_k: 50
    kwargs_negative:
      do_sample: False
      top_k: None
    expected_outcome:
      outcome: error
      emission_channel: logger_warning_once
      normalised_fields: []
    message_template: 'top_k must be set when do_sample is True'
    added_by: llm_miner_pass3
    added_at: '2026-05-25'

  - id: transformers_min_length_set_when_do_sample_true
    engine: transformers
    library: transformers
    invariant_under_test: GenerationConfig.validate checks min_length when do_sample is True
    severity: error
    native_type: transformers.GenerationConfig
    miner_source:
      path: generation_config.py
      method: validate
      line_at_scan: 211
    match:
      engine: transformers
      fields:
        transformers.sampling.do_sample: true
        transformers.sampling.min_length:
          present: true
    kwargs_positive:
      do_sample: True
      min_length: 10
    kwargs_negative:
      do_sample: False
      min_length: None
    expected_outcome:
      outcome: error
      emission_channel: logger_warning_once
      normalised_fields: []
    message_template: 'min_length must be set when do_sample is True'
    added_by: llm_miner_pass3
    added_at: '2026-05-25'

  - id: transformers_max_length_set_when_do_sample_true
    engine: transformers
    library: transformers
    invariant_under_test: GenerationConfig.validate checks max_length when do_sample is True
    severity: error
    native_type: transformers.GenerationConfig
    miner_source:
      path: generation_config.py
      method: validate
      line_at_scan: 233
    match:
      engine: transformers
      fields:
        transformers.sampling.do_sample: true
        transformers.sampling.max_length:
          present: true
    kwargs_positive:
      do_sample: True
      max_length: 100
    kwargs_negative:
      do_sample: False
      max_length: None
    expected_outcome:
      outcome: error
      emission_channel: logger_warning_once
      normalised_fields: []
    message_template: 'max_length must be set when do_sample is True'
    added_by: llm_miner_pass3
    added_at: '2026-05-25'
```
