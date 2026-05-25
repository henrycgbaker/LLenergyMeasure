# invariants_pass3_extend extraction transcript: validate_section_00_1._Validation_of_individual_attributes

- chunk_description: GenerationConfig.validate section: 1._Validation_of_individual_attributes
- expected_namespaces: ['transformers.sampling']
- attempts: 1
- elapsed_sec: 134.85
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
- id: transformers_max_length_lt_0
  severity: error
  match:
    engine: transformers
    fields:
      transformers.sampling.max_length:
        <: 0
  invariant_under_test: GenerationConfig.validate flags max_length < 0
- id: transformers_min_length_lt_0
  severity: error
  match:
    engine: transformers
    fields:
      transformers.sampling.min_length:
        <: 0
  invariant_under_test: GenerationConfig.validate flags min_length < 0
- id: transformers_no_repeat_ngram_size_lt_0
  severity: error
  match:
    engine: transformers
    fields:
      transformers.sampling.no_repeat_ngram_size:
        <: 0
  invariant_under_test: GenerationConfig.validate flags no_repeat_ngram_size < 0
- id: transformers_num_beams_lt_1
  severity: error
  match:
    engine: transformers
    fields:
      transformers.sampling.num_beams:
        <: 1
  invariant_under_test: GenerationConfig.validate flags num_beams < 1
- id: transformers_num_return_sequences_lt_1
  severity: error
  match:
    engine: transformers
    fields:
      transformers.sampling.num_return_sequences:
        <: 1
  invariant_under_test: GenerationConfig.validate flags num_return_sequences < 1
- id: transformers_num_beam_groups_lt_1
  severity: error
  match:
    engine: transformers
    fields:
      transformers.sampling.num_beam_groups:
        <: 1
  invariant_under_test: GenerationConfig.validate flags num_beam_groups < 1
- id: transformers_diversity_penalty_lt_0
  severity: error
  match:
    engine: transformers
    fields:
      transformers.sampling.diversity_penalty:
        <: 0
  invariant_under_test: GenerationConfig.validate flags diversity_penalty < 0
- id: transformers_repetition_penalty_lt_0
  severity: error
  match:
    engine: transformers
    fields:
      transformers.sampling.repetition_penalty:
        <: 0
  invariant_under_test: GenerationConfig.validate flags repetition_penalty < 0
- id: transformers_temperature_lt_0
  severity: error
  match:
    engine: transformers
    fields:
      transformers.sampling.temperature:
        <: 0
  invariant_under_test: GenerationConfig.validate flags temperature < 0
- id: transformers_top_k_lt_0
  severity: error
  match:
    engine: transformers
    fields:
      transformers.sampling.top_k:
        <: 0
  invariant_under_test: GenerationConfig.validate flags top_k < 0
- id: transformers_top_p_lt_0
  severity: error
  match:
    engine: transformers
    fields:
      transformers.sampling.top_p:
        <: 0
  invariant_under_test: GenerationConfig.validate flags top_p < 0
- id: transformers_typical_p_lt_0
  severity: error
  match:
    engine: transformers
    fields:
      transformers.sampling.typical_p:
        <: 0
  invariant_under_test: GenerationConfig.validate flags typical_p < 0
- id: transformers_exponential_decay_length_penalty_lt_0
  severity: error
  match:
    engine: transformers
    fields:
      transformers.sampling.exponential_decay_length_penalty:
        <: 0
  invariant_under_test: GenerationConfig.validate flags exponential_decay_length_penalty
    < 0
- id: transformers_exponential_decay_bias_lt_0
  severity: error
  match:
    engine: transformers
    fields:
      transformers.sampling.exponential_decay_bias:
        <: 0
  invariant_under_test: GenerationConfig.validate flags exponential_decay_bias < 0
- id: transformers_length_penalty_lt_0
  severity: error
  match:
    engine: transformers
    fields:
      transformers.sampling.length_penalty:
        <: 0
  invariant_under_test: GenerationConfig.validate flags length_penalty < 0
- id: transformers_early_stopping_not_in_allowlist
  severity: error
  match:
    engine: transformers
    fields:
      transformers.sampling.early_stopping:
        present: true
        not_in:
        - true
        - false
        - never
  invariant_under_test: GenerationConfig.validate flags early_stopping not in allowlist


INPUT 2 - PASS 2 FLAGGED ISSUES (for context; consider whether they hint
at related misses):

flagged:
- id: transformers_do_sample_not_in_allowlist
  reason: Source does not validate do_sample against an allowlist.
  fix: drop
- id: transformers_output_scores_not_in_allowlist
  reason: Source does not validate output_scores against an allowlist.
  fix: drop
- id: transformers_output_attentions_not_in_allowlist
  reason: Source does not validate output_attentions against an allowlist.
  fix: drop
- id: transformers_output_hidden_states_not_in_allowlist
  reason: Source does not validate output_hidden_states against an allowlist.
  fix: drop
- id: transformers_remove_invalid_values_not_in_allowlist
  reason: Source does not validate remove_invalid_values against an allowlist.
  fix: drop
- id: transformers_early_stopping_not_in_allowlist
  reason: Source does not validate early_stopping against an allowlist with 'never'
    as a valid value.
  fix: correct_kwargs_positive


INPUT 3 - THE SOURCE PASS 1 READ:

=== CONTEXT: GenerationConfig.validate() ===
This is one logical section. Other sections are mined separately. Focus only on validations visible in THIS section.

=== SOURCE: validate() section 1 (1._Validation_of_individual_attributes) ===
# 1. Validation of individual attributes



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
   d. Less-common predicates: `<=`, `
...<truncated>...
```

### Response

```
invariants:
- id: transformers_do_sample_not_in_allowlist
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.validate checks do_sample against an allowlist
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: generation_config.py
    method: validate
    line_at_scan: 123
  match:
    engine: transformers
    fields:
      transformers.sampling.do_sample: {present: true, not_in: [true, false]}
  kwargs_positive:
    do_sample: null
  kwargs_negative:
    do_sample: true
  expected_outcome:
    outcome: error
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: 'do_sample must be one of [True, False]'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
- id: transformers_output_scores_not_in_allowlist
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.validate checks output_scores against an allowlist
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: generation_config.py
    method: validate
    line_at_scan: 145
  match:
    engine: transformers
    fields:
      transformers.sampling.output_scores: {present: true, not_in: [true, false]}
  kwargs_positive:
    output_scores: null
  kwargs_negative:
    output_scores: true
  expected_outcome:
    outcome: error
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: 'output_scores must be one of [True, False]'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
- id: transformers_output_attentions_not_in_allowlist
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.validate checks output_attentions against an allowlist
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: generation_config.py
    method: validate
    line_at_scan: 167
  match:
    engine: transformers
    fields:
      transformers.sampling.output_attentions: {present: true, not_in: [true, false]}
  kwargs_positive:
    output_attentions: null
  kwargs_negative:
    output_attentions: true
  expected_outcome:
    outcome: error
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: 'output_attentions must be one of [True, False]'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
- id: transformers_output_hidden_states_not_in_allowlist
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.validate checks output_hidden_states against an allowlist
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: generation_config.py
    method: validate
    line_at_scan: 189
  match:
    engine: transformers
    fields:
      transformers.sampling.output_hidden_states: {present: true, not_in: [true, false]}
  kwargs_positive:
    output_hidden_states: null
  kwargs_negative:
    output_hidden_states: true
  expected_outcome:
    outcome: error
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: 'output_hidden_states must be one of [True, False]'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
- id: transformers_remove_invalid_values_not_in_allowlist
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.validate checks remove_invalid_values against an allowlist
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: generation_config.py
    method: validate
    line_at_scan: 211
  match:
    engine: transformers
    fields:
      transformers.sampling.remove_invalid_values: {present: true, not_in: [true, false]}
  kwargs_positive:
    remove_invalid_values: null
  kwargs_negative:
    remove_invalid_values: true
  expected_outcome:
    outcome: error
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: 'remove_invalid_values must be one of [True, False]'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
```
