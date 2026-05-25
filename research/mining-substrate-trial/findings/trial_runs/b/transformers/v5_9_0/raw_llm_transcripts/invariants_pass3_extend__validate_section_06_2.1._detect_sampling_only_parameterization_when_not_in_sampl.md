# invariants_pass3_extend extraction transcript: validate_section_06_2.1._detect_sampling_only_parameterization_when_not_in_sampl

- chunk_description: GenerationConfig.validate section: 2.1._detect_sampling_only_parameterization_when_not_in_sampl
- expected_namespaces: ['transformers.sampling']
- attempts: 1
- elapsed_sec: 122.80
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser doing PASS 3 (extension) of multi-pass
invariant extraction. PASS 1 already extracted N invariants from
transformers v5.9.0 for ONE chunk of source. Your job is to
identify ADDITIONAL invariants PASS 1 missed.

INPUT 1 - PASS 1 EXTRACTED INVARIANTS (already known; do NOT re-emit):

invariants:
- id: transformers_temperature_set_when_do_sample_false
  severity: dormant
  match:
    engine: transformers
    fields:
      transformers.sampling.do_sample: false
      transformers.sampling.temperature:
        present: true
  invariant_under_test: GenerationConfig.validate flags temperature set when do_sample=False
- id: transformers_top_p_set_when_do_sample_false
  severity: dormant
  match:
    engine: transformers
    fields:
      transformers.sampling.do_sample: false
      transformers.sampling.top_p:
        present: true
  invariant_under_test: GenerationConfig.validate flags top_p set when do_sample=False
- id: transformers_top_k_set_when_do_sample_false
  severity: dormant
  match:
    engine: transformers
    fields:
      transformers.sampling.do_sample: false
      transformers.sampling.top_k:
        present: true
  invariant_under_test: GenerationConfig.validate flags top_k set when do_sample=False
- id: transformers_min_length_set_when_do_sample_false
  severity: dormant
  match:
    engine: transformers
    fields:
      transformers.sampling.do_sample: false
      transformers.sampling.min_length:
        present: true
  invariant_under_test: GenerationConfig.validate flags min_length set when do_sample=False
- id: transformers_max_length_set_when_do_sample_false
  severity: dormant
  match:
    engine: transformers
    fields:
      transformers.sampling.do_sample: false
      transformers.sampling.max_length:
        present: true
  invariant_under_test: GenerationConfig.validate flags max_length set when do_sample=False
- id: transformers_no_repeat_ngram_size_set_when_do_sample_false
  severity: dormant
  match:
    engine: transformers
    fields:
      transformers.sampling.do_sample: false
      transformers.sampling.no_repeat_ngram_size:
        present: true
  invariant_under_test: GenerationConfig.validate flags no_repeat_ngram_size set when
    do_sample=False
- id: transformers_num_beams_set_when_do_sample_false
  severity: dormant
  match:
    engine: transformers
    fields:
      transformers.sampling.do_sample: false
      transformers.sampling.num_beams:
        present: true
  invariant_under_test: GenerationConfig.validate flags num_beams set when do_sample=False
- id: transformers_num_return_sequences_set_when_do_sample_false
  severity: dormant
  match:
    engine: transformers
    fields:
      transformers.sampling.do_sample: false
      transformers.sampling.num_return_sequences:
        present: true
  invariant_under_test: GenerationConfig.validate flags num_return_sequences set when
    do_sample=False
- id: transformers_num_beam_groups_set_when_do_sample_false
  severity: dormant
  match:
    engine: transformers
    fields:
      transformers.sampling.do_sample: false
      transformers.sampling.num_beam_groups:
        present: true
  invariant_under_test: GenerationConfig.validate flags num_beam_groups set when do_sample=False
- id: transformers_diversity_penalty_set_when_do_sample_false
  severity: dormant
  match:
    engine: transformers
    fields:
      transformers.sampling.do_sample: false
      transformers.sampling.diversity_penalty:
        present: true
  invariant_under_test: GenerationConfig.validate flags diversity_penalty set when
    do_sample=False
- id: transformers_repetition_penalty_set_when_do_sample_false
  severity: dormant
  match:
    engine: transformers
    fields:
      transformers.sampling.do_sample: false
      transformers.sampling.repetition_penalty:
        present: true
  invariant_under_test: GenerationConfig.validate flags repetition_penalty set when
    do_sample=False
- id: transformers_repetition_penalty_range_set_when_do_sample_false
  severity: dormant
  match:
    engine: transformers
    fields:
      transformers.sampling.do_sample: false
      transformers.sampling.repetition_penalty_range:
        present: true
  invariant_under_test: GenerationConfig.validate flags repetition_penalty_range set
    when do_sample=False
- id: transformers_repetition_penalty_slope_set_when_do_sample_false
  severity: dormant
  match:
    engine: transformers
    fields:
      transformers.sampling.do_sample: false
      transformers.sampling.repetition_penalty_slope:
        present: true
  invariant_under_test: GenerationConfig.validate flags repetition_penalty_slope set
    when do_sample=False
- id: transformers_encoder_no_repeat_ngram_size_set_when_do_sample_false
  severity: dormant
  match:
    engine: transformers
    fields:
      transformers.sampling.do_sample: false
      transformers.sampling.encoder_no_repeat_ngram_size:
        present: true
  invariant_under_test: GenerationConfig.validate flags encoder_no_repeat_ngram_size
    set when do_sample=False
- id: transformers_encoder_repetition_penalty_set_when_do_sample_false
  severity: dormant
  match:
    engine: transformers
    fields:
      transformers.sampling.do_sample: false
      transformers.sampling.encoder_repetition_penalty:
        presen
# ...<truncated>...


INPUT 2 - PASS 2 FLAGGED ISSUES (for context; consider whether they hint
at related misses):

(no flags)

INPUT 3 - THE SOURCE PASS 1 READ:

=== CONTEXT: GenerationConfig.validate() ===
This is one logical section. Other sections are mined separately. Focus only on validations visible in THIS section.

IMPORTANT: Each `if self.<field> is not None and <cond>:` block must produce ONE invariant per <field>. Do NOT collapse multiple field checks into one invariant keyed by `do_sample` or `num_beams`. Each per-field check is its OWN dormant invariant. Key by the FIELD being flagged (temperature, top_p, min_p, etc.), NOT by the gating field (do_sample / num_beams).


=== SOURCE: validate() section 7 (2.1._detect_sampling_only_parameterization_when_not_in_sampl) ===
# 2.1. detect sampling-only parameterization when not in sampling mode



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

...<truncated>...
```

### Response

```
invariants:
- id: transformers_typical_p_set_when_do_sample_false
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.validate flags typical_p set when do_sample=False
  severity: dormant
  native_type: transformers.GenerationConfig
  miner_source:
    path: generation_config.py
    method: validate
    line_at_scan: 123
  match:
    engine: transformers
    fields:
      transformers.sampling.do_sample: false
      transformers.sampling.typical_p:
        present: true
  kwargs_positive:
    typical_p: 0.5
  kwargs_negative:
    typical_p: null
  expected_outcome:
    outcome: dormant_announced
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: 'typical_p must be set when do_sample=False'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: transformers_epsilon_cutoff_set_when_do_sample_false
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.validate flags epsilon_cutoff set when do_sample=False
  severity: dormant
  native_type: transformers.GenerationConfig
  miner_source:
    path: generation_config.py
    method: validate
    line_at_scan: 126
  match:
    engine: transformers
    fields:
      transformers.sampling.do_sample: false
      transformers.sampling.epsilon_cutoff:
        present: true
  kwargs_positive:
    epsilon_cutoff: 0.1
  kwargs_negative:
    epsilon_cutoff: null
  expected_outcome:
    outcome: dormant_announced
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: 'epsilon_cutoff must be set when do_sample=False'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: transformers_eta_cutoff_set_when_do_sample_false
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.validate flags eta_cutoff set when do_sample=False
  severity: dormant
  native_type: transformers.GenerationConfig
  miner_source:
    path: generation_config.py
    method: validate
    line_at_scan: 129
  match:
    engine: transformers
    fields:
      transformers.sampling.do_sample: false
      transformers.sampling.eta_cutoff:
        present: true
  kwargs_positive:
    eta_cutoff: 0.5
  kwargs_negative:
    eta_cutoff: null
  expected_outcome:
    outcome: dormant_announced
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: 'eta_cutoff must be set when do_sample=False'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: transformers_temperature_must_be_positive
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.validate temperature is positive
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: generation_config.py
    method: validate
    line_at_scan: 132
  match:
    engine: transformers
    fields:
      transformers.sampling.temperature: {'>': 0}
  kwargs_positive:
    temperature: 1.0
  kwargs_negative:
    temperature: -1.0
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: 'temperature must be positive'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
```
