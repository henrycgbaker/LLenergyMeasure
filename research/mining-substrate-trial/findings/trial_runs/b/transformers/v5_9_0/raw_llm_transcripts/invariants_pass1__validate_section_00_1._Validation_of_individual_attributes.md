# invariants_pass1 extraction transcript: validate_section_00_1._Validation_of_individual_attributes

- chunk_description: GenerationConfig.validate section: 1._Validation_of_individual_attributes
- expected_namespaces: ['transformers.sampling']
- attempts: 1
- elapsed_sec: 468.25
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser extracting validation invariants from
transformers library v5.9.0. An "invariant" is a rule the
library checks at runtime - typically `if <predicate>: raise ValueError(...)`
or `if <predicate>: minor_issues[...] = ...` (which surfaces as a
warning).

You will be shown ONE CHUNK of validation source. Extract every
invariant visible in this chunk.

OUTPUT FORMAT: YAML document matching EXACTLY this shape:

invariants:
- id: <snake_case_unique_id>
  engine: transformers
  library: transformers
  invariant_under_test: <one-line: what the library checks here>
  severity: <error|dormant|warning>
  native_type: transformers.GenerationConfig
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: <validate|__init__>
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
  message_template: '<the exact error/warning string from source, with {} placeholders preserved>'
  added_by: llm_miner
  added_at: '2026-05-25'

INVARIANT TYPES TO EXTRACT (one per `if ... :` block typically):

1. ERROR (raises ValueError at construction or validate()):
   - Field value not in allowed enum -> severity: error, predicate: not_in
   - Field type mismatch (e.g. `not isinstance(x, T)`) -> severity: error, predicate: type_is_not
   - Field value out of range (e.g. <= 0) -> severity: error, predicate: gt/lt
   - Cross-field invalid combo (e.g. num_return_sequences > num_beams) -> severity: error

2. DORMANT (logs warning, parameter silently ignored or normalised):
   - Sampling-only param set when do_sample=False -> severity: dormant
   - Beam-only param set when num_beams=1 -> severity: dormant
   - Cache-related dormancy -> severity: dormant

3. WARNING (logs, execution continues with user value):
   - pad_token_id < 0 -> severity: warning
   - Other non-blocking minor_issues entries -> severity: warning

PREDICATE FORMS for the `match.fields` block (use the EXACT keys shown):
- Exact value:         `transformers.sampling.field: value`
- Not in list:         `transformers.sampling.field: {present: true, not_in: [a, b]}`
- Not equal:           `transformers.sampling.field: {present: true, not_equal: value}`
- Greater than:        `transformers.sampling.field: {'>': value}`
- Less than:           `transformers.sampling.field: {'<': value}`
- Greater or equal:    `transformers.sampling.field: {'>=': value}`
- Less or equal:       `transformers.sampling.field: {'<=': value}`
- Type not in:         `transformers.sampling.field: {present: true, type_is_not: [TypeName]}`
- Presence (any value):`transformers.sampling.field: {present: true}`

CRITICAL RULES:
1. Return ONLY the YAML document. NO markdown code fences (no ```yaml).
   NO commentary, no preamble. First character must be `i` (from
   `invariants:`).
2. Extract ONLY invariants VISIBLE in the source below. Do not invent.
3. Use snake_case_with_engine_prefix for `id` (e.g. `transformers_cache_implementation_not_in_allowlist`).
4. Each `if <cond>: raise / minor_issues[...] = ` block = ONE invariant.
5. Set `severity: error` when the source has `raise ValueError(...)`.
   Set `severity: dormant` when the source assigns to `minor_issues`
   AND comments / context indicate the parameter is silently ignored.
   Set `severity: warning` when the source assigns to `minor_issues`
   AND there's no silent-ignore semantics.
6. For `kwargs_positive`: provide a concrete dict that WOULD trigger
   the invariant (so a downstream validator can replay it).
7. For `kwargs_negative`: provide a concrete dict that would NOT
   trigger (so the negative case is checkable).
8. `message_template`: the EXACT f-string literal from `raise` /
   `minor_issues[...]`. Preserve `{}` placeholders. Do NOT
   substitute concrete values.

FEW-SHOT EXAMPLES (from transformers v4.57.3 reference):

Example 1 (ERROR, enum violation):
Source: ``if self.early_stopping not in {True, False, "never"}: raise ValueError(f"`early_stopping` must be a boolean or 'never', but is {self.early_stopping}.")``
Emit:
- id: transformers_early_stopping_not_in_allowlist
  engine: transformers
  invariant_under_test: GenerationConfig.validate flags `early_stopping` not in allowlist
  severity: error
  match:
    engine: transformers
    fields:
      transformers.sampling.early_stopping: {present: true, not_in: [true, false, "never"]}
  kwargs_positive:
    early_stopping: "invalid"
  kwargs_negative:
    early_stopping: true
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "`early_stopping` must be a boolean or 'never', but is {self.early_stopping}."

Example 2 (DORMANT, sampling-only when greedy):
Source (under `if self.do_sample is False:`):
``if self.temperature is not None and self.temperature != 1.0:
    minor_issues["temperature"] = greedy_wrong_parameter_msg.format(...)``
Emit:
- id: transformers_temperature_set_when_do_sample_false
  engine: transformers
  invariant_under_test: GenerationConfig.validate flags `temperature` set when do_sample=False
  severity: dormant
  match:
    engine: transformers
    fields:
      transformers.sampling.do_sample: false
  kwargs_positive:
    do_sample: false
    temperature: 0.5
  kwargs_negative:
    do_sample: true
    temperature: 0.5
  expected_outcome:
    outcome: dormant_announced
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: "`do_sample` is set to `False`. However, `{flag_name}` is set to `{flag_value}` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `{flag_name}`."

Example 3 (CROSS-FIELD ERROR):
Source: ``if self.num_return_sequences > self.num_beams: raise ValueError(...)``
Emit:
- id: transformers_num_return_sequences_gt_num_beams
  engine: transformers
  invariant_under_test: GenerationConfig.validate flags num_return_sequences > num_beams
  severity: error
  match:
    engine: transformers
    fields:
      transformers.sampling.num_return_sequences: {'>': 1}
  kwargs_positive:
    num_beams: 2
    num_return_sequences: 3
  kwargs_negative:
    num_beams: 5
    num_return_sequences: 2
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "`num_return_sequences` ({self.num_return_sequences}) has to be smaller or equal to `num_beams` ({self.num_beams})."

=== CONTEXT: GenerationConfig.validate() ===
This is one logical section. Other sections are mined separately. Focus only on validations visible in THIS section.

=== SOURCE: validate() section 1 (1._Validation_of_individual_attributes) ===
# 1. Validation of individual attributes



Emit the YAML now:

```

### Response

```
invariants:
- id: transformers_max_length_lt_0
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.validate flags max_length < 0
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: validate
    line_at_scan: 123
  match:
    engine: transformers
    fields:
      transformers.sampling.max_length: {'<': 0}
  kwargs_positive:
    max_length: -1
  kwargs_negative:
    max_length: 10
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "`max_length` must be a positive integer, but is {self.max_length}."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: transformers_min_length_lt_0
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.validate flags min_length < 0
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: validate
    line_at_scan: 126
  match:
    engine: transformers
    fields:
      transformers.sampling.min_length: {'<': 0}
  kwargs_positive:
    min_length: -1
  kwargs_negative:
    min_length: 10
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "`min_length` must be a positive integer, but is {self.min_length}."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: transformers_no_repeat_ngram_size_lt_0
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.validate flags no_repeat_ngram_size < 0
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: validate
    line_at_scan: 129
  match:
    engine: transformers
    fields:
      transformers.sampling.no_repeat_ngram_size: {'<': 0}
  kwargs_positive:
    no_repeat_ngram_size: -1
  kwargs_negative:
    no_repeat_ngram_size: 10
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "`no_repeat_ngram_size` must be a positive integer, but is {self.no_repeat_ngram_size}."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: transformers_num_beams_lt_1
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.validate flags num_beams < 1
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: validate
    line_at_scan: 132
  match:
    engine: transformers
    fields:
      transformers.sampling.num_beams: {'<': 1}
  kwargs_positive:
    num_beams: 0
  kwargs_negative:
    num_beams: 10
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "`num_beams` must be a positive integer, but is {self.num_beams}."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: transformers_num_return_sequences_lt_1
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.validate flags num_return_sequences < 1
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: validate
    line_at_scan: 135
  match:
    engine: transformers
    fields:
      transformers.sampling.num_return_sequences: {'<': 1}
  kwargs_positive:
    num_return_sequences: 0
  kwargs_negative:
    num_return_sequences: 10
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "`num_return_sequences` must be a positive integer, but is {self.num_return_sequences}."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: transformers_num_beam_groups_lt_1
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.validate flags num_beam_groups < 1
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: validate
    line_at_scan: 138
  match:
    engine: transformers
    fields:
      transformers.sampling.num_beam_groups: {'<': 1}
  kwargs_positive:
    num_beam_groups: 0
  kwargs_negative:
    num_beam_groups: 10
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "`num_beam_groups` must be a positive integer, but is {self.num_beam_groups}."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: transformers_diversity_penalty_lt_0
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.validate flags diversity_penalty < 0
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: validate
    line_at_scan: 141
  match:
    engine: transformers
    fields:
      transformers.sampling.diversity_penalty: {'<': 0}
  kwargs_positive:
    diversity_penalty: -1.0
  kwargs_negative:
    diversity_penalty: 10.0
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "`diversity_penalty` must be a positive float, but is {self.diversity_penalty}."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: transformers_repetition_penalty_lt_0
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.validate flags repetition_penalty < 0
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: validate
    line_at_scan: 144
  match:
    engine: transformers
    fields:
      transformers.sampling.repetition_penalty: {'<': 0}
  kwargs_positive:
    repetition_penalty: -1.0
  kwargs_negative:
    repetition_penalty: 10.0
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "`repetition_penalty` must be a positive float, but is {self.repetition_penalty}."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: transformers_length_penalty_lt_0
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.validate flags length_penalty < 0
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: validate
    line_at_scan: 147
  match:
    engine: transformers
    fields:
      transformers.sampling.length_penalty: {'<': 0}
  kwargs_positive:
    length_penalty: -1.0
  kwargs_negative:
    length_penalty: 10.0
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "`length_penalty` must be a positive float, but is {self.length_penalty}."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: transformers_early_stopping_not_in_allowlist
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.validate flags early_stopping not in allowlist
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: validate
    line_at_scan: 150
  match:
    engine: transformers
    fields:
      transformers.sampling.early_stopping: {present: true, not_in: [true, false, "never"]}
  kwargs_positive:
    early_stopping: "invalid"
  kwargs_negative:
    early_stopping: true
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "`early_stopping` must be a boolean or 'never', but is {self.early_stopping}."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: transformers_typical_p_not_in_allowlist
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.validate flags typical_p not in allowlist
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    pa
...<truncated>...
```
