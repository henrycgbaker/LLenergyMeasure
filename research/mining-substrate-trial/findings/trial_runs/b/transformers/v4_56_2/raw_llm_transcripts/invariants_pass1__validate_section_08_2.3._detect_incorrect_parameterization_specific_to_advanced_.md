# invariants_pass1 extraction transcript: validate_section_08_2.3._detect_incorrect_parameterization_specific_to_advanced_

- chunk_description: GenerationConfig.validate section: 2.3._detect_incorrect_parameterization_specific_to_advanced_
- expected_namespaces: ['transformers.sampling']
- attempts: 1
- elapsed_sec: 108.45
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser extracting validation invariants from
transformers library v4.56.2. An "invariant" is a rule the
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

=== SOURCE: validate() section 9 (2.3._detect_incorrect_parameterization_specific_to_advanced_) ===
# 2.3. detect incorrect parameterization specific to advanced beam modes
        else:
            # constrained beam search
            if self.constraints is not None or self.force_words_ids is not None:
                constrained_wrong_parameter_msg = (
                    "one of `constraints`, `force_words_ids` is not `None`, triggering constrained beam search. "
                    "However, `{flag_name}` is set to `{flag_value}`, which is incompatible with this generation "
                    "mode. Set `constraints` and `force_words_ids` to `None` or unset `{flag_name}` to continue."
                )
                if self.do_sample is True:
                    raise ValueError(
                        constrained_wrong_parameter_msg.format(flag_name="do_sample", flag_value=self.do_sample)
                    )
                if self.num_beam_groups is not None and self.num_beam_groups != 1:
                    raise ValueError(
                        constr
...<truncated>...
```

### Response

```
invariants:
- id: transformers_do_sample_set_when_constrained_beam_search
  engine: transformers
  invariant_under_test: GenerationConfig.validate flags do_sample=True when constrained beam search is triggered
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: validate
    line_at_scan: 1234
  match:
    engine: transformers
    fields:
      transformers.sampling.constraints: {present: true}
      transformers.sampling.force_words_ids: {present: true}
      transformers.sampling.do_sample: true
  kwargs_positive:
    do_sample: true
    constraints: []
  kwargs_negative:
    do_sample: false
    constraints: []
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "one of `constraints`, `force_words_ids` is not `None`, triggering constrained beam search. However, `{flag_name}` is set to `{flag_value}`, which is incompatible with this generation mode. Set `constraints` and `force_words_ids` to `None` or unset `{flag_name}` to continue."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: transformers_num_beam_groups_set_when_constrained_beam_search
  engine: transformers
  invariant_under_test: GenerationConfig.validate flags num_beam_groups!=1 when constrained beam search is triggered
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: validate
    line_at_scan: 1234
  match:
    engine: transformers
    fields:
      transformers.sampling.constraints: {present: true}
      transformers.sampling.force_words_ids: {present: true}
      transformers.sampling.num_beam_groups: {'>': 1}
  kwargs_positive:
    num_beam_groups: 2
    constraints: []
  kwargs_negative:
    num_beam_groups: 1
    constraints: []
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "one of `constraints`, `force_words_ids` is not `None`, triggering constrained beam search. However, `{flag_name}` is set to `{flag_value}`, which is incompatible with this generation mode. Set `constraints` and `force_words_ids` to `None` or unset `{flag_name}` to continue."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: transformers_do_sample_set_when_group_beam_search
  engine: transformers
  invariant_under_test: GenerationConfig.validate flags do_sample=True when group beam search is triggered
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: validate
    line_at_scan: 1234
  match:
    engine: transformers
    fields:
      transformers.sampling.diversity_penalty: {'>': 0.0}
      transformers.sampling.num_beam_groups: {'>': 1}
      transformers.sampling.do_sample: true
  kwargs_positive:
    do_sample: true
    diversity_penalty: 0.5
  kwargs_negative:
    do_sample: false
    diversity_penalty: 0.5
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "`diversity_penalty` is not 0.0 or `num_beam_groups` is not 1, triggering group beam search. In this generation mode, `{flag_name}` must be set to `{flag_value}`"
  added_by: llm_miner
  added_at: '2026-05-25'

- id: transformers_num_beams_not_divisible_by_num_beam_groups_when_group_beam_search
  engine: transformers
  invariant_under_test: GenerationConfig.validate flags num_beams not divisible by num_beam_groups when group beam search is triggered
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: validate
    line_at_scan: 1234
  match:
    engine: transformers
    fields:
      transformers.sampling.diversity_penalty: {'>': 0.0}
      transformers.sampling.num_beam_groups: {'>': 1}
      transformers.sampling.num_beams: {not_divisible_by: transformers.sampling.num_beam_groups}
  kwargs_positive:
    num_beams: 5
    num_beam_groups: 3
  kwargs_negative:
    num_beams: 6
    num_beam_groups: 3
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "`diversity_penalty` is not 0.0 or `num_beam_groups` is not 1, triggering group beam search. In this generation mode, `{flag_name}` should be divisible by `{other_flag_name}`"
  added_by: llm_miner
  added_at: '2026-05-25'

- id: transformers_diversity_penalty_zero_when_group_beam_search
  engine: transformers
  invariant_under_test: GenerationConfig.validate flags diversity_penalty=0.0 when group beam search is triggered
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: validate
    line_at_scan: 1234
  match:
    engine: transformers
    fields:
      transformers.sampling.diversity_penalty: {'<=': 0.0}
      transformers.sampling.num_beam_groups: {'>': 1}
  kwargs_positive:
    diversity_penalty: 0.0
    num_beam_groups: 2
  kwargs_negative:
    diversity_penalty: 0.5
    num_beam_groups: 2
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "`diversity_penalty` is not 0.0 or `num_beam_groups` is not 1, triggering group beam search. In this generation mode, `{flag_name}` should be greater than `0.0`, otherwise your groups will be identical."
  added_by: llm_miner
  added_at: '2026-05-25'
```
