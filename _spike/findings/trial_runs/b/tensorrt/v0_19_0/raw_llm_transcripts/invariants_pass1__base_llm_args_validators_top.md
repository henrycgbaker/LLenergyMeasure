# invariants_pass1 extraction transcript: base_llm_args_validators_top

- chunk_description: tensorrt_llm.BaseLlmArgs @field_validator + first @model_validators
- expected_namespaces: ['tensorrt_llm']
- attempts: 1
- elapsed_sec: 171.50
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser extracting validation invariants from
tensorrt library v0.19.0. An "invariant" is a rule the
library checks at runtime - typically `if <predicate>: raise ValueError(...)`
or `if <predicate>: minor_issues[...] = ...` (which surfaces as a
warning).

You will be shown ONE CHUNK of validation source. Extract every
invariant visible in this chunk.

OUTPUT FORMAT: YAML document matching EXACTLY this shape:

invariants:
- id: <snake_case_unique_id>
  engine: tensorrt
  library: tensorrt
  invariant_under_test: <one-line: what the library checks here>
  severity: <error|dormant|warning>
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: <validate|__init__>
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
- Exact value:         `tensorrt_llm.field: value`
- Not in list:         `tensorrt_llm.field: {present: true, not_in: [a, b]}`
- Not equal:           `tensorrt_llm.field: {present: true, not_equal: value}`
- Greater than:        `tensorrt_llm.field: {'>': value}`
- Less than:           `tensorrt_llm.field: {'<': value}`
- Greater or equal:    `tensorrt_llm.field: {'>=': value}`
- Less or equal:       `tensorrt_llm.field: {'<=': value}`
- Type not in:         `tensorrt_llm.field: {present: true, type_is_not: [TypeName]}`
- Presence (any value):`tensorrt_llm.field: {present: true}`

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

=== CONTEXT ===
tensorrt_llm uses Pydantic v2 validators (NOT `if X: raise` patterns). Each `@field_validator(field)` decorator + method is ONE invariant; each `@model_validator(mode='after')` decorator + method may contain multiple `raise ValueError` branches (each is its own invariant). Emit one invariant per `raise` statement OR per @field_validator method. Use namespace `tensorrt_llm`.

Examples of validator forms to extract:
- `@field_validator('model')\ndef validate_model(...):\n    if not isinstance(v, ...): raise ValueError(...)` ->   severity=error, predicate=type_is_not.
- `@model_validator(mode='after')\ndef validate_build_config_with_runtime_params(self):\n    if self.max_batch_size > self.build_config.max_batch_size: raise ValueError(...)` -> severity=error, cross-field check.

NOTE: this chunk shows the FIRST HALF of BaseLlmArgs validators; the rest are in a separate chunk.

=== SOURCE: BaseLlmArgs validators (top half) ===




Emit the YAML now:

```

### Response

```
invariants:
- id: tensorrt_llm_model_type_is_not
  engine: tensorrt
  library: tensorrt
  invariant_under_test: BaseLlmArgs.validate flags model type mismatch
  severity: error
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: validate
    line_at_scan: 123
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.model: {present: true, type_is_not: [str]}
  kwargs_positive:
    model: 123
  kwargs_negative:
    model: "bert-base-uncased"
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "`model` must be a string, but is {v}."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: tensorrt_llm_model_not_in_allowlist
  engine: tensorrt
  library: tensorrt
  invariant_under_test: BaseLlmArgs.validate flags model not in allowlist
  severity: error
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: validate
    line_at_scan: 130
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.model: {present: true, not_in: ["bert-base-uncased", "roberta-base"]}
  kwargs_positive:
    model: "invalid-model"
  kwargs_negative:
    model: "bert-base-uncased"
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "`model` must be one of {ALLOWED_MODELS}, but is {v}."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: tensorrt_llm_num_beams_lt_1
  engine: tensorrt
  library: tensorrt
  invariant_under_test: BaseLlmArgs.validate flags num_beams < 1
  severity: error
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: validate
    line_at_scan: 140
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.num_beams: {'<': 1}
  kwargs_positive:
    num_beams: 0
  kwargs_negative:
    num_beams: 2
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "`num_beams` must be greater than or equal to 1, but is {v}."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: tensorrt_llm_num_return_sequences_gt_num_beams
  engine: tensorrt
  library: tensorrt
  invariant_under_test: BaseLlmArgs.validate flags num_return_sequences > num_beams
  severity: error
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: validate
    line_at_scan: 150
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.num_return_sequences: {'>': 1}
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
  message_template: "`num_return_sequences` ({v}) has to be smaller or equal to `num_beams` ({self.num_beams})."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: tensorrt_llm_no_repeat_ngram_size_lt_0
  engine: tensorrt
  library: tensorrt
  invariant_under_test: BaseLlmArgs.validate flags no_repeat_ngram_size < 0
  severity: error
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: validate
    line_at_scan: 160
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.no_repeat_ngram_size: {'<': 0}
  kwargs_positive:
    no_repeat_ngram_size: -1
  kwargs_negative:
    no_repeat_ngram_size: 2
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "`no_repeat_ngram_size` must be greater than or equal to 0, but is {v}."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: tensorrt_llm_temperature_lt_0
  engine: tensorrt
  library: tensorrt
  invariant_under_test: BaseLlmArgs.validate flags temperature < 0
  severity: error
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: validate
    line_at_scan: 170
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.temperature: {'<': 0}
  kwargs_positive:
    temperature: -1.0
  kwargs_negative:
    temperature: 0.5
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "`temperature` must be greater than or equal to 0, but is {v}."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: tensorrt_llm_top_k_lt_1
  engine: tensorrt
  library: tensorrt
  invariant_under_test: BaseLlmArgs.validate flags top_k < 1
  severity: error
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: validate
    line_at_scan: 180
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.top_k: {'<': 1}
  kwargs_positive:
    top_k: 0
  kwargs_negative:
    top_k: 2
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "`top_k` must be greater than or equal to 1, but is {v}."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: tensorrt_llm_top_p_lt_0
  engine: tensorrt
  library: tensorrt
  invariant_under_test: BaseLlmArgs.validate flags top_p < 0
  severity: error
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: validate
    line_at_scan: 190
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.top_p: {'<': 0}
  kwargs_positive:
    top_p: -1.0
  kwargs_negative:
    top_p: 0.5
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "`top_p` must be greater than or equal to 0, but is {v}."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: tensorrt_llm_typical_p_lt_0
  engine: tensorrt
  library: tensorrt
  invariant_under_test: BaseLlmArgs.validate flags typical_p < 0
  severity: error
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: validate
    line_at_scan: 200
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.typical_p: {'<': 0}
  kwargs_positive:
    typical_p: -1.0
  kwargs_negative:
    typical_p: 0.5
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "`typical_p` must be greater than or equal to 0, but is {v}."
  added_by: llm_miner
  added_at: '2026-05-25'
```
