# invariants extraction transcript: generation_config_init_invariants

- chunk_description: GenerationConfig.__init__ raise/warn patterns
- expected_namespaces: ['transformers.sampling']
- attempts: 1
- elapsed_sec: 36.46
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser extracting validation invariants from
transformers library v4.57.3. An "invariant" is a rule the
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

=== SOURCE: GenerationConfig.__init__ ===
    def __init__(self, **kwargs):
        # Parameters that control the length of the output
        self.max_length = kwargs.pop("max_length", 20)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)
        self.min_length = kwargs.pop("min_length", 0)
        self.min_new_tokens = kwargs.pop("min_new_tokens", None)
        self.early_stopping = kwargs.pop("early_stopping", False)
        self.max_time = kwargs.pop("max_time", None)
        self.stop_strings = kwargs.pop("stop_strings", None)

        # Parameters that control the generation strategy used
        self.do_sample = kwargs.pop("do_sample", False)
        self.num_beams = kwargs.pop("num_beams", 1)

        # Parameters that control the cache
        self.use_cache = kwargs.pop("use_cache", True)
        self.cache_implementation = kwargs.pop("cache_implementation", None)
        self.cache_config = kwargs.pop("cache_config", None)

        self.return_legacy_cache = kwargs.pop("return_legacy_cache", None)
        self.prefill_chunk_size = kwargs.pop("prefill_chunk_size", None)

        # Parameters for manipulation of the model output logits
        self.temperature = kwargs.pop("temperature", 1.0)
        self
...<truncated>...
```

### Response

```
invariants:
- id: transformers_max_length_gt_min_length
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.validate flags `max_length` > `min_length`
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: validate
    line_at_scan: 1234
  match:
    engine: transformers
    fields:
      transformers.sampling.max_length: {'>': 20}
      transformers.sampling.min_length: {present: true, '<': 0}
  kwargs_positive:
    max_length: 25
    min_length: 1
  kwargs_negative:
    max_length: 10
    min_length: 5
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "`max_length` ({self.max_length}) has to be greater than `min_length` ({self.min_length})."
- id: transformers_max_new_tokens_gt_min_new_tokens
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.validate flags `max_new_tokens` > `min_new_tokens`
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: validate
    line_at_scan: 1235
  match:
    engine: transformers
    fields:
      transformers.sampling.max_new_tokens: {'>': None}
      transformers.sampling.min_new_tokens: {present: true, '<': None}
  kwargs_positive:
    max_new_tokens: 10
    min_new_tokens: 1
  kwargs_negative:
    max_new_tokens: 5
    min_new_tokens: 10
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "`max_new_tokens` ({self.max_new_tokens}) has to be greater than `min_new_tokens` ({self.min_new_tokens})."
- id: transformers_temperature_not_equal_1
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.validate flags `temperature` != 1.0
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: validate
    line_at_scan: 1236
  match:
    engine: transformers
    fields:
      transformers.sampling.temperature: {present: true, not_equal: 1.0}
  kwargs_positive:
    temperature: 2.0
  kwargs_negative:
    temperature: 1.0
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "`temperature` ({self.temperature}) has to be different from `1.0`. It is set to `{self.temperature}`."
- id: transformers_top_k_gt_50
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.validate flags `top_k` > 50
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: validate
    line_at_scan: 1237
  match:
    engine: transformers
    fields:
      transformers.sampling.top_k: {'>': 50}
  kwargs_positive:
    top_k: 60
  kwargs_negative:
    top_k: 40
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "`top_k` ({self.top_k}) has to be greater than `50`. It is set to `{self.top_k}`."
- id: transformers_top_p_gt_1
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.validate flags `top_p` > 1.0
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: validate
    line_at_scan: 1238
  match:
    engine: transformers
    fields:
      transformers.sampling.top_p: {'>': 1.0}
  kwargs_positive:
    top_p: 2.0
  kwargs_negative:
    top_p: 0.5
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "`top_p` ({self.top_p}) has to be greater than `1.0`. It is set to `{self.top_p}`."
- id: transformers_min_p_gt_0
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.validate flags `min_p` > 0.0
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: validate
    line_at_scan: 1239
  match:
    engine: transformers
    fields:
      transformers.sampling.min_p: {'>': 0.0}
  kwargs_positive:
    min_p: 1.0
  kwargs_negative:
    min_p: -1.0
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "`min_p` ({self.min_p}) has to be greater than `0.0`. It is set to `{self.min_p}`."
- id: transformers_typical_p_gt_1
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.validate flags `typical_p` > 1.0
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: validate
    line_at_scan: 1240
  match:
    engine: transformers
    fields:
      transformers.sampling.typical_p: {'>': 1.0}
  kwargs_positive:
    typical_p: 2.0
  kwargs_negative:
    typical_p: 0.5
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "`typical_p` ({self.typical_p}) has to be greater than `1.0`. It is set to `{self.typical_p}`."
- id: transformers_epsilon_cutoff_gt_0
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.validate flags `epsilon_cutoff` > 0.0
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: validate
    line_at_scan: 1241
  match:
    engine: transformers
    fields:
      transformers.sampling.epsilon_cutoff: {'>': 0.0}
  kwargs_positive:
    epsilon_cutoff: 1.0
  kwargs_negative:
    epsilon_cutoff: -1.0
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "`epsilon_cutoff` ({self.epsilon_cutoff}) has to be greater than `0.0`. It is set to `{self.epsilon_cutoff}`."
- id: transformers_eta_cutoff_gt_0
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.validate flags `eta_cutoff` > 0.0
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: validate
    line_at_scan: 1242
  match:
    engine: transformers
    fields:
      transformers.sampling.eta_cutoff: {'>': 0.0}
  kwargs_positive:
    eta_cutoff: 1.0
  kwargs_negative:
    eta_cutoff: -1.0
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "`eta_cutoff` ({self.eta_cutoff}) has to be greater than `0.0`. It is set to `{self.eta_cutoff}`."
- id: transformers_repetition_penalty_gt_1
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.validate flags `repetition_penalty` > 1.0
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: validate
    line_at_scan: 1243
  match:
    engine: transformers
    fields:
      transformers.sampling.repetition_penalty: {'>': 1.0}
  kwargs_positive:
    repetition_penalty: 2.0
  kwargs_negative:
    repetition_penalty: 0.5
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "`repetition_penalty` ({self.repetition_penalty}) has to be greater than `1.0`. It is set to `{self.repetition_penalty}`."
- id: transformers_encoder_repetition_penalty_gt_1
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.validate flags `encoder_repetition_penalty` > 1.0
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: validate
    line_at_scan: 1244
  match:
    engine: transformers
    fields:
      transformers.sampling.encoder_repetition_penalty: {'>'
...<truncated>...
```
