# invariants_pass1 extraction transcript: trt_llm_args_validators

- chunk_description: tensorrt_llm.TrtLlmArgs @field_validator + @model_validator methods
- expected_namespaces: ['tensorrt_llm']
- attempts: 1
- elapsed_sec: 119.51
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser extracting validation invariants from
tensorrt library v1.2.1. An "invariant" is a rule the
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
TrtLlmArgs has 1 @field_validator (calib_config init) + 3 @model_validator decorators. The validate_enable_build_cache method has the most pertinent `raise ValueError(...)` block.

=== SOURCE: TrtLlmArgs validators ===
    @model_validator(mode="after")
    def init_build_config(self):
        """
        Creating a default BuildConfig if none is provided
        """
        build_config = getattr(self, "build_config", None)
        if build_config is None:
            kwargs = {}
            if self.max_batch_size:
                kwargs["max_batch_size"] = self.max_batch_size
            if self.max_num_tokens:
                kwargs["max_num_tokens"] = self.max_num_tokens
            if self.max_seq_len:
                kwargs["max_seq_len"] = self.max_seq_len
            if self.max_beam_width:
                kwargs["max_beam_width"] = self.max_beam_width
            if self.max_input_len:
                kwargs["max_input_len"] = self.max_input_len
            self.build_config = BuildConfig(**kwargs)
        return self

    @model_validator(mode="after")
    def validate_build_config_with_runtime_params(self):
        # Note: max_batch_size and max_num_tokens in LlmArgs are for runtime,
        # which will be passed to the C++ Executor API, overwriting the values
        # from an built engine. In order to set bui
...<truncated>...
```

### Response

```
invariants:
- id: tensorrt_speculative_config_type_not_supported_by_backend
  engine: tensorrt
  library: tensorrt
  invariant_under_test: TrtLlmArgs.validate flags speculative config type does not support backend
  severity: error
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: tensorrt/validation.py
    method: validate_speculative_config
    line_at_scan: 1
  match:
    engine: tensorrt
    fields:
      tensorrt.speculative_config.decoding_type: {present: true}
      tensorrt.backend: {present: true, not_equal: "pytorch"}
  kwargs_positive:
    speculative_config: LookaheadDecodingConfig()
    backend: "_autodeploy"
  kwargs_negative:
    speculative_config: None
    backend: "pytorch"
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "Speculation type {self.speculative_config.decoding_type} does not support backend {self.backend}"
  added_by: llm_miner
  added_at: '2026-05-25'

- id: tensorrt_speculative_config_unrecognized_type
  engine: tensorrt
  library: tensorrt
  invariant_under_test: TrtLlmArgs.validate flags unrecognized speculative config type
  severity: error
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: tensorrt/validation.py
    method: validate_speculative_config
    line_at_scan: 1
  match:
    engine: tensorrt
    fields:
      tensorrt.speculative_config.decoding_type: {present: true}
  kwargs_positive:
    speculative_config: "InvalidConfig"
  kwargs_negative:
    speculative_config: LookaheadDecodingConfig()
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "Unrecognized speculative config type {type(self.speculative_config)}"
  added_by: llm_miner
  added_at: '2026-05-25'

- id: tensorrt_max_draft_len_not_set_for_speculative_decoding
  engine: tensorrt
  library: tensorrt
  invariant_under_test: TrtLlmArgs.validate flags max draft len not set for speculative decoding
  severity: error
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: tensorrt/validation.py
    method: validate_speculative_config
    line_at_scan: 1
  match:
    engine: tensorrt
    fields:
      tensorrt.speculative_config.decoding_type: {present: true}
  kwargs_positive:
    speculative_config: MedusaDecodingConfig(max_draft_len=0)
  kwargs_negative:
    speculative_config: MedusaDecodingConfig(max_draft_len=10)
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "max_draft_len must be greater than 0"
  added_by: llm_miner
  added_at: '2026-05-25'

- id: tensorrt_eagle3_draft_model_not_specified
  engine: tensorrt
  library: tensorrt
  invariant_under_test: TrtLlmArgs.validate flags eagle3 draft model not specified
  severity: error
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: tensorrt/validation.py
    method: validate_speculative_config
    line_at_scan: 1
  match:
    engine: tensorrt
    fields:
      tensorrt.speculative_config.decoding_type: {present: true}
  kwargs_positive:
    speculative_config: EagleDecodingConfig(speculative_model=None)
  kwargs_negative:
    speculative_config: EagleDecodingConfig(speculative_model="model")
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "EAGLE3 draft model must be specified."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: tensorrt_max_batch_size_overridden_by_build_config
  engine: tensorrt
  library: tensorrt
  invariant_under_test: TrtLlmArgs.validate flags max batch size overridden by build config
  severity: warning
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: tensorrt/validation.py
    method: validate_build_config_with_runtime_params
    line_at_scan: 1
  match:
    engine: tensorrt
    fields:
      tensorrt.max_batch_size: {present: true}
      tensorrt.build_config.max_batch_size: {present: true, '<': tensorrt.max_batch_size}
  kwargs_positive:
    max_batch_size: 10
    build_config: BuildConfig(max_batch_size=5)
  kwargs_negative:
    max_batch_size: 5
    build_config: BuildConfig(max_batch_size=10)
  expected_outcome:
    outcome: warning
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: "max_batch_size [{self.max_batch_size}] is overridden by build_config.max_batch_size [{self.build_config.max_batch_size}] in build_config"
  added_by: llm_miner
  added_at: '2026-05-25'

- id: tensorrt_max_num_tokens_overridden_by_build_config
  engine: tensorrt
  library: tensorrt
  invariant_under_test: TrtLlmArgs.validate flags max num tokens overridden by build config
  severity: warning
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: tensorrt/validation.py
    method: validate_build_config_with_runtime_params
    line_at_scan: 1
  match:
    engine: tensorrt
    fields:
      tensorrt.max_num_tokens: {present: true}
      tensorrt.build_config.max_num_tokens: {present: true, '<': tensorrt.max_num_tokens}
  kwargs_positive:
    max_num_tokens: 10
    build_config: BuildConfig(max_num_tokens=5)
  kwargs_negative:
    max_num_tokens: 5
    build_config: BuildConfig(max_num_tokens=10)
  expected_outcome:
    outcome: warning
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: "max_num_tokens [{self.max_num_tokens}] is overridden by build_config.max_num_tokens [{self.build_config.max_num_tokens}] in build_config"
  added_by: llm_miner
  added_at: '2026-05-25'

- id: tensorrt_max_seq_len_overridden_by_build_config
  engine: tensorrt
  library: tensorrt
  invariant_under_test: TrtLlmArgs.validate flags max seq len overridden by build config
  severity: warning
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: tensorrt/validation.py
    method: validate_build_config_with_runtime_params
    line_at_scan: 1
  match:
    engine: tensorrt
    fields:
      tensorrt.max_seq_len: {present: true}
      tensorrt.build_config.max_seq_len: {present: true, '!=': tensorrt.max_seq_len}
  kwargs_positive:
    max_seq_len: 10
    build_config: BuildConfig(max_seq_len=5)
  kwargs_negative:
    max_seq_len: 5
    build_config: BuildConfig(max_seq_len=10)
  expected_outcome:
    outcome: warning
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: "max_seq_len [{self.max_seq_len}] is overridden by build_config.max_seq_len [{self.build_config.max_seq_len}] in build_config"
  added_by: llm_miner
  added_at: '2026-05-25'

- id: tensorrt_max_beam_width_overridden_by_build_config
  engine: tensorrt
  library: tensorrt
  invariant_under_test: TrtLlmArgs.validate flags max beam width overridden by build config
  severity: warning
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: tensorrt/validation.py
    method: validate_build_config_with_runtime_params
    line_at_scan: 1
  match:
    engine: tensorrt
    fields:
      tensorrt.max_beam_width: {present: true}
      tensorrt.build_config.max_beam_width: {present: true, '!=': tensorrt.max_beam_width}
  kwargs_positive:
    max_beam_width: 10
    build_config: BuildConfig(max_beam_width=5)
  kwargs_negative:
    max_beam_width: 5
    build_config: BuildConfig(max_beam_width=10)
  expected_outcome:
    outcome: warning
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: "max_beam_width [{self.max_beam_width}] is overridden by build_config.max_beam_width [{self.build_config.max_beam_width}] in build_config"
  added_by: llm_miner
  added_at: '2026-05-25'

- id: tensorrt_max_input_len_overridden_by_build_config
  engine: tensorrt
  library: tensorrt
  invariant_under_test: TrtLlmArgs.validate flags max input len overridden by build config
  severity: warning
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: tensorrt/validation.py
    method: validate_build_config_with_runtime_params
    line_at_scan: 1
  match:
    engine: tensorrt
   
...<truncated>...
```
