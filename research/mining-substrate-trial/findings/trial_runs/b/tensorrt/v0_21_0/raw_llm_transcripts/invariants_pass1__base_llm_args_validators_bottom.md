# invariants_pass1 extraction transcript: base_llm_args_validators_bottom

- chunk_description: tensorrt_llm.BaseLlmArgs remaining @model_validators
- expected_namespaces: ['tensorrt_llm']
- attempts: 1
- elapsed_sec: 77.19
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser extracting validation invariants from
tensorrt library v0.21.0. An "invariant" is a rule the
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
Continuation of BaseLlmArgs validators (second half). Same rules as the top half. Use namespace `tensorrt_llm`.

=== SOURCE: BaseLlmArgs validators (bottom half) ===

    @model_validator(mode="after")
    def validate_speculative_config(self):
        if self.speculative_config:
            if isinstance(self.speculative_config, LookaheadDecodingConfig):
                lookahead_config = self.speculative_config
                # Update the build config
                _, _, max_draft_tokens, _ = lookahead_config.calculate_speculative_resource(
                )
                self.build_config.speculative_decoding_mode = SpeculativeDecodingMode.LOOKAHEAD_DECODING
                if max_draft_tokens > self.build_config.max_draft_len:
                    self.build_config.max_draft_len = max_draft_tokens

                self.decoding_config = DecodingConfig(
                    decoding_mode=DecodingMode.Lookahead(),
                    lookahead_decoding_config=PybindMirror.maybe_to_pybind(
                        lookahead_config))
            elif isinstance(self.speculative_config, MedusaDecodingConfig):
                self.build_config.speculative_decoding_mode = SpeculativeDecodingMode.MEDUSA

                assert self.speculative_config.max_draft_len > 0
                self.build_config.max_draft_len = self.s
...<truncated>...
```

### Response

```
invariants:
- id: tensorrt_llm_speculative_config_type_not_recognized
  engine: tensorrt
  library: tensorrt
  invariant_under_test: BaseLlmArgs.validate flags speculative config type not recognized
  severity: error
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: tensorrt_llm/configuration_utils.py
    method: validate_speculative_config
    line_at_scan: 10
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.speculative_config: {present: true, type_is_not: [LookaheadDecodingConfig, MedusaDecodingConfig, EagleDecodingConfig, NGramDecodingConfig, DraftTargetDecodingConfig, MTPDecodingConfig]}
  kwargs_positive:
    speculative_config: "invalid"
  kwargs_negative:
    speculative_config: LookaheadDecodingConfig()
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "Speculative config type not recognized: {self.speculative_config}"
  added_by: llm_miner
  added_at: '2026-05-25'

- id: tensorrt_llm_max_lora_rank_ignored_when_lora_config_provided
  engine: tensorrt
  library: tensorrt
  invariant_under_test: BaseLlmArgs.validate flags max_lora_rank ignored when lora_config is provided
  severity: warning
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: tensorrt_llm/configuration_utils.py
    method: validate_lora_config_consistency
    line_at_scan: 20
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.lora_config: {present: true}
      tensorrt_llm.max_lora_rank: {present: true, not_equal: None}
  kwargs_positive:
    lora_config: {}
    max_lora_rank: 10
  kwargs_negative:
    lora_config: None
    max_lora_rank: 10
  expected_outcome:
    outcome: warning
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: "max_lora_rank is ignored when lora_config is provided."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: tensorrt_llm_max_loras_ignored_when_lora_config_provided
  engine: tensorrt
  library: tensorrt
  invariant_under_test: BaseLlmArgs.validate flags max_loras ignored when lora_config is provided
  severity: warning
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: tensorrt_llm/configuration_utils.py
    method: validate_lora_config_consistency
    line_at_scan: 25
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.lora_config: {present: true}
      tensorrt_llm.max_loras: {present: true, not_equal: None}
  kwargs_positive:
    lora_config: {}
    max_loras: 10
  kwargs_negative:
    lora_config: None
    max_loras: 10
  expected_outcome:
    outcome: warning
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: "max_loras is ignored when lora_config is provided."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: tensorrt_llm_max_cpu_loras_ignored_when_lora_config_provided
  engine: tensorrt
  library: tensorrt
  invariant_under_test: BaseLlmArgs.validate flags max_cpu_loras ignored when lora_config is provided
  severity: warning
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: tensorrt_llm/configuration_utils.py
    method: validate_lora_config_consistency
    line_at_scan: 30
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.lora_config: {present: true}
      tensorrt_llm.max_cpu_loras: {present: true, not_equal: None}
  kwargs_positive:
    lora_config: {}
    max_cpu_loras: 10
  kwargs_negative:
    lora_config: None
    max_cpu_loras: 10
  expected_outcome:
    outcome: warning
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: "max_cpu_loras is ignored when lora_config is provided."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: tensorrt_llm_enable_lora_ignored_when_lora_config_provided_for_pytorch_backend
  engine: tensorrt
  library: tensorrt
  invariant_under_test: BaseLlmArgs.validate flags enable_lora ignored when lora_config is provided for pytorch backend
  severity: warning
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: tensorrt_llm/configuration_utils.py
    method: validate_lora_config_consistency
    line_at_scan: 40
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.lora_config: {present: true}
      tensorrt_llm.enable_lora: {present: true, not_equal: None}
      tensorrt_llm.backend: pytorch
  kwargs_positive:
    lora_config: {}
    enable_lora: True
    backend: pytorch
  kwargs_negative:
    lora_config: None
    enable_lora: True
    backend: pytorch
  expected_outcome:
    outcome: warning
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: "enable_lora is ignored when lora_config is provided for {self.backend} backend."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: tensorrt_llm_both_lora_dir_and_lora_target_modules_empty
  engine: tensorrt
  library: tensorrt
  invariant_under_test: BaseLlmArgs.validate flags both lora_dir and lora_target_modules empty
  severity: warning
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: tensorrt_llm/configuration_utils.py
    method: validate_lora_config_consistency
    line_at_scan: 50
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.lora_config: {present: true}
      tensorrt_llm.lora_dir: []
      tensorrt_llm.lora_target_modules: []
  kwargs_positive:
    lora_config: {}
    lora_dir: []
    lora_target_modules: []
  kwargs_negative:
    lora_config: None
    lora_dir: ["dir"]
    lora_target_modules: ["module"]
  expected_outcome:
    outcome: warning
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: "Both lora_dir and lora_target_modules are empty, so all LoRA modules will be expected. This will lead to serious memory consumption. Please provide either lora_dir or lora_target_modules if this behavior is not what you expect."
  added_by: llm_miner
  added_at: '2026-05-25'
```
