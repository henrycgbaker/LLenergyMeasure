# invariants_pass1 extraction transcript: base_llm_args_validators_bottom

- chunk_description: tensorrt_llm.BaseLlmArgs remaining @model_validators
- expected_namespaces: ['tensorrt_llm']
- attempts: 1
- elapsed_sec: 134.03
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
Continuation of BaseLlmArgs validators (second half). Same rules as the top half. Use namespace `tensorrt_llm`.

=== SOURCE: BaseLlmArgs validators (bottom half) ===

    @model_validator(mode="after")
    def validate_runtime_args(self):
        if self.max_batch_size is not None and self.max_num_tokens is not None:
            if self.max_batch_size > self.max_num_tokens:
                logger.warning(
                    f"max_batch_size [{self.max_batch_size}] should be less than or equal to max_num_tokens [{self.max_num_tokens}]"
                )
        return self

    @model_validator(mode="after")
    def validate_lora_config_consistency(self):
        if self.lora_config:
            if len(self.lora_config.lora_dir) == 0:
                # TODO [TRTLLM-5173]
                logger.warning(
                    "lora_dir is empty, so custom embedding or lm head will not be applied."
                )

        if self.enable_lora and self.lora_config is not None and self.backend in [
                'pytorch', '_autodeploy'
        ]:
            logger.warning(
                f"enable_lora is ignored when lora_config is provided for {self.backend} backend."
            )

        if self.lora_config is not None:
            if len(self.lora_config.lora_dir) == 0 and len(
                    self.lora_config.lor
...<truncated>...
```

### Response

```
invariants:
- id: tensorrt_max_batch_size_gt_max_num_tokens
  engine: tensorrt
  library: tensorrt
  invariant_under_test: BaseLlmArgs.validate flags max_batch_size > max_num_tokens
  severity: warning
  native_type: tensorrt.BaseLlmArgs
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: validate_runtime_args
    line_at_scan: 5
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.max_batch_size: {'>': 1}
  kwargs_positive:
    max_batch_size: 10
    max_num_tokens: 5
  kwargs_negative:
    max_batch_size: 5
    max_num_tokens: 10
  expected_outcome:
    outcome: warning
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: "max_batch_size [{self.max_batch_size}] should be less than or equal to max_num_tokens [{self.max_num_tokens}]"
  added_by: llm_miner
  added_at: '2026-05-25'

- id: tensorrt_lora_dir_empty
  engine: tensorrt
  library: tensorrt
  invariant_under_test: BaseLlmArgs.validate flags lora_dir empty
  severity: warning
  native_type: tensorrt.BaseLlmArgs
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: validate_lora_config_consistency
    line_at_scan: 10
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.lora_dir: {present: true, not_equal: ""}
  kwargs_positive:
    lora_dir: ""
  kwargs_negative:
    lora_dir: "/path/to/dir"
  expected_outcome:
    outcome: warning
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: "lora_dir is empty, so custom embedding or lm head will not be applied."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: tensorrt_enable_lora_ignored_with_lora_config
  engine: tensorrt
  library: tensorrt
  invariant_under_test: BaseLlmArgs.validate flags enable_lora ignored with lora_config
  severity: warning
  native_type: tensorrt.BaseLlmArgs
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: validate_lora_config_consistency
    line_at_scan: 15
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.enable_lora: true
      tensorrt_llm.lora_config: {present: true}
      tensorrt_llm.backend: {not_in: ['pytorch', '_autodeploy']}
  kwargs_positive:
    enable_lora: true
    lora_config: {}
    backend: 'pytorch'
  kwargs_negative:
    enable_lora: false
    lora_config: {}
    backend: 'pytorch'
  expected_outcome:
    outcome: warning
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: "enable_lora is ignored when lora_config is provided for {self.backend} backend."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: tensorrt_both_lora_dir_and_lora_target_modules_empty
  engine: tensorrt
  library: tensorrt
  invariant_under_test: BaseLlmArgs.validate flags both lora_dir and lora_target_modules empty
  severity: warning
  native_type: tensorrt.BaseLlmArgs
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: validate_lora_config_consistency
    line_at_scan: 20
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.lora_dir: {present: true, not_equal: ""}
      tensorrt_llm.lora_target_modules: {present: true, not_equal: []}
  kwargs_positive:
    lora_dir: ""
    lora_target_modules: []
  kwargs_negative:
    lora_dir: "/path/to/dir"
    lora_target_modules: ["module1", "module2"]
  expected_outcome:
    outcome: warning
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: "Both lora_dir and lora_target_modules are empty, so all LoRA modules will be expected. This will lead to serious memory consumption. Please provide either lora_dir or lora_target_modules if this behavior is not what you expect."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: tensorrt_lora_prefetch_dir_set_while_not_supported
  engine: tensorrt
  library: tensorrt
  invariant_under_test: BaseLlmArgs.validate flags lora_prefetch_dir set while not supported
  severity: error
  native_type: tensorrt.BaseLlmArgs
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: validate_peft_cache_config
    line_at_scan: 25
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.peft_cache_config.lora_prefetch_dir: {present: true, not_equal: ""}
  kwargs_positive:
    peft_cache_config: {"lora_prefetch_dir": "/path/to/dir"}
  kwargs_negative:
    peft_cache_config: {}
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "lora_prefetch_dir was set to '{self.peft_cache_config.lora_prefetch_dir}' while LoRA prefetch is not supported"
  added_by: llm_miner
  added_at: '2026-05-25'
```
