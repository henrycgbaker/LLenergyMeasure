# invariants extraction transcript: parallel_config_invariants

- chunk_description: vllm.ParallelConfig __post_init__ + _verify_args
- expected_namespaces: ['vllm']
- attempts: 1
- elapsed_sec: 32.79
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser extracting validation invariants from
vllm library v0.7.3. An "invariant" is a rule the
library checks at runtime - typically `if <predicate>: raise ValueError(...)`
or `if <predicate>: minor_issues[...] = ...` (which surfaces as a
warning).

You will be shown ONE CHUNK of validation source. Extract every
invariant visible in this chunk.

OUTPUT FORMAT: YAML document matching EXACTLY this shape:

invariants:
- id: <snake_case_unique_id>
  engine: vllm
  library: vllm
  invariant_under_test: <one-line: what the library checks here>
  severity: <error|dormant|warning>
  native_type: vllm.GenerationConfig
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: <validate|__init__>
    line_at_scan: <approximate line number if visible>
  match:
    engine: vllm
    fields:
      vllm.<field>: <value or predicate>
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
- Exact value:         `vllm.field: value`
- Not in list:         `vllm.field: {present: true, not_in: [a, b]}`
- Not equal:           `vllm.field: {present: true, not_equal: value}`
- Greater than:        `vllm.field: {'>': value}`
- Less than:           `vllm.field: {'<': value}`
- Greater or equal:    `vllm.field: {'>=': value}`
- Less or equal:       `vllm.field: {'<=': value}`
- Type not in:         `vllm.field: {present: true, type_is_not: [TypeName]}`
- Presence (any value):`vllm.field: {present: true}`

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

=== E6 FIELD ANCHOR ===
The engine's declared model fields are listed below. For EACH field,
look in the source for a validation pattern (`if X.field <pred>: raise`
or `minor_issues[...] = ...`) and emit an invariant if you find one.

DOMAIN CONSTRAINT: ONLY propose invariants on fields that appear in
this list. Do NOT propose invariants on fields you cannot find in the
declared __fields__ below - those are out of scope (extra=allow
fields are not in scope for this prompt).

--- ModelConfig.__fields__ ---
allowed_local_media_path, code_revision, config_format, disable_mm_preprocessor_cache, disable_sliding_window, dtype, enable_sleep_mode, enforce_eager, generation_config, hf_overrides, limit_mm_per_prompt, logits_processor_pattern, max_logprobs, max_model_len, max_seq_len_to_capture, mm_processor_kwargs, model, model_impl, override_generation_config, override_neuron_config, override_pooler_config, quantization, revision, rope_scaling, rope_theta, seed, served_model_name, skip_tokenizer_init, spec_target_max_model_len, task, tokenizer, tokenizer_mode, tokenizer_revision, trust_remote_code, use_async_output_proc

--- CacheConfig.__fields__ ---
block_size, cache_dtype, calculate_kv_scales, cpu_offload_gb, enable_prefix_caching, gpu_memory_utilization, is_attention_free, num_gpu_blocks_override, sliding_window, swap_space

--- ParallelConfig.__fields__ ---
disable_custom_all_reduce, distributed_executor_backend, max_parallel_loading_worker
...<truncated>...
```

### Response

```
invariants:
- id: vllm_distributed_executor_backend_not_in_allowlist
  engine: vllm
  invariant_under_test: ParallelConfig._verify_args flags distributed_executor_backend not in allowlist
  severity: error
  native_type: vllm.ParallelConfig
  miner_source:
    path: vllm/parallel_config.py
    method: _verify_args
    line_at_scan: 10
  match:
    engine: vllm
    fields:
      vllm.parallel.distributed_executor_backend: {present: true, not_in: ["ray", "mp", "uni", "external_launcher"]}
  kwargs_positive:
    distributed_executor_backend: "invalid"
  kwargs_negative:
    distributed_executor_backend: "ray"
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "Unrecognized distributed executor backend {self.distributed_executor_backend}. Supported values are 'ray', 'mp' 'uni', 'external_launcher' or custom ExecutorBase subclass."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: vllm_ray_workers_use_nsight_without_ray
  engine: vllm
  invariant_under_test: ParallelConfig._verify_args flags ray_workers_use_nsight without use_ray
  severity: error
  native_type: vllm.ParallelConfig
  miner_source:
    path: vllm/parallel_config.py
    method: _verify_args
    line_at_scan: 20
  match:
    engine: vllm
    fields:
      vllm.parallel.ray_workers_use_nsight: {present: true}
      vllm.parallel.use_ray: false
  kwargs_positive:
    ray_workers_use_nsight: true
    use_ray: false
  kwargs_negative:
    ray_workers_use_nsight: true
    use_ray: true
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "Unable to use nsight profiling unless workers run with Ray."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: vllm_rocm_disable_custom_all_reduce
  engine: vllm
  invariant_under_test: ParallelConfig._verify_args flags disable_custom_all_reduce on ROCm
  severity: dormant
  native_type: vllm.ParallelConfig
  miner_source:
    path: vllm/parallel_config.py
    method: _verify_args
    line_at_scan: 15
  match:
    engine: vllm
    fields:
      vllm.platforms.current_platform.is_rocm: true
  kwargs_positive:
    disable_custom_all_reduce: false
  kwargs_negative:
    disable_custom_all_reduce: true
  expected_outcome:
    outcome: dormant_announced
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: "Disabled the custom all-reduce kernel because it is not supported on AMD GPUs."
  added_by: llm_miner
  added_at: '2026-05-25'
```
