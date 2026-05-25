# invariants_pass1 extraction transcript: parallel_config_invariants

- chunk_description: vllm.ParallelConfig __post_init__ + _verify_args
- expected_namespaces: ['vllm']
- attempts: 1
- elapsed_sec: 151.80
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser extracting validation invariants from
vllm library v0.6.6.post1. An "invariant" is a rule the
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

=== CONTEXT ===
vllm.ParallelConfig validates `distributed_executor_backend` enum and `pipeline_parallel_size`/`tensor_parallel_size`.

=== SOURCE: ParallelConfig.__post_init__ ===
    def __post_init__(self) -> None:
        self.world_size = self.pipeline_parallel_size * \
            self.tensor_parallel_size

        if self.worker_use_ray:
            if self.distributed_executor_backend is None:
                self.distributed_executor_backend = "ray"
            elif not self.use_ray:
                raise ValueError(f"worker-use-ray can't be used with "
                                 f"distributed executor backend "
                                 f"'{self.distributed_executor_backend}'.")
        ray_only_devices = ["tpu", "hpu"]
        if (current_platform.device_type in ray_only_devices
                and self.world_size > 1):
            if self.distributed_executor_backend is None:
                self.distributed_executor_backend = "ray"
            if self.distributed_executor_backend != "ray":
                raise ValueError(
                    f"{current_platform.device_type.upper()} backend only "
                    "supports Ray for distributed inference.")

        if self.distributed_executor_backend is None and self.world_size > 1:
            # We use multiprocessing by default if world_size fits on the
            # current node and we aren't in a ray placement group.

            from vllm.executo
...<truncated>...
```

### Response

```
invariants:
- id: vllm_distributed_executor_backend_not_in_allowlist
  engine: vllm
  library: vllm
  invariant_under_test: ParallelConfig._verify_args flags distributed_executor_backend not in allowlist
  severity: error
  native_type: vllm.ParallelConfig
  miner_source:
    path: configuration_utils.py
    method: _verify_args
    line_at_scan: 10
  match:
    engine: vllm
    fields:
      vllm.parallel.distributed_executor_backend: {present: true, not_in: ["ray", "mp"]}
  kwargs_positive:
    distributed_executor_backend: "invalid"
  kwargs_negative:
    distributed_executor_backend: "ray"
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "Unrecognized distributed executor backend {self.distributed_executor_backend}. Supported values are 'ray', 'mp' or custom ExecutorBase subclass."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: vllm_ray_workers_use_nsight_when_not_use_ray
  engine: vllm
  library: vllm
  invariant_under_test: ParallelConfig._verify_args flags ray_workers_use_nsight when not use_ray
  severity: error
  native_type: vllm.ParallelConfig
  miner_source:
    path: configuration_utils.py
    method: _verify_args
    line_at_scan: 15
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

- id: vllm_worker_use_ray_with_distributed_executor_backend_not_ray
  engine: vllm
  library: vllm
  invariant_under_test: ParallelConfig.__post_init__ flags worker-use-ray with distributed executor backend not ray
  severity: error
  native_type: vllm.ParallelConfig
  miner_source:
    path: configuration_utils.py
    method: __post_init__
    line_at_scan: 5
  match:
    engine: vllm
    fields:
      vllm.parallel.worker_use_ray: true
      vllm.parallel.distributed_executor_backend: {present: true, not_equal: "ray"}
  kwargs_positive:
    worker_use_ray: true
    distributed_executor_backend: "mp"
  kwargs_negative:
    worker_use_ray: false
    distributed_executor_backend: "ray"
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "worker-use-ray can't be used with distributed executor backend '{self.distributed_executor_backend}'."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: vllm_distributed_executor_backend_not_ray_on_tpu_hpu
  engine: vllm
  library: vllm
  invariant_under_test: ParallelConfig.__post_init__ flags distributed executor backend not ray on tpu/hpu
  severity: error
  native_type: vllm.ParallelConfig
  miner_source:
    path: configuration_utils.py
    method: __post_init__
    line_at_scan: 10
  match:
    engine: vllm
    fields:
      vllm.parallel.distributed_executor_backend: {present: true, not_equal: "ray"}
      current_platform.device_type: {present: true, in: ["tpu", "hpu"]}
  kwargs_positive:
    distributed_executor_backend: "mp"
    world_size: 2
  kwargs_negative:
    distributed_executor_backend: "ray"
    world_size: 1
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "{current_platform.device_type.upper()} backend only supports Ray for distributed inference."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: vllm_ray_not_available_for_multi_node_inference
  engine: vllm
  library: vllm
  invariant_under_test: ParallelConfig.__post_init__ flags ray not available for multi-node inference
  severity: error
  native_type: vllm.ParallelConfig
  miner_source:
    path: configuration_utils.py
    method: __post_init__
    line_at_scan: 20
  match:
    engine: vllm
    fields:
      current_platform.is_cuda(): true
      cuda_device_count_stateless(): {'<': world_size}
  kwargs_positive:
    world_size: 2
  kwargs_negative:
    world_size: 1
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "Unable to load Ray which is required for multi-node inference, please install Ray with `pip install ray`."
  added_by: llm_miner
  added_at: '2026-05-25'
```
