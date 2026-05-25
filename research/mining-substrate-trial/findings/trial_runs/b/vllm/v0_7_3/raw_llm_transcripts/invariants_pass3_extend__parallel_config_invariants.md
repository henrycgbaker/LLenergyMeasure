# invariants_pass3_extend extraction transcript: parallel_config_invariants

- chunk_description: vllm.ParallelConfig __post_init__ + _verify_args
- expected_namespaces: ['vllm']
- attempts: 1
- elapsed_sec: 35.26
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser doing PASS 3 (extension) of multi-pass
invariant extraction. PASS 1 already extracted N invariants from
vllm v0.7.3 for ONE chunk of source. Your job is to
identify ADDITIONAL invariants PASS 1 missed.

INPUT 1 - PASS 1 EXTRACTED INVARIANTS (already known; do NOT re-emit):

invariants:
- id: vllm_distributed_executor_backend_not_in_allowlist
  severity: error
  match:
    engine: vllm
    fields:
      vllm.parallel.distributed_executor_backend:
        present: true
        not_in:
        - ray
        - mp
        - uni
        - external_launcher
  invariant_under_test: ParallelConfig._verify_args flags distributed_executor_backend
    not in allowlist
- id: vllm_ray_workers_use_nsight_without_ray
  severity: error
  match:
    engine: vllm
    fields:
      vllm.parallel.ray_workers_use_nsight:
        present: true
      vllm.parallel.use_ray: false
  invariant_under_test: ParallelConfig._verify_args flags ray_workers_use_nsight without
    use_ray
- id: vllm_tpu_backend_not_ray_for_distributed_inference
  severity: error
  match:
    engine: vllm
    fields:
      vllm.parallel.world_size:
        '>': 1
      vllm.platforms.current_platform.device_type:
        present: true
        not_in:
        - tpu
      vllm.parallel.distributed_executor_backend:
        present: true
        not_equal: ray
  invariant_under_test: ParallelConfig.__post_init__ flags tpu backend not ray for
    distributed inference


INPUT 2 - PASS 2 FLAGGED ISSUES (for context; consider whether they hint
at related misses):

flagged:
- id: vllm_ray_not_available_for_multi_node_inference
  reason: Source does not raise an error for multi-node inference with ray, it only
    checks if ray is available.
  fix: drop
- id: vllm_tpu_backend_not_ray_for_distributed_inference
  reason: Source sets distributed_executor_backend to "ray" when current_platform.device_type
    is "tpu" and world_size > 1, but does not raise an error for other backends.
  fix: correct_predicate:not_equal


INPUT 3 - THE SOURCE PASS 1 READ:

=== CONTEXT ===
vllm.ParallelConfig validates `distributed_executor_backend` enum and `pipeline_parallel_size`/`tensor_parallel_size`.

=== SOURCE: ParallelConfig.__post_init__ ===
    def __post_init__(self) -> None:
        self.world_size = self.pipeline_parallel_size * \
            self.tensor_parallel_size

        ray_only_devices = ["tpu"]
        from vllm.platforms import current_platform
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

            from vllm.executor import ray_utils
            backend = "mp"
            ray_found = ray_utils.ray_is_available()
            if current_platform.is_neuron():
                # neuron uses single process to control multiple devices
                backend = "uni"
            elif (current_platform.is_cuda()
                  and cuda_device_count_stateless() < self.world_size):
                if not ray_found:
                    raise ValueError("Unable to load Ray which is "
                                     "required for multi-node inference, "
                                     "please install Ray with `pip install "
                                     "ray`.") from ray_utils.ray_import_err
                backend = "ray"
            elif ray_found:
                if self.placement_group:
                    backend = "ray"
                else:
                    from ray import is_initialized as ray_is_initialized
                    if ray_is_initialized():
                        from ray.util import get_current_placement_group
                        if get_current_placement_group():
                            backend = "ray"
            self.distributed_executor_backend = backend
            logger.info("Defaulting to use %s for distributed inference",
                        backend)

        if self.distributed_executor_backend is None and self.world_size == 1:
            self.distributed_executor_backend = "uni"

        self._verify_args()


=== SOURCE: ParallelConfig._verify_args ===
    def _verify_args(self) -> None:
        # Lazy import to avoid circular import
        from vllm.executor.executor_base import ExecutorBase
        from vllm.platforms import current_platform
        if self.distributed_executor_backend not in (
                "ray", "mp", "uni",
                "external_launcher", None) and not (isinstance(
                    self.distributed_executor_backend, type) and issubclass(
                        self.distributed_executor_backend, ExecutorBase)):
            raise ValueError(
                "Unrecognized distributed executor backend "
                f"{self.distributed_executor_backend}. Supported "
                "values are 'ray', 'mp' 'uni', 'external_launcher' or"
                " custom ExecutorBase subclass.")
        if self.use_ray:
            from vllm.executor import ray_utils
            ray_utils.assert_ray_available()
        if current_platform.is_rocm():
            self.disable_custom_all_reduce = True
            logger.info(
                "Disabled the custom all-reduce kernel because it is not "
                "supported on AMD GPUs.")
        if self.ray_workers_use_nsight and not self.use_ray:
            raise ValueError("Unable to use nsight profiling unless workers "
                             "run with Ray.")





OUTPUT FORMAT: YAML document with the SAME shape as the original
invariants prompt's output:

invariants:
- id: <snake_case_unique_id>
  engine: vllm
  library: vllm
  invariant_under_test: <one-line: what the library checks here>
  severity: <error|dormant|warning>
  native_type: <e.g. vllm.GenerationConfig or vllm.EngineArgs>
  miner_source:
    path: <file path>
    method: <validate|__init__|_verify_args|...>
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
  message_template: '<the exact error/warning string from source>'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

PREDICATE FORMS (use the EXACT keys shown):
- Exact value:         `vllm.field: value`
- Not in list:         `vllm.field: {present: true, not_in: [a, b]}`
- Greater than:        `vllm.field: {'>': value}`
- Less than:           `vllm.field: {'<': value}`
- Type not in:         `vllm.field: {present: true, type_is_not: [TypeName]}`
- Presence:            `vllm.field: {present: true}`

CRITICAL RULES FOR PASS 3:
1. Return ONLY the YAML document. NO markdown code fences. NO commentary.
   First character must be `i` (from `invariants:`).
2. ONLY emit invariants that are NOT already in PASS 1's list (check by
   ID and by (field, predicate_kind) tuple).
3. Look ESPECIALLY for these PASS 1 blind spots:
   a. PER-FIELD invariants where PASS 1 emitted ONE entry covering many
      fields. Example: PASS 1 emitted `transformers_temperature_set_when_do_sample_false`
      but the source has SIMILAR if-blocks for top_p, top_k, min_p,
      typical_p, epsilon_cutoff, eta_cutoff. Emit ONE per field PASS 1
      missed.
   b. Multi-clause `if A
...<truncated>...
```

### Response

```
invariants:
- id: vllm_ray_workers_use_nsight_without_placement_group
  engine: vllm
  library: vllm
  invariant_under_test: ParallelConfig._verify_args flags ray_workers_use_nsight without placement group
  severity: error
  native_type: vllm.ParallelConfig
  miner_source:
    path: ParallelConfig.py
    method: _verify_args
    line_at_scan: 10
  match:
    engine: vllm
    fields:
      vllm.ray_workers_use_nsight:
        present: true
      vllm.placement_group:
        not_equal: True
  kwargs_positive:
    ray_workers_use_nsight: True
    placement_group: False
  kwargs_negative:
    ray_workers_use_nsight: False
    placement_group: True
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: 'Unable to use nsight profiling unless workers run with Ray.'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
- id: vllm_use_ray_without_ray_available
  engine: vllm
  library: vllm
  invariant_under_test: ParallelConfig._verify_args flags use_ray without ray available
  severity: error
  native_type: vllm.ParallelConfig
  miner_source:
    path: ParallelConfig.py
    method: _verify_args
    line_at_scan: 15
  match:
    engine: vllm
    fields:
      vllm.use_ray:
        present: true
      vllm.ray_available:
        not_equal: True
  kwargs_positive:
    use_ray: True
    ray_available: False
  kwargs_negative:
    use_ray: False
    ray_available: True
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: 'Unable to load Ray which is required for multi-node inference, please install Ray with `pip install ray`.'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
- id: vllm_distributed_executor_backend_not_supported_on_rocm
  engine: vllm
  library: vllm
  invariant_under_test: ParallelConfig._verify_args flags distributed_executor_backend not supported on ROCm
  severity: error
  native_type: vllm.ParallelConfig
  miner_source:
    path: ParallelConfig.py
    method: _verify_args
    line_at_scan: 20
  match:
    engine: vllm
    fields:
      vllm.distributed_executor_backend:
        present: true
      vllm.current_platform.device_type:
        equal: rocm
  kwargs_positive:
    distributed_executor_backend: ray
    current_platform.device_type: rocm
  kwargs_negative:
    distributed_executor_backend: mp
    current_platform.device_type: cuda
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: 'Disabled the custom all-reduce kernel because it is not supported on AMD GPUs.'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
```
