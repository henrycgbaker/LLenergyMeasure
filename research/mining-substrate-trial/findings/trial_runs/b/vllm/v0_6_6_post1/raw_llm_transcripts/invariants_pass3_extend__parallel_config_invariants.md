# invariants_pass3_extend extraction transcript: parallel_config_invariants

- chunk_description: vllm.ParallelConfig __post_init__ + _verify_args
- expected_namespaces: ['vllm']
- attempts: 1
- elapsed_sec: 49.72
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser doing PASS 3 (extension) of multi-pass
invariant extraction. PASS 1 already extracted N invariants from
vllm v0.6.6.post1 for ONE chunk of source. Your job is to
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
  invariant_under_test: ParallelConfig._verify_args flags distributed_executor_backend
    not in allowlist
- id: vllm_ray_workers_use_nsight_when_not_use_ray
  severity: error
  match:
    engine: vllm
    fields:
      vllm.parallel.ray_workers_use_nsight:
        present: true
      vllm.parallel.use_ray: false
  invariant_under_test: ParallelConfig._verify_args flags ray_workers_use_nsight when
    not use_ray
- id: vllm_worker_use_ray_with_distributed_executor_backend_not_ray
  severity: error
  match:
    engine: vllm
    fields:
      vllm.parallel.worker_use_ray: true
      vllm.parallel.distributed_executor_backend:
        present: true
        not_equal: ray
  invariant_under_test: ParallelConfig.__post_init__ flags worker-use-ray with distributed
    executor backend not ray
- id: vllm_distributed_executor_backend_not_ray_on_tpu_hpu
  severity: error
  match:
    engine: vllm
    fields:
      vllm.parallel.distributed_executor_backend:
        present: true
        not_equal: ray
      current_platform.device_type:
        present: true
        in:
        - tpu
        - hpu
  invariant_under_test: ParallelConfig.__post_init__ flags distributed executor backend
    not ray on tpu/hpu
- id: vllm_ray_not_available_for_multi_node_inference
  severity: error
  match:
    engine: vllm
    fields:
      current_platform.is_cuda(): true
      cuda_device_count_stateless():
        <: world_size
  invariant_under_test: ParallelConfig.__post_init__ flags ray not available for multi-node
    inference


INPUT 2 - PASS 2 FLAGGED ISSUES (for context; consider whether they hint
at related misses):

(no flags)

INPUT 3 - THE SOURCE PASS 1 READ:

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

            from vllm.executor import ray_utils
            backend = "mp"
            ray_found = ray_utils.ray_is_available()
            if (current_platform.is_cuda()
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

        self._verify_args()


=== SOURCE: ParallelConfig._verify_args ===
    def _verify_args(self) -> None:
        # Lazy import to avoid circular import
        from vllm.executor.executor_base import ExecutorBase

        if self.distributed_executor_backend not in (
                "ray", "mp", None) and not (isinstance(
                    self.distributed_executor_backend, type) and issubclass(
                        self.distributed_executor_backend, ExecutorBase)):
            raise ValueError(
                "Unrecognized distributed executor backend "
                f"{self.distributed_executor_backend}. Supported "
                "values are 'ray', 'mp' or custom ExecutorBase subclass.")
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
      typical_p
...<truncated>...
```

### Response

```
invariants:
- id: vllm_distributed_executor_backend_not_none_when_world_size_greater_than_one
  engine: vllm
  library: vllm
  invariant_under_test: ParallelConfig.__post_init__ checks distributed executor backend is not None when world size greater than one
  severity: error
  native_type: vllm.ParallelConfig
  miner_source:
    path: ParallelConfig.py
    method: __post_init__
    line_at_scan: 10
  match:
    engine: vllm
    fields:
      vllm.parallel.distributed_executor_backend:
        present: true
        not_equal: null
      vllm.parallel.world_size:
        '>': 1
  kwargs_positive:
    distributed_executor_backend: null
    world_size: 2
  kwargs_negative:
    distributed_executor_backend: ray
    world_size: 1
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: 'Unable to load Ray which is required for multi-node inference, please install Ray with `pip install ray`.'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
- id: vllm_ray_workers_use_nsight_with_use_ray_false_type_error
  engine: vllm
  library: vllm
  invariant_under_test: ParallelConfig._verify_args checks ray workers use nsight with use ray false type error
  severity: error
  native_type: vllm.ParallelConfig
  miner_source:
    path: ParallelConfig.py
    method: _verify_args
    line_at_scan: 20
  match:
    engine: vllm
    fields:
      vllm.parallel.ray_workers_use_nsight:
        present: true
      vllm.parallel.use_ray:
        exact: false
  kwargs_positive:
    ray_workers_use_nsight: True
    use_ray: False
  kwargs_negative:
    ray_workers_use_nsight: False
    use_ray: True
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: 'Unable to use nsight profiling unless workers run with Ray.'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
- id: vllm_distributed_executor_backend_type_error
  engine: vllm
  library: vllm
  invariant_under_test: ParallelConfig._verify_args checks distributed executor backend type error
  severity: error
  native_type: vllm.ParallelConfig
  miner_source:
    path: ParallelConfig.py
    method: _verify_args
    line_at_scan: 15
  match:
    engine: vllm
    fields:
      vllm.parallel.distributed_executor_backend:
        present: true
        type_is_not: [str, ExecutorBase]
  kwargs_positive:
    distributed_executor_backend: 123
  kwargs_negative:
    distributed_executor_backend: ray
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: 'Unrecognized distributed executor backend {self.distributed_executor_backend}. Supported values are \'ray\', \'mp\' or custom ExecutorBase subclass.'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
```
