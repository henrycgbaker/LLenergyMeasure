# invariants_pass2_verify extraction transcript: parallel_config_invariants

- chunk_description: vllm.ParallelConfig __post_init__ + _verify_args
- expected_namespaces: ['vllm']
- attempts: 1
- elapsed_sec: 8.57
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser doing PASS 2 (verification) of multi-pass
invariant extraction. PASS 1 already extracted invariants from
vllm v0.7.3 for ONE chunk of source. Your job is to
REVIEW each emitted invariant against the source and flag any that
look WRONG.

INPUT 1 - PASS 1 EXTRACTED INVARIANTS (the candidate list):

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
- id: vllm_ray_not_available_for_multi_node_inference
  severity: error
  match:
    engine: vllm
    fields:
      vllm.parallel.world_size:
        '>': 1
  invariant_under_test: ParallelConfig.__post_init__ flags ray not available for multi-node
    inference
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


INPUT 2 - THE SOURCE PASS 1 READ:

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





OUTPUT FORMAT: a YAML document with TWO sections:

confirmed:
- <id-of-pass1-invariant>  # one ID per line, no further detail needed.

flagged:
- id: <id-of-pass1-invariant>
  reason: <one-line: what looks wrong>
  fix: <one of: "drop", "correct_severity:<new-severity>", "correct_predicate:<new-kind>", "correct_kwargs_positive">

RULES:
1. Every PASS 1 invariant MUST appear in EITHER `confirmed` OR `flagged`
   (not both, not neither). If you're unsure, place in `confirmed` (the
   bar for flagging is "obviously wrong against the source").
2. Flag reasons must be CONCRETE - cite the source line that contradicts
   the invariant, or note the specific shape mismatch.
3. NO markdown code fences. NO commentary outside the YAML.
4. First character must be `c` (from `confirmed:`).

CRITERIA FOR FLAGGING:
- Severity wrong: source has `raise ValueError` but invariant says
  `severity: dormant`; or source has `minor_issues[...] = ...` but
  invariant says `severity: error`.
- Predicate wrong: source has `if X.field not in {a, b}: raise`
  but invariant emits `predicate_kind: exact`.
- Kwargs_positive wrong: source's predicate is "value < 0 raises" but
  invariant's kwargs_positive shows `field: 1` (which would NOT trigger).
- Hallucinated: invariant references a field name that does NOT appear
  in the source.

Emit the YAML now:

```

### Response

```
confirmed:
- vllm_distributed_executor_backend_not_in_allowlist
- vllm_ray_workers_use_nsight_without_ray

flagged:
- id: vllm_ray_not_available_for_multi_node_inference
  reason: Source does not raise an error for multi-node inference with ray, it only checks if ray is available.
  fix: drop

- id: vllm_tpu_backend_not_ray_for_distributed_inference
  reason: Source sets distributed_executor_backend to "ray" when current_platform.device_type is "tpu" and world_size > 1, but does not raise an error for other backends.
  fix: correct_predicate:not_equal
```
