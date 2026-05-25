# invariants_pass3_extend extraction transcript: parallel_config_invariants

- chunk_description: vllm.ParallelConfig __post_init__ + _verify_args
- expected_namespaces: ['vllm']
- attempts: 1
- elapsed_sec: 191.16
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser doing PASS 3 (extension) of multi-pass
invariant extraction. PASS 1 already extracted N invariants from
vllm v0.9.2 for ONE chunk of source. Your job is to
identify ADDITIONAL invariants PASS 1 missed.

INPUT 1 - PASS 1 EXTRACTED INVARIANTS (already known; do NOT re-emit):

invariants:
- id: vllm_data_parallel_size_local_gt_data_parallel_size
  severity: error
  match:
    engine: vllm
    fields:
      vllm.data_parallel_size_local:
        '>': 1
  invariant_under_test: ParallelConfig.__post_init__ flags data_parallel_size_local
    > data_parallel_size
- id: vllm_data_parallel_rank_out_of_range
  severity: error
  match:
    engine: vllm
    fields:
      vllm.data_parallel_rank:
        <: 0
  invariant_under_test: ParallelConfig.__post_init__ flags data_parallel_rank out
    of range
- id: vllm_num_redundant_experts_lt_0_when_enable_eplb
  severity: error
  match:
    engine: vllm
    fields:
      vllm.num_redundant_experts:
        <: 0
      vllm.enable_eplb:
        present: true
        not_equal: false
  invariant_under_test: ParallelConfig.__post_init__ flags num_redundant_experts <
    0 when enable_eplb
- id: vllm_num_redundant_experts_neq_0_when_not_enable_eplb
  severity: error
  match:
    engine: vllm
    fields:
      vllm.num_redundant_experts:
        '!=': 0
      vllm.enable_eplb:
        present: true
        not_equal: true
  invariant_under_test: ParallelConfig.__post_init__ flags num_redundant_experts !=
    0 when not enable_eplb
- id: vllm_distributed_executor_backend_not_in_allowlist
  severity: error
  match:
    engine: vllm
    fields:
      vllm.distributed_executor_backend:
        present: true
        not_in:
        - ray
        - mp
        - uni
        - external_launcher
  invariant_under_test: ParallelConfig._verify_args flags distributed_executor_backend
    not in allowlist
- id: vllm_ray_workers_use_nsight_when_not_use_ray
  severity: error
  match:
    engine: vllm
    fields:
      vllm.ray_workers_use_nsight:
        present: true
      vllm.use_ray:
        present: true
        not_equal: true
  invariant_under_test: ParallelConfig._verify_args flags ray_workers_use_nsight when
    not use_ray


INPUT 2 - PASS 2 FLAGGED ISSUES (for context; consider whether they hint
at related misses):

flagged:
- id: vllm_data_parallel_external_lb_when_data_parallel_size_eq_1
  reason: Source raises ValueError when data_parallel_external_lb is set and data_parallel_size
    > 1, not == 1.
  fix: drop
- id: vllm_num_redundant_experts_neq_0_when_not_enable_eplb
  reason: Source raises ValueError when num_redundant_experts != 0 and enable_eplb
    is False, but invariant says != 0 is incorrect.
  fix: correct_predicate:not_equal


INPUT 3 - THE SOURCE PASS 1 READ:

=== CONTEXT ===
vllm.ParallelConfig validates `distributed_executor_backend` enum and `pipeline_parallel_size`/`tensor_parallel_size`.

=== SOURCE: ParallelConfig.__post_init__ ===
    def __post_init__(self) -> None:
        self.world_size = self.pipeline_parallel_size * \
            self.tensor_parallel_size

        if self.data_parallel_size_local > self.data_parallel_size:
            raise ValueError(
                f"data_parallel_size_local ({self.data_parallel_size_local}) "
                f"must be <= data_parallel_size ({self.data_parallel_size})")

        if self.data_parallel_size > 1 or self.data_parallel_size_local == 0:
            # Data parallel was specified in the engine args.
            self.data_parallel_master_port = get_open_port()

            if not (0 <= self.data_parallel_rank < self.data_parallel_size):
                raise ValueError(
                    f"data_parallel_rank ({self.data_parallel_rank})"
                    f" must be in the range [0, {self.data_parallel_size})")
        else:
            # Otherwise fall back to env vars (e.g. for offline SPMD case).
            self.data_parallel_size = envs.VLLM_DP_SIZE
            self.data_parallel_rank = envs.VLLM_DP_RANK
            self.data_parallel_rank_local = envs.VLLM_DP_RANK_LOCAL
            self.data_parallel_master_ip = envs.VLLM_DP_MASTER_IP
            self.data_parallel_master_port = envs.VLLM_DP_MASTER_PORT

            if self.data_parallel_external_lb:
                raise ValueError("data_parallel_external_lb can only "
                                 "be set when data_parallel_size > 1")

        if self.distributed_executor_backend == "external_launcher":
            import os
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
            logger.info("Disabling V1 multiprocessing for external launcher.")

        if self.enable_eplb:
            if not current_platform.is_cuda():
                raise ValueError(
                    "Expert parallelism load balancing is only supported on "
                    "CUDA devices now.")
            if self.num_redundant_experts < 0:
                raise ValueError(
                    "num_redundant_experts must be non-negative, but got "
                    f"{self.num_redundant_experts}.")
        else:
            if self.num_redundant_experts != 0:
                raise ValueError(
                    "num_redundant_experts should be used with EPLB."
                    f"{self.num_redundant_experts}.")
        if self.distributed_executor_backend is None and self.world_size > 1:
            # We use multiprocessing by default if world_size fits on the
            # current node and we aren't in a ray placement group.

            from vllm.executor import ray_utils
            backend: DistributedExecutorBackend = "mp"
            ray_found = ray_utils.ray_is_available()
            if current_platform.is_neuron():
                # neuron uses single process to control multiple devices
                backend = "uni"
            elif current_platform.is_tpu() and envs.VLLM_XLA_USE_SPMD:
                backend = "uni"
            elif (current_platform.is_cuda()
                  and cuda_device_count_stateless() < self.world_size):
                if not ray_found:
                    raise ValueError("Unable to load Ray which is "
                                     "required for multi-node inference, "
                                     "please install Ray with `pip install "
                                     "ray`.") from ray_utils.ray_import_err
                backend = "ray"
            elif self.data_parallel_backend == "ray":
                logger.info("Using ray distributed inference because "
                            "data_parallel_backend is ray")
                backend = "ray"
            elif ray_found:
                if self.placement_group:
                    backend = "ray"
                else:
                    from ray import is_initialized as ray_is_initialized
                    if ray_is_initialized():
                        from ray.util import get_current_placement_group
                        if get_current_placement_group():

=== SOURCE: ParallelConfig._verify_args ===
    def _verify_args(self) -> Self:
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

        if not current_platform.use_custom_allreduce():
     
...<truncated>...
```

### Response

```
invariants:
- id: vllm_data_parallel_external_lb_when_data_parallel_size_eq_1
  engine: vllm
  library: vllm
  invariant_under_test: ParallelConfig.__post_init__ flags data_parallel_external_lb when data_parallel_size == 1
  severity: error
  native_type: vllm.ParallelConfig
  miner_source:
    path: ParallelConfig.py
    method: __post_init__
    line_at_scan: 15
  match:
    engine: vllm
    fields:
      vllm.data_parallel_external_lb:
        present: true
      vllm.data_parallel_size:
        exact: 1
  kwargs_positive:
    data_parallel_external_lb: True
    data_parallel_size: 1
  kwargs_negative:
    data_parallel_external_lb: False
    data_parallel_size: 2
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: 'data_parallel_external_lb can only be set when data_parallel_size > 1'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: vllm_num_redundant_experts_lt_0_when_enable_eplb_corrected_predicate
  engine: vllm
  library: vllm
  invariant_under_test: ParallelConfig.__post_init__ flags num_redundant_experts < 0 when enable_eplb (corrected predicate)
  severity: error
  native_type: vllm.ParallelConfig
  miner_source:
    path: ParallelConfig.py
    method: __post_init__
    line_at_scan: 25
  match:
    engine: vllm
    fields:
      vllm.num_redundant_experts:
        '<': 0
      vllm.enable_eplb:
        present: true
        not_equal: false
  kwargs_positive:
    num_redundant_experts: -1
    enable_eplb: True
  kwargs_negative:
    num_redundant_experts: 0
    enable_eplb: False
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: 'num_redundant_experts must be non-negative, but got {self.num_redundant_experts}.'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: vllm_distributed_executor_backend_not_in_allowlist_custom_subclass
  engine: vllm
  library: vllm
  invariant_under_test: ParallelConfig._verify_args flags distributed_executor_backend not in allowlist (custom subclass)
  severity: error
  native_type: vllm.ParallelConfig
  miner_source:
    path: ParallelConfig.py
    method: _verify_args
    line_at_scan: 5
  match:
    engine: vllm
    fields:
      vllm.distributed_executor_backend:
        present: true
        type_is_not: [ExecutorBase]
  kwargs_positive:
    distributed_executor_backend: 'custom_subclass'
  kwargs_negative:
    distributed_executor_backend: ExecutorBase
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: 'Unrecognized distributed executor backend {self.distributed_executor_backend}. Supported values are \'ray\', \'mp\' \'uni\', \'external_launcher\' or custom ExecutorBase subclass.'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: vllm_ray_workers_use_nsight_when_not_use_ray_corrected_predicate
  engine: vllm
  library: vllm
  invariant_under_test: ParallelConfig._verify_args flags ray_workers_use_nsight when not use_ray (corrected predicate)
  severity: error
  native_type: vllm.ParallelConfig
  miner_source:
    path: ParallelConfig.py
    method: _verify_args
    line_at_scan: 20
  match:
    engine: vllm
    fields:
      vllm.ray_workers_use_nsight:
        present: true
      vllm.use_ray:
        present: true
        not_equal: true
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
```
