# invariants_pass3_extend extraction transcript: parallel_config_invariants

- chunk_description: vllm.ParallelConfig __post_init__ + _verify_args
- expected_namespaces: ['vllm']
- attempts: 1
- elapsed_sec: 120.90
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser doing PASS 3 (extension) of multi-pass
invariant extraction. PASS 1 already extracted N invariants from
vllm v0.6.0 for ONE chunk of source. Your job is to
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
- id: vllm_ray_workers_use_nsight_when_not_using_ray
  severity: error
  match:
    engine: vllm
    fields:
      vllm.parallel.use_ray: false
      vllm.parallel.ray_workers_use_nsight:
        present: true
  invariant_under_test: ParallelConfig._verify_args flags ray_workers_use_nsight when
    not using Ray


INPUT 2 - PASS 2 FLAGGED ISSUES (for context; consider whether they hint
at related misses):

flagged:
- id: vllm_distributed_executor_backend_not_in_allowlist
  reason: Source allows custom ExecutorBase subclass, but invariant does not.
  fix: correct_predicate:not_in_or_subclass


INPUT 3 - THE SOURCE PASS 1 READ:

=== CONTEXT ===
vllm.ParallelConfig validates `distributed_executor_backend` enum and `pipeline_parallel_size`/`tensor_parallel_size`.

=== SOURCE: ParallelConfig.__post_init__ ===


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
        if is_hip():
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
   b. Multi-clause `if A and B and C:` invariants where PASS 1 keyed on
      only A. Emit one invariant per (A, B, C) tuple if each is independent.
   c. Type-check invariants `if not isinstance(x, T): raise` that PASS 1
      may have skipped (these often follow a similar pattern repeated
      per field).
   d. Less-common predicates: `<=`, `>=`, `not_equal`, `present`. PASS 1
      may have collapsed these to `exact`.
4. If PASS 1 covered every visible invariant, emit `invariants: []` (empty
   list) - do NOT fabricate.
5. Use snake_case_with_engine_prefix for `id`.
6. Set `added_by: llm_miner_pass3` (NOT `llm_miner`) so the multi-pass
   bookkeeping can track pass-3 contributions.

Emit the YAML now:

```

### Response

```
invariants:
- id: vllm_use_ray_asserts_ray_available
  engine: vllm
  library: vllm
  invariant_under_test: ParallelConfig._verify_args asserts Ray availability when use_ray is True
  severity: error
  native_type: vllm.ParallelConfig
  miner_source:
    path: vllm/parallel.py
    method: _verify_args
    line_at_scan: 10
  match:
    engine: vllm
    fields:
      vllm.use_ray: true
  kwargs_positive:
    use_ray: true
  kwargs_negative:
    use_ray: false
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: 'Ray is not available.'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
- id: vllm_ray_workers_use_nsight_requires_use_ray
  engine: vllm
  library: vllm
  invariant_under_test: ParallelConfig._verify_args requires use_ray when ray_workers_use_nsight is True
  severity: error
  native_type: vllm.ParallelConfig
  miner_source:
    path: vllm/parallel.py
    method: _verify_args
    line_at_scan: 15
  match:
    engine: vllm
    fields:
      vllm.ray_workers_use_nsight: true
      vllm.use_ray: false
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
  message_template: 'Unable to use nsight profiling unless workers run with Ray.'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
- id: vllm_disable_custom_all_reduce_on_hip
  engine: vllm
  library: vllm
  invariant_under_test: ParallelConfig._verify_args disables custom all-reduce kernel on HIP devices
  severity: dormant
  native_type: vllm.ParallelConfig
  miner_source:
    path: vllm/parallel.py
    method: _verify_args
    line_at_scan: 20
  match:
    engine: vllm
    fields:
      is_hip: true
  kwargs_positive:
    is_hip: true
  kwargs_negative:
    is_hip: false
  expected_outcome:
    outcome: dormant_announced
    emission_channel: logger_info
    normalised_fields: []
  message_template: 'Disabled the custom all-reduce kernel because it is not supported on AMD GPUs.'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
```
