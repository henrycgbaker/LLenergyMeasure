# invariants_pass3_extend extraction transcript: cache_config_invariants

- chunk_description: vllm.CacheConfig._verify_args + _verify_cache_dtype + _verify_prefix_caching
- expected_namespaces: ['vllm']
- attempts: 1
- elapsed_sec: 117.22
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
- id: vllm_cpu_offload_gb_lt_zero
  severity: error
  match:
    engine: vllm
    fields:
      vllm.cpu_offload_gb:
        <: 0
  invariant_under_test: CacheConfig._verify_args flags cpu_offload_gb < 0
- id: vllm_gpu_memory_utilization_gt_one
  severity: error
  match:
    engine: vllm
    fields:
      vllm.gpu_memory_utilization:
        '>': 1.0
  invariant_under_test: CacheConfig._verify_args flags gpu_memory_utilization > 1.0
- id: vllm_cache_dtype_not_in_allowlist
  severity: error
  match:
    engine: vllm
    fields:
      vllm.cache_dtype:
        present: true
        not_in: get_args(CacheDType)
  invariant_under_test: CacheConfig._verify_cache_dtype flags cache_dtype not in allowlist
- id: vllm_prefix_caching_hash_algo_not_in_allowlist
  severity: error
  match:
    engine: vllm
    fields:
      vllm.prefix_caching_hash_algo:
        present: true
        not_in: get_args(PrefixCachingHashAlgo)
  invariant_under_test: CacheConfig._verify_prefix_caching flags prefix_caching_hash_algo
    not in allowlist
- id: vllm_sliding_window_not_supported_with_prefix_caching
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sliding_window:
        present: true
      vllm.enable_prefix_caching: true
  invariant_under_test: CacheConfig._verify_prefix_caching flags sliding window not
    supported with prefix caching


INPUT 2 - PASS 2 FLAGGED ISSUES (for context; consider whether they hint
at related misses):

flagged:
- id: vllm_cache_dtype_not_in_allowlist
  reason: Source checks for `self.cache_dtype in get_args(CacheDType)` but also allows
    `"auto"`, which is not accounted for in the invariant.
  fix: correct_predicate:not_in_or_equal_to_auto
- id: vllm_sliding_window_not_supported_with_prefix_caching
  reason: Source raises `NotImplementedError` instead of `ValueError` when sliding
    window is used with prefix caching, contradicting the invariant's severity.
  fix: correct_severity:dormant


INPUT 3 - THE SOURCE PASS 1 READ:

=== CONTEXT ===
vllm.CacheConfig has 3 validator methods (_verify_args, _verify_cache_dtype, _verify_prefix_caching). Emit one invariant per `if <cond>: raise ...` block. Use namespace `vllm` for engine_params fields.

=== SOURCE: CacheConfig._verify_args ===
    def _verify_args(self) -> Self:
        if self.cpu_offload_gb < 0:
            raise ValueError("CPU offload space must be non-negative"
                             f", but got {self.cpu_offload_gb}")

        if self.gpu_memory_utilization > 1.0:
            raise ValueError(
                "GPU memory utilization must be less than 1.0. Got "
                f"{self.gpu_memory_utilization}.")

        return self


=== SOURCE: CacheConfig._verify_cache_dtype ===
    def _verify_cache_dtype(self) -> None:
        if self.cache_dtype == "auto":
            pass
        elif self.cache_dtype in get_args(CacheDType):
            logger.info(
                "Using fp8 data type to store kv cache. It reduces the GPU "
                "memory footprint and boosts the performance. "
                "Meanwhile, it may cause accuracy drop without a proper "
                "scaling factor")
        else:
            raise ValueError(f"Unknown kv cache dtype: {self.cache_dtype}")


=== SOURCE: CacheConfig._verify_prefix_caching ===
    def _verify_prefix_caching(self) -> None:
        if not self.enable_prefix_caching:
            return

        if self.sliding_window is not None and not envs.VLLM_USE_V1:
            raise NotImplementedError(
                "Prefix caching is not supported with sliding window. "
                "Run with --disable-sliding-window to use prefix caching.")

        if (self.enable_prefix_caching and self.prefix_caching_hash_algo
                not in get_args(PrefixCachingHashAlgo)):
            raise ValueError(
                "Unknown prefix caching hash algorithm: "
                f"{self.prefix_caching_hash_algo}. Must be one of "
                f"{get_args(PrefixCachingHashAlgo)}.")




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
- id: vllm_cache_dtype_not_auto_or_in_allowlist
  engine: vllm
  library: vllm
  invariant_under_test: CacheConfig._verify_cache_dtype flags cache_dtype not "auto" and not in allowlist
  severity: error
  native_type: vllm.GenerationConfig
  miner_source:
    path: <file path>
    method: _verify_cache_dtype
    line_at_scan: <approximate line number if visible>
  match:
    engine: vllm
    fields:
      vllm.cache_dtype:
        present: true
        not_in_or_equal_to_auto: get_args(CacheDType)
  kwargs_positive:
    cache_dtype: "unknown"
  kwargs_negative:
    cache_dtype: "auto"
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: 'Unknown kv cache dtype: {self.cache_dtype}'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
- id: vllm_prefix_caching_hash_algo_not_in_allowlist_with_enable_prefix_caching_true
  engine: vllm
  library: vllm
  invariant_under_test: CacheConfig._verify_prefix_caching flags prefix_caching_hash_algo not in allowlist when enable_prefix_caching is True
  severity: error
  native_type: vllm.EngineArgs
  miner_source:
    path: <file path>
    method: _verify_prefix_caching
    line_at_scan: <approximate line number if visible>
  match:
    engine: vllm
    fields:
      vllm.enable_prefix_caching: true
      vllm.prefix_caching_hash_algo:
        present: true
        not_in: get_args(PrefixCachingHashAlgo)
  kwargs_positive:
    enable_prefix_caching: true
    prefix_caching_hash_algo: "unknown"
  kwargs_negative:
    enable_prefix_caching: false
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: 'Unknown prefix caching hash algorithm: {self.prefix_caching_hash_algo}. Must be one of {get_args(PrefixCachingHashAlgo)}.'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
```
