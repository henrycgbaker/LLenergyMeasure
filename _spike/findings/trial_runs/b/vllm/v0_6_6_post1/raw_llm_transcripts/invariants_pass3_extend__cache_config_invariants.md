# invariants_pass3_extend extraction transcript: cache_config_invariants

- chunk_description: vllm.CacheConfig._verify_args + _verify_cache_dtype + _verify_prefix_caching
- expected_namespaces: ['vllm']
- attempts: 1
- elapsed_sec: 153.57
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
- id: vllm_gpu_memory_utilization_gt_1
  severity: error
  match:
    engine: vllm
    fields:
      vllm.gpu_memory_utilization:
        '>': 1.0
  invariant_under_test: CacheConfig._verify_args flags gpu_memory_utilization > 1.0
- id: vllm_block_size_gt_32_cuda
  severity: error
  match:
    engine: vllm
    fields:
      vllm.block_size:
        '>': 32
  invariant_under_test: CacheConfig._verify_args flags block_size > 32 on CUDA
- id: vllm_cache_dtype_not_in_allowlist
  severity: error
  match:
    engine: vllm
    fields:
      vllm.cache_dtype:
        present: true
        not_in:
        - auto
        - fp8
        - fp8_e4m3
        - fp8_e5m2
  invariant_under_test: CacheConfig._verify_cache_dtype flags cache_dtype not in allowlist
- id: vllm_prefix_caching_with_sliding_window
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sliding_window:
        present: true
  invariant_under_test: CacheConfig._verify_prefix_caching flags prefix caching with
    sliding window


INPUT 2 - PASS 2 FLAGGED ISSUES (for context; consider whether they hint
at related misses):

flagged:
- id: vllm_prefix_caching_with_sliding_window
  reason: Source raises NotImplementedError, not ValueError.
  fix: correct_severity:error


INPUT 3 - THE SOURCE PASS 1 READ:

=== CONTEXT ===
vllm.CacheConfig has 3 validator methods (_verify_args, _verify_cache_dtype, _verify_prefix_caching). Emit one invariant per `if <cond>: raise ...` block. Use namespace `vllm` for engine_params fields.

=== SOURCE: CacheConfig._verify_args ===
    def _verify_args(self) -> None:
        if self.gpu_memory_utilization > 1.0:
            raise ValueError(
                "GPU memory utilization must be less than 1.0. Got "
                f"{self.gpu_memory_utilization}.")
        if (current_platform.is_cuda() and self.block_size is not None
                and self.block_size > 32):
            raise ValueError("CUDA Paged Attention kernel only supports "
                             f"block sizes up to 32. Got {self.block_size}.")


=== SOURCE: CacheConfig._verify_cache_dtype ===
    def _verify_cache_dtype(self) -> None:
        if self.cache_dtype == "auto":
            pass
        elif self.cache_dtype in ("fp8", "fp8_e4m3", "fp8_e5m2"):
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

        if self.sliding_window is not None:
            raise NotImplementedError(
                "Prefix caching is not supported with sliding window. "
                "Run with --disable-sliding-window to use prefix caching.")




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
- id: vllm_cache_dtype_is_auto
  engine: vllm
  library: vllm
  invariant_under_test: CacheConfig._verify_cache_dtype checks cache_dtype is "auto"
  severity: dormant
  native_type: vllm.CacheConfig
  miner_source:
    path: <file_path>
    method: _verify_cache_dtype
    line_at_scan: <approximate_line_number>
  match:
    engine: vllm
    fields:
      vllm.cache_dtype: "auto"
  kwargs_positive:
    cache_dtype: "auto"
  kwargs_negative:
    cache_dtype: "fp8"
  expected_outcome:
    outcome: dormant_announced
    emission_channel: logger_info
    normalised_fields: []
  message_template: 'Using fp8 data type to store kv cache. It reduces the GPU memory footprint and boosts the performance. Meanwhile, it may cause accuracy drop without a proper scaling factor'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
```
