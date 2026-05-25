# invariants_pass2_verify extraction transcript: cache_config_invariants

- chunk_description: vllm.CacheConfig._verify_args + _verify_cache_dtype + _verify_prefix_caching
- expected_namespaces: ['vllm']
- attempts: 1
- elapsed_sec: 44.52
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser doing PASS 2 (verification) of multi-pass
invariant extraction. PASS 1 already extracted invariants from
vllm v0.9.2 for ONE chunk of source. Your job is to
REVIEW each emitted invariant against the source and flag any that
look WRONG.

INPUT 1 - PASS 1 EXTRACTED INVARIANTS (the candidate list):

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


INPUT 2 - THE SOURCE PASS 1 READ:

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
- vllm_cpu_offload_gb_lt_zero
- vllm_gpu_memory_utilization_gt_one
- vllm_prefix_caching_hash_algo_not_in_allowlist

flagged:
- id: vllm_cache_dtype_not_in_allowlist
  reason: Source checks for `self.cache_dtype in get_args(CacheDType)` but also allows `"auto"`, which is not accounted for in the invariant.
  fix: correct_predicate:not_in_or_equal_to_auto
- id: vllm_sliding_window_not_supported_with_prefix_caching
  reason: Source raises `NotImplementedError` instead of `ValueError` when sliding window is used with prefix caching, contradicting the invariant's severity.
  fix: correct_severity:dormant
```
