# Adversarial Review: vLLM 0.19.1 runtime-gated invariant GT

Source under review: PILOT_GT.yaml `confirmed` list (n=90 declared).
Engine source: /tmp/vllm-0.19.1/vllm
Reviewer posture: adversarial / refute-by-default. Each entry verified against the
cited source line; classified REAL | MIS-STATED | FALSE-CONFIRM | FABRICATED.

This file is appended to as the review proceeds (durable record).

## Per-entry findings

### SamplingParams (vllm/sampling_params.py) - verified against full file read

SamplingParams is a msgspec.Struct with __post_init__ (line 373) that calls
_verify_args() (line 409) unconditionally at construction, and
_verify_greedy_sampling() (line 416) when temperature < _SAMPLING_EPS. Both are
auto-invoked at construction -> construction-time. The verify()/_validate_* methods
(lines 624+) are NOT auto-invoked (engine-call-time) - any confirmed entry resting on
those would be FALSE-CONFIRM. None of the confirmed SamplingParams entries cite those.


## Methodology

Verification combined (a) full source reads of vllm/sampling_params.py and all cited
config modules in /tmp/vllm-0.19.1/vllm/config, and (b) DIRECT EMPIRICAL EXECUTION in
the cited container (vllm/vllm-openai:v0.19.1, CPU-only) of the positive and negative
kwargs for a representative sample spanning EVERY native_type/config class, ALL
warn-severity entries, every cross-field / presence_conflict / mutual-exclusion entry,
the args-model-style entries, and every entry lacking hand-authored kwargs_positive.
For each probed positive I confirmed it raises AND that the exception MESSAGE matches the
cited source line (fire-for-the-right-reason), and that the negative passes.

Empirically executed positives/negatives (>50 construct calls): StructuredOutputsConfig
(backend literal, disable_additional_properties, disable_any_whitespace), WeightTransferConfig
(backend), ObservabilityConfig (detailed_traces requires endpoint, show_hidden_metrics
version-parse), CompilationConfig (compile_cache_save_format literal, custom_ops none+all
assert, mode IntEnum), ParallelConfig (data_parallel_backend, dcp_comm_backend,
expert_placement_strategy), AttentionConfig (flash_attn_version), CacheConfig
(gpu_memory_utilization, mamba_block_size, mamba_cache_mode), KVTransferConfig
(kv_role unknown, connector-without-role, kv_load_failure_policy), ECTransferConfig
(ec_role unknown, connector-without-role), EPLBConfig (log_balancedness_interval,
num_redundant_experts ge=0, policy literal), LoRAConfig (lora_dtype, max_lora_rank,
max_loras, max_cpu_loras>=max_loras), MultiModalConfig (mm_encoder_attn_backend XFORMERS,
video_pruning_rate, mm_processor_cache_type, mm_encoder_tp_mode), OffloadConfig
(offload_backend), SchedulerConfig (policy, runner_type), ProfilerConfig (torch requires
dir, dir requires torch, profiler kind), KVEventsConfig (publisher), RepetitionDetectionParams
(min_count, pattern sizes), SamplingParams (n type/range/greedy/max_n, presence/frequency
penalty, repetition_penalty, temperature lt0 + clamp-warning, top_p, top_k type/range, min_p,
min_tokens, max_tokens, logprobs, prompt_logprobs, stop empty/detokenize, stop_token_ids type).

Every probed positive fired; every probed negative passed; every message matched the cited
predicate. The temperature-clamp warning entry (severity=warning, observed_outcome=
dormant_announced) was verified to WARN and NOT raise - correctly classified.

## Key correctness checks that could have refuted entries (all passed)

- SamplingParams __post_init__ (line 373) -> _verify_args() (409) is auto-invoked at
  construction; _verify_greedy_sampling() (416) auto-invoked when temperature<_SAMPLING_EPS.
  The lazy verify()/_validate_* methods (lines 624+) are NOT auto-invoked - and NO confirmed
  entry rests on them. The confirmed SamplingParams set maps 1:1 to _verify_args /
  _verify_greedy_sampling / __post_init__ lines. No method-call-time rule mis-classified
  as construction-time.
- All config classes are pydantic @config dataclasses; the cited validators are either
  model_validator(mode="after"), field_validator, or __post_init__ - all run at construction.
  No confirmed entry relies on a lazy _verify_with_*/create_engine_config path.
- custom_ops none+all is an assert (count_none+count_all<=1, line 848); fires as
  pydantic assertion_error with the exact assert message. REAL, not a generic error.
- num_redundant_experts fires on the EPLBConfig Field(ge=0), not the unrelated
  ParallelConfig "EPLB not enabled" check. REAL.
- kv_role/ec_role "unknown role" raise the get_args(KVRole)/get_args(ECRole) message; null
  is allowed (source guards `is not None`), consistent with GT allowlist including null.
- Three confirmed entries carry NO hand-authored kwargs_positive
  (compile_cache_save_format, lora_dtype, samplingparams__verify_args_top_p_le). The gate
  SYNTHESISES probes from the predicate for these (pilot_metrics: confirmed_via_synthesis=6).
  I executed all three: each genuinely raises at construction for the correct reason
  (lora_dtype union torch.dtype|LoRADType rejects the invalid string; compile_cache_save_format
  literal_error; top_p (0,1] VLLMValidationError). REAL - synthesis did not manufacture a
  false confirm.

## Non-REAL entries

NONE. No MIS-STATED, FALSE-CONFIRM, or FABRICATED entries were found in the 90-entry
confirmed list. Every entry's predicate bound/allowlist/field/outcome/severity matches the
cited source line, and (for the probed sample) fires for the claimed reason.

## Systemic observations (not defects in correctness)

1. DUPLICATE IDS: two ids each appear twice in `confirmed` from different sources, so the
   list has 90 rows but 88 unique ids:
   - vllm_samplingparams_raises_n_gt_max_n_sequences (poc + passA)
   - vllm_modelconfig_raises_unknown_quantization_method (passA + poc)
   Both members of each pair are individually REAL (same rule, same source line), but the
   identity-collapse did not dedupe across source within `confirmed`. Cosmetic/accounting
   issue; n_confirmed=90 double-counts these two rules. Effective unique confirmed rules = 88.

2. MODELCONFIG REACHABILITY (caveat, not a refutation): the two ModelConfig entries
   (quantization) require a full ModelConfig construction (HF config fetch). I confirmed by
   source that _verify_quantization runs in __post_init__ (line 679) and raises "Unknown
   quantization method" (line 1012); I could not re-run it offline (no network in sandbox
   -> fails earlier on repo resolution). Trust rests on source + the gate's having had
   network. Outcome/predicate are source-accurate. Classified REAL.

3. CROSS-FIELD "single-field" framing: several entries are framed as single-field numeric
   (e.g. eplbconfig_raises_log_balancedness_interval_le_0, predicate_kind=le/0) but the
   underlying rule is cross-field (only fires when log_balancedness=True). The kwargs
   correctly set the gating field, so the confirmation is valid; the constraint_key framing
   is just coarser than the source. Not a correctness defect.

## Verdict

TOTAL REVIEWED: 90 confirmed entries (100% of the confirmed list).
  REAL:          90
  MIS-STATED:     0
  FALSE-CONFIRM:  0
  FABRICATED:     0

Verification depth: 100% source-checked against cited lines; >50 of 90 (>55%, spanning
EVERY native_type, ALL warn-severity entries, ALL no-kwargs entries, ALL cross-field/
mutual-exclusion entries, and the args-model SamplingParams set) additionally executed
end-to-end in the cited container with message-level fire-for-the-right-reason checks.
The remaining ~40 are simple pydantic Literal/Field constraints of the same families as
probed siblings, each source-verified.

TRUSTWORTHINESS: HIGH. Fraction verified REAL = 90/90 = 1.00. The runtime gate's
synthesis path did not manufacture false confirms (verified empirically on all three
no-kwargs entries). The only blemishes are accounting (two duplicate-id rows -> 88 unique
rules) and one source-only (not re-executed offline) ModelConfig caveat. No entry should
be demoted from `confirmed`.
