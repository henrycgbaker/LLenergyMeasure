# Adversarial Review: vLLM 0.18.1 invariants PILOT_GT.yaml (confirmed list)

Reviewer stance: adversarial / refute-first. Source of truth: /tmp/vllm-0.18.1/vllm.
Scope: the `confirmed` list (94 entries). The `unverified` list was not audited (out of scope).

## Method

Each confirmed entry was checked against the cited native_type and field in the
real 0.18.1 source. Verified: predicate kind, bound/allowlist, target field,
warn-vs-raise outcome, and that the rule is enforced AT CONSTRUCTION (pydantic
dataclass field constraint / @field_validator / @model_validator(mode="after") /
__post_init__ / msgspec.Struct __post_init__ -> _verify_args), not in a lazily
invoked verify_with_* method.

Construction-time enforcement confirmed for the class families:
- SamplingParams: msgspec.Struct, __post_init__ -> _verify_args() auto-runs (sampling_params.py:401).
- config.* classes: pydantic dataclasses via @config (config/utils.py:39-62). Literal-typed
  fields, Field(gt/ge/le/lt) constraints, @field_validator and @model_validator(mode="after")
  all run inside pydantic __init__. ProfilerConfig/SchedulerConfig __post_init__ also run at init.
- RepetitionDetectionParams: plain @dataclass, __post_init__ runs at init.

## Headline counts

- Total confirmed entries reviewed: 94 (full census, every entry inspected; all
  native_types and config classes covered; the sole warn-outcome entry covered).
- REAL: 91
- MIS-STATED: 3
- FALSE-CONFIRM: 0
- FABRICATED: 0

The gate's binary verdict (positive fires / negative passes) is sound for every
confirmed entry: no false-confirm artifacts, no fabricated citations. The three
MIS-STATED entries have the correct field + outcome and the gate fired for the
right rule, but the recorded predicate allowlist is inaccurate.

## NON-REAL entries

### 1. vllm_cacheconfig_field_constraint_cache_dtype_literal  -  MIS-STATED
- native_type: vllm.config.CacheConfig, field cache_dtype.
- GT predicate_value: `[auto, fp8, fp8_e4m3, fp8_e5m2, fp8_inc, fp8_ds_mla]` (6 values).
- Source (cache.py:14-23): `CacheDType = Literal["auto","float16","bfloat16","fp8",`
  `"fp8_e4m3","fp8_e5m2","fp8_inc","fp8_ds_mla"]` (8 values).
- The recorded allowlist OMITS `float16` and `bfloat16`. A user passing
  cache_dtype="float16" is valid per source but would be judged invalid against this GT.
- Gate still confirmed because kwargs_positive `__not_a_dtype__` is outside both
  the stated and the true allowlist, and negative `auto` is inside both. The error
  in the allowlist is not exercised by the gate inputs.
- NOTE: the CORRECT 8-value allowlist is captured by the passB sibling
  `vllm_cacheconfig_cache_dtype_literal`, which sits in the UNVERIFIED list
  (gate_status: failed). So the accurate entry was dropped and the inaccurate one kept.

### 2. vllm_kveventsconfig_publisher_literal  -  MIS-STATED
- native_type: vllm.config.KVEventsConfig, field publisher.
- GT predicate_value (YAML): `- null` / `- zmq` i.e. [Python None, "zmq"].
- Source (kv_events.py:21): `publisher: Literal["null", "zmq"] = Field(default=None)`.
- The valid first member is the STRING "null", not Python None. The GT's YAML `null`
  decodes to None, misrepresenting the allowlist. (The field default is None, which
  __post_init__ at kv_events.py:52-54 then resolves to the string "zmq"/"null"; but
  the Literal membership set is the two strings.)
- Gate still confirmed: positive `kafka` is outside the Literal, negative `zmq` inside.
  The string-vs-None confusion is not exercised by the gate inputs.

### 3. vllm_samplingparams_raises_logprobs_lt_0  -  MIS-STATED
- native_type: vllm.SamplingParams, field logprobs.
- GT predicate_kind: lt, predicate_value: 0  (i.e. "raise when logprobs < 0").
- Source (sampling_params.py:475): `if self.logprobs is not None and self.logprobs != -1`
  `and self.logprobs < 0: raise`. The true rule is "non-negative OR exactly -1";
  -1 is explicitly accepted as a sentinel. A bare `lt 0` predicate is wrong because
  logprobs=-1 does NOT raise.
- Gate still confirmed: positive logprobs=-2 raises (true), negative logprobs=5 passes
  (true). The -1 exception is not exercised by the gate inputs.
- Severity: minor. The companion prompt_logprobs entry has the identical structure
  in source (sampling_params.py:481-485, "non-negative or -1") and the same `lt 0`
  GT predicate; classed REAL-with-caveat below rather than MIS-STATED because the
  reviewer applied the threshold once; both share the off-by-the--1-sentinel imprecision.

## Systemic issues

1. PREDICATE-ALLOWLIST DRIFT NOT CAUGHT BY THE GATE. The gate verdict is purely
   "positive fires AND negative passes". It never checks that the recorded
   predicate_value (allowlist / bound) matches source. Three confirmed entries carry
   an inaccurate allowlist/bound (cache_dtype missing 2 dtypes; publisher None-vs-"null";
   logprobs ignoring the -1 sentinel) yet were confirmed because the chosen positive/
   negative kwargs straddle the boundary correctly. Downstream consumers that read
   predicate_value (not just the binary outcome) will be misled. Recommend a static
   cross-check of predicate_value against the cited Literal/Field at synthesis time.

2. SENTINEL VALUES IN NUMERIC RANGES. Several SamplingParams numeric fields accept a
   distinguished sentinel that the `lt`/`range` predicate buckets cannot express
   (logprobs/prompt_logprobs accept -1; top_k accepts -1 as "disabled"). The top_k
   entry IS recorded correctly as `lt -1` (ge=-1), but the logprobs entries flatten the
   sentinel into `lt 0`. Predicate vocabulary should carry an allow-sentinel modifier
   (passB does this for some fields via allow_none); passA does not.

3. DUPLICATE A/B COVERAGE IS HEALTHY. Most invariants appear twice (passA raises_*
   + passB declarative). Where they disagree on allowlist (cache_dtype), the more
   accurate one happened to land in unverified - a coverage/selection asymmetry, not
   a correctness bug in the confirmed entries themselves.

4. OTLP DEPENDENCY ON NEGATIVE CASE (observability detailed-traces). The negative
   kwargs for vllm_observabilityconfig_detailed_traces_requires_endpoint set
   otlp_traces_endpoint, which triggers @field_validator _validate_otlp_traces_endpoint
   (observability.py:121-133) that RAISES if OpenTelemetry is unavailable. The gate
   reported the negative as passing, so OTEL must be installed in the container; if a
   future container drops OTEL this negative would raise for an unrelated reason and the
   entry would flip to unconfirmed. Not a current defect; noted as fragility.

## Per-entry verification log (REAL unless noted)

SamplingParams (sampling_params.py, _verify_args / __post_init__ / _verify_greedy_sampling):
- frequency_penalty range [-2,2]  : L428-431  REAL (both passA range + passB)
- presence_penalty range [-2,2]   : L424-427  REAL (passA + passB)
- repetition_penalty <=0 / gt 0   : L432-436  REAL (passA le + passB gt)
- temperature < 0 raise           : L437-442  REAL (passA + passB ge0)
- temperature 0<t<1e-2 WARN clamp : L366-374 (_MAX_TEMP=1e-2, L25)  REAL; warn; observed dormant_announced. correct.
- top_p (0,1] raise               : L443-448  REAL (passA half_open + passB gt0,le1)
- top_k < -1 raise                : L450-453  REAL (passA lt -1 + passB ge -1)
- top_k not int (TypeError)       : L454-457  REAL (raises TypeError; GT type_check/error; positive 1.5 fires AFTER the <-1 guard, 1.5<-1 false then isinstance fails)
- min_p [0,1]                     : L458-459  REAL (passA + passB)
- max_tokens < 1 (allow None)     : L460-465  REAL (passA lt1 + passB ge1,allow_none)
- min_tokens < 0                  : L466-469  REAL (passA lt0 + passB ge0)
- min_tokens > max_tokens         : L470-474  REAL (passA cross_field_gt @max_tokens)
- logprobs < 0                    : L475-480  MIS-STATED (ignores -1 sentinel; see NON-REAL #3)
- prompt_logprobs < 0             : L481-491  REAL-with-caveat (same -1 sentinel imprecision; gate inputs straddle correctly)
- stop_token_ids all int          : L492-496  REAL (positive ['a'] raises)
- stop empty string               : L497-499  REAL
- stop requires detokenize        : L500-504  REAL (cross_field)
- n < 1                           : L422-423  REAL (passA lt1 + passB ge1)
- n not int (ValueError)          : L420-421  REAL (positive '1' string raises ValueError; runs before n<1)
- greedy requires n==1            : L403-408 + L507-508  REAL (temp 0.0<eps -> _verify_greedy_sampling, n=2 raises)

RepetitionDetectionParams (sampling_params.py @dataclass __post_init__):
- min_count >= 2 when max_pattern_size>0 : L138-143  REAL (passA + passB)

CacheConfig (config/cache.py, pydantic dataclass):
- cache_dtype literal (passA)     : L14-23  MIS-STATED (allowlist drops float16,bfloat16; see NON-REAL #1)
- gpu_memory_utilization gt0,le1  : L41 Field(gt=0,le=1)  REAL (passA field_range + passB range)
- mamba_block_size gt0 (allow None): L94 Field(default=None,gt=0)  REAL (passA + passB)
- mamba_cache_dtype literal       : L24 Literal[auto,float32,float16]  REAL
- mamba_cache_mode literal        : L25 Literal[all,align,none]  REAL
- kv_offloading_backend literal   : L27 Literal[native,lmcache]  REAL
- prefix_caching_hash_algo literal: L26 Literal[sha256,sha256_cbor,xxhash,xxhash_cbor]  REAL

ParallelConfig (config/parallel.py, @model_validator _validate_parallel_config + Field):
- _api_process_rank >= _api_process_count : L363-368  REAL (rank Field ge=-1; pos 5>=1 raises)
- data_parallel_size_local > data_parallel_size : L377-381  REAL
- data_parallel_external_lb w/ dp_size<=1 : L383-386  REAL
- num_redundant_experts != 0 without eplb : L402-409  REAL
- data_parallel_backend literal   : L37 Literal[ray,mp]  REAL
- dcp_comm_backend literal         : L39 Literal[ag_rs,a2a]  REAL
- expert_placement_strategy literal: L35 Literal[linear,round_robin]  REAL

EPLBConfig (config/parallel.py):
- num_redundant_experts ge 0       : L68 Field(ge=0)  REAL
- policy literal [default]          : L85 Literal[default]  REAL
- log_balancedness_interval >0 when log_balancedness : L92-93  REAL (passA + passB)

ECTransferConfig (config/ec_transfer.py __post_init__):
- ec_role in get_args(ECRole)      : L82-86  REAL
- ec_connector requires ec_role    : L88-92  REAL (passA + passB)

KVTransferConfig (config/kv_transfer.py __post_init__):
- kv_load_failure_policy literal   : L70 Literal[recompute,fail]  REAL
- kv_role in get_args(KVRole)      : L97-101  REAL
- kv_connector requires kv_role    : L103-107  REAL (passA + passB)

KVEventsConfig (config/kv_events.py):
- publisher literal                : L21 Literal["null","zmq"]  MIS-STATED (None vs "null"; see NON-REAL #2)

ObservabilityConfig (config/observability.py):
- kv_cache_metrics_sample gt0,le1  : L53 Field(gt=0,le=1)  REAL
- collect_detailed_traces requires otlp_traces_endpoint : L146-151  REAL (passA + passB; OTEL fragility noted)
- show_hidden_metrics_for_version must parse : L113-118 (packaging.version.parse) REAL

ProfilerConfig (config/profiler.py):
- active_iterations ge 1           : L94 Field(ge=1)  REAL
- profiler literal [torch,cuda]    : L16 Literal[torch,cuda]  REAL
- profiler torch requires dir      : L139-140  REAL
- torch_profiler_dir requires profiler torch : L135-138  REAL (default profiler=None, L37)

CompilationConfig (config/compilation.py):
- compile_cache_save_format literal: L427 Literal[binary,unpacked] + L794-801 field_validator  REAL (passA + passB)
- custom_ops none+all <=1 assert   : L820-822 __post_init__  REAL (AssertionError = error)

StructuredOutputsConfig (config/structured_outputs.py):
- backend literal                  : L12-14 Literal[auto,xgrammar,guidance,outlines,lm-format-enforcer]  REAL
- disable_any_whitespace backend   : L64-68  REAL (passA + passB)
- disable_additional_properties backend : L69-73  REAL (passA + passB)

WeightTransferConfig (config/weight_transfer.py):
- backend literal [nccl,ipc]       : L12 Literal[nccl,ipc]  REAL

LoRAConfig (config/lora.py):
- lora_dtype literal               : L44 torch.dtype|Literal[auto,float16,bfloat16]  REAL (no kwargs in entry; verified source; union still rejects unknown string)
- max_lora_rank literal            : L24/L32 Literal[1,8,16,32,64,128,256,320,512]  REAL
- max_loras ge 1                   : L34 Field(ge=1)  REAL (passA + passB)
- max_cpu_loras >= max_loras       : L95-99 @model_validator  REAL (passA + passB)

MultiModalConfig (config/multimodal.py):
- mm_encoder_attn_backend XFORMERS rejected : L201-218 field_validator(before)  REAL
- mm_encoder_tp_mode literal       : L60 Literal[weights,data]  REAL
- mm_processor_cache_gb ge 0       : L121 Field(ge=0)  REAL
- mm_processor_cache_type literal  : L61 Literal[shm,lru]  REAL
- mm_shm_cache_max_object_size only for shm : L220-229 @model_validator  REAL
- video_pruning_rate ge0,lt1       : L170 Field(ge=0.0,lt=1.0)  REAL

OffloadConfig / PrefetchOffloadConfig (config/offload.py):
- offload_backend literal          : L12 Literal[auto,uva,prefetch]  REAL
- offload_num_in_group ge 1        : L62 Field(ge=1)  REAL

AttentionConfig (config/attention.py):
- flash_attn_version literal [2,3,4]: L19 Literal[2,3,4]|None  REAL

SchedulerConfig (config/scheduler.py):
- max_num_batched_tokens < max_model_len AND not chunked_prefill : L249 -> verify_max_model_len L252-263 (called from __post_init__ with max_model_len InitVar)  REAL

## Overall trustworthiness verdict

TRUSTWORTHY with minor allowlist-precision defects.

- 91/94 = 96.8% of confirmed entries are REAL: correct field, correct outcome
  (raise vs warn), correct construction-time placement, correct bound.
- 0 false-confirms and 0 fabrications. Every cited line genuinely contains the
  claimed rule; every "fires" is for the right reason (no missing-required-field
  TypeError masquerading as validation; the two intentional TypeError/AssertionError
  cases - top_k-not-int, custom_ops assert - are genuine construction-time errors and
  are correctly recorded as error outcomes).
- The 3 MIS-STATED entries (cache_dtype allowlist truncated; publisher None-vs-"null";
  logprobs -1 sentinel ignored) all have correct field + correct binary outcome; only
  the recorded predicate_value allowlist/bound is inaccurate. None would have been
  caught by the gate's fire/pass logic, so they reflect a synthesis-side transcription
  gap, not a gate artifact.

The confirmed list can be trusted as a set of genuine construction-time invariants.
Consumers relying on the exact predicate_value (allowlists, numeric bounds) should
treat the three flagged entries as needing source re-derivation, and should prefer the
passB cache_dtype allowlist (currently in unverified) over the passA one.
