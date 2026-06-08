# Adversarial GT review - vllm 0.22.0 invariants

Reviewer: adversarial GT auditor (refute-first).
Source under audit: /tmp/vllm-0.22.0/vllm (package root; citation paths resolve from /tmp/vllm-0.22.0/).
GT under audit: PILOT_GT.yaml `confirmed` list (n_confirmed = 118).
Citations resolved by id via passA_entrypoint.yaml and passB_classtree.yaml.
All 118 confirmed ids resolve to a citation entry in their declared source pass (0 unresolved).

## Method
For each sampled entry: resolved id -> citation (file:line:qualname), opened the cited source
line in SOURCE, verified predicate_kind + predicate_value + native_field + bound/allowlist +
outcome (warn vs raise) match the source EXACTLY. Multi-line Literal aliases, enums, role
Literals, sampling constants, and validator auto-invocation chased to definitions to rule out
FALSE-CONFIRM and FABRICATION.

## Sampling scope
Sampled 68 of 118 confirmed entries (58%). Coverage:
- native_type: 25 of 25 distinct config/param classes covered (every class), including the
  0.22-new LoadConfig and DynamicShapesConfig, plus the single model-dependent ModelConfig entry.
- predicate_kind: 20 of 20 distinct kinds covered (every kind), including all complex kinds
  (cross_field_combo, presence_conflict, backend_dispatch, cross_runtime_combo, cross_field_gt/lt,
  any_falsy_in_list, all_type_check, type_check, type_is, in_open_range, not_in,
  not_in_range_inclusive/half_open, strenum_in).
- outcomes: the ONLY non-error confirmed entry (temperature clamp, cit_outcome=warn,
  observed_outcome=dormant_announced) is included.
- Both confirmed entries with kwargs_positive=None (gate-synthesised) included and chased:
  vllm_eplbconfig_raises_log_balancedness_interval_le_0, vllm_loraconfig_lora_dtype_literal.

## Counts (sample of 68)
- REAL: 68
- MIS-STATED: 0
- FALSE-CONFIRM: 0
- FABRICATED: 0

No non-real entry surfaced in the sample, so per the thoroughness override no full expansion was
triggered. (Confirmed-outcome distribution across all 118: 117 error, 1 dormant_announced;
cit_outcome 117 invalid, 1 warn; severity 117 error, 1 warning.)

## Cross-checks performed (FALSE-CONFIRM / FABRICATION hunts)
- SamplingParams validators are construction-time: __post_init__ (sampling_params.py:398) calls
  self._verify_args() (line 434) unconditionally and gates _verify_greedy_sampling() (line 441) on
  temperature < _SAMPLING_EPS. Constants: _SAMPLING_EPS = 1e-5 (line 25), _MAX_TEMP = 1e-2 (line 26).
  Match cited predicate values.
- Multi-line / aliased Literals & enums verified member-for-member vs predicate_value:
  - MoEBackend (kernel.py): 14 members INCLUDING the 0.22-new 'flashinfer_b12x'; exact match to
    moe_backend entry's 14-member predval (version-correct drift vs 0.21's 13).
  - LinearBackend (kernel.py): 15 members; exact match to the new linear_backend entry's predval
    (KernelConfig.linear_backend is new at 0.22).
  - DynamicShapesType (compilation.py:322/327/332) = backed/unbacked/backed_size_oblivious; exact
    match to the new DynamicShapesConfig.type strenum predval.
  - CacheDType: 15-member cache_dtype allowlist matches. AttentionConfig.flash_attn_version
    Literal[2,3,4] (attention.py:20): exact.
  - MLAPrefillBackendEnum (imported attention.py:9, used in validate_mla_prefill_backend_before
    via MLAPrefillBackendEnum[value.upper()] at line 110): invalid string -> KeyError at the
    mode=before field_validator -> construction-time membership rejection. predval @registry. REAL.
  - OffloadBackend = [auto,uva,prefetch]; EPLBCommunicatorBackend = [torch_nccl,torch_gloo,nixl,
    pynccl]; KVEventsConfig.publisher Literal[null,zmq]: all exact.
  - LoRADType (lora.py:25) = [auto,float16,bfloat16]: exact (gate-synthesised lora_dtype entry).
  - KVRole (kv_transfer.py:13) and ECRole (ec_transfer.py:12) expand to {*_producer,*_both,
    *_consumer}; the strenum_in and not_in (incl. None) role entries both match. REAL.
- ModelConfig quantization (model.py): citation points at the EXACT raise line 1084 ("Unknown
  quantization method: ... Must be one of ...") inside _verify_quantization, fired for
  self.quantization not in supported_quantization. Gate negative (quantization=None) PASSED on a
  successfully constructed model, so the positive ('invalid_quant') error is attributable to the
  membership check, not a model-load failure -> not a FALSE-CONFIRM. REAL. (Note: 0.22's citation
  is more precise than 0.20/0.21, which anchored on the method def-line.)
- Gate-synthesised entries (report: 3 synthesised): the two sampled (eplb
  log_balancedness_interval<=0 at parallel.py:102, lora_dtype literal at lora.py:46) carry
  kwargs_positive=None because the gate synthesised the probe; both underlying rules are real and
  exact in source.

## Per-entry verification (sampled = all REAL)
Every sampled cited line contains the claimed rule with matching predicate, field, bound/allowlist,
and outcome. Representative confirmations:
- SamplingParams: presence/frequency_penalty [-2,2] (sp.py:464/468), min_p [0,1] (498),
  top_p (0,1] (483), n type_check (453), n>VLLM_MAX_N_SEQUENCES (458), top_k type_check (494),
  min_tokens<=max_tokens (510), stop empty-string (538), stop requires detokenize (540),
  stop_token_ids all int (533), bad_words empty-string (new at 0.22, sp.py:546), greedy n==1
  (553/554), temperature clamp warn (399/400) -> REAL.
- Config presence_conflict / cross_field: ec/kv connector-requires-role + unknown-role
  (ec_transfer.py:82/88, kv_transfer.py:97/103), lora max_cpu_loras>=max_loras (lora.py:112),
  profiler torch/dir pairing (profiler.py:135/139), pooler logit_sigma!=0 (pooler.py:152/153),
  observability detailed_traces requires endpoint (observability.py:148/149) + version parse
  (observability.py:113/118), structured_outputs backend gating (structured_outputs.py:64/69),
  multimodal shm-cache cross-field (multimodal.py:247-251) + XFORMERS-removed (multimodal.py:226-232),
  parallel dcp a2a requires dcp>1 (new at 0.22, parallel.py:484/485), dp_local<=dp (432),
  external_lb (438), scheduler max_num_batched_tokens>=max_model_len gating chunked-prefill
  (scheduler.py:261-264), compilation custom_ops none+all assert (892) + compile_cache_save_format
  (867), repetition min_count>=2 (sp.py:143/144) -> REAL.
- pydantic Field(...) constraints: gpu_memory_utilization gt=0,le=1 (cache.py:66),
  kv_cache_metrics_sample gt=0,le=1 (observability.py:53), cpu_offload_gb ge=0 (offload.py:23;
  expressed both as range{ge:0} and as violation-predicate lt 0 - equivalent), num_in_group ge=1
  (offload.py:62), max_num_seqs ge=1 (scheduler.py:63), safetensors_prefetch_num_threads ge=1
  (new LoadConfig, load.py:86), num_frames gt=0 (multimodal.py:27) -> REAL.

## Systemic issues
None. Citation precision improved over 0.20/0.21 (ModelConfig quantization now anchors the exact
raise line 1084). Two confirmed entries carry kwargs_positive=None because they were gate-SYNTHESISED
(report notes 3 synthesised probes); their underlying rules are real and exact. The dual
cpu_offload_gb entries (range vs lt) and dual logit_sigma entries (presence_conflict vs eq) are
intentional alternative encodings of the same source check, both correct.

## Overall verdict
TRUSTWORTHY. Fraction of the 68-entry representative sample verified REAL: 68/68 = 100%.
Sample spans all 25 native_types (incl. 0.22-new LoadConfig, DynamicShapesConfig, KernelConfig
linear_backend), all 20 predicate_kinds, the sole non-error outcome, and the gate-synthesised
entries. No MIS-STATED, FALSE-CONFIRM, or FABRICATED entries found. Version-specific additions
(bad_words checks, dcp a2a gate, MoEBackend 'flashinfer_b12x', DynamicShapesType, MLAPrefillBackendEnum,
safetensors prefetch threads) are correctly captured with precise citations.
