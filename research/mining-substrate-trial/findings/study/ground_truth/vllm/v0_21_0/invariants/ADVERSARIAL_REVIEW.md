# Adversarial GT review - vllm 0.21.0 invariants

Reviewer: adversarial GT auditor (refute-first).
Source under audit: /tmp/vllm-0.21.0/vllm (package root; citation paths resolve from /tmp/vllm-0.21.0/).
GT under audit: PILOT_GT.yaml `confirmed` list (n_confirmed = 132).
Citations resolved by id via passA_entrypoint.yaml and passB_classtree.yaml.
All 132 confirmed ids resolve to a citation entry in their declared source pass (0 unresolved).

## Method
For each sampled entry: resolved id -> citation (file:line:qualname), opened the cited source
line in SOURCE, verified predicate_kind + predicate_value + native_field + bound/allowlist +
outcome (warn vs raise) match the source EXACTLY. Multi-line Literal aliases, role Literals,
sampling constants, and validator auto-invocation chased to definitions to rule out
FALSE-CONFIRM and FABRICATION.

## Sampling scope
Sampled 74 of 132 confirmed entries (56%). Coverage:
- native_type: 23 of 23 distinct config/param classes covered (every class), including BOTH
  ModelConfig entries (model-dependent, treated as suspicious).
- predicate_kind: 21 of 21 distinct kinds covered (every kind), including all complex/new kinds
  (cross_field_combo, presence_conflict, backend_dispatch, cross_runtime_combo, cross_field_gt/lt,
  any_falsy_in_list, all_type_check, type_check, type_is, in_open_range, not_in,
  not_in_range_inclusive/half_open, strenum_in (new at 0.21), is_true (new at 0.21)).
- outcomes: the ONLY non-error confirmed entry (temperature clamp, cit_outcome=warn,
  observed_outcome=dormant_announced) is included.
- Both confirmed entries with kwargs_positive=None (gate-synthesised) included and chased:
  vllm_modelconfig_logprobs_mode_literal, vllm_loraconfig_lora_dtype_literal.

## Counts (sample of 74)
- REAL: 74
- MIS-STATED: 0
- FALSE-CONFIRM: 0
- FABRICATED: 0

No non-real entry surfaced in the sample, so per the thoroughness override no full expansion was
triggered. (Confirmed-outcome distribution across all 132: 131 error, 1 dormant_announced;
cit_outcome 131 invalid, 1 warn; severity 131 error, 1 warning.)

## Cross-checks performed (FALSE-CONFIRM / FABRICATION hunts)
- SamplingParams validators are construction-time: __post_init__ (sampling_params.py:391) calls
  self._verify_args() (line 427) unconditionally and gates _verify_greedy_sampling() (line 434) on
  temperature < _SAMPLING_EPS. Constants confirmed: _SAMPLING_EPS = 1e-5 (line 25),
  _MAX_TEMP = 1e-2 (line 26). Match cited predicate values.
- Multi-line / aliased Literals verified member-for-member vs predicate_value:
  - CacheDType (cache.py): cache_dtype 15-member allowlist matches.
  - MoEBackend (kernel.py:108-122): 13 members INCLUDING the 0.21-new 'humming' and
    'triton_unfused'; exact match to the entry's 13-member predval (version-correct drift).
  - LogprobsMode (model.py:90) = [raw_logits,raw_logprobs,processed_logits,processed_logprobs]:
    exact match (this is the gate-synthesised logprobs_mode entry; rule is a real pydantic Literal).
  - LoRADType (lora.py:25) = [auto,float16,bfloat16]: exact match to lora_dtype entry's predval
    (gate-synthesised). Field type torch.dtype | LoRADType (lora.py:46); invalid string is rejected
    construction-time either by pydantic or by __post_init__ getattr(torch, value) failure - still a
    construction-time membership rule, citation correct.
  - OffloadBackend (offload.py) = [auto,uva,prefetch]; EPLBCommunicatorBackend =
    [torch_nccl,torch_gloo,nixl,pynccl]; WeightTransferConfig.backend Literal[nccl,ipc];
    KVEventsConfig.publisher Literal[null,zmq] (both publisher entries point here): all exact.
  - KVRole (kv_transfer.py:13) = Literal[KVProducer,KVConsumer] -> members
    {kv_producer,kv_both,kv_consumer}; strenum_in entry predval [kv_producer,kv_consumer,kv_both]
    and not_in entry predval [...,None] both match. ECRole (ec_transfer.py:12) analogous; both REAL.
- ModelConfig quantization (model.py:949 qualname _verify_quantization, invoked from __post_init__
  after model_arch_config load): actual raise "Unknown quantization method ... Must be one of ..."
  fires for self.quantization not in me_quant.QUANTIZATION_METHODS regardless of model quant_cfg.
  Because the gate's negative (quantization=None) PASSED on a successfully constructed model, the
  positive ('invalid_quant') error is attributable to the membership check, not a model-load
  failure -> not a FALSE-CONFIRM. Classified REAL. Citation anchors on the def-line (949), not the
  exact raise line; message_template and rule are correct (citation-line imprecision only).
- New 0.21 attention.use_cudnn_prefill removal (is_true / presence_conflict, attention.py:117-122):
  both the is_true entry and the presence_conflict variant point at the same `if self.use_cudnn_prefill:
  raise ValueError("...cuDNN MLA prefill backend has been removed...")` block. is_true with
  predval True is the correct violation predicate. REAL.

## Per-entry verification (sampled = all REAL)
Every sampled cited line contains the claimed rule with matching predicate, field, bound/allowlist,
and outcome. Representative confirmations:
- SamplingParams: presence/frequency_penalty [-2,2] (sp.py:457/461), min_p [0,1] (491),
  top_p (0,1] (476), n type_check (447), n>VLLM_MAX_N_SEQUENCES (451), top_k type_check (487),
  min_tokens<=max_tokens (503), stop empty-string (531/532), stop requires detokenize (533/534),
  stop_token_ids all int (526/527), greedy n==1 (540/541), temperature clamp warn (392) -> REAL.
- Config presence_conflict / cross_field: ec/kv connector-requires-role + unknown-role
  (ec_transfer.py:82/88, kv_transfer.py:97/103), lora max_cpu_loras>=max_loras (lora.py:105/106),
  profiler torch/dir pairing (profiler.py:135/136/139/140), pooler logit_sigma!=0 (pooler.py:152/153),
  observability detailed_traces requires endpoint (observability.py:147/148) + version parse
  (observability.py:115/118), structured_outputs backend gating (structured_outputs.py:64/69),
  multimodal shm-cache cross-field (multimodal.py:248-251) + XFORMERS-removed (multimodal.py:226-235),
  parallel api_process_rank (414), dp_local<=dp (428), external_lb (434), numa_bind_nodes validator
  (parallel.py:376-382), scheduler max_num_batched_tokens<max_model_len gating chunked-prefill
  (scheduler.py:261-264), compilation custom_ops none+all assert (883), invalid-syntax (971),
  encoder_cudagraph_max_vision_items>=0 (994), repetition min_count>=2 (sp.py:143/144) -> REAL.
- pydantic Field(...) constraints: gpu_memory_utilization gt=0,le=1 (cache.py:66),
  kv_cache_metrics_sample gt=0,le=1 (observability.py:53), video_pruning_rate ge=0,lt=1
  (multimodal.py:190), cpu_offload_gb ge=0 (offload.py:23), offload_group_size ge=0 /
  num_in_group ge=1 (offload.py:54/62), active_iterations ge=1 (profiler.py:94; entry expressed as
  violation predicate lt 1, equivalent and correct) -> REAL.

## Systemic issues
None material. Same cosmetic pattern as 0.20.0: a few citations anchor on the validator method's
def-line rather than the exact raise/assert line (ModelConfig._verify_quantization cites 949 vs
raise downstream; ObservabilityConfig version validator cites 115 def vs parse() at 118). In every
sampled case method, message_template, predicate and outcome are correct, so no classification
changes. Two confirmed entries carry kwargs_positive=None because they were gate-SYNTHESISED (the
report notes 3 synthesised probes); their underlying pydantic-Literal rules are real and exact.

## Overall verdict
TRUSTWORTHY. Fraction of the 74-entry representative sample verified REAL: 74/74 = 100%.
Sample spans all 23 native_types, all 21 predicate_kinds (incl. the 0.21-new strenum_in / is_true),
the sole non-error outcome, and both gate-synthesised entries. No MIS-STATED, FALSE-CONFIRM, or
FABRICATED entries found. Version-specific drift (MoEBackend grew 'humming'/'triton_unfused';
cuDNN-prefill removal) is correctly captured.
