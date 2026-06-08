# Adversarial GT review - vllm 0.20.0 invariants

Reviewer: adversarial GT auditor (refute-first).
Source under audit: /tmp/vllm-0.20.0/vllm (package root; citation paths resolve from /tmp/vllm-0.20.0/).
GT under audit: PILOT_GT.yaml `confirmed` list (n_confirmed = 119).
Citations resolved by id via passA_entrypoint.yaml (100 invariants) and passB_classtree.yaml (92).
All 119 confirmed ids resolve to a citation entry in their declared source pass (0 unresolved).

## Method
For each sampled entry: resolved id -> citation (file:line:qualname), opened the cited
source line in SOURCE, and verified predicate_kind + predicate_value + native_field +
bound/allowlist + outcome (warn vs raise) match the source line EXACTLY. Multi-line Literal
aliases and validator auto-invocation were chased to their definitions to rule out
FALSE-CONFIRM (gate artifact) and FABRICATION.

## Sampling scope
Sampled 68 of 119 confirmed entries (57%). Coverage:
- native_type: 25 of 25 distinct config/param classes covered (every class).
- predicate_kind: 21 of 21 distinct kinds covered (every kind), including ALL semantically
  complex kinds (cross_field_combo, presence_conflict, platform_combo, backend_dispatch,
  cross_runtime_combo, cross_field_gt/lt, regex_no_match, any_falsy_in_list, all_type_check,
  type_check, type_is, not_in_range_inclusive/half_open, in_open_range).
- outcomes: the ONLY non-`invalid`/non-error confirmed entry (the temperature clamp,
  cit_outcome=warn, observed_outcome=dormant_announced) is included.
- All entries flagged as suspicious during triage (ModelConfig quantization, offload
  num_in_group default-driven fire, SamplingParams type_check via ValueError, greedy gate)
  were explicitly chased.

## Counts (sample of 68)
- REAL: 68
- MIS-STATED: 0
- FALSE-CONFIRM: 0
- FABRICATED: 0

No non-real entry surfaced in the sample, so per the thoroughness override no full expansion
was triggered. (Confirmed-outcome distribution across all 119: 118 error, 1 dormant_announced;
cit_outcome 118 invalid, 1 warn; severity 118 error, 1 warning.)

## Cross-checks performed (FALSE-CONFIRM / FABRICATION hunts)
- SamplingParams._verify_args and _verify_greedy_sampling ARE construction-time: __post_init__
  (sampling_params.py:387) calls self._verify_args() (line 423) unconditionally, and gates
  _verify_greedy_sampling() (line 430) on `temperature < _SAMPLING_EPS`. Constants confirmed:
  _SAMPLING_EPS = 1e-5 (line 25), _MAX_TEMP = 1e-2 (line 26). Both match cited predicate values.
  Greedy-n entries (kwargs temperature=0.0 -> below EPS -> greedy -> n>1 raises) fire for the
  claimed reason, not an unrelated error.
- Multi-line Literal aliases verified member-for-member against predicate_value:
  - CacheDType (cache.py:18): 15 members, exact match to cache_dtype allowlist.
  - MoEBackend (kernel.py:108): 11 members, exact match to moe_backend allowlist.
  - OffloadBackend (offload.py:12) = [auto,uva,prefetch]: exact.
  - EPLBCommunicatorBackend (parallel.py:39) = [torch_nccl,torch_gloo,nixl,pynccl]: exact.
  - SchedulerPolicy (scheduler.py:22) = [fcfs,priority]; RunnerType (scheduler.py:21) =
    [generate,pooling,draft]: exact.
  - WeightTransferConfig.backend Literal[nccl,ipc] (weight_transfer.py:12): exact.
  - KVEventsConfig.publisher Literal[null,zmq] (kv_events.py:19): exact.
  - AttentionConfig.flash_attn_version Literal[2,3,4] (attention.py:19): exact.
- OffloadConfig num_in_group<=group_size (offload.py:100-106): kwargs_positive sets only
  offload_backend=prefetch; defaults offload_group_size=0 (offload.py:54), offload_num_in_group=1
  (offload.py:62), so the `prefetch` branch is entered and 1 > 0 raises -> fires for the claimed
  reason. kwargs_negative offload_backend=auto with group_size=0 does not enter the branch -> passes.
- ModelConfig quantization (model.py:932 qualname _verify_quantization, called from __post_init__
  line 687 after model_arch_config is loaded line 523): the actual raise "Unknown quantization
  method: ... Must be one of ..." is at line 1012, reached for any self.quantization not in
  me_quant.QUANTIZATION_METHODS regardless of quant_cfg. Because the gate's kwargs_negative
  (quantization=None) PASSED with a successfully constructed model, the kwargs_positive
  (quantization='invalid_quant') error is attributable to the quantization membership check and
  NOT to a model-load failure. Classified REAL. Minor note: the citation line (932 = method def)
  is the qualname anchor, not the exact raise line (1012); message_template and rule are correct,
  so this is a citation-line imprecision, NOT a fabrication or mis-statement.

## Per-entry verification (sampled = all REAL)
Every sampled entry's cited source line contains the claimed rule with matching predicate,
field, bound/allowlist, and outcome. Representative confirmations:
- SamplingParams numeric ranges: presence_penalty/frequency_penalty [-2,2] (sp.py:453/457),
  min_p [0,1] (487), top_p (0,1] (472), logprobs<0 (504), n>VLLM_MAX_N_SEQUENCES (447),
  n type_check (443), top_k type_check (483), min_tokens<=max_tokens (499), stop empty-string
  (528), stop requires detokenize (530), stop_token_ids all int (523), temperature clamp warn
  (388) -> all REAL, exact.
- Config presence_conflict / cross_field rules verified at the exact branch:
  ec/kv connector-requires-role (ec_transfer.py:88-89, kv_transfer.py:103-104),
  lora max_cpu_loras>=max_loras (lora.py:105-106), profiler torch/dir pairing
  (profiler.py:136/140), pooler logit_bias/mean + logit_sigma!=0 (pooler.py:125/153),
  observability detailed_traces requires endpoint (observability.py:148-149),
  structured_outputs backend gating (structured_outputs.py:64/69),
  multimodal shm-cache cross-field (multimodal.py:228-232) and XFORMERS-removed
  backend_dispatch (multimodal.py:207-216), parallel dp_local<=dp / external_lb / numa_bind /
  eplb-requires-cuda (parallel.py:426/432/437/444), compilation custom_ops none+all assert
  (compilation.py:883) and invalid-syntax raise (compilation.py:971),
  repetition-detection min_count>=2 (sampling_params.py:139-140) -> all REAL, exact.
- pydantic Field(...) constraints: gpu_memory_utilization gt=0,le=1 (cache.py:66),
  kv_cache_metrics_sample gt=0,le=1 (observability.py:53), video_pruning_rate ge=0,lt=1
  (multimodal.py:171), count ge=0 (multimodal.py:19), num_frames gt=0 (multimodal.py:26),
  cpu_offload_gb ge=0 (offload.py:23), offload_group_size ge=0 / num_in_group ge=1
  (offload.py:54/62), active_iterations ge=1 (profiler.py) -> all REAL, exact.

## Systemic issues
None material. One cosmetic pattern: a few citations anchor on the qualname's def-line rather
than the exact raise/assert line within the method (e.g. ModelConfig._verify_quantization
cites 932 vs raise at 1012). In every sampled case the method, message_template, predicate and
outcome are still correct, so this does not change any classification.

## Overall verdict
TRUSTWORTHY. Fraction of the 68-entry representative sample verified REAL: 68/68 = 100%.
Sample spans all 25 native_types, all 21 predicate_kinds, and the sole non-error outcome.
No MIS-STATED, FALSE-CONFIRM, or FABRICATED entries found. The vllm CPU-only fire/pass gate
model is sound for these entries (validators are construction-time auto-invoked; the one
model-dependent ModelConfig entry is corroborated by its passing negative).
