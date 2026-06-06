# Pilot GT report - tensorrt 1.2.1 invariants (union + gate)

Round 0 pilot: union the 4 GT sources by tolerant identity (leaf_native_field, coarse_predicate_bucket), runtime-gate every kwargs-bearing candidate in the production validator inside nvcr.io/nvidia/tensorrt-llm/release:1.2.1, keep gate-confirmed as GT.

## Pilot verdict

**The union + gate pipeline works end-to-end.** 4 sources -> tolerant dedup
(144 identities) -> production validator in-container -> 40 runtime-confirmed
identities + a labelled candidate corpus.

**One product-gate bug was found and fixed.** The first run confirmed 0/78
candidates: the gate's `_construct_trtllm` resolved `native_type` only against
a 4-entry 0.21-era map under the `tensorrt_llm.` python namespace, but the
study sources tag types in the LLEM engine namespace (`tensorrt.X`) / bare
names, and 1.2.x's primary args class `TorchLlmArgs` plus several `llmapi`
configs and `PluginConfig` were unresolvable. Fix: resolve the bare class name
across the canonical TRT-LLM export modules and inject the `model` placeholder
for any `*LlmArgs` class (`scripts/validate_invariants.py`).

**The gate-validated GT is materially richer and more trustworthy than the
PoC N=1 GT:**

- **GT-growth = +8** runtime-confirmed identities the PoC GT lacked
  (`allreduce_strategy`, 5 plugin-literal constraints, 2 cache-transceiver
  timeouts) - real, non-tautological additions.
- **Every confirmed entry is runtime-proven** (positive fires + negative
  constructs clean), vs the PoC's single-pass assertion.
- **PoC unique contribution = 0**: every PoC tolerant identity is also in the
  union; the PoC GT is a strict subset.
- **68/78 gateable candidates confirmed (87%)**. The 10 non-confirmations are
  4 redundant duplicates (their tolerant identity is confirmed via a sibling
  source) + 6 genuinely cross-field / `from_dict`-dispatch / heavy-`LLM()`
  probes that a single-field kwargs probe cannot isolate (correctly held
  unverified, not asserted).

**Coverage limitation (the fan-out decision point).** Only 40 / 144 union
identities (28%) are runtime-confirmed. 98 are runtime-UNREACHABLE - not wrong,
but carrying no kwargs probe: all 110 mechanical entries (match.fields only),
all 92 PoC entries (no native_type), and the declarative subset of the Opus
passes. The mechanical miner alone holds 51 unique-but-unverified identities.
These entries DO carry `predicate_kind` + `predicate_value` + `native_field`,
so the gate could SYNTHESIZE positive/negative probes from the declared
predicate (literal_in -> out-of-set / in-set; range -> out-of-range /
in-range; ...), mirroring the planned schema gate's construct-with-probe. That
is the single biggest lever on confirmed-GT coverage and should be decided
before fan-out.

## Per-source candidate counts

| source | raw candidates | tolerant keys | gateable | unique tolerant keys |
|---|---|---|---|---|
| passA | 99 | 86 | 35 | 4 |
| passB | 100 | 88 | 43 | 5 |
| mech | 110 | 95 | 0 | 51 |
| poc | 92 | 80 | 0 | 0 |

## Union + gate

- Union size (distinct tolerant identities across 4 sources): **144**
- Gate-confirmed tolerant identities: **40**
- Gateable candidates (native_type + kwargs): **78** (confirmed=68, failed=10, skipped/hardware=0, infra_error=0)

Group status breakdown (per tolerant identity):

- confirmed: 40
- failed: 6
- unreachable: 98

## GT-growth vs PoC N=1 GT

PoC GT contributed **80** tolerant identities. The gate-confirmed union grows GT by **8** confirmed identities the PoC GT lacked:

- allreduce_strategy [membership]
- bert_attention_plugin [membership]
- gemm_allreduce_plugin [membership]
- gemm_swiglu_plugin [membership]
- kv_transfer_sender_future_timeout_ms [numeric]
- kv_transfer_timeout_ms [numeric]
- low_latency_gemm_plugin [membership]
- low_latency_gemm_swiglu_plugin [membership]

## Gate REJECTIONS (kwargs-bearing candidates that ran and were not confirmed)

| id | source | native_type | observed_outcome |
|---|---|---|---|
| tensorrt_LLM_pytorch_rejects_trt_specific_kwargs | passA | tensorrt.LLM | error |
| tensorrt_baseSparseAttentionConfig_from_dict_algorithm_dispatch | passA | tensorrt.BaseSparseAttentionConfig | error |
| tensorrt_baseSparseAttentionConfig_from_dict_algorithm_required | passA | tensorrt.BaseSparseAttentionConfig | no_op |
| tensorrt_samplingParams_best_of_ge_n | passA | tensorrt.SamplingParams | error |
| tensorrt_baseLlmArgs_guided_decoding_backend_literal | passB | BaseLlmArgs | no_op |
| tensorrt_baseLlmArgs_orchestrator_type_literal | passB | BaseLlmArgs | no_op |
| tensorrt_baseLlmArgs_tokenizer_mode_literal | passB | BaseLlmArgs | no_op |
| tensorrt_guidedDecodingParams_at_most_one_guide | passB | GuidedDecodingParams | no_op |
| tensorrt_samplingParams_best_of_ge_n | passB | SamplingParams | error |
| tensorrt_torchLlmArgs_load_format_enum | passB | TorchLlmArgs | error |

### warn_on_unstable_feature_usage flag (passB flagged as possibly-invalid)

- `tensorrt_torchLlmArgs_warn_on_unstable_feature_usage` (source=passA, gateable=False, verdict=ungated, observed=n/a)
- `tensorrt_torchLlmArgs_warn_on_unstable_feature_usage` (source=passB, gateable=False, verdict=ungated, observed=n/a)
- `tensorrt_torchLlmArgs_warn_on_unstable_feature_usage` (source=poc, gateable=False, verdict=ungated, observed=n/a)
