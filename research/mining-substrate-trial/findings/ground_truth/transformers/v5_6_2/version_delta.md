# Version delta - transformers v4.57.3 -> v5.6.2

Explicit diff of this ground-truth corpus against the v4.57.3 ground truth
(`research/mining-substrate-trial/findings/ground_truth/transformers/v4_57_3/`).
This is a major-version bump; the surface changed substantially.

Confirmed against the v5.0.0 release notes
(github.com/huggingface/transformers/releases/tag/v5.0.0).

## 0. File-size deltas (orientation)

| File | v4.57.3 LOC | v5.6.2 LOC |
| --- | --- | --- |
| generation/configuration_utils.py | 1478 | 1834 |
| utils/quantization_config.py | 2110 | 2002 |
| modeling_utils.py | 6165 | 4998 |
| utils/hub.py | 1162 | 949 |
| cache_utils.py | (monolithic) | 1574 (refactored) |

## 1. GenerationConfig

### 1.1 ADDED fields

| Field | Notes | Source |
| --- | --- | --- |
| `top_h` | Entropy-based top-H sampling. New greedy-strip dormant warning. | `configuration_utils.py:376` |
| `continuous_batching_config` | New ContinuousBatchingConfig dataclass ($def added). Not validated in validate(). | `configuration_utils.py:432` |

### 1.2 REMOVED fields

| Field | Notes |
| --- | --- |
| `return_legacy_cache` | Gone from `__init__` and from the cache-strip arm of `validate()`. |

### 1.3 SEMANTICS CHANGED (same field name)

| Field | v4.57.3 | v5.6.2 |
| --- | --- | --- |
| ALL fields | Concrete literal defaults inline in `__init__` (max_length=20, temperature=1.0, ...). | **BREAKING:** every field is `kwargs.pop(name, None)`. Effective defaults moved to `_get_default_generation_params()` (line 551), applied lazily at generate-time. The config object holds `None` until then. |
| `is_assistant` | Hard-coded `self.is_assistant = False` (not a kwarg surface). | Now a regular `kwargs.pop("is_assistant", None)` (line 418); user-settable. |
| `transformers_version` | "4.57.3" | "5.6.2" |

### 1.4 validate() invariant deltas

| Change | v4.57.3 | v5.6.2 |
| --- | --- | --- |
| generate-only rejection tuple | 9 entries (incl. `use_model_defaults`). | **8 entries**: `use_model_defaults` REMOVED. The other 8 retained. (line 727) |
| greedy-strip set | (no top_h) | `top_h` greedy-strip ADDED (line 652). |
| cache-when-use_cache-false loop | iterated 3 arms (cache_implementation, cache_config, return_legacy_cache). | iterates 2 arms (cache_implementation, cache_config); return_legacy_cache removed (line 711). |
| cache_implementation allowlist | 11 values. | 11 values + `paged` added at validate time (line 618). |

### 1.5 save_pretrained

| Change | v4.57.3 | v5.6.2 |
| --- | --- | --- |
| `use_auth_token` deprecation + mutual-exclusion handling | Present (2 invariants). | REMOVED. Only the strict-validate gate + file-not-dir assertion remain. |

## 2. from_pretrained / modeling_utils

### 2.1 ADDED kwargs

`weights_only` (now explicit, default True), `fusion_config`, `disable_mmap`
(promoted to explicit signature); `tqdm_class`, `allow_all_kernels`,
`kernel_config`, `experts_implementation` (new popped kwargs).

### 2.2 REMOVED kwargs

| Field | v4.57.3 | v5.6.2 |
| --- | --- | --- |
| `load_in_8bit` / `load_in_4bit` | Deprecated aliases that synthesised a BNB config. | **Gone entirely** (count 0 in modeling_utils). Use quantization_config. |
| `use_auth_token` | Deprecated alias of token (raised on mutual-exclusion). | **Gone entirely.** |
| `resume_download` | Silently popped no-op. | Gone entirely. |
| `from_tf` / `from_flax` | Selected the TF/Flax loader. | Demoted to silently-popped no-ops (dead-kwargs sweep line 4008). TF/Flax loading removed. |
| `low_cpu_mem_usage`, `offload_state_dict`, `mirror`, `_fast_init` | Silently popped no-ops (scattered). | Consolidated into one dead-kwargs sweep at line 4008. |

### 2.3 SEMANTICS CHANGED

| Field | v4.57.3 | v5.6.2 |
| --- | --- | --- |
| `dtype` default | Unresolved default None. | Resolves None -> 'auto' (line 4014). Behavioural change; closes the v4.57.3 pipeline-vs-model divergence. |
| `tp_plan` | Only 'auto' accepted as non-null string; "only auto" gate at modeling_utils. | Now accepts a `dict[str, str]` plan too; the "only auto" gate is gone. TP setup relocated. |
| device_map gates | Inline in modeling_utils.py (~line 4786). | Relocated to `integrations/accelerate.py:check_and_set_device_map` (same 4-value allowlist + negative-int + deepspeed + accelerate + meta-anti-pattern). |
| tp gates | Inline in modeling_utils. | Relocated to `integrations/tensor_parallel.py:initialize_tensor_parallelism`. New gates: torch>=2.5 required; MPS rejected; RANK env required. |

### 2.4 ADDED invariants (from_pretrained)

`from_pretrained_tp_requires_torch_25`, `from_pretrained_tp_not_supported_on_mps`,
`from_pretrained_kernel_config_auto_enables_use_kernels`,
`from_pretrained_dtype_none_resolves_to_auto`. The dead-kwargs sweep is now a
single normalisation invariant covering 6 fields.

### 2.5 REMOVED invariants (from_pretrained)

`from_pretrained_use_auth_token_vs_token_mutual_exclusion`,
`from_pretrained_use_auth_token_deprecation_warning`,
`from_pretrained_load_in_bit_vs_quantization_config_mutual_exclusion`,
`from_pretrained_load_in_bit_deprecated_warning` - all gone because the
underlying kwargs were removed.

## 3. Quantization configs

### 3.1 ADDED classes (3)

| Class | Notes | Source |
| --- | --- | --- |
| `MetalConfig` | Apple Silicon MPS affine quant. bits {2,4,8}. | `quantization_config.py:1798` |
| `FourOverSixConfig` | Adaptive NVFP4 (arxiv 2512.02010). 16 fields, no post_init gates. | `quantization_config.py:1842` |
| `SinqConfig` | SINQ / A-SINQ. method {sinq,asinq}; group_size%8 advisory. | `quantization_config.py:1937` |

New `QuantizationMethod` enum members: `METAL`, `FOUR_OVER_SIX`, `SINQ`.

### 3.2 RENAMED / REWORKED

| Class | Change |
| --- | --- |
| `AwqConfig` | **REWORKED:** now SUBCLASSES `GPTQConfig` (was standalone). `version` field replaced by `format` (AwqFormat enum). `backend` is now an AwqBackend enum (13 values). REMOVED fields: `version` (legacy kwarg still honoured), `do_fuse`, `fuse_max_seq_len`, `modules_to_fuse`, `exllama_config`. New AwqFormat enum adds `gemv_fast`; AwqBackend adds auto_trainable/machete/marlin/exllama_v1/exllama_v2/gemm_triton/torch_awq/torch_fused_awq. |
| `GPTQConfig` | `checkpoint_format` RENAMED to `format` (legacy key accepted). NEW field `act_group_aware` (default True, auto-disabled when desc_act=True). REMOVED fields: `use_exllama`, `use_cuda_fp16`, `exllama_config`. New normalisation invariant for act_group_aware. |
| `FineGrainedFP8Config` | NEW field `dequantize`. `activation_scheme` enum widened to {dynamic,static}. quant_method now QuantizationMethod.FP8. |
| `FPQuantConfig` | `backward_dtype` enum widened from bf16-only to {bf16,mxfp8,mxfp4}. New cross-field gate (non-bf16 backward requires mxfp4 forward). |

### 3.3 UNCHANGED classes (field surface stable)

BitsAndBytesConfig, AqlmConfig, VptqConfig, QuantoConfig, EetqConfig,
HqqConfig, AutoRoundConfig, FbgemmFp8Config, HiggsConfig, TorchAoConfig,
BitNetQuantConfig, SpQRConfig, QuarkConfig, Mxfp4Config. (CompressedTensorsConfig
constructor restructured but loader-facing surface stable.)

Net: v4.57.3 had 19 quant configs; v5.6.2 has 22 (+3, with AwqConfig reworked).

## 4. Cache taxonomy

**BREAKING refactor.** v4.57.3 had 13 monolithic top-level Cache classes.
v5.6.2 has 4 (`DynamicCache`, `StaticCache`, `QuantizedCache`,
`EncoderDecoderCache`) plus an 11-class `CacheLayerMixin` composition layer.

REMOVED as top-level classes (refactored into layers or aliased):
`SlidingWindowCache` (now `= StaticCache` alias, line 1574), `HybridCache`,
`HybridChunkedCache`, `OffloadedCache`, `OffloadedStaticCache`,
`OffloadedHybridCache`, `SinkCache`, `MambaCache`, `QuantoQuantizedCache`,
`HQQQuantizedCache`.

ADDED layer classes: CacheLayerMixin, DynamicLayer, DynamicSlidingWindowLayer,
StaticLayer, StaticSlidingWindowLayer, QuantizedLayer, QuantoQuantizedLayer,
HQQQuantizedLayer, LinearAttentionCacheLayerMixin, LinearAttentionLayer,
LinearAttentionAndFullAttentionLayer.

## 5. Pipeline

| Change | v4.57.3 | v5.6.2 |
| --- | --- | --- |
| `framework` kwarg | Present ('pt'/'tf'/None). | REMOVED (TF support dropped). |
| kwarg count | 17. | 16. |

## 6. Environment variables

### 6.1 REMOVED (transformers-owned reads)

`TRANSFORMERS_CACHE`, `PYTORCH_TRANSFORMERS_CACHE`,
`PYTORCH_PRETRAINED_BERT_CACHE`, `HUGGINGFACE_CO_RESOLVE_ENDPOINT`,
`HUGGINGFACE_CO_STAGING`, `TORCH_HOME`, `XDG_CACHE_HOME` (no longer read by
hub.py), `TRANSFORMERS_OFFLINE` (no longer read by transformers at all).

### 6.2 SEMANTICS CHANGED

`HF_HUB_OFFLINE`: in v4.57.3 the gate was a transformers-local import-time
binding (`_is_offline_mode` at hub.py:81) and `TRANSFORMERS_OFFLINE` was a
no-op foot-gun. In v5, `hub.py` imports `is_offline_mode` directly from
`huggingface_hub` (line 40); the transformers-local binding is gone and
offline gating is fully delegated upstream. The v4.57.3 "set HF_HUB_OFFLINE
before import" gotcha is now whatever huggingface_hub does, not a transformers
artefact.

### 6.3 ADDED reads

`CI` (logging flush behaviour, logging.py:114), `RANK` (TP process-group init,
tensor_parallel.py:62). The SageMaker/ECS telemetry probes
(`ECS_CONTAINER_METADATA_URI`, `SM_FRAMEWORK_PARAMS`, `TRAINING_JOB_ARN`,
`SM_FRAMEWORK_MODULE`, `AWS_REGION`, `SM_NUM_GPUS`, `SM_NUM_CPUS`) are now
itemised (present in v4.57.3 source too but not enumerated in that GT).

## 7. Net version-delta counts

| Axis | Added | Removed | Renamed/Reworked |
| --- | --- | --- | --- |
| GenerationConfig fields | 2 (top_h, continuous_batching_config) | 1 (return_legacy_cache) | 1 semantics (is_assistant) + global lazy-default refactor |
| from_pretrained kwargs | 7 (weights_only*, fusion_config, disable_mmap, tqdm_class, allow_all_kernels, kernel_config, experts_implementation) | 4 (load_in_8bit, load_in_4bit, use_auth_token, resume_download) | 6 demoted to dead no-ops (from_tf, from_flax, low_cpu_mem_usage, offload_state_dict, mirror, _fast_init) |
| Quantization classes | 3 (Metal, FourOverSix, Sinq) | 0 | 4 (AwqConfig, GPTQConfig, FineGrainedFP8Config, FPQuantConfig) |
| Cache classes (top-level) | 0 | 9 (refactored to layers/aliases) | 1 alias (SlidingWindowCache=StaticCache) + 11 new layer classes |
| Pipeline kwargs | 0 | 1 (framework) | 0 |
| Env vars | 2 (CI, RANK) | 8 (cache/endpoint/offline reads) | 1 semantics (HF_HUB_OFFLINE binding) |
| Invariants | ~5 new (tp torch/mps gates, kernel_config, dtype-auto) | ~6 (use_auth_token x2, load_in_bit x2, return_legacy_cache arm, use_model_defaults rejection) | several relocated to integrations/* |

**Totals:** roughly **19 added** surface elements, **23 removed**, **24
renamed/reworked/semantics-changed** across all axes. This is a high-churn
bump: a miner pinned to the v4.57.3 landmark set would mis-report defaults
(the lazy-default refactor), miss 3 quant classes, surface 4 dead kwargs as
live, and carry a stale TRANSFORMERS_OFFLINE warning.
