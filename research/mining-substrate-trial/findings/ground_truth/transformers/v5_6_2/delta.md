# Delta vs LLEM mined baseline - transformers v5.6.2

## Baseline note

`engine_versions/transformers/v5_6_2/` ships only `producers/` (the miner
scripts). It has **no `outputs/` directory** - the mining substrate has not
been run against v5.6.2 yet. The reference baseline for "what the miner
surfaces" is therefore the v4.57.3 LLEM-mined output
(`engine_versions/transformers/v4_57_3/outputs/`):

- `schema.discovered.json`: 39 `engine_params`, 68 `sampling_params`, 1 nested
  `$def` (`CompileConfig`). (The miner's BNB fields are folded into
  `engine_params`, e.g. `bnb_4bit_quant_type`.)
- `invariants.proposed.yaml` / `invariants.validated.yaml`: ~37 invariants,
  all on the GenerationConfig + BitsAndBytes axis.

Ground truth (this corpus, v5.6.2):
- Schema: 46 engine_params + 68 sampling_params + 22 quantization `$defs`
  + 2 watermarking `$defs` + 1 CompileConfig `$def` + 1 ContinuousBatchingConfig
  `$def` + 16 pipeline kwargs + 38 env-var entries + 4 top-level cache classes
  (+ 11 cache-layer classes).
- Invariants: 118 entries.

**Headline:** The thin slice the miner would surface (GenerationConfig +
BitsAndBytes type checks) is 4-5x smaller than the real v5.6.2 control
surface once quantization configs, the pipeline factory, env vars, the
`from_pretrained` pre-flight gates, and the v5-new constructs
(ContinuousBatchingConfig, the 3 new quant classes, the new from_pretrained
kwargs) are included.

Below: only ADDITIONS over what the miner would surface. The format mirrors
the ground-truth envelopes so each entry is grep-able to its source.

## 1. Schema additions

### 1.1 `from_pretrained` kwargs the baseline omits

| Field | Why it matters | Source |
| --- | --- | --- |
| `weights_only` (default True) | Security-relevant: blocks pickle code execution. Now an explicit v5 signature kwarg. | `modeling_utils.py:3742` |
| `fusion_config` | NEW v5: module-fusion control. | `modeling_utils.py:3743` |
| `disable_mmap` | NEW v5: disable mmap safetensors loading; matters for shared-storage benchmark rigs. | `modeling_utils.py:3744` |
| `tqdm_class` | NEW v5: custom progress-bar class. | `modeling_utils.py:3990` |
| `adapter_kwargs` / `adapter_name` | PEFT/LoRA loader entry-points. | `modeling_utils.py:4004-4005` |
| `generation_config` | Overrides the bundled GenerationConfig; reproducibility-critical. | `modeling_utils.py:4006` |
| `gguf_file` | Hits multiple mutual-exclusion gates. | `modeling_utils.py:4007` |
| `tp_plan` / `tp_size` / `distributed_config` / `device_mesh` | Tensor-parallel surface. `tp_plan` now accepts a dict plan. | `modeling_utils.py:4008-4011` |
| `allow_all_kernels` / `use_kernels` / `kernel_config` | NEW v5 kernel-hub surface. `kernel_config` auto-enables `use_kernels`. | `modeling_utils.py:3998-4001` |
| `experts_implementation` | NEW v5: MoE experts kernel selection (parallel to attn_implementation). | `modeling_utils.py:4093` |
| `key_mapping` | State-dict rename map. | `modeling_utils.py:4002` |
| `torch_dtype` | Deprecated alias for `dtype`. | `modeling_utils.py:3995` |
| `mirror`, `_fast_init`, `low_cpu_mem_usage`, `from_tf`, `from_flax`, `offload_state_dict` | All six silently popped no-ops (dead-kwargs sweep). | `modeling_utils.py:4008` |

**Why it matters:** the six dead kwargs are accepted but discarded. A pinned
config setting `low_cpu_mem_usage=True` (or `from_tf=True` migrated from a
TF-era script) executes but the flag is dead - the miner would pass it
through as "documented".

### 1.2 GenerationConfig: the lazy-default refactor

The miner surfaces `sampling_params` with concrete defaults (max_length=20,
temperature=1.0, ...). In v5 those literal defaults are **gone from
`__init__`**: every field is `kwargs.pop(name, None)`. The effective defaults
live in `GenerationConfig._get_default_generation_params()` (line 551) and are
applied lazily at generate-time. Ground truth records both the `init_default`
(None) and the effective `default`.

**Why it matters:** a miner that constructs a `GenerationConfig()` and reads
attributes back will see `None` for almost every field in v5, not the
documented defaults. Any introspection-based default extraction breaks on the
bump unless it knows to consult `_get_default_generation_params()`.

### 1.3 New sampling field: `top_h`

NEW in v5 (`configuration_utils.py:376`): entropy-based top-H sampling
control. Has a greedy-strip dormant warning in `validate()` (line 652).

### 1.4 Quantization-config classes entirely missing from the miner slice

The miner surfaces only BitsAndBytes fields. Ground truth adds 21 other
`$defs`, including the **3 v5-new classes**:

- `MetalConfig` (`quantization_config.py:1798`) - Apple Silicon MPS affine
  quant. bits in {2,4,8}; positive group_size.
- `FourOverSixConfig` (`quantization_config.py:1842`) - adaptive NVFP4
  (arxiv 2512.02010). 16 fields, no post_init gates.
- `SinqConfig` (`quantization_config.py:1937`) - SINQ / A-SINQ weight-only
  quant. method allowlist {sinq, asinq}; group_size multiple-of-8 advisory.

Plus the 18 carried-forward classes (GPTQ, AWQ, Aqlm, Vptq, Quanto, Eetq,
Hqq, AutoRound, CompressedTensors, FbgemmFp8, Higgs, FPQuant, TorchAo, BitNet,
SpQR, FineGrainedFP8, Quark, Mxfp4).

**Why it matters:** quantization is the highest-leverage knob in
energy-per-token measurement. The miner can validate BNB but a user passing
`GPTQConfig(bits=5)` or `SinqConfig(method='xyz')` slips through to a runtime
ValueError.

### 1.5 Watermarking configs, CompileConfig, ContinuousBatchingConfig

- `WatermarkingConfig` (5 fields) + `SynthIDTextWatermarkingConfig` (7 fields)
  - referenced from `GenerationConfig.watermarking_config`, unchanged from
  v4.57.3.
- `CompileConfig` (5 fields) - the miner already surfaces this (correct).
- `ContinuousBatchingConfig` (19 fields) - **NEW v5 dataclass** reachable via
  `GenerationConfig.continuous_batching_config`. Entirely absent from the
  miner slice.

### 1.6 Cache taxonomy

The miner lifts the `cache_implementation` string enum (correct). Ground truth
additionally enumerates the v5-rearchitected class taxonomy: 4 top-level Cache
classes + 11 `CacheLayerMixin` subclasses, and flags that `SlidingWindowCache`
is now a deprecated alias of `StaticCache` (`cache_utils.py:1574`).

### 1.7 Pipeline factory kwargs

`pipeline()` (`pipelines/__init__.py:440`) is entirely absent from the miner
slice. Ground truth adds all 16 keyword params. Notable: `framework` was
**dropped in v5** (TF support removed); `dtype` defaults to `"auto"`
(and from_pretrained now also resolves None->'auto', closing the v4.57.3
divergence).

### 1.8 Environment variables

The miner does not enumerate env vars. Ground truth adds 38. The v5-relevant
ones:

- Transformers-owned: `HF_MODULES_CACHE`, `TRANSFORMERS_VERBOSITY`,
  `TRANSFORMERS_NO_ADVISORY_WARNINGS`, `TRANSFORMERS_IS_CI`, `CI` (new),
  `TRUST_REMOTE_CODE`, `WORLD_SIZE`, `LOCAL_RANK`, `RANK` (new TP dependency),
  plus the SageMaker/ECS telemetry probes.
- huggingface_hub pass-through (carried, Medium confidence): the `HF_*` /
  `HF_HUB_*` family.

**HIGH-VALUE CALLOUT - the offline-mode binding changed in v5.** In v4.57.3,
`transformers/utils/hub.py` bound `_is_offline_mode` once at import and
`TRANSFORMERS_OFFLINE` was a no-op foot-gun. In v5, `hub.py` imports
`is_offline_mode` directly from `huggingface_hub` (line 40); the
transformers-local binding AND the `TRANSFORMERS_OFFLINE` read are **gone**.
Offline gating is now fully delegated to `huggingface_hub.constants.HF_HUB_OFFLINE`.
A submission validator carried over from a v4.57.3 mental model will
mis-describe this.

## 2. Invariant additions

Miner slice = ~37 invariants (GenerationConfig + BNB type checks). Ground
truth = 118. Categories below mirror the v4.57.3 delta but updated for v5.

### 2.1 GenerationConfig `validate()` - 8 generate-only-kwarg rejections

The validate body rejects 8 generate-only kwargs (`logits_processor`,
`stopping_criteria`, `prefix_allowed_tokens_fn`, `synced_gpus`,
`assistant_model`, `streamer`, `negative_prompt_ids`,
`negative_prompt_attention_mask`) at lines 727-742. **NOTE: this is 8, not 9
- v5 dropped `use_model_defaults` from the rejection tuple.** A user sticking
`streamer` on the config dict still hits the error.

### 2.2 GenerationConfig.validate() - full dormant/greedy/beam set

Adds the greedy-strips (temperature, top_p, min_p, **top_h (new)**, typical_p,
top_k, epsilon_cutoff, eta_cutoff), the beam-strips (early_stopping,
length_penalty), the 4 output-flag dormant warnings (incl. output_logits),
the 2 cache-when-use_cache-false warnings (**note: only 2 arms now, not 3 -
return_legacy_cache removed**), and the num_return_sequences gates.

### 2.3 save_pretrained / from_model_config

- `save_pretrained` runs `validate(strict=True)`: dormant warnings become
  refusal-to-save errors. (v5 dropped the `use_auth_token` handling.)
- `save_directory` must not be a file.
- `from_model_config` silently sets `return_dict_in_generate=True` when any
  output flag is True.

### 2.4 Quantization invariants (~62 across 22 classes)

All BNB type checks plus the full post_init gate set for the other 21 classes,
including the v5-new ones:
- `MetalConfig`: bits {2,4,8}; positive group_size.
- `SinqConfig`: method {sinq,asinq}; nbits int-coercible; group_size%8 advisory.
- `GPTQConfig` NEW: `act_group_aware` auto-disabled when desc_act=True.
- `AwqConfig` REWORKED: format allowlist {gemm,gemv,gemv_fast,llm-awq};
  backend allowlist (13 values); llm-awq normalisation.
- `FineGrainedFP8Config`: activation_scheme now {dynamic,static}.
- `FPQuantConfig`: backward_dtype now {bf16,mxfp8,mxfp4} + cross-field gate.

### 2.5 `from_pretrained` pre-flight gates (~22)

Adds: state_dict mutual exclusion; gguf-vs-quant / gguf-vs-disk; gguf requires
accelerate; device_map string allowlist / negative-int / deepspeed / accelerate
/ meta-anti-pattern (relocated to `integrations/accelerate.py`); tp_size
requires tp_plan; tp_plan vs device_map; tp requires torch>=2.5; tp rejects
MPS; device_mesh tp-dim (relocated to `integrations/tensor_parallel.py`); the
6-field dead-kwargs sweep; dtype None->'auto'; torch_dtype->dtype; kernel_config
auto-enables use_kernels.

## 3. Net delta count

| Category | Miner slice (v4.57.3 outputs) | Ground truth (v5.6.2) | Net adds |
| --- | --- | --- | --- |
| `engine_params` | 39 | 46 | +7 net (and several renamed/removed; see version_delta) |
| `sampling_params` | 68 | 68 | ~0 count; but +`top_h`, -`return_legacy_cache`, + the lazy-default semantics |
| `quantization_configs` ($defs) | 0 (BNB folded in) | 22 | +22 |
| `watermarking_configs` ($defs) | 0 | 2 | +2 |
| `ContinuousBatchingConfig` ($def) | 0 | 1 | +1 |
| `cache_classes` | 0 enumerated | 4 top + 11 layer | +15 |
| `pipeline_kwargs` | 0 | 16 | +16 |
| `env_vars` | 0 | 38 | +38 |
| Invariants | ~37 | 118 | +81 |

## 4. Three concrete high-value additions worth fixing in the miner first

1. **The GenerationConfig lazy-default refactor.** Any miner that introspects
   a constructed `GenerationConfig()` to read defaults will see `None` for
   nearly every field in v5. The miner must learn to consult
   `_get_default_generation_params()` (line 551) or it will report a wrong
   schema on the v4->v5 bump.

2. **The 3 new quantization classes + the AWQ rework.** `MetalConfig`,
   `FourOverSixConfig`, `SinqConfig` are new; `AwqConfig` changed from a
   standalone class to a `GPTQConfig` subclass with a different field set
   (version->format, backend enum). Pointing the AST walker at all 22
   `QuantizationConfigMixin` subclasses catches `SinqConfig(method='xyz')`
   and `MetalConfig(bits=3)` at submission.

3. **The offline-mode binding change.** A submission validator that warns
   "TRANSFORMERS_OFFLINE is a no-op, use HF_HUB_OFFLINE before import" (correct
   for v4.57.3) is now describing a binding that no longer exists in
   transformers. v5 delegates entirely to huggingface_hub. The warning text
   must be version-gated.
