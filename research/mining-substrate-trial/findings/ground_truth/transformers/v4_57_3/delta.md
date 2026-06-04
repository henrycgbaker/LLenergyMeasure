# Delta vs LLEM mined baseline - transformers v4.57.3

Baseline:
- `engine_versions/transformers/v4_57_3/outputs/schema.discovered.json`
  (38 `engine_params`, 58 `sampling_params`, 1 nested `$def` for
  `CompileConfig`).
- `engine_versions/transformers/v4_57_3/outputs/invariants.proposed.yaml`
  (37 invariants total, all on the GenerationConfig / BitsAndBytes axis).
- `engine_versions/transformers/v4_57_3/outputs/invariants.validated.yaml`
  (39 validated case rows; same coverage).

Ground truth (this corpus):
- Schema: ~38 engine_params + 65 sampling_params + 19 quantization `$defs`
  + 17 pipeline kwargs + 32 env-var entries + 13 cache classes.
- Invariants: ~115 entries.

**Headline:** The mined baseline covers a thin slice (GenerationConfig +
BitsAndBytes type checks). The actual control surface LLEM users touch is
4-5x larger when quantization configs, pipeline factory, env vars, and
`from_pretrained` pre-flight gates are included.

Below: only ADDITIONS (and one correction). The format mirrors the
ground-truth envelopes so each entry is grep-able to its source.

## 1. Schema additions

### 1.1 `from_pretrained` kwargs the baseline omits

The baseline mines `from_pretrained` kwargs via a docstring walker. That
catches the documented set but misses kwargs the function `pop`s from
`**kwargs` without docstring entries.

| Field | Why it matters | Source |
| --- | --- | --- |
| `use_auth_token` | Deprecated alias of `token`. Hits a deprecation warning + a mutual-exclusion error with `token`. LLEM users with pinned scripts from 4.30-era still set it. | `modeling_utils.py:4625` |
| `adapter_kwargs` | PEFT adapter loader kwargs. Default `{}`. Whenever LLEM loads a LoRA checkpoint this is the actual entry-point. | `modeling_utils.py:4640` |
| `adapter_name` | PEFT adapter name. Default `"default"`. | `modeling_utils.py:4641` |
| `generation_config` | Overrides the model's bundled `GenerationConfig`. Critical for reproducibility-sensitive benchmarking. | `modeling_utils.py:4642` |
| `gguf_file` | Path to a GGUF checkpoint. Hits multiple mutual-exclusion gates (quantization, disk offload). | `modeling_utils.py:4643` |
| `distributed_config` | Forces `tp_plan='auto'` when set. | `modeling_utils.py:4646` |
| `use_kernels` | Opts into HF kernel-hub attention kernels. | `modeling_utils.py:4649` |
| `key_mapping` | State-dict rename map; auto-populated for VLM classes. | `modeling_utils.py:4651` |
| `low_cpu_mem_usage` | Silently popped no-op in 4.57.3. **Baseline curated.yaml comment claims this is dropped from the docstring; the kwarg is still accepted but is a silent no-op.** This is a behavioural change worth surfacing as an invariant. | `modeling_utils.py:4665` |
| `offload_state_dict` | Silently popped no-op in 4.57.3. | `modeling_utils.py:4666` |
| `torch_dtype` | Deprecated alias for `dtype`; aliased with `warning_once`. Baseline only has `dtype`. | `modeling_utils.py:4629` |
| `state_dict` | The baseline has it but as a bare `{"type": "object"}` with no x-source line. Ground truth pins it to the `kwargs.pop` site at line 4620. | `modeling_utils.py:4620` |
| `_from_pipeline` | Internal kwarg; affects telemetry user-agent. Surface because pipeline() sets it. | `modeling_utils.py:4626` |
| `_from_auto` | Internal kwarg set by Auto* classes. | `modeling_utils.py:4627` |
| `_commit_hash` | Internal kwarg; short-circuits hash resolution. | `modeling_utils.py:4638` |
| `resume_download` | Silently popped (no-op, deprecated). | `modeling_utils.py:4662` |
| `mirror` | Silently popped (no-op, deprecated). | `modeling_utils.py:4663` |
| `_fast_init` | Silently popped (no-op). | `modeling_utils.py:4664` |

**Why it matters:** LLEM treats the `**kwargs` catch-all as an opaque
discovery limitation. The baseline's "tracks the catchall" note in
`schema.discovered.json` Lines 18-19 is correct, but the catch-all
actually contains six silently-discarded kwargs and at least four
deprecated aliases. If a user's pinned config sets `low_cpu_mem_usage=True`
in 4.57.3, the run will execute but the flag is dead - LLEM should flag
that explicitly, not pass it through as "documented".

### 1.2 GenerationConfig fields the baseline marks as "no type annotation"

The baseline's `discovery_limitations` block correctly notes that 31
sampling fields have runtime default `None` and no type annotation, so the
introspector returns `{"description": "runtime default was None; ..."}`.
Ground truth resolves these from the docstring + source pops:

| Field | Resolved type | Default | Source |
| --- | --- | --- | --- |
| `max_new_tokens` | int \| null | null | `configuration_utils.py:336` |
| `min_new_tokens` | int \| null | null | `configuration_utils.py:338` |
| `max_time` | number \| null | null | `configuration_utils.py:340` |
| `stop_strings` | str \| list[str] \| null | null | `configuration_utils.py:341` |
| `cache_config` | object \| null | null | `configuration_utils.py:350` |
| `return_legacy_cache` | bool \| null | null | `configuration_utils.py:352` |
| `prefill_chunk_size` | int \| null | null | `configuration_utils.py:353` |
| `min_p` | number \| null | null | `configuration_utils.py:359` |
| `bad_words_ids` | list[list[int]] \| null | null | `configuration_utils.py:367` |
| `forced_bos_token_id` | int \| null | null | `configuration_utils.py:369` |
| `forced_eos_token_id` | int \| list[int] \| null | null | `configuration_utils.py:370` |
| `exponential_decay_length_penalty` | tuple[int, float] \| null | null | `configuration_utils.py:372` |
| `suppress_tokens` | list[int] \| null | null | `configuration_utils.py:373` |
| `begin_suppress_tokens` | list[int] \| null | null | `configuration_utils.py:374` |
| `sequence_bias` | dict[tuple[int], float] \| null | null | `configuration_utils.py:375` |
| `guidance_scale` | number \| null | null | `configuration_utils.py:377` |
| `watermarking_config` | $ref WatermarkingConfig \| $ref SynthIDTextWatermarkingConfig \| object \| null | null | `configuration_utils.py:379` |
| `output_logits` | bool \| null | null | `configuration_utils.py:392` |
| `pad_token_id` | int \| null | null | `configuration_utils.py:396` |
| `bos_token_id` | int \| null | null | `configuration_utils.py:397` |
| `eos_token_id` | int \| list[int] \| null | null | `configuration_utils.py:398` |
| `decoder_start_token_id` | int \| list[int] \| null | null | `configuration_utils.py:402` |
| `prompt_lookup_num_tokens` | int \| null | null | `configuration_utils.py:409` |
| `max_matching_ngram_size` | int \| null | null | `configuration_utils.py:410` |
| `assistant_early_exit` | int \| null | null | `configuration_utils.py:411` |
| `low_memory` | bool \| null (deprecated v4.62) | null | `configuration_utils.py:421` |
| `penalty_alpha` | number \| null (deprecated v4.62) | null | `configuration_utils.py:422` |
| `dola_layers` | str \| list[int] \| null (deprecated v4.62) | null | `configuration_utils.py:423` |
| `constraints` | list \| null | null | `configuration_utils.py:426` |
| `force_words_ids` | list \| null | null | `configuration_utils.py:427` |
| `num_assistant_tokens_schedule` | str (enum) | "constant" | `configuration_utils.py:407` |

**Why it matters:** Six of these (`low_memory`, `penalty_alpha`,
`dola_layers`, `diversity_penalty`, `num_beam_groups`, `constraints`,
`force_words_ids`) are **explicitly marked for removal in v4.62.0** in the
source (lines 420-427). The baseline surfaces them with no deprecation
context. LLEM users pinning 4.57.3 today will silently break on the bump.

### 1.3 Sampling fields the baseline omits entirely

| Field | Why missed | Source |
| --- | --- | --- |
| `is_assistant` | Hard-coded `self.is_assistant = False` (no `kwargs.pop`); not user-tunable. The baseline reports default=`false` and type=`bool` correctly, but ground truth flags it as **not a kwarg surface**, which downstream rebinds should respect. | `configuration_utils.py:405` |
| `num_assistant_tokens_schedule` enum | Baseline has type=string default="constant" but no enum. Valid set is `{constant, heuristic, heuristic_transient}` (docstring at line 293). | `configuration_utils.py:293-296` |

### 1.4 Quantization-config classes ENTIRELY missing from baseline

The baseline `schema.discovered.json` exposes only BitsAndBytesConfig
fields (lifted from its docstring). All 18 OTHER quantization config
classes are absent. Ground truth adds them as `$defs`:

| Class | Why it matters | Source line range |
| --- | --- | --- |
| `GPTQConfig` | The post-AWQ default for 4-bit quantization in many open-source benchmarks. 22 fields, 7 post_init invariants. | `quantization_config.py:641-874` |
| `AwqConfig` | Activation-aware quant; common for Llama-3.x. 10 fields, 6 post_init invariants incl. CUDA compute-capability gate. | `quantization_config.py:878-1052` |
| `AqlmConfig` | Additive Quantization; surfacing for 4-bit Llama variants. 5 fields. | `quantization_config.py:1055-1112` |
| `VptqConfig` + `VptqLayerConfig` | Vector-PTQ; layer-by-layer config. | `quantization_config.py:1115-1207` |
| `QuantoConfig` | HF-native quanto. 3 fields, 2 enum gates (weights, activations). | `quantization_config.py:1211-1248` |
| `EetqConfig` | EETQ int8. 2 fields, 1 enum gate. | `quantization_config.py:1252-1282` |
| `HqqConfig` | HQQ; uses an HQQ-side BaseQuantizeConfig pass-through. 6 fields. | `quantization_config.py:280-401` |
| `AutoRoundConfig` | Intel AutoRound. 4 fields, 2 post_init gates. | `quantization_config.py:211-276` |
| `CompressedTensorsConfig` | Used by Neural Magic / vLLM ports. 9 fields. | `quantization_config.py:1285-1465` |
| `FbgemmFp8Config` | Meta's FBGEMM FP8. 2 fields. | `quantization_config.py:1468-1495` |
| `HiggsConfig` | HIGGS quant. 6 fields, 4 post_init gates incl. divisibility. | `quantization_config.py:1499-1551` |
| `FPQuantConfig` | NVFP4/MXFP4 quant. 8 fields, 7 cross-field post_init gates. | `quantization_config.py:1555-1636` |
| `TorchAoConfig` | torchao integration. Fields + dynamic version-gated type check. | `quantization_config.py:1640-1874` |
| `BitNetQuantConfig` | BitNet. 5 fields, 2 enum gates. | `quantization_config.py:1878-1931` |
| `SpQRConfig` | SpQR. 5 fields, 4 identity gates (bits=3, beta1=16, beta2=16). | `quantization_config.py:1935-1994` |
| `FineGrainedFP8Config` | DeepSeek-style FP8. 3 fields, 3 cross-field gates. | `quantization_config.py:1998-2034` |
| `QuarkConfig` | AMD Quark; supports both legacy and 1.0+ export configs. | `quantization_config.py:2037-2075` |
| `Mxfp4Config` | MX FP4. 2 fields. | `quantization_config.py:2079-2110` |

**Why it matters:** Quantization is THE highest-leverage knob in
energy-per-token measurement. LLEM as a benchmarking-as-a-service product
must reject incoherent `quantization_config` kwargs at submission time;
the existing baseline can validate that the user picked BNB correctly,
but a user passing `GPTQConfig(bits=5)` slips straight through to a
runtime ValueError instead.

### 1.5 Watermarking configs

`WatermarkingConfig` and `SynthIDTextWatermarkingConfig` are referenced
from `GenerationConfig.watermarking_config` but not enumerated in the
baseline `$defs`. Ground truth adds both with their full field schemas.

- `WatermarkingConfig` (5 fields, 3 validate-time invariants):
  `greenlist_ratio` (0.0-1.0), `bias`, `hashing_key`, `seeding_scheme`
  (enum `lefthash`/`selfhash`), `context_width` (>=1).
- `SynthIDTextWatermarkingConfig` (7 fields, 1 validate-time invariant):
  `sampling_table_size` <= 2**24.

Source: `configuration_utils.py:1262-1426`.

### 1.6 Cache classes

The baseline lifts the `cache_implementation` string enum
(`{static, offloaded_static, sliding_window, hybrid, hybrid_chunked,
offloaded_hybrid, offloaded_hybrid_chunked, dynamic, dynamic_full,
offloaded, quantized}`) - correct, via the `ALL_CACHE_IMPLEMENTATIONS`
landmark.

Ground truth additionally enumerates the **13 concrete Cache classes**
that those strings dispatch to (with line citations into
`cache_utils.py`). The string enum and the class names are NOT 1:1: e.g.
`sliding_window` -> `SlidingWindowCache`, but the string `dynamic_full`
has no separate class (uses `DynamicCache(full=True)` semantics). This
mismatch matters for users instantiating cache classes directly via
`kwargs={"past_key_values": DynamicCache(...)}`.

### 1.7 Pipeline factory kwargs

The `pipeline()` factory at `transformers/pipelines/__init__.py:637` is
**entirely absent from the baseline**. Ground truth adds all 17 keyword
parameters. Notable:

- `pipeline()` default `dtype` is `"auto"`, but
  `PreTrainedModel.from_pretrained` default is `None`. A user moving from
  `pipeline(...)` to direct `AutoModel.from_pretrained(...)` will silently
  lose the auto-dtype detection. Ground truth flags this.
- `processor` is a new (multimodal) kwarg that the baseline omits.
- `device` + `device_map` are mutually exclusive (documented in the tip
  block at line 786-790). Baseline does not enforce.

### 1.8 Environment variables

The baseline does **not** enumerate env vars at all. Ground truth adds 32:

**Transformers-owned (read directly):**
- `TRANSFORMERS_CACHE` (deprecated -> HF_HOME)
- `PYTORCH_TRANSFORMERS_CACHE` (deprecated)
- `PYTORCH_PRETRAINED_BERT_CACHE` (deprecated)
- `HF_MODULES_CACHE` (where trust_remote_code modules cache)
- `HUGGINGFACE_CO_RESOLVE_ENDPOINT` (deprecated -> HF_ENDPOINT)
- `HF_ENDPOINT`
- `HUGGINGFACE_CO_STAGING`
- `TRANSFORMERS_VERBOSITY` (enum: debug/info/warning/error/critical/detail)
- `TRANSFORMERS_NO_ADVISORY_WARNINGS`
- `TRANSFORMERS_IS_CI` (suppresses telemetry)
- `TRUST_REMOTE_CODE` (legacy: only RAG / RealmRetriever / TransfoXL paths read it)
- `WORLD_SIZE`, `LOCAL_RANK` (consumed by `from_pretrained` for distributed-vs-auto warnings)
- `TORCH_HOME`, `XDG_CACHE_HOME` (used in default-cache resolution)

**huggingface_hub pass-through (transformers honours via import):**
- `HF_HOME`, `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, `HF_ASSETS_CACHE`,
  `HF_XET_CACHE`, `HF_TOKEN`, `HF_TOKEN_PATH`, `HF_STORED_TOKENS_PATH`,
  `HF_HUB_OFFLINE`, `HF_HUB_DISABLE_TELEMETRY`,
  `HF_HUB_DISABLE_IMPLICIT_TOKEN`, `HF_HUB_DISABLE_PROGRESS_BARS`,
  `HF_HUB_DISABLE_SYMLINKS_WARNING`, `HF_HUB_DISABLE_EXPERIMENTAL_WARNING`,
  `HF_HUB_DISABLE_XET`, `HF_HUB_ENABLE_HF_TRANSFER`,
  `HF_HUB_DOWNLOAD_TIMEOUT`, `HF_HUB_ETAG_TIMEOUT`,
  `HF_HUB_USER_AGENT_ORIGIN`, `HF_DEBUG`, `HF_TRANSFER_CONCURRENCY`.

**HIGH-VALUE CALLOUT - the `HF_HUB_OFFLINE` import-time binding:**
`transformers/utils/hub.py:81` binds `_is_offline_mode` to
`huggingface_hub.constants.HF_HUB_OFFLINE` **once at import**. Setting the
env var **after** transformers is imported has NO effect on
`is_offline_mode()`. The well-known `TRANSFORMERS_OFFLINE` env var that
LLEM users (and many docker pinned-image setups) still set is **effectively
a no-op in 4.57.3**: only `HF_HUB_OFFLINE` works, and it must be set
before `import transformers`. This is a foot-gun the baseline cannot
detect with introspection alone.

## 2. Invariant additions

Baseline = 37 invariants. Ground truth = ~115. Net additions = ~78.
Categories below.

### 2.1 GenerationConfig `validate()` invariants the baseline misses

The baseline captures the well-known greedy-strips (epsilon, eta, min_p,
top_p, top_k, typical_p, temperature) and the beam-strips (early_stopping,
length_penalty). Ground truth adds the rest:

| Invariant id | Severity | Why missed | Source |
| --- | --- | --- | --- |
| `generationconfig_dormant_output_logits_without_return_dict` | dormant | The baseline has the other three `extra_output_flags` (output_attentions, output_hidden_states, output_scores) but NOT `output_logits`. The validate body iterates `self.extra_output_flags` (line 645) which includes all four. | `configuration_utils.py:644-649` |
| `generationconfig_dormant_cache_config_set_when_use_cache_false` | dormant | Baseline has `cache_implementation` arm of the loop; misses `cache_config` and `return_legacy_cache` arms (line 637 iterates all three). | `configuration_utils.py:637` |
| `generationconfig_dormant_return_legacy_cache_set_when_use_cache_false` | dormant | Same loop, third tuple element. | `configuration_utils.py:637` |
| 9x `generationconfig_rejects_{logits_processor, stopping_criteria, prefix_allowed_tokens_fn, synced_gpus, assistant_model, streamer, negative_prompt_ids, negative_prompt_attention_mask, use_model_defaults}` | error | The validate body at line 653-668 rejects nine generate-only kwargs that bled into the config. Baseline has zero of these. **LLEM users routinely set `streamer` or `stopping_criteria` on the config dict for convenience; this is the validator that catches it.** | `configuration_utils.py:653-668` |
| `generationconfig_watermarking_config_rejects_nonmapping` | error | When `watermarking_config` is a non-dict non-BaseWatermarkingConfig value, `WatermarkingConfig(**watermarking_config)` raises a TypeError. Baseline catches this for integer specifically but mis-classifies it: ground truth pins the predicate to the WatermarkingConfig `**` unpack, not to a synthetic isinstance check. | `configuration_utils.py:382` |
| `generationconfig_watermarking_config_coerces_dict` | normalisation | Dict -> `WatermarkingConfig.from_dict()` is a silent normalisation. Baseline misses entirely. | `configuration_utils.py:385` |

### 2.2 WatermarkingConfig / SynthIDTextWatermarkingConfig

ZERO invariants on these in the baseline. Ground truth adds 4:

- `watermarking_seeding_scheme_allowlist` (selfhash / lefthash)
- `watermarking_greenlist_ratio_in_unit_interval` (0.0 <= x <= 1.0)
- `watermarking_context_width_positive` (>= 1)
- `synthid_sampling_table_size_max` (<= 2**24)

### 2.3 Quantization config invariants

Baseline has 9 BNB invariants (all type checks). Ground truth keeps those
and adds ~50 invariants across the other 18 quant classes:

**GPTQConfig** (9 added): bits allowlist [2,3,4,8]; group_size range;
damp_percent open unit interval; dataset string allowlist
[wikitext2, c4, c4-new]; ptb/ptb-new explicit rejection; dataset type
check; exllama_config requires version key; exllama version allowlist
[1,2]; modules_in_block_to_quantize requires optimum >= 1.15.0.

**AwqConfig** (7 added): backend allowlist [autoawq, llm-awq]; version
allowlist [gemm, gemv, exllama, ipex]; llm-awq requires CUDA or XPU;
llm-awq requires compute capability >= 8.0; do_fuse requires
fuse_max_seq_len; modules_to_fuse requires 7 keys when do_fuse=True;
AWQ exllama version allowlist.

**AqlmConfig** (5 added): int type checks for 4 size fields;
linear_weights_not_to_quantize None -> [] normalisation.

**VptqConfig** (2 added): enable_proxy_error must be False;
is_indice_packed must be True (per-layer).

**QuantoConfig** (2 added): weights enum [float8, int8, int4, int2];
activations enum [None, int8, float8].

**EetqConfig** (1 added): weights must be int8.

**HqqConfig** (2 added): axis allowlist [0, 1]; axis None -> 1
normalisation with info log.

**HiggsConfig** (4 added): bits [2,3,4]; p [1,2]; group_size
[64,128,256]; hadamard_size % group_size == 0.

**FPQuantConfig** (7 added): forward_dtype [mxfp4, nvfp4]; mxfp4
forward_method [abs_max, quest]; mxfp4 hadamard_group_size [32, 64, 128];
nvfp4 forward_method must be abs_max; nvfp4 hadamard_group_size
[16, 32, 64, 128]; backward_dtype must be bf16; transform_init
[hadamard, identity, gsr].

**TorchAoConfig** (3 added): torchao installed gate; quant_type type
check (str or AOBaseConfig for ao>0.9.0); str-only gate for ao<=0.9.0.

**BitNetQuantConfig** (2 added): linear_class [bitlinear, autobitlinear];
quantization_mode [online, offline].

**SpQRConfig** (5 added): bits type check; bits must be 3; beta1 must be
16; beta2 must be 16; shapes must be dict.

**FineGrainedFP8Config** (3 added): activation_scheme must be dynamic;
weight_block_size len 2; weight_block_size positive entries.

**AutoRoundConfig** (2 added): bits allowlist; group_size range.

### 2.4 `from_pretrained` pre-flight gates

Baseline has ZERO invariants on `from_pretrained` (it only covers
`GenerationConfig` and the BNB type-check shells). Ground truth adds 19:

| Invariant id | Severity | Why critical |
| --- | --- | --- |
| `from_pretrained_state_dict_vs_model_name_mutual_exclusion` | error | `state_dict` + `pretrained_model_name_or_path` or `state_dict` + `gguf_file` -> raise. |
| `from_pretrained_tp_size_requires_tp_plan` | error | LLEM tensor-parallel paths. |
| `from_pretrained_tp_plan_only_auto` | error | Custom tp_plan strings rejected. |
| `from_pretrained_tp_plan_vs_device_map_mutual_exclusion` | error | Common LLEM benchmark misconfiguration. |
| `from_pretrained_device_map_auto_with_world_size_warns` | warning | Distributed launch + device_map=auto. |
| `from_pretrained_use_auth_token_vs_token_mutual_exclusion` | error | Deprecation foot-gun. |
| `from_pretrained_use_auth_token_deprecation_warning` | warning | Same; warning side. |
| `from_pretrained_gguf_requires_accelerate` | error | accelerate peer-package gate. |
| `from_pretrained_meta_device_context_raises` | error | torch.set_default_device('meta') anti-pattern. |
| `from_pretrained_device_map_string_allowlist` | error | The four named values {auto, balanced, balanced_low_0, sequential}. Baseline does not enforce. |
| `from_pretrained_device_map_negative_int` | error | int < 0 rejected. |
| `from_pretrained_device_map_requires_accelerate` | error | accelerate peer-package gate. |
| `from_pretrained_device_map_vs_deepspeed_zero3` | error | DS-Z3 + device_map mutual exclusion. |
| `from_pretrained_load_in_bit_vs_quantization_config_mutual_exclusion` | error | The exact gate the baseline schema covers as a field but not as an invariant. |
| `from_pretrained_load_in_bit_deprecated_warning` | warning | load_in_8bit / load_in_4bit warning text. |
| `from_pretrained_quantization_vs_gguf_mutual_exclusion` | error | gguf_file + quantization_config -> raise. |
| `from_pretrained_gguf_disallows_disk_offload` | error | gguf + disk in device_map -> raise. |
| `from_pretrained_safetensors_format_metadata_allowlist` | error | format enum [pt, tf, flax, mlx]. |
| `from_pretrained_transformers_weights_must_be_safetensors` | error | `config.transformers_weights` enforced to be safetensors. |
| `from_pretrained_normalises_torch_dtype_to_dtype` | warning | torch_dtype + dtype both set -> dtype wins; warn. |

Plus five `normalisation` invariants for silently-popped no-op kwargs
(`resume_download`, `low_cpu_mem_usage`, `offload_state_dict`, `_fast_init`,
`mirror`).

### 2.5 `GenerationConfig.save_pretrained` invariants

Baseline does not cover the save-time path. Ground truth adds 4:

- `save_pretrained` runs `validate(strict=True)`: any dormant warning
  becomes a refusal-to-save error.
- `save_pretrained` use_auth_token deprecation warning.
- `save_pretrained` use_auth_token vs token mutual exclusion.
- `save_directory` must not be a file.

**Why this matters:** LLEM users often persist `GenerationConfig` post-run
for reproducibility. A config that LOADS fine (dormant warnings only) will
FAIL TO SAVE. This is the highest-leverage of all the missed invariants
because the failure mode is asymmetric (silently OK at runtime, broken at
artefact persistence).

### 2.6 `GenerationConfig.from_model_config` normalisation

`from_model_config` silently sets `return_dict_in_generate=True` when any
`output_*` flag was True. Baseline misses; ground truth adds one
normalisation invariant.

## 3. Things the baseline got RIGHT (sanity check)

- `cache_implementation` 11-value enum: correct.
- `compile_config` $ref + the CompileConfig 5-field schema: correct (via
  the #671 nested-dataclass walker).
- The 9 BNB type-check invariants: correct and well-pinned to setter
  qualname.
- The `early_stopping` allowlist {True, False, 'never'}: correct.
- The `max_new_tokens > 0` gate: correct.
- The `num_return_sequences > num_beams` and "greedy + num_return_sequences > 1"
  gates: correct, both captured.

## 4. Single correction

The baseline's `invariants.proposed.yaml` line 87-90 marks the
`compile_config` predicate as `'>': 0`. This is the dynamic miner's
artefact of seeing `compile_config: 42` work as a positive trigger; the
ACTUAL predicate is `not isinstance(self.compile_config, CompileConfig)`
(line 560). The error message
("You provided `compile_config` as an instance of `<class 'int'>`...") is
the same, so the validated case row at `invariants.validated.yaml:80-90`
passes; the predicate description in `invariants.proposed.yaml` is just
misleading. Ground truth's
`generationconfig_compile_config_type` uses
`predicate_kind: type_check` + `isinstance: CompileConfig`.

## 5. Net delta count

| Category | Baseline | Ground truth | Net adds |
| --- | --- | --- | --- |
| `engine_params` (schema) | 38 | ~56 | +18 |
| `sampling_params` (schema) | 58 | 65 + 31 resolved-from-untyped | +7 + 31 type resolutions |
| `quantization_configs` (schema) | 0 ($defs) | 19 | +19 |
| `watermarking_configs` (schema) | 0 ($defs) | 2 | +2 |
| `cache_classes` (schema) | 0 enumerated | 13 | +13 |
| `pipeline_kwargs` (schema) | 0 | 17 | +17 |
| `env_vars` (schema) | 0 | 32 | +32 |
| Invariants (validate gates) | 37 | ~115 | +78 |

## 6. Three concrete high-value additions worth fixing in the miner first

1. **`HF_HUB_OFFLINE` import-time binding + `TRANSFORMERS_OFFLINE` no-op.**
   A whole class of LLEM users are running benchmark suites with
   `TRANSFORMERS_OFFLINE=1` set, believing it gates the Hub. In 4.57.3 it
   doesn't. The miner currently cannot detect this because it's a binding
   artefact (`_is_offline_mode = ...` at import time, line 81), not a
   `validate()` predicate. Add an env-var corpus pass that names the
   binding semantics so LLEM can warn at submission.

2. **The 9 generate-only kwargs that GenerationConfig rejects**
   (`logits_processor`, `stopping_criteria`, `streamer`, etc.). These are
   user-friendly errors but the baseline has none of them. Adding them is
   one tight-loop walk of `configuration_utils.py:653-668`; mining them
   would catch a common LLEM user mistake (sticking `streamer` on the
   config dict) at submission, not after a 4-hour H100 run.

3. **Quantization-config coverage beyond BNB.** GPTQ and AWQ are the
   highest-traffic quant choices on Hugging Face today (per HF trending
   model browser). Lacking even the bits-allowlist gate
   (`bits in [2, 3, 4, 8]`) means LLEM cannot reject `GPTQConfig(bits=5)`
   at submission. The mining-substrate AST walker already handles the
   AST shape (`raise ValueError(...)` inside `post_init`); pointing it at
   the 18 additional QuantizationConfigMixin classes should be a one-day
   patch on the miner LANDMARKS tuple.
