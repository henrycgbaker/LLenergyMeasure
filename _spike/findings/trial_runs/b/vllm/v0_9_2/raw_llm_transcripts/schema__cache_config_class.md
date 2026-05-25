# schema extraction transcript: cache_config_class

- chunk_description: vllm.CacheConfig dataclass (KV cache engine_params)
- expected_namespaces: ['engine_params']
- attempts: 1
- elapsed_sec: 144.58
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser extracting the parameter schema for the
vllm library, version 0.9.2.

You will be shown ONE CHUNK of source code. Extract the parameter
schema for the fields visible in this chunk. Other chunks are mined
separately and merged later - do not include fields from outside
this chunk.

OUTPUT FORMAT: a single JSON object matching EXACTLY this shape:

{
  "engine": "vllm",
  "engine_version": "0.9.2",
  "chunk_name": "cache_config_class",
  "chunk_fields": {
    "<field_name>": {
      "namespace": "<one of: engine_params, sampling_params, $defs.CompileConfig, ...>",
      "type": "<one of: string, integer, number, boolean, array, object, null>",
      "default": <value_or_null>,
      "description": "<brief one-liner>",
      "enum": [<values>],
      "anyOf": [{"type": "..."}, ...]
    }
  }
}

EXPECTED NAMESPACES FOR THIS CHUNK: engine_params
(Other namespaces are extracted from other chunks. If you see fields
that belong to other namespaces, ignore them in this chunk.)

CRITICAL RULES:
1. Return ONLY the JSON document. NO markdown code fences (no ```).
   NO commentary, no preamble, no postamble. The first character of
   your response must be `{`.
2. Extract ONLY fields VISIBLE in the source below. Do not invent or
   hallucinate fields. If a field is referenced but its source is not
   shown, omit it.
3. Skip internal-plumbing fields: any name starting with `_`
   (e.g. `_commit_hash`, `_from_auto`) and these explicit names:
   `adapter_kwargs`, `model_kwargs`, `torch_dtype`.
4. For fields with NO clear type annotation, OMIT the "type" key. Do
   NOT guess. Only set "type" when the source explicitly annotates.
5. For Optional[X] / Union[X, None] / X | None: set "type" to X and
   "default" to null if applicable. If multiple non-null types use
   "anyOf": [{"type": "X"}, {"type": "Y"}, {"type": "null"}].
6. For typing.Union[A, B] without None: use "anyOf" not "type".
7. For defaults that are None, use null. For defaults that are
   complex objects, use null (the schema only carries simple defaults).
8. For Sphinx-documented kwargs (pulled by name from a `kwargs.pop(...)`
   call) the docstring usually documents the type and default - read
   it and emit accordingly. If the docstring is shown, USE IT.

FEW-SHOT EXAMPLES (from transformers v4.57.3 reference catalogue):

Example 1 (engine_params, simple bool):
  Source: `force_download: bool = False,`
  Emit: `"force_download": {"namespace": "engine_params", "type": "boolean", "default": false}`

Example 2 (engine_params, Optional Union with PathLike):
  Source: `cache_dir: Optional[Union[str, os.PathLike]] = None,`
  Emit: `"cache_dir": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "object", "description": "PathLike"}, {"type": "null"}], "default": null}`

Example 3 (engine_params, BitsAndBytesConfig field with default):
  Source: `bnb_4bit_compute_dtype=None,` (in BitsAndBytesConfig.__init__)
  Emit: `"bnb_4bit_compute_dtype": {"namespace": "engine_params", "default": null, "description": "BitsAndBytesConfig quantisation field"}`

Example 4 (sampling_params with enum from validate()):
  Source: GenerationConfig docstring mentions `cache_implementation (str, *optional*)` and validate() checks `not in ALL_CACHE_IMPLEMENTATIONS`.
  Emit: `"cache_implementation": {"namespace": "sampling_params", "type": "string"}`

Example 5 ($defs entry for CompileConfig):
  Source: CompileConfig dataclass with `fullgraph: bool = True`
  Emit: `"fullgraph": {"namespace": "$defs.CompileConfig", "type": "boolean", "default": true}`

Example 6 (sampling_params, unannotated default-None - common in GenerationConfig.__init__):
  Source: `temperature = kwargs.pop("temperature", None)` and docstring `temperature (float, *optional*, defaults to 1.0)`
  Emit: `"temperature": {"namespace": "sampling_params", "type": "number", "default": 1.0}`

CRITICAL: every field name MUST appear in the chunk_fields object.
Do NOT nest by namespace at the top level - the "namespace" key
inside each field's object is the namespace marker.

=== SOURCE: vllm.CacheConfig (dataclass field list) ===
class CacheConfig:
    """Configuration for the KV cache."""

    block_size: SkipValidation[BlockSize] = None  # type: ignore
    """Size of a contiguous cache block in number of tokens. This is ignored on
    neuron devices and set to `--max-model-len`. On CUDA devices, only block
    sizes up to 32 are supported. On HPU devices, block size defaults to 128.

    This config has no static default. If left unspecified by the user, it will
    be set in `Platform.check_and_update_config()` based on the current
    platform."""
    gpu_memory_utilization: float = 0.9
    """The fraction of GPU memory to be used for the model executor, which can
    range from 0 to 1. For example, a value of 0.5 would imply 50% GPU memory
    utilization. If unspecified, will use the default value of 0.9. This is a
    per-instance limit, and only applies to the current vLLM instance. It does
    not matter if you have another vLLM instance running on the same GPU. For
    example, if you have two vLLM instances running on the same GPU, you can
    set the GPU memory utilization to 0.5 for each instance."""
    swap_space: float = 4
    """Size of the CPU swap space per GPU (in GiB)."""
    cache_dtype: CacheDType = "auto"
    """Data type for kv cache storage. If "auto", will use model data type.
    CUDA 11.8+ supports fp8 (=fp8_e4m3) and fp8_e5m2. ROCm (AMD GPU) supports
    fp8 (=fp8_e4m3)."""
    is_attention_free: bool = False
    """Whether the model is attention-free. This is primarily set in
    `ModelConfig` and that value should be manually duplicated here."""
    num_gpu_blocks_override: Optional[int] = None
    """Number of GPU blocks to use. This overrides the profiled `num_gpu_blocks`
    if specified. Does nothing if `None`. Used for testing preemption."""
    sliding_window: Optional[int] = None
    """Sliding window size for the KV cache. This is primarily set in
    `ModelConfig` and that value should be manually duplicated here."""
    enable_prefix_caching: Optional[bool] = None
    """Whether to enable prefix caching. Disabled by default for V0. Enabled by
    default for V1."""
    prefix_caching_hash_algo: PrefixCachingHashAlgo = "builtin"
    """Set the hash algorithm for prefix caching:\n
    - "builtin" is Python's built-in hash.\n
    - "sha256" is collision resistant but with certain overheads."""
    cpu_offload_gb: float = 0
    """The space in GiB to offload to CPU, per GPU. Default is 0, which means
    no offloading. Intuitively, this argument can be seen as a virtual way to
    increase the GPU memory size. For example, if you have one 24 GB GPU and
    set this to 10, virtually you can think of it as a 34 GB GPU. Then you can
    load a 13B model with BF16 weight, which requires at least 26GB GPU memory.
    Note that this requires fast CPU-GPU interconnect, as part of the model is
    loaded from CPU memory to GPU memory on the fly in each model forward pass.
    """
    calculate_kv_scales: bool = False
    """This enables dynamic calculation of `k_scale` and `v_scale` when
    kv_cache_dtype is fp8. If `False`, the scales will be loaded from the model
    checkpoint if available. Otherwise, the scales will default to 1.0."""
    cpu_kvcache_space_bytes: Optional[int] = None
    """(CPU backend only) CPU key-value cache space."""

    # Will be set after profiling.
    num_gpu_blocks: Optional[int] = field(default=None, init=False)
    """The number of blocks to allocate for GPU memory."""
    num_cpu_blocks: Optional[int] = field(default=None, init=False)
    """The number of blocks to allocate for CPU memory."""

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
     
...<truncated>...
```

### Response

```
{
  "engine": "vllm",
  "engine_version": "0.9.2",
  "chunk_name": "cache_config_class",
  "chunk_fields": {
    "block_size": {"namespace": "engine_params", "default": null, "description": "Size of a contiguous cache block in number of tokens"},
    "gpu_memory_utilization": {"namespace": "engine_params", "type": "number", "default": 0.9},
    "swap_space": {"namespace": "engine_params", "type": "number", "default": 4},
    "cache_dtype": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "null"}], "default": null, "description": "Data type for kv cache storage"},
    "is_attention_free": {"namespace": "engine_params", "type": "boolean", "default": false},
    "num_gpu_blocks_override": {"namespace": "engine_params", "anyOf": [{"type": "integer"}, {"type": "null"}], "default": null, "description": "Number of GPU blocks to use"},
    "sliding_window": {"namespace": "engine_params", "anyOf": [{"type": "integer"}, {"type": "null"}], "default": null},
    "enable_prefix_caching": {"namespace": "engine_params", "anyOf": [{"type": "boolean"}, {"type": "null"}], "default": null, "description": "Whether to enable prefix caching"},
    "prefix_caching_hash_algo": {"namespace": "engine_params", "type": "string", "default": "builtin", "enum": ["builtin", "sha256"]},
    "cpu_offload_gb": {"namespace": "engine_params", "type": "number", "default": 0, "description": "The space in GiB to offload to CPU, per GPU"},
    "calculate_kv_scales": {"namespace": "engine_params", "type": "boolean", "default": false},
    "cpu_kvcache_space_bytes": {"namespace": "engine_params", "anyOf": [{"type": "integer"}, {"type": "null"}], "default": null, "description": "(CPU backend only) CPU key-value cache space"}
  }
}
```
