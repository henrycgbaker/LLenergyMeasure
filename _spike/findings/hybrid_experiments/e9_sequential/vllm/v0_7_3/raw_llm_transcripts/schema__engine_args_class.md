# schema extraction transcript: engine_args_class

- chunk_description: vllm.EngineArgs dataclass (CLI args -> engine_params)
- expected_namespaces: ['engine_params']
- attempts: 1
- elapsed_sec: 231.81
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser extracting the parameter schema for the
vllm library, version 0.7.3.

You will be shown ONE CHUNK of source code. Extract the parameter
schema for the fields visible in this chunk. Other chunks are mined
separately and merged later - do not include fields from outside
this chunk.

OUTPUT FORMAT: a single JSON object matching EXACTLY this shape:

{
  "engine": "vllm",
  "engine_version": "0.7.3",
  "chunk_name": "engine_args_class",
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

=== SOURCE: vllm.EngineArgs (dataclass field list) ===
class EngineArgs:
    """Arguments for vLLM engine."""
    model: str = 'facebook/opt-125m'
    served_model_name: Optional[Union[str, List[str]]] = None
    tokenizer: Optional[str] = None
    task: TaskOption = "auto"
    skip_tokenizer_init: bool = False
    tokenizer_mode: str = 'auto'
    trust_remote_code: bool = False
    allowed_local_media_path: str = ""
    download_dir: Optional[str] = None
    load_format: str = 'auto'
    config_format: ConfigFormat = ConfigFormat.AUTO
    dtype: str = 'auto'
    kv_cache_dtype: str = 'auto'
    seed: int = 0
    max_model_len: Optional[int] = None
    # Note: Specifying a custom executor backend by passing a class
    # is intended for expert use only. The API may change without
    # notice.
    distributed_executor_backend: Optional[Union[str,
                                                 Type[ExecutorBase]]] = None
    # number of P/D disaggregation (or other disaggregation) workers
    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 1
    max_parallel_loading_workers: Optional[int] = None
    block_size: Optional[int] = None
    enable_prefix_caching: Optional[bool] = None
    disable_sliding_window: bool = False
    use_v2_block_manager: bool = True
    swap_space: float = 4  # GiB
    cpu_offload_gb: float = 0  # GiB
    gpu_memory_utilization: float = 0.90
    max_num_batched_tokens: Optional[int] = None
    max_num_partial_prefills: Optional[int] = 1
    max_long_partial_prefills: Optional[int] = 1
    long_prefill_token_threshold: Optional[int] = 0
    max_num_seqs: Optional[int] = None
    max_logprobs: int = 20  # Default value for OpenAI Chat Completions API
    disable_log_stats: bool = False
    revision: Optional[str] = None
    code_revision: Optional[str] = None
    rope_scaling: Optional[Dict[str, Any]] = None
    rope_theta: Optional[float] = None
    hf_overrides: Optional[HfOverrides] = None
    tokenizer_revision: Optional[str] = None
    quantization: Optional[str] = None
    enforce_eager: Optional[bool] = None
    max_seq_len_to_capture: int = 8192
    disable_custom_all_reduce: bool = False
    tokenizer_pool_size: int = 0
    # Note: Specifying a tokenizer pool by passing a class
    # is intended for expert use only. The API may change without
    # notice.
    tokenizer_pool_type: Union[str, Type["BaseTokenizerGroup"]] = "ray"
    tokenizer_pool_extra_config: Optional[Dict[str, Any]] = None
    limit_mm_per_prompt: Optional[Mapping[str, int]] = None
    mm_processor_kwargs: Optional[Dict[str, Any]] = None
    disable_mm_preprocessor_cache: bool = False
    enable_lora: bool = False
    enable_lora_bias: bool = False
    max_loras: int = 1
    max_lora_rank: int = 16
    enable_prompt_adapter: bool = False
    max_prompt_adapters: int = 1
    max_prompt_adapter_token: int = 0
    fully_sharded_loras: bool = False
    lora_extra_vocab_size: int = 256
    long_lora_scaling_factors: Optional[Tuple[float]] = None
    lora_dtype: Optional[Union[str, torch.dtype]] = 'auto'
    max_cpu_loras: Optional[int] = None
    device: str = 'auto'
    num_scheduler_steps: int = 1
    multi_step_stream_outputs: bool = True
    ray_workers_use_nsight: bool = False
    num_gpu_blocks_override: Optional[int] = None
    num_lookahead_slots: int = 0
    model_loader_extra_config: Optional[dict] = None
    ignore_patterns: Optional[Union[str, List[str]]] = None
    preemption_mode: Optional[str] = None

    scheduler_delay_factor: float = 0.0
    enable_chunked_prefill: Optional[bool] = None

    guided_decoding_backend: str = 'xgrammar'
    logits_processor_pattern: Optional[str] = None
    # Speculative decoding configuration.
    speculative_model: Optional[str] = None
    speculative_model_quantization: Optional[str] = None
    speculative_draft_tensor_parallel_size: Optional[int] = None
    num_speculative_tokens: Optional[int] = None

...<truncated>...
```

### Response

```
{
  "engine": "vllm",
  "engine_version": "0.7.3",
  "chunk_name": "engine_args_class",
  "chunk_fields": {
    "model": {"namespace": "engine_params", "type": "string", "default": "facebook/opt-125m"},
    "served_model_name": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}, {"type": "null"}], "default": null},
    "tokenizer": {"namespace": "engine_params", "type": "string", "default": null},
    "task": {"namespace": "engine_params", "type": "string", "enum": ["auto"]},
    "skip_tokenizer_init": {"namespace": "engine_params", "type": "boolean", "default": false},
    "tokenizer_mode": {"namespace": "engine_params", "type": "string", "default": "auto"},
    "trust_remote_code": {"namespace": "engine_params", "type": "boolean", "default": false},
    "allowed_local_media_path": {"namespace": "engine_params", "type": "string", "default": ""},
    "download_dir": {"namespace": "engine_params", "type": "string", "default": null},
    "load_format": {"namespace": "engine_params", "type": "string", "default": "auto"},
    "config_format": {"namespace": "engine_params", "enum": ["AUTO"]},
    "dtype": {"namespace": "engine_params", "type": "string", "default": "auto"},
    "kv_cache_dtype": {"namespace": "engine_params", "type": "string", "default": "auto"},
    "seed": {"namespace": "engine_params", "type": "integer", "default": 0},
    "max_model_len": {"namespace": "engine_params", "type": "integer", "default": null},
    "distributed_executor_backend": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "object", "description": "ExecutorBase class"}, {"type": "null"}], "default": null},
    "pipeline_parallel_size": {"namespace": "engine_params", "type": "integer", "default": 1},
    "tensor_parallel_size": {"namespace": "engine_params", "type": "integer", "default": 1},
    "max_parallel_loading_workers": {"namespace": "engine_params", "type": "integer", "default": null},
    "block_size": {"namespace": "engine_params", "type": "integer", "default": null},
    "enable_prefix_caching": {"namespace": "engine_params", "type": "boolean", "default": null},
    "disable_sliding_window": {"namespace": "engine_params", "type": "boolean", "default": false},
    "use_v2_block_manager": {"namespace": "engine_params", "type": "boolean", "default": true},
    "swap_space": {"namespace": "engine_params", "type": "number", "default": 4.0},
    "cpu_offload_gb": {"namespace": "engine_params", "type": "number", "default": 0.0},
    "gpu_memory_utilization": {"namespace": "engine_params", "type": "number", "default": 0.9},
    "max_num_batched_tokens": {"namespace": "engine_params", "type": "integer", "default": null},
    "max_num_partial_prefills": {"namespace": "engine_params", "type": "integer", "default": 1},
    "max_long_partial_prefills": {"namespace": "engine_params", "type": "integer", "default": 1},
    "long_prefill_token_threshold": {"namespace": "engine_params", "type": "integer", "default": 0},
    "max_num_seqs": {"namespace": "engine_params", "type": "integer", "default": null},
    "max_logprobs": {"namespace": "engine_params", "type": "integer", "default": 20},
    "disable_log_stats": {"namespace": "engine_params", "type": "boolean", "default": false},
    "revision": {"namespace": "engine_params", "type": "string", "default": null},
    "code_revision": {"namespace": "engine_params", "type": "string", "default": null},
    "rope_scaling": {"namespace": "engine_params", "anyOf": [{"type": "object"}, {"type": "null"}], "default": null},
    "rope_theta": {"namespace": "engine_params", "type": "number", "default": null},
    "hf_overrides": {"namespace": "engine_params", "anyOf": [{"type": "object"}, {"type": "null"}], "default": null},
    "tokenizer_revision": {"namespace": "engine_params", "type": "string", "default": null},
    "quantization": {"namespace": "engine_params", "type": "string", "default": null},
    "enforce_eager": {"namespace": "engine_params", "type": "boolean", "default": null},
    "max_seq_len_to_capture": {"namespace": "engine_params", "type": "integer", "default": 8192},
    "disable_custom_all_reduce": {"namespace": "engine_params", "type": "boolean", "default": false},
    "tokenizer_pool_size": {"namespace": "engine_params", "type": "integer", "default": 0},
    "tokenizer_pool_type": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "object", "description": "BaseTokenizerGroup class"}], "enum": ["ray"]},
    "tokenizer_pool_extra_config": {"namespace": "engine_params", "anyOf": [{"type": "object"}, {"type": "null"}], "default": null},
    "limit_mm_per_prompt": {"namespace": "engine_params", "anyOf": [{"type": "object"}, {"type": "null"}], "default": null},
    "mm_processor_kwargs": {"namespace": "engine_params", "anyOf": [{"type": "object"}, {"type": "null"}], "default": null},
    "disable_mm_preprocessor_cache": {"namespace": "engine_params", "type": "boolean", "default": false},
    "enable_lora": {"namespace": "engine_params", "type": "boolean", "default": false},
    "enable_lora_bias": {"namespace": "engine_params", "type": "boolean", "default": false},
    "max_loras": {"namespace": "engine_params", "type": "integer", "default": 1},
    "max_lora_rank": {"namespace": "engine_params", "type": "integer", "default": 16},
    "enable_prompt_adapter": {"namespace": "engine_params", "type": "boolean", "default": false},
    "max_prompt_adapters": {"namespace": "engine_params", "type": "integer", "default": 1},
    "max_prompt_adapter_token": {"namespace": "engine_params", "type": "integer", "default": 0},
    "fully_sharded_loras": {"namespace": "engine_params", "type": "boolean", "default": false},
    "lora_extra_vocab_size": {"namespace": "engine_params", "type": "integer", "default": 256},
    "long_lora_scaling_factors": {"namespace": "engine_params", "anyOf": [{"type": "array", "items": {"type": "number"}}, {"type": "null"}], "default": null},
    "lora_dtype": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "object", "description": "torch.dtype"}, {"type": "null"}], "enum": ["auto"]},
    "max_cpu_loras": {"namespace": "engine_params", "type": "integer", "default": null},
    "device": {"namespace": "engine_params", "type": "string", "default": "auto"},
    "num_scheduler_steps": {"namespace": "engine_params", "type": "integer", "default": 1},
    "multi_step_stream_outputs": {"namespace": "engine_params", "type": "boolean", "default": true},
    "ray_workers_use_nsight": {"namespace": "engine_params", "type": "boolean", "default": false},
    "num_gpu_blocks_override": {"namespace": "engine_params", "type": "integer", "default": null},
    "num_lookahead_slots": {"namespace": "engine_params", "type": "integer", "default": 0},
    "model_loader_extra_config": {"namespace": "engine_params", "anyOf": [{"type": "object"}, {"type": "null"}], "default": null},
    "ignore_patterns": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}, {"type": "null"}], "default": null},
    "preemption_mode": {"namespace": "engine_params", "type": "string", "default": null},
    "scheduler_delay_factor": {"namespace": "engine_params", "type": "number", "default": 0.0},
    "enable_chunked_prefill": {"namespace": "engine_params", "type": "boolean", "default": null},
    "guided_decoding_backend": {"namespace": "engine_params", "type": "string", "enum": ["xgrammar"]},
    "logits_processor_pattern": {"namespace": "engine_params", "type": "string", "default": null},
    "speculative_model": {"namespace": "engine_params", "type": "string", "default": null},
    "speculative_model_quantization": {"namespace": "engine_params", "type": "string", "default": null},
    "speculative_draft_tensor_parallel_size": {"namespace": "engine_params", "type": "integer", "default": null},
    "num_speculative_tokens": {"namespace": "engine_params", "type": "integer", "default"
...<truncated>...
```
