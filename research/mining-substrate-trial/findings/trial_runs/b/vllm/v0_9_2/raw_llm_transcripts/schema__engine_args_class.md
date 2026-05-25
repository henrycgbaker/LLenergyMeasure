# schema extraction transcript: engine_args_class

- chunk_description: vllm.EngineArgs dataclass (CLI args -> engine_params)
- expected_namespaces: ['engine_params']
- attempts: 1
- elapsed_sec: 476.92
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
    model: str = ModelConfig.model
    served_model_name: Optional[Union[
        str, List[str]]] = ModelConfig.served_model_name
    tokenizer: Optional[str] = ModelConfig.tokenizer
    hf_config_path: Optional[str] = ModelConfig.hf_config_path
    task: TaskOption = ModelConfig.task
    skip_tokenizer_init: bool = ModelConfig.skip_tokenizer_init
    enable_prompt_embeds: bool = ModelConfig.enable_prompt_embeds
    tokenizer_mode: TokenizerMode = ModelConfig.tokenizer_mode
    trust_remote_code: bool = ModelConfig.trust_remote_code
    allowed_local_media_path: str = ModelConfig.allowed_local_media_path
    download_dir: Optional[str] = LoadConfig.download_dir
    load_format: str = LoadConfig.load_format
    config_format: str = ModelConfig.config_format
    dtype: ModelDType = ModelConfig.dtype
    kv_cache_dtype: CacheDType = CacheConfig.cache_dtype
    seed: Optional[int] = ModelConfig.seed
    max_model_len: Optional[int] = ModelConfig.max_model_len
    cuda_graph_sizes: list[int] = get_field(SchedulerConfig,
                                            "cuda_graph_sizes")
    # Note: Specifying a custom executor backend by passing a class
    # is intended for expert use only. The API may change without
    # notice.
    distributed_executor_backend: Optional[Union[
        DistributedExecutorBackend,
        Type[ExecutorBase]]] = ParallelConfig.distributed_executor_backend
    # number of P/D disaggregation (or other disaggregation) workers
    pipeline_parallel_size: int = ParallelConfig.pipeline_parallel_size
    tensor_parallel_size: int = ParallelConfig.tensor_parallel_size
    data_parallel_size: int = ParallelConfig.data_parallel_size
    data_parallel_rank: Optional[int] = None
    data_parallel_size_local: Optional[int] = None
    data_parallel_address: Optional[str] = None
    data_parallel_rpc_port: Optional[int] = None
    data_parallel_backend: str = ParallelConfig.data_parallel_backend
    enable_expert_parallel: bool = ParallelConfig.enable_expert_parallel
    enable_eplb: bool = ParallelConfig.enable_eplb
    num_redundant_experts: int = ParallelConfig.num_redundant_experts
    eplb_window_size: int = ParallelConfig.eplb_window_size
    eplb_step_interval: int = ParallelConfig.eplb_step_interval
    eplb_log_balancedness: bool = ParallelConfig.eplb_log_balancedness
    max_parallel_loading_workers: Optional[
        int] = ParallelConfig.max_parallel_loading_workers
    block_size: Optional[BlockSize] = CacheConfig.block_size
    enable_prefix_caching: Optional[bool] = CacheConfig.enable_prefix_caching
    prefix_caching_hash_algo: PrefixCachingHashAlgo = \
        CacheConfig.prefix_caching_hash_algo
    disable_sliding_window: bool = ModelConfig.disable_sliding_window
    disable_cascade_attn: bool = ModelConfig.disable_cascade_attn
    use_v2_block_manager: bool = True
    swap_space: float = CacheConfig.swap_space
    cpu_offload_gb: float = CacheConfig.cpu_offload_gb
    gpu_memory_utilization: float = CacheConfig.gpu_memory_utilization
    max_num_batched_tokens: Optional[
        int] = SchedulerConfig.max_num_batched_tokens
    max_num_partial_prefills: int = SchedulerConfig.max_num_partial_prefills
    max_long_partial_prefills: int = SchedulerConfig.max_long_partial_prefills
    long_prefill_token_threshold: int = \
        SchedulerConfig.long_prefill_token_threshold
    max_num_seqs: Optional[int] = SchedulerConfig.max_num_seqs
    max_logprobs: int = ModelConfig.max_logprobs
    disable_log_stats: bool = False
    revision: Optional[str] = ModelConfig.revision
    code_revision: Optional[str] = ModelConfig.code_revision
    rope_scaling: dict[str, Any] = get_field(ModelConfig, "rope_scaling")
    rope_theta: Optional[float] = ModelConfig.rope_theta
    hf_token: Optional[Union[bool, str]] = ModelConfig.hf_token
    hf_o
...<truncated>...
```

### Response

```
{
  "engine": "vllm",
  "engine_version": "0.9.2",
  "chunk_name": "engine_args_class",
  "chunk_fields": {
    "model": {"namespace": "engine_params", "type": "string"},
    "served_model_name": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "array"}], "default": null},
    "tokenizer": {"namespace": "engine_params", "type": "string", "default": null},
    "hf_config_path": {"namespace": "engine_params", "type": "string", "default": null},
    "task": {"namespace": "engine_params"},
    "skip_tokenizer_init": {"namespace": "engine_params", "type": "boolean"},
    "enable_prompt_embeds": {"namespace": "engine_params", "type": "boolean"},
    "tokenizer_mode": {"namespace": "engine_params"},
    "trust_remote_code": {"namespace": "engine_params", "type": "boolean"},
    "allowed_local_media_path": {"namespace": "engine_params", "type": "string"},
    "download_dir": {"namespace": "engine_params", "type": "string", "default": null},
    "load_format": {"namespace": "engine_params", "type": "string"},
    "config_format": {"namespace": "engine_params", "type": "string"},
    "dtype": {"namespace": "engine_params"},
    "kv_cache_dtype": {"namespace": "engine_params"},
    "seed": {"namespace": "engine_params", "type": "integer", "default": null},
    "max_model_len": {"namespace": "engine_params", "type": "integer", "default": null},
    "cuda_graph_sizes": {"namespace": "engine_params", "type": "array"},
    "distributed_executor_backend": {"namespace": "engine_params", "anyOf": [{"type": "object"}, {"type": "null"}], "default": null},
    "pipeline_parallel_size": {"namespace": "engine_params", "type": "integer"},
    "tensor_parallel_size": {"namespace": "engine_params", "type": "integer"},
    "data_parallel_size": {"namespace": "engine_params", "type": "integer"},
    "data_parallel_rank": {"namespace": "engine_params", "type": "integer", "default": null},
    "data_parallel_size_local": {"namespace": "engine_params", "type": "integer", "default": null},
    "data_parallel_address": {"namespace": "engine_params", "type": "string", "default": null},
    "data_parallel_rpc_port": {"namespace": "engine_params", "type": "integer", "default": null},
    "data_parallel_backend": {"namespace": "engine_params", "type": "string"},
    "enable_expert_parallel": {"namespace": "engine_params", "type": "boolean"},
    "enable_eplb": {"namespace": "engine_params", "type": "boolean"},
    "num_redundant_experts": {"namespace": "engine_params", "type": "integer"},
    "eplb_window_size": {"namespace": "engine_params", "type": "integer"},
    "eplb_step_interval": {"namespace": "engine_params", "type": "integer"},
    "eplb_log_balancedness": {"namespace": "engine_params", "type": "boolean"},
    "max_parallel_loading_workers": {"namespace": "engine_params", "type": "integer", "default": null},
    "block_size": {"namespace": "engine_params", "anyOf": [{"type": "object"}, {"type": "null"}], "default": null},
    "enable_prefix_caching": {"namespace": "engine_params", "type": "boolean", "default": null},
    "prefix_caching_hash_algo": {"namespace": "engine_params"},
    "disable_sliding_window": {"namespace": "engine_params", "type": "boolean"},
    "disable_cascade_attn": {"namespace": "engine_params", "type": "boolean"},
    "use_v2_block_manager": {"namespace": "engine_params", "type": "boolean"},
    "swap_space": {"namespace": "engine_params", "type": "number"},
    "cpu_offload_gb": {"namespace": "engine_params", "type": "number"},
    "gpu_memory_utilization": {"namespace": "engine_params", "type": "number"},
    "max_num_batched_tokens": {"namespace": "engine_params", "type": "integer", "default": null},
    "max_num_partial_prefills": {"namespace": "engine_params", "type": "integer"},
    "max_long_partial_prefills": {"namespace": "engine_params", "type": "integer"},
    "long_prefill_token_threshold": {"namespace": "engine_params", "type": "integer"},
    "max_num_seqs": {"namespace": "engine_params", "type": "integer", "default": null},
    "max_logprobs": {"namespace": "engine_params", "type": "integer"},
    "disable_log_stats": {"namespace": "engine_params", "type": "boolean"},
    "revision": {"namespace": "engine_params", "type": "string", "default": null},
    "code_revision": {"namespace": "engine_params", "type": "string", "default": null},
    "rope_scaling": {"namespace": "engine_params", "type": "object"},
    "rope_theta": {"namespace": "engine_params", "type": "number", "default": null},
    "hf_token": {"namespace": "engine_params", "anyOf": [{"type": "boolean"}, {"type": "string"}], "default": null},
    "hf_overrides": {"namespace": "engine_params"},
    "tokenizer_revision": {"namespace": "engine_params", "type": "string", "default": null},
    "quantization": {"namespace": "engine_params", "anyOf": [{"type": "object"}, {"type": "null"}], "default": null},
    "enforce_eager": {"namespace": "engine_params", "type": "boolean"},
    "max_seq_len_to_capture": {"namespace": "engine_params", "type": "integer"},
    "disable_custom_all_reduce": {"namespace": "engine_params", "type": "boolean"},
    "tokenizer_pool_size": {"namespace": "engine_params", "type": "integer"},
    "tokenizer_pool_type": {"namespace": "engine_params", "type": "string"},
    "tokenizer_pool_extra_config": {"namespace": "engine_params", "type": "object"},
    "limit_mm_per_prompt": {"namespace": "engine_params", "type": "object"},
    "media_io_kwargs": {"namespace": "engine_params", "type": "object"},
    "mm_processor_kwargs": {"namespace": "engine_params", "anyOf": [{"type": "object"}, {"type": "null"}], "default": null},
    "disable_mm_preprocessor_cache": {"namespace": "engine_params", "type": "boolean"},
    "enable_lora": {"namespace": "engine_params", "type": "boolean"},
    "enable_lora_bias": {"namespace": "engine_params", "type": "boolean"},
    "max_loras": {"namespace": "engine_params", "type": "integer"},
    "max_lora_rank": {"namespace": "engine_params", "type": "integer"},
    "fully_sharded_loras": {"namespace": "engine_params", "type": "boolean"},
    "max_cpu_loras": {"namespace": "engine_params", "type": "integer", "default": null},
    "lora_dtype": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "object"}], "default": null},
    "lora_extra_vocab_size": {"namespace": "engine_params", "type": "integer"},
    "long_lora_scaling_factors": {"namespace": "engine_params", "anyOf": [{"type": "array"}, {"type": "null"}], "default": null},
    "enable_prompt_adapter": {"namespace": "engine_params", "type": "boolean"},
    "max_prompt_adapters": {"namespace": "engine_params", "type": "integer"},
    "max_prompt_adapter_token": {"namespace": "engine_params", "type": "integer"},
    "device": {"namespace": "engine_params"},
    "num_scheduler_steps": {"namespace": "engine_params", "type": "integer"},
    "multi_step_stream_outputs": {"namespace": "engine_params", "type": "boolean"},
    "ray_workers_use_nsight": {"namespace": "engine_params", "type": "boolean"},
    "num_gpu_blocks_override": {"namespace": "engine_params", "type": "integer", "default": null},
    "num_lookahead_slots": {"namespace": "engine_params", "type": "integer"},
    "model_loader_extra_config": {"namespace": "engine_params", "type": "object"},
    "ignore_patterns": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "array"}], "default": null},
    "preemption_mode": {"namespace": "engine_params", "type": "string", "default": null},
    "scheduler_delay_factor": {"namespace": "engine_params", "type": "number"},
    "enable_chunked_prefill": {"namespace": "engine_params", "type": "boolean", "default": null},
    "disable_chunked_mm_input": {"namespace": "engine_params", "type": "boolean"},
    "disable_hybrid_kv_cache_manager": {"namespace": "engine_params", "type": "boolean"},
    "guided_decoding_backend": {"namespace": "engine_params"},
    "guided_decoding_disable_fallback": {"namespace": "engine_params", "type": "b
...<truncated>...
```
