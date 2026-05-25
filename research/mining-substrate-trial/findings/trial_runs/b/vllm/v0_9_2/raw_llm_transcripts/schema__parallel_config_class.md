# schema extraction transcript: parallel_config_class

- chunk_description: vllm.ParallelConfig dataclass (distributed engine_params)
- expected_namespaces: ['engine_params']
- attempts: 1
- elapsed_sec: 181.41
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
  "chunk_name": "parallel_config_class",
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

=== SOURCE: vllm.ParallelConfig (dataclass field list) ===
class ParallelConfig:
    """Configuration for the distributed execution."""

    pipeline_parallel_size: int = 1
    """Number of pipeline parallel groups."""
    tensor_parallel_size: int = 1
    """Number of tensor parallel groups."""
    data_parallel_size: int = 1
    """Number of data parallel groups. MoE layers will be sharded according to
    the product of the tensor parallel size and data parallel size."""
    data_parallel_size_local: int = 1
    """Number of local data parallel groups."""
    data_parallel_rank: int = 0
    """Rank of the data parallel group."""
    data_parallel_rank_local: Optional[int] = None
    """Local rank of the data parallel group,
    set only in SPMD mode."""
    data_parallel_master_ip: str = "127.0.0.1"
    """IP of the data parallel master."""
    data_parallel_rpc_port: int = 29550
    """Port for data parallel messaging."""
    data_parallel_master_port: int = 29500
    """Port of the data parallel master."""
    data_parallel_backend: str = "mp"
    """Backend to use for data parallel, either "mp" or "ray"."""
    data_parallel_external_lb: bool = False
    """Whether to use "external" DP LB mode. Applies only to online serving
    and when data_parallel_size > 0. Set implicitly when
    data_parallel_rank is provided explicitly to vllm serve."""
    enable_expert_parallel: bool = False
    """Use expert parallelism instead of tensor parallelism for MoE layers."""
    enable_eplb: bool = False
    """Enable expert parallelism load balancing for MoE layers."""
    num_redundant_experts: int = 0
    """Number of redundant experts to use for expert parallelism."""
    eplb_window_size: int = 1000
    """Window size for expert load recording."""
    eplb_step_interval: int = 3000
    """
    Interval for rearranging experts in expert parallelism.

    Note that if this is greater than the EPLB window size, only the metrics
    of the last `eplb_window_size` steps will be used for rearranging experts.
    """
    eplb_log_balancedness: bool = False
    """
    Log the balancedness each step of expert parallelism.
    This is turned off by default since it will cause communication overhead.
    """

    max_parallel_loading_workers: Optional[int] = None
    """Maximum number of parallel loading workers when loading model
    sequentially in multiple batches. To avoid RAM OOM when using tensor
    parallel and large models."""

    disable_custom_all_reduce: bool = False
    """Disable the custom all-reduce kernel and fall back to NCCL."""

    tokenizer_pool_config: Optional[TokenizerPoolConfig] = None
    """This parameter is deprecated and will be removed in a future release.
    Please remove it from your configs"""

    ray_workers_use_nsight: bool = False
    """Whether to profile Ray workers with nsight, see https://docs.ray.io/en/latest/ray-observability/user-guides/profiling.html#profiling-nsight-profiler."""

    placement_group: Optional["PlacementGroup"] = None
    """ray distributed model workers placement group."""

    distributed_executor_backend: Optional[Union[DistributedExecutorBackend,
                                                 type["ExecutorBase"]]] = None
    """Backend to use for distributed model
    workers, either "ray" or "mp" (multiprocessing). If the product
    of pipeline_parallel_size and tensor_parallel_size is less than
    or equal to the number of GPUs available, "mp" will be used to
    keep processing on a single host. Otherwise, this will default
    to "ray" if Ray is installed and fail otherwise. Note that tpu
    and hpu only support Ray for distributed inference."""

    worker_cls: str = "auto"
    """The full name of the worker class to use. If "auto", the worker class
    will be determined based on the platform."""
    sd_worker_cls: str = "auto"
    """The full name of the worker class to use for speculative de
...<truncated>...
```

### Response

```
{
  "engine": "vllm",
  "engine_version": "0.9.2",
  "chunk_name": "parallel_config_class",
  "chunk_fields": {
    "pipeline_parallel_size": {"namespace": "engine_params", "type": "integer", "default": 1, "description": "Number of pipeline parallel groups."},
    "tensor_parallel_size": {"namespace": "engine_params", "type": "integer", "default": 1, "description": "Number of tensor parallel groups."},
    "data_parallel_size": {"namespace": "engine_params", "type": "integer", "default": 1, "description": "Number of data parallel groups. MoE layers will be sharded according to the product of the tensor parallel size and data parallel size."},
    "data_parallel_size_local": {"namespace": "engine_params", "type": "integer", "default": 1, "description": "Number of local data parallel groups."},
    "data_parallel_rank": {"namespace": "engine_params", "type": "integer", "default": 0, "description": "Rank of the data parallel group."},
    "data_parallel_rank_local": {"namespace": "engine_params", "anyOf": [{"type": "integer"}, {"type": "null"}], "default": null, "description": "Local rank of the data parallel group, set only in SPMD mode."},
    "data_parallel_master_ip": {"namespace": "engine_params", "type": "string", "default": "127.0.0.1", "description": "IP of the data parallel master."},
    "data_parallel_rpc_port": {"namespace": "engine_params", "type": "integer", "default": 29550, "description": "Port for data parallel messaging."},
    "data_parallel_master_port": {"namespace": "engine_params", "type": "integer", "default": 29500, "description": "Port of the data parallel master."},
    "data_parallel_backend": {"namespace": "engine_params", "type": "string", "default": "mp", "enum": ["mp", "ray"], "description": "Backend to use for data parallel, either \"mp\" or \"ray\"."},
    "data_parallel_external_lb": {"namespace": "engine_params", "type": "boolean", "default": false, "description": "Whether to use \"external\" DP LB mode. Applies only to online serving and when data_parallel_size > 0. Set implicitly when data_parallel_rank is provided explicitly to vllm serve."},
    "enable_expert_parallel": {"namespace": "engine_params", "type": "boolean", "default": false, "description": "Use expert parallelism instead of tensor parallelism for MoE layers."},
    "enable_eplb": {"namespace": "engine_params", "type": "boolean", "default": false, "description": "Enable expert parallelism load balancing for MoE layers."},
    "num_redundant_experts": {"namespace": "engine_params", "type": "integer", "default": 0, "description": "Number of redundant experts to use for expert parallelism."},
    "eplb_window_size": {"namespace": "engine_params", "type": "integer", "default": 1000, "description": "Window size for expert load recording."},
    "eplb_step_interval": {"namespace": "engine_params", "type": "integer", "default": 3000, "description": "Interval for rearranging experts in expert parallelism."},
    "eplb_log_balancedness": {"namespace": "engine_params", "type": "boolean", "default": false, "description": "Log the balancedness each step of expert parallelism. This is turned off by default since it will cause communication overhead."},
    "max_parallel_loading_workers": {"namespace": "engine_params", "anyOf": [{"type": "integer"}, {"type": "null"}], "default": null, "description": "Maximum number of parallel loading workers when loading model sequentially in multiple batches. To avoid RAM OOM when using tensor parallel and large models."},
    "disable_custom_all_reduce": {"namespace": "engine_params", "type": "boolean", "default": false, "description": "Disable the custom all-reduce kernel and fall back to NCCL."},
    "tokenizer_pool_config": {"namespace": "engine_params", "anyOf": [{"type": "object"}, {"type": "null"}], "default": null, "description": "This parameter is deprecated and will be removed in a future release. Please remove it from your configs"},
    "ray_workers_use_nsight": {"namespace": "engine_params", "type": "boolean", "default": false, "description": "Whether to profile Ray workers with nsight, see https://docs.ray.io/en/latest/ray-observability/user-guides/profiling.html#profiling-nsight-profiler."},
    "placement_group": {"namespace": "engine_params", "anyOf": [{"type": "object"}, {"type": "null"}], "default": null, "description": "ray distributed model workers placement group."},
    "distributed_executor_backend": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "object"}, {"type": "null"}], "enum": ["mp", "ray"], "default": null, "description": "Backend to use for distributed model workers, either \"ray\" or \"mp\" (multiprocessing). If the product of pipeline_parallel_size and tensor_parallel_size is less than or equal to the number of GPUs available, \"mp\" will be used to keep processing on a single host. Otherwise, this will default to \"ray\" if Ray is installed and fail otherwise. Note that tpu and hpu only support Ray for distributed inference."},
    "worker_cls": {"namespace": "engine_params", "type": "string", "default": "auto", "description": "The full name of the worker class to use. If \"auto\", the worker class will be determined based on the platform."},
    "sd_worker_cls": {"namespace": "engine_params", "type": "string", "default": "auto", "description": "The full name of the worker class to use for speculative decoding. If \"auto\", the worker class will be determined based on the platform."},
    "worker_extension_cls": {"namespace": "engine_params", "type": "string", "default": "", "description": "The full name of the worker extension class to use. The worker extension class is dynamically inherited by the worker class. This is used to inject new attributes and methods to the worker class for use in collective_rpc calls."},
    "rank": {"namespace": "engine_params", "type": "integer", "default": 0, "description": "Global rank in distributed setup."},
    "enable_multimodal_encoder_data_parallel": {"namespace": "engine_params", "type": "boolean", "default": false, "description": "Use data parallelism instead of tensor parallelism for vision encoder. Only support LLama4 for now"}
  }
}
```
