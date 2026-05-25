# schema extraction transcript: scheduler_peft_configs

- chunk_description: tensorrt_llm.{SchedulerConfig, PeftCacheConfig, DynamicBatchConfig}
- expected_namespaces: ['engine_params', '$defs.SchedulerConfig', '$defs.PeftCacheConfig', '$defs.DynamicBatchConfig']
- attempts: 1
- elapsed_sec: 182.02
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser extracting the parameter schema for the
tensorrt library, version 0.19.0.

You will be shown ONE CHUNK of source code. Extract the parameter
schema for the fields visible in this chunk. Other chunks are mined
separately and merged later - do not include fields from outside
this chunk.

OUTPUT FORMAT: a single JSON object matching EXACTLY this shape:

{
  "engine": "tensorrt",
  "engine_version": "0.19.0",
  "chunk_name": "scheduler_peft_configs",
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

EXPECTED NAMESPACES FOR THIS CHUNK: engine_params, $defs.SchedulerConfig, $defs.PeftCacheConfig, $defs.DynamicBatchConfig
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

=== CONTEXT ===
Three more Pydantic nested configs. SchedulerConfig has StrEnum-typed fields (capacity_scheduler_policy, context_chunking_policy); these gate the value to one of the enum members.

=== SOURCE: SchedulerConfig ===
class SchedulerConfig(BaseModel, PybindMirror):
    capacity_scheduler_policy: CapacitySchedulerPolicy = Field(
        default=CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
        description="The capacity scheduler policy to use")

    context_chunking_policy: Optional[ContextChunkingPolicy] = Field(
        default=None, description="The context chunking policy to use")

    dynamic_batch_config: Optional[DynamicBatchConfig] = Field(
        default=None, description="The dynamic batch config to use")

    def _to_pybind(self):
        return _SchedulerConfig(
            capacity_scheduler_policy=self.capacity_scheduler_policy._to_pybind(
            ),
            context_chunking_policy=self.context_chunking_policy._to_pybind()
            if self.context_chunking_policy else None,
            dynamic_batch_config=self.dynamic_batch_config._to_pybind()
            if self.dynamic_batch_config else None)



=== SOURCE: PeftCacheConfig ===
class PeftCacheConfig(BaseModel, PybindMirror):
    """
    Configuration for the PEFT cache.
    """
    num_host_module_layer: int = Field(
        default=0,
        description=
        "number of max sized 1-layer 1-module adapterSize=1 sets of weights that can be stored in host cache"
    )
    num_device_module_layer: int = Field(
        default=0,
        description=
        "number of max sized 1-layer 1-module sets of weights that can be stored in host cache"
    )
    optimal_adapter_size: int = Field(
        default=
        8,  # There are tests to keep the default value consistent with the pybind default value
        description="optimal adapter size used to set page width")
    max_adapter_size: int = Field(
        default=64,
        description="max supported adapter size. Used to compute minimum")
    num_put_workers: int = Field(
        default=1,
        description=
        "number of worker threads used to put weights into host cache")
    num_ensure_workers: int = Field(
        default=1,
        description=
        "number of worker threads used to copy weights from host to device")
    num_copy_streams: int = Field(
        default=1,
        description="number of streams used to copy weights from host to device"
    )
    max_pages_per_block_host: int = Field(
        default=24,
        description="Number of cache pages per allocation block (host)")
    max_pages_per_block_device: int = Field(
        default=8,
        description="Number of cache pages per allocation block (device)")
    device_cache_percent: Optional[float] = Field(
        default=None,
        description="percent of memory after engine load to use for cache")
    host_cache_size: Optional[int] = Field(
        default=None, description="size in bytes to use for host cache")
    lora_prefetch_dir: Optional[str] = Field(
        default=None,
        description=
        "folder to store the LoRA weights we hope to load during engine initialization"
    )

    def _to_pybind(self):
        return _PeftCacheConfig(
            num_host_module_layer=self.num_host_module_layer,
            num_device_module_layer=self.num_device_module_layer,
            optimal_adapter_size=self.optimal_adapter_size,
            max_adapter_size=self.max_adapter_size,
            num_put_workers=self.num_put_workers,
            num_ensure_workers=self.num_ensure_workers,
            num_copy_streams=self.num_copy_streams,
            max_pages_per_block_host=self.max_pages_per_block_host,
            max_pages_per_block_device=self.max_pages_per_block_device,
            device_cache_percent=self.device_cache_percent,
       
...<truncated>...
```

### Response

```
{
  "engine": "tensorrt",
  "engine_version": "0.19.0",
  "chunk_name": "scheduler_peft_configs",
  "chunk_fields": {
    "capacity_scheduler_policy": {
      "namespace": "$defs.SchedulerConfig",
      "type": "string",
      "enum": ["GUARANTEED_NO_EVICT"],
      "description": "The capacity scheduler policy to use"
    },
    "context_chunking_policy": {
      "namespace": "$defs.SchedulerConfig",
      "anyOf": [{"type": "string"}, {"type": "null"}],
      "default": null,
      "description": "The context chunking policy to use"
    },
    "dynamic_batch_config": {
      "namespace": "$defs.SchedulerConfig",
      "anyOf": [{"$ref": "#/$defs/DynamicBatchConfig"}, {"type": "null"}],
      "default": null,
      "description": "The dynamic batch config to use"
    },
    "num_host_module_layer": {
      "namespace": "$defs.PeftCacheConfig",
      "type": "integer",
      "default": 0,
      "description": "number of max sized 1-layer 1-module adapterSize=1 sets of weights that can be stored in host cache"
    },
    "num_device_module_layer": {
      "namespace": "$defs.PeftCacheConfig",
      "type": "integer",
      "default": 0,
      "description": "number of max sized 1-layer 1-module sets of weights that can be stored in host cache"
    },
    "optimal_adapter_size": {
      "namespace": "$defs.PeftCacheConfig",
      "type": "integer",
      "default": 8,
      "description": "optimal adapter size used to set page width"
    },
    "max_adapter_size": {
      "namespace": "$defs.PeftCacheConfig",
      "type": "integer",
      "default": 64,
      "description": "max supported adapter size. Used to compute minimum"
    },
    "num_put_workers": {
      "namespace": "$defs.PeftCacheConfig",
      "type": "integer",
      "default": 1,
      "description": "number of worker threads used to put weights into host cache"
    },
    "num_ensure_workers": {
      "namespace": "$defs.PeftCacheConfig",
      "type": "integer",
      "default": 1,
      "description": "number of worker threads used to copy weights from host to device"
    },
    "num_copy_streams": {
      "namespace": "$defs.PeftCacheConfig",
      "type": "integer",
      "default": 1,
      "description": "number of streams used to copy weights from host to device"
    },
    "max_pages_per_block_host": {
      "namespace": "$defs.PeftCacheConfig",
      "type": "integer",
      "default": 24,
      "description": "Number of cache pages per allocation block (host)"
    },
    "max_pages_per_block_device": {
      "namespace": "$defs.PeftCacheConfig",
      "type": "integer",
      "default": 8,
      "description": "Number of cache pages per allocation block (device)"
    },
    "device_cache_percent": {
      "namespace": "$defs.PeftCacheConfig",
      "anyOf": [{"type": "number"}, {"type": "null"}],
      "default": null,
      "description": "percent of memory after engine load to use for cache"
    },
    "host_cache_size": {
      "namespace": "$defs.PeftCacheConfig",
      "anyOf": [{"type": "integer"}, {"type": "null"}],
      "default": null,
      "description": "size in bytes to use for host cache"
    },
    "lora_prefetch_dir": {
      "namespace": "$defs.PeftCacheConfig",
      "anyOf": [{"type": "string"}, {"type": "null"}],
      "default": null,
      "description": "folder to store the LoRA weights we hope to load during engine initialization"
    },
    "enable_batch_size_tuning": {
      "namespace": "$defs.DynamicBatchConfig",
      "type": "boolean",
      "description": "Controls if the batch size should be tuned dynamically"
    },
    "enable_max_num_tokens_tuning": {
      "namespace": "$defs.DynamicBatchConfig",
      "type": "boolean",
      "description": "Controls if the max num tokens should be tuned dynamically"
    },
    "dynamic_batch_moving_average_window": {
      "namespace": "$defs.DynamicBatchConfig",
      "type": "integer",
      "description": "The window size for moving average of input and output length which is used to calculate dynamic batch size and max num tokens"
    }
  }
}
```
