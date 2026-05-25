# schema extraction transcript: calib_kv_cache_configs

- chunk_description: tensorrt_llm.{CalibConfig, KvCacheConfig} Pydantic classes
- expected_namespaces: ['engine_params', '$defs.CalibConfig', '$defs.KvCacheConfig']
- attempts: 1
- elapsed_sec: 159.03
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser extracting the parameter schema for the
tensorrt library, version 1.0.0.

You will be shown ONE CHUNK of source code. Extract the parameter
schema for the fields visible in this chunk. Other chunks are mined
separately and merged later - do not include fields from outside
this chunk.

OUTPUT FORMAT: a single JSON object matching EXACTLY this shape:

{
  "engine": "tensorrt",
  "engine_version": "1.0.0",
  "chunk_name": "calib_kv_cache_configs",
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

EXPECTED NAMESPACES FOR THIS CHUNK: engine_params, $defs.CalibConfig, $defs.KvCacheConfig
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
CalibConfig and KvCacheConfig are nested Pydantic configs. Each Field(...) declaration is an engine_params entry. Note the `Literal['cuda', 'cpu']` on CalibConfig.device - emit this as `enum: [cuda, cpu]` in the schema entry.

=== SOURCE: CalibConfig ===
class CalibConfig(StrictBaseModel):
    """
    Calibration configuration.
    """
    device: Literal['cuda',
                    'cpu'] = Field(default='cuda',
                                   description="The device to run calibration.")
    calib_dataset: str = Field(
        default='cnn_dailymail',
        description="The name or local path of calibration dataset.")
    calib_batches: int = Field(
        default=512,
        description="The number of batches that the calibration runs.")
    calib_batch_size: int = Field(
        default=1, description="The batch size that the calibration runs.")
    calib_max_seq_length: int = Field(
        default=512,
        description="The maximum sequence length that the calibration runs.")
    random_seed: int = Field(
        default=1234, description="The random seed used for calibration.")
    tokenizer_max_seq_length: int = Field(
        default=2048,
        description=
        "The maximum sequence length to initialize tokenizer for calibration.")

    @classmethod
    def from_dict(cls, config: dict) -> 'CalibConfig':
        """Create a CalibConfig instance from a dict.

        Args:
            config (dict): The dict used to create CalibConfig.

        Returns:
            tensorrt_llm.llmapi.CalibConfig: The CalibConfig created from dict.
        """
        return cls(**config)

    def to_dict(self) -> dict:
        """Dump a CalibConfig instance to a dict.

        Returns:
            dict: The dict dumped from CalibConfig.
        """
        return self.model_dump()



=== SOURCE: KvCacheConfig ===
class KvCacheConfig(StrictBaseModel, PybindMirror):
    """
    Configuration for the KV cache.
    """
    enable_block_reuse: bool = Field(
        default=True,
        description=
        "Controls if KV cache blocks can be reused for different requests.")
    max_tokens: Optional[int] = Field(
        default=None,
        description=
        "The maximum number of tokens that should be stored in the KV cache. If both `max_tokens` and `free_gpu_memory_fraction` are specified, memory corresponding to the minimum will be used."
    )
    max_attention_window: Optional[List[int]] = Field(
        default=None,
        description=
        "Size of the attention window for each sequence. Only the last tokens will be stored in the KV cache. If the number of elements in `max_attention_window` is less than the number of layers, `max_attention_window` will be repeated multiple times to the number of layers."
    )
    sink_token_length: Optional[int] = Field(
        default=None,
        description=
        "Number of sink tokens (tokens to always keep in attention window).")
    free_gpu_memory_fraction: Optional[float] = Field(
        default=None,
        description=
        "The fraction of GPU memory fraction that should be allocated for the KV cache. Default is 90%. If both `max_tokens` and `free_gpu_memory_fraction` are specified, memory corresponding to the minimum will be used."
    )
    host_cache_size: Optional[int] = Field(
        default=None,
        description=
        "Size of the host cache in bytes. If both `max_tokens` and `host_cache_size` are specified, memory corresponding to the minimum will be used."
    )
    onboard_blocks: bool = Field(
        default=True, description="Controls if blocks are onboarded.")
    cross_kv_cache_fraction: Optional[float] = Field(
        default=None,
        description=
        "The fraction of the KV Cache memory should be reserved for cross attention. If set to p, self attention will use 1-p of KV Cache memory and
...<truncated>...
```

### Response

```
{
  "engine": "tensorrt",
  "engine_version": "1.0.0",
  "chunk_name": "calib_kv_cache_configs",
  "chunk_fields": {
    "device": {
      "namespace": "$defs.CalibConfig",
      "type": "string",
      "enum": ["cuda", "cpu"],
      "default": "cuda",
      "description": "The device to run calibration."
    },
    "calib_dataset": {
      "namespace": "$defs.CalibConfig",
      "type": "string",
      "default": "cnn_dailymail",
      "description": "The name or local path of calibration dataset."
    },
    "calib_batches": {
      "namespace": "$defs.CalibConfig",
      "type": "integer",
      "default": 512,
      "description": "The number of batches that the calibration runs."
    },
    "calib_batch_size": {
      "namespace": "$defs.CalibConfig",
      "type": "integer",
      "default": 1,
      "description": "The batch size that the calibration runs."
    },
    "calib_max_seq_length": {
      "namespace": "$defs.CalibConfig",
      "type": "integer",
      "default": 512,
      "description": "The maximum sequence length that the calibration runs."
    },
    "random_seed": {
      "namespace": "$defs.CalibConfig",
      "type": "integer",
      "default": 1234,
      "description": "The random seed used for calibration."
    },
    "tokenizer_max_seq_length": {
      "namespace": "$defs.CalibConfig",
      "type": "integer",
      "default": 2048,
      "description": "The maximum sequence length to initialize tokenizer for calibration."
    },
    "enable_block_reuse": {
      "namespace": "$defs.KvCacheConfig",
      "type": "boolean",
      "default": true,
      "description": "Controls if KV cache blocks can be reused for different requests."
    },
    "max_tokens": {
      "namespace": "$defs.KvCacheConfig",
      "anyOf": [{"type": "integer"}, {"type": "null"}],
      "default": null,
      "description": "The maximum number of tokens that should be stored in the KV cache. If both `max_tokens` and `free_gpu_memory_fraction` are specified, memory corresponding to the minimum will be used."
    },
    "max_attention_window": {
      "namespace": "$defs.KvCacheConfig",
      "anyOf": [{"type": "array", "items": {"type": "integer"}}, {"type": "null"}],
      "default": null,
      "description": "Size of the attention window for each sequence. Only the last tokens will be stored in the KV cache. If the number of elements in `max_attention_window` is less than the number of layers, `max_attention_window` will be repeated multiple times to the number of layers."
    },
    "sink_token_length": {
      "namespace": "$defs.KvCacheConfig",
      "anyOf": [{"type": "integer"}, {"type": "null"}],
      "default": null,
      "description": "Number of sink tokens (tokens to always keep in attention window)."
    },
    "free_gpu_memory_fraction": {
      "namespace": "$defs.KvCacheConfig",
      "anyOf": [{"type": "number"}, {"type": "null"}],
      "default": null,
      "description": "The fraction of GPU memory fraction that should be allocated for the KV cache. Default is 90%. If both `max_tokens` and `free_gpu_memory_fraction` are specified, memory corresponding to the minimum will be used."
    },
    "host_cache_size": {
      "namespace": "$defs.KvCacheConfig",
      "anyOf": [{"type": "integer"}, {"type": "null"}],
      "default": null,
      "description": "Size of the host cache in bytes. If both `max_tokens` and `host_cache_size` are specified, memory corresponding to the minimum will be used."
    },
    "onboard_blocks": {
      "namespace": "$defs.KvCacheConfig",
      "type": "boolean",
      "default": true,
      "description": "Controls if blocks are onboarded."
    },
    "cross_kv_cache_fraction": {
      "namespace": "$defs.KvCacheConfig",
      "anyOf": [{"type": "number"}, {"type": "null"}],
      "default": null,
      "description": "The fraction of the KV Cache memory should be reserved for cross attention. If set to p, self attention will use 1-p of KV Cache memory and cross attention will use p of KV Cache memory. Default is 50%. Should only be set when using encoder-decoder model."
    },
    "secondary_offload_min_priority": {
      "namespace": "$defs.KvCacheConfig",
      "anyOf": [{"type": "integer"}, {"type": "null"}],
      "default": null,
      "description": "Only blocks with priority > mSecondaryOfflineMinPriority can be offloaded to secondary memory."
    },
    "event_buffer_max_size": {
      "namespace": "$defs.KvCacheConfig",
      "type": "integer",
      "default": 0,
      "description": "Maximum size of the event buffer. If set to 0, the event buffer will not be used."
    },
    "enable_partial_reuse": {
      "namespace": "$defs.KvCacheConfig",
      "type": "boolean",
      "default": true,
      "description": "Whether blocks that are only partially matched can be reused."
    },
    "copy_on_partial_reuse": {
      "namespace": "$defs.KvCacheConfig",
      "type": "boolean",
      "default": true,
      "description": "Whether partially matched blocks that are in use can be reused after copying them."
    },
    "use_uvm": {
      "namespace": "$defs.KvCacheConfig",
      "type": "boolean",
      "default": false,
      "description": "Whether to use UVM for the KV cache."
    },
    "dtype": {
      "namespace": "$defs.KvCacheConfig",
      "type": "string",
      "default": "auto",
      "description": "The data type to use for the KV cache."
    }
  }
}
```
