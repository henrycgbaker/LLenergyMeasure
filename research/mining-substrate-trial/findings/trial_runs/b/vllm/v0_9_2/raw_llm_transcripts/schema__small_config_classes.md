# schema extraction transcript: small_config_classes

- chunk_description: vllm.{LoRAConfig, PromptAdapterConfig, TokenizerPoolConfig, DecodingConfig}
- expected_namespaces: ['engine_params']
- attempts: 1
- elapsed_sec: 171.11
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
  "chunk_name": "small_config_classes",
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

=== SOURCE: vllm.LoRAConfig ===
class LoRAConfig:
    """Configuration for LoRA."""

    max_lora_rank: int = 16
    """Max LoRA rank."""
    max_loras: int = 1
    """Max number of LoRAs in a single batch."""
    fully_sharded_loras: bool = False
    """By default, only half of the LoRA computation is sharded with tensor
    parallelism. Enabling this will use the fully sharded layers. At high
    sequence length, max rank or tensor parallel size, this is likely faster.
    """
    max_cpu_loras: Optional[int] = None
    """Maximum number of LoRAs to store in CPU memory. Must be >= than
    `max_loras`."""
    lora_dtype: Union[torch.dtype, LoRADType] = "auto"
    """Data type for LoRA. If auto, will default to base model dtype."""
    lora_extra_vocab_size: int = 256
    """Maximum size of extra vocabulary that can be present in a LoRA adapter
    (added to the base model vocabulary)."""
    lora_vocab_padding_size: ClassVar[int] = current_platform\
        .get_lora_vocab_padding_size()
    long_lora_scaling_factors: Optional[tuple[float, ...]] = None
    """Specify multiple scaling factors (which can be different from base model
    scaling factor - see eg. Long LoRA) to allow for multiple LoRA adapters
    trained with those scaling factors to be used at the same time. If not
    specified, only adapters trained with the base model scaling factor are
    allowed."""
    bias_enabled: bool = False
    """Enable bias for LoRA adapters."""

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []
        factors.append(self.max_lora_rank)
        factors.append(self.max_loras)
        factors.append(self.fully_sharded_loras)
        factors.append(self.lora_dtype)
        factors.append(self.lora_extra_vocab_size)
        factors.append(self.lora_vocab_padding_size)
        factors.append(self.long_lora_scaling_factors)
        factors.append(self.bias_enabled)
        hash_str = hashlib.md5(str(factors).encode(),
                               usedforsecurity=False).hexdigest()
        return hash_str

    def __post_init__(self):
        # Setting the maximum rank to 512 should be able to satisfy the vast
        # majority of applications.
        possible_max_ranks = (8, 16, 32, 64, 128, 256, 320, 512)
        possible_lora_extra_vocab_size = (256, 512)
        if self.max_lora_rank not in possible_max_ranks:
            raise ValueError(
                f"max_lora_rank ({self.max_lora_rank}) must be one of "
                f"{possible_max_ranks}.")
        if self.lora_extra_vocab_size not in possible_lora_extra_vocab_size:
            raise ValueError(
                f"lora_extra_vocab_size ({self.lora_extra_vocab_size}) "
                f"must be one of {possible_lora_extra_vocab_size}.")
        if self.max_loras < 1:
            raise ValueError(f"max_loras ({self.max_loras}) must be >= 1.")
        if self.max_cpu_loras is None:
            self.max_cpu_loras = self.max_loras
        elif self.max_cpu_loras < self.max_loras:
            raise ValueError(
                f"max_cpu_loras ({self.max_cpu_loras}) must be >= "
                f"max_loras ({self.max_loras})")

    def verify_with_cache_config(self, cache_config: CacheConfig):
        if cache_config.cpu_offload_gb > 0 and not envs.VLLM_USE_V1:
            raise ValueError(
                "V0 LoRA does not support CPU offload, please use V1.")

    def verify_with_model_config(self, model_config: ModelConfig):
    
...<truncated>...
```

### Response

```
{
  "engine": "vllm",
  "engine_version": "0.9.2",
  "chunk_name": "small_config_classes",
  "chunk_fields": {
    "max_lora_rank": {"namespace": "engine_params", "type": "integer", "default": 16, "description": "Max LoRA rank."},
    "max_loras": {"namespace": "engine_params", "type": "integer", "default": 1, "description": "Max number of LoRAs in a single batch."},
    "fully_sharded_loras": {"namespace": "engine_params", "type": "boolean", "default": false, "description": "By default, only half of the LoRA computation is sharded with tensor parallelism. Enabling this will use the fully sharded layers."},
    "max_cpu_loras": {"namespace": "engine_params", "anyOf": [{"type": "integer"}, {"type": "null"}], "default": null, "description": "Maximum number of LoRAs to store in CPU memory. Must be >= than `max_loras`."},
    "lora_dtype": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "object", "description": "torch.dtype"}], "default": "auto", "description": "Data type for LoRA. If auto, will default to base model dtype."},
    "lora_extra_vocab_size": {"namespace": "engine_params", "type": "integer", "default": 256, "description": "Maximum size of extra vocabulary that can be present in a LoRA adapter (added to the base model vocabulary)."},
    "long_lora_scaling_factors": {"namespace": "engine_params", "anyOf": [{"type": "array", "items": {"type": "number"}}, {"type": "null"}], "default": null, "description": "Specify multiple scaling factors (which can be different from base model scaling factor - see eg. Long LoRA) to allow for multiple LoRA adapters trained with those scaling factors to be used at the same time."},
    "bias_enabled": {"namespace": "engine_params", "type": "boolean", "default": false, "description": "Enable bias for LoRA adapters."},
    "max_prompt_adapters": {"namespace": "engine_params", "type": "integer", "default": 1, "description": "Max number of PromptAdapters in a batch."},
    "max_prompt_adapter_token": {"namespace": "engine_params", "type": "integer", "default": 0, "description": "Max number of PromptAdapters tokens."},
    "max_cpu_prompt_adapters": {"namespace": "engine_params", "anyOf": [{"type": "integer"}, {"type": "null"}], "default": null, "description": "Maximum number of PromptAdapters to store in CPU memory. Must be >= than `max_prompt_adapters`."},
    "prompt_adapter_dtype": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "object", "description": "torch.dtype"}], "default": "auto", "description": "Data type for PromptAdapter. If auto, will default to base model dtype."},
    "pool_size": {"namespace": "engine_params", "type": "integer", "default": 0, "description": "This parameter is deprecated and will be removed in a future release."},
    "pool_type": {"namespace": "engine_params", "type": "string", "default": "ray", "description": "This parameter is deprecated and will be removed in a future release."},
    "extra_config": {"namespace": "engine_params", "type": "object", "default": {}, "description": "This parameter is deprecated and will be removed in a future release."},
    "backend": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "null"}], "default": null, "description": "Which engine will be used for guided decoding (JSON schema / regex etc) by default. With \"auto\", we will make opinionated choices based on request contents and what the backend libraries currently support, so the behavior is subject to change in each release."},
    "disable_fallback": {"namespace": "engine_params", "type": "boolean", "default": false, "description": "If `True`, vLLM will not fallback to a different backend on error."},
    "disable_any_whitespace": {"namespace": "engine_params", "type": "boolean", "default": false, "description": "If `True`, the model will not generate any whitespace during guided decoding. This is only supported for xgrammar and guidance backends."},
    "disable_additional_properties": {"namespace": "engine_params", "type": "boolean", "default": false, "description": "If `True`, the `guidance` backend will not use `additionalProperties` in the JSON schema. This is only supported for the `guidance` backend and is used to better align its behaviour with `outlines` and `xgrammar`."},
    "reasoning_backend": {"namespace": "engine_params", "type": "string", "default": "", "description": "Select the reasoning parser depending on the model that you're using. This is used to parse the reasoning content into OpenAI API format."}
  }
}
```
