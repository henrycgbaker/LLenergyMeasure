# schema extraction transcript: small_config_classes

- chunk_description: vllm.{LoRAConfig, PromptAdapterConfig, TokenizerPoolConfig, DecodingConfig}
- expected_namespaces: ['engine_params']
- attempts: 1
- elapsed_sec: 192.98
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser extracting the parameter schema for the
vllm library, version 0.6.0.

You will be shown ONE CHUNK of source code. Extract the parameter
schema for the fields visible in this chunk. Other chunks are mined
separately and merged later - do not include fields from outside
this chunk.

OUTPUT FORMAT: a single JSON object matching EXACTLY this shape:

{
  "engine": "vllm",
  "engine_version": "0.6.0",
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
    max_lora_rank: int
    max_loras: int
    fully_sharded_loras: bool = False
    max_cpu_loras: Optional[int] = None
    lora_dtype: Optional[torch.dtype] = None
    lora_extra_vocab_size: int = 256
    # This is a constant.
    lora_vocab_padding_size: ClassVar[int] = 256
    long_lora_scaling_factors: Optional[Tuple[float]] = None

    def __post_init__(self):
        # Setting the maximum rank to 256 should be able to satisfy the vast
        # majority of applications.
        possible_max_ranks = (8, 16, 32, 64, 128, 256)
        possible_lora_extra_vocab_size = (0, 256, 512)
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

    def verify_with_model_config(self, model_config: ModelConfig):
        if self.lora_dtype in (None, "auto"):
            self.lora_dtype = model_config.dtype
        elif isinstance(self.lora_dtype, str):
            self.lora_dtype = getattr(torch, self.lora_dtype)
        if model_config.quantization and model_config.quantization not in [
                "awq", "gptq"
        ]:
            # TODO support marlin and squeezellm
            logger.warning("%s quantization is not tested with LoRA yet.",
                           model_config.quantization)

    def verify_with_scheduler_config(self, scheduler_config: SchedulerConfig):
        if scheduler_config.chunked_prefill_enabled:
            raise ValueError("LoRA is not supported with chunked prefill yet.")


@dataclass

=== SOURCE: vllm.PromptAdapterConfig ===
class PromptAdapterConfig:
    max_prompt_adapters: int
    max_prompt_adapter_token: int
    max_cpu_prompt_adapters: Optional[int] = None
    prompt_adapter_dtype: Optional[torch.dtype] = None

    def __post_init__(self):
        library_name = 'peft'
        try:
            __import__(library_name)
        except ImportError as e:
            raise ImportError(
                f"'{library_name}' is not installed for prompt adapter support."
                f"Please install it using 'pip install {library_name}'."
            ) from e

        if self.max_prompt_adapters < 1:
            raise ValueError(f"max_prompt_adapters "
                             f"({self.max_prompt_adapters}) must be >= 1.")
        if self.max_prompt_adapter_token == 0:
            raise ValueError("max_prompt_adapter_token must be set.")
        if self.max_cpu_prompt_adapters is None:
            self.max_cpu_prompt_adapters = self.max_prompt_adapters

    def verify_with_model_config(self, model_config: ModelConfig):
        if self.prompt_adapter_dtype in (None, "auto"):
            self.prompt_adapter_dtype = model_config.dtype
        elif isinstance(self.prompt_adapter_dtype, str):
            self.prompt_adapter_dtype = getattr(torch,
                                                self.prompt_adapter_dtype)


@dataclass

=== SOURCE: vllm.TokenizerPoolConfig ===
class TokenizerPoolConfig:
    """Configuration for the tokenizer pool.

    Args:
        pool_size: Number of tokenizer workers in the pool.
        pool_type: Type of the pool.
        extra_config: Additional config for the pool.
        
...<truncated>...
```

### Response

```
{
  "engine": "vllm",
  "engine_version": "0.6.0",
  "chunk_name": "small_config_classes",
  "chunk_fields": {
    "max_lora_rank": {"namespace": "engine_params", "type": "integer"},
    "max_loras": {"namespace": "engine_params", "type": "integer"},
    "fully_sharded_loras": {"namespace": "engine_params", "type": "boolean", "default": false},
    "max_cpu_loras": {"namespace": "engine_params", "anyOf": [{"type": "integer"}, {"type": "null"}], "default": null},
    "lora_dtype": {"namespace": "engine_params", "anyOf": [{"type": "object", "description": "torch.dtype"}, {"type": "null"}], "default": null},
    "lora_extra_vocab_size": {"namespace": "engine_params", "type": "integer", "default": 256},
    "long_lora_scaling_factors": {"namespace": "engine_params", "anyOf": [{"type": "array", "items": {"type": "number"}}, {"type": "null"}], "default": null},
    "max_prompt_adapters": {"namespace": "engine_params", "type": "integer"},
    "max_prompt_adapter_token": {"namespace": "engine_params", "type": "integer"},
    "max_cpu_prompt_adapters": {"namespace": "engine_params", "anyOf": [{"type": "integer"}, {"type": "null"}], "default": null},
    "prompt_adapter_dtype": {"namespace": "engine_params", "anyOf": [{"type": "object", "description": "torch.dtype"}, {"type": "null"}], "default": null},
    "pool_size": {"namespace": "engine_params", "type": "integer"},
    "pool_type": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "object", "description": "BaseTokenizerGroup"}]},
    "extra_config": {"namespace": "engine_params", "type": "object"},
    "guided_decoding_backend": {"namespace": "engine_params", "type": "string", "default": "outlines"}
  }
}
```
