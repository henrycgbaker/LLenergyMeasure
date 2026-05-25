# schema extraction transcript: base_llm_args_class

- chunk_description: tensorrt_llm.BaseLlmArgs Pydantic dataclass (engine_params)
- expected_namespaces: ['engine_params']
- attempts: 1
- elapsed_sec: 140.54
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
  "chunk_name": "base_llm_args_class",
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

=== CONTEXT ===
BaseLlmArgs is the parent class of TrtLlmArgs and TorchLlmArgs. Most fields used at the LLM-API surface live here. Pydantic Field(default=..., description=...) is the common pattern. Extract each field as engine_params with its type + default.

=== SOURCE: BaseLlmArgs (Pydantic class body) ===
class BaseLlmArgs(StrictBaseModel):
    """
    Base class for both TorchLlmArgs and TrtLlmArgs. It contains all the arguments that are common to both.
    """
    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "forbid",
    }

    # Explicit arguments
    model: Union[str, Path] = Field(
        description=
        "The path to the model checkpoint or the model name from the Hugging Face Hub."
    )

    tokenizer: Optional[Union[
        str, Path, TokenizerBase, PreTrainedTokenizerBase]] = Field(
            description=
            "The path to the tokenizer checkpoint or the tokenizer name from the Hugging Face Hub.",
            default=None)

    tokenizer_mode: Literal['auto', 'slow'] = Field(
        default='auto',
        description="The mode to initialize the tokenizer.",
        json_schema_extra={"type": "Literal['auto', 'slow']"})

    skip_tokenizer_init: bool = Field(
        default=False,
        description="Whether to skip the tokenizer initialization.")

    trust_remote_code: bool = Field(
        default=False, description="Whether to trust the remote code.")

    tensor_parallel_size: int = Field(default=1,
                                      description="The tensor parallel size.")

    dtype: str = Field(default="auto",
                       description="The data type to use for the model.")

    revision: Optional[str] = Field(
        default=None, description="The revision to use for the model.")

    tokenizer_revision: Optional[str] = Field(
        default=None, description="The revision to use for the tokenizer.")

    # Below are all remaining arguments

    pipeline_parallel_size: int = Field(
        default=1, description="The pipeline parallel size.")

    context_parallel_size: int = Field(default=1,
                                       description="The context parallel size.")

    gpus_per_node: Optional[int] = Field(
        default=None,
        description="The number of GPUs per node.",
        status="beta",
        validate_default=True)

    moe_cluster_parallel_size: Optional[int] = Field(
        default=None,
        description="The cluster parallel size for MoE models's expert weights.",
        status="beta")

    moe_tensor_parallel_size: Optional[int] = Field(
        default=None,
        description="The tensor parallel size for MoE models's expert weights.")

    moe_expert_parallel_size: Optional[int] = Field(
        default=None,
        description="The expert parallel size for MoE models's expert weights.")

    enable_attention_dp: bool = Field(
        default=False,
        description="Enable attention data parallel.",
        status="beta")

    cp_config: Optional[dict] = Field(default_factory=dict,
                                      description="Context parallel config.",
                                      status="prototype")

    load_format: Literal['auto', 'dummy'] = Field(
        default='auto',
        description="The format to load the model.",
        json_schema_extra={"type": "Literal['auto', 'dummy']"})

    fail_fast_on_attention_window_too_large: bool = Field(
        default=False,
        description=
        "Fail fast when attention window is too large to fit even a single sequence in the KV cache.",
        status="prototype")

    # LoRA arguments
    enable_lora: bool = Field(default=False, description="Enable LoRA.")

    lora_config: Optional[LoraConfig] = Field(
        default=None, description="LoRA configuration for the model.")

    # Several options from ExecutorConfig, expanded here for less hierarchy
    kv_
...<truncated>...
```

### Response

```
{
  "engine": "tensorrt",
  "engine_version": "1.0.0",
  "chunk_name": "base_llm_args_class",
  "chunk_fields": {
    "model": {"namespace": "engine_params", "type": "string", "description": "The path to the model checkpoint or the model name from the Hugging Face Hub."},
    "tokenizer": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "object", "description": "PathLike"}, {"type": "null"}], "default": null, "description": "The path to the tokenizer checkpoint or the tokenizer name from the Hugging Face Hub."},
    "tokenizer_mode": {"namespace": "engine_params", "type": "string", "enum": ["auto", "slow"], "default": "auto", "description": "The mode to initialize the tokenizer."},
    "skip_tokenizer_init": {"namespace": "engine_params", "type": "boolean", "default": false, "description": "Whether to skip the tokenizer initialization."},
    "trust_remote_code": {"namespace": "engine_params", "type": "boolean", "default": false, "description": "Whether to trust the remote code."},
    "tensor_parallel_size": {"namespace": "engine_params", "type": "integer", "default": 1, "description": "The tensor parallel size."},
    "dtype": {"namespace": "engine_params", "type": "string", "default": "auto", "description": "The data type to use for the model."},
    "revision": {"namespace": "engine_params", "type": "string", "default": null, "description": "The revision to use for the model."},
    "tokenizer_revision": {"namespace": "engine_params", "type": "string", "default": null, "description": "The revision to use for the tokenizer."},
    "pipeline_parallel_size": {"namespace": "engine_params", "type": "integer", "default": 1, "description": "The pipeline parallel size."},
    "context_parallel_size": {"namespace": "engine_params", "type": "integer", "default": 1, "description": "The context parallel size."},
    "gpus_per_node": {"namespace": "engine_params", "type": "integer", "default": null, "description": "The number of GPUs per node.", "status": "beta"},
    "moe_cluster_parallel_size": {"namespace": "engine_params", "type": "integer", "default": null, "description": "The cluster parallel size for MoE models's expert weights.", "status": "beta"},
    "moe_tensor_parallel_size": {"namespace": "engine_params", "type": "integer", "default": null, "description": "The tensor parallel size for MoE models's expert weights."},
    "moe_expert_parallel_size": {"namespace": "engine_params", "type": "integer", "default": null, "description": "The expert parallel size for MoE models's expert weights."},
    "enable_attention_dp": {"namespace": "engine_params", "type": "boolean", "default": false, "description": "Enable attention data parallel.", "status": "beta"},
    "cp_config": {"namespace": "engine_params", "type": "object", "default": null, "description": "Context parallel config.", "status": "prototype"},
    "load_format": {"namespace": "engine_params", "type": "string", "enum": ["auto", "dummy"], "default": "auto", "description": "The format to load the model."},
    "fail_fast_on_attention_window_too_large": {"namespace": "engine_params", "type": "boolean", "default": false, "description": "Fail fast when attention window is too large to fit even a single sequence in the KV cache.", "status": "prototype"},
    "enable_lora": {"namespace": "engine_params", "type": "boolean", "default": false, "description": "Enable LoRA."},
    "lora_config": {"namespace": "engine_params", "type": "object", "default": null, "description": "LoRA configuration for the model."},
    "kv_cache_config": {"namespace": "engine_params", "type": "object", "description": "KV cache config."},
    "enable_chunked_prefill": {"namespace": "engine_params", "type": "boolean", "default": false, "description": "Enable chunked prefill."},
    "guided_decoding_backend": {"namespace": "engine_params", "type": "string", "enum": ["xgrammar", "llguidance"], "default": null, "description": "Guided decoding backend. llguidance is supported in PyTorch backend only."},
    "batched_logits_processor": {"namespace": "engine_params", "type": "object", "default": null, "description": "Batched logits processor."},
    "iter_stats_max_iterations": {"namespace": "engine_params", "type": "integer", "default": null, "description": "The maximum number of iterations for iter stats.", "status": "prototype"},
    "request_stats_max_iterations": {"namespace": "engine_params", "type": "integer", "default": null, "description": "The maximum number of iterations for request stats.", "status": "prototype"},
    "peft_cache_config": {"namespace": "engine_params", "type": "object", "default": null, "description": "PEFT cache config.", "status": "prototype"},
    "scheduler_config": {"namespace": "engine_params", "type": "object", "description": "Scheduler config.", "status": "prototype"},
    "cache_transceiver_config": {"namespace": "engine_params", "type": "object", "default": null, "description": "Cache transceiver config.", "status": "prototype"},
    "speculative_config": {"namespace": "engine_params", "type": "object", "default": null, "description": "Speculative decoding config."},
    "max_batch_size": {"namespace": "engine_params", "type": "integer", "default": null, "description": "The maximum batch size."},
    "max_input_len": {"namespace": "engine_params", "type": "integer", "default": null, "description": "The maximum input length."},
    "max_seq_len": {"namespace": "engine_params", "type": "integer", "default": null, "description": "The maximum sequence length."},
    "max_beam_width": {"namespace": "engine_params", "type": "integer", "default": null, "description": "The maximum beam width."},
    "max_num_tokens": {"namespace": "engine_params", "type": "integer", "default": null, "description": "The maximum number of tokens."},
    "gather_generation_logits": {"namespace": "engine_params", "type": "boolean", "default": false, "description": "Gather generation logits.", "status": "prototype"},
    "num_postprocess_workers": {"namespace": "engine_params", "type": "integer", "default": 0, "description": "The number of processes used for postprocessing the generated tokens, including detokenization.", "status": "prototype"},
    "postprocess_tokenizer_dir": {"namespace": "engine_params", "type": "string", "default": null, "description": "The path to the tokenizer directory for postprocessing.", "status": "prototype"},
    "reasoning_parser": {"namespace": "engine_params", "type": "string", "default": null, "description": "The parser to separate reasoning content from output.", "status": "prototype"},
    "decoding_config": {"namespace": "engine_params", "type": "object", "default": null, "description": "The decoding config.", "status": "deprecated", "deprecated": "Use speculative_config instead."},
    "mpi_session": {"namespace": "engine_params", "type": "object", "default": null, "description": "The optional MPI session to use for this LLM instance."},
    "backend": {"namespace": "engine_params", "type": "string", "default": null, "description": "The backend to use for this LLM instance.", "status": "deprecated"}
  }
}
```
