# schema extraction transcript: model_config_class

- chunk_description: vllm.ModelConfig dataclass (model-side engine_params)
- expected_namespaces: ['engine_params']
- attempts: 1
- elapsed_sec: 119.38
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
  "chunk_name": "model_config_class",
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

=== SOURCE: vllm.ModelConfig (dataclass field list) ===
class ModelConfig:
    """Configuration for the model."""

    model: str = "facebook/opt-125m"
    """Name or path of the Hugging Face model to use. It is also used as the
    content for `model_name` tag in metrics output when `served_model_name` is
    not specified."""
    task: Literal[TaskOption, Literal["draft"]] = "auto"
    """The task to use the model for. Each vLLM instance only supports one
    task, even if the same model can be used for multiple tasks. When the model
    only supports one task, "auto" can be used to select it; otherwise, you
    must specify explicitly which task to use."""
    tokenizer: SkipValidation[str] = None  # type: ignore
    """Name or path of the Hugging Face tokenizer to use. If unspecified, model
    name or path will be used."""
    tokenizer_mode: TokenizerMode = "auto"
    """Tokenizer mode:\n
    - "auto" will use the fast tokenizer if available.\n
    - "slow" will always use the slow tokenizer.\n
    - "mistral" will always use the tokenizer from `mistral_common`.\n
    - "custom" will use --tokenizer to select the preregistered tokenizer."""
    trust_remote_code: bool = False
    """Trust remote code (e.g., from HuggingFace) when downloading the model
    and tokenizer."""
    dtype: Union[ModelDType, torch.dtype] = "auto"
    """Data type for model weights and activations:\n
    - "auto" will use FP16 precision for FP32 and FP16 models, and BF16
    precision for BF16 models.\n
    - "half" for FP16. Recommended for AWQ quantization.\n
    - "float16" is the same as "half".\n
    - "bfloat16" for a balance between precision and range.\n
    - "float" is shorthand for FP32 precision.\n
    - "float32" for FP32 precision."""
    seed: Optional[int] = None
    """Random seed for reproducibility. Initialized to None in V0, but
    initialized to 0 in V1."""
    hf_config_path: Optional[str] = None
    """Name or path of the Hugging Face config to use. If unspecified, model
    name or path will be used."""
    allowed_local_media_path: str = ""
    """Allowing API requests to read local images or videos from directories
    specified by the server file system. This is a security risk. Should only
    be enabled in trusted environments."""
    revision: Optional[str] = None
    """The specific model version to use. It can be a branch name, a tag name,
    or a commit id. If unspecified, will use the default version."""
    code_revision: Optional[str] = None
    """The specific revision to use for the model code on the Hugging Face Hub.
    It can be a branch name, a tag name, or a commit id. If unspecified, will
    use the default version."""
    rope_scaling: dict[str, Any] = field(default_factory=dict)
    """RoPE scaling configuration. For example,
    `{"rope_type":"dynamic","factor":2.0}`."""
    rope_theta: Optional[float] = None
    """RoPE theta. Use with `rope_scaling`. In some cases, changing the RoPE
    theta improves the performance of the scaled model."""
    tokenizer_revision: Optional[str] = None
    """The specific revision to use for the tokenizer on the Hugging Face Hub.
    It can be a branch name, a tag name, or a commit id. If unspecified, will
    use the default version."""
    max_model_len: SkipValidation[int] = None  # type: ignore
    """Model context length (prompt and output). If unspecified, will be
    automatically derived from the model config.

    When passing via `--max-model-len`, supports k/m/g/K/M/G in human-readable
    format. Examples:\n
    - 1k -> 1000\n
    - 1K -> 1024\n
    - 25.6k -> 25,600"""
    spec_target_max_model_len: Optional[int] = None
    """Specify the maximum length for spec decoding draft models."""
    quantization: SkipValidation[Optional[QuantizationMethods]] = None
    """Method used to quantize the weights. If `None`, we first check the
    `quantization_config` attribute in the model config 
...<truncated>...
```

### Response

```
{
  "engine": "vllm",
  "engine_version": "0.9.2",
  "chunk_name": "model_config_class",
  "chunk_fields": {
    "model": {"namespace": "engine_params", "type": "string"},
    "task": {"namespace": "engine_params", "enum": ["auto", "draft"]},
    "tokenizer": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "null"}], "default": null},
    "tokenizer_mode": {"namespace": "engine_params", "enum": ["auto", "slow", "mistral", "custom"]},
    "trust_remote_code": {"namespace": "engine_params", "type": "boolean", "default": false},
    "dtype": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "object"}], "enum": ["auto", "half", "float16", "bfloat16", "float", "float32"]},
    "seed": {"namespace": "engine_params", "type": "integer", "default": null},
    "hf_config_path": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "null"}], "default": null},
    "allowed_local_media_path": {"namespace": "engine_params", "type": "string"},
    "revision": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "null"}], "default": null},
    "code_revision": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "null"}], "default": null},
    "rope_scaling": {"namespace": "engine_params", "type": "object"},
    "rope_theta": {"namespace": "engine_params", "type": "number", "default": null},
    "tokenizer_revision": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "null"}], "default": null},
    "max_model_len": {"namespace": "engine_params", "type": "integer", "default": null},
    "spec_target_max_model_len": {"namespace": "engine_params", "type": "integer", "default": null},
    "quantization": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "null"}], "default": null},
    "enforce_eager": {"namespace": "engine_params", "type": "boolean", "default": false},
    "max_seq_len_to_capture": {"namespace": "engine_params", "type": "integer"},
    "max_logprobs": {"namespace": "engine_params", "type": "integer"},
    "disable_sliding_window": {"namespace": "engine_params", "type": "boolean", "default": false},
    "disable_cascade_attn": {"namespace": "engine_params", "type": "boolean", "default": false},
    "skip_tokenizer_init": {"namespace": "engine_params", "type": "boolean", "default": false},
    "enable_prompt_embeds": {"namespace": "engine_params", "type": "boolean", "default": false},
    "served_model_name": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "array"}, {"type": "null"}], "default": null},
    "limit_mm_per_prompt": {"namespace": "engine_params", "type": "object"},
    "media_io_kwargs": {"namespace": "engine_params", "type": "object"}
  }
}
```
