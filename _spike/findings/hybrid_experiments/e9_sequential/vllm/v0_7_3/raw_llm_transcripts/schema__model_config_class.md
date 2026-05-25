# schema extraction transcript: model_config_class

- chunk_description: vllm.ModelConfig dataclass (model-side engine_params)
- expected_namespaces: ['engine_params']
- attempts: 1
- elapsed_sec: 52.53
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
    """Configuration for the model.

    Args:
        model: Name or path of the huggingface model to use.
            It is also used as the content for `model_name` tag in metrics
            output when `served_model_name` is not specified.
        task: The task to use the model for. Each vLLM instance only supports
            one task, even if the same model can be used for multiple tasks.
            When the model only supports one task, "auto" can be used to select
            it; otherwise, you must specify explicitly which task to use.
        tokenizer: Name or path of the huggingface tokenizer to use.
        tokenizer_mode: Tokenizer mode. "auto" will use the fast tokenizer if
            available, "slow" will always use the slow tokenizer,
            "mistral" will always use the tokenizer from `mistral_common`, and
            "custom" will use --tokenizer to select the preregistered tokenizer.
        trust_remote_code: Trust remote code (e.g., from HuggingFace) when
            downloading the model and tokenizer.
        allowed_local_media_path: Allowing API requests to read local images or
            videos from directories specified by the server file system.
            This is a security risk. Should only be enabled in trusted
            environments.
        dtype: Data type for model weights and activations. The "auto" option
            will use FP16 precision for FP32 and FP16 models, and BF16 precision
            for BF16 models.
        seed: Random seed for reproducibility.
        revision: The specific model version to use. It can be a branch name,
            a tag name, or a commit id. If unspecified, will use the default
            version.
        code_revision: The specific revision to use for the model code on
            Hugging Face Hub. It can be a branch name, a tag name, or a
            commit id. If unspecified, will use the default version.
        tokenizer_revision: The specific tokenizer version to use. It can be a
            branch name, a tag name, or a commit id. If unspecified, will use
            the default version.
        max_model_len: Maximum length of a sequence (including prompt and
            output). If None, will be derived from the model.
        spec_target_max_model_len: Specify the the maximum length for spec
            decoding draft models.
        quantization: Quantization method that was used to quantize the model
            weights. If None, we assume the model weights are not quantized.
        enforce_eager: Whether to enforce eager execution. If True, we will
            disable CUDA graph and always execute the model in eager mode.
            If False, we will use CUDA graph and eager execution in hybrid.
            If None, the user did not specify, so default to False.
        max_seq_len_to_capture: Maximum sequence len covered by CUDA graphs.
            When a sequence has context length larger than this, we fall back
            to eager mode. Additionally for encoder-decoder models, if the
            sequence length of the encoder input is larger than this, we fall
            back to the eager mode.
        max_logprobs: Maximum number of log probabilities. Defaults to 20.
        disable_sliding_window: Whether to disable sliding window. If True,
            we will disable the sliding window functionality of the model.
            If the model does not support sliding window, this argument is
            ignored.
        skip_tokenizer_init: If true, skip initialization of tokenizer and
            detokenizer.
        served_model_name: The model name used in metrics tag `model_name`,
            matches the model name exposed via the APIs. If multiple model
            names provided, the first name will be used. If not specified,
            the model name will be the same as `model`.
...<truncated>...
```

### Response

```
{
  "engine": "vllm",
  "engine_version": "0.7.3",
  "chunk_name": "model_config_class",
  "chunk_fields": {
    "allowed_local_media_path": {"namespace": "engine_params", "type": "boolean"},
    "config_format": {"namespace": "engine_params", "default": "auto"},
    "disable_mm_preprocessor_cache": {"namespace": "engine_params", "type": "boolean"},
    "disable_sliding_window": {"namespace": "engine_params", "type": "boolean"},
    "dtype": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "null"}]},
    "enforce_eager": {"namespace": "engine_params", "anyOf": [{"type": "boolean"}, {"type": "null"}]},
    "generation_config": {"namespace": "engine_params"},
    "hf_overrides": {"namespace": "engine_params"},
    "limit_mm_per_prompt": {"namespace": "engine_params"},
    "logits_processor_pattern": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "null"}]},
    "max_logprobs": {"namespace": "engine_params", "default": 20},
    "max_model_len": {"namespace": "engine_params", "anyOf": [{"type": "integer"}, {"type": "null"}]},
    "max_seq_len_to_capture": {"namespace": "engine_params"},
    "model": {"namespace": "engine_params"},
    "model_impl": {"namespace": "engine_params", "enum": ["auto", "vllm", "transformers"]},
    "mm_processor_kwargs": {"namespace": "engine_params"},
    "override_generation_config": {"namespace": "engine_params"},
    "override_neuron_config": {"namespace": "engine_params"},
    "override_pooler_config": {"namespace": "engine_params"},
    "quantization": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "null"}]},
    "revision": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "null"}]},
    "served_model_name": {"namespace": "engine_params"},
    "seed": {"namespace": "engine_params"},
    "skip_tokenizer_init": {"namespace": "engine_params", "type": "boolean"},
    "spec_target_max_model_len": {"namespace": "engine_params"},
    "task": {"namespace": "engine_params"},
    "tokenizer": {"namespace": "engine_params"},
    "tokenizer_mode": {"namespace": "engine_params", "enum": ["auto", "slow", "mistral", "custom"]},
    "tokenizer_revision": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "null"}]},
    "trust_remote_code": {"namespace": "engine_params", "type": "boolean"},
    "use_async_output_proc": {"namespace": "engine_params", "default": true}
  }
}
```
