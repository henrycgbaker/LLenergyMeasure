# schema extraction transcript: bitsandbytes_compile_configs

- chunk_description: BitsAndBytesConfig + CompileConfig + WatermarkingConfig fields
- expected_namespaces: ['engine_params', '$defs.CompileConfig']
- attempts: 1
- elapsed_sec: 58.30
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser extracting the parameter schema for the
transformers library, version 4.57.3.

You will be shown ONE CHUNK of source code. Extract the parameter
schema for the fields visible in this chunk. Other chunks are mined
separately and merged later - do not include fields from outside
this chunk.

OUTPUT FORMAT: a single JSON object matching EXACTLY this shape:

{
  "engine": "transformers",
  "engine_version": "4.57.3",
  "chunk_name": "bitsandbytes_compile_configs",
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

EXPECTED NAMESPACES FOR THIS CHUNK: engine_params, $defs.CompileConfig
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

=== SOURCE: BitsAndBytesConfig (quantisation fields that surface as engine_params kwargs) ===
@dataclass
class BitsAndBytesConfig(QuantizationConfigMixin):
    """
    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `bitsandbytes`.

    This replaces `load_in_8bit` or `load_in_4bit`therefore both options are mutually exclusive.

    Currently only supports `LLM.int8()`, `FP4`, and `NF4` quantization. If more methods are added to `bitsandbytes`,
    then more arguments will be added to this class.

    Args:
        load_in_8bit (`bool`, *optional*, defaults to `False`):
            This flag is used to enable 8-bit quantization with LLM.int8().
        load_in_4bit (`bool`, *optional*, defaults to `False`):
            This flag is used to enable 4-bit quantization by replacing the Linear layers with FP4/NF4 layers from
            `bitsandbytes`.
        llm_int8_threshold (`float`, *optional*, defaults to 6.0):
            This corresponds to the outlier threshold for outlier detection as described in `LLM.int8() : 8-bit Matrix
            Multiplication for Transformers at Scale` paper: https://huggingface.co/papers/2208.07339 Any hidden states value
            that is above this threshold will be considered an outlier and the operation on those values will be done
            in fp16. Values are usually normally distributed, that is, most values are in the range [-3.5, 3.5], but
            there are some exceptional systematic outliers that are very differently distributed for large models.
            These outliers are often in the interval [-60, -6] or [6, 60]. Int8 quantization works well for values of
            magnitude ~5, but beyond that, there is a significant performance penalty. A good default threshold is 6,
            but a lower threshold might be needed for more unstable models (small models, fine-tuning).
        llm_int8_skip_modules (`list[str]`, *optional*):
            An explicit list of the modules that we do not want to convert in 8-bit. This is useful for models such as
            Jukebox that has several heads in different places and not necessarily at the last position. For example
            for `CausalLM` models, the last `lm_head` is kept in its original `dtype`.
        llm_int8_enable_fp32_cpu_offload (`bool`, *optional*, defaults to `False`):
            This flag is used for advanced use cases and users that are aware of this feature. If you want to split
            your model in different parts and run some parts in int8 on GPU and some parts in fp32 on CPU, you can use
            this flag. This is useful for offloading large models such as `google/flan-t5-xxl`. Note that the int8
            operations will not be run on CPU.
        llm_int8_has_fp16_weight (`bool`, *optional*, defaults to `False`):
            This flag runs LLM.int8() with 16-bit main weights. This is useful for fine-tuning as the weights do not
            have to be converted back and forth for the backward pass.
        bnb_4bit_compute_dtype (`torch.dtype` or str, *optional*, defaults to `torch.float32`):
            This sets the computational type which might be different than the input type. For example, inputs might be
            fp32, but computation can be set to bf16 for speedups.
        bnb_4bit_quant_type (`str`,  *optional*, defaults to `"fp4"`):
            This sets the quantization data type in the bnb.nn.Linear4Bit layers. Options are FP4 and NF4 data types
            which are specified by `fp4` or `nf4`.
        bnb_4bit_use_double_quant (`bool`, *optional*, defaults to `False`):
            This flag is used for nested quantization where the quantization constants from the first quantization are
            quantized again.
        bnb_4bit_quant_storage (`torch.dtype` or str, *optional*, defau
...<truncated>...
```

### Response

```
{
  "engine": "transformers",
  "engine_version": "4.57.3",
  "chunk_name": "bitsandbytes_compile_configs",
  "chunk_fields": {
    "load_in_8bit": {"namespace": "engine_params", "type": "boolean", "default": false},
    "load_in_4bit": {"namespace": "engine_params", "type": "boolean", "default": false},
    "llm_int8_threshold": {"namespace": "engine_params", "type": "number", "default": 6.0},
    "llm_int8_skip_modules": {"namespace": "engine_params", "anyOf": [{"type": "array"}, {"type": "null"}], "default": null},
    "llm_int8_enable_fp32_cpu_offload": {"namespace": "engine_params", "type": "boolean", "default": false},
    "llm_int8_has_fp16_weight": {"namespace": "engine_params", "type": "boolean", "default": false},
    "bnb_4bit_compute_dtype": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "object", "description": "torch.dtype"}, {"type": "null"}], "default": null},
    "bnb_4bit_quant_type": {"namespace": "engine_params", "type": "string", "enum": ["fp4", "nf4"], "default": "fp4"},
    "bnb_4bit_use_double_quant": {"namespace": "engine_params", "type": "boolean", "default": false},
    "bnb_4bit_quant_storage": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "object", "description": "torch.dtype"}, {"type": "null"}], "default": null},
    "fullgraph": {"namespace": "$defs.CompileConfig", "type": "boolean", "default": false},
    "dynamic": {"namespace": "$defs.CompileConfig", "anyOf": [{"type": "boolean"}, {"type": "null"}], "default": null},
    "backend": {"namespace": "$defs.CompileConfig", "anyOf": [{"type": "string"}, {"type": "object", "description": "Callable"}, {"type": "null"}], "default": "inductor"},
    "mode": {"namespace": "$defs.CompileConfig", "type": "string", "default": "reduce-overhead"},
    "options": {"namespace": "$defs.CompileConfig", "anyOf": [{"type": "object"}, {"type": "null"}], "default": null},
    "greenlist_ratio": {"namespace": "engine_params", "type": "number", "default": 0.25},
    "bias": {"namespace": "engine_params", "type": "number", "default": 2.0},
    "hashing_key": {"namespace": "engine_params", "type": "integer", "default": 15485863},
    "seeding_scheme": {"namespace": "engine_params", "type": "string", "enum": ["lefthash", "selfhash"], "default": "lefthash"},
    "context_width": {"namespace": "engine_params", "type": "integer", "default": 1}
  }
}
```
