# schema extraction transcript: trt_llm_args_class

- chunk_description: tensorrt_llm.TrtLlmArgs Pydantic dataclass (TRT-specific engine_params)
- expected_namespaces: ['engine_params']
- attempts: 1
- elapsed_sec: 108.24
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser extracting the parameter schema for the
tensorrt library, version 1.2.1.

You will be shown ONE CHUNK of source code. Extract the parameter
schema for the fields visible in this chunk. Other chunks are mined
separately and merged later - do not include fields from outside
this chunk.

OUTPUT FORMAT: a single JSON object matching EXACTLY this shape:

{
  "engine": "tensorrt",
  "engine_version": "1.2.1",
  "chunk_name": "trt_llm_args_class",
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
TrtLlmArgs extends BaseLlmArgs with TRT-specific build/calib fields (extended_runtime_perf_knob_config, calib_config, build_config, fast_build, enable_build_cache). Extract each Field(...) as engine_params.

=== SOURCE: TrtLlmArgs ===
class TrtLlmArgs(BaseLlmArgs):
    enable_tqdm: bool = Field(default=False,
                              description="Enable tqdm for progress bar.")

    workspace: Optional[str] = Field(default=None,
                                     description="The workspace for the model.")

    # Once set, the model will reuse the build_cache
    enable_build_cache: object = Field(
        default=False,
        description="Enable build cache.",
        json_schema_extra={
            "type": f"Union[{get_type_repr(BuildCacheConfig)}, bool]"
        })

    extended_runtime_perf_knob_config: Optional[
        ExtendedRuntimePerfKnobConfig] = Field(
            default=None, description="Extended runtime perf knob config.")

    calib_config: Optional[CalibConfig] = Field(
        default=None, description="Calibration config.", validate_default=True)

    # Quantization and calibration configurations
    quant_config: Optional[QuantConfig] = Field(
        default=None, description="Quantization config.", validate_default=True)

    embedding_parallel_mode: str = Field(
        default='SHARDING_ALONG_VOCAB',
        description="The embedding parallel mode.")

    fast_build: bool = Field(default=False, description="Enable fast build.")

    # BuildConfig is introduced to give users a familiar interface to configure the model building.
    build_config: Optional[BuildConfig] = Field(default=None,
                                                description="Build config.")

    # Prompt adapter arguments
    enable_prompt_adapter: bool = Field(default=False,
                                        description="Enable prompt adapter.")

    max_prompt_adapter_token: int = Field(
        default=0, description="The maximum number of prompt adapter tokens.")

    batching_type: Optional[BatchingType] = Field(default=None,
                                                  description="Batching type.")

    normalize_log_probs: bool = Field(
        default=False, description="Normalize log probabilities.")

    # Private attributes
    # This is used to hold the options for convert_checkpoint
    _convert_checkpoint_options: Dict[str,
                                      Any] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def init_build_config(self):
        """
        Creating a default BuildConfig if none is provided
        """
        build_config = getattr(self, "build_config", None)
        if build_config is None:
            kwargs = {}
            if self.max_batch_size:
                kwargs["max_batch_size"] = self.max_batch_size
            if self.max_num_tokens:
                kwargs["max_num_tokens"] = self.max_num_tokens
            if self.max_seq_len:
                kwargs["max_seq_len"] = self.max_seq_len
            if self.max_beam_width:
                kwargs["max_beam_width"] = self.max_beam_width
            if self.max_input_len:
                kwargs["max_input_len"] = self.max_input_len
            self.build_config = BuildConfig(**kwargs)
        return self

    @model_validator(mode="after")
    def validate_build_config_with_runtime_params(self):
        # Note: max_batch_size and max_num_tokens in LlmArgs are for runtime,
        # which will be passed to the C++ Executor API, overwriting the values
        # from an built engine. In order to set build configuration, it is
        # recommended to use build_config instead.
        assert isinstance(
            self.build_config, BuildConfig
        ), f"build_config is not initialized: {self.build_config}"

        if self.max_batch_size is not None:
            if self.max_batch_size > self.build_con
...<truncated>...
```

### Response

```
{
  "engine": "tensorrt",
  "engine_version": "1.2.1",
  "chunk_name": "trt_llm_args_class",
  "chunk_fields": {
    "enable_tqdm": {"namespace": "engine_params", "type": "boolean", "default": false, "description": "Enable tqdm for progress bar."},
    "workspace": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "null"}], "default": null, "description": "The workspace for the model."},
    "enable_build_cache": {"namespace": "engine_params", "anyOf": [{"type": "object", "description": "BuildCacheConfig"}, {"type": "boolean"}], "default": false, "description": "Enable build cache."},
    "extended_runtime_perf_knob_config": {"namespace": "engine_params", "anyOf": [{"type": "object", "description": "ExtendedRuntimePerfKnobConfig"}, {"type": "null"}], "default": null, "description": "Extended runtime perf knob config."},
    "calib_config": {"namespace": "engine_params", "anyOf": [{"type": "object", "description": "CalibConfig"}, {"type": "null"}], "default": null, "description": "Calibration config.", "validate_default": true},
    "quant_config": {"namespace": "engine_params", "anyOf": [{"type": "object", "description": "QuantConfig"}, {"type": "null"}], "default": null, "description": "Quantization config.", "validate_default": true},
    "embedding_parallel_mode": {"namespace": "engine_params", "type": "string", "default": "SHARDING_ALONG_VOCAB", "description": "The embedding parallel mode."},
    "fast_build": {"namespace": "engine_params", "type": "boolean", "default": false, "description": "Enable fast build."},
    "build_config": {"namespace": "engine_params", "anyOf": [{"type": "object", "description": "BuildConfig"}, {"type": "null"}], "default": null, "description": "Build config."},
    "enable_prompt_adapter": {"namespace": "engine_params", "type": "boolean", "default": false, "description": "Enable prompt adapter."},
    "max_prompt_adapter_token": {"namespace": "engine_params", "type": "integer", "default": 0, "description": "The maximum number of prompt adapter tokens."},
    "batching_type": {"namespace": "engine_params", "anyOf": [{"type": "object", "description": "BatchingType"}, {"type": "null"}], "default": null, "description": "Batching type."},
    "normalize_log_probs": {"namespace": "engine_params", "type": "boolean", "default": false, "description": "Normalize log probabilities."}
  }
}
```
