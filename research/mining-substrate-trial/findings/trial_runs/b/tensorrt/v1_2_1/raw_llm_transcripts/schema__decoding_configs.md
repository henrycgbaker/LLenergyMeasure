# schema extraction transcript: decoding_configs

- chunk_description: tensorrt_llm decoding configs (Lookahead, Medusa, Eagle, NGram, MTP)
- expected_namespaces: ['engine_params', '$defs.LookaheadDecodingConfig', '$defs.MedusaDecodingConfig', '$defs.EagleDecodingConfig', '$defs.NGramDecodingConfig', '$defs.MTPDecodingConfig']
- attempts: 1
- elapsed_sec: 153.74
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
  "chunk_name": "decoding_configs",
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

EXPECTED NAMESPACES FOR THIS CHUNK: engine_params, $defs.LookaheadDecodingConfig, $defs.MedusaDecodingConfig, $defs.EagleDecodingConfig, $defs.NGramDecodingConfig, $defs.MTPDecodingConfig
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
Speculative decoding configs (one per algorithm). Each has a small Pydantic field set; LookaheadDecodingConfig has a @field_validator on three positive-value fields.

=== SOURCE: DecodingBaseConfig ===
class DecodingBaseConfig(StrictBaseModel):
    # The number of the drafter layers.
    max_draft_len: Optional[int] = None
    # The number of draft tokens in the draft tokens tree.
    # If it's a linear tree, each draft layer will only generate one draft token.
    # In this case, max_draft_len == max_total_draft_tokens.
    # If it's a static or dynamic tree, each draft layer may generate more than one draft token.
    # In this case, max_total_draft_tokens >= max_draft_len.
    max_total_draft_tokens: Optional[int] = None
    # The speculative (draft) model. Accepts either:
    # - A HuggingFace Hub model ID (str), e.g., "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
    #   which will be automatically downloaded.
    # - A local filesystem path to a downloaded model directory.
    speculative_model: Optional[Union[str, Path]] = Field(
        default=None,
        validation_alias=AliasChoices("speculative_model",
                                      "speculative_model_dir"))

    # PyTorch only.
    # When specified, speculation will be disabled at batch sizes above
    # this value. Otherwise, speculation will always be on.
    max_concurrency: Optional[int] = None

    # Developer interface: dynamically adjust draft length based on active batch size in runtime.
    # Maps batch size to draft lengths. For example:
    # {1: 4, 4: 2, 8: 0} means:
    # - batch_size >= 1: use draft_len=4
    # - batch_size >= 4: use draft_len=2
    # - batch_size >= 8: use draft_len=0 (disable speculation)
    # draft_len_schedule is enforced to contain batch_size=1 and its according draft_len equals max_draft_len for consistency
    # for example, if max_draft_len=4, the schedule must contain {1: 4}
    draft_len_schedule: Optional[dict[int, int]] = None

    load_format: Optional[str] = None
    # PyTorch only.
    # Rolling average window size (N) for acceptance length across completed requests.
    # If not set or set to 0, the feature is disabled.
    acceptance_window: Optional[i

=== SOURCE: LookaheadDecodingConfig ===
class LookaheadDecodingConfig(DecodingBaseConfig, PybindMirror):
    """
    Configuration for lookahead speculative decoding.
    """

    max_window_size: int = Field(
        default=_LookaheadDecodingConfig.get_default_lookahead_decoding_window(
        ),
        description="Number of NGrams in lookahead branch per step.")
    max_ngram_size: int = Field(
        default=_LookaheadDecodingConfig.get_default_lookahead_decoding_ngram(),
        description="Number of tokens per NGram.")
    max_verification_set_size: int = Field(
        default=_LookaheadDecodingConfig.
        get_default_lookahead_decoding_verification_set(),
        description="Number of NGrams in verification branch per step.")

    @field_validator('max_window_size', 'max_ngram_size',
                     'max_verification_set_size')
    @classmethod
    def validate_positive_values(cls, v):
        if v <= 0:
            raise ValueError(f"Value must be positive, got {v}")
        return v

    def __init__(self, **data):
        super().__init__(**data)
        self.max_total_draft_tokens = self.max_draft_len  # Current Lookahead only support linear tree
        self._check_fields()

    def calculate_speculative_resource(self):
        return _LookaheadDecodingConfig.calculate_speculative_resource_tuple(
            self.max_window_size, self.max_ngram_size,
            self.max_verification_set_size)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    def _to_pybind(self):
        return _Lo
...<truncated>...
```

### Response

```
{
  "engine": "tensorrt",
  "engine_version": "1.2.1",
  "chunk_name": "decoding_configs",
  "chunk_fields": {
    "max_draft_len": {"namespace": "engine_params", "type": "integer", "default": null},
    "max_total_draft_tokens": {"namespace": "engine_params", "type": "integer", "default": null},
    "speculative_model": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "object", "description": "PathLike"}, {"type": "null"}], "default": null},
    "max_concurrency": {"namespace": "engine_params", "type": "integer", "default": null},
    "draft_len_schedule": {"namespace": "engine_params", "type": "object", "default": null},
    "load_format": {"namespace": "engine_params", "type": "string", "default": null},
    "acceptance_window": {"namespace": "engine_params", "type": "integer", "default": null},
    "max_window_size": {"namespace": "$defs.LookaheadDecodingConfig", "type": "integer"},
    "max_ngram_size": {"namespace": "$defs.LookaheadDecodingConfig", "type": "integer"},
    "max_verification_set_size": {"namespace": "$defs.LookaheadDecodingConfig", "type": "integer"},
    "medusa_choices": {"namespace": "$defs.MedusaDecodingConfig", "anyOf": [{"type": "array"}, {"type": "null"}], "default": null},
    "num_medusa_heads": {"namespace": "$defs.MedusaDecodingConfig", "type": "integer", "default": null},
    "eagle_choices": {"namespace": "$defs.EagleDecodingConfig", "anyOf": [{"type": "array"}, {"type": "null"}], "default": null},
    "greedy_sampling": {"namespace": "$defs.EagleDecodingConfig", "type": "boolean", "default": true},
    "posterior_threshold": {"namespace": "$defs.EagleDecodingConfig", "type": "number", "default": null},
    "use_dynamic_tree": {"namespace": "$defs.EagleDecodingConfig", "type": "boolean", "default": false},
    "dynamic_tree_max_topK": {"namespace": "$defs.EagleDecodingConfig", "type": "integer", "default": null},
    "num_eagle_layers": {"namespace": "$defs.EagleDecodingConfig", "type": "integer", "default": null},
    "max_non_leaves_per_layer": {"namespace": "$defs.EagleDecodingConfig", "type": "integer", "default": null},
    "eagle3_one_model": {"namespace": "$defs.EagleDecodingConfig", "type": "boolean", "default": true},
    "eagle3_layers_to_capture": {"namespace": "$defs.EagleDecodingConfig", "anyOf": [{"type": "array"}, {"type": "null"}], "default": null},
    "eagle3_model_arch": {"namespace": "$defs.EagleDecodingConfig", "type": "string"},
    "max_matching_ngram_size": {"namespace": "$defs.NGramDecodingConfig", "type": "integer"},
    "is_keep_all": {"namespace": "$defs.NGramDecodingConfig", "type": "boolean", "default": true},
    "is_use_oldest": {"namespace": "$defs.NGramDecodingConfig", "type": "boolean", "default": true},
    "is_public_pool": {"namespace": "$defs.NGramDecodingConfig", "type": "boolean", "default": true},
    "num_nextn_predict_layers": {"namespace": "$defs.MTPDecodingConfig", "type": "integer"},
    "use_relaxed_acceptance_for_thinking": {"namespace": "$defs.MTPDecodingConfig", "type": "boolean", "default": false},
    "relaxed_topk": {"namespace": "$defs.MTPDecodingConfig", "type": "integer"},
    "relaxed_delta": {"namespace": "$defs.MTPDecodingConfig", "type": "number", "default": 0.0},
    "use_mtp_vanilla": {"namespace": "$defs.MTPDecodingConfig", "type": "boolean", "default": false},
    "mtp_eagle_one_model": {"namespace": "$defs.MTPDecodingConfig", "type": "boolean", "default": true},
    "num_nextn_predict_layers_from_model_config": {"namespace": "$defs.MTPDecodingConfig", "type": "integer"},
    "begin_thinking_phase_token": {"namespace": "$defs.MTPDecodingConfig", "type": "integer"},
    "end_thinking_phase_token": {"namespace": "$defs.MTPDecodingConfig", "type": "integer"}
  }
}
```
