# schema extraction transcript: scheduler_config_class

- chunk_description: vllm.SchedulerConfig dataclass (scheduler engine_params)
- expected_namespaces: ['engine_params']
- attempts: 1
- elapsed_sec: 133.49
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
  "chunk_name": "scheduler_config_class",
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

=== SOURCE: vllm.SchedulerConfig (dataclass field list) ===
class SchedulerConfig:
    """Scheduler configuration."""

    runner_type: RunnerType = "generate"
    """The runner type to launch for the model."""

    max_num_batched_tokens: SkipValidation[int] = None  # type: ignore
    """Maximum number of tokens to be processed in a single iteration.

    This config has no static default. If left unspecified by the user, it will
    be set in `EngineArgs.create_engine_config` based on the usage context."""

    max_num_seqs: SkipValidation[int] = None  # type: ignore
    """Maximum number of sequences to be processed in a single iteration.

    This config has no static default. If left unspecified by the user, it will
    be set in `EngineArgs.create_engine_config` based on the usage context."""

    max_model_len: SkipValidation[int] = None  # type: ignore
    """Maximum length of a sequence (including prompt and generated text). This
    is primarily set in `ModelConfig` and that value should be manually
    duplicated here."""

    max_num_partial_prefills: int = 1
    """For chunked prefill, the maximum number of sequences that can be
    partially prefilled concurrently."""

    max_long_partial_prefills: int = 1
    """For chunked prefill, the maximum number of prompts longer than
    long_prefill_token_threshold that will be prefilled concurrently. Setting
    this less than max_num_partial_prefills will allow shorter prompts to jump
    the queue in front of longer prompts in some cases, improving latency."""

    long_prefill_token_threshold: int = 0
    """For chunked prefill, a request is considered long if the prompt is
    longer than this number of tokens."""

    num_lookahead_slots: int = 0
    """The number of slots to allocate per sequence per
    step, beyond the known token ids. This is used in speculative
    decoding to store KV activations of tokens which may or may not be
    accepted.

    NOTE: This will be replaced by speculative config in the future; it is
    present to enable correctness tests until then."""

    cuda_graph_sizes: list[int] = field(default_factory=lambda: [512])
    """Cuda graph capture sizes, default is 512.
    1. if one value is provided, then the capture list would follow the
    pattern: [1, 2, 4] + [i for i in range(8, cuda_graph_sizes + 1, 8)]
    2. more than one value (e.g. 1 2 128) is provided, then the capture list
    will follow the provided list."""

    delay_factor: float = 0.0
    """Apply a delay (of delay factor multiplied by previous
    prompt latency) before scheduling next prompt."""

    enable_chunked_prefill: SkipValidation[bool] = None  # type: ignore
    """If True, prefill requests can be chunked based
    on the remaining max_num_batched_tokens."""

    is_multimodal_model: bool = False
    """True if the model is multimodal."""

    # TODO (ywang96): Make this configurable.
    max_num_encoder_input_tokens: int = field(init=False)
    """Multimodal encoder compute budget, only used in V1.

    NOTE: This is not currently configurable. It will be overridden by
    max_num_batched_tokens in case max multimodal embedding size is larger."""

    # TODO (ywang96): Make this configurable.
    encoder_cache_size: int = field(init=False)
    """Multimodal encoder cache size, only used in V1.

    NOTE: This is not currently configurable. It will be overridden by
    max_num_batched_tokens in case max multimodal embedding size is larger."""

    preemption_mode: Optional[PreemptionMode] = None
    """Whether to perform preemption by swapping or
    recomputation. If not specified, we determine the mode as follows:
    We use recomputation by default since it incurs lower overhead than
    swapping. However, when the sequence group has multiple sequences
    (e.g., beam search), recomputation is not currently supported. In
    such a case, we use swapping instead."""

    num_scheduler_st
...<truncated>...
```

### Response

```
{
  "engine": "vllm",
  "engine_version": "0.9.2",
  "chunk_name": "scheduler_config_class",
  "chunk_fields": {
    "runner_type": {"namespace": "engine_params", "type": "string", "default": "generate"},
    "max_num_batched_tokens": {"namespace": "engine_params", "type": "integer", "default": null},
    "max_num_seqs": {"namespace": "engine_params", "type": "integer", "default": null},
    "max_model_len": {"namespace": "engine_params", "type": "integer", "default": null},
    "max_num_partial_prefills": {"namespace": "engine_params", "type": "integer", "default": 1},
    "max_long_partial_prefills": {"namespace": "engine_params", "type": "integer", "default": 1},
    "long_prefill_token_threshold": {"namespace": "engine_params", "type": "integer", "default": 0},
    "num_lookahead_slots": {"namespace": "engine_params", "type": "integer", "default": 0},
    "cuda_graph_sizes": {"namespace": "engine_params", "type": "array", "items": {"type": "integer"}, "default": [512]},
    "delay_factor": {"namespace": "engine_params", "type": "number", "default": 0.0},
    "enable_chunked_prefill": {"namespace": "engine_params", "type": "boolean", "default": null},
    "is_multimodal_model": {"namespace": "engine_params", "type": "boolean", "default": false},
    "max_num_encoder_input_tokens": {"namespace": "engine_params", "type": "integer"},
    "encoder_cache_size": {"namespace": "engine_params", "type": "integer"},
    "preemption_mode": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "null"}], "default": null},
    "num_scheduler_steps": {"namespace": "engine_params", "type": "integer", "default": 1},
    "multi_step_stream_outputs": {"namespace": "engine_params", "type": "boolean", "default": true},
    "send_delta_data": {"namespace": "engine_params", "type": "boolean", "default": false},
    "policy": {"namespace": "engine_params", "type": "string", "default": "fcfs"},
    "chunked_prefill_enabled": {"namespace": "engine_params", "type": "boolean"},
    "disable_chunked_mm_input": {"namespace": "engine_params", "type": "boolean", "default": false},
    "scheduler_cls": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "object"}], "default": "vllm.core.scheduler.Scheduler"}
  }
}
```
