# schema extraction transcript: scheduler_config_class

- chunk_description: vllm.SchedulerConfig dataclass (scheduler engine_params)
- expected_namespaces: ['engine_params']
- attempts: 1
- elapsed_sec: 45.11
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

    runner_type: str = "generate"  # The runner type to launch for the model.

    # Maximum number of tokens to be processed in a single iteration.
    max_num_batched_tokens: int = field(default=None)  # type: ignore

    # Maximum number of sequences to be processed in a single iteration.
    max_num_seqs: int = 128

    # Maximum length of a sequence (including prompt and generated text).
    max_model_len: int = 8192

    # Maximum number of sequences that can be partially prefilled concurrently
    max_num_partial_prefills: int = 1

    # Maximum number of "very long prompt" sequences that can be prefilled
    # concurrently (long is defined by long_prefill_threshold)
    max_long_partial_prefills: int = 1

    # calculate context length that determines which sequences are
    # considered "long"
    long_prefill_token_threshold: int = 0

    # The number of slots to allocate per sequence per
    # step, beyond the known token ids. This is used in speculative
    # decoding to store KV activations of tokens which may or may not be
    # accepted.
    num_lookahead_slots: int = 0

    # Apply a delay (of delay factor multiplied by previous
    # prompt latency) before scheduling next prompt.
    delay_factor: float = 0.0

    # If True, prefill requests can be chunked based
    # on the remaining max_num_batched_tokens.
    enable_chunked_prefill: bool = False

    is_multimodal_model: bool = False

    # NOTE: The following multimodal encoder budget will be initialized to
    # max_num_batched_tokens and overridden in case max multimodal embedding
    # size is larger.
    # TODO (ywang96): Make these configurable.
    # Multimodal encoder compute budget, only used in V1
    max_num_encoder_input_tokens: int = field(default=None)  # type: ignore

    # Multimodal encoder cache size, only used in V1
    encoder_cache_size: int = field(default=None)  # type: ignore

    # Whether to perform preemption by swapping or
    # recomputation. If not specified, we determine the mode as follows:
    # We use recomputation by default since it incurs lower overhead than
    # swapping. However, when the sequence group has multiple sequences
    # (e.g., beam search), recomputation is not currently supported. In
    # such a case, we use swapping instead.
    preemption_mode: Optional[str] = None

    num_scheduler_steps: int = 1

    multi_step_stream_outputs: bool = False

    # Private API. If used, scheduler sends delta data to
    # workers instead of an entire data. It should be enabled only
    # when SPMD worker architecture is enabled. I.e.,
    # VLLM_USE_RAY_SPMD_WORKER=1
    send_delta_data: bool = False

    # The scheduling policy to use. "fcfs" (default) or "priority".
    policy: str = "fcfs"

    chunked_prefill_enabled: bool = field(init=False)

    # scheduler class or path. "vllm.core.scheduler.Scheduler" (default)
    # or "mod.custom_class".
    scheduler_cls: Union[str, Type[object]] = "vllm.core.scheduler.Scheduler"

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
        # no factors to consider.
        # this config will not affect the computation graph.
        factors: List[Any] = []
        hash_str = hashlib.md5(str(factors).encode()).hexdigest()
        return hash_str

    def __post_init__(self) -> None:
        if self.max_num_batched_tokens is None:
            if 
...<truncated>...
```

### Response

```
{
  "engine": "vllm",
  "engine_version": "0.7.3",
  "chunk_name": "scheduler_config_class",
  "chunk_fields": {
    "runner_type": {"namespace": "engine_params", "type": "string", "default": "generate"},
    "max_num_batched_tokens": {"namespace": "engine_params", "type": "integer", "default": null},
    "max_num_seqs": {"namespace": "engine_params", "type": "integer", "default": 128},
    "max_model_len": {"namespace": "engine_params", "type": "integer", "default": 8192},
    "max_num_partial_prefills": {"namespace": "engine_params", "type": "integer", "default": 1},
    "max_long_partial_prefills": {"namespace": "engine_params", "type": "integer", "default": 1},
    "long_prefill_token_threshold": {"namespace": "engine_params", "type": "integer", "default": 0},
    "num_lookahead_slots": {"namespace": "engine_params", "type": "integer", "default": 0},
    "delay_factor": {"namespace": "engine_params", "type": "number", "default": 0.0},
    "enable_chunked_prefill": {"namespace": "engine_params", "type": "boolean", "default": false},
    "is_multimodal_model": {"namespace": "engine_params", "type": "boolean", "default": false},
    "max_num_encoder_input_tokens": {"namespace": "engine_params", "type": "integer", "default": null},
    "encoder_cache_size": {"namespace": "engine_params", "type": "integer", "default": null},
    "preemption_mode": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "null"}], "default": null},
    "num_scheduler_steps": {"namespace": "engine_params", "type": "integer", "default": 1},
    "multi_step_stream_outputs": {"namespace": "engine_params", "type": "boolean", "default": false},
    "send_delta_data": {"namespace": "engine_params", "type": "boolean", "default": false},
    "policy": {"namespace": "engine_params", "type": "string", "default": "fcfs"},
    "scheduler_cls": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "object"}], "default": "vllm.core.scheduler.Scheduler"}
  }
}
```
