# schema extraction transcript: scheduler_config_class

- chunk_description: vllm.SchedulerConfig dataclass (scheduler engine_params)
- expected_namespaces: ['engine_params']
- attempts: 1
- elapsed_sec: 110.60
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
    """Scheduler configuration.

    Args:
        max_num_batched_tokens: Maximum number of tokens to be processed in
            a single iteration.
        max_num_seqs: Maximum number of sequences to be processed in a single
            iteration.
        max_model_len: Maximum length of a sequence (including prompt
            and generated text).
        use_v2_block_manager: Whether to use the BlockSpaceManagerV2 or not.
        num_lookahead_slots: The number of slots to allocate per sequence per
            step, beyond the known token ids. This is used in speculative
            decoding to store KV activations of tokens which may or may not be
            accepted.
        delay_factor: Apply a delay (of delay factor multiplied by previous
            prompt latency) before scheduling next prompt.
        enable_chunked_prefill: If True, prefill requests can be chunked based
            on the remaining max_num_batched_tokens.
        embedding_mode: Whether the running model is for embedding.
        preemption_mode: Whether to perform preemption by swapping or 
            recomputation. If not specified, we determine the mode as follows:
            We use recomputation by default since it incurs lower overhead than
            swapping. However, when the sequence group has multiple sequences
            (e.g., beam search), recomputation is not currently supported. In
            such a case, we use swapping instead.
        send_delta_data: Private API. If used, scheduler sends delta data to
            workers instead of an entire data. It should be enabled only
            when SPMD worker architecture is enabled. I.e.,
            VLLM_USE_RAY_SPMD_WORKER=1

    """

    def __init__(self,
                 max_num_batched_tokens: Optional[int],
                 max_num_seqs: int,
                 max_model_len: int,
                 use_v2_block_manager: bool = False,
                 num_lookahead_slots: int = 0,
                 delay_factor: float = 0.0,
                 enable_chunked_prefill: bool = False,
                 embedding_mode: bool = False,
                 is_multimodal_model: bool = False,
                 preemption_mode: Optional[str] = None,
                 num_scheduler_steps: int = 1,
                 send_delta_data: bool = False) -> None:
        if max_num_batched_tokens is None:
            if enable_chunked_prefill:
                # It is the values that have the best balance between ITL
                # and TTFT on A100. Note it is not optimized for throughput.
                max_num_batched_tokens = 512
            else:
                # If max_model_len is too short, use 2048 as the default value
                # for higher throughput.
                max_num_batched_tokens = max(max_model_len, 2048)

            if embedding_mode:
                # For embedding, choose specific value for higher throughput
                max_num_batched_tokens = max(
                    max_num_batched_tokens,
                    _EMBEDDING_MODEL_MAX_NUM_BATCHED_TOKENS,
                )
            if is_multimodal_model:
                # The value needs to be at least the number of multimodal tokens
                max_num_batched_tokens = max(
                    max_num_batched_tokens,
                    _MULTIMODAL_MODEL_MAX_NUM_BATCHED_TOKENS,
                )

        self.max_num_batched_tokens = max_num_batched_tokens

        if enable_chunked_prefill:
            logger.info(
                "Chunked prefill is enabled with max_num_batched_tokens=%d.",
                self.max_num_batched_tokens)

        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.use_v2_block_manager = use_v2_block_manager
        self.num_lookahead_slots = num_lookahead_slots
        self.delay_factor = delay
...<truncated>...
```

### Response

```
{
  "engine": "vllm",
  "engine_version": "0.6.0",
  "chunk_name": "scheduler_config_class",
  "chunk_fields": {
    "max_num_batched_tokens": {
      "namespace": "engine_params",
      "type": "integer",
      "default": null,
      "description": "Maximum number of tokens to be processed in a single iteration."
    },
    "max_num_seqs": {
      "namespace": "engine_params",
      "type": "integer",
      "default": null,
      "description": "Maximum number of sequences to be processed in a single iteration."
    },
    "max_model_len": {
      "namespace": "engine_params",
      "type": "integer",
      "default": null,
      "description": "Maximum length of a sequence (including prompt and generated text)."
    },
    "use_v2_block_manager": {
      "namespace": "engine_params",
      "type": "boolean",
      "default": false,
      "description": "Whether to use the BlockSpaceManagerV2 or not."
    },
    "num_lookahead_slots": {
      "namespace": "engine_params",
      "type": "integer",
      "default": 0,
      "description": "The number of slots to allocate per sequence per step, beyond the known token ids."
    },
    "delay_factor": {
      "namespace": "engine_params",
      "type": "number",
      "default": 0.0,
      "description": "Apply a delay (of delay factor multiplied by previous prompt latency) before scheduling next prompt."
    },
    "enable_chunked_prefill": {
      "namespace": "engine_params",
      "type": "boolean",
      "default": false,
      "description": "If True, prefill requests can be chunked based on the remaining max_num_batched_tokens."
    },
    "embedding_mode": {
      "namespace": "engine_params",
      "type": "boolean",
      "default": false,
      "description": "Whether the running model is for embedding."
    },
    "is_multimodal_model": {
      "namespace": "engine_params",
      "type": "boolean",
      "default": false,
      "description": ""
    },
    "preemption_mode": {
      "namespace": "engine_params",
      "anyOf": [{"type": "string"}, {"type": "null"}],
      "default": null,
      "description": "Whether to perform preemption by swapping or recomputation."
    },
    "num_scheduler_steps": {
      "namespace": "engine_params",
      "type": "integer",
      "default": 1,
      "description": ""
    },
    "send_delta_data": {
      "namespace": "engine_params",
      "type": "boolean",
      "default": false,
      "description": "Private API. If used, scheduler sends delta data to workers instead of an entire data."
    }
  }
}
```
