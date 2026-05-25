# schema extraction transcript: sampling_params_class

- chunk_description: vllm.SamplingParams msgspec.Struct fields + GuidedDecodingParams companion
- expected_namespaces: ['sampling_params', '$defs.GuidedDecodingParams']
- attempts: 1
- elapsed_sec: 72.52
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
  "chunk_name": "sampling_params_class",
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

EXPECTED NAMESPACES FOR THIS CHUNK: sampling_params, $defs.GuidedDecodingParams
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

=== SOURCE: vllm.SamplingParams (msgspec.Struct definition) ===
class SamplingParams(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
        # required for @cached_property.
        dict=True):  # type: ignore[call-arg]
    """Sampling parameters for text generation.

    Overall, we follow the sampling parameters from the OpenAI text completion
    API (https://platform.openai.com/docs/api-reference/completions/create).
    In addition, we support beam search, which is not supported by OpenAI.

    Args:
        n: Number of output sequences to return for the given prompt.
        best_of: Number of output sequences that are generated from the prompt.
            From these `best_of` sequences, the top `n` sequences are returned.
            `best_of` must be greater than or equal to `n`. This is treated as
            the beam width when `use_beam_search` is True. By default, `best_of`
            is set to `n`.
        presence_penalty: Float that penalizes new tokens based on whether they
            appear in the generated text so far. Values > 0 encourage the model
            to use new tokens, while values < 0 encourage the model to repeat
            tokens.
        frequency_penalty: Float that penalizes new tokens based on their
            frequency in the generated text so far. Values > 0 encourage the
            model to use new tokens, while values < 0 encourage the model to
            repeat tokens.
        repetition_penalty: Float that penalizes new tokens based on whether
            they appear in the prompt and the generated text so far. Values > 1
            encourage the model to use new tokens, while values < 1 encourage
            the model to repeat tokens.
        temperature: Float that controls the randomness of the sampling. Lower
            values make the model more deterministic, while higher values make
            the model more random. Zero means greedy sampling.
        top_p: Float that controls the cumulative probability of the top tokens
            to consider. Must be in (0, 1]. Set to 1 to consider all tokens.
        top_k: Integer that controls the number of top tokens to consider. Set
            to -1 to consider all tokens.
        min_p: Float that represents the minimum probability for a token to be
            considered, relative to the probability of the most likely token.
            Must be in [0, 1]. Set to 0 to disable this.
        seed: Random seed to use for the generation.
        use_beam_search: Whether to use beam search instead of sampling.
        length_penalty: Float that penalizes sequences based on their length.
            Used in beam search.
        early_stopping: Controls the stopping condition for beam search. It
            accepts the following values: `True`, where the generation stops as
            soon as there are `best_of` complete candidates; `False`, where an
            heuristic is applied and the generation stops when is it very
            unlikely to find better candidates; `"never"`, where the beam search
            procedure only stops when there cannot be better candidates
            (canonical beam search algorithm).
        stop: List of strings that stop the generation when they are generated.
            The returned output will not contain the stop strings.
        stop_token_ids: List of tokens that stop the generation when they are
            generated. The returned output will contain the stop tokens unless
            the stop tokens are special tokens.
        include_stop_str_in_output: Whether to include the stop strings in
            output text. Defaults to False.
        ignore_eos: Whether to ignore the EOS token and continue generating
            tokens after the EOS token is generated.
        max_tokens: Maximum number of tokens to generate per output sequence.
        min_tokens
...<truncated>...
```

### Response

```
{
  "engine": "vllm",
  "engine_version": "0.6.0",
  "chunk_name": "sampling_params_class",
  "chunk_fields": {
    "n": {"namespace": "sampling_params", "type": "integer", "default": 1},
    "best_of": {"namespace": "sampling_params", "type": "integer", "default": null},
    "presence_penalty": {"namespace": "sampling_params", "type": "number", "default": 0.0},
    "frequency_penalty": {"namespace": "sampling_params", "type": "number", "default": 0.0},
    "repetition_penalty": {"namespace": "sampling_params", "type": "number", "default": 1.0},
    "temperature": {"namespace": "sampling_params", "type": "number", "default": 1.0},
    "top_p": {"namespace": "sampling_params", "type": "number", "default": 1.0},
    "top_k": {"namespace": "sampling_params", "type": "integer", "default": -1},
    "min_p": {"namespace": "sampling_params", "type": "number", "default": 0.0},
    "seed": {"namespace": "sampling_params", "type": "integer", "default": null},
    "use_beam_search": {"namespace": "sampling_params", "type": "boolean", "default": false},
    "length_penalty": {"namespace": "sampling_params", "type": "number", "default": 1.0},
    "early_stopping": {"namespace": "sampling_params", "anyOf": [{"type": "boolean"}, {"type": "string"}], "default": false},
    "stop": {"namespace": "sampling_params", "anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}], "default": null},
    "stop_token_ids": {"namespace": "sampling_params", "type": "array", "items": {"type": "integer"}, "default": null},
    "ignore_eos": {"namespace": "sampling_params", "type": "boolean", "default": false},
    "max_tokens": {"namespace": "sampling_params", "type": "integer", "default": 16},
    "min_tokens": {"namespace": "sampling_params", "type": "integer", "default": 0},
    "logprobs": {"namespace": "sampling_params", "type": "integer", "default": null},
    "prompt_logprobs": {"namespace": "sampling_params", "type": "integer", "default": null},
    "detokenize": {"namespace": "sampling_params", "type": "boolean", "default": true},
    "skip_special_tokens": {"namespace": "sampling_params", "type": "boolean", "default": true},
    "spaces_between_special_tokens": {"namespace": "sampling_params", "type": "boolean", "default": true},
    "logits_processors": {"namespace": "sampling_params", "anyOf": [{"type": "array", "items": {"type": "object"}}, {"type": "null"}], "default": null},
    "include_stop_str_in_output": {"namespace": "sampling_params", "type": "boolean", "default": false},
    "truncate_prompt_tokens": {"namespace": "sampling_params", "anyOf": [{"type": "integer"}, {"type": "null"}], "default": null}
  }
}
```
