# schema extraction transcript: generation_config_docstring

- chunk_description: GenerationConfig docstring FULL (documents all kwargs.pop'd sampling params)
- expected_namespaces: ['sampling_params']
- attempts: 1
- elapsed_sec: 595.28
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser extracting the parameter schema for the
transformers library, version 5.9.0.

You will be shown ONE CHUNK of source code. Extract the parameter
schema for the fields visible in this chunk. Other chunks are mined
separately and merged later - do not include fields from outside
this chunk.

OUTPUT FORMAT: a single JSON object matching EXACTLY this shape:

{
  "engine": "transformers",
  "engine_version": "5.9.0",
  "chunk_name": "generation_config_docstring",
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

EXPECTED NAMESPACES FOR THIS CHUNK: sampling_params
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
Below is GenerationConfig's FULL docstring. It documents every sampling param the __init__ kwargs.pop's. Extract as sampling_params using the Sphinx pattern `name (type, *optional*, defaults to X)`.

=== SOURCE: GenerationConfig docstring ===
Class that holds a configuration for a generation task. A `generate` call supports the following generation methods
for text-decoder, text-to-text, speech-to-text, and vision-to-text models:

    - *greedy decoding* if `num_beams=1` and `do_sample=False`
    - *multinomial sampling* if `num_beams=1` and `do_sample=True`
    - *beam-search decoding* if `num_beams>1` and `do_sample=False`
    - *beam-search multinomial sampling* if `num_beams>1` and `do_sample=True`
    - *assisted decoding* if `assistant_model` or `prompt_lookup_num_tokens` is passed to `.generate()`

To learn more about decoding strategies refer to the [text generation strategies guide](../generation_strategies).

<Tip>

A large number of these flags control the logits or the stopping criteria of the generation. Make sure you check
the [generate-related classes](https://huggingface.co/docs/transformers/internal/generation_utils) for a full
description of the possible manipulations, as well as examples of their usage.

</Tip>

Note: the configuration fields that are still `None` will be overridden by `GenerationConfig._get_default_generation_params()`
during the generation loop. If you want to use different values for these fields, make sure to explicitly set them in the
generation config.

Args:
    > Parameters that control the length of the output

    max_length (`int`, *optional*):
        `max_new_tokens` is recommended for controlling how many tokens the model generates.
        `max_length` remains for backward compatibility.

    max_new_tokens (`int`, *optional*):
        The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
    min_length (`int`, *optional*):
        The minimum length of the sequence to be generated. Corresponds to the length of the input prompt +
        `min_new_tokens`. Its effect is overridden by `min_new_tokens`, if also set.
    min_new_tokens (`int`, *optional*):
        The minimum numbers of tokens to generate, ignoring the number of tokens in the prompt.
    early_stopping (`bool` or `str`, *optional*):
        Controls the stopping condition for beam-based methods, like beam-search. It accepts the following values:
        `True`, where the generation stops as soon as there are `num_beams` complete candidates; `False`, where an
        heuristic is applied and the generation stops when is it very unlikely to find better candidates;
        `"never"`, where the beam search procedure only stops when there cannot be better candidates (canonical
        beam search algorithm).
    max_time (`float`, *optional*):
        The maximum amount of time you allow the computation to run for in seconds. generation will still finish
        the current pass after allocated time has been passed.
    stop_strings (`str or list[str]`, *optional*):
        A string or a list of strings that should terminate generation if the model outputs them.

    > Parameters that control the generation strategy used

    do_sample (`bool`):
        Whether or not to use sampling ; use greedy decoding otherwise.
    num_beams (`int`, *optional*):
        Number of beams for beam search. 1 means no beam search.

    > Parameters that control the cache

    use_cache (`bool`):
        Whether or not the model should use the past last key/values attentions (if applicable to the model) to
        speed up decoding.
    cache_implementation (`str`, *optional*):
        Name of the cache class that will be instantiated in `generate`, for faster decoding. Possible values are:

        - `"dynamic"`: [`DynamicCache`]
        - `"static"`: [`StaticCache`]
        - `"offloaded"`: [`Dy
...<truncated>...
```

### Response

```
{
  "engine": "transformers",
  "engine_version": "5.9.0",
  "chunk_name": "generation_config_docstring",
  "chunk_fields": {
    "max_length": {"namespace": "sampling_params", "type": "integer"},
    "max_new_tokens": {"namespace": "sampling_params", "type": "integer"},
    "min_length": {"namespace": "sampling_params", "type": "integer"},
    "min_new_tokens": {"namespace": "sampling_params", "type": "integer"},
    "early_stopping": {"namespace": "sampling_params", "anyOf": [{"type": "boolean"}, {"type": "string"}]},
    "max_time": {"namespace": "sampling_params", "type": "number"},
    "stop_strings": {"namespace": "sampling_params", "anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]},
    "do_sample": {"namespace": "sampling_params", "type": "boolean"},
    "num_beams": {"namespace": "sampling_params", "type": "integer"},
    "use_cache": {"namespace": "sampling_params", "type": "boolean"},
    "cache_implementation": {"namespace": "sampling_params", "type": "string", "enum": ["dynamic", "static", "offloaded", "offloaded_static", "quantized"]},
    "cache_config": {"namespace": "sampling_params", "type": "object"},
    "temperature": {"namespace": "sampling_params", "type": "number"},
    "top_k": {"namespace": "sampling_params", "type": "integer"},
    "top_p": {"namespace": "sampling_params", "type": "number"},
    "min_p": {"namespace": "sampling_params", "type": "number"},
    "top_h": {"namespace": "sampling_params", "type": "number"},
    "typical_p": {"namespace": "sampling_params", "type": "number"},
    "epsilon_cutoff": {"namespace": "sampling_params", "type": "number"},
    "eta_cutoff": {"namespace": "sampling_params", "type": "number"},
    "repetition_penalty": {"namespace": "sampling_params", "type": "number"},
    "encoder_repetition_penalty": {"namespace": "sampling_params", "type": "number"},
    "length_penalty": {"namespace": "sampling_params", "type": "number"},
    "no_repeat_ngram_size": {"namespace": "sampling_params", "type": "integer"},
    "bad_words_ids": {"namespace": "sampling_params", "type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
    "renormalize_logits": {"namespace": "sampling_params", "type": "boolean"},
    "forced_bos_token_id": {"namespace": "sampling_params", "type": "integer"},
    "forced_eos_token_id": {"namespace": "sampling_params", "anyOf": [{"type": "integer"}, {"type": "array", "items": {"type": "integer"}}]},
    "remove_invalid_values": {"namespace": "sampling_params", "type": "boolean"},
    "exponential_decay_length_penalty": {"namespace": "sampling_params", "type": "array", "items": [{"type": "integer"}, {"type": "number"}]},
    "suppress_tokens": {"namespace": "sampling_params", "type": "array", "items": {"type": "integer"}},
    "begin_suppress_tokens": {"namespace": "sampling_params", "type": "array", "items": {"type": "integer"}},
    "sequence_bias": {"namespace": "sampling_params", "type": "object"},
    "token_healing": {"namespace": "sampling_params", "type": "boolean"},
    "guidance_scale": {"namespace": "sampling_params", "type": "number"},
    "watermarking_config": {"namespace": "sampling_params", "anyOf": [{"type": "object"}, {"type": "null"}]},
    "num_return_sequences": {"namespace": "sampling_params", "type": "integer"},
    "output_attentions": {"namespace": "sampling_params", "type": "boolean"},
    "output_hidden_states": {"namespace": "sampling_params", "type": "boolean"},
    "output_scores": {"namespace": "sampling_params", "type": "boolean"},
    "output_logits": {"namespace": "sampling_params", "type": "boolean"},
    "return_dict_in_generate": {"namespace": "sampling_params", "type": "boolean"},
    "pad_token_id": {"namespace": "sampling_params", "type": "integer"},
    "bos_token_id": {"namespace": "sampling_params", "type": "integer"},
    "eos_token_id": {"namespace": "sampling_params", "anyOf": [{"type": "integer"}, {"type": "array", "items": {"type": "integer"}}]},
    "encoder_no_repeat_ngram_size": {"namespace": "sampling_params", "type": "integer"},
    "decoder_start_token_id": {"namespace": "sampling_params", "anyOf": [{"type": "integer"}, {"type": "array", "items": {"type": "integer"}}]},
    "is_assistant": {"namespace": "sampling_params", "type": "boolean"},
    "num_assistant_tokens": {"namespace": "sampling_params", "type": "integer"},
    "num_assistant_tokens_schedule": {"namespace": "sampling_params", "anyOf": [{"type": "string"}, {"type": "null"}]},
    "assistant_confidence_threshold": {"namespace": "sampling_params", "type": "number"}
  }
}
```
