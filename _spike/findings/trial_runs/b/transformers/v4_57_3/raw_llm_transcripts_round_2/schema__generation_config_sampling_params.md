# schema extraction transcript: generation_config_sampling_params

- chunk_description: GenerationConfig.__init__ + docstring (sampling_params)
- expected_namespaces: ['sampling_params']
- attempts: 1
- elapsed_sec: 115.90
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
  "chunk_name": "generation_config_sampling_params",
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

=== SOURCE: GenerationConfig.__init__ ===
    def __init__(self, **kwargs):
        # Parameters that control the length of the output
        self.max_length = kwargs.pop("max_length", 20)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)
        self.min_length = kwargs.pop("min_length", 0)
        self.min_new_tokens = kwargs.pop("min_new_tokens", None)
        self.early_stopping = kwargs.pop("early_stopping", False)
        self.max_time = kwargs.pop("max_time", None)
        self.stop_strings = kwargs.pop("stop_strings", None)

        # Parameters that control the generation strategy used
        self.do_sample = kwargs.pop("do_sample", False)
        self.num_beams = kwargs.pop("num_beams", 1)

        # Parameters that control the cache
        self.use_cache = kwargs.pop("use_cache", True)
        self.cache_implementation = kwargs.pop("cache_implementation", None)
        self.cache_config = kwargs.pop("cache_config", None)

        self.return_legacy_cache = kwargs.pop("return_legacy_cache", None)
        self.prefill_chunk_size = kwargs.pop("prefill_chunk_size", None)

        # Parameters for manipulation of the model output logits
        self.temperature = kwargs.pop("temperature", 1.0)
        self.top_k = kwargs.pop("top_k", 50)
        self.top_p = kwargs.pop("top_p", 1.0)
        self.min_p = kwargs.pop("min_p", None)
        self.typical_p = kwargs.pop("typical_p", 1.0)
        self.epsilon_cutoff = kwargs.pop("epsilon_cutoff", 0.0)
        self.eta_cutoff = kwargs.pop("eta_cutoff", 0.0)
        self.repetition_penalty = kwargs.pop("repetition_penalty", 1.0)
        self.encoder_repetition_penalty = kwargs.pop("encoder_repetition_penalty", 1.0)
        self.length_penalty = kwargs.pop("length_penalty", 1.0)
        self.no_repeat_ngram_size = kwargs.pop("no_repeat_ngram_size", 0)
        self.bad_words_ids = kwargs.pop("bad_words_ids", None)
        self.renormalize_logits = kwargs.pop("renormalize_logits", False)
        self.forced_bos_token_id = kwargs.pop("forced_bos_token_id", None)
        self.forced_eos_token_id = kwargs.pop("forced_eos_token_id", None)
        self.remove_invalid_values = kwargs.pop("remove_invalid_values", False)
        self.exponential_decay_length_penalty = kwargs.pop("exponential_decay_length_penalty", None)
        self.suppress_tokens = kwargs.pop("suppress_tokens", None)
        self.begin_suppress_tokens = kwargs.pop("begin_suppress_tokens", None)
        self.sequence_bias = kwargs.pop("sequence_bias", None)
        self.token_healing = kwargs.pop("token_healing", False)
        self.guidance_scale = kwargs.pop("guidance_scale", None)

        watermarking_config = kwargs.pop("watermarking_config", None)
        if watermarking_config is None:
            self.watermarking_config = None
        elif isinstance(watermarking_config, BaseWatermarkingConfig):
            self.watermarking_config = watermarking_config
        else:
            self.watermarking_config = WatermarkingConfig.from_dict(watermarking_config)

        # Parameters that define the output variables of `generate`
        self.num_return_sequences = kwargs.pop("num_return_sequences", 1)
        self.output_attentions = kwargs.pop("output_attentions", False)
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        self.output_scores = kwargs.pop("output_scores", False)
        self.output_logits = kwargs.pop("output_logits", None)
        self.return_dict_in_generate = kwargs.pop("return_dict_in_generate", False)

        # Special tokens that can be used at generation time
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)

        # Generation parameters exclusive to encoder-decoder models
        self.encoder_no_repeat_ngram_size 
...<truncated>...
```

### Response

```
{
  "engine": "transformers",
  "engine_version": "4.57.3",
  "chunk_name": "generation_config_sampling_params",
  "chunk_fields": {
    "max_length": {"namespace": "sampling_params", "type": "integer", "default": 20, "description": "The maximum length the generated tokens can have."},
    "max_new_tokens": {"namespace": "sampling_params", "type": "integer", "default": null, "description": "The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt."},
    "min_length": {"namespace": "sampling_params", "type": "integer", "default": 0, "description": "The minimum length of the sequence to be generated."},
    "min_new_tokens": {"namespace": "sampling_params", "type": "integer", "default": null, "description": "The minimum numbers of tokens to generate, ignoring the number of tokens in the prompt."},
    "early_stopping": {"namespace": "sampling_params", "type": "boolean", "default": false, "description": "Controls the stopping condition for beam-based methods, like beam-search."},
    "max_time": {"namespace": "sampling_params", "type": "number", "default": null, "description": "The maximum amount of time you allow the computation to run for in seconds."},
    "stop_strings": {"namespace": "sampling_params", "type": "string", "default": null, "description": "A string or a list of strings that should terminate generation if the model outputs them."},
    "do_sample": {"namespace": "sampling_params", "type": "boolean", "default": false, "description": "Whether or not to use sampling ; use greedy decoding otherwise."},
    "num_beams": {"namespace": "sampling_params", "type": "integer", "default": 1, "description": "Number of beams for beam search. 1 means no beam search."},
    "use_cache": {"namespace": "sampling_params", "type": "boolean", "default": true, "description": "Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding."},
    "cache_implementation": {"namespace": "sampling_params", "anyOf": [{"type": "string"}, {"type": "null"}], "default": null, "description": "Name of the cache class that will be instantiated in `generate`, for faster decoding."},
    "cache_config": {"namespace": "sampling_params", "type": "object", "default": null, "description": "Arguments used in the key-value cache class can be passed in `cache_config`."},
    "return_legacy_cache": {"namespace": "sampling_params", "type": "boolean", "default": true, "description": "Whether to return the legacy or new format of the cache when `DynamicCache` is used by default."},
    "temperature": {"namespace": "sampling_params", "type": "number", "default": 1.0, "description": "The value used to module the next token probabilities."},
    "top_k": {"namespace": "sampling_params", "type": "integer", "default": 50, "description": "The number of highest probability vocabulary tokens to keep for top-k-filtering."},
    "top_p": {"namespace": "sampling_params", "type": "number", "default": 1.0, "description": "If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or higher are kept for generation."},
    "min_p": {"namespace": "sampling_params", "type": "number", "default": null, "description": "Minimum token probability, which will be scaled by the probability of the most likely token."},
    "typical_p": {"namespace": "sampling_params", "type": "number", "default": 1.0, "description": "Local typicality measures how similar the conditional probability of predicting a target token next is to the expected conditional probability of predicting a random token next, given the partial text already generated."},
    "epsilon_cutoff": {"namespace": "sampling_params", "type": "number", "default": 0.0, "description": "If set to float strictly between 0 and 1, only tokens with a conditional probability greater than `epsilon_cutoff` will be sampled."},
    "eta_cutoff": {"namespace": "sampling_params", "type": "number", "default": 0.0, "description": "Eta sampling is a hybrid of locally typical sampling and epsilon sampling."},
    "repetition_penalty": {"namespace": "sampling_params", "type": "number", "default": 1.0, "description": "The parameter for repetition penalty."},
    "encoder_repetition_penalty": {"namespace": "sampling_params", "type": "number", "default": 1.0, "description": "The parameter for encoder_repetition_penalty."},
    "length_penalty": {"namespace": "sampling_params", "type": "number", "default": 1.0, "description": "Exponential penalty to the length that is used"},
    "no_repeat_ngram_size": {"namespace": "sampling_params", "type": "integer", "default": 0, "description": ""},
    "bad_words_ids": {"namespace": "sampling_params", "type": "array", "default": null, "description": ""},
    "renormalize_logits": {"namespace": "sampling_params", "type": "boolean", "default": false, "description": ""},
    "forced_bos_token_id": {"namespace": "sampling_params", "type": "integer", "default": null, "description": ""},
    "forced_eos_token_id": {"namespace": "sampling_params", "type": "integer", "default": null, "description": ""},
    "remove_invalid_values": {"namespace": "sampling_params", "type": "boolean", "default": false, "description": ""},
    "exponential_decay_length_penalty": {"namespace": "sampling_params", "type": "number", "default": null, "description": ""},
    "suppress_tokens": {"namespace": "sampling_params", "type": "array", "default": null, "description": ""},
    "begin_suppress_tokens": {"namespace": "sampling_params", "type": "array", "default": null, "description": ""},
    "sequence_bias": {"namespace": "sampling_params", "type": "number", "default": null, "description": ""},
    "token_healing": {"namespace": "sampling_params", "type": "boolean", "default": false, "description": ""},
    "guidance_scale": {"namespace": "sampling_params", "type": "number", "default": null, "description": ""}
  }
}
```
