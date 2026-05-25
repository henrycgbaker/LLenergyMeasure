# schema extraction transcript: h6_whole_file_schema

- chunk_description: H6 single-shot whole-source schema
- expected_namespaces: ['engine_params', 'sampling_params']
- attempts: 1
- elapsed_sec: 257.38
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
  "chunk_name": "h6_whole_file",
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

EXPECTED NAMESPACES FOR THIS CHUNK: engine_params, sampling_params, $defs.CompileConfig, $defs.WatermarkingConfig
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

=== SOURCE: PreTrainedModel.from_pretrained (signature + leading docstring) -> engine_params ===
def from_pretrained(
        cls: type[SpecificPreTrainedModelType],
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: Optional[bool] = None,
        weights_only: bool = True,
        **kwargs,
    ) -> SpecificPreTrainedModelType:
r"""
        Instantiate a pretrained pytorch model from a pre-trained model configuration.

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
        the model, you should first set it back in training mode with `model.train()`.

        The warning *Weights from XXX not initialized from pretrained model* means that the weights of XXX do not come
        pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
        task.

        The warning *Weights from XXX not used in YYY* means that the layer XXX is not used by YYY, therefore those
        weights are discarded.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
                      this case, `from_tf` should be set to `True` and a configuration object should be provided as
                      `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
                      PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
                    - A path or url to a model folder containing a *flax checkpoint file* in *.msgpack* format (e.g,
                      `./flax_model/` containing `flax_model.msgpack`). In this case, `from_flax` should be set to
                      `True`.
                    - `None` if you are both providing the configuration and state dictionary (resp. with keyword
                      arguments `config` and `state_dict`).
            model_args (sequence of positional arguments, *optional*):
                All remaining positional arguments will be passed to the underlying model's `__init__` method.
            config (`Union[PretrainedConfig, str, os.PathLike]`, *optional*):
                Can be either:

                    - an instance of a class derived from [`PretrainedConfig`],
                    - a string or path valid as input to [`~PretrainedConfig.from_pretrained`].

                Configuration for the model to use instead of an automatically loaded configuration. Configuration can
                be automatically loaded when:

                    - The model is a model provided by the library (loaded with the *model id* string of a pretrained
                      model).
                    - The model was saved using [`~PreTrainedModel.save_pretrained`] and is reloaded by supplying the
                      save directory.
                    - The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
                      configuration JSON
...<truncated>...
```

### Response

```
{
  "engine": "transformers",
  "engine_version": "4.57.3",
  "chunk_name": "h6_whole_file",
  "chunk_fields": {
    "pretrained_model_name_or_path": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "object", "description": "PathLike"}, {"type": "null"}], "default": null},
    "config": {"namespace": "engine_params", "anyOf": [{"type": "object", "description": "PretrainedConfig"}, {"type": "string"}, {"type": "object", "description": "PathLike"}, {"type": "null"}], "default": null},
    "cache_dir": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "object", "description": "PathLike"}, {"type": "null"}], "default": null},
    "ignore_mismatched_sizes": {"namespace": "engine_params", "type": "boolean", "default": false},
    "force_download": {"namespace": "engine_params", "type": "boolean", "default": false},
    "local_files_only": {"namespace": "engine_params", "type": "boolean", "default": false},
    "token": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "boolean"}, {"type": "null"}], "default": null},
    "revision": {"namespace": "engine_params", "type": "string", "default": "main"},
    "use_safetensors": {"namespace": "engine_params", "anyOf": [{"type": "boolean"}, {"type": "null"}], "default": null},
    "weights_only": {"namespace": "engine_params", "type": "boolean", "default": true},
    "dtype": {"namespace": "engine_params", "description": "torch.dtype or str"},
    "device_map": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "object", "description": "dict[str, Union[int, str, torch.device]]"}, {"type": "integer"}, {"type": "object", "description": "torch.device"}], "default": null},
    "max_memory": {"namespace": "engine_params", "type": "object"},
    "tp_plan": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "null"}], "default": null},
    "tp_size": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "null"}], "default": null},
    "device_mesh": {"namespace": "engine_params", "type": "object"},
    "offload_folder": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "object", "description": "PathLike"}, {"type": "null"}], "default": null},
    "offload_buffers": {"namespace": "engine_params", "type": "boolean", "default": false},
    "quantization_config": {"namespace": "engine_params", "anyOf": [{"type": "object", "description": "QuantizationConfigMixin"}, {"type": "object", "description": "Dict"}, {"type": "null"}], "default": null},
    "subfolder": {"namespace": "engine_params", "type": "string", "default": ""},
    "variant": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "null"}], "default": null},
    "load_in_8bit": {"namespace": "engine_params", "type": "boolean", "default": false},
    "load_in_4bit": {"namespace": "engine_params", "type": "boolean", "default": false},
    "llm_int8_threshold": {"namespace": "engine_params", "type": "number", "default": 6.0},
    "llm_int8_skip_modules": {"namespace": "engine_params", "anyOf": [{"type": "array"}, {"type": "null"}], "default": null},
    "llm_int8_enable_fp32_cpu_offload": {"namespace": "engine_params", "type": "boolean", "default": false},
    "llm_int8_has_fp16_weight": {"namespace": "engine_params", "type": "boolean", "default": false},
    "bnb_4bit_compute_dtype": {"namespace": "engine_params", "description": "torch.dtype or str"},
    "bnb_4bit_quant_type": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "null"}], "default": "fp4"},
    "bnb_4bit_use_double_quant": {"namespace": "engine_params", "type": "boolean", "default": false},
    "bnb_4bit_quant_storage": {"namespace": "engine_params", "description": "torch.dtype or str"},
    "fullgraph": {"namespace": "$defs.CompileConfig", "type": "boolean", "default": false},
    "dynamic": {"namespace": "$defs.CompileConfig", "anyOf": [{"type": "boolean"}, {"type": "null"}], "default": null},
    "backend": {"namespace": "$defs.CompileConfig", "anyOf": [{"type": "string"}, {"type": "object", "description": "Callable"}, {"type": "null"}], "default": "inductor"},
    "mode": {"namespace": "$defs.CompileConfig", "type": "string", "default": "reduce-overhead"},
    "options": {"namespace": "$defs.CompileConfig", "anyOf": [{"type": "object", "description": "dict"}, {"type": "null"}], "default": null},
    "greenlist_ratio": {"namespace": "$defs.WatermarkingConfig", "type": "number", "default": 0.25},
    "bias": {"namespace": "$defs.WatermarkingConfig", "type": "number", "default": 2.0},
    "hashing_key": {"namespace": "$defs.WatermarkingConfig", "type": "integer", "default": 15485863},
    "seeding_scheme": {"namespace": "$defs.WatermarkingConfig", "anyOf": [{"type": "string"}, {"type": "null"}], "enum": ["lefthash", "selfhash"], "default": "lefthash"},
    "context_width": {"namespace": "$defs.WatermarkingConfig", "type": "integer", "default": 1},
    "max_length": {"namespace": "sampling_params", "type": "integer", "default": 20},
    "max_new_tokens": {"namespace": "sampling_params", "anyOf": [{"type": "integer"}, {"type": "null"}], "default": null},
    "min_length": {"namespace": "sampling_params", "type": "integer", "default": 0},
    "min_new_tokens": {"namespace": "sampling_params", "anyOf": [{"type": "integer"}, {"type": "null"}], "default": null},
    "early_stopping": {"namespace": "sampling_params", "anyOf": [{"type": "boolean"}, {"type": "string"}], "default": false},
    "max_time": {"namespace": "sampling_params", "anyOf": [{"type": "number"}, {"type": "null"}], "default": null},
    "stop_strings": {"namespace": "sampling_params", "anyOf": [{"type": "string"}, {"type": "array"}, {"type": "null"}], "default": null},
    "do_sample": {"namespace": "sampling_params", "type": "boolean", "default": false},
    "num_beams": {"namespace": "sampling_params", "type": "integer", "default": 1},
    "use_cache": {"namespace": "sampling_params", "type": "boolean", "default": true},
    "cache_implementation": {"namespace": "sampling_params", "anyOf": [{"type": "string"}, {"type": "null"}], "enum": ["dynamic", "static", "offloaded", "offloaded_static", "quantized"], "default": null},
    "cache_config": {"namespace": "sampling_params", "anyOf": [{"type": "object", "description": "dict"}, {"type": "null"}], "default": null},
    "return_legacy_cache": {"namespace": "sampling_params", "anyOf": [{"type": "boolean"}, {"type": "null"}], "default": true},
    "prefill_chunk_size": {"namespace": "sampling_params", "anyOf": [{"type": "integer"}, {"type": "null"}], "default": null},
    "temperature": {"namespace": "sampling_params", "type": "number", "default": 1.0},
    "top_k": {"namespace": "sampling_params", "type": "integer", "default": 50},
    "top_p": {"namespace": "sampling_params", "type": "number", "default": 1.0},
    "min_p": {"namespace": "sampling_params", "anyOf": [{"type": "number"}, {"type": "null"}], "default": null},
    "typical_p": {"namespace": "sampling_params", "type": "number", "default": 1.0},
    "epsilon_cutoff": {"namespace": "sampling_params", "type": "number", "default": 0.0},
    "eta_cutoff": {"namespace": "sampling_params", "type": "number", "default": 0.0},
    "repetition_penalty": {"namespace": "sampling_params", "type": "number", "default": 1.0},
    "encoder_repetition_penalty": {"namespace": "sampling_params", "type": "number", "default": 1.0},
    "length_penalty": {"namespace": "sampling_params", "type": "number", "default": 1.0},
    "no_repeat_ngram_size": {"namespace": "sampling_params", "type": "integer", "default": 0},
    "bad_words_ids": {"namespace": "sampling_params", "anyOf": [{"type": "array"}, {"type": "null"}], "default": null},
    "renormalize_logits": {"namespace": "sampling_params", "type": "boolean", "default": false},
    "forced_bos_token_id": {"namespace": "sampling_params", "anyOf": [{"type": "integer"}, {"type": "null"}], "default": null},
    "forced_eos_token_id": {"namesp
...<truncated>...
```
