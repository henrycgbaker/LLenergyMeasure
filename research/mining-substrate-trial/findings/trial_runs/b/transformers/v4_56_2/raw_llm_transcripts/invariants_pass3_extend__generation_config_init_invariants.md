# invariants_pass3_extend extraction transcript: generation_config_init_invariants

- chunk_description: GenerationConfig.__init__ raise/warn patterns
- expected_namespaces: ['transformers.sampling']
- attempts: 1
- elapsed_sec: 49.99
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser doing PASS 3 (extension) of multi-pass
invariant extraction. PASS 1 already extracted N invariants from
transformers v4.56.2 for ONE chunk of source. Your job is to
identify ADDITIONAL invariants PASS 1 missed.

INPUT 1 - PASS 1 EXTRACTED INVARIANTS (already known; do NOT re-emit):

invariants:
- id: transformers_watermarking_config_type_is_not
  severity: error
  match:
    engine: transformers
    fields:
      transformers.sampling.watermarking_config:
        present: true
        type_is_not:
        - BaseWatermarkingConfig
  invariant_under_test: GenerationConfig.__init__ flags watermarking_config type mismatch


INPUT 2 - PASS 2 FLAGGED ISSUES (for context; consider whether they hint
at related misses):

(no flags)

INPUT 3 - THE SOURCE PASS 1 READ:

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
        self.num_beam_groups = kwargs.pop("num_beam_groups", 1)

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
        self.diversity_penalty = kwargs.pop("diversity_penalty", 0.0)
        self.repetition_penalty = kwargs.pop("repetition_penalty", 1.0)
        self.encoder_repetition_penalty = kwargs.pop("encoder_repetition_penalty", 1.0)
        self.length_penalty = kwargs.pop("length_penalty", 1.0)
        self.no_repeat_ngram_size = kwargs.pop("no_repeat_ngram_size", 0)
        self.bad_words_ids = kwargs.pop("bad_words_ids", None)
        self.force_words_ids = kwargs.pop("force_words_ids", None)
        self.renormalize_logits = kwargs.pop("renormalize_logits", False)
        self.constraints = kwargs.pop("constraints", None)
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
        self.encoder_no_repeat_ngram_size = kwargs.pop("encoder_no_repeat_ngram_size", 0)
        self.decoder_start_token_id = kwargs.pop("decoder_start_token_id", None)

        # Assistant generation
        self.is_assistant = False
        self.num_assistant_tokens = kwargs.pop("num_assistant_tokens", 20)
        self.num_assistant_tokens_schedule = kwargs.pop("num_assistant_tokens_schedule", "constant")
        self.assistant_confidence_threshold = kwargs.pop("assistant_confidence_threshold", 0.4)
        self.prompt_lookup_num_tokens = kwargs.pop("prompt_lookup_num_tokens", None)
        self.max_matching_ngram_size = kwargs.pop("max_matching_ngram_size", None)
        self.assistant_early_exit = kwargs.pop("assistant_early_exit", None)
        ## assistant generation for different tokenizers, the windows size for assistant/target model
        self.assistant_lookbehind = kwargs.pop("assistant_lookbehind", 10)
        self.target_lookbehind = kwargs.pop("target_lookbehind", 10)

        # Performance
        self.compile_config = kwargs.pop("compile_config", None)
        self.disable_compile = kwargs.pop("disable_compile", False)

        # Deprecated (moved to the Hub). TODO joao, manuel: remove in v4.62.0
        self.low_memory = kwargs.pop("low_memory", None)
        self.penalty_alpha = kwargs.pop("penalty_alpha", None)
        self.dola_layers = kwargs.pop("dola_layers", None)

        # The remaining attributes do not parametrize `.generate()`, but are informative and/or used by the hub
        # interface.
        self._from_model_config = kwargs.pop("_from_model_config", False)
        self._commit_hash = kwargs.pop("_commit_hash", None)
        self.transformers_version = kwargs.pop("transformers_version", __version__)

        # Additional attributes without default values
        if not self._from_model_config:
            # we don't want to copy values from the model config if we're initializing a `GenerationConfig` from a
            # model's default configuration file
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err

        # Validate the values of the attributes
        self.validate()

=== COMPANION SOURCE: CompileConfig (referenced in init) ===
class CompileConfig:
    """
    Class that holds arguments relative to `torch.compile` behavior, when using automatic compilation in `generate`.
    See [`torch.compile`](https://pytorch.org/docs/stable/generated/torch.compile.html) for more details on the arguments.

    Args:
        fullgraph (`bool`, *optional*, defaults to `False`):
            If False (default), attempts to discover compileable regions that will be optimized. If True, then require
            that the entire function be capturable into a single graph. If this is not possible (that is, if there are
            graph breaks), then an error will be raised.
        dynamic (`bool`
...<truncated>...
```

### Response

```
invariants:
- id: transformers_generationconfig_compile_config_type_is_not
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.__init__ flags compile_config type mismatch
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: generation_config.py
    method: __init__
    line_at_scan: 123
  match:
    engine: transformers
    fields:
      transformers.sampling.compile_config:
        present: true
        type_is_not:
        - CompileConfig
  kwargs_positive:
    compile_config: "invalid_type"
  kwargs_negative:
    compile_config: CompileConfig(fullgraph=True)
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: 'Invalid type for compile_config'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: transformers_generationconfig_dynamic_must_be_bool_or_none
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.__init__ flags dynamic must be bool or None
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: generation_config.py
    method: __init__
    line_at_scan: 145
  match:
    engine: transformers
    fields:
      transformers.sampling.compile_config.dynamic:
        present: true
        not_in: [bool, None]
  kwargs_positive:
    compile_config: CompileConfig(dynamic="invalid_type")
  kwargs_negative:
    compile_config: CompileConfig(dynamic=True)
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: 'Invalid type for dynamic'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: transformers_generationconfig_backend_must_be_str_or_callable
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.__init__ flags backend must be str or callable
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: generation_config.py
    method: __init__
    line_at_scan: 155
  match:
    engine: transformers
    fields:
      transformers.sampling.compile_config.backend:
        present: true
        not_in: [str, Callable]
  kwargs_positive:
    compile_config: CompileConfig(backend=123)
  kwargs_negative:
    compile_config: CompileConfig(backend="inductor")
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: 'Invalid type for backend'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: transformers_generationconfig_mode_must_be_str
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.__init__ flags mode must be str
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: generation_config.py
    method: __init__
    line_at_scan: 165
  match:
    engine: transformers
    fields:
      transformers.sampling.compile_config.mode:
        present: true
        not_in: [str]
  kwargs_positive:
    compile_config: CompileConfig(mode=123)
  kwargs_negative:
    compile_config: CompileConfig(mode="reduce-overhead")
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: 'Invalid type for mode'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
```
