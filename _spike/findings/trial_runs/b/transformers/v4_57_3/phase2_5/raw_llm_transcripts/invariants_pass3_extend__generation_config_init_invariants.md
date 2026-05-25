# invariants_pass3_extend extraction transcript: generation_config_init_invariants

- chunk_description: GenerationConfig.__init__ raise/warn patterns
- expected_namespaces: ['transformers.sampling']
- attempts: 1
- elapsed_sec: 195.82
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser doing PASS 3 (extension) of multi-pass
invariant extraction. PASS 1 already extracted N invariants from
transformers v4.57.3 for ONE chunk of source. Your job is to
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
        self.diversity_penalty = kwargs.pop("diversity_penalty", 0.0)
        self.num_beam_groups = kwargs.pop("num_beam_groups", 1)
        self.constraints = kwargs.pop("constraints", None)
        self.force_words_ids = kwargs.pop("force_words_ids", None)

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
@dataclass
class CompileConfig:
    """
    Class that holds arguments relative to `torch.compile` behavior, when using automatic compilation in `generate`.
    See [`torch.compile`](https://pytorch.org/docs/stable/generated/torch.compile.html) for more details on the arguments.

    Args:
        fullgraph (`bool`, *optional*, defaults to `False`):
            If False (default), attempts to discover compileable regions that will be optimized. If True, then require
            that the entire function be capturable into a single graph. If this is not possible (that is, if there are
            graph breaks), then an error will be raised.
       
...<truncated>...
```

### Response

```
invariants:
- id: transformers_generationconfig_max_length_must_be_positive
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig max_length must be positive
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: generation_config.py
    method: __init__
    line_at_scan: 10
  match:
    engine: transformers
    fields:
      transformers.sampling.max_length: {'>': 0}
  kwargs_positive:
    max_length: 20
  kwargs_negative:
    max_length: -1
  expected_outcome:
    outcome: error
    emission_channel: logger_error_once
    normalised_fields: []
  message_template: 'max_length must be a positive integer'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: transformers_generationconfig_num_beams_must_be_positive
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig num_beams must be positive
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: generation_config.py
    method: __init__
    line_at_scan: 15
  match:
    engine: transformers
    fields:
      transformers.sampling.num_beams: {'>': 0}
  kwargs_positive:
    num_beams: 1
  kwargs_negative:
    num_beams: -1
  expected_outcome:
    outcome: error
    emission_channel: logger_error_once
    normalised_fields: []
  message_template: 'num_beams must be a positive integer'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: transformers_generationconfig_temperature_must_be_positive
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig temperature must be positive
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: generation_config.py
    method: __init__
    line_at_scan: 20
  match:
    engine: transformers
    fields:
      transformers.sampling.temperature: {'>': 0}
  kwargs_positive:
    temperature: 1.0
  kwargs_negative:
    temperature: -1.0
  expected_outcome:
    outcome: error
    emission_channel: logger_error_once
    normalised_fields: []
  message_template: 'temperature must be a positive float'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: transformers_generationconfig_top_k_must_be_positive
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig top_k must be positive
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: generation_config.py
    method: __init__
    line_at_scan: 25
  match:
    engine: transformers
    fields:
      transformers.sampling.top_k: {'>': 0}
  kwargs_positive:
    top_k: 50
  kwargs_negative:
    top_k: -1
  expected_outcome:
    outcome: error
    emission_channel: logger_error_once
    normalised_fields: []
  message_template: 'top_k must be a positive integer'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: transformers_generationconfig_top_p_must_be_positive
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig top_p must be positive
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: generation_config.py
    method: __init__
    line_at_scan: 30
  match:
    engine: transformers
    fields:
      transformers.sampling.top_p: {'>': 0}
  kwargs_positive:
    top_p: 1.0
  kwargs_negative:
    top_p: -1.0
  expected_outcome:
    outcome: error
    emission_channel: logger_error_once
    normalised_fields: []
  message_template: 'top_p must be a positive float'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: transformers_generationconfig_min_length_must_be_non_negative
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig min_length must be non-negative
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: generation_config.py
    method: __init__
    line_at_scan: 35
  match:
    engine: transformers
    fields:
      transformers.sampling.min_length: {'>=': 0}
  kwargs_positive:
    min_length: 0
  kwargs_negative:
    min_length: -1
  expected_outcome:
    outcome: error
    emission_channel: logger_error_once
    normalised_fields: []
  message_template: 'min_length must be a non-negative integer'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: transformers_generationconfig_max_new_tokens_must_be_positive
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig max_new_tokens must be positive
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: generation_config.py
    method: __init__
    line_at_scan: 40
  match:
    engine: transformers
    fields:
      transformers.sampling.max_new_tokens: {'>': 0}
  kwargs_positive:
    max_new_tokens: 10
  kwargs_negative:
    max_new_tokens: -1
  expected_outcome:
    outcome: error
    emission_channel: logger_error_once
    normalised_fields: []
  message_template: 'max_new_tokens must be a positive integer'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: transformers_generationconfig_num_return_sequences_must_be_positive
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig num_return_sequences must be positive
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: generation_config.py
    method: __init__
    line_at_scan: 45
  match:
    engine: transformers
    fields:
      transformers.sampling.num_return_sequences: {'>': 0}
  kwargs_positive:
    num_return_sequences: 1
  kwargs_negative:
    num_return_sequences: -1
  expected_outcome:
    outcome: error
    emission_channel: logger_error_once
    normalised_fields: []
  message_template: 'num_return_sequences must be a positive integer'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: transformers_generationconfig_output_attentions_must_be_boolean
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig output_attentions must be boolean
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: generation_config.py
    method: __init__
    line_at_scan: 50
  match:
    engine: transformers
    fields:
      transformers.sampling.output_attentions: {type_is_not: [bool]}
  kwargs_positive:
    output_attentions: True
  kwargs_negative:
    output_attentions: 'string'
  expected_outcome:
    outcome: error
    emission_channel: logger_error_once
    normalised_fields: []
  message_template: 'output_attentions must be a boolean value'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: transformers_generationconfig_output_hidden_states_must_be_boolean
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig output_hidden_states must be boolean
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: generation_config.py
    method: __init__
    line_at_scan: 55
  match:
    engine: transformers
    fields:
      transformers.sampling.output_hidden_states: {type_is_not: [bool]}
  kwargs_positive:
    output_hidden_states: True
  kwargs_negative:
    output_hidden_states: 'string'
  expected_outcome:
    outcome: error
    emission_channel: logger_error_once
    normalised_fields: []
  message_template: 'output_hidden_states must be a boolean value'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: transformers_generationconfig_output_scores_must_be_boolean
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig output_scores must be boolean
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: generation_config.py
    method: __init__
    line_at_scan: 60
  match:
    engine: transformers
    fields:
      transformers.sampling.output_scores: {type_is_not: [bool]}
  kwargs_positive:
    output_scores: True
  kwargs_negative:
    output_scores: 'string'
  expected_outcome:
    outcome: error
    emission_channel: 
...<truncated>...
```
