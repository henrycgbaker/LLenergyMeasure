# invariants_pass3_extend extraction transcript: generation_config_init_invariants

- chunk_description: GenerationConfig.__init__ raise/warn patterns
- expected_namespaces: ['transformers.sampling']
- attempts: 1
- elapsed_sec: 108.53
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser doing PASS 3 (extension) of multi-pass
invariant extraction. PASS 1 already extracted N invariants from
transformers v5.9.0 for ONE chunk of source. Your job is to
identify ADDITIONAL invariants PASS 1 missed.

INPUT 1 - PASS 1 EXTRACTED INVARIANTS (already known; do NOT re-emit):

invariants:
- id: transformers_pad_token_id_lt_zero
  severity: warning
  match:
    engine: transformers
    fields:
      transformers.sampling.pad_token_id:
        <: 0
  invariant_under_test: GenerationConfig.validate flags pad_token_id < 0


INPUT 2 - PASS 2 FLAGGED ISSUES (for context; consider whether they hint
at related misses):

(no flags)

INPUT 3 - THE SOURCE PASS 1 READ:

=== SOURCE: GenerationConfig.__init__ ===
def __init__(self, **kwargs):
        # Snapshot of the attributes the caller explicitly provided (before the `kwargs.pop(...)` calls below
        # consume them). Used by `validate()` to restrict "minor issue" warnings to flags actually set by the user,
        # as opposed to defaults inherited from a model's `generation_config.json`.
        user_set_attributes = set(kwargs.keys())

        # Parameters that control the length of the output
        self.max_length = kwargs.pop("max_length", None)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)
        self.min_length = kwargs.pop("min_length", None)
        self.min_new_tokens = kwargs.pop("min_new_tokens", None)
        self.early_stopping = kwargs.pop("early_stopping", None)
        self.max_time = kwargs.pop("max_time", None)
        self.stop_strings = kwargs.pop("stop_strings", None)

        # Parameters that control the generation strategy used
        self.do_sample = kwargs.pop("do_sample", None)
        self.num_beams = kwargs.pop("num_beams", None)

        # Parameters that control the cache
        self.use_cache = kwargs.pop("use_cache", None)
        self.cache_implementation = kwargs.pop("cache_implementation", None)
        self.cache_config = kwargs.pop("cache_config", None)

        # Parameters for manipulation of the model output logits
        self.temperature = kwargs.pop("temperature", None)
        self.top_k = kwargs.pop("top_k", None)
        self.top_p = kwargs.pop("top_p", None)
        self.min_p = kwargs.pop("min_p", None)
        self.top_h = kwargs.pop("top_h", None)
        self.typical_p = kwargs.pop("typical_p", None)
        self.epsilon_cutoff = kwargs.pop("epsilon_cutoff", None)
        self.eta_cutoff = kwargs.pop("eta_cutoff", None)
        self.repetition_penalty = kwargs.pop("repetition_penalty", None)
        self.encoder_repetition_penalty = kwargs.pop("encoder_repetition_penalty", None)
        self.length_penalty = kwargs.pop("length_penalty", None)
        self.no_repeat_ngram_size = kwargs.pop("no_repeat_ngram_size", None)
        self.bad_words_ids = kwargs.pop("bad_words_ids", None)
        self.renormalize_logits = kwargs.pop("renormalize_logits", None)
        self.forced_bos_token_id = kwargs.pop("forced_bos_token_id", None)
        self.forced_eos_token_id = kwargs.pop("forced_eos_token_id", None)
        self.remove_invalid_values = kwargs.pop("remove_invalid_values", None)
        self.exponential_decay_length_penalty = kwargs.pop("exponential_decay_length_penalty", None)
        self.suppress_tokens = kwargs.pop("suppress_tokens", None)
        self.begin_suppress_tokens = kwargs.pop("begin_suppress_tokens", None)
        self.sequence_bias = kwargs.pop("sequence_bias", None)
        self.token_healing = kwargs.pop("token_healing", None)
        self.guidance_scale = kwargs.pop("guidance_scale", None)

        self.watermarking_config = kwargs.pop("watermarking_config", None)
        if isinstance(self.watermarking_config, dict):
            self.watermarking_config = WatermarkingConfig.from_dict(self.watermarking_config)

        # Parameters that define the output variables of `generate`
        self.num_return_sequences = kwargs.pop("num_return_sequences", None)
        self.output_attentions = kwargs.pop("output_attentions", None)
        self.output_hidden_states = kwargs.pop("output_hidden_states", None)
        self.output_scores = kwargs.pop("output_scores", None)
        self.output_logits = kwargs.pop("output_logits", None)
        self.return_dict_in_generate = kwargs.pop("return_dict_in_generate", None)

        # Special tokens that can be used at generation time
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)

        # Generation parameters exclusive to encoder-decoder models
        self.encoder_no_repeat_ngram_size = kwargs.pop("encoder_no_repeat_ngram_size", None)
        self.decoder_start_token_id = kwargs.pop("decoder_start_token_id", None)

        # Assistant generation
        self.is_assistant = kwargs.pop("is_assistant", None)
        self.num_assistant_tokens = kwargs.pop("num_assistant_tokens", None)
        self.num_assistant_tokens_schedule = kwargs.pop("num_assistant_tokens_schedule", None)
        self.assistant_confidence_threshold = kwargs.pop("assistant_confidence_threshold", None)
        self.prompt_lookup_num_tokens = kwargs.pop("prompt_lookup_num_tokens", None)
        self.max_matching_ngram_size = kwargs.pop("max_matching_ngram_size", None)
        self.assistant_early_exit = kwargs.pop("assistant_early_exit", None)
        self.assistant_lookbehind = kwargs.pop("assistant_lookbehind", None)
        self.target_lookbehind = kwargs.pop("target_lookbehind", None)

        # Performance
        self.compile_config = kwargs.pop("compile_config", None)
        self.disable_compile = kwargs.pop("disable_compile", None)

        self.continuous_batching_config = kwargs.pop("continuous_batching_config", None)

        # Deprecated (moved to the Hub). TODO remove for v5
        self.low_memory = kwargs.pop("low_memory", None)
        self.penalty_alpha = kwargs.pop("penalty_alpha", None)
        self.dola_layers = kwargs.pop("dola_layers", None)
        self.diversity_penalty = kwargs.pop("diversity_penalty", None)
        self.num_beam_groups = kwargs.pop("num_beam_groups", None)
        self.constraints = kwargs.pop("constraints", None)
        self.force_words_ids = kwargs.pop("force_words_ids", None)

        self.prefill_chunk_size = kwargs.pop("prefill_chunk_size", None)

        # Common attributes
        self._commit_hash = kwargs.pop("_commit_hash", None)
        self._from_model_config = kwargs.pop("_from_model_config", None)
        self.transformers_version = kwargs.pop("transformers_version", None)

        # Additional attributes without default values
        if not self._from_model_config:
            # we don't want to copy values from the model config if we're initializing
            # a `GenerationConfig` from a model's default configuration file
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err
        else:
            # Ensure backward compatibility for models that use `forced_bos_token_id` within their config
            if kwargs.get("force_bos_token_to_be_generated", False):
                self.forced_bos_token_id = self.bos_token_id
                logger.warning_once(
                    f"Please make sure the generation config includes `forced_bos_token_id={self.bos_token_id}`. "
                )

        # Validate the values of the attributes
        self.validate(user_set_attributes=user_set_attributes)

=== COMPANION SOURCE: CompileConfig (referenced in init) ===
class CompileConfig:
    """
    Class that holds arguments relative to `torch.compile` behavior, when using automatic compilation in `generate`.
    See [`torch.compile`](https://pytorch.org/docs/stable/generated/torch.c
...<truncated>...
```

### Response

```
invariants:
- id: transformers_top_p_set_when_do_sample_false
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.validate flags top_p when do_sample is False
  severity: warning
  native_type: transformers.GenerationConfig
  miner_source:
    path: generation_config.py
    method: validate
    line_at_scan: 123
  match:
    engine: transformers
    fields:
      transformers.sampling.top_p: {present: true}
      transformers.sampling.do_sample: {'<': 1}
  kwargs_positive:
    top_p: 0.5
    do_sample: False
  kwargs_negative:
    top_p: None
    do_sample: True
  expected_outcome:
    outcome: warning
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: 'top_p is set when do_sample=False'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: transformers_top_k_set_when_do_sample_false
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.validate flags top_k when do_sample is False
  severity: warning
  native_type: transformers.GenerationConfig
  miner_source:
    path: generation_config.py
    method: validate
    line_at_scan: 126
  match:
    engine: transformers
    fields:
      transformers.sampling.top_k: {present: true}
      transformers.sampling.do_sample: {'<': 1}
  kwargs_positive:
    top_k: 5
    do_sample: False
  kwargs_negative:
    top_k: None
    do_sample: True
  expected_outcome:
    outcome: warning
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: 'top_k is set when do_sample=False'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: transformers_min_p_set_when_do_sample_false
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.validate flags min_p when do_sample is False
  severity: warning
  native_type: transformers.GenerationConfig
  miner_source:
    path: generation_config.py
    method: validate
    line_at_scan: 129
  match:
    engine: transformers
    fields:
      transformers.sampling.min_p: {present: true}
      transformers.sampling.do_sample: {'<': 1}
  kwargs_positive:
    min_p: 0.2
    do_sample: False
  kwargs_negative:
    min_p: None
    do_sample: True
  expected_outcome:
    outcome: warning
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: 'min_p is set when do_sample=False'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: transformers_typical_p_set_when_do_sample_false
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.validate flags typical_p when do_sample is False
  severity: warning
  native_type: transformers.GenerationConfig
  miner_source:
    path: generation_config.py
    method: validate
    line_at_scan: 132
  match:
    engine: transformers
    fields:
      transformers.sampling.typical_p: {present: true}
      transformers.sampling.do_sample: {'<': 1}
  kwargs_positive:
    typical_p: 0.8
    do_sample: False
  kwargs_negative:
    typical_p: None
    do_sample: True
  expected_outcome:
    outcome: warning
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: 'typical_p is set when do_sample=False'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: transformers_epsilon_cutoff_set_when_do_sample_false
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.validate flags epsilon_cutoff when do_sample is False
  severity: warning
  native_type: transformers.GenerationConfig
  miner_source:
    path: generation_config.py
    method: validate
    line_at_scan: 135
  match:
    engine: transformers
    fields:
      transformers.sampling.epsilon_cutoff: {present: true}
      transformers.sampling.do_sample: {'<': 1}
  kwargs_positive:
    epsilon_cutoff: 0.01
    do_sample: False
  kwargs_negative:
    epsilon_cutoff: None
    do_sample: True
  expected_outcome:
    outcome: warning
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: 'epsilon_cutoff is set when do_sample=False'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: transformers_eta_cutoff_set_when_do_sample_false
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.validate flags eta_cutoff when do_sample is False
  severity: warning
  native_type: transformers.GenerationConfig
  miner_source:
    path: generation_config.py
    method: validate
    line_at_scan: 138
  match:
    engine: transformers
    fields:
      transformers.sampling.eta_cutoff: {present: true}
      transformers.sampling.do_sample: {'<': 1}
  kwargs_positive:
    eta_cutoff: 0.05
    do_sample: False
  kwargs_negative:
    eta_cutoff: None
    do_sample: True
  expected_outcome:
    outcome: warning
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: 'eta_cutoff is set when do_sample=False'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
```
