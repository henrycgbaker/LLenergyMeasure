# invariants_pass2_verify extraction transcript: base_llm_args_validators_bottom

- chunk_description: tensorrt_llm.BaseLlmArgs remaining @model_validators
- expected_namespaces: ['tensorrt_llm']
- attempts: 1
- elapsed_sec: 71.38
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser doing PASS 2 (verification) of multi-pass
invariant extraction. PASS 1 already extracted invariants from
tensorrt v1.0.0 for ONE chunk of source. Your job is to
REVIEW each emitted invariant against the source and flag any that
look WRONG.

INPUT 1 - PASS 1 EXTRACTED INVARIANTS (the candidate list):

invariants:
- id: tensorrt_speculative_config_type_not_recognized
  severity: error
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.speculative_config:
        present: true
        type_is_not:
        - LookaheadDecodingConfig
        - MedusaDecodingConfig
        - EagleDecodingConfig
        - NGramDecodingConfig
        - DraftTargetDecodingConfig
        - MTPDecodingConfig
        - UserProvidedDecodingConfig
        - AutoDecodingConfig
  invariant_under_test: BaseLlmArgs.validate flags speculative config type not recognized
- id: tensorrt_speculative_config_supports_backend_false
  severity: error
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.speculative_config.supports_backend: false
  invariant_under_test: BaseLlmArgs.validate flags speculative config does not support
    backend
- id: tensorrt_lora_prefetch_dir_set_while_not_supported
  severity: error
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.peft_cache_config.lora_prefetch_dir:
        present: true
  invariant_under_test: BaseLlmArgs.validate flags lora prefetch dir set while not
    supported
- id: tensorrt_lora_dir_empty_warning
  severity: warning
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.lora_config.lora_dir:
        present: true
        not_equal: []
  invariant_under_test: BaseLlmArgs.validate flags lora dir empty warning
- id: tensorrt_enable_lora_ignored_when_lora_config_provided_warning
  severity: warning
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.enable_lora: true
      tensorrt_llm.lora_config:
        present: true
  invariant_under_test: BaseLlmArgs.validate flags enable lora ignored when lora config
    provided warning
- id: tensorrt_both_lora_dir_and_lora_target_modules_empty_warning
  severity: warning
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.lora_config.lora_dir:
        present: true
        not_equal: []
      tensorrt_llm.lora_config.lora_target_modules:
        present: true
        not_equal: []
  invariant_under_test: BaseLlmArgs.validate flags both lora dir and lora target modules
    empty warning


INPUT 2 - THE SOURCE PASS 1 READ:

=== CONTEXT ===
Continuation of BaseLlmArgs validators (second half). Same rules as the top half. Use namespace `tensorrt_llm`.

=== SOURCE: BaseLlmArgs validators (bottom half) ===

    @model_validator(mode="after")
    def validate_build_config_remaining(self):
        is_trt_llm_args = isinstance(self, TrtLlmArgs)

        # TODO: remove the checker when manage weights support all data types
        if is_trt_llm_args and self.fast_build and (self.quant_config.quant_algo
                                                    is QuantAlgo.FP8):
            self._update_plugin_config("manage_weights", True)

        if self.parallel_config._world_size == 1 and self.build_config:
            self.build_config.plugin_config.nccl_plugin = None

        if self.enable_lora and self.backend != 'pytorch':
            self.build_config.plugin_config.lora_plugin = 'auto'
            if self.lora_config is not None:
                self.build_config.lora_config.max_lora_rank = self.lora_config.max_lora_rank

        if hasattr(self,
                   'enable_prompt_adapter') and self.enable_prompt_adapter:
            self.build_config.max_prompt_embedding_table_size = self.max_prompt_adapter_token * self.build_config.max_batch_size

        if self.max_beam_width is None:
            if self.build_config:
                self.max_beam_width = self.build_config.max_beam_width
            else:
                self.max_beam_width = 1

        return self

    @model_validator(mode="after")
    def validate_speculative_config(self):
        if self.speculative_config:
            if not self.speculative_config.supports_backend(self.backend):
                raise ValueError(
                    f"Speculation type {self.speculative_config.decoding_type} does not "
                    f"support backend {self.backend}")

            # Below, we only need to set speculative_decoding_mode/decoding_config for speculation
            # on the TRT backend.
            if isinstance(self.speculative_config, LookaheadDecodingConfig):
                max_draft_len = self.speculative_config.calculate_speculative_resource(
                )[2]
                assert max_draft_len > 0
                self.build_config.speculative_decoding_mode = SpeculativeDecodingMode.LOOKAHEAD_DECODING
                self.build_config.max_draft_len = max(
                    self.build_config.max_draft_len, max_draft_len)
                self.decoding_config = DecodingConfig(
                    decoding_mode=DecodingMode.Lookahead(),
                    lookahead_decoding_config=PybindMirror.maybe_to_pybind(
                        self.speculative_config))

            elif isinstance(self.speculative_config, MedusaDecodingConfig):
                assert self.speculative_config.max_draft_len > 0
                self.build_config.speculative_decoding_mode = SpeculativeDecodingMode.MEDUSA
                self.build_config.max_draft_len = self.speculative_config.max_draft_len
                self.decoding_config = DecodingConfig(
                    decoding_mode=DecodingMode.Medusa(),
                    medusa_choices=self.speculative_config.medusa_choices)

            elif isinstance(self.speculative_config, EagleDecodingConfig):
                assert self.speculative_config.max_draft_len > 0
                assert self.speculative_config.speculative_model_dir is not None, "Path to EAGLE3 weights must be specified."
                self.build_config.max_draft_len = self.speculative_config.max_draft_len
                self.build_config.speculative_decoding_mode = SpeculativeDecodingMode.EAGLE
                if self.backend not in ['pytorch', '_autodeploy']:
                    eagle_config = _EagleConfig(
                        self.speculative_config.eagle_choices,
                        self.speculative_config.greedy_sampling,
                        self.speculative_config.posterior_threshold,
                        self.speculative_config.use_dynamic_tree,
                        self.speculative_config.dynamic_tree_max_topK)
                    self.decoding_config = DecodingConfig(
                        decoding_mode=DecodingMode.Eagle(),
                        eagle_config=eagle_config)

            elif isinstance(self.speculative_config, NGramDecodingConfig):
                assert self.backend in ['pytorch', '_autodeploy']
                assert self.speculative_config.max_draft_len > 0 and self.speculative_config.max_matching_ngram_size > 0
                self.build_config.speculative_decoding_mode = SpeculativeDecodingMode.NGRAM
                self.build_config.max_draft_len = self.speculative_config.max_draft_len

            elif isinstance(self.speculative_config, DraftTargetDecodingConfig):
                assert self.backend in ['pytorch']
                assert self.speculative_config.max_draft_len > 0
                assert self.speculative_config.speculative_model_dir is not None, "Path to draft model must be specified."
                self.build_config.speculative_decoding_mode = SpeculativeDecodingMode.DRAFT_TOKENS_EXTERNAL
                self.build_config.max_draft_len = self.speculative_config.max_draft_len

            elif isinstance(self.speculative_config, MTPDecodingConfig):
                assert self.speculative_config.num_nextn_predict_layers > 0
                self.speculative_config.max_draft_len = self.speculative_config.num_nextn_predict_layers

            elif isinstance(self.speculative_c
...<truncated>...
```

### Response

```
confirmed:
- tensorrt_speculative_config_type_not_recognized
- tensorrt_lora_prefetch_dir_set_while_not_supported
- tensorrt_enable_lora_ignored_when_lora_config_provided_warning
- tensorrt_both_lora_dir_and_lora_target_modules_empty_warning

flagged:
- id: tensorrt_speculative_config_supports_backend_false
  reason: source has `if not self.speculative_config.supports_backend(self.backend): raise ValueError` but invariant does not check for specific backend
  fix: correct_predicate:exact
- id: tensorrt_lora_dir_empty_warning
  reason: source has `logger.warning("lora_dir is empty, so custom embedding or lm head will not be applied.")` but invariant severity is warning instead of dormant
  fix: correct_severity:dormant
```
