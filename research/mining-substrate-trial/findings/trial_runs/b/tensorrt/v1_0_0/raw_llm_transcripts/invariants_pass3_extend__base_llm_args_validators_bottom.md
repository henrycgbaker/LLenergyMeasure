# invariants_pass3_extend extraction transcript: base_llm_args_validators_bottom

- chunk_description: tensorrt_llm.BaseLlmArgs remaining @model_validators
- expected_namespaces: ['tensorrt_llm']
- attempts: 1
- elapsed_sec: 86.91
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser doing PASS 3 (extension) of multi-pass
invariant extraction. PASS 1 already extracted N invariants from
tensorrt v1.0.0 for ONE chunk of source. Your job is to
identify ADDITIONAL invariants PASS 1 missed.

INPUT 1 - PASS 1 EXTRACTED INVARIANTS (already known; do NOT re-emit):

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


INPUT 2 - PASS 2 FLAGGED ISSUES (for context; consider whether they hint
at related misses):

(no flags)

INPUT 3 - THE SOURCE PASS 1 READ:

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
                self.speculative_config.max_draft_len = self.speculat
...<truncated>...
```

### Response

```
invariants:
- id: tensorrt_speculative_config_max_draft_len_not_set
  engine: tensorrt
  library: tensorrt
  invariant_under_test: BaseLlmArgs.validate sets max_draft_len for speculative config
  severity: error
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: BaseLlmArgs validators (bottom half)
    method: validate_speculative_config
    line_at_scan: 34
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.speculative_config:
        present: true
      tensorrt_llm.build_config.max_draft_len:
        {'<': 1}
  kwargs_positive:
    speculative_config: LookaheadDecodingConfig(max_draft_len=0)
    build_config: GenerationConfig(max_draft_len=0)
  kwargs_negative:
    speculative_config: LookaheadDecodingConfig(max_draft_len=10)
    build_config: GenerationConfig(max_draft_len=10)
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: 'max_draft_len must be greater than 0'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: tensorrt_speculative_config_eagle_model_dir_not_set
  engine: tensorrt
  library: tensorrt
  invariant_under_test: BaseLlmArgs.validate sets eagle model dir for speculative config
  severity: error
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: BaseLlmArgs validators (bottom half)
    method: validate_speculative_config
    line_at_scan: 53
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.speculative_config:
        present: true
        type_is_not:
          - LookaheadDecodingConfig
          - MedusaDecodingConfig
          - NGramDecodingConfig
          - DraftTargetDecodingConfig
          - MTPDecodingConfig
          - UserProvidedDecodingConfig
          - AutoDecodingConfig
      tensorrt_llm.speculative_config.speculative_model_dir:
        present: false
  kwargs_positive:
    speculative_config: EagleDecodingConfig(speculative_model_dir=None)
  kwargs_negative:
    speculative_config: EagleDecodingConfig(speculative_model_dir='/path/to/model')
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: 'Path to EAGLE3 weights must be specified.'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: tensorrt_speculative_config_draft_model_dir_not_set
  engine: tensorrt
  library: tensorrt
  invariant_under_test: BaseLlmArgs.validate sets draft model dir for speculative config
  severity: error
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: BaseLlmArgs validators (bottom half)
    method: validate_speculative_config
    line_at_scan: 73
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
          - MTPDecodingConfig
          - UserProvidedDecodingConfig
          - AutoDecodingConfig
      tensorrt_llm.speculative_config.speculative_model_dir:
        present: false
  kwargs_positive:
    speculative_config: DraftTargetDecodingConfig(speculative_model_dir=None)
  kwargs_negative:
    speculative_config: DraftTargetDecodingConfig(speculative_model_dir='/path/to/model')
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: 'Path to draft model must be specified.'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: tensorrt_lora_config_empty_warning
  engine: tensorrt
  library: tensorrt
  invariant_under_test: BaseLlmArgs.validate flags lora config empty warning
  severity: warning
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: BaseLlmArgs validators (bottom half)
    method: validate_lora_config_consistency
    line_at_scan: 10
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.lora_config:
        present: true
      tensorrt_llm.lora_config.lora_dir:
        not_equal: []
  kwargs_positive:
    lora_config: LoraConfig(lora_dir=[])
  kwargs_negative:
    lora_config: LoraConfig(lora_dir=['/path/to/lora'])
  expected_outcome:
    outcome: warning
    emission_channel: logger_warning_once
    normalised_fields: ['lora_dir']
  message_template: 'lora_dir is empty, so custom embedding or lm head will not be applied.'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: tensorrt_lora_config_both_empty_warning
  engine: tensorrt
  library: tensorrt
  invariant_under_test: BaseLlmArgs.validate flags both lora dir and target modules empty warning
  severity: warning
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: BaseLlmArgs validators (bottom half)
    method: validate_lora_config_consistency
    line_at_scan: 24
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.lora_config:
        present: true
      tensorrt_llm.lora_config.lora_dir:
        not_equal: []
      tensorrt_llm.lora_config.lora_target_modules:
        not_equal: []
  kwargs_positive:
    lora_config: LoraConfig(lora_dir=[], lora_target_modules=[])
  kwargs_negative:
    lora_config: LoraConfig(lora_dir=['/path/to/lora'], lora_target_modules=['module1'])
  expected_outcome:
    outcome: warning
    emission_channel: logger_warning_once
    normalised_fields: ['lora_dir', 'lora_target_modules']
  message_template: 'Both lora_dir and lora_target_modules are empty, so all LoRA modules will be expected. This will lead to serious memory consumption. Please provide either lora_dir or lora_target_modules if this behavior is not what you expect.'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: tensorrt_peft_cache_config_lora_prefetch_dir_set_while_not_supported
  engine: tensorrt
  library: tensorrt
  invariant_under_test: BaseLlmArgs.validate flags lora prefetch dir set while not supported
  severity: error
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: BaseLlmArgs validators (bottom half)
    method: validate_peft_cache_config
    line_at_scan: 3
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.peft_cache_config.lora_prefetch_dir:
        present: true
  kwargs_positive:
    peft_cache_config: PeftCacheConfig(lora_prefetch_dir='/path/to/prefetch')
  kwargs_negative:
    peft_cache_config: PeftCacheConfig(lora_prefetch_dir=None)
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: 'lora_prefetch_dir was set to \'{}\' while LoRA prefetch is not supported'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
```
