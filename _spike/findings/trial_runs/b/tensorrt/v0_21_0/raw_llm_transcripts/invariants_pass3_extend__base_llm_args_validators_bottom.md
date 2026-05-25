# invariants_pass3_extend extraction transcript: base_llm_args_validators_bottom

- chunk_description: tensorrt_llm.BaseLlmArgs remaining @model_validators
- expected_namespaces: ['tensorrt_llm']
- attempts: 1
- elapsed_sec: 85.71
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser doing PASS 3 (extension) of multi-pass
invariant extraction. PASS 1 already extracted N invariants from
tensorrt v0.21.0 for ONE chunk of source. Your job is to
identify ADDITIONAL invariants PASS 1 missed.

INPUT 1 - PASS 1 EXTRACTED INVARIANTS (already known; do NOT re-emit):

invariants:
- id: tensorrt_llm_speculative_config_type_not_recognized
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
  invariant_under_test: BaseLlmArgs.validate flags speculative config type not recognized
- id: tensorrt_llm_max_lora_rank_ignored_when_lora_config_provided
  severity: warning
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.lora_config:
        present: true
      tensorrt_llm.max_lora_rank:
        present: true
        not_equal: None
  invariant_under_test: BaseLlmArgs.validate flags max_lora_rank ignored when lora_config
    is provided
- id: tensorrt_llm_max_loras_ignored_when_lora_config_provided
  severity: warning
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.lora_config:
        present: true
      tensorrt_llm.max_loras:
        present: true
        not_equal: None
  invariant_under_test: BaseLlmArgs.validate flags max_loras ignored when lora_config
    is provided
- id: tensorrt_llm_max_cpu_loras_ignored_when_lora_config_provided
  severity: warning
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.lora_config:
        present: true
      tensorrt_llm.max_cpu_loras:
        present: true
        not_equal: None
  invariant_under_test: BaseLlmArgs.validate flags max_cpu_loras ignored when lora_config
    is provided
- id: tensorrt_llm_enable_lora_ignored_when_lora_config_provided_for_pytorch_backend
  severity: warning
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.lora_config:
        present: true
      tensorrt_llm.enable_lora:
        present: true
        not_equal: None
      tensorrt_llm.backend: pytorch
  invariant_under_test: BaseLlmArgs.validate flags enable_lora ignored when lora_config
    is provided for pytorch backend
- id: tensorrt_llm_both_lora_dir_and_lora_target_modules_empty
  severity: warning
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.lora_config:
        present: true
      tensorrt_llm.lora_dir: []
      tensorrt_llm.lora_target_modules: []
  invariant_under_test: BaseLlmArgs.validate flags both lora_dir and lora_target_modules
    empty


INPUT 2 - PASS 2 FLAGGED ISSUES (for context; consider whether they hint
at related misses):

flagged:
- id: tensorrt_llm_enable_lora_ignored_when_lora_config_provided_for_pytorch_backend
  reason: Source has a more specific condition `self.enable_lora and self.lora_config
    is not None and self.backend in ['pytorch', '_autodeploy']` which is not fully
    captured by the invariant.
  fix: correct_predicate:exact
- id: tensorrt_llm_both_lora_dir_and_lora_target_modules_empty
  reason: Source has a more specific condition `len(self.lora_config.lora_dir) ==
    0 and len(self.lora_config.lora_target_modules) == 0` which is not fully captured
    by the invariant.
  fix: correct_predicate:exact


INPUT 3 - THE SOURCE PASS 1 READ:

=== CONTEXT ===
Continuation of BaseLlmArgs validators (second half). Same rules as the top half. Use namespace `tensorrt_llm`.

=== SOURCE: BaseLlmArgs validators (bottom half) ===

    @model_validator(mode="after")
    def validate_speculative_config(self):
        if self.speculative_config:
            if isinstance(self.speculative_config, LookaheadDecodingConfig):
                lookahead_config = self.speculative_config
                # Update the build config
                _, _, max_draft_tokens, _ = lookahead_config.calculate_speculative_resource(
                )
                self.build_config.speculative_decoding_mode = SpeculativeDecodingMode.LOOKAHEAD_DECODING
                if max_draft_tokens > self.build_config.max_draft_len:
                    self.build_config.max_draft_len = max_draft_tokens

                self.decoding_config = DecodingConfig(
                    decoding_mode=DecodingMode.Lookahead(),
                    lookahead_decoding_config=PybindMirror.maybe_to_pybind(
                        lookahead_config))
            elif isinstance(self.speculative_config, MedusaDecodingConfig):
                self.build_config.speculative_decoding_mode = SpeculativeDecodingMode.MEDUSA

                assert self.speculative_config.max_draft_len > 0
                self.build_config.max_draft_len = self.speculative_config.max_draft_len
                self.decoding_config = DecodingConfig(
                    decoding_mode=DecodingMode.Medusa(),
                    medusa_choices=self.speculative_config.medusa_choices)
            elif isinstance(self.speculative_config, EagleDecodingConfig):
                self.build_config.speculative_decoding_mode = SpeculativeDecodingMode.EAGLE
                assert self.speculative_config.max_draft_len > 0

                self.build_config.max_draft_len = self.speculative_config.max_draft_len

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
                else:
                    from tensorrt_llm._torch.speculative import Eagle3Config
                    self.speculative_config = Eagle3Config(
                        max_draft_tokens=self.speculative_config.max_draft_len,
                        draft_model_path=self.speculative_config.
                        pytorch_weights_path,
                        eagle3_one_model=self.speculative_config.
                        eagle3_one_model)
            elif isinstance(self.speculative_config, NGramDecodingConfig):
                self.build_config.speculative_decoding_mode = SpeculativeDecodingMode.NGRAM
                assert self.backend in ['pytorch', '_autodeploy']
                assert self.speculative_config.prompt_lookup_num_tokens > 0 and self.speculative_config.max_matching_ngram_size > 0
                self.build_config.max_draft_len = self.speculative_config.max_draft_len
                from tensorrt_llm._torch.speculative import NGramConfig
                self.speculative_config = NGramConfig(
                    prompt_lookup_num_tokens=self.speculative_config.
                    prompt_lookup_num_tokens,
                    max_matching_ngram_size=self.speculative_config.
                    max_matching_ngram_size,
                    is_keep_all=self.speculative_config.is_keep_all,
                    is_use_oldest=self.speculative_config.is_use_oldest,
                    is_public_pool=self.speculative_config.is_public_pool,
                )
            elif isinstance(self.speculative_config, DraftTargetDecodingConfig):
                self.build_config.speculative_decoding_mode = SpeculativeDecodingMode.DRAFT_TOKENS_EXTERNAL
                assert self.backend == 'pytorch'
                assert self.speculative_config.max_draft_len > 0
                self.build_config.max_draft_len = self.speculative_config.max_draft_len
                from tensorrt_llm._torch.speculative import DraftTargetConfig
                self.speculative_config = DraftTargetConfig(
                    ma
...<truncated>...
```

### Response

```
invariants:
- id: tensorrt_llm_speculative_config_lookahead_decoding_max_draft_len_update
  engine: tensorrt
  library: tensorrt
  invariant_under_test: BaseLlmArgs.validate updates max_draft_len when speculative config is LookaheadDecodingConfig
  severity: dormant
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: BaseLlmArgs validators (bottom half)
    method: validate_speculative_config
    line_at_scan: 10
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.speculative_config:
        present: true
        type_is: LookaheadDecodingConfig
  kwargs_positive:
    speculative_config: LookaheadDecodingConfig(max_draft_len=100)
  kwargs_negative:
    speculative_config: MedusaDecodingConfig()
  expected_outcome:
    outcome: dormant_announced
    emission_channel: none
    normalised_fields: []
  message_template: ''
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: tensorrt_llm_speculative_config_medusa_decoding_max_draft_len_update
  engine: tensorrt
  library: tensorrt
  invariant_under_test: BaseLlmArgs.validate updates max_draft_len when speculative config is MedusaDecodingConfig
  severity: dormant
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: BaseLlmArgs validators (bottom half)
    method: validate_speculative_config
    line_at_scan: 20
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.speculative_config:
        present: true
        type_is: MedusaDecodingConfig
  kwargs_positive:
    speculative_config: MedusaDecodingConfig(max_draft_len=100)
  kwargs_negative:
    speculative_config: LookaheadDecodingConfig()
  expected_outcome:
    outcome: dormant_announced
    emission_channel: none
    normalised_fields: []
  message_template: ''
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: tensorrt_llm_speculative_config_eagle_decoding_max_draft_len_update
  engine: tensorrt
  library: tensorrt
  invariant_under_test: BaseLlmArgs.validate updates max_draft_len when speculative config is EagleDecodingConfig
  severity: dormant
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: BaseLlmArgs validators (bottom half)
    method: validate_speculative_config
    line_at_scan: 30
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.speculative_config:
        present: true
        type_is: EagleDecodingConfig
  kwargs_positive:
    speculative_config: EagleDecodingConfig(max_draft_len=100)
  kwargs_negative:
    speculative_config: MedusaDecodingConfig()
  expected_outcome:
    outcome: dormant_announced
    emission_channel: none
    normalised_fields: []
  message_template: ''
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: tensorrt_llm_speculative_config_ngram_decoding_max_draft_len_update
  engine: tensorrt
  library: tensorrt
  invariant_under_test: BaseLlmArgs.validate updates max_draft_len when speculative config is NGramDecodingConfig
  severity: dormant
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: BaseLlmArgs validators (bottom half)
    method: validate_speculative_config
    line_at_scan: 40
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.speculative_config:
        present: true
        type_is: NGramDecodingConfig
  kwargs_positive:
    speculative_config: NGramDecodingConfig(max_draft_len=100)
  kwargs_negative:
    speculative_config: EagleDecodingConfig()
  expected_outcome:
    outcome: dormant_announced
    emission_channel: none
    normalised_fields: []
  message_template: ''
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: tensorrt_llm_speculative_config_draft_target_decoding_max_draft_len_update
  engine: tensorrt
  library: tensorrt
  invariant_under_test: BaseLlmArgs.validate updates max_draft_len when speculative config is DraftTargetDecodingConfig
  severity: dormant
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: BaseLlmArgs validators (bottom half)
    method: validate_speculative_config
    line_at_scan: 50
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.speculative_config:
        present: true
        type_is: DraftTargetDecodingConfig
  kwargs_positive:
    speculative_config: DraftTargetDecodingConfig(max_draft_len=100)
  kwargs_negative:
    speculative_config: NGramDecodingConfig()
  expected_outcome:
    outcome: dormant_announced
    emission_channel: none
    normalised_fields: []
  message_template: ''
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: tensorrt_llm_lora_config_max_loras_ignored_when_lora_dir_empty
  engine: tensorrt
  library: tensorrt
  invariant_under_test: BaseLlmArgs.validate flags max_loras ignored when lora_config is provided and lora_dir is empty
  severity: warning
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: BaseLlmArgs validators (bottom half)
    method: validate_lora_config_consistency
    line_at_scan: 60
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.lora_config:
        present: true
      tensorrt_llm.max_loras:
        present: true
        not_equal: None
      tensorrt_llm.lora_dir:
        present: true
        exact: []
  kwargs_positive:
    lora_config: LoraConfig(max_loras=10, lora_dir=[])
    max_loras: 20
  kwargs_negative:
    lora_config: LoraConfig(max_loras=10, lora_dir=['dir'])
    max_loras: 20
  expected_outcome:
    outcome: warning
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: "max_loras is ignored when lora_config is provided and lora_dir is empty."
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: tensorrt_llm_lora_config_max_cpu_loras_ignored_when_lora_target_modules_empty
  engine: tensorrt
  library: tensorrt
  invariant_under_test: BaseLlmArgs.validate flags max_cpu_loras ignored when lora_config is provided and lora_target_modules is empty
  severity: warning
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: BaseLlmArgs validators (bottom half)
    method: validate_lora_config_consistency
    line_at_scan: 70
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.lora_config:
        present: true
      tensorrt_llm.max_cpu_loras:
        present: true
        not_equal: None
      tensorrt_llm.lora_target_modules:
        present: true
        exact: []
  kwargs_positive:
    lora_config: LoraConfig(max_cpu_loras=10, lora_target_modules=[])
    max_cpu_loras: 20
  kwargs_negative:
    lora_config: LoraConfig(max_cpu_loras=10, lora_target_modules=['module'])
    max_cpu_loras: 20
  expected_outcome:
    outcome: warning
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: "max_cpu_loras is ignored when lora_config is provided and lora_target_modules is empty."
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
```
