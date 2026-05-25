# invariants_pass2_verify extraction transcript: base_llm_args_validators_bottom

- chunk_description: tensorrt_llm.BaseLlmArgs remaining @model_validators
- expected_namespaces: ['tensorrt_llm']
- attempts: 1
- elapsed_sec: 13.50
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser doing PASS 2 (verification) of multi-pass
invariant extraction. PASS 1 already extracted invariants from
tensorrt v0.21.0 for ONE chunk of source. Your job is to
REVIEW each emitted invariant against the source and flag any that
look WRONG.

INPUT 1 - PASS 1 EXTRACTED INVARIANTS (the candidate list):

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


INPUT 2 - THE SOURCE PASS 1 READ:

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
                    max_draft_tokens=self.speculative_config.max_draft_len,
                    draft_model_path=self.speculative_config.
                    pytorch_weights_path)
            elif isinstance(self.speculative_config, MTPDecodingConfig):
                from tensorrt_llm._torch.speculative import MTPConfig
                self.speculative_config = MTPConfig(
                    num_nextn_predict_layers=self.speculative_config.
                    num_nextn_predict_layers,
                    max_batch_size=self.build_config.max_batch_size,
                    use_relaxed_acceptance_for_thinking=self.speculative_config.
                    use_relaxed_acceptance_for_thinking,
     
...<truncated>...
```

### Response

```
confirmed:
- tensorrt_llm_speculative_config_type_not_recognized
- tensorrt_llm_max_lora_rank_ignored_when_lora_config_provided
- tensorrt_llm_max_loras_ignored_when_lora_config_provided
- tensorrt_llm_max_cpu_loras_ignored_when_lora_config_provided

flagged:
- id: tensorrt_llm_enable_lora_ignored_when_lora_config_provided_for_pytorch_backend
  reason: Source has a more specific condition `self.enable_lora and self.lora_config is not None and self.backend in ['pytorch', '_autodeploy']` which is not fully captured by the invariant.
  fix: correct_predicate:exact

- id: tensorrt_llm_both_lora_dir_and_lora_target_modules_empty
  reason: Source has a more specific condition `len(self.lora_config.lora_dir) == 0 and len(self.lora_config.lora_target_modules) == 0` which is not fully captured by the invariant.
  fix: correct_predicate:exact
```
