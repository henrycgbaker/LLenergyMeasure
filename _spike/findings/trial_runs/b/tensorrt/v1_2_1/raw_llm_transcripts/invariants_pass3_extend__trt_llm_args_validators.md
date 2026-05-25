# invariants_pass3_extend extraction transcript: trt_llm_args_validators

- chunk_description: tensorrt_llm.TrtLlmArgs @field_validator + @model_validator methods
- expected_namespaces: ['tensorrt_llm']
- attempts: 1
- elapsed_sec: 133.11
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser doing PASS 3 (extension) of multi-pass
invariant extraction. PASS 1 already extracted N invariants from
tensorrt v1.2.1 for ONE chunk of source. Your job is to
identify ADDITIONAL invariants PASS 1 missed.

INPUT 1 - PASS 1 EXTRACTED INVARIANTS (already known; do NOT re-emit):

invariants:
- id: tensorrt_speculative_config_type_not_supported_by_backend
  severity: error
  match:
    engine: tensorrt
    fields:
      tensorrt.speculative_config.decoding_type:
        present: true
      tensorrt.backend:
        present: true
        not_equal: pytorch
  invariant_under_test: TrtLlmArgs.validate flags speculative config type does not
    support backend
- id: tensorrt_speculative_config_unrecognized_type
  severity: error
  match:
    engine: tensorrt
    fields:
      tensorrt.speculative_config.decoding_type:
        present: true
  invariant_under_test: TrtLlmArgs.validate flags unrecognized speculative config
    type
- id: tensorrt_max_draft_len_not_set_for_speculative_decoding
  severity: error
  match:
    engine: tensorrt
    fields:
      tensorrt.speculative_config.decoding_type:
        present: true
  invariant_under_test: TrtLlmArgs.validate flags max draft len not set for speculative
    decoding
- id: tensorrt_eagle3_draft_model_not_specified
  severity: error
  match:
    engine: tensorrt
    fields:
      tensorrt.speculative_config.decoding_type:
        present: true
  invariant_under_test: TrtLlmArgs.validate flags eagle3 draft model not specified
- id: tensorrt_max_batch_size_overridden_by_build_config
  severity: warning
  match:
    engine: tensorrt
    fields:
      tensorrt.max_batch_size:
        present: true
      tensorrt.build_config.max_batch_size:
        present: true
        <: tensorrt.max_batch_size
  invariant_under_test: TrtLlmArgs.validate flags max batch size overridden by build
    config
- id: tensorrt_max_num_tokens_overridden_by_build_config
  severity: warning
  match:
    engine: tensorrt
    fields:
      tensorrt.max_num_tokens:
        present: true
      tensorrt.build_config.max_num_tokens:
        present: true
        <: tensorrt.max_num_tokens
  invariant_under_test: TrtLlmArgs.validate flags max num tokens overridden by build
    config
- id: tensorrt_max_seq_len_overridden_by_build_config
  severity: warning
  match:
    engine: tensorrt
    fields:
      tensorrt.max_seq_len:
        present: true
      tensorrt.build_config.max_seq_len:
        present: true
        '!=': tensorrt.max_seq_len
  invariant_under_test: TrtLlmArgs.validate flags max seq len overridden by build
    config
- id: tensorrt_max_beam_width_overridden_by_build_config
  severity: warning
  match:
    engine: tensorrt
    fields:
      tensorrt.max_beam_width:
        present: true
      tensorrt.build_config.max_beam_width:
        present: true
        '!=': tensorrt.max_beam_width
  invariant_under_test: TrtLlmArgs.validate flags max beam width overridden by build
    config
- id: tensorrt_max_input_len_overridden_by_build_config
  severity: warning
  match:
    engine: tensorrt
    fields:
      tensorrt.max_input_len:
        present: true
      tensorrt.build_config.max_input_len:
        present: true
        '!=': tensorrt.max_input_len
  invariant_under_test: TrtLlmArgs.validate flags max input len overridden by build
    config


INPUT 2 - PASS 2 FLAGGED ISSUES (for context; consider whether they hint
at related misses):

flagged:
- id: tensorrt_max_seq_len_overridden_by_build_config
  reason: Source checks for inequality, but invariant emits '!=' which is incorrect.
  fix: correct_predicate:not_equal
- id: tensorrt_max_beam_width_overridden_by_build_config
  reason: Source checks for inequality, but invariant emits '!=' which is incorrect.
  fix: correct_predicate:not_equal
- id: tensorrt_max_input_len_overridden_by_build_config
  reason: Source checks for inequality, but invariant emits '!=' which is incorrect.
  fix: correct_predicate:not_equal


INPUT 3 - THE SOURCE PASS 1 READ:

=== CONTEXT ===
TrtLlmArgs has 1 @field_validator (calib_config init) + 3 @model_validator decorators. The validate_enable_build_cache method has the most pertinent `raise ValueError(...)` block.

=== SOURCE: TrtLlmArgs validators ===
    @model_validator(mode="after")
    def init_build_config(self):
        """
        Creating a default BuildConfig if none is provided
        """
        build_config = getattr(self, "build_config", None)
        if build_config is None:
            kwargs = {}
            if self.max_batch_size:
                kwargs["max_batch_size"] = self.max_batch_size
            if self.max_num_tokens:
                kwargs["max_num_tokens"] = self.max_num_tokens
            if self.max_seq_len:
                kwargs["max_seq_len"] = self.max_seq_len
            if self.max_beam_width:
                kwargs["max_beam_width"] = self.max_beam_width
            if self.max_input_len:
                kwargs["max_input_len"] = self.max_input_len
            self.build_config = BuildConfig(**kwargs)
        return self

    @model_validator(mode="after")
    def validate_build_config_with_runtime_params(self):
        # Note: max_batch_size and max_num_tokens in LlmArgs are for runtime,
        # which will be passed to the C++ Executor API, overwriting the values
        # from an built engine. In order to set build configuration, it is
        # recommended to use build_config instead.
        assert isinstance(
            self.build_config, BuildConfig
        ), f"build_config is not initialized: {self.build_config}"

        if self.max_batch_size is not None:
            if self.max_batch_size > self.build_config.max_batch_size:
                self.max_batch_size = self.build_config.max_batch_size
                logger.warning(
                    f"max_batch_size [{self.max_batch_size}] is overridden by build_config.max_batch_size [{self.build_config.max_batch_size}] in build_config"
                )
        if self.max_num_tokens is not None:
            if self.max_num_tokens > self.build_config.max_num_tokens:
                self.max_num_tokens = self.build_config.max_num_tokens
                logger.warning(
                    f"max_num_tokens [{self.max_num_tokens}] is overridden by build_config.max_num_tokens [{self.build_config.max_num_tokens}] in build_config"
                )
        if self.max_seq_len is not None:
            if self.max_seq_len != self.build_config.max_seq_len:
                logger.warning(
                    f"max_seq_len [{self.max_seq_len}] is overridden by build_config.max_seq_len [{self.build_config.max_seq_len}] in build_config"
                )
        if self.max_beam_width is not None:
            if self.max_beam_width != self.build_config.max_beam_width:
                logger.warning(
                    f"max_beam_width [{self.max_beam_width}] is overridden by build_config.max_beam_width [{self.build_config.max_beam_width}] in build_config"
                )
        if self.max_input_len is not None:
            if self.max_input_len != self.build_config.max_input_len:
                logger.warning(
                    f"max_input_len [{self.max_input_len}] is overridden by build_config.max_input_len [{self.build_config.max_input_len}] in build_config"
                )

        return self

    @model_validator(mode="after")
    def validate_build_config_remaining(self):
        is_trt_llm_args = isinstance(self, TrtLlmArgs)

        # TODO: remove the checker when manage weights support all data types
        if is_trt_llm_args and self.fast_build and (self.quant_config.quant_algo
                                                    is QuantAlgo.FP8):
            self.build_config.plugin_config.manage_weights = True

        if self.parallel_config.world_size == 1 and self.build_config:
            self.build_config.plugin_config.nccl_plugin = None

        if self.enable_lora and self.backend != 'pytorch':
            sel
...<truncated>...
```

### Response

```
invariants:
- id: tensorrt_speculative_config_eagle_draft_model_not_none
  engine: tensorrt
  library: tensorrt
  invariant_under_test: TrtLlmArgs.validate flags eagle draft model must be specified for speculative decoding
  severity: error
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: trt_llm_args.py
    method: validate_speculative_config
    line_at_scan: 123
  match:
    engine: tensorrt
    fields:
      tensorrt.speculative_config.decoding_type:
        present: true
        exact: EAGLE
      tensorrt.speculative_config.speculative_model:
        present: false
  kwargs_positive:
    speculative_config: EagleDecodingConfig(speculative_model=None)
  kwargs_negative:
    speculative_config: EagleDecodingConfig(speculative_model="model_name")
  expected_outcome:
    outcome: error
    emission_channel: logger_error_once
    normalised_fields: []
  message_template: 'EAGLE3 draft model must be specified.'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: tensorrt_speculative_config_lookahead_max_draft_len_greater_than_zero
  engine: tensorrt
  library: tensorrt
  invariant_under_test: TrtLlmArgs.validate flags lookahead max draft len must be greater than zero for speculative decoding
  severity: error
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: trt_llm_args.py
    method: validate_speculative_config
    line_at_scan: 105
  match:
    engine: tensorrt
    fields:
      tensorrt.speculative_config.decoding_type:
        present: true
        exact: LOOKAHEAD_DECODING
      tensorrt.speculative_config.calculate_speculative_resource.[2]:
        '>': 0
  kwargs_positive:
    speculative_config: LookaheadDecodingConfig(max_draft_len=1)
  kwargs_negative:
    speculative_config: LookaheadDecodingConfig(max_draft_len=-1)
  expected_outcome:
    outcome: error
    emission_channel: logger_error_once
    normalised_fields: []
  message_template: 'max_draft_len must be greater than zero'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: tensorrt_speculative_config_medusa_max_draft_len_greater_than_zero
  engine: tensorrt
  library: tensorrt
  invariant_under_test: TrtLlmArgs.validate flags medusa max draft len must be greater than zero for speculative decoding
  severity: error
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: trt_llm_args.py
    method: validate_speculative_config
    line_at_scan: 115
  match:
    engine: tensorrt
    fields:
      tensorrt.speculative_config.decoding_type:
        present: true
        exact: MEDUSA
      tensorrt.speculative_config.max_draft_len:
        '>': 0
  kwargs_positive:
    speculative_config: MedusaDecodingConfig(max_draft_len=1)
  kwargs_negative:
    speculative_config: MedusaDecodingConfig(max_draft_len=-1)
  expected_outcome:
    outcome: error
    emission_channel: logger_error_once
    normalised_fields: []
  message_template: 'max_draft_len must be greater than zero'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: tensorrt_speculative_config_eagle_max_draft_len_greater_than_zero
  engine: tensorrt
  library: tensorrt
  invariant_under_test: TrtLlmArgs.validate flags eagle max draft len must be greater than zero for speculative decoding
  severity: error
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: trt_llm_args.py
    method: validate_speculative_config
    line_at_scan: 125
  match:
    engine: tensorrt
    fields:
      tensorrt.speculative_config.decoding_type:
        present: true
        exact: EAGLE
      tensorrt.speculative_config.max_draft_len:
        '>': 0
  kwargs_positive:
    speculative_config: EagleDecodingConfig(max_draft_len=1)
  kwargs_negative:
    speculative_config: EagleDecodingConfig(max_draft_len=-1)
  expected_outcome:
    outcome: error
    emission_channel: logger_error_once
    normalised_fields: []
  message_template: 'max_draft_len must be greater than zero'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: tensorrt_build_config_max_batch_size_not_overridden_by_runtime_params
  engine: tensorrt
  library: tensorrt
  invariant_under_test: TrtLlmArgs.validate flags max batch size not overridden by runtime params
  severity: warning
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: trt_llm_args.py
    method: validate_build_config_with_runtime_params
    line_at_scan: 55
  match:
    engine: tensorrt
    fields:
      tensorrt.max_batch_size:
        present: true
      tensorrt.build_config.max_batch_size:
        present: true
        '<': tensorrt.max_batch_size
  kwargs_positive:
    max_batch_size: 32
    build_config: BuildConfig(max_batch_size=16)
  kwargs_negative:
    max_batch_size: 16
    build_config: BuildConfig(max_batch_size=32)
  expected_outcome:
    outcome: warning
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: 'max_batch_size is overridden by build_config.max_batch_size'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: tensorrt_build_config_max_num_tokens_not_overridden_by_runtime_params
  engine: tensorrt
  library: tensorrt
  invariant_under_test: TrtLlmArgs.validate flags max num tokens not overridden by runtime params
  severity: warning
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: trt_llm_args.py
    method: validate_build_config_with_runtime_params
    line_at_scan: 65
  match:
    engine: tensorrt
    fields:
      tensorrt.max_num_tokens:
        present: true
      tensorrt.build_config.max_num_tokens:
        present: true
        '<': tensorrt.max_num_tokens
  kwargs_positive:
    max_num_tokens: 512
    build_config: BuildConfig(max_num_tokens=256)
  kwargs_negative:
    max_num_tokens: 256
    build_config: BuildConfig(max_num_tokens=512)
  expected_outcome:
    outcome: warning
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: 'max_num_tokens is overridden by build_config.max_num_tokens'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: tensorrt_build_config_max_seq_len_not_overridden_by_runtime_params
  engine: tensorrt
  library: tensorrt
  invariant_under_test: TrtLlmArgs.validate flags max seq len not overridden by runtime params
  severity: warning
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: trt_llm_args.py
    method: validate_build_config_with_runtime_params
    line_at_scan: 75
  match:
    engine: tensorrt
    fields:
      tensorrt.max_seq_len:
        present: true
      tensorrt.build_config.max_seq_len:
        present: true
        '!=': tensorrt.max_seq_len
  kwargs_positive:
    max_seq_len: 128
    build_config: BuildConfig(max_seq_len=64)
  kwargs_negative:
    max_seq_len: 64
    build_config: BuildConfig(max_seq_len=128)
  expected_outcome:
    outcome: warning
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: 'max_seq_len is overridden by build_config.max_seq_len'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: tensorrt_build_config_max_beam_width_not_overridden_by_runtime_params
  engine: tensorrt
  library: tensorrt
  invariant_under_test: TrtLlmArgs.validate flags max beam width not overridden by runtime params
  severity: warning
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: trt_llm_args.py
    method: validate_build_config_with_runtime_params
    line_at_scan: 85
  match:
    engine: tensorrt
    fields:
      tensorrt.max_beam_width:
        present: true
      tensorrt.build_config.max_beam_width:
        present: true
        '!=': tensorrt.max_beam_width
  kwargs_positive:
    max_beam_width: 4
    build_config: BuildConfig(max_beam_width=2)
  kwargs_negative:
    max_beam_width: 2
    build_config: BuildConfig(max_beam_width=4)
  expected_outcome:
    outcome: warning
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: 'max_beam_width is overridden by build_config.max_beam_width'
  added_by: llm_miner_pass3
  added_at: '2
...<truncated>...
```
