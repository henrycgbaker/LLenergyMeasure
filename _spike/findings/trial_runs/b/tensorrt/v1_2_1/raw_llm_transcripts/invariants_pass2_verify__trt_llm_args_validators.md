# invariants_pass2_verify extraction transcript: trt_llm_args_validators

- chunk_description: tensorrt_llm.TrtLlmArgs @field_validator + @model_validator methods
- expected_namespaces: ['tensorrt_llm']
- attempts: 1
- elapsed_sec: 15.03
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser doing PASS 2 (verification) of multi-pass
invariant extraction. PASS 1 already extracted invariants from
tensorrt v1.2.1 for ONE chunk of source. Your job is to
REVIEW each emitted invariant against the source and flag any that
look WRONG.

INPUT 1 - PASS 1 EXTRACTED INVARIANTS (the candidate list):

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


INPUT 2 - THE SOURCE PASS 1 READ:

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
                self.max_b
...<truncated>...
```

### Response

```
confirmed:
- tensorrt_speculative_config_type_not_supported_by_backend
- tensorrt_speculative_config_unrecognized_type
- tensorrt_max_draft_len_not_set_for_speculative_decoding
- tensorrt_eagle3_draft_model_not_specified
- tensorrt_max_batch_size_overridden_by_build_config
- tensorrt_max_num_tokens_overridden_by_build_config

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
```
