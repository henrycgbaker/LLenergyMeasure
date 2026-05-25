# invariants_pass3_extend extraction transcript: base_llm_args_validators_top

- chunk_description: tensorrt_llm.BaseLlmArgs @field_validator + first @model_validators
- expected_namespaces: ['tensorrt_llm']
- attempts: 1
- elapsed_sec: 70.39
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
- id: tensorrt_llm_model_not_str_or_path
  severity: error
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.model:
        present: true
        type_is_not:
        - str
        - Path
  invariant_under_test: BaseLlmArgs.validate flags `model` not str or Path
- id: tensorrt_llm_max_batch_size_gt_build_config_max_batch_size
  severity: error
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.max_batch_size:
        '>': 1
  invariant_under_test: BaseLlmArgs.validate flags `max_batch_size` > build_config.max_batch_size
- id: tensorrt_llm_max_num_tokens_gt_build_config_max_num_tokens
  severity: error
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.max_num_tokens:
        '>': 1
  invariant_under_test: BaseLlmArgs.validate flags `max_num_tokens` > build_config.max_num_tokens
- id: tensorrt_llm_max_seq_len_not_equal_build_config_max_seq_len
  severity: warning
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.max_seq_len:
        present: true
        not_equal: 1
  invariant_under_test: BaseLlmArgs.validate flags `max_seq_len` != build_config.max_seq_len
- id: tensorrt_llm_max_beam_width_not_equal_build_config_max_beam_width
  severity: warning
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.max_beam_width:
        present: true
        not_equal: 1
  invariant_under_test: BaseLlmArgs.validate flags `max_beam_width` != build_config.max_beam_width
- id: tensorrt_llm_max_input_len_not_equal_build_config_max_input_len
  severity: warning
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.max_input_len:
        present: true
        not_equal: 1
  invariant_under_test: BaseLlmArgs.validate flags `max_input_len` != build_config.max_input_len


INPUT 2 - PASS 2 FLAGGED ISSUES (for context; consider whether they hint
at related misses):

(no flags)

INPUT 3 - THE SOURCE PASS 1 READ:

=== CONTEXT ===
tensorrt_llm uses Pydantic v2 validators (NOT `if X: raise` patterns). Each `@field_validator(field)` decorator + method is ONE invariant; each `@model_validator(mode='after')` decorator + method may contain multiple `raise ValueError` branches (each is its own invariant). Emit one invariant per `raise` statement OR per @field_validator method. Use namespace `tensorrt_llm`.

Examples of validator forms to extract:
- `@field_validator('model')\ndef validate_model(...):\n    if not isinstance(v, ...): raise ValueError(...)` ->   severity=error, predicate=type_is_not.
- `@model_validator(mode='after')\ndef validate_build_config_with_runtime_params(self):\n    if self.max_batch_size > self.build_config.max_batch_size: raise ValueError(...)` -> severity=error, cross-field check.

NOTE: this chunk shows the FIRST HALF of BaseLlmArgs validators; the rest are in a separate chunk.

=== SOURCE: BaseLlmArgs validators (top half) ===
    @field_validator("dtype")
    @classmethod
    def validate_dtype(cls, v, info):
        if torch.cuda.get_device_properties(0).major < 8:
            if v == 'auto':
                v = 'float16'
            if v == 'bfloat16':
                raise RuntimeError("Pre SM 80 GPUs do not support bfloat16")
        return v

    @field_validator("quant_config", mode='before')
    @classmethod
    def validate_quant_config(cls, v, info):
        if v is None:
            v = QuantConfig()
        return v

    @field_validator("gpus_per_node", mode='before')
    @classmethod
    def validate_gpus_per_node(cls, v, info):
        if v is None:
            logger.warning(
                f"Using default gpus_per_node: {torch.cuda.device_count()}")
            v = torch.cuda.device_count()
        return v

    @field_validator("model")
    @classmethod
    def validate_model(cls, v, info):
        if not isinstance(v, (str, Path)):
            raise ValueError(f"Invalid model: {v}")
        return v

    @model_validator(mode="after")
    def validate_parallel_config(self):
        if self.moe_cluster_parallel_size is None:
            self.moe_cluster_parallel_size = -1

        if self.moe_tensor_parallel_size is None:
            self.moe_tensor_parallel_size = -1

        if self.moe_expert_parallel_size is None:
            self.moe_expert_parallel_size = -1

        self._parallel_config = _ParallelConfig(
            tp_size=self.tensor_parallel_size,
            pp_size=self.pipeline_parallel_size,
            cp_size=self.context_parallel_size,
            gpus_per_node=self.gpus_per_node,
            moe_cluster_size=self.moe_cluster_parallel_size,
            moe_tp_size=self.moe_tensor_parallel_size,
            moe_ep_size=self.moe_expert_parallel_size,
            enable_attention_dp=self.enable_attention_dp,
            cp_config=self.cp_config)
        return self

    @model_validator(mode="after")
    def set_default_max_input_len(self):
        if self.max_input_len is None:
            self.max_input_len = 1024
        return self

    @model_validator(mode="after")
    def validate_and_init_tokenizer(self):
        """Initialize tokenizer based on configuration."""
        if self.skip_tokenizer_init:
            self.tokenizer = None
        else:
            self.tokenizer = tokenizer_factory(
                self.tokenizer,
                trust_remote_code=self.trust_remote_code,
                use_fast=self.tokenizer_mode != 'slow')
        return self

    @model_validator(mode="after")
    def validate_model_format_misc(self):
        '''
        Load the model format, and do the following:

        1. Load the build_config if got an engine.
        2. Load the parallel_config if got a checkpoint.
        '''
        model_obj = _ModelWrapper(self.model)

        if model_obj.is_local_model and self.backend not in [
                'pytorch', '_autodeploy'
        ]:
            # Load parallel_config from the engine.
            model_format = get_model_format(self.model)

            if model_format is _ModelFormatKind.TLLM_ENGINE:
                if self.build_config is not None:
                    logger.warning(
                        "The build_config is ignored for model format of TLLM_ENGINE."
                    )
                self._load_config_from_engine(model_obj.model_dir)
                runtime_defaults = self._pretrained_config.runtime_defaults
                if runtime_defaults:
                    self.kv_cache_config.fill_empty_fields_from_runtime_defaults(
                        runtime_defaults)

            # Load parallel_config from the checkpoint.
            elif model_format is _ModelFormatKind.TLLM_CKPT:
                # We need to create a temporary instance to call _load_config_from_ckpt
                self._load_config_from_ckpt(model_obj.model_dir)
        else:
            model_format = _ModelFormatKind.HF

        # Store the model format in the values
        self._model_format = model_format
        return self

    @model_validator(mode="after")
    def init_build_config(self):
        """
        Creating a default BuildConfig if none is provided
        """
        if self.build_config is None:
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

        assert isinstance(
            self.bui
...<truncated>...
```

### Response

```
invariants:
- id: tensorrt_llm_dtype_not_bfloat16_on_pre_sm80_gpus
  engine: tensorrt
  library: tensorrt
  invariant_under_test: BaseLlmArgs.validate flags `dtype` not bfloat16 on pre SM 80 GPUs
  severity: error
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: BaseLlmArgs validators (top half)
    method: validate_dtype
    line_at_scan: 5
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.dtype:
        present: true
        type_is_not:
        - str
  kwargs_positive:
    dtype: bfloat16
  kwargs_negative:
    dtype: float16
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: 'Pre SM 80 GPUs do not support bfloat16'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: tensorrt_llm_quant_config_not_none
  engine: tensorrt
  library: tensorrt
  invariant_under_test: BaseLlmArgs.validate flags `quant_config` not None
  severity: dormant
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: BaseLlmArgs validators (top half)
    method: validate_quant_config
    line_at_scan: 15
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.quant_config:
        present: true
  kwargs_positive:
    quant_config: None
  kwargs_negative:
    quant_config: QuantConfig()
  expected_outcome:
    outcome: dormant_announced
    emission_channel: none
    normalised_fields: []
  message_template: ''
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: tensorrt_llm_gpus_per_node_not_none
  engine: tensorrt
  library: tensorrt
  invariant_under_test: BaseLlmArgs.validate flags `gpus_per_node` not None
  severity: dormant
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: BaseLlmArgs validators (top half)
    method: validate_gpus_per_node
    line_at_scan: 25
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.gpus_per_node:
        present: true
  kwargs_positive:
    gpus_per_node: None
  kwargs_negative:
    gpus_per_node: torch.cuda.device_count()
  expected_outcome:
    outcome: dormant_announced
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: 'Using default gpus_per_node: {torch.cuda.device_count()}'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: tensorrt_llm_max_batch_size_not_equal_build_config_max_batch_size
  engine: tensorrt
  library: tensorrt
  invariant_under_test: BaseLlmArgs.validate flags `max_batch_size` != build_config.max_batch_size
  severity: warning
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: BaseLlmArgs validators (top half)
    method: validate_build_config_with_runtime_params
    line_at_scan: 105
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.max_batch_size:
        present: true
        not_equal: 1
  kwargs_positive:
    max_batch_size: 2
    build_config:
      max_batch_size: 1
  kwargs_negative:
    max_batch_size: 1
    build_config:
      max_batch_size: 1
  expected_outcome:
    outcome: warning
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: 'max_batch_size [{self.max_batch_size}] is overridden by build_config.max_batch_size [{self.build_config.max_batch_size}] in build_config'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: tensorrt_llm_max_beam_width_not_equal_build_config_max_beam_width
  engine: tensorrt
  library: tensorrt
  invariant_under_test: BaseLlmArgs.validate flags `max_beam_width` != build_config.max_beam_width
  severity: warning
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: BaseLlmArgs validators (top half)
    method: validate_build_config_with_runtime_params
    line_at_scan: 115
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.max_beam_width:
        present: true
        not_equal: 1
  kwargs_positive:
    max_beam_width: 2
    build_config:
      max_beam_width: 1
  kwargs_negative:
    max_beam_width: 1
    build_config:
      max_beam_width: 1
  expected_outcome:
    outcome: warning
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: 'max_beam_width [{self.max_beam_width}] is overridden by build_config.max_beam_width [{self.build_config.max_beam_width}] in build_config'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: tensorrt_llm_max_input_len_not_equal_build_config_max_input_len
  engine: tensorrt
  library: tensorrt
  invariant_under_test: BaseLlmArgs.validate flags `max_input_len` != build_config.max_input_len
  severity: warning
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: BaseLlmArgs validators (top half)
    method: validate_build_config_with_runtime_params
    line_at_scan: 125
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.max_input_len:
        present: true
        not_equal: 1
  kwargs_positive:
    max_input_len: 1025
    build_config:
      max_input_len: 1024
  kwargs_negative:
    max_input_len: 1024
    build_config:
      max_input_len: 1024
  expected_outcome:
    outcome: warning
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: 'max_input_len [{self.max_input_len}] is overridden by build_config.max_input_len [{self.build_config.max_input_len}] in build_config'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
```
