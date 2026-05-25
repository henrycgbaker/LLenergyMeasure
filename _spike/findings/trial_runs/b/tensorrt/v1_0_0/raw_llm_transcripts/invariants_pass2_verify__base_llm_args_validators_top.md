# invariants_pass2_verify extraction transcript: base_llm_args_validators_top

- chunk_description: tensorrt_llm.BaseLlmArgs @field_validator + first @model_validators
- expected_namespaces: ['tensorrt_llm']
- attempts: 1
- elapsed_sec: 56.07
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
- id: tensorrt_llm_max_batch_size_gt_max_num_tokens
  severity: warning
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.max_batch_size:
        '>': 1
  invariant_under_test: BaseLlmArgs.validate flags `max_batch_size` > max_num_tokens
- id: tensorrt_llm_bfloat16_on_pre_sm80_gpu
  severity: error
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.dtype: bfloat16
  invariant_under_test: BaseLlmArgs.validate flags bfloat16 on pre-SM80 GPU


INPUT 2 - THE SOURCE PASS 1 READ:

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
            model_format = get_model_format(
                self.model, trust_remote_code=self.trust_remote_code)

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
        else:
            assert isinstance(
                build_config,
                BuildConfig), f"build_config is not initialized: {build_config}"
        return self

    @model_validator(mode="after")
    def set_runtime_knobs_from_build_config(self):
        # TODO: remove this after PyT become default to adapt PyT with build_config as input
        assert self.build_config is not None, "build_config is not initialized"
        if self.backend == "pytorch":
            if self.build_config:
                for key in [
                        "max_batch_size", "max_num_tokens", "max_seq_len",
                        "max_input_len",
...<truncated>...
```

### Response

```
confirmed:
- tensorrt_llm_model_not_str_or_path
- tensorrt_llm_bfloat16_on_pre_sm80_gpu

flagged:
- id: tensorrt_llm_max_batch_size_gt_build_config_max_batch_size
  reason: Source has `if self.max_batch_size > self.build_config.max_batch_size` but invariant says `> 1`
  fix: correct_predicate:cross-field check
- id: tensorrt_llm_max_num_tokens_gt_build_config_max_num_tokens
  reason: Same as above, source has `if self.max_num_tokens > self.build_config.max_num_tokens` but invariant says `> 1`
  fix: correct_predicate:cross-field check
- id: tensorrt_llm_max_batch_size_gt_max_num_tokens
  reason: Source does not have a direct comparison between max_batch_size and max_num_tokens, only a warning when max_batch_size > max_num_tokens
  fix: drop
```
