# invariants_pass3_extend extraction transcript: base_llm_args_validators_top

- chunk_description: tensorrt_llm.BaseLlmArgs @field_validator + first @model_validators
- expected_namespaces: ['tensorrt_llm']
- attempts: 1
- elapsed_sec: 114.62
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
  invariant_under_test: BaseLlmArgs.validate flags model not str or Path
- id: tensorrt_llm_bfloat16_on_pre_sm80_gpu
  severity: error
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.dtype: bfloat16
  invariant_under_test: BaseLlmArgs.validate flags bfloat16 on pre-SM80 GPU
- id: tensorrt_llm_custom_tokenizer_invalid
  severity: error
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.custom_tokenizer:
        present: true
  invariant_under_test: BaseLlmArgs.validate flags custom tokenizer invalid
- id: tensorrt_llm_custom_tokenizer_not_compatible_with_tokenizer_object
  severity: error
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.custom_tokenizer:
        present: true
      tensorrt_llm.tokenizer:
        present: true
        type_is_not:
        - None
  invariant_under_test: BaseLlmArgs.validate flags custom tokenizer not compatible
    with tokenizer object


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
    @field_validator('env_overrides', mode='before')
    @classmethod
    def coerce_env_overrides_to_str(cls, v):
        """Coerce env_overrides values to strings for os.environ compatibility."""
        if v is None:
            return v
        return {str(k): str(val) for k, val in v.items()}

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
        if os.getenv("RAY_LOCAL_WORLD_SIZE") is not None:
            return info.data.get("tensor_parallel_size")
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
            enable_lm_head_tp_in_adp=self.enable_lm_head_tp_in_adp,
            pp_partition=self.pp_partition,
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
        elif self.custom_tokenizer:
            # If tokenizer is already a tokenizer object, custom_tokenizer is not compatible
            if isinstance(self.tokenizer,
                          (TokenizerBase, PreTrainedTokenizerBase)):
                raise ValueError(
                    "Cannot use custom_tokenizer when tokenizer is already a tokenizer object. "
                    "Please specify a tokenizer path or leave it as None to load from model path."
                )

            # Support short aliases for built-in tokenizers
            TOKENIZER_ALIASES = {
                'deepseek_v32':
                'tensorrt_llm.tokenizer.deepseek_v32.DeepseekV32Tokenizer',
            }

            tokenizer_path = TOKENIZER_ALIASES.get(self.custom_tokenizer,
                                                   self.custom_tokenizer)

            # Dynamically import and use custom tokenizer
            from importlib import import_module
            try:
                module_path, class_name = tokenizer_path.rsplit('.', 1)
                module = import_module(module_path)
                tokenizer_class = getattr(module, class_name)
                # Use tokenizer path if specified, otherwise use model path
                load_path = self.tokenizer if self.tokenizer else self.model
                self.tokenizer = tokenizer_class.from_pretrained(
                    load_path,
                    trust_remote_code=self.trust_remote_code,
                    use_fast=self.tokenizer_mode != 'slow')
            except (ValueError, ImportError, AttributeError) as e:
                raise ValueError(
                    f"Failed to load custom tokenizer '{self.custom_tokenizer}': {e}. "
                    "Expected format: 'module.path.ClassName' or a recognized alias."
                ) from e
        else:
            self.tokenizer = tokenizer_factory(
                self.tokenizer,
                trust_remote_code=self.trust_remote_code,
                use_fast=self.tokenizer_mode != 'slow')
        return self




OUTPUT FORMAT: YAML document with the SAME shape as the original
invariants prompt's output:

invariants:
- id: <snake_case_unique_id>
  engine: tensorrt
  library: tensorrt
  invariant_under_test: <one-line: what the library checks here>
  severity: <error|dormant|warning>
  native_type: <e.g. tensorrt.GenerationConfig or tensorrt.EngineArgs>
  miner_source:
    path: <file path>
    method: <validate|__init__|_verify_args|...>
    line_at_scan: <approximate line number if visible>
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.<field>: <value or predicate>
  kwargs_positive:
    <field>: <value that TRIGGERS the invariant>
  kwargs_negative:
    <field>: <value that does NOT trigger>
  expected_outcome:
    outcome: <error|dormant_announced|warn
...<truncated>...
```

### Response

```
invariants:
- id: tensorrt_llm_dtype_not_bfloat16_on_pre_sm80_gpu
  engine: tensorrt
  library: tensorrt
  invariant_under_test: BaseLlmArgs.validate flags dtype not bfloat16 on pre-SM80 GPU
  severity: error
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: BaseLlmArgs.py
    method: validate_dtype
    line_at_scan: 123
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.dtype: {present: true, type_is_not: [str]}
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

- id: tensorrt_llm_custom_tokenizer_not_compatible_with_tokenizer_object_alias
  engine: tensorrt
  library: tensorrt
  invariant_under_test: BaseLlmArgs.validate flags custom tokenizer not compatible with tokenizer object alias
  severity: error
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: BaseLlmArgs.py
    method: validate_and_init_tokenizer
    line_at_scan: 234
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.custom_tokenizer: {present: true, not_in: ['deepseek_v32']}
  kwargs_positive:
    custom_tokenizer: deepseek_v32
  kwargs_negative:
    custom_tokenizer: other_alias
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: 'Cannot use custom_tokenizer when tokenizer is already a tokenizer object. Please specify a tokenizer path or leave it as None to load from model path.'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: tensorrt_llm_custom_tokenizer_invalid_import_module_error
  engine: tensorrt
  library: tensorrt
  invariant_under_test: BaseLlmArgs.validate flags custom tokenizer invalid import module error
  severity: error
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: BaseLlmArgs.py
    method: validate_and_init_tokenizer
    line_at_scan: 345
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.custom_tokenizer: {present: true}
  kwargs_positive:
    custom_tokenizer: invalid_module_path
  kwargs_negative:
    custom_tokenizer: valid_module_path
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: 'Failed to load custom tokenizer \'{}\''.format(self.custom_tokenizer)
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
```
