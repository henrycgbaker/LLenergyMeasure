# hybrid extraction transcript: hybrid_d_ab_tensorrt_extension

- chunk_description: Hybrid d-ab on tensorrt: (a) output + source -> extension
- expected_namespaces: ['tensorrt.*']
- attempts: 1
- elapsed_sec: 44.59
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser working as a SECOND PASS on top of a
deterministic miner's output. The deterministic miner has already
extracted invariants from tensorrt v1.2.1; your job is to
find what it MISSED, find what looks SPURIOUS, and diagnose WHY it
missed what it missed.

INPUT 1 - DETERMINISTIC MINER'S OUTPUT (this is what (a) found):

- tensorrt__autodeployllmargs_mla_backend_in_1_values: severity=error, field=tensorrt.mla_backend
- tensorrt__autodeployllmargs_model_factory_in_2_values: severity=error, field=tensorrt.model_factory
- tensorrt_autodeploy_free_mem_ratio_out_of_range: severity=error, field=tensorrt.free_mem_ratio
- tensorrt_basellmargs_load_format_in_2_values: severity=error, field=tensorrt.load_format
- tensorrt_basellmargs_tokenizer_mode_in_2_values: severity=error, field=tensorrt.tokenizer_mode
- tensorrt_batching_type_in_2_values: severity=error, field=tensorrt.batching_type
- tensorrt_build_cache_max_records_ge_1: severity=error, field=tensorrt.max_records
- tensorrt_calibconfig_device_in_2_values: severity=error, field=tensorrt.device
- tensorrt_quant_config_kv_cache_quant_algo_in_allowlist: severity=error, field=tensorrt.kv_cache_quant_algo
- tensorrt_quant_config_quant_algo_in_allowlist: severity=error, field=tensorrt.quant_algo
- tensorrt_raises_cuda_graph_max_batch_size_lt_0_cuda_graph_max_batch_size: severity=error, field=tensorrt.cuda_graph_max_batch_size
- tensorrt_raises_cuda_graph_max_batch_size_ne_0_cuda_graph_config: severity=error, field=tensorrt.cuda_graph_batch_sizes
- tensorrt_raises_dtype_eq_bfloat16_dtype: severity=error, field=tensorrt.dtype
- tensorrt_raises_enable_build_cache_not_type_buildcacheconfig_enable_build_cache: severity=error, field=tensorrt.enable_build_cache
- tensorrt_raises_max_batch_size_set_True_build_config_with_runtime_params: severity=error, field=tensorrt.max_batch_size
- tensorrt_raises_max_ngram_size_le_0_positive_values: severity=error, field=tensorrt.max_ngram_size
- tensorrt_raises_max_num_tokens_set_True_build_config_with_runtime_params: severity=error, field=tensorrt.max_num_tokens
- tensorrt_raises_max_verification_set_size_le_0_positive_values: severity=error, field=tensorrt.max_verification_set_size
- tensorrt_raises_max_window_size_le_0_positive_values: severity=error, field=tensorrt.max_window_size
- tensorrt_raises_model_not_type_model: severity=error, field=tensorrt.model
- tensorrt_raises_moe_load_balancer_type_str_moe_load_balancer: severity=error, field=tensorrt.moe_load_balancer
- tensorrt_raises_speculative_config_set_True_speculative_config: severity=error, field=tensorrt.speculative_config
- tensorrt_torch_llm_load_format_invalid: severity=error, field=tensorrt.load_format
- tensorrt_warns_backend_in_lora_config_consistency: severity=warn, field=tensorrt.enable_lora
- tensorrt_warns_build_config_set_True_model_format_misc: severity=warn, field=tensorrt.backend
- tensorrt_warns_lora_config_set_True_lora_config_consistency: severity=warn, field=tensorrt.lora_config
- tensorrt_warns_max_batch_size_set_True_set_runtime_knobs_from_build_config: severity=warn, field=tensorrt.backend
- tensorrt_warns_max_beam_width_set_True_build_config_with_runtime_params: severity=warn, field=tensorrt.max_beam_width
- tensorrt_warns_max_beam_width_set_True_set_runtime_knobs_from_build_config: severity=warn, field=tensorrt.backend
- tensorrt_warns_max_input_len_set_True_build_config_with_runtime_params: severity=warn, field=tensorrt.max_input_len
- tensorrt_warns_max_input_len_set_True_set_runtime_knobs_from_build_config: severity=warn, field=tensorrt.backend
- tensorrt_warns_max_lora_rank_set_True_lora_config_consistency: severity=warn, field=tensorrt.lora_config
- tensorrt_warns_max_num_tokens_set_True_set_runtime_knobs_from_build_config: severity=warn, field=tensorrt.backend
- tensorrt_warns_max_seq_len_set_True_build_config_with_runtime_params: severity=warn, field=tensorrt.max_seq_len
- tensorrt_warns_max_seq_len_set_True_set_runtime_knobs_from_build_config: severity=warn, field=tensorrt.backend

INPUT 2 - ENGINE SOURCE (the same source the deterministic miner read):

=== CHUNK: base_llm_args_validators_top ===
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
            pp_partition=self.pp_partit

=== CHUNK: base_llm_args_validators_bottom ===
=== CONTEXT ===
Continuation of BaseLlmArgs validators (second half). Same rules as the top half. Use namespace `tensorrt_llm`.

=== SOURCE: BaseLlmArgs validators (bottom half) ===

    @model_validator(mode="after")
    def validate_runtime_args(self):
        if self.max_batch_size is not None and self.max_num_tokens is not None:
            if self.max_batch_size > self.max_num_tokens:
                logger.warning(
                    f"max_batch_size [{self.max_batch_size}] should be less than or equal to max_num_tokens [{self.max_num_tokens}]"
                )
        return self

    @model_validator(mode="after")
    def validate_lora_c
...<truncated>...
```

### Response

```
added_by_llm_verifier:
  - id: tensorrt__calibconfig_device_not_in
    field: tensorrt.device
    predicate_kind: not_in
    values: ['cuda', 'cpu']
    severity: error
    added_by: llm_verifier
    flagged_for_review: true

  - id: tensorrt_lookaheaddecodingconfig_max_window_size_le_0
    field: tensorrt.max_window_size
    predicate_kind: <= 0
    severity: error
    added_by: llm_verifier
    flagged_for_review: true

  - id: tensorrt_lookaheaddecodingconfig_max_ngram_size_le_0
    field: tensorrt.max_ngram_size
    predicate_kind: <= 0
    severity: error
    added_by: llm_verifier
    flagged_for_review: true

  - id: tensorrt_lookaheaddecodingconfig_max_verification_set_size_le_0
    field: tensorrt.max_verification_set_size
    predicate_kind: <= 0
    severity: error
    added_by: llm_verifier
    flagged_for_review: true

flagged_spurious_in_deterministic:
  - id: tensorrt_warns_backend_in_lora_config_consistency
    reason: warning instead of error, and unclear why this is an invariant

missed_diagnosis:
  - id: tensorrt__calibconfig_device_not_in
    why_missed: Pydantic's Literal type was not recognized as a source of invariants

  - id: tensorrt_lookaheaddecodingconfig_max_window_size_le_0
    why_missed: field_validator applied to multiple fields, but only one invariant was emitted

  - id: tensorrt_lookaheaddecodingconfig_max_ngram_size_le_0
    why_missed: same reason as above

  - id: tensorrt_lookaheaddecodingconfig_max_verification_set_size_le_0
    why_missed: same reason as above
```
