# hybrid extraction transcript: hybrid_d_ab_tensorrt_extension

- chunk_description: Hybrid d-ab on tensorrt: (a) output + source -> extension
- expected_namespaces: ['tensorrt.*']
- attempts: 1
- elapsed_sec: 63.12
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser working as a SECOND PASS on top of a
deterministic miner's output. The deterministic miner has already
extracted invariants from tensorrt v0.20.0; your job is to
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




=== CHUNK: base_llm_args_validators_bottom ===
=== CONTEXT ===
Continuation of BaseLlmArgs validators (second half). Same rules as the top half. Use namespace `tensorrt_llm`.

=== SOURCE: BaseLlmArgs validators (bottom half) ===




=== CHUNK: trt_llm_args_validators ===
=== CONTEXT ===
TrtLlmArgs has 1 @field_validator (calib_config init) + 3 @model_validator decorators. The validate_enable_build_cache method has the most pertinent `raise ValueError(...)` block.

=== SOURCE: TrtLlmArgs validators ===




=== CHUNK: lookahead_validator ===
=== CONTEXT ===
LookaheadDecodingConfig has ONE @field_validator decorator applied to THREE fields (max_window_size, max_ngram_size, max_verification_set_size). This MUST emit THREE separate invariants - one per field. Each invariant has predicate `<= 0` (i.e. `<= 0` triggers ValueError). Use namespace `tensorrt_llm`.

=== SOURCE: LookaheadDecodingConfig ===
class LookaheadDecodingConfig(DecodingBaseConfig, PybindMirror):
    """
    Configuration for lookahead speculative decoding.
    """

    max_window_size: int = Field(
        default=_LookaheadDecodingConfig.get_default_lookahead_decoding_window(
        ),
        description="Number of NGrams in lookahead branch per step.")
    max_ngram_size: int = Field(
        default=_LookaheadDecodingConfig.get_default_lookahead_decoding_ngram(),
        description="Number of tokens per NGram.")
    max_verification_set_size: int = Field(
        default=_LookaheadDecodingConfig.
        get_default_lookahead_decoding_verification_set(),
        description="Number of NGrams in verification branch per step.")

    @validator('max_window_size', 'max_ngram_size', 'max_verification_set_size')
    def validate_positive_values(cls, v):
        if v <= 0:
            raise ValueError(f"Value must be positive, got {v}")
        return v

    def __init__(self, **data):
        super().__init__(**data)
        self._check_fields()

    def calculate_speculative_resource(self):
        return _LookaheadDecodingConfig.calculate_speculative_resource_tuple(
            self.max_window_size, self.max_ngram_size,
            self.max_verification_set_size)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    def _to_pybind(self):
        return _LookaheadDecodingConfig(self.max_window_size,
                                        self.max_ngram_size,
                                        self.max_verification_set_size)

    decoding_type: ClassVar[str] = "Lookahead"





=== CHUNK: calib_config_literal ===
=== CONTEXT ===
CalibConfig.device is typed `Literal['cuda', 'cpu']`. Pydantic enforces this at construction time - any value other than 'cuda' or 'cpu' raises ValidationError. Emit this as ONE invariant with predicate=not_in, severity=error.

=== SOURCE: CalibConfig ===
class CalibConfig(BaseModel):
    """
    Calibration config
...<truncated>...
```

### Response

```
added_by_llm_verifier:
  - id: tensorrt_lookaheaddecodingconfig_max_window_size_le_0
    field: max_window_size
    predicate_kind: le
    severity: error
    added_by: llm_verifier
    flagged_for_review: true
  - id: tensorrt_lookaheaddecodingconfig_max_ngram_size_le_0
    field: max_ngram_size
    predicate_kind: le
    severity: error
    added_by: llm_verifier
    flagged_for_review: true
  - id: tensorrt_lookaheaddecodingconfig_max_verification_set_size_le_0
    field: max_verification_set_size
    predicate_kind: le
    severity: error
    added_by: llm_verifier
    flagged_for_review: true

flagged_spurious_in_deterministic:
  - id: tensorrt_torch_llm_load_format_invalid
    reason: unclear what this invariant is checking, as the load_format field is not defined in the provided source code.

missed_diagnosis:
  - id: tensorrt_calibconfig_device_not_cuda_or_cpu
    why_missed: The deterministic miner missed this invariant because it was defined using a Literal type annotation, which is not explicitly checked by the miner.
  - id: tensorrt_batchingtype_not_static_or_inflight
    why_missed: The deterministic miner missed this invariant because it was defined using an StrEnum class, which is not explicitly checked by the miner.
```
