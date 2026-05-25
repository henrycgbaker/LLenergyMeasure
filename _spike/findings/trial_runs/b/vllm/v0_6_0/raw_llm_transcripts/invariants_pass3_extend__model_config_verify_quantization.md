# invariants_pass3_extend extraction transcript: model_config_verify_quantization

- chunk_description: vllm.ModelConfig._verify_quantization
- expected_namespaces: ['vllm']
- attempts: 1
- elapsed_sec: 51.22
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser doing PASS 3 (extension) of multi-pass
invariant extraction. PASS 1 already extracted N invariants from
vllm v0.6.0 for ONE chunk of source. Your job is to
identify ADDITIONAL invariants PASS 1 missed.

INPUT 1 - PASS 1 EXTRACTED INVARIANTS (already known; do NOT re-emit):

invariants:
- id: vllm_quantization_method_not_in_allowlist
  severity: error
  match:
    engine: vllm
    fields:
      vllm.quantization:
        present: true
        not_in:
        - awq
        - gptq
        - squeezellm
        - fp8
  invariant_under_test: ModelConfig._verify_quantization flags quantization not in
    allowlist
- id: vllm_rocm_quantization_method_not_in_allowlist
  severity: error
  match:
    engine: vllm
    fields:
      vllm.quantization:
        present: true
        not_in:
        - awq
        - gptq
        - squeezellm
        - fp8
  invariant_under_test: ModelConfig._verify_quantization flags ROCm quantization not
    in allowlist
- id: vllm_tpu_quantization_method_not_in_allowlist
  severity: error
  match:
    engine: vllm
    fields:
      vllm.quantization:
        present: true
        not_in:
        - tpu_int8
  invariant_under_test: ModelConfig._verify_quantization flags TPU quantization not
    in allowlist
- id: vllm_neuron_quantization_method_not_in_allowlist
  severity: error
  match:
    engine: vllm
    fields:
      vllm.quantization:
        present: true
        not_in:
        - neuron_quant
  invariant_under_test: ModelConfig._verify_quantization flags Neuron quantization
    not in allowlist
- id: vllm_awq_quantization_not_optimized
  severity: warning
  match:
    engine: vllm
    fields:
      vllm.quantization: awq
  invariant_under_test: ModelConfig._verify_quantization flags AWQ quantization not
    optimized


INPUT 2 - PASS 2 FLAGGED ISSUES (for context; consider whether they hint
at related misses):

flagged:
- id: vllm_awq_quantization_not_optimized
  reason: Source raises a warning for non-optimized quantization methods, but the
    invariant does not match the exact list of optimized methods.
  fix: correct_predicate:not_in
- id: vllm_awq_quantization_rocm_not_enabled
  reason: Source sets an environment variable when AWQ is used with ROCm and VLLM_USE_TRITON_AWQ
    is not set, but the invariant does not capture this behavior.
  fix: drop


INPUT 3 - THE SOURCE PASS 1 READ:

=== CONTEXT ===
vllm.ModelConfig._verify_quantization - emit one invariant per `if <cond>: raise ...` block. Use namespace `vllm` for engine_params fields.

=== SOURCE: ModelConfig._verify_quantization ===
    def _verify_quantization(self) -> None:
        supported_quantization = [*QUANTIZATION_METHODS]
        rocm_supported_quantization = ["awq", "gptq", "squeezellm", "fp8"]
        optimized_quantization_methods = [
            "fp8", "marlin", "gptq_marlin_24", "gptq_marlin", "awq_marlin",
            "fbgemm_fp8", "compressed_tensors", "compressed-tensors",
            "experts_int8"
        ]
        tpu_supported_quantization = ["tpu_int8"]
        neuron_supported_quantization = ["neuron_quant"]
        if self.quantization is not None:
            self.quantization = self.quantization.lower()

        # Parse quantization method from the HF model config, if available.
        quant_cfg = self._parse_quant_hf_config()

        if quant_cfg is not None:
            quant_method = quant_cfg.get("quant_method", "").lower()

            # Detect which checkpoint is it
            for _, method in QUANTIZATION_METHODS.items():
                quantization_override = method.override_quantization_method(
                    quant_cfg, self.quantization)
                if quantization_override:
                    quant_method = quantization_override
                    self.quantization = quantization_override
                    break

            # Verify quantization configurations.
            if self.quantization is None:
                self.quantization = quant_method
            elif self.quantization != quant_method:
                raise ValueError(
                    "Quantization method specified in the model config "
                    f"({quant_method}) does not match the quantization "
                    f"method specified in the `quantization` argument "
                    f"({self.quantization}).")

        if self.quantization is not None:
            if self.quantization not in supported_quantization:
                raise ValueError(
                    f"Unknown quantization method: {self.quantization}. Must "
                    f"be one of {supported_quantization}.")
            if is_hip(
            ) and self.quantization not in rocm_supported_quantization:
                raise ValueError(
                    f"{self.quantization} quantization is currently not "
                    f"supported in ROCm.")
            if current_platform.is_tpu(
            ) and self.quantization not in tpu_supported_quantization:
                raise ValueError(
                    f"{self.quantization} quantization is currently not "
                    f"supported in TPU Backend.")
            if self.quantization not in optimized_quantization_methods:
                logger.warning(
                    "%s quantization is not fully "
                    "optimized yet. The speed can be slower than "
                    "non-quantized models.", self.quantization)
            if (self.quantization == "awq" and is_hip()
                    and not envs.VLLM_USE_TRITON_AWQ):
                logger.warning(
                    "Using AWQ quantization with ROCm, but VLLM_USE_TRITON_AWQ"
                    " is not set, enabling VLLM_USE_TRITON_AWQ.")
                envs.VLLM_USE_TRITON_AWQ = True
            if is_neuron(
            ) and self.quantization not in neuron_supported_quantization:
                raise ValueError(
                    f"{self.quantization} quantization is currently not "
                    f"supported in Neuron Backend.")




OUTPUT FORMAT: YAML document with the SAME shape as the original
invariants prompt's output:

invariants:
- id: <snake_case_unique_id>
  engine: vllm
  library: vllm
  invariant_under_test: <one-line: what the library checks here>
  severity: <error|dormant|warning>
  native_type: <e.g. vllm.GenerationConfig or vllm.EngineArgs>
  miner_source:
    path: <file path>
    method: <validate|__init__|_verify_args|...>
    line_at_scan: <approximate line number if visible>
  match:
    engine: vllm
    fields:
      vllm.<field>: <value or predicate>
  kwargs_positive:
    <field>: <value that TRIGGERS the invariant>
  kwargs_negative:
    <field>: <value that does NOT trigger>
  expected_outcome:
    outcome: <error|dormant_announced|warning>
    emission_channel: <none|logger_warning_once|logger_warning>
    normalised_fields: []
  message_template: '<the exact error/warning string from source>'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

PREDICATE FORMS (use the EXACT keys shown):
- Exact value:         `vllm.field: value`
- Not in list:         `vllm.field: {present: true, not_in: [a, b]}`
- Greater than:        `vllm.field: {'>': value}`
- Less than:           `vllm.field: {'<': value}`
- Type not in:         `vllm.field: {present: true, type_is_not: [TypeName]}`
- Presence:            `vllm.field: {present: true}`

CRITICAL RULES FOR PASS 3:
1. Return ONLY the YAML document. NO markdown code fences. NO commentary.
   First character must be `i` (from `invariants:`).
2. ONLY emit invariants that are NOT already in PASS 1's list (check by
   ID and by (field, predicate_kind) tuple).
3. Look ESPECIALLY for these PASS 1 blind spots:
   a. PER-FIELD invariants where PASS 1 emitted ONE entry covering many
      fields. Example: PASS 1 emitted `transformers_temperature_set_when_do_sample_false`
      but the source has SIMILAR if-blocks for top_p, top_k, min_p,
      typical_p, epsilon_cutoff, eta_cutoff. Emit ONE per field PASS 1
      misse
...<truncated>...
```

### Response

```
invariants:
- id: vllm_quantization_mismatch_between_hf_config_and_arg
  engine: vllm
  library: vllm
  invariant_under_test: ModelConfig._verify_quantization flags mismatch between quantization method in HF model config and `quantization` argument
  severity: error
  native_type: vllm.ModelConfig
  miner_source:
    path: ModelConfig.py
    method: _verify_quantization
    line_at_scan: 15
  match:
    engine: vllm
    fields:
      vllm.quantization: {present: true}
      vllm.quant_method_from_hf_config: {not_equal: vllm.quantization}
  kwargs_positive:
    quantization: awq
    quant_method_from_hf_config: gptq
  kwargs_negative:
    quantization: awq
    quant_method_from_hf_config: awq
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: 'Quantization method specified in the model config ({quant_method_from_hf_config}) does not match the quantization method specified in the `quantization` argument ({vllm.quantization}).'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: vllm_quantization_not_optimized_for_rocm
  engine: vllm
  library: vllm
  invariant_under_test: ModelConfig._verify_quantization flags non-optimized quantization methods for ROCm
  severity: warning
  native_type: vllm.ModelConfig
  miner_source:
    path: ModelConfig.py
    method: _verify_quantization
    line_at_scan: 25
  match:
    engine: vllm
    fields:
      vllm.quantization: {present: true}
      vllm.is_rocm: {present: true}
      vllm.optimized_quantization_methods_for_rocm: {not_in: [vllm.quantization]}
  kwargs_positive:
    quantization: awq
    is_rocm: True
  kwargs_negative:
    quantization: fp8
    is_rocm: True
  expected_outcome:
    outcome: warning
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: '%s quantization is not fully optimized yet for ROCm. The speed can be slower than non-quantized models.'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: vllm_awq_quantization_rocm_not_enabled_v2
  engine: vllm
  library: vllm
  invariant_under_test: ModelConfig._verify_quantization flags AWQ quantization with ROCm but VLLM_USE_TRITON_AWQ is not set
  severity: warning
  native_type: vllm.ModelConfig
  miner_source:
    path: ModelConfig.py
    method: _verify_quantization
    line_at_scan: 30
  match:
    engine: vllm
    fields:
      vllm.quantization: awq
      vllm.is_rocm: {present: true}
      vllm.VLLM_USE_TRITON_AWQ: {not_equal: True}
  kwargs_positive:
    quantization: awq
    is_rocm: True
  kwargs_negative:
    quantization: fp8
    is_rocm: True
  expected_outcome:
    outcome: warning
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: 'Using AWQ quantization with ROCm, but VLLM_USE_TRITON_AWQ is not set, enabling VLLM_USE_TRITON_AWQ.'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
```
