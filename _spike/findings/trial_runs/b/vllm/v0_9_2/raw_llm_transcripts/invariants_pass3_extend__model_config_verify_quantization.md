# invariants_pass3_extend extraction transcript: model_config_verify_quantization

- chunk_description: vllm.ModelConfig._verify_quantization
- expected_namespaces: ['vllm']
- attempts: 1
- elapsed_sec: 83.38
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser doing PASS 3 (extension) of multi-pass
invariant extraction. PASS 1 already extracted N invariants from
vllm v0.9.2 for ONE chunk of source. Your job is to
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
        - fp8
        - marlin
        - modelopt
        - gptq_marlin_24
        - gptq_marlin
        - awq_marlin
        - fbgemm_fp8
        - compressed-tensors
        - experts_int8
        - quark
        - modelopt_fp4
        - bitblas
        - gptq_bitblas
  invariant_under_test: ModelConfig._verify_quantization flags quantization not in
    allowlist
- id: vllm_quantization_method_override_not_in_allowlist
  severity: error
  match:
    engine: vllm
    fields:
      vllm.quantization:
        present: true
        not_in:
        - marlin
        - bitblas
        - gptq_marlin_24
        - gptq_marlin
        - gptq_bitblas
        - awq_marlin
        - ipex
        - moe_wna16
  invariant_under_test: ModelConfig._verify_quantization flags quantization override
    not in allowlist
- id: vllm_quantization_method_mismatch
  severity: error
  match:
    engine: vllm
    fields:
      vllm.quantization:
        present: true
        not_equal: quant_method
  invariant_under_test: ModelConfig._verify_quantization flags quantization method
    mismatch


INPUT 2 - PASS 2 FLAGGED ISSUES (for context; consider whether they hint
at related misses):

flagged:
- id: vllm_quantization_method_override_not_in_allowlist
  reason: Source checks for overrides in a specific order, but invariant does not
    reflect this.
  fix: correct_predicate:not_equal


INPUT 3 - THE SOURCE PASS 1 READ:

=== CONTEXT ===
vllm.ModelConfig._verify_quantization - emit one invariant per `if <cond>: raise ...` block. Use namespace `vllm` for engine_params fields.

=== SOURCE: ModelConfig._verify_quantization ===
    def _verify_quantization(self) -> None:
        supported_quantization = me_quant.QUANTIZATION_METHODS
        optimized_quantization_methods = [
            "fp8", "marlin", "modelopt", "gptq_marlin_24", "gptq_marlin",
            "awq_marlin", "fbgemm_fp8", "compressed-tensors", "experts_int8",
            "quark", "modelopt_fp4", "bitblas", "gptq_bitblas"
        ]
        if self.quantization is not None:
            self.quantization = cast(me_quant.QuantizationMethods,
                                     self.quantization)

        # Parse quantization method from the HF model config, if available.
        quant_cfg = self._parse_quant_hf_config()

        if quant_cfg is not None:
            quant_method = quant_cfg.get("quant_method", "").lower()
            quant_method = quant_method.replace("compressed_tensors",
                                                "compressed-tensors")
            quant_cfg["quant_method"] = quant_method

            # Quantization methods which are overrides (i.e. they have a
            # `override_quantization_method` method) must be checked in order
            # of preference (this is particularly important for GPTQ).
            overrides = [
                "marlin",
                "bitblas",
                "gptq_marlin_24",
                "gptq_marlin",
                "gptq_bitblas",
                "awq_marlin",
                "ipex",
                "moe_wna16",
            ]
            quantization_methods = [
                q for q in supported_quantization if q not in overrides
            ]
            # Any custom overrides will be in quantization_methods so we place
            # them at the start of the list so custom overrides have preference
            # over the built in ones.
            quantization_methods = quantization_methods + overrides

            # Detect which checkpoint is it
            for name in quantization_methods:
                method = me_quant.get_quantization_config(name)
                quantization_override = method.override_quantization_method(
                    quant_cfg, self.quantization)
                if quantization_override is not None:
                    # Raise error if the override is not custom (custom would
                    # be in QUANTIZATION_METHODS but not QuantizationMethods)
                    # and hasn't been added to the overrides list.
                    if (name in get_args(me_quant.QuantizationMethods)
                            and name not in overrides):
                        raise ValueError(
                            f"Quantization method {name} is an override but "
                            "is has not been added to the `overrides` list "
                            "above. This is necessary to ensure that the "
                            "overrides are checked in order of preference.")
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
            from vllm.platforms import current_platform
            current_platform.verify_quantization(self.quantization)
            if self.quantization not in optimized_quantization_methods:
                logger.warning(



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
      but the source has SIMILAR if-blo
...<truncated>...
```

### Response

```
invariants:
- id: vllm_quantization_method_override_order_mismatch
  engine: vllm
  library: vllm
  invariant_under_test: ModelConfig._verify_quantization flags quantization override order mismatch
  severity: error
  native_type: vllm.ModelConfig
  miner_source:
    path: model_config.py
    method: _verify_quantization
    line_at_scan: 123
  match:
    engine: vllm
    fields:
      vllm.quantization:
        present: true
        not_equal: quant_method
  kwargs_positive:
    quantization: marlin
    quant_cfg: {quant_method: bitblas}
  kwargs_negative:
    quantization: fp8
    quant_cfg: {quant_method: fp8}
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: 'Quantization method override order mismatch'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: vllm_quantization_method_not_supported_on_platform
  engine: vllm
  library: vllm
  invariant_under_test: ModelConfig._verify_quantization flags quantization method not supported on platform
  severity: error
  native_type: vllm.ModelConfig
  miner_source:
    path: model_config.py
    method: _verify_quantization
    line_at_scan: 145
  match:
    engine: vllm
    fields:
      vllm.quantization:
        present: true
        not_in:
        - fp8
        - marlin
        - modelopt
        - gptq_marlin_24
        - gptq_marlin
        - awq_marlin
        - fbgemm_fp8
        - compressed-tensors
        - experts_int8
        - quark
        - modelopt_fp4
        - bitblas
        - gptq_bitblas
  kwargs_positive:
    quantization: ipex
  kwargs_negative:
    quantization: fp8
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: 'Unknown quantization method'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: vllm_quantization_method_mismatch_between_config_and_arg
  engine: vllm
  library: vllm
  invariant_under_test: ModelConfig._verify_quantization flags quantization method mismatch between config and arg
  severity: error
  native_type: vllm.ModelConfig
  miner_source:
    path: model_config.py
    method: _verify_quantization
    line_at_scan: 165
  match:
    engine: vllm
    fields:
      vllm.quantization:
        present: true
        not_equal: quant_method
  kwargs_positive:
    quantization: marlin
    quant_cfg: {quant_method: bitblas}
  kwargs_negative:
    quantization: fp8
    quant_cfg: {quant_method: fp8}
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: 'Quantization method specified in the model config does not match the quantization method specified in the `quantization` argument'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
```
