# invariants_pass3_extend extraction transcript: model_config_verify_quantization

- chunk_description: vllm.ModelConfig._verify_quantization
- expected_namespaces: ['vllm']
- attempts: 1
- elapsed_sec: 40.27
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser doing PASS 3 (extension) of multi-pass
invariant extraction. PASS 1 already extracted N invariants from
vllm v0.6.6.post1 for ONE chunk of source. Your job is to
identify ADDITIONAL invariants PASS 1 missed.

INPUT 1 - PASS 1 EXTRACTED INVARIANTS (already known; do NOT re-emit):

invariants:
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
- id: vllm_quantization_not_in_allowlist
  severity: error
  match:
    engine: vllm
    fields:
      vllm.quantization:
        present: true
        not_in: supported_quantization
  invariant_under_test: ModelConfig._verify_quantization flags quantization not in
    allowlist


INPUT 2 - PASS 2 FLAGGED ISSUES (for context; consider whether they hint
at related misses):

(no flags)

INPUT 3 - THE SOURCE PASS 1 READ:

=== CONTEXT ===
vllm.ModelConfig._verify_quantization - emit one invariant per `if <cond>: raise ...` block. Use namespace `vllm` for engine_params fields.

=== SOURCE: ModelConfig._verify_quantization ===
    def _verify_quantization(self) -> None:
        supported_quantization = QUANTIZATION_METHODS
        optimized_quantization_methods = [
            "fp8", "marlin", "modelopt", "gptq_marlin_24", "gptq_marlin",
            "awq_marlin", "fbgemm_fp8", "compressed_tensors",
            "compressed-tensors", "experts_int8"
        ]
        if self.quantization is not None:
            self.quantization = self.quantization.lower()

        # Parse quantization method from the HF model config, if available.
        quant_cfg = self._parse_quant_hf_config()

        if quant_cfg is not None:
            quant_method = quant_cfg.get("quant_method", "").lower()

            # Detect which checkpoint is it
            for name in QUANTIZATION_METHODS:
                method = get_quantization_config(name)
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
            current_platform.verify_quantization(self.quantization)
            if self.quantization not in optimized_quantization_methods:
                logger.warning(
                    "%s quantization is not fully "
                    "optimized yet. The speed can be slower than "
                    "non-quantized models.", self.quantization)




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
      missed.
   b. Multi-clause `if A and B and C:` invariants where PASS 1 keyed on
      only A. Emit one invariant per (A, B, C) tuple if each is independent.
   c. Type-check invariants `if not isinstance(x, T): raise` that PASS 1
      may have skipped (these often follow a similar pattern repeated
      per field).
   d. Less-common predicates: `<=`, `>=`, `not_equal`, `present`. PASS 1
      may have collapsed these to `exact`.
4. If PASS 1 covered every visible invariant, emit `invariants: []` (empty
   list) - do NOT fabricate.
5. Use snake_case_with_engine_prefix for `id`.
6. Set `added_by: llm_miner_pass3` (NOT `llm_miner`) so the multi-pass
   bookkeeping can track pass-3 contributions.

Emit the YAML now:

```

### Response

```
invariants:
- id: vllm_quantization_not_optimized
  engine: vllm
  library: vllm
  invariant_under_test: ModelConfig._verify_quantization checks for optimized quantization methods
  severity: warning
  native_type: vllm.ModelConfig
  miner_source:
    path: model_config.py
    method: _verify_quantization
    line_at_scan: 30
  match:
    engine: vllm
    fields:
      vllm.quantization:
        present: true
        not_in: optimized_quantization_methods
  kwargs_positive:
    quantization: awq_marlin
  kwargs_negative:
    quantization: fp8
  expected_outcome:
    outcome: warning
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: '%s quantization is not fully optimized yet. The speed can be slower than non-quantized models.'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
- id: vllm_quantization_mismatch_in_hf_config
  engine: vllm
  library: vllm
  invariant_under_test: ModelConfig._verify_quantization checks for quantization method mismatch in HF config
  severity: error
  native_type: vllm.ModelConfig
  miner_source:
    path: model_config.py
    method: _verify_quantization
    line_at_scan: 20
  match:
    engine: vllm
    fields:
      vllm.quantization:
        present: true
        not_equal: quant_method
  kwargs_positive:
    quantization: fp8
    quant_cfg: {quant_method: marlin}
  kwargs_negative:
    quantization: fp8
    quant_cfg: {quant_method: fp8}
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: 'Quantization method specified in the model config (%s) does not match the quantization method specified in the `quantization` argument (%s).'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
```
