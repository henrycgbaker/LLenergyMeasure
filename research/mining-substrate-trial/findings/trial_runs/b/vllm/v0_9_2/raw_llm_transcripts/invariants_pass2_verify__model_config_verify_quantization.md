# invariants_pass2_verify extraction transcript: model_config_verify_quantization

- chunk_description: vllm.ModelConfig._verify_quantization
- expected_namespaces: ['vllm']
- attempts: 1
- elapsed_sec: 19.45
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser doing PASS 2 (verification) of multi-pass
invariant extraction. PASS 1 already extracted invariants from
vllm v0.9.2 for ONE chunk of source. Your job is to
REVIEW each emitted invariant against the source and flag any that
look WRONG.

INPUT 1 - PASS 1 EXTRACTED INVARIANTS (the candidate list):

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


INPUT 2 - THE SOURCE PASS 1 READ:

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



OUTPUT FORMAT: a YAML document with TWO sections:

confirmed:
- <id-of-pass1-invariant>  # one ID per line, no further detail needed.

flagged:
- id: <id-of-pass1-invariant>
  reason: <one-line: what looks wrong>
  fix: <one of: "drop", "correct_severity:<new-severity>", "correct_predicate:<new-kind>", "correct_kwargs_positive">

RULES:
1. Every PASS 1 invariant MUST appear in EITHER `confirmed` OR `flagged`
   (not both, not neither). If you're unsure, place in `confirmed` (the
   bar for flagging is "obviously wrong against the source").
2. Flag reasons must be CONCRETE - cite the source line that contradicts
   the invariant, or note the specific shape mismatch.
3. NO markdown code fences. NO commentary outside the YAML.
4. First character must be `c` (from `confirmed:`).

CRITERIA FOR FLAGGING:
- Severity wrong: source has `raise ValueError` but invariant says
  `severity: dormant`; or source has `minor_issues[...] = ...` but
  invariant says `severity: error`.
- Predicate wrong: source has `if X.field not in {a, b}: raise`
  but invariant emits `predicate_kind: exact`.
- Kwargs_positive wrong: source's predicate is "value < 0 raises" but
  invariant's kwargs_positive shows `field: 1` (which would NOT trigger).
- Hallucinated: invariant references a field name that does NOT appear
  in the source.

Emit the YAML now:

```

### Response

```
confirmed:
- vllm_quantization_method_not_in_allowlist
- vllm_quantization_method_mismatch

flagged:
- id: vllm_quantization_method_override_not_in_allowlist
  reason: Source checks for overrides in a specific order, but invariant does not reflect this.
  fix: correct_predicate:not_equal
```
