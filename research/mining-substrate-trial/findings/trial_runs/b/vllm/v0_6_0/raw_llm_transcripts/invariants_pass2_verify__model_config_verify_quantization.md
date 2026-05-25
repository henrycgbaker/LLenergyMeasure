# invariants_pass2_verify extraction transcript: model_config_verify_quantization

- chunk_description: vllm.ModelConfig._verify_quantization
- expected_namespaces: ['vllm']
- attempts: 1
- elapsed_sec: 83.95
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser doing PASS 2 (verification) of multi-pass
invariant extraction. PASS 1 already extracted invariants from
vllm v0.6.0 for ONE chunk of source. Your job is to
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
- id: vllm_awq_quantization_rocm_not_enabled
  severity: warning
  match:
    engine: vllm
    fields:
      vllm.quantization: awq
  invariant_under_test: ModelConfig._verify_quantization flags AWQ quantization ROCm
    not enabled


INPUT 2 - THE SOURCE PASS 1 READ:

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
- vllm_rocm_quantization_method_not_in_allowlist
- vllm_tpu_quantization_method_not_in_allowlist
- vllm_neuron_quantization_method_not_in_allowlist

flagged:
- id: vllm_awq_quantization_not_optimized
  reason: Source raises a warning for non-optimized quantization methods, but the invariant does not match the exact list of optimized methods.
  fix: correct_predicate:not_in
- id: vllm_awq_quantization_rocm_not_enabled
  reason: Source sets an environment variable when AWQ is used with ROCm and VLLM_USE_TRITON_AWQ is not set, but the invariant does not capture this behavior.
  fix: drop
```
