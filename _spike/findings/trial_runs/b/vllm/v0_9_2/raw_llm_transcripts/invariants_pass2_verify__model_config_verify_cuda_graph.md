# invariants_pass2_verify extraction transcript: model_config_verify_cuda_graph

- chunk_description: vllm.ModelConfig._verify_cuda_graph
- expected_namespaces: ['vllm']
- attempts: 1
- elapsed_sec: 27.10
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
- id: vllm_cuda_graph_capture_not_supported_on_rocm
  severity: warning
  match:
    engine: vllm
    fields:
      vllm.model_type:
        present: true
        not_in:
        - mllama
      vllm.is_encoder_decoder: false
      vllm.enforce_eager: false
  invariant_under_test: ModelConfig._verify_cuda_graph flags CUDA graph capture not
    supported on ROCm


INPUT 2 - THE SOURCE PASS 1 READ:

=== CONTEXT ===
vllm.ModelConfig._verify_cuda_graph - emit one invariant per `if <cond>: raise ...` block. Use namespace `vllm` for engine_params fields.

=== SOURCE: ModelConfig._verify_cuda_graph ===
    def _verify_cuda_graph(self) -> None:
        self.max_seq_len_to_capture = min(self.max_seq_len_to_capture,
                                          self.max_model_len)
        # CUDAGraph capture not supported for enc-dec models and mllama on ROCm
        ROCM_UNSUPPORTED_MODELS = ['mllama']
        unsupported_rocm = (self.hf_config.model_type
                            in ROCM_UNSUPPORTED_MODELS
                            or self.is_encoder_decoder)

        if (unsupported_rocm and not self.enforce_eager
                and current_platform.is_rocm()):
            logger.warning(
                "CUDA graph is not supported for %s on ROCm yet, fallback "
                "to eager mode.", self.hf_config.model_type)
            self.enforce_eager = True




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
- vllm_cuda_graph_capture_not_supported_on_rocm
```
