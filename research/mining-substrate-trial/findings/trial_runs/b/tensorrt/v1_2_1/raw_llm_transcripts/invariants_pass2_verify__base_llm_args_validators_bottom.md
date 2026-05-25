# invariants_pass2_verify extraction transcript: base_llm_args_validators_bottom

- chunk_description: tensorrt_llm.BaseLlmArgs remaining @model_validators
- expected_namespaces: ['tensorrt_llm']
- attempts: 1
- elapsed_sec: 16.78
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser doing PASS 2 (verification) of multi-pass
invariant extraction. PASS 1 already extracted invariants from
tensorrt v1.2.1 for ONE chunk of source. Your job is to
REVIEW each emitted invariant against the source and flag any that
look WRONG.

INPUT 1 - PASS 1 EXTRACTED INVARIANTS (the candidate list):

invariants:
- id: tensorrt_max_batch_size_gt_max_num_tokens
  severity: warning
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.max_batch_size:
        '>': 1
  invariant_under_test: BaseLlmArgs.validate flags max_batch_size > max_num_tokens
- id: tensorrt_lora_dir_empty
  severity: warning
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.lora_dir:
        present: true
        not_equal: ''
  invariant_under_test: BaseLlmArgs.validate flags lora_dir empty
- id: tensorrt_enable_lora_ignored_with_lora_config
  severity: warning
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.enable_lora: true
      tensorrt_llm.lora_config:
        present: true
      tensorrt_llm.backend:
        not_in:
        - pytorch
        - _autodeploy
  invariant_under_test: BaseLlmArgs.validate flags enable_lora ignored with lora_config
- id: tensorrt_both_lora_dir_and_lora_target_modules_empty
  severity: warning
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.lora_dir:
        present: true
        not_equal: ''
      tensorrt_llm.lora_target_modules:
        present: true
        not_equal: []
  invariant_under_test: BaseLlmArgs.validate flags both lora_dir and lora_target_modules
    empty
- id: tensorrt_lora_prefetch_dir_set_while_not_supported
  severity: error
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.peft_cache_config.lora_prefetch_dir:
        present: true
        not_equal: ''
  invariant_under_test: BaseLlmArgs.validate flags lora_prefetch_dir set while not
    supported


INPUT 2 - THE SOURCE PASS 1 READ:

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
    def validate_lora_config_consistency(self):
        if self.lora_config:
            if len(self.lora_config.lora_dir) == 0:
                # TODO [TRTLLM-5173]
                logger.warning(
                    "lora_dir is empty, so custom embedding or lm head will not be applied."
                )

        if self.enable_lora and self.lora_config is not None and self.backend in [
                'pytorch', '_autodeploy'
        ]:
            logger.warning(
                f"enable_lora is ignored when lora_config is provided for {self.backend} backend."
            )

        if self.lora_config is not None:
            if len(self.lora_config.lora_dir) == 0 and len(
                    self.lora_config.lora_target_modules) == 0:
                logger.warning(
                    "Both lora_dir and lora_target_modules are empty, so all LoRA modules will be expected. "
                    "This will lead to serious memory consumption. Please provide either lora_dir or lora_target_modules if this behavior is not what you expect."
                )
                default_trtllm_modules_to_hf_modules = get_default_trtllm_modules_to_hf_modules(
                )
                self.lora_config.lora_target_modules = list(
                    default_trtllm_modules_to_hf_modules.keys())
        return self

    @model_validator(mode="after")
    def validate_peft_cache_config(self):
        if self.peft_cache_config is not None and self.peft_cache_config.lora_prefetch_dir is not None:
            raise ValueError(
                f"lora_prefetch_dir was set to '{self.peft_cache_config.lora_prefetch_dir}' "
                "while LoRA prefetch is not supported")
        return self




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
- tensorrt_max_batch_size_gt_max_num_tokens
- tensorrt_lora_dir_empty
- tensorrt_enable_lora_ignored_with_lora_config
- tensorrt_both_lora_dir_and_lora_target_modules_empty

flagged:
- id: tensorrt_lora_prefetch_dir_set_while_not_supported
  reason: Source raises ValueError, but invariant has severity warning.
  fix: correct_severity:error
```
