# invariants_pass3_extend extraction transcript: base_llm_args_validators_bottom

- chunk_description: tensorrt_llm.BaseLlmArgs remaining @model_validators
- expected_namespaces: ['tensorrt_llm']
- attempts: 1
- elapsed_sec: 94.71
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


INPUT 2 - PASS 2 FLAGGED ISSUES (for context; consider whether they hint
at related misses):

flagged:
- id: tensorrt_lora_prefetch_dir_set_while_not_supported
  reason: Source raises ValueError, but invariant has severity warning.
  fix: correct_severity:error


INPUT 3 - THE SOURCE PASS 1 READ:

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
    outcome: <error|dormant_announced|warning>
    emission_channel: <none|logger_warning_once|logger_warning>
    normalised_fields: []
  message_template: '<the exact error/warning string from source>'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

PREDICATE FORMS (use the EXACT keys shown):
- Exact value:         `tensorrt_llm.field: value`
- Not in list:         `tensorrt_llm.field: {present: true, not_in: [a, b]}`
- Greater than:        `tensorrt_llm.field: {'>': value}`
- Less than:           `tensorrt_llm.field: {'<': value}`
- Type not in:         `tensorrt_llm.field: {present: true, type_is_not: [TypeName]}`
- Presence:            `tensorrt_llm.field: {present: true}`

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
- id: tensorrt_max_batch_size_leq_max_num_tokens
  engine: tensorrt
  library: tensorrt
  invariant_under_test: BaseLlmArgs.validate flags max_batch_size <= max_num_tokens
  severity: warning
  native_type: tensorrt.GenerationConfig
  miner_source:
    path: BaseLlmArgs validators (bottom half)
    method: validate_runtime_args
    line_at_scan: 5
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.max_batch_size:
        '<=': tensorrt_llm.max_num_tokens
  kwargs_positive:
    max_batch_size: 10
    max_num_tokens: 10
  kwargs_negative:
    max_batch_size: 11
    max_num_tokens: 10
  expected_outcome:
    outcome: warning
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: 'max_batch_size [{self.max_batch_size}] should be less than or equal to max_num_tokens [{self.max_num_tokens}]'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
```
