# invariants_pass3_extend extraction transcript: model_config_verify_bnb_config

- chunk_description: vllm.ModelConfig._verify_bnb_config
- expected_namespaces: ['vllm']
- attempts: 1
- elapsed_sec: 13.94
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser doing PASS 3 (extension) of multi-pass
invariant extraction. PASS 1 already extracted N invariants from
vllm v0.7.3 for ONE chunk of source. Your job is to
identify ADDITIONAL invariants PASS 1 missed.

INPUT 1 - PASS 1 EXTRACTED INVARIANTS (already known; do NOT re-emit):

invariants:
- id: vllm_cuda_graph_not_supported_on_bitsandbytes_8bit
  severity: warning
  match:
    engine: vllm
    fields:
      vllm.quantization: bitsandbytes
      vllm.hf_config.quantization_config.load_in_8bit: true
      vllm.enforce_eager: false
  invariant_under_test: ModelConfig._verify_bnb_config flags CUDA graph not supported
    on BitAndBytes 8-bit


INPUT 2 - PASS 2 FLAGGED ISSUES (for context; consider whether they hint
at related misses):

(no flags)

INPUT 3 - THE SOURCE PASS 1 READ:

=== CONTEXT ===
vllm.ModelConfig._verify_bnb_config - emit one invariant per `if <cond>: raise ...` block. Use namespace `vllm` for engine_params fields.

=== SOURCE: ModelConfig._verify_bnb_config ===
    def _verify_bnb_config(self) -> None:
        """
        The current version of bitsandbytes (0.44.0) with 8-bit models does not
        yet support CUDA graph.
        """
        is_bitsandbytes = self.quantization == "bitsandbytes"
        has_quantization_config = (getattr(self.hf_config,
                                           "quantization_config", None)
                                   is not None)
        is_8bit = (self.hf_config.quantization_config.get(
            "load_in_8bit", False) if has_quantization_config else False)
        if all([
                is_bitsandbytes,
                has_quantization_config,
                is_8bit,
                not self.enforce_eager,
        ]):
            logger.warning(
                "CUDA graph is not supported on BitAndBytes 8bit yet, "
                "fallback to the eager mode.")
            self.enforce_eager = True




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
  - id: vllm_cuda_graph_not_supported_on_bitsandbytes_8bit_enforce_eager_set
    engine: vllm
    library: vllm
    invariant_under_test: ModelConfig._verify_bnb_config sets enforce_eager to True when CUDA graph is not supported on BitAndBytes 8-bit
    severity: warning
    native_type: vllm.ModelConfig
    miner_source:
      path: model_config.py
      method: _verify_bnb_config
      line_at_scan: 123
    match:
      engine: vllm
      fields:
        vllm.quantization: bitsandbytes
        vllm.hf_config.quantization_config.load_in_8bit: true
        vllm.enforce_eager: false
    kwargs_positive:
      enforce_eager: false
    kwargs_negative:
      enforce_eager: true
    expected_outcome:
      outcome: warning
      emission_channel: logger_warning_once
      normalised_fields: []
    message_template: 'CUDA graph is not supported on BitAndBytes 8bit yet, fallback to the eager mode.'
    added_by: llm_miner_pass3
    added_at: '2026-05-25'
```
