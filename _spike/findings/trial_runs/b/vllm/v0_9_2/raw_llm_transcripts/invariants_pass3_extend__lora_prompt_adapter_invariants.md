# invariants_pass3_extend extraction transcript: lora_prompt_adapter_invariants

- chunk_description: vllm.{LoRAConfig, PromptAdapterConfig, TokenizerPoolConfig}.__post_init__
- expected_namespaces: ['vllm']
- attempts: 1
- elapsed_sec: 21.94
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
- id: vllm_max_lora_rank_not_in_allowlist
  severity: error
  match:
    engine: vllm
    fields:
      vllm.max_lora_rank:
        present: true
        not_in:
        - 8
        - 16
        - 32
        - 64
        - 128
        - 256
        - 320
        - 512
  invariant_under_test: LoRAConfig flags max_lora_rank not in allowlist
- id: vllm_lora_extra_vocab_size_not_in_allowlist
  severity: error
  match:
    engine: vllm
    fields:
      vllm.lora_extra_vocab_size:
        present: true
        not_in:
        - 256
        - 512
  invariant_under_test: LoRAConfig flags lora_extra_vocab_size not in allowlist
- id: vllm_max_loras_lt_1
  severity: error
  match:
    engine: vllm
    fields:
      vllm.max_loras:
        <: 1
  invariant_under_test: LoRAConfig flags max_loras < 1
- id: vllm_max_cpu_loras_lt_max_loras
  severity: error
  match:
    engine: vllm
    fields:
      vllm.max_cpu_loras:
        <: 1
  invariant_under_test: LoRAConfig flags max_cpu_loras < max_loras
- id: vllm_max_prompt_adapters_lt_1
  severity: error
  match:
    engine: vllm
    fields:
      vllm.max_prompt_adapters:
        <: 1
  invariant_under_test: PromptAdapterConfig flags max_prompt_adapters < 1
- id: vllm_max_prompt_adapter_token_eq_0
  severity: error
  match:
    engine: vllm
    fields:
      vllm.max_prompt_adapter_token:
        present: true
        not_equal: 0
  invariant_under_test: PromptAdapterConfig flags max_prompt_adapter_token == 0


INPUT 2 - PASS 2 FLAGGED ISSUES (for context; consider whether they hint
at related misses):

flagged:
- id: vllm_max_cpu_loras_lt_max_loras
  reason: Source checks for `max_cpu_loras < self.max_loras`, but invariant only checks
    for `< 1`.
  fix: correct_predicate:not_less_than


INPUT 3 - THE SOURCE PASS 1 READ:

=== CONTEXT ===
Three small dataclasses share namespace `vllm`. Each `__post_init__` emits ValueError on bad shapes (max_loras<1, max_cpu_loras<max_loras, max_prompt_adapters<1, max_prompt_adapter_token==0, pool_type not in {ray}, extra_config not a dict).

=== SOURCE: LoRAConfig.__post_init__ ===
    def __post_init__(self):
        # Setting the maximum rank to 512 should be able to satisfy the vast
        # majority of applications.
        possible_max_ranks = (8, 16, 32, 64, 128, 256, 320, 512)
        possible_lora_extra_vocab_size = (256, 512)
        if self.max_lora_rank not in possible_max_ranks:
            raise ValueError(
                f"max_lora_rank ({self.max_lora_rank}) must be one of "
                f"{possible_max_ranks}.")
        if self.lora_extra_vocab_size not in possible_lora_extra_vocab_size:
            raise ValueError(
                f"lora_extra_vocab_size ({self.lora_extra_vocab_size}) "
                f"must be one of {possible_lora_extra_vocab_size}.")
        if self.max_loras < 1:
            raise ValueError(f"max_loras ({self.max_loras}) must be >= 1.")
        if self.max_cpu_loras is None:
            self.max_cpu_loras = self.max_loras
        elif self.max_cpu_loras < self.max_loras:
            raise ValueError(
                f"max_cpu_loras ({self.max_cpu_loras}) must be >= "
                f"max_loras ({self.max_loras})")


=== SOURCE: PromptAdapterConfig.__post_init__ ===
    def __post_init__(self):

        if self.max_prompt_adapters < 1:
            raise ValueError(f"max_prompt_adapters "
                             f"({self.max_prompt_adapters}) must be >= 1.")
        if self.max_prompt_adapter_token == 0:
            raise ValueError("max_prompt_adapter_token must be set.")
        if self.max_cpu_prompt_adapters is None:
            self.max_cpu_prompt_adapters = self.max_prompt_adapters


=== SOURCE: TokenizerPoolConfig.__post_init__ ===
    def __post_init__(self) -> None:
        logger.warning_once(
            "TokenizerPoolConfig is deprecated and will be removed in a "
            "future release. Passing this parameter will have no effect. "
            "Please remove it from your configurations.")





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
- id: vllm_max_cpu_loras_not_lt_max_loras
  engine: vllm
  library: vllm
  invariant_under_test: LoRAConfig flags max_cpu_loras < max_loras
  severity: error
  native_type: vllm.LoRAConfig
  miner_source:
    path: LoRAConfig.__post_init__
    method: __post_init__
    line_at_scan: 15
  match:
    engine: vllm
    fields:
      vllm.max_cpu_loras:
        <: vllm.max_loras
  kwargs_positive:
    max_cpu_loras: 1
    max_loras: 2
  kwargs_negative:
    max_cpu_loras: 2
    max_loras: 1
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: 'max_cpu_loras ({self.max_cpu_loras}) must be >= max_loras ({self.max_loras}).'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
```
