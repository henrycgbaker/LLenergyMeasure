# invariants_pass3_extend extraction transcript: scheduler_config_invariants

- chunk_description: vllm.SchedulerConfig __post_init__ + _verify_args
- expected_namespaces: ['vllm']
- attempts: 1
- elapsed_sec: 145.92
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
- id: vllm_max_num_batched_tokens_lt_max_model_len
  severity: error
  match:
    engine: vllm
    fields:
      vllm.chunked_prefill_enabled: false
      vllm.max_num_batched_tokens:
        <: 1
  invariant_under_test: SchedulerConfig._verify_args flags max_num_batched_tokens
    < max_model_len when chunked_prefill_enabled=False
- id: vllm_max_num_batched_tokens_lt_max_num_seqs
  severity: error
  match:
    engine: vllm
    fields:
      vllm.max_num_batched_tokens:
        <: 1
  invariant_under_test: SchedulerConfig._verify_args flags max_num_batched_tokens
    < max_num_seqs
- id: vllm_num_lookahead_slots_lt_zero
  severity: error
  match:
    engine: vllm
    fields:
      vllm.num_lookahead_slots:
        <: 0
  invariant_under_test: SchedulerConfig._verify_args flags num_lookahead_slots < 0
- id: vllm_num_scheduler_steps_lt_one
  severity: error
  match:
    engine: vllm
    fields:
      vllm.num_scheduler_steps:
        <: 1
  invariant_under_test: SchedulerConfig._verify_args flags num_scheduler_steps < 1


INPUT 2 - PASS 2 FLAGGED ISSUES (for context; consider whether they hint
at related misses):

flagged:
- id: vllm_num_lookahead_slots_lt_zero
  reason: Source checks for `num_lookahead_slots < 0`, but invariant has `< 1` in
    the match.
  fix: correct_predicate:ge


INPUT 3 - THE SOURCE PASS 1 READ:

=== CONTEXT ===
vllm.SchedulerConfig validates `max_num_batched_tokens` vs `max_num_seqs` and the `policy` enum. Use namespace `vllm`.

=== SOURCE: SchedulerConfig.__post_init__ ===


=== SOURCE: SchedulerConfig._verify_args ===
    def _verify_args(self) -> None:
        if (self.max_num_batched_tokens < self.max_model_len
                and not self.chunked_prefill_enabled):
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) is "
                f"smaller than max_model_len ({self.max_model_len}). "
                "This effectively limits the maximum sequence length to "
                "max_num_batched_tokens and makes vLLM reject longer "
                "sequences. Please increase max_num_batched_tokens or "
                "decrease max_model_len.")

        if self.max_num_batched_tokens < self.max_num_seqs:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) must "
                "be greater than or equal to max_num_seqs "
                f"({self.max_num_seqs}).")

        if self.num_lookahead_slots < 0:
            raise ValueError(
                "num_lookahead_slots "
                f"({self.num_lookahead_slots}) must be greater than or "
                "equal to 0.")

        if self.num_scheduler_steps < 1:
            raise ValueError(
                "num_scheduler_steps "
                f"({self.num_scheduler_steps}) must be greater than or "
                "equal to 1.")




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
- id: vllm_max_model_len_gt_zero
  engine: vllm
  library: vllm
  invariant_under_test: SchedulerConfig._verify_args flags max_model_len > 0
  severity: error
  native_type: vllm.SchedulerConfig
  miner_source:
    path: SchedulerConfig.py
    method: _verify_args
    line_at_scan: 5
  match:
    engine: vllm
    fields:
      vllm.max_model_len:
        {'>': 0}
  kwargs_positive:
    max_model_len: 1
  kwargs_negative:
    max_model_len: 0
  expected_outcome:
    outcome: error
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: 'max_model_len must be greater than 0.'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
- id: vllm_max_num_seqs_gt_zero
  engine: vllm
  library: vllm
  invariant_under_test: SchedulerConfig._verify_args flags max_num_seqs > 0
  severity: error
  native_type: vllm.SchedulerConfig
  miner_source:
    path: SchedulerConfig.py
    method: _verify_args
    line_at_scan: 10
  match:
    engine: vllm
    fields:
      vllm.max_num_seqs:
        {'>': 0}
  kwargs_positive:
    max_num_seqs: 1
  kwargs_negative:
    max_num_seqs: 0
  expected_outcome:
    outcome: error
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: 'max_num_seqs must be greater than 0.'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
- id: vllm_chunked_prefill_enabled_type_check
  engine: vllm
  library: vllm
  invariant_under_test: SchedulerConfig._verify_args flags chunked_prefill_enabled is bool
  severity: error
  native_type: vllm.SchedulerConfig
  miner_source:
    path: SchedulerConfig.py
    method: _verify_args
    line_at_scan: 15
  match:
    engine: vllm
    fields:
      vllm.chunked_prefill_enabled:
        {type_is_not: [bool]}
  kwargs_positive:
    chunked_prefill_enabled: True
  kwargs_negative:
    chunked_prefill_enabled: 'not a bool'
  expected_outcome:
    outcome: error
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: 'chunked_prefill_enabled must be a boolean.'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
```
