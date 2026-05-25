# invariants_pass3_extend extraction transcript: enum_allowlist_invariants

- chunk_description: tensorrt_llm StrEnum allowlists as Pydantic-enforced invariants
- expected_namespaces: ['tensorrt_llm']
- attempts: 1
- elapsed_sec: 33.74
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser doing PASS 3 (extension) of multi-pass
invariant extraction. PASS 1 already extracted N invariants from
tensorrt v0.19.0 for ONE chunk of source. Your job is to
identify ADDITIONAL invariants PASS 1 missed.

INPUT 1 - PASS 1 EXTRACTED INVARIANTS (already known; do NOT re-emit):

invariants:
- id: tensorrt_batching_type_not_in_allowlist
  severity: error
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.batching_type:
        present: true
        not_in:
        - STATIC
        - INFLIGHT
  invariant_under_test: TrtLlmArgs.validate flags batching_type not in allowlist
- id: tensorrt_capacity_scheduler_policy_not_in_allowlist
  severity: error
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.capacity_scheduler_policy:
        present: true
        not_in:
        - MAX_UTILIZATION
        - GUARANTEED_NO_EVICT
        - STATIC_BATCH
  invariant_under_test: SchedulerConfig.validate flags capacity_scheduler_policy not
    in allowlist
- id: tensorrt_context_chunking_policy_not_in_allowlist
  severity: error
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.context_chunking_policy:
        present: true
        not_in:
        - FIRST_COME_FIRST_SERVED
        - EQUAL_PROGRESS
  invariant_under_test: SchedulerConfig.validate flags context_chunking_policy not
    in allowlist


INPUT 2 - PASS 2 FLAGGED ISSUES (for context; consider whether they hint
at related misses):

(no flags)

INPUT 3 - THE SOURCE PASS 1 READ:

=== CONTEXT ===
These StrEnum classes are the type for fields on TrtLlmArgs (batching_type uses BatchingType) and SchedulerConfig (capacity_scheduler_policy uses CapacitySchedulerPolicy, context_chunking_policy uses ContextChunkingPolicy). Pydantic rejects any value outside the enum. Emit ONE invariant per field: predicate=not_in, severity=error, namespace=tensorrt_llm.

=== SOURCE: BatchingType (field: batching_type on TrtLlmArgs) ===
class BatchingType(StrEnum, metaclass=PybindMirrorEnumMeta):
    STATIC = "STATIC"
    INFLIGHT = "INFLIGHT"

    def _to_pybind(self):
        return getattr(_BatchingType, self.value)



=== SOURCE: CapacitySchedulerPolicy (field: capacity_scheduler_policy on SchedulerConfig) ===
class CapacitySchedulerPolicy(StrEnum, metaclass=PybindMirrorEnumMeta):
    MAX_UTILIZATION = "MAX_UTILIZATION"
    GUARANTEED_NO_EVICT = "GUARANTEED_NO_EVICT"
    STATIC_BATCH = "STATIC_BATCH"

    def _to_pybind(self):
        return getattr(_CapacitySchedulerPolicy, self.value)



=== SOURCE: ContextChunkingPolicy (field: context_chunking_policy on SchedulerConfig) ===
class ContextChunkingPolicy(StrEnum, metaclass=PybindMirrorEnumMeta):
    ''' Context chunking policy. '''
    FIRST_COME_FIRST_SERVED = "FIRST_COME_FIRST_SERVED"
    EQUAL_PROGRESS = "EQUAL_PROGRESS"

    def _to_pybind(self):
        return getattr(_ContextChunkingPolicy, self.value)





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
  - id: tensorrt_batching_type_not_static
    engine: tensorrt
    library: tensorrt
    invariant_under_test: TrtLlmArgs.validate flags batching_type not static
    severity: error
    native_type: tensorrt.GenerationConfig
    miner_source:
      path: trt_llm_args.py
      method: validate
      line_at_scan: 123
    match:
      engine: tensorrt
      fields:
        tensorrt_llm.batching_type:
          present: true
          not_in:
            - STATIC
    kwargs_positive:
      batching_type: INFLIGHT
    kwargs_negative:
      batching_type: STATIC
    expected_outcome:
      outcome: error
      emission_channel: logger_warning_once
      normalised_fields: []
    message_template: 'Batching type must be static'
    added_by: llm_miner_pass3
    added_at: '2026-05-25'

  - id: tensorrt_capacity_scheduler_policy_not_max_utilization
    engine: tensorrt
    library: tensorrt
    invariant_under_test: SchedulerConfig.validate flags capacity_scheduler_policy not max utilization
    severity: error
    native_type: tensorrt.EngineArgs
    miner_source:
      path: scheduler_config.py
      method: validate
      line_at_scan: 456
    match:
      engine: tensorrt
      fields:
        tensorrt_llm.capacity_scheduler_policy:
          present: true
          not_in:
            - MAX_UTILIZATION
    kwargs_positive:
      capacity_scheduler_policy: GUARANTEED_NO_EVICT
    kwargs_negative:
      capacity_scheduler_policy: MAX_UTILIZATION
    expected_outcome:
      outcome: error
      emission_channel: logger_warning_once
      normalised_fields: []
    message_template: 'Capacity scheduler policy must be max utilization'
    added_by: llm_miner_pass3
    added_at: '2026-05-25'

  - id: tensorrt_context_chunking_policy_not_first_come_first_served
    engine: tensorrt
    library: tensorrt
    invariant_under_test: SchedulerConfig.validate flags context_chunking_policy not first come first served
    severity: error
    native_type: tensorrt.EngineArgs
    miner_source:
      path: scheduler_config.py
      method: validate
      line_at_scan: 789
    match:
      engine: tensorrt
      fields:
        tensorrt_llm.context_chunking_policy:
          present: true
          not_in:
            - FIRST_COME_FIRST_SERVED
    kwargs_positive:
      context_chunking_policy: EQUAL_PROGRESS
    kwargs_negative:
      context_chunking_policy: FIRST_COME_FIRST_SERVED
    expected_outcome:
      outcome: error
      emission_channel: logger_warning_once
      normalised_fields: []
    message_template: 'Context chunking policy must be first come first served'
    added_by: llm_miner_pass3
    added_at: '2026-05-25'
```
