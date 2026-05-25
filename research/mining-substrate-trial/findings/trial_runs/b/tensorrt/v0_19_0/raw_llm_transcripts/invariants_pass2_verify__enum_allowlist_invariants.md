# invariants_pass2_verify extraction transcript: enum_allowlist_invariants

- chunk_description: tensorrt_llm StrEnum allowlists as Pydantic-enforced invariants
- expected_namespaces: ['tensorrt_llm']
- attempts: 1
- elapsed_sec: 37.47
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser doing PASS 2 (verification) of multi-pass
invariant extraction. PASS 1 already extracted invariants from
tensorrt v0.19.0 for ONE chunk of source. Your job is to
REVIEW each emitted invariant against the source and flag any that
look WRONG.

INPUT 1 - PASS 1 EXTRACTED INVARIANTS (the candidate list):

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


INPUT 2 - THE SOURCE PASS 1 READ:

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
- tensorrt_batching_type_not_in_allowlist
- tensorrt_capacity_scheduler_policy_not_in_allowlist
- tensorrt_context_chunking_policy_not_in_allowlist
```
