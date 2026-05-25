# invariants_pass2_verify extraction transcript: trt_llm_args_validators

- chunk_description: tensorrt_llm.TrtLlmArgs @field_validator + @model_validator methods
- expected_namespaces: ['tensorrt_llm']
- attempts: 1
- elapsed_sec: 10.18
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser doing PASS 2 (verification) of multi-pass
invariant extraction. PASS 1 already extracted invariants from
tensorrt v0.20.0 for ONE chunk of source. Your job is to
REVIEW each emitted invariant against the source and flag any that
look WRONG.

INPUT 1 - PASS 1 EXTRACTED INVARIANTS (the candidate list):

invariants:
- id: tensorrt_calib_batch_size_not_in_allowlist
  severity: error
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.calib_batch_size:
        present: true
        not_in:
        - 1
        - 2
        - 4
        - 8
  invariant_under_test: TrtLlmArgs.validate flags calib_batch_size not in allowlist
- id: tensorrt_calib_max_batch_size_not_in_allowlist
  severity: error
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.calib_max_batch_size:
        present: true
        not_in:
        - 1
        - 2
        - 4
        - 8
  invariant_under_test: TrtLlmArgs.validate flags calib_max_batch_size not in allowlist
- id: tensorrt_calib_min_batch_size_not_in_allowlist
  severity: error
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.calib_min_batch_size:
        present: true
        not_in:
        - 1
        - 2
        - 4
        - 8
  invariant_under_test: TrtLlmArgs.validate flags calib_min_batch_size not in allowlist
- id: tensorrt_calib_batch_size_lt_calib_min_batch_size
  severity: error
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.calib_batch_size:
        <: 1
  invariant_under_test: TrtLlmArgs.validate flags calib_batch_size < calib_min_batch_size


INPUT 2 - THE SOURCE PASS 1 READ:

=== CONTEXT ===
TrtLlmArgs has 1 @field_validator (calib_config init) + 3 @model_validator decorators. The validate_enable_build_cache method has the most pertinent `raise ValueError(...)` block.

=== SOURCE: TrtLlmArgs validators ===




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
- tensorrt_calib_batch_size_not_in_allowlist
- tensorrt_calib_max_batch_size_not_in_allowlist
- tensorrt_calib_min_batch_size_not_in_allowlist

flagged:
- id: tensorrt_calib_batch_size_lt_calib_min_batch_size
  reason: Source does not contain a comparison between calib_batch_size and calib_min_batch_size.
  fix: drop
```
