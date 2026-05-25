# invariants_pass2_verify extraction transcript: base_llm_args_validators_top

- chunk_description: tensorrt_llm.BaseLlmArgs @field_validator + first @model_validators
- expected_namespaces: ['tensorrt_llm']
- attempts: 1
- elapsed_sec: 98.56
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
- id: tensorrt_llm_model_type_is_not
  severity: error
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.model:
        present: true
        type_is_not:
        - str
  invariant_under_test: BaseLlmArgs.validate flags model type mismatch
- id: tensorrt_llm_model_not_in_allowlist
  severity: error
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.model:
        present: true
        not_in:
        - bert-base-uncased
        - roberta-base
  invariant_under_test: BaseLlmArgs.validate flags model not in allowlist
- id: tensorrt_llm_num_beams_lt_1
  severity: error
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.num_beams:
        <: 1
  invariant_under_test: BaseLlmArgs.validate flags num_beams < 1
- id: tensorrt_llm_num_return_sequences_gt_num_beams
  severity: error
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.num_return_sequences:
        '>': 1
  invariant_under_test: BaseLlmArgs.validate flags num_return_sequences > num_beams
- id: tensorrt_llm_no_repeat_ngram_size_lt_0
  severity: error
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.no_repeat_ngram_size:
        <: 0
  invariant_under_test: BaseLlmArgs.validate flags no_repeat_ngram_size < 0
- id: tensorrt_llm_temperature_lt_0
  severity: error
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.temperature:
        <: 0
  invariant_under_test: BaseLlmArgs.validate flags temperature < 0
- id: tensorrt_llm_top_k_lt_1
  severity: error
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.top_k:
        <: 1
  invariant_under_test: BaseLlmArgs.validate flags top_k < 1
- id: tensorrt_llm_top_p_lt_0
  severity: error
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.top_p:
        <: 0
  invariant_under_test: BaseLlmArgs.validate flags top_p < 0
- id: tensorrt_llm_typical_p_lt_0
  severity: error
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.typical_p:
        <: 0
  invariant_under_test: BaseLlmArgs.validate flags typical_p < 0


INPUT 2 - THE SOURCE PASS 1 READ:

=== CONTEXT ===
tensorrt_llm uses Pydantic v2 validators (NOT `if X: raise` patterns). Each `@field_validator(field)` decorator + method is ONE invariant; each `@model_validator(mode='after')` decorator + method may contain multiple `raise ValueError` branches (each is its own invariant). Emit one invariant per `raise` statement OR per @field_validator method. Use namespace `tensorrt_llm`.

Examples of validator forms to extract:
- `@field_validator('model')\ndef validate_model(...):\n    if not isinstance(v, ...): raise ValueError(...)` ->   severity=error, predicate=type_is_not.
- `@model_validator(mode='after')\ndef validate_build_config_with_runtime_params(self):\n    if self.max_batch_size > self.build_config.max_batch_size: raise ValueError(...)` -> severity=error, cross-field check.

NOTE: this chunk shows the FIRST HALF of BaseLlmArgs validators; the rest are in a separate chunk.

=== SOURCE: BaseLlmArgs validators (top half) ===




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
- tensorrt_llm_model_type_is_not
- tensorrt_llm_num_beams_lt_1
- tensorrt_llm_temperature_lt_0
- tensorrt_llm_top_k_lt_1
- tensorrt_llm_top_p_lt_0
- tensorrt_llm_typical_p_lt_0

flagged:
- id: tensorrt_llm_model_not_in_allowlist
  reason: Source does not contain a validator for model allowlist.
  fix: drop
- id: tensorrt_llm_num_return_sequences_gt_num_beams
  reason: Source does not contain a validator for num_return_sequences > num_beams.
  fix: drop
- id: tensorrt_llm_no_repeat_ngram_size_lt_0
  reason: Source does not contain a validator for no_repeat_ngram_size < 0.
  fix: drop
```
