# invariants_pass3_extend extraction transcript: lookahead_validator

- chunk_description: tensorrt_llm.LookaheadDecodingConfig @field_validator (positive values)
- expected_namespaces: ['tensorrt_llm']
- attempts: 1
- elapsed_sec: 30.66
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser doing PASS 3 (extension) of multi-pass
invariant extraction. PASS 1 already extracted N invariants from
tensorrt v0.21.0 for ONE chunk of source. Your job is to
identify ADDITIONAL invariants PASS 1 missed.

INPUT 1 - PASS 1 EXTRACTED INVARIANTS (already known; do NOT re-emit):

invariants:
- id: tensorrt_llm_max_window_size_le_zero
  severity: error
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.max_window_size:
        <=: 0
  invariant_under_test: LookaheadDecodingConfig.validate flags max_window_size <=
    0
- id: tensorrt_llm_max_ngram_size_le_zero
  severity: error
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.max_ngram_size:
        <=: 0
  invariant_under_test: LookaheadDecodingConfig.validate flags max_ngram_size <= 0
- id: tensorrt_llm_max_verification_set_size_le_zero
  severity: error
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.max_verification_set_size:
        <=: 0
  invariant_under_test: LookaheadDecodingConfig.validate flags max_verification_set_size
    <= 0


INPUT 2 - PASS 2 FLAGGED ISSUES (for context; consider whether they hint
at related misses):

(no flags)

INPUT 3 - THE SOURCE PASS 1 READ:

=== CONTEXT ===
LookaheadDecodingConfig has ONE @field_validator decorator applied to THREE fields (max_window_size, max_ngram_size, max_verification_set_size). This MUST emit THREE separate invariants - one per field. Each invariant has predicate `<= 0` (i.e. `<= 0` triggers ValueError). Use namespace `tensorrt_llm`.

=== SOURCE: LookaheadDecodingConfig ===
class LookaheadDecodingConfig(DecodingBaseConfig, PybindMirror):
    """
    Configuration for lookahead speculative decoding.
    """

    max_window_size: int = Field(
        default=_LookaheadDecodingConfig.get_default_lookahead_decoding_window(
        ),
        description="Number of NGrams in lookahead branch per step.")
    max_ngram_size: int = Field(
        default=_LookaheadDecodingConfig.get_default_lookahead_decoding_ngram(),
        description="Number of tokens per NGram.")
    max_verification_set_size: int = Field(
        default=_LookaheadDecodingConfig.
        get_default_lookahead_decoding_verification_set(),
        description="Number of NGrams in verification branch per step.")

    @field_validator('max_window_size', 'max_ngram_size',
                     'max_verification_set_size')
    @classmethod
    def validate_positive_values(cls, v):
        if v <= 0:
            raise ValueError(f"Value must be positive, got {v}")
        return v

    def __init__(self, **data):
        super().__init__(**data)
        self._check_fields()

    def calculate_speculative_resource(self):
        return _LookaheadDecodingConfig.calculate_speculative_resource_tuple(
            self.max_window_size, self.max_ngram_size,
            self.max_verification_set_size)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    def _to_pybind(self):
        return _LookaheadDecodingConfig(self.max_window_size,
                                        self.max_ngram_size,
                                        self.max_verification_set_size)

    decoding_type: ClassVar[str] = "Lookahead"





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
  - id: tensorrt_llm_max_window_size_positive
    engine: tensorrt
    library: tensorrt
    invariant_under_test: LookaheadDecodingConfig.validate flags max_window_size > 0
    severity: error
    native_type: tensorrt.LookaheadDecodingConfig
    miner_source:
      path: lookahead_decoding_config.py
      method: validate_positive_values
      line_at_scan: 15
    match:
      engine: tensorrt
      fields:
        tensorrt_llm.max_window_size:
          '>': 0
    kwargs_positive:
      max_window_size: 1
    kwargs_negative:
      max_window_size: -1
    expected_outcome:
      outcome: error
      emission_channel: logger_warning_once
      normalised_fields: []
    message_template: 'Value must be positive, got {v}'
    added_by: llm_miner_pass3
    added_at: '2026-05-25'
  - id: tensorrt_llm_max_ngram_size_positive
    engine: tensorrt
    library: tensorrt
    invariant_under_test: LookaheadDecodingConfig.validate flags max_ngram_size > 0
    severity: error
    native_type: tensorrt.LookaheadDecodingConfig
    miner_source:
      path: lookahead_decoding_config.py
      method: validate_positive_values
      line_at_scan: 15
    match:
      engine: tensorrt
      fields:
        tensorrt_llm.max_ngram_size:
          '>': 0
    kwargs_positive:
      max_ngram_size: 1
    kwargs_negative:
      max_ngram_size: -1
    expected_outcome:
      outcome: error
      emission_channel: logger_warning_once
      normalised_fields: []
    message_template: 'Value must be positive, got {v}'
    added_by: llm_miner_pass3
    added_at: '2026-05-25'
  - id: tensorrt_llm_max_verification_set_size_positive
    engine: tensorrt
    library: tensorrt
    invariant_under_test: LookaheadDecodingConfig.validate flags max_verification_set_size > 0
    severity: error
    native_type: tensorrt.LookaheadDecodingConfig
    miner_source:
      path: lookahead_decoding_config.py
      method: validate_positive_values
      line_at_scan: 15
    match:
      engine: tensorrt
      fields:
        tensorrt_llm.max_verification_set_size:
          '>': 0
    kwargs_positive:
      max_verification_set_size: 1
    kwargs_negative:
      max_verification_set_size: -1
    expected_outcome:
      outcome: error
      emission_channel: logger_warning_once
      normalised_fields: []
    message_template: 'Value must be positive, got {v}'
    added_by: llm_miner_pass3
    added_at: '2026-05-25'
```
