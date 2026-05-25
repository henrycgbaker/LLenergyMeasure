# invariants_pass3_extend extraction transcript: calib_config_literal

- chunk_description: tensorrt_llm.CalibConfig Literal['cuda', 'cpu'] (Pydantic-enforced enum)
- expected_namespaces: ['tensorrt_llm']
- attempts: 1
- elapsed_sec: 112.19
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser doing PASS 3 (extension) of multi-pass
invariant extraction. PASS 1 already extracted N invariants from
tensorrt v0.20.0 for ONE chunk of source. Your job is to
identify ADDITIONAL invariants PASS 1 missed.

INPUT 1 - PASS 1 EXTRACTED INVARIANTS (already known; do NOT re-emit):

invariants:
- id: tensorrt_device_not_in_allowlist
  severity: error
  match:
    engine: tensorrt
    fields:
      tensorrt_llm.device:
        present: true
        not_in:
        - cuda
        - cpu
  invariant_under_test: CalibConfig construction flags device not in allowlist


INPUT 2 - PASS 2 FLAGGED ISSUES (for context; consider whether they hint
at related misses):

(no flags)

INPUT 3 - THE SOURCE PASS 1 READ:

=== CONTEXT ===
CalibConfig.device is typed `Literal['cuda', 'cpu']`. Pydantic enforces this at construction time - any value other than 'cuda' or 'cpu' raises ValidationError. Emit this as ONE invariant with predicate=not_in, severity=error.

=== SOURCE: CalibConfig ===
class CalibConfig(BaseModel):
    """
    Calibration configuration.
    """
    device: Literal['cuda',
                    'cpu'] = Field(default='cuda',
                                   description="The device to run calibration.")
    calib_dataset: str = Field(
        default='cnn_dailymail',
        description="The name or local path of calibration dataset.")
    calib_batches: int = Field(
        default=512,
        description="The number of batches that the calibration runs.")
    calib_batch_size: int = Field(
        default=1, description="The batch size that the calibration runs.")
    calib_max_seq_length: int = Field(
        default=512,
        description="The maximum sequence length that the calibration runs.")
    random_seed: int = Field(
        default=1234, description="The random seed used for calibration.")
    tokenizer_max_seq_length: int = Field(
        default=2048,
        description=
        "The maximum sequence length to initialize tokenizer for calibration.")

    @classmethod
    def from_dict(cls, config: dict) -> 'CalibConfig':
        """Create a CalibConfig instance from a dict.

        Args:
            config (dict): The dict used to create CalibConfig.

        Returns:
            tensorrt_llm.llmapi.CalibConfig: The CalibConfig created from dict.
        """
        return cls(**config)

    def to_dict(self) -> dict:
        """Dump a CalibConfig instance to a dict.

        Returns:
            dict: The dict dumped from CalibConfig.
        """
        return self.model_dump()





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
  - id: tensorrt_calib_dataset_not_empty
    engine: tensorrt
    library: tensorrt
    invariant_under_test: CalibConfig calib_dataset not empty
    severity: error
    native_type: tensorrt.CalibConfig
    miner_source:
      path: CalibConfig.py
      method: __init__
      line_at_scan: 10
    match:
      engine: tensorrt
      fields:
        tensorrt_llm.calib_dataset: {present: true, not_equal: ''}
    kwargs_positive:
      calib_dataset: ''
    kwargs_negative:
      calib_dataset: 'cnn_dailymail'
    expected_outcome:
      outcome: error
      emission_channel: logger_error_once
      normalised_fields: []
    message_template: "CalibConfig calib_dataset cannot be empty"
    added_by: llm_miner_pass3
    added_at: '2026-05-25'

  - id: tensorrt_calib_batches_positive
    engine: tensorrt
    library: tensorrt
    invariant_under_test: CalibConfig calib_batches positive
    severity: error
    native_type: tensorrt.CalibConfig
    miner_source:
      path: CalibConfig.py
      method: __init__
      line_at_scan: 15
    match:
      engine: tensorrt
      fields:
        tensorrt_llm.calib_batches: {'>': 0}
    kwargs_positive:
      calib_batches: -1
    kwargs_negative:
      calib_batches: 512
    expected_outcome:
      outcome: error
      emission_channel: logger_error_once
      normalised_fields: []
    message_template: "CalibConfig calib_batches must be positive"
    added_by: llm_miner_pass3
    added_at: '2026-05-25'

  - id: tensorrt_calib_batch_size_positive
    engine: tensorrt
    library: tensorrt
    invariant_under_test: CalibConfig calib_batch_size positive
    severity: error
    native_type: tensorrt.CalibConfig
    miner_source:
      path: CalibConfig.py
      method: __init__
      line_at_scan: 20
    match:
      engine: tensorrt
      fields:
        tensorrt_llm.calib_batch_size: {'>': 0}
    kwargs_positive:
      calib_batch_size: -1
    kwargs_negative:
      calib_batch_size: 1
    expected_outcome:
      outcome: error
      emission_channel: logger_error_once
      normalised_fields: []
    message_template: "CalibConfig calib_batch_size must be positive"
    added_by: llm_miner_pass3
    added_at: '2026-05-25'

  - id: tensorrt_calib_max_seq_length_positive
    engine: tensorrt
    library: tensorrt
    invariant_under_test: CalibConfig calib_max_seq_length positive
    severity: error
    native_type: tensorrt.CalibConfig
    miner_source:
      path: CalibConfig.py
      method: __init__
      line_at_scan: 25
    match:
      engine: tensorrt
      fields:
        tensorrt_llm.calib_max_seq_length: {'>': 0}
    kwargs_positive:
      calib_max_seq_length: -1
    kwargs_negative:
      calib_max_seq_length: 512
    expected_outcome:
      outcome: error
      emission_channel: logger_error_once
      normalised_fields: []
    message_template: "CalibConfig calib_max_seq_length must be positive"
    added_by: llm_miner_pass3
    added_at: '2026-05-25'

  - id: tensorrt_random_seed_non_negative
    engine: tensorrt
    library: tensorrt
    invariant_under_test: CalibConfig random_seed non-negative
    severity: error
    native_type: tensorrt.CalibConfig
    miner_source:
      path: CalibConfig.py
      method: __init__
      line_at_scan: 30
    match:
      engine: tensorrt
      fields:
        tensorrt_llm.random_seed: {'>=': 0}
    kwargs_positive:
      random_seed: -1
    kwargs_negative:
      random_seed: 1234
    expected_outcome:
      outcome: error
      emission_channel: logger_error_once
      normalised_fields: []
    message_template: "CalibConfig random_seed must be non-negative"
    added_by: llm_miner_pass3
    added_at: '2026-05-25'

  - id: tensorrt_tokenizer_max_seq_length_positive
    engine: tensorrt
    library: tensorrt
    invariant_under_test: CalibConfig tokenizer_max_seq_length positive
    severity: error
    native_type: tensorrt.CalibConfig
    miner_source:
      path: CalibConfig.py
      method: __init__
      line_at_scan: 35
    match:
      engine: tensorrt
      fields:
        tensorrt_llm.tokenizer_max_seq_length: {'>': 0}
    kwargs_positive:
      tokenizer_max_seq_length: -1
    kwargs_negative:
      tokenizer_max_seq_length: 2048
    expected_outcome:
      outcome: error
      emission_channel: logger_error_once
      normalised_fields: []
    message_template: "CalibConfig tokenizer_max_seq_length must be positive"
    added_by: llm_miner_pass3
    added_at: '2026-05-25'
```
