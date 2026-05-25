# invariants_pass2_verify extraction transcript: lookahead_validator

- chunk_description: tensorrt_llm.LookaheadDecodingConfig @field_validator (positive values)
- expected_namespaces: ['tensorrt_llm']
- attempts: 1
- elapsed_sec: 33.36
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser doing PASS 2 (verification) of multi-pass
invariant extraction. PASS 1 already extracted invariants from
tensorrt v1.0.0 for ONE chunk of source. Your job is to
REVIEW each emitted invariant against the source and flag any that
look WRONG.

INPUT 1 - PASS 1 EXTRACTED INVARIANTS (the candidate list):

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


INPUT 2 - THE SOURCE PASS 1 READ:

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

    def supports_backend(self, backend: str) -> bool:
        return backend not in ("pytorch", "_autodeploy")

    decoding_type: ClassVar[str] = "Lookahead"


SpeculativeConfig: TypeAlias = Optional[Union[
    DraftTargetDecodingConfig,
    EagleDecodingConfig,
    LookaheadDecodingConfig,
    MedusaDecodingConfig,
    MTPDecodingConfig,
    NGramDecodingConfig,
    UserProvidedDecodingConfig,
    AutoDecodingConfig,
]]





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
- tensorrt_llm_max_window_size_le_zero
- tensorrt_llm_max_ngram_size_le_zero
- tensorrt_llm_max_verification_set_size_le_zero
```
