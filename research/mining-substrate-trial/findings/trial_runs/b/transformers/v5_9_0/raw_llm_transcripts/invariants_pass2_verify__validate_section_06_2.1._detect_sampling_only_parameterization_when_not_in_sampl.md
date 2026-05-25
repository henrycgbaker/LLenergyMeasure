# invariants_pass2_verify extraction transcript: validate_section_06_2.1._detect_sampling_only_parameterization_when_not_in_sampl

- chunk_description: GenerationConfig.validate section: 2.1._detect_sampling_only_parameterization_when_not_in_sampl
- expected_namespaces: ['transformers.sampling']
- attempts: 1
- elapsed_sec: 188.17
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser doing PASS 2 (verification) of multi-pass
invariant extraction. PASS 1 already extracted invariants from
transformers v5.9.0 for ONE chunk of source. Your job is to
REVIEW each emitted invariant against the source and flag any that
look WRONG.

INPUT 1 - PASS 1 EXTRACTED INVARIANTS (the candidate list):

invariants:
- id: transformers_temperature_set_when_do_sample_false
  severity: dormant
  match:
    engine: transformers
    fields:
      transformers.sampling.do_sample: false
      transformers.sampling.temperature:
        present: true
  invariant_under_test: GenerationConfig.validate flags temperature set when do_sample=False
- id: transformers_top_p_set_when_do_sample_false
  severity: dormant
  match:
    engine: transformers
    fields:
      transformers.sampling.do_sample: false
      transformers.sampling.top_p:
        present: true
  invariant_under_test: GenerationConfig.validate flags top_p set when do_sample=False
- id: transformers_top_k_set_when_do_sample_false
  severity: dormant
  match:
    engine: transformers
    fields:
      transformers.sampling.do_sample: false
      transformers.sampling.top_k:
        present: true
  invariant_under_test: GenerationConfig.validate flags top_k set when do_sample=False
- id: transformers_min_length_set_when_do_sample_false
  severity: dormant
  match:
    engine: transformers
    fields:
      transformers.sampling.do_sample: false
      transformers.sampling.min_length:
        present: true
  invariant_under_test: GenerationConfig.validate flags min_length set when do_sample=False
- id: transformers_max_length_set_when_do_sample_false
  severity: dormant
  match:
    engine: transformers
    fields:
      transformers.sampling.do_sample: false
      transformers.sampling.max_length:
        present: true
  invariant_under_test: GenerationConfig.validate flags max_length set when do_sample=False
- id: transformers_no_repeat_ngram_size_set_when_do_sample_false
  severity: dormant
  match:
    engine: transformers
    fields:
      transformers.sampling.do_sample: false
      transformers.sampling.no_repeat_ngram_size:
        present: true
  invariant_under_test: GenerationConfig.validate flags no_repeat_ngram_size set when
    do_sample=False
- id: transformers_num_beams_set_when_do_sample_false
  severity: dormant
  match:
    engine: transformers
    fields:
      transformers.sampling.do_sample: false
      transformers.sampling.num_beams:
        present: true
  invariant_under_test: GenerationConfig.validate flags num_beams set when do_sample=False
- id: transformers_num_return_sequences_set_when_do_sample_false
  severity: dormant
  match:
    engine: transformers
    fields:
      transformers.sampling.do_sample: false
      transformers.sampling.num_return_sequences:
        present: true
  invariant_under_test: GenerationConfig.validate flags num_return_sequences set when
    do_sample=False
- id: transformers_num_beam_groups_set_when_do_sample_false
  severity: dormant
  match:
    engine: transformers
    fields:
      transformers.sampling.do_sample: false
      transformers.sampling.num_beam_groups:
        present: true
  invariant_under_test: GenerationConfig.validate flags num_beam_groups set when do_sample=False
- id: transformers_diversity_penalty_set_when_do_sample_false
  severity: dormant
  match:
    engine: transformers
    fields:
      transformers.sampling.do_sample: false
      transformers.sampling.diversity_penalty:
        present: true
  invariant_under_test: GenerationConfig.validate flags diversity_penalty set when
    do_sample=False
- id: transformers_repetition_penalty_set_when_do_sample_false
  severity: dormant
  match:
    engine: transformers
    fields:
      transformers.sampling.do_sample: false
      transformers.sampling.repetition_penalty:
        present: true
  invariant_under_test: GenerationConfig.validate flags repetition_penalty set when
    do_sample=False
- id: transformers_repetition_penalty_range_set_when_do_sample_false
  severity: dormant
  match:
    engine: transformers
    fields:
      transformers.sampling.do_sample: false
      transformers.sampling.repetition_penalty_range:
        present: true
  invariant_under_test: GenerationConfig.validate flags repetition_penalty_range set
    when do_sample=False
- id: transformers_repetition_penalty_slope_set_when_do_sample_false
  severity: dormant
  match:
    engine: transformers
    fields:
      transformers.sampling.do_sample: false
      transformers.sampling.repetition_penalty_slope:
        present: true
  invariant_under_test: GenerationConfig.validate flags repetition_penalty_slope set
    when do_sample=False
- id: transformers_encoder_no_repeat_ngram_size_set_when_do_sample_false
  severity: dormant
  match:
    engine: transformers
    fields:
      transformers.sampling.do_sample: false
      transformers.sampling.encoder_no_repeat_ngram_size:
        present: true
  invariant_under_test: GenerationConfig.validate flags encoder_no_repeat_ngram_size
    set when do_sample=False
- id: transformers_encoder_repetition_penalty_set_when_do_sample_false
  severity: dormant
  match:
    engine: transformers
    fields:
      transformers.sampling.do_sample: false
      transformers.sampling.encoder_repetition_penalty:
        present: true
  invariant_under_test: GenerationConfig.validate flags encoder_repetition_penalty
    set when do_sample=False


INPUT 2 - THE SOURCE PASS 1 READ:

=== CONTEXT: GenerationConfig.validate() ===
This is one logical section. Other sections are mined separately. Focus only on validations visible in THIS section.

IMPORTANT: Each `if self.<field> is not None and <cond>:` block must produce ONE invariant per <field>. Do NOT collapse multiple field checks into one invariant keyed by `do_sample` or `num_beams`. Each per-field check is its OWN dormant invariant. Key by the FIELD being flagged (temperature, top_p, min_p, etc.), NOT by the gating field (do_sample / num_beams).


=== SOURCE: validate() section 7 (2.1._detect_sampling_only_parameterization_when_not_in_sampl) ===
# 2.1. detect sampling-only parameterization when not in sampling mode



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
- transformers_temperature_set_when_do_sample_false
- transformers_top_p_set_when_do_sample_false
- transformers_top_k_set_when_do_sample_false
- transformers_min_length_set_when_do_sample_false
- transformers_max_length_set_when_do_sample_false
- transformers_no_repeat_ngram_size_set_when_do_sample_false
- transformers_num_beams_set_when_do_sample_false
- transformers_num_return_sequences_set_when_do_sample_false
- transformers_num_beam_groups_set_when_do_sample_false
- transformers_diversity_penalty_set_when_do_sample_false
- transformers_repetition_penalty_set_when_do_sample_false
- transformers_repetition_penalty_range_set_when_do_sample_false
- transformers_repetition_penalty_slope_set_when_do_sample_false
- transformers_encoder_no_repeat_ngram_size_set_when_do_sample_false
- transformers_encoder_repetition_penalty_set_when_do_sample_false
```
