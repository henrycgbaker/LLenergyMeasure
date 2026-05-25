# invariants_pass3_extend extraction transcript: validate_section_07_2.2._detect_beam_only_parameterization_when_not_in_beam_mode

- chunk_description: GenerationConfig.validate section: 2.2._detect_beam_only_parameterization_when_not_in_beam_mode
- expected_namespaces: ['transformers.sampling']
- attempts: 1
- elapsed_sec: 97.40
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser doing PASS 3 (extension) of multi-pass
invariant extraction. PASS 1 already extracted N invariants from
transformers v4.57.6 for ONE chunk of source. Your job is to
identify ADDITIONAL invariants PASS 1 missed.

INPUT 1 - PASS 1 EXTRACTED INVARIANTS (already known; do NOT re-emit):

invariants:
- id: transformers_early_stopping_set_when_num_beams_one
  severity: dormant
  match:
    engine: transformers
    fields:
      transformers.sampling.num_beams:
        ==: 1
  invariant_under_test: GenerationConfig.validate flags early_stopping set when num_beams=1
- id: transformers_length_penalty_set_when_num_beams_one
  severity: dormant
  match:
    engine: transformers
    fields:
      transformers.sampling.num_beams:
        ==: 1
  invariant_under_test: GenerationConfig.validate flags length_penalty set when num_beams=1


INPUT 2 - PASS 2 FLAGGED ISSUES (for context; consider whether they hint
at related misses):

(no flags)

INPUT 3 - THE SOURCE PASS 1 READ:

=== CONTEXT: GenerationConfig.validate() ===
This is one logical section. Other sections are mined separately. Focus only on validations visible in THIS section.

IMPORTANT: Each `if self.<field> is not None and <cond>:` block must produce ONE invariant per <field>. Do NOT collapse multiple field checks into one invariant keyed by `do_sample` or `num_beams`. Each per-field check is its OWN dormant invariant. Key by the FIELD being flagged (temperature, top_p, min_p, etc.), NOT by the gating field (do_sample / num_beams).


=== SOURCE: validate() section 8 (2.2._detect_beam_only_parameterization_when_not_in_beam_mode) ===
# 2.2. detect beam-only parameterization when not in beam mode
        if self.num_beams == 1:
            single_beam_wrong_parameter_msg = (
                "`num_beams` is set to 1. However, `{flag_name}` is set to `{flag_value}` -- this flag is only used "
                "in beam-based generation modes. You should set `num_beams>1` or unset `{flag_name}`."
            )
            if self.early_stopping is not False:
                minor_issues["early_stopping"] = single_beam_wrong_parameter_msg.format(
                    flag_name="early_stopping", flag_value=self.early_stopping
                )
            if self.length_penalty is not None and self.length_penalty != 1.0:
                minor_issues["length_penalty"] = single_beam_wrong_parameter_msg.format(
                    flag_name="length_penalty", flag_value=self.length_penalty
                )



OUTPUT FORMAT: YAML document with the SAME shape as the original
invariants prompt's output:

invariants:
- id: <snake_case_unique_id>
  engine: transformers
  library: transformers
  invariant_under_test: <one-line: what the library checks here>
  severity: <error|dormant|warning>
  native_type: <e.g. transformers.GenerationConfig or transformers.EngineArgs>
  miner_source:
    path: <file path>
    method: <validate|__init__|_verify_args|...>
    line_at_scan: <approximate line number if visible>
  match:
    engine: transformers
    fields:
      transformers.sampling.<field>: <value or predicate>
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
- Exact value:         `transformers.sampling.field: value`
- Not in list:         `transformers.sampling.field: {present: true, not_in: [a, b]}`
- Greater than:        `transformers.sampling.field: {'>': value}`
- Less than:           `transformers.sampling.field: {'<': value}`
- Type not in:         `transformers.sampling.field: {present: true, type_is_not: [TypeName]}`
- Presence:            `transformers.sampling.field: {present: true}`

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
- id: transformers_length_penalty_set_when_num_beams_one_not_equal_to_1.0
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.validate flags length_penalty set when num_beams=1 and not equal to 1.0
  severity: dormant
  native_type: transformers.GenerationConfig
  miner_source:
    path: generation_config.py
    method: validate
    line_at_scan: 123
  match:
    engine: transformers
    fields:
      transformers.sampling.num_beams:
        ==: 1
      transformers.sampling.length_penalty:
        not_equal: 1.0
  kwargs_positive:
    num_beams: 1
    length_penalty: 2.0
  kwargs_negative:
    num_beams: 1
    length_penalty: 1.0
  expected_outcome:
    outcome: dormant_announced
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: '`num_beams` is set to 1. However, `length_penalty` is set to `{flag_value}` -- this flag is only used in beam-based generation modes. You should set `num_beams>1` or unset `length_penalty`.'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
```
