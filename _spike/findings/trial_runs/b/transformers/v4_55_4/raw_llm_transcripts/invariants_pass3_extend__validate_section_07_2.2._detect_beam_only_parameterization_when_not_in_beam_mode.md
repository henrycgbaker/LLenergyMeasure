# invariants_pass3_extend extraction transcript: validate_section_07_2.2._detect_beam_only_parameterization_when_not_in_beam_mode

- chunk_description: GenerationConfig.validate section: 2.2._detect_beam_only_parameterization_when_not_in_beam_mode
- expected_namespaces: ['transformers.sampling']
- attempts: 1
- elapsed_sec: 84.96
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser doing PASS 3 (extension) of multi-pass
invariant extraction. PASS 1 already extracted N invariants from
transformers v4.55.4 for ONE chunk of source. Your job is to
identify ADDITIONAL invariants PASS 1 missed.

INPUT 1 - PASS 1 EXTRACTED INVARIANTS (already known; do NOT re-emit):

invariants:
- id: transformers_early_stopping_set_when_num_beams_one
  severity: dormant
  match:
    engine: transformers
    fields:
      transformers.sampling.num_beams:
        present: true
        equal: 1
  invariant_under_test: GenerationConfig.validate flags early_stopping set when num_beams=1
- id: transformers_num_beam_groups_set_when_num_beams_one
  severity: dormant
  match:
    engine: transformers
    fields:
      transformers.sampling.num_beams:
        present: true
        equal: 1
  invariant_under_test: GenerationConfig.validate flags num_beam_groups set when num_beams=1
- id: transformers_diversity_penalty_set_when_num_beams_one
  severity: dormant
  match:
    engine: transformers
    fields:
      transformers.sampling.num_beams:
        present: true
        equal: 1
  invariant_under_test: GenerationConfig.validate flags diversity_penalty set when
    num_beams=1
- id: transformers_length_penalty_set_when_num_beams_one
  severity: dormant
  match:
    engine: transformers
    fields:
      transformers.sampling.num_beams:
        present: true
        equal: 1
  invariant_under_test: GenerationConfig.validate flags length_penalty set when num_beams=1
- id: transformers_constraints_set_when_num_beams_one
  severity: dormant
  match:
    engine: transformers
    fields:
      transformers.sampling.num_beams:
        present: true
        equal: 1
  invariant_under_test: GenerationConfig.validate flags constraints set when num_beams=1
- id: transformers_repetition_penalty_set_when_dola_layers_and_num_beams_one
  severity: warning
  match:
    engine: transformers
    fields:
      transformers.sampling.num_beams:
        present: true
        equal: 1
      transformers.sampling.dola_layers:
        present: true
  invariant_under_test: GenerationConfig.validate flags repetition_penalty set when
    dola_layers and num_beams=1


INPUT 2 - PASS 2 FLAGGED ISSUES (for context; consider whether they hint
at related misses):

flagged:
- id: transformers_repetition_penalty_set_when_dola_layers_and_num_beams_one
  reason: Source checks for `repetition_penalty < 1.2`, but invariant has no such
    condition.
  fix: correct_predicate:range


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
            if self.num_beam_groups is not None and self.num_beam_groups != 1:
                minor_issues["num_beam_groups"] = single_beam_wrong_parameter_msg.format(
                    flag_name="num_beam_groups", flag_value=self.num_beam_groups
                )
            if self.diversity_penalty is not None and self.diversity_penalty != 0.0:
                minor_issues["diversity_penalty"] = single_beam_wrong_parameter_msg.format(
                    flag_name="diversity_penalty", flag_value=self.diversity_penalty
                )
            if self.length_penalty is not None and self.length_penalty != 1.0:
                minor_issues["length_penalty"] = single_beam_wrong_parameter_msg.format(
                    flag_name="length_penalty", flag_value=self.length_penalty
                )
            if self.constraints is not None:
                minor_issues["constraints"] = single_beam_wrong_parameter_msg.format(
                    flag_name="constraints", flag_value=self.constraints
                )
            # DoLa generation needs num_beams == 1
            if self.dola_layers is not None and (self.repetition_penalty is None or self.repetition_penalty < 1.2):
                minor_issues["repetition_penalty"] = (
                    "`dola_layers` is set to trigger DoLa decoding, but `repetition_penalty` is set to a value of "
                    f"{self.repetition_penalty}, which could induce unwanted repetition. The recommended value for "
                    "DoLa decoding is `repetition_penalty>=1.2`.",
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

...<truncated>...
```

### Response

```
invariants:
- id: transformers_repetition_penalty_range_check_when_dola_layers_set
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.validate checks repetition_penalty >= 1.2 when dola_layers is set
  severity: warning
  native_type: transformers.GenerationConfig
  miner_source:
    path: generation_config.py
    method: validate
    line_at_scan: 123
  match:
    engine: transformers
    fields:
      transformers.sampling.dola_layers: {present: true}
      transformers.sampling.repetition_penalty: {'>=': 1.2}
  kwargs_positive:
    dola_layers: True
    repetition_penalty: 1.3
  kwargs_negative:
    dola_layers: False
    repetition_penalty: 0.9
  expected_outcome:
    outcome: warning
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: '`dola_layers` is set to trigger DoLa decoding, but `repetition_penalty` is set to a value of {self.repetition_penalty}, which could induce unwanted repetition. The recommended value for DoLa decoding is `repetition_penalty>=1.2`.'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
```
