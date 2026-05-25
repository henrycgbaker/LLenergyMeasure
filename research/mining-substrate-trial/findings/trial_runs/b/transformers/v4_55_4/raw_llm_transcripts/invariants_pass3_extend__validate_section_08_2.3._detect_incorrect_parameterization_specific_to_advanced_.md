# invariants_pass3_extend extraction transcript: validate_section_08_2.3._detect_incorrect_parameterization_specific_to_advanced_

- chunk_description: GenerationConfig.validate section: 2.3._detect_incorrect_parameterization_specific_to_advanced_
- expected_namespaces: ['transformers.sampling']
- attempts: 1
- elapsed_sec: 89.21
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
- id: transformers_do_sample_set_when_constrained_beam_search
  severity: error
  match:
    engine: transformers
    fields:
      transformers.sampling.constraints:
        present: true
      transformers.sampling.force_words_ids:
        present: true
      transformers.sampling.do_sample: true
  invariant_under_test: GenerationConfig.validate flags do_sample=True when constrained
    beam search is triggered
- id: transformers_num_beam_groups_set_when_constrained_beam_search
  severity: error
  match:
    engine: transformers
    fields:
      transformers.sampling.constraints:
        present: true
      transformers.sampling.force_words_ids:
        present: true
      transformers.sampling.num_beam_groups:
        '>=': 2
  invariant_under_test: GenerationConfig.validate flags num_beam_groups!=1 when constrained
    beam search is triggered
- id: transformers_num_beams_not_divisible_by_num_beam_groups_when_group_beam_search
  severity: error
  match:
    engine: transformers
    fields:
      transformers.sampling.diversity_penalty:
        '>': 0.0
      transformers.sampling.num_beam_groups:
        '>=': 2
      transformers.sampling.num_beams:
        present: true
        not_divisible_by:
        - transformers.sampling.num_beam_groups
  invariant_under_test: GenerationConfig.validate flags num_beams not divisible by
    num_beam_groups when group beam search is triggered
- id: transformers_diversity_penalty_zero_when_group_beam_search
  severity: error
  match:
    engine: transformers
    fields:
      transformers.sampling.diversity_penalty:
        <=: 0.0
      transformers.sampling.num_beam_groups:
        '>=': 2
  invariant_under_test: GenerationConfig.validate flags diversity_penalty=0.0 when
    group beam search is triggered


INPUT 2 - PASS 2 FLAGGED ISSUES (for context; consider whether they hint
at related misses):

flagged:
- id: transformers_do_sample_set_when_group_beam_search
  reason: source code raises ValueError when `do_sample` is True, but invariant says
    it should be set to True
  fix: drop
- id: transformers_num_beams_not_divisible_by_num_beam_groups_when_group_beam_search
  reason: source code requires `num_beams` to be divisible by `num_beam_groups`, but
    invariant says the opposite
  fix: correct_predicate:not_divisible_by -> divisible_by


INPUT 3 - THE SOURCE PASS 1 READ:

=== CONTEXT: GenerationConfig.validate() ===
This is one logical section. Other sections are mined separately. Focus only on validations visible in THIS section.

=== SOURCE: validate() section 9 (2.3._detect_incorrect_parameterization_specific_to_advanced_) ===
# 2.3. detect incorrect parameterization specific to advanced beam modes
        else:
            # constrained beam search
            if self.constraints is not None or self.force_words_ids is not None:
                constrained_wrong_parameter_msg = (
                    "one of `constraints`, `force_words_ids` is not `None`, triggering constrained beam search. "
                    "However, `{flag_name}` is set to `{flag_value}`, which is incompatible with this generation "
                    "mode. Set `constraints` and `force_words_ids` to `None` or unset `{flag_name}` to continue."
                )
                if self.do_sample is True:
                    raise ValueError(
                        constrained_wrong_parameter_msg.format(flag_name="do_sample", flag_value=self.do_sample)
                    )
                if self.num_beam_groups is not None and self.num_beam_groups != 1:
                    raise ValueError(
                        constrained_wrong_parameter_msg.format(
                            flag_name="num_beam_groups", flag_value=self.num_beam_groups
                        )
                    )
            # group beam search
            elif self.diversity_penalty != 0.0 or self.num_beam_groups != 1:
                group_error_prefix = (
                    "`diversity_penalty` is not 0.0 or `num_beam_groups` is not 1, triggering group beam search. In "
                    "this generation mode, "
                )
                if self.do_sample is True:
                    raise ValueError(group_error_prefix + "`do_sample` must be set to `False`")
                if self.num_beams % self.num_beam_groups != 0:
                    raise ValueError(group_error_prefix + "`num_beams` should be divisible by `num_beam_groups`")
                if self.diversity_penalty == 0.0:
                    raise ValueError(
                        group_error_prefix
                        + "`diversity_penalty` should be greater than `0.0`, otherwise your groups will be identical."
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
- id: transformers_do_sample_set_when_group_beam_search_is_false
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.validate flags do_sample=False when group beam search is triggered
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: generation_config.py
    method: validate
    line_at_scan: 123
  match:
    engine: transformers
    fields:
      transformers.sampling.do_sample:
        exact: false
      transformers.sampling.diversity_penalty:
        '>': 0.0
      transformers.sampling.num_beam_groups:
        '>=': 2
  kwargs_positive:
    do_sample: true
    diversity_penalty: 1.0
    num_beam_groups: 2
  kwargs_negative:
    do_sample: false
    diversity_penalty: 0.0
    num_beam_groups: 1
  expected_outcome:
    outcome: error
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: '`do_sample` must be set to `False`'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: transformers_diversity_penalty_greater_than_zero_when_group_beam_search
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.validate flags diversity_penalty>0.0 when group beam search is triggered
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: generation_config.py
    method: validate
    line_at_scan: 123
  match:
    engine: transformers
    fields:
      transformers.sampling.diversity_penalty:
        '>': 0.0
      transformers.sampling.num_beam_groups:
        '>=': 2
  kwargs_positive:
    diversity_penalty: 1.0
    num_beam_groups: 2
  kwargs_negative:
    diversity_penalty: 0.0
    num_beam_groups: 1
  expected_outcome:
    outcome: error
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: '`diversity_penalty` should be greater than `0.0`, otherwise your groups will be identical.'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: transformers_num_beams_divisible_by_num_beam_groups_when_group_beam_search
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.validate flags num_beams divisible by num_beam_groups when group beam search is triggered
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: generation_config.py
    method: validate
    line_at_scan: 123
  match:
    engine: transformers
    fields:
      transformers.sampling.num_beams:
        present: true
        divisible_by:
        - transformers.sampling.num_beam_groups
      transformers.sampling.diversity_penalty:
        '>': 0.0
      transformers.sampling.num_beam_groups:
        '>=': 2
  kwargs_positive:
    num_beams: 4
    diversity_penalty: 1.0
    num_beam_groups: 2
  kwargs_negative:
    num_beams: 3
    diversity_penalty: 0.0
    num_beam_groups: 1
  expected_outcome:
    outcome: error
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: '`num_beams` should be divisible by `num_beam_groups`'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
```
