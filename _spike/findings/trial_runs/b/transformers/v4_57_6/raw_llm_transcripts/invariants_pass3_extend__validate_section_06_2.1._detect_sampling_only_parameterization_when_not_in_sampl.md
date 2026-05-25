# invariants_pass3_extend extraction transcript: validate_section_06_2.1._detect_sampling_only_parameterization_when_not_in_sampl

- chunk_description: GenerationConfig.validate section: 2.1._detect_sampling_only_parameterization_when_not_in_sampl
- expected_namespaces: ['transformers.sampling']
- attempts: 1
- elapsed_sec: 248.65
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
- id: transformers_temperature_set_when_do_sample_false
  severity: dormant
  match:
    engine: transformers
    fields:
      transformers.sampling.do_sample: false
      transformers.sampling.temperature:
        present: true
        not_equal: 1.0
  invariant_under_test: GenerationConfig.validate flags temperature set when do_sample=False
- id: transformers_top_p_set_when_do_sample_false
  severity: dormant
  match:
    engine: transformers
    fields:
      transformers.sampling.do_sample: false
      transformers.sampling.top_p:
        present: true
        not_equal: 1.0
  invariant_under_test: GenerationConfig.validate flags top_p set when do_sample=False
- id: transformers_min_p_set_when_do_sample_false
  severity: dormant
  match:
    engine: transformers
    fields:
      transformers.sampling.do_sample: false
      transformers.sampling.min_p:
        present: true
  invariant_under_test: GenerationConfig.validate flags min_p set when do_sample=False
- id: transformers_typical_p_set_when_do_sample_false
  severity: dormant
  match:
    engine: transformers
    fields:
      transformers.sampling.do_sample: false
      transformers.sampling.typical_p:
        present: true
        not_equal: 1.0
  invariant_under_test: GenerationConfig.validate flags typical_p set when do_sample=False
- id: transformers_top_k_set_when_do_sample_false
  severity: dormant
  match:
    engine: transformers
    fields:
      transformers.sampling.do_sample: false
      transformers.sampling.top_k:
        present: true
        not_equal: 50
  invariant_under_test: GenerationConfig.validate flags top_k set when do_sample=False
- id: transformers_epsilon_cutoff_set_when_do_sample_false
  severity: dormant
  match:
    engine: transformers
    fields:
      transformers.sampling.do_sample: false
      transformers.sampling.epsilon_cutoff:
        present: true
        not_equal: 0.0
  invariant_under_test: GenerationConfig.validate flags epsilon_cutoff set when do_sample=False
- id: transformers_eta_cutoff_set_when_do_sample_false
  severity: dormant
  match:
    engine: transformers
    fields:
      transformers.sampling.do_sample: false
      transformers.sampling.eta_cutoff:
        present: true
        not_equal: 0.0
  invariant_under_test: GenerationConfig.validate flags eta_cutoff set when do_sample=False


INPUT 2 - PASS 2 FLAGGED ISSUES (for context; consider whether they hint
at related misses):

(no flags)

INPUT 3 - THE SOURCE PASS 1 READ:

=== CONTEXT: GenerationConfig.validate() ===
This is one logical section. Other sections are mined separately. Focus only on validations visible in THIS section.

IMPORTANT: Each `if self.<field> is not None and <cond>:` block must produce ONE invariant per <field>. Do NOT collapse multiple field checks into one invariant keyed by `do_sample` or `num_beams`. Each per-field check is its OWN dormant invariant. Key by the FIELD being flagged (temperature, top_p, min_p, etc.), NOT by the gating field (do_sample / num_beams).


=== SOURCE: validate() section 7 (2.1._detect_sampling_only_parameterization_when_not_in_sampl) ===
# 2.1. detect sampling-only parameterization when not in sampling mode
        if self.do_sample is False:
            greedy_wrong_parameter_msg = (
                "`do_sample` is set to `False`. However, `{flag_name}` is set to `{flag_value}` -- this flag is only "
                "used in sample-based generation modes. You should set `do_sample=True` or unset `{flag_name}`."
            )
            if self.temperature is not None and self.temperature != 1.0:
                minor_issues["temperature"] = greedy_wrong_parameter_msg.format(
                    flag_name="temperature", flag_value=self.temperature
                )
            if self.top_p is not None and self.top_p != 1.0:
                minor_issues["top_p"] = greedy_wrong_parameter_msg.format(flag_name="top_p", flag_value=self.top_p)
            if self.min_p is not None:
                minor_issues["min_p"] = greedy_wrong_parameter_msg.format(flag_name="min_p", flag_value=self.min_p)
            if self.typical_p is not None and self.typical_p != 1.0:
                minor_issues["typical_p"] = greedy_wrong_parameter_msg.format(
                    flag_name="typical_p", flag_value=self.typical_p
                )
            if self.top_k is not None and self.top_k != 50:
                minor_issues["top_k"] = greedy_wrong_parameter_msg.format(flag_name="top_k", flag_value=self.top_k)
            if self.epsilon_cutoff is not None and self.epsilon_cutoff != 0.0:
                minor_issues["epsilon_cutoff"] = greedy_wrong_parameter_msg.format(
                    flag_name="epsilon_cutoff", flag_value=self.epsilon_cutoff
                )
            if self.eta_cutoff is not None and self.eta_cutoff != 0.0:
                minor_issues["eta_cutoff"] = greedy_wrong_parameter_msg.format(
                    flag_name="eta_cutoff", flag_value=self.eta_cutoff
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
5. Use
...<truncated>...
```

### Response

```
invariants:
  - id: transformers_temperature_set_when_do_sample_false_and_not_equal_one
    engine: transformers
    library: transformers
    invariant_under_test: GenerationConfig.validate flags temperature set when do_sample=False and not equal to 1.0
    severity: dormant
    native_type: transformers.GenerationConfig
    miner_source:
      path: generation_config.py
      method: validate
      line_at_scan: 123
    match:
      engine: transformers
      fields:
        transformers.sampling.do_sample: false
        transformers.sampling.temperature:
          present: true
          not_equal: 1.0
    kwargs_positive:
      temperature: 0.7
    kwargs_negative:
      temperature: 1.0
    expected_outcome:
      outcome: dormant_announced
      emission_channel: logger_warning_once
      normalised_fields: []
    message_template: '`do_sample` is set to `False`. However, `{flag_name}` is set to `{flag_value}` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `{flag_name}`.'
    added_by: llm_miner_pass3
    added_at: '2026-05-25'
  - id: transformers_top_p_set_when_do_sample_false_and_not_equal_one
    engine: transformers
    library: transformers
    invariant_under_test: GenerationConfig.validate flags top_p set when do_sample=False and not equal to 1.0
    severity: dormant
    native_type: transformers.GenerationConfig
    miner_source:
      path: generation_config.py
      method: validate
      line_at_scan: 123
    match:
      engine: transformers
      fields:
        transformers.sampling.do_sample: false
        transformers.sampling.top_p:
          present: true
          not_equal: 1.0
    kwargs_positive:
      top_p: 0.7
    kwargs_negative:
      top_p: 1.0
    expected_outcome:
      outcome: dormant_announced
      emission_channel: logger_warning_once
      normalised_fields: []
    message_template: '`do_sample` is set to `False`. However, `{flag_name}` is set to `{flag_value}` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `{flag_name}`.'
    added_by: llm_miner_pass3
    added_at: '2026-05-25'
  - id: transformers_typical_p_set_when_do_sample_false_and_not_equal_one
    engine: transformers
    library: transformers
    invariant_under_test: GenerationConfig.validate flags typical_p set when do_sample=False and not equal to 1.0
    severity: dormant
    native_type: transformers.GenerationConfig
    miner_source:
      path: generation_config.py
      method: validate
      line_at_scan: 123
    match:
      engine: transformers
      fields:
        transformers.sampling.do_sample: false
        transformers.sampling.typical_p:
          present: true
          not_equal: 1.0
    kwargs_positive:
      typical_p: 0.7
    kwargs_negative:
      typical_p: 1.0
    expected_outcome:
      outcome: dormant_announced
      emission_channel: logger_warning_once
      normalised_fields: []
    message_template: '`do_sample` is set to `False`. However, `{flag_name}` is set to `{flag_value}` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `{flag_name}`.'
    added_by: llm_miner_pass3
    added_at: '2026-05-25'
  - id: transformers_top_k_set_when_do_sample_false_and_not_equal_fifty
    engine: transformers
    library: transformers
    invariant_under_test: GenerationConfig.validate flags top_k set when do_sample=False and not equal to 50
    severity: dormant
    native_type: transformers.GenerationConfig
    miner_source:
      path: generation_config.py
      method: validate
      line_at_scan: 123
    match:
      engine: transformers
      fields:
        transformers.sampling.do_sample: false
        transformers.sampling.top_k:
          present: true
          not_equal: 50
    kwargs_positive:
      top_k: 20
    kwargs_negative:
      top_k: 50
    expected_outcome:
      outcome: dormant_announced
      emission_channel: logger_warning_once
      normalised_fields: []
    message_template: '`do_sample` is set to `False`. However, `{flag_name}` is set to `{flag_value}` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `{flag_name}`.'
    added_by: llm_miner_pass3
    added_at: '2026-05-25'
  - id: transformers_epsilon_cutoff_set_when_do_sample_false_and_not_equal_zero
    engine: transformers
    library: transformers
    invariant_under_test: GenerationConfig.validate flags epsilon_cutoff set when do_sample=False and not equal to 0.0
    severity: dormant
    native_type: transformers.GenerationConfig
    miner_source:
      path: generation_config.py
      method: validate
      line_at_scan: 123
    match:
      engine: transformers
      fields:
        transformers.sampling.do_sample: false
        transformers.sampling.epsilon_cutoff:
          present: true
          not_equal: 0.0
    kwargs_positive:
      epsilon_cutoff: 0.1
    kwargs_negative:
      epsilon_cutoff: 0.0
    expected_outcome:
      outcome: dormant_announced
      emission_channel: logger_warning_once
      normalised_fields: []
    message_template: '`do_sample` is set to `False`. However, `{flag_name}` is set to `{flag_value}` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `{flag_name}`.'
    added_by: llm_miner_pass3
    added_at: '2026-05-25'
  - id: transformers_eta_cutoff_set_when_do_sample_false_and_not_equal_zero
    engine: transformers
    library: transformers
    invariant_under_test: GenerationConfig.validate flags eta_cutoff set when do_sample=False and not equal to 0.0
    severity: dormant
    native_type: transformers.GenerationConfig
    miner_source:
      path: generation_config.py
      method: validate
      line_at_scan: 123
    match:
      engine: transformers
      fields:
        transformers.sampling.do_sample: false
        transformers.sampling.eta_cutoff:
          present: true
          not_equal: 0.0
    kwargs_positive:
      eta_cutoff: 0.1
    kwargs_negative:
      eta_cutoff: 0.0
    expected_outcome:
      outcome: dormant_announced
      emission_channel: logger_warning_once
      normalised_fields: []
    message_template: '`do_sample` is set to `False`. However, `{flag_name}` is set to `{flag_value}` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `{flag_name}`.'
    added_by: llm_miner_pass3
    added_at: '2026-05-25'
```
