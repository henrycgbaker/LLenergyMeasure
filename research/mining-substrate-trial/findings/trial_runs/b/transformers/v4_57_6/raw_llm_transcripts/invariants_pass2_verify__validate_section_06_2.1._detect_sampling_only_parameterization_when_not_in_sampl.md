# invariants_pass2_verify extraction transcript: validate_section_06_2.1._detect_sampling_only_parameterization_when_not_in_sampl

- chunk_description: GenerationConfig.validate section: 2.1._detect_sampling_only_parameterization_when_not_in_sampl
- expected_namespaces: ['transformers.sampling']
- attempts: 1
- elapsed_sec: 25.92
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser doing PASS 2 (verification) of multi-pass
invariant extraction. PASS 1 already extracted invariants from
transformers v4.57.6 for ONE chunk of source. Your job is to
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


INPUT 2 - THE SOURCE PASS 1 READ:

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
- transformers_min_p_set_when_do_sample_false
- transformers_typical_p_set_when_do_sample_false
- transformers_top_k_set_when_do_sample_false
- transformers_epsilon_cutoff_set_when_do_sample_false
- transformers_eta_cutoff_set_when_do_sample_false
```
