# invariants_pass3_extend extraction transcript: validate_section_11_3._Check_common_issue_passing_generate_arguments_inside_the_

- chunk_description: GenerationConfig.validate section: 3._Check_common_issue_passing_generate_arguments_inside_the_
- expected_namespaces: ['transformers.sampling']
- attempts: 1
- elapsed_sec: 97.86
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser doing PASS 3 (extension) of multi-pass
invariant extraction. PASS 1 already extracted N invariants from
transformers v5.9.0 for ONE chunk of source. Your job is to
identify ADDITIONAL invariants PASS 1 missed.

INPUT 1 - PASS 1 EXTRACTED INVARIANTS (already known; do NOT re-emit):

invariants:
- id: transformers_generate_arguments_not_in_config
  severity: error
  match:
    engine: transformers
    fields:
      transformers.sampling.logits_processor:
        present: true
      transformers.sampling.stopping_criteria:
        present: true
      transformers.sampling.prefix_allowed_tokens_fn:
        present: true
      transformers.sampling.synced_gpus:
        present: true
      transformers.sampling.assistant_model:
        present: true
      transformers.sampling.streamer:
        present: true
      transformers.sampling.negative_prompt_ids:
        present: true
      transformers.sampling.negative_prompt_attention_mask:
        present: true
  invariant_under_test: GenerationConfig.validate flags generate arguments not in
    config


INPUT 2 - PASS 2 FLAGGED ISSUES (for context; consider whether they hint
at related misses):

(no flags)

INPUT 3 - THE SOURCE PASS 1 READ:

=== CONTEXT: GenerationConfig.validate() ===
This is one logical section. Other sections are mined separately. Focus only on validations visible in THIS section.

=== SOURCE: validate() section 12 (3._Check_common_issue_passing_generate_arguments_inside_the_) ===
# 3. Check common issue: passing `generate` arguments inside the generation config
        generate_arguments = (
            "logits_processor",
            "stopping_criteria",
            "prefix_allowed_tokens_fn",
            "synced_gpus",
            "assistant_model",
            "streamer",
            "negative_prompt_ids",
            "negative_prompt_attention_mask",
        )
        for arg in generate_arguments:
            if hasattr(self, arg):
                raise ValueError(
                    f"Argument `{arg}` is not a valid argument of `GenerationConfig`. It should be passed to "
                    "`generate()` (or a pipeline) directly."
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
  - id: transformers_generate_arguments_not_in_config_per_field_logits_processor
    engine: transformers
    library: transformers
    invariant_under_test: GenerationConfig.validate flags generate arguments not in config for logits_processor
    severity: error
    native_type: transformers.GenerationConfig
    miner_source:
      path: generation_config.py
      method: validate
      line_at_scan: 123
    match:
      engine: transformers
      fields:
        transformers.sampling.logits_processor: {present: true}
    kwargs_positive:
      logits_processor: "some_value"
    kwargs_negative:
      logits_processor: None
    expected_outcome:
      outcome: error
      emission_channel: logger_warning_once
      normalised_fields: []
    message_template: 'Argument `logits_processor` is not a valid argument of `GenerationConfig`. It should be passed to `generate()` (or a pipeline) directly.'
    added_by: llm_miner_pass3
    added_at: '2026-05-25'

  - id: transformers_generate_arguments_not_in_config_per_field_stopping_criteria
    engine: transformers
    library: transformers
    invariant_under_test: GenerationConfig.validate flags generate arguments not in config for stopping_criteria
    severity: error
    native_type: transformers.GenerationConfig
    miner_source:
      path: generation_config.py
      method: validate
      line_at_scan: 123
    match:
      engine: transformers
      fields:
        transformers.sampling.stopping_criteria: {present: true}
    kwargs_positive:
      stopping_criteria: "some_value"
    kwargs_negative:
      stopping_criteria: None
    expected_outcome:
      outcome: error
      emission_channel: logger_warning_once
      normalised_fields: []
    message_template: 'Argument `stopping_criteria` is not a valid argument of `GenerationConfig`. It should be passed to `generate()` (or a pipeline) directly.'
    added_by: llm_miner_pass3
    added_at: '2026-05-25'

  - id: transformers_generate_arguments_not_in_config_per_field_prefix_allowed_tokens_fn
    engine: transformers
    library: transformers
    invariant_under_test: GenerationConfig.validate flags generate arguments not in config for prefix_allowed_tokens_fn
    severity: error
    native_type: transformers.GenerationConfig
    miner_source:
      path: generation_config.py
      method: validate
      line_at_scan: 123
    match:
      engine: transformers
      fields:
        transformers.sampling.prefix_allowed_tokens_fn: {present: true}
    kwargs_positive:
      prefix_allowed_tokens_fn: "some_value"
    kwargs_negative:
      prefix_allowed_tokens_fn: None
    expected_outcome:
      outcome: error
      emission_channel: logger_warning_once
      normalised_fields: []
    message_template: 'Argument `prefix_allowed_tokens_fn` is not a valid argument of `GenerationConfig`. It should be passed to `generate()` (or a pipeline) directly.'
    added_by: llm_miner_pass3
    added_at: '2026-05-25'

  - id: transformers_generate_arguments_not_in_config_per_field_synced_gpus
    engine: transformers
    library: transformers
    invariant_under_test: GenerationConfig.validate flags generate arguments not in config for synced_gpus
    severity: error
    native_type: transformers.GenerationConfig
    miner_source:
      path: generation_config.py
      method: validate
      line_at_scan: 123
    match:
      engine: transformers
      fields:
        transformers.sampling.synced_gpus: {present: true}
    kwargs_positive:
      synced_gpus: "some_value"
    kwargs_negative:
      synced_gpus: None
    expected_outcome:
      outcome: error
      emission_channel: logger_warning_once
      normalised_fields: []
    message_template: 'Argument `synced_gpus` is not a valid argument of `GenerationConfig`. It should be passed to `generate()` (or a pipeline) directly.'
    added_by: llm_miner_pass3
    added_at: '2026-05-25'

  - id: transformers_generate_arguments_not_in_config_per_field_assistant_model
    engine: transformers
    library: transformers
    invariant_under_test: GenerationConfig.validate flags generate arguments not in config for assistant_model
    severity: error
    native_type: transformers.GenerationConfig
    miner_source:
      path: generation_config.py
      method: validate
      line_at_scan: 123
    match:
      engine: transformers
      fields:
        transformers.sampling.assistant_model: {present: true}
    kwargs_positive:
      assistant_model: "some_value"
    kwargs_negative:
      assistant_model: None
    expected_outcome:
      outcome: error
      emission_channel: logger_warning_once
      normalised_fields: []
    message_template: 'Argument `assistant_model` is not a valid argument of `GenerationConfig`. It should be passed to `generate()` (or a pipeline) directly.'
    added_by: llm_miner_pass3
    added_at: '2026-05-25'

  - id: transformers_generate_arguments_not_in_config_per_field_streamer
    engine: transformers
    library: transformers
    invariant_under_test: GenerationConfig.validate flags generate arguments not in config for streamer
    severity: error
    native_type: transformers.GenerationConfig
    miner_source:
      path: generation_config.py
      method: validate
      line_at_scan: 123
    match:
      engine: transformers
      fields:
        transformers.sampling.streamer: {present: true}
    kwargs_positive:
      streamer: "some_value"
    kwargs_negative:
      streamer: None
    expected_outcome:
      outcome: error
      emission_channel: logger_warning_once
      normalised_fields: []
    message_template: 'Argument `streamer` is not a valid argument of `GenerationConfig`. It should be passed to `generate()` (or a pipeline) directly.'
    added_by: llm_miner_pass3
    added_at: '2026-05-25'

  - id: transformers_generate_arguments_not_in_config_per_field_negative_prompt_ids
    engine: transformers
    library: transformers
    invariant_under_test: GenerationConfig.validate flags generate arguments not in config for negative_prompt_ids
    severity: error
    native_type: transformers.GenerationConfig
    miner_source:
      path: generation_config.py
      method: validate
      line_at_scan: 123
    match:
      engine: transformers
      fields:
        transformers.sampling.negative_prompt_ids: {present: true}
    kwargs_positive:
      negative_prompt_ids: "some_value"
    kwargs_negative:
      negative_prompt_ids: None
    expected_outcome:
      outcome: error
      emission_channel: logger_warning_once
      normalised_fields: []
    message_template: 'Argument `negative_prompt_ids` is not a valid argument of `GenerationConfig`. It should be passed to `generate()` (or a pipeline) directly.'
    added_by: llm_miner_pass3
    added_at: '2026-05-25'

  - id: transformers_generate_arguments_not_in_config_per_field_negative_prompt_attention_mask
    engine: transformers
    library: transformers
    invariant_under_test: GenerationConfig.validate flags generate arguments not in config for negative_prompt_attention_mask
    severity: error
    native_type: transformers.GenerationConfig
    miner_source:
      path: generation_config.py
      method: validate
      line_at_scan: 123
    match:
      engine: transformers
      fields:
        transformers.sampling.negative_prompt_attention_mask: {present: true}
    kwargs_positive:
      negative_prompt_attention_mask: "some_value"
    kwargs_negative:
      negative_prompt_attention_mask: None
    expected_outcome:
      outcome: error
      emission_channel: logger_warning_once
      normalised_fields: []
    message_template: 'Argument `negative_prompt_attention_mask` is not a valid argument of `GenerationConfig`. It should be passed to `generate()` (or a pipeline) directly.'
    added_by: llm_miner_pass3
    added_at: '2026-05-25'
```
