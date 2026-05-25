# invariants extraction transcript: lora_prompt_adapter_invariants

- chunk_description: vllm.{LoRAConfig, PromptAdapterConfig, TokenizerPoolConfig}.__post_init__
- expected_namespaces: ['vllm']
- attempts: 1
- elapsed_sec: 81.17
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser extracting validation invariants from
vllm library v0.7.3. An "invariant" is a rule the
library checks at runtime - typically `if <predicate>: raise ValueError(...)`
or `if <predicate>: minor_issues[...] = ...` (which surfaces as a
warning).

You will be shown ONE CHUNK of validation source. Extract every
invariant visible in this chunk.

OUTPUT FORMAT: YAML document matching EXACTLY this shape:

invariants:
- id: <snake_case_unique_id>
  engine: vllm
  library: vllm
  invariant_under_test: <one-line: what the library checks here>
  severity: <error|dormant|warning>
  native_type: vllm.GenerationConfig
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: <validate|__init__>
    line_at_scan: <approximate line number if visible>
  match:
    engine: vllm
    fields:
      vllm.<field>: <value or predicate>
  kwargs_positive:
    <field>: <value that TRIGGERS the invariant>
  kwargs_negative:
    <field>: <value that does NOT trigger>
  expected_outcome:
    outcome: <error|dormant_announced|warning>
    emission_channel: <none|logger_warning_once|logger_warning>
    normalised_fields: []
  message_template: '<the exact error/warning string from source, with {} placeholders preserved>'
  added_by: llm_miner
  added_at: '2026-05-25'

INVARIANT TYPES TO EXTRACT (one per `if ... :` block typically):

1. ERROR (raises ValueError at construction or validate()):
   - Field value not in allowed enum -> severity: error, predicate: not_in
   - Field type mismatch (e.g. `not isinstance(x, T)`) -> severity: error, predicate: type_is_not
   - Field value out of range (e.g. <= 0) -> severity: error, predicate: gt/lt
   - Cross-field invalid combo (e.g. num_return_sequences > num_beams) -> severity: error

2. DORMANT (logs warning, parameter silently ignored or normalised):
   - Sampling-only param set when do_sample=False -> severity: dormant
   - Beam-only param set when num_beams=1 -> severity: dormant
   - Cache-related dormancy -> severity: dormant

3. WARNING (logs, execution continues with user value):
   - pad_token_id < 0 -> severity: warning
   - Other non-blocking minor_issues entries -> severity: warning

PREDICATE FORMS for the `match.fields` block (use the EXACT keys shown):
- Exact value:         `vllm.field: value`
- Not in list:         `vllm.field: {present: true, not_in: [a, b]}`
- Not equal:           `vllm.field: {present: true, not_equal: value}`
- Greater than:        `vllm.field: {'>': value}`
- Less than:           `vllm.field: {'<': value}`
- Greater or equal:    `vllm.field: {'>=': value}`
- Less or equal:       `vllm.field: {'<=': value}`
- Type not in:         `vllm.field: {present: true, type_is_not: [TypeName]}`
- Presence (any value):`vllm.field: {present: true}`

CRITICAL RULES:
1. Return ONLY the YAML document. NO markdown code fences (no ```yaml).
   NO commentary, no preamble. First character must be `i` (from
   `invariants:`).
2. Extract ONLY invariants VISIBLE in the source below. Do not invent.
3. Use snake_case_with_engine_prefix for `id` (e.g. `transformers_cache_implementation_not_in_allowlist`).
4. Each `if <cond>: raise / minor_issues[...] = ` block = ONE invariant.
5. Set `severity: error` when the source has `raise ValueError(...)`.
   Set `severity: dormant` when the source assigns to `minor_issues`
   AND comments / context indicate the parameter is silently ignored.
   Set `severity: warning` when the source assigns to `minor_issues`
   AND there's no silent-ignore semantics.
6. For `kwargs_positive`: provide a concrete dict that WOULD trigger
   the invariant (so a downstream validator can replay it).
7. For `kwargs_negative`: provide a concrete dict that would NOT
   trigger (so the negative case is checkable).
8. `message_template`: the EXACT f-string literal from `raise` /
   `minor_issues[...]`. Preserve `{}` placeholders. Do NOT
   substitute concrete values.

FEW-SHOT EXAMPLES (from transformers v4.57.3 reference):

Example 1 (ERROR, enum violation):
Source: ``if self.early_stopping not in {True, False, "never"}: raise ValueError(f"`early_stopping` must be a boolean or 'never', but is {self.early_stopping}.")``
Emit:
- id: transformers_early_stopping_not_in_allowlist
  engine: transformers
  invariant_under_test: GenerationConfig.validate flags `early_stopping` not in allowlist
  severity: error
  match:
    engine: transformers
    fields:
      transformers.sampling.early_stopping: {present: true, not_in: [true, false, "never"]}
  kwargs_positive:
    early_stopping: "invalid"
  kwargs_negative:
    early_stopping: true
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "`early_stopping` must be a boolean or 'never', but is {self.early_stopping}."

Example 2 (DORMANT, sampling-only when greedy):
Source (under `if self.do_sample is False:`):
``if self.temperature is not None and self.temperature != 1.0:
    minor_issues["temperature"] = greedy_wrong_parameter_msg.format(...)``
Emit:
- id: transformers_temperature_set_when_do_sample_false
  engine: transformers
  invariant_under_test: GenerationConfig.validate flags `temperature` set when do_sample=False
  severity: dormant
  match:
    engine: transformers
    fields:
      transformers.sampling.do_sample: false
  kwargs_positive:
    do_sample: false
    temperature: 0.5
  kwargs_negative:
    do_sample: true
    temperature: 0.5
  expected_outcome:
    outcome: dormant_announced
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: "`do_sample` is set to `False`. However, `{flag_name}` is set to `{flag_value}` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `{flag_name}`."

Example 3 (CROSS-FIELD ERROR):
Source: ``if self.num_return_sequences > self.num_beams: raise ValueError(...)``
Emit:
- id: transformers_num_return_sequences_gt_num_beams
  engine: transformers
  invariant_under_test: GenerationConfig.validate flags num_return_sequences > num_beams
  severity: error
  match:
    engine: transformers
    fields:
      transformers.sampling.num_return_sequences: {'>': 1}
  kwargs_positive:
    num_beams: 2
    num_return_sequences: 3
  kwargs_negative:
    num_beams: 5
    num_return_sequences: 2
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "`num_return_sequences` ({self.num_return_sequences}) has to be smaller or equal to `num_beams` ({self.num_beams})."

=== E9 SEQUENTIAL EXTRACTION ===
You are doing a METHODICAL FILE-BY-FILE sweep of vllm v0.7.3.
This is chunk 10/10. You have already extracted
invariants from prior chunks (running notes below). For THIS chunk:

1. EXTRACT new invariants from the source below.
2. DO NOT re-emit invariants already in the running notes (dedup by
   field name + predicate kind).
3. WATCH FOR CROSS-CLASS INVARIANTS: if this chunk's source references
   a field that also appears in an earlier chunk's invariants, note
   whether it adds a cross-class constraint (e.g.
   `max_num_batched_tokens >= max_num_seqs` couples two SchedulerConfig
   fields).
4. If THIS chunk adds NO new invariants beyond the prior list, emit
   `invariants: []` (do not fabricate).

=== RUNNING NOTES (prior extractions) ===
PRIOR EXTRACTIONS (39 total invariants from earlier chunks):
  - id=vllm_n_must_be_at_least_1 sev=error field=vllm.sampling.n
  - id=vllm_n_must_be_int sev=error field=vllm.sampling.n
  - id=vllm_presence_penalty_must_be_in_range sev=error field=vllm.sampling.presence_penalty
  - id=vllm_frequency_penalty_must_be_in_range sev=error field=vllm.sampling.frequency_penalty
  - id=vllm_repetition_penalty_must_be_in_range sev=error field=vllm.sampling.repetition_penalty
  - id=vllm_temperature_must_be_non_negative sev=error field=vllm.sampling.temperature
  - id=vllm_top_p_must_be_in_range sev=error field=vllm.sampling.top_p
  - id=vllm_top_k_must_be_valid sev=error field
...<truncated>...
```

### Response

```
invariants:
- id: vllm_max_lora_rank_not_in_allowlist
  engine: vllm
  invariant_under_test: LoRAConfig flags max_lora_rank not in allowlist
  severity: error
  native_type: vllm.LoRAConfig
  miner_source:
    path: configuration_utils.py
    method: __post_init__
    line_at_scan: 5
  match:
    engine: vllm
    fields:
      vllm.max_lora_rank: {present: true, not_in: [8, 16, 32, 64, 128, 256]}
  kwargs_positive:
    max_lora_rank: 512
  kwargs_negative:
    max_lora_rank: 128
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "max_lora_rank ({self.max_lora_rank}) must be one of {possible_max_ranks}."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: vllm_lora_extra_vocab_size_not_in_allowlist
  engine: vllm
  invariant_under_test: LoRAConfig flags lora_extra_vocab_size not in allowlist
  severity: error
  native_type: vllm.LoRAConfig
  miner_source:
    path: configuration_utils.py
    method: __post_init__
    line_at_scan: 10
  match:
    engine: vllm
    fields:
      vllm.lora_extra_vocab_size: {present: true, not_in: [0, 256, 512]}
  kwargs_positive:
    lora_extra_vocab_size: 1024
  kwargs_negative:
    lora_extra_vocab_size: 256
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "lora_extra_vocab_size ({self.lora_extra_vocab_size}) must be one of {possible_lora_extra_vocab_size}."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: vllm_max_loras_must_be_at_least_1
  engine: vllm
  invariant_under_test: LoRAConfig flags max_loras must be at least 1
  severity: error
  native_type: vllm.LoRAConfig
  miner_source:
    path: configuration_utils.py
    method: __post_init__
    line_at_scan: 15
  match:
    engine: vllm
    fields:
      vllm.max_loras: {'<': 1}
  kwargs_positive:
    max_loras: 0
  kwargs_negative:
    max_loras: 2
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "max_loras ({self.max_loras}) must be >= 1."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: vllm_max_cpu_loras_must_be_greater_than_or_equal_to_max_loras
  engine: vllm
  invariant_under_test: LoRAConfig flags max_cpu_loras must be greater than or equal to max_loras
  severity: error
  native_type: vllm.LoRAConfig
  miner_source:
    path: configuration_utils.py
    method: __post_init__
    line_at_scan: 20
  match:
    engine: vllm
    fields:
      vllm.max_cpu_loras: {'<': 'vllm.max_loras'}
  kwargs_positive:
    max_loras: 2
    max_cpu_loras: 1
  kwargs_negative:
    max_loras: 2
    max_cpu_loras: 3
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "max_cpu_loras ({self.max_cpu_loras}) must be >= max_loras ({self.max_loras})."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: vllm_max_prompt_adapters_must_be_at_least_1
  engine: vllm
  invariant_under_test: PromptAdapterConfig flags max_prompt_adapters must be at least 1
  severity: error
  native_type: vllm.PromptAdapterConfig
  miner_source:
    path: configuration_utils.py
    method: __post_init__
    line_at_scan: 25
  match:
    engine: vllm
    fields:
      vllm.max_prompt_adapters: {'<': 1}
  kwargs_positive:
    max_prompt_adapters: 0
  kwargs_negative:
    max_prompt_adapters: 2
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "max_prompt_adapters ({self.max_prompt_adapters}) must be >= 1."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: vllm_max_prompt_adapter_token_must_be_set
  engine: vllm
  invariant_under_test: PromptAdapterConfig flags max_prompt_adapter_token must be set
  severity: error
  native_type: vllm.PromptAdapterConfig
  miner_source:
    path: configuration_utils.py
    method: __post_init__
    line_at_scan: 30
  match:
    engine: vllm
    fields:
      vllm.max_prompt_adapter_token: {present: true, not_equal: 0}
  kwargs_positive:
    max_prompt_adapter_token: 0
  kwargs_negative:
    max_prompt_adapter_token: 2
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "max_prompt_adapter_token must be set."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: vllm_pool_type_not_in_allowlist
  engine: vllm
  invariant_under_test: TokenizerPoolConfig flags pool_type not in allowlist
  severity: error
  native_type: vllm.TokenizerPoolConfig
  miner_source:
    path: configuration_utils.py
    method: __post_init__
    line_at_scan: 35
  match:
    engine: vllm
    fields:
      vllm.pool_type: {present: true, not_in: ["ray"]}
  kwargs_positive:
    pool_type: "invalid"
  kwargs_negative:
    pool_type: "ray"
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "Unknown pool type: {self.pool_type}"
  added_by: llm_miner
  added_at: '2026-05-25'

- id: vllm_extra_config_must_be_dict
  engine: vllm
  invariant_under_test: TokenizerPoolConfig flags extra_config must be dict
  severity: error
  native_type: vllm.TokenizerPoolConfig
  miner_source:
    path: configuration_utils.py
    method: __post_init__
    line_at_scan: 40
  match:
    engine: vllm
    fields:
      vllm.extra_config: {present: true, type_is_not: [dict]}
  kwargs_positive:
    extra_config: "invalid"
  kwargs_negative:
    extra_config: {}
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "extra_config must be a dictionary."
  added_by: llm_miner
  added_at: '2026-05-25'
```
