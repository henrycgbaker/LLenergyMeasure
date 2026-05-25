# invariants extraction transcript: sampling_params_invariants

- chunk_description: SamplingParams __post_init__ + _verify_args + _verify_greedy_sampling
- expected_namespaces: ['vllm.sampling']
- attempts: 1
- elapsed_sec: 197.09
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
      vllm.sampling.<field>: <value or predicate>
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
- Exact value:         `vllm.sampling.field: value`
- Not in list:         `vllm.sampling.field: {present: true, not_in: [a, b]}`
- Not equal:           `vllm.sampling.field: {present: true, not_equal: value}`
- Greater than:        `vllm.sampling.field: {'>': value}`
- Less than:           `vllm.sampling.field: {'<': value}`
- Greater or equal:    `vllm.sampling.field: {'>=': value}`
- Less or equal:       `vllm.sampling.field: {'<=': value}`
- Type not in:         `vllm.sampling.field: {present: true, type_is_not: [TypeName]}`
- Presence (any value):`vllm.sampling.field: {present: true}`

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
This is chunk 1/10. You have already extracted
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
(no invariants extracted yet; this is the first chunk)

=== END RUNNING NOTES ===

Now apply the standard invariant extraction prompt below to THIS chunk's
source.


=== CONTEXT ===
vllm.SamplingParams enforces validation via `_verify_args` called from `__post_init__`. The `_verify_args` body contains the bulk of the `if not <cond>: raise ValueError(...)` checks. `__post_init__` also adds dormant warning paths (e.g. temperature below threshold). Use namespace `vllm.sampling`.

=== SOURCE: SamplingParams.__post_init__ ===
    def __post_init__(self) -> None:
        # how we deal
...<truncated>...
```

### Response

```
invariants:
- id: vllm_n_must_be_at_least_1
  engine: vllm
  invariant_under_test: SamplingParams._verify_args flags n < 1
  severity: error
  native_type: vllm.SamplingParams
  miner_source:
    path: sampling.py
    method: _verify_args
    line_at_scan: 10
  match:
    engine: vllm
    fields:
      vllm.sampling.n: {'<': 1}
  kwargs_positive:
    n: 0
  kwargs_negative:
    n: 2
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "n must be at least 1, got {self.n}."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: vllm_n_must_be_int
  engine: vllm
  invariant_under_test: SamplingParams._verify_args flags n not int
  severity: error
  native_type: vllm.SamplingParams
  miner_source:
    path: sampling.py
    method: _verify_args
    line_at_scan: 5
  match:
    engine: vllm
    fields:
      vllm.sampling.n: {present: true, type_is_not: [int]}
  kwargs_positive:
    n: "a"
  kwargs_negative:
    n: 2
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "n must be an int, but is of {type(self.n)}"
  added_by: llm_miner
  added_at: '2026-05-25'

- id: vllm_presence_penalty_must_be_in_range
  engine: vllm
  invariant_under_test: SamplingParams._verify_args flags presence_penalty out of range
  severity: error
  native_type: vllm.SamplingParams
  miner_source:
    path: sampling.py
    method: _verify_args
    line_at_scan: 15
  match:
    engine: vllm
    fields:
      vllm.sampling.presence_penalty: {'<': -2.0, '>': 2.0}
  kwargs_positive:
    presence_penalty: 3.0
  kwargs_negative:
    presence_penalty: 1.5
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "presence_penalty must be in [-2, 2], got {self.presence_penalty}."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: vllm_frequency_penalty_must_be_in_range
  engine: vllm
  invariant_under_test: SamplingParams._verify_args flags frequency_penalty out of range
  severity: error
  native_type: vllm.SamplingParams
  miner_source:
    path: sampling.py
    method: _verify_args
    line_at_scan: 20
  match:
    engine: vllm
    fields:
      vllm.sampling.frequency_penalty: {'<': -2.0, '>': 2.0}
  kwargs_positive:
    frequency_penalty: 3.0
  kwargs_negative:
    frequency_penalty: 1.5
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "frequency_penalty must be in [-2, 2], got {self.frequency_penalty}."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: vllm_repetition_penalty_must_be_in_range
  engine: vllm
  invariant_under_test: SamplingParams._verify_args flags repetition_penalty out of range
  severity: error
  native_type: vllm.SamplingParams
  miner_source:
    path: sampling.py
    method: _verify_args
    line_at_scan: 25
  match:
    engine: vllm
    fields:
      vllm.sampling.repetition_penalty: {'<': 0.0, '>': 2.0}
  kwargs_positive:
    repetition_penalty: -1.0
  kwargs_negative:
    repetition_penalty: 1.5
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "repetition_penalty must be in (0, 2], got {self.repetition_penalty}."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: vllm_temperature_must_be_non_negative
  engine: vllm
  invariant_under_test: SamplingParams._verify_args flags temperature < 0.0
  severity: error
  native_type: vllm.SamplingParams
  miner_source:
    path: sampling.py
    method: _verify_args
    line_at_scan: 30
  match:
    engine: vllm
    fields:
      vllm.sampling.temperature: {'<': 0.0}
  kwargs_positive:
    temperature: -1.0
  kwargs_negative:
    temperature: 1.5
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "temperature must be non-negative, got {self.temperature}."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: vllm_top_p_must_be_in_range
  engine: vllm
  invariant_under_test: SamplingParams._verify_args flags top_p out of range
  severity: error
  native_type: vllm.SamplingParams
  miner_source:
    path: sampling.py
    method: _verify_args
    line_at_scan: 35
  match:
    engine: vllm
    fields:
      vllm.sampling.top_p: {'<': 0.0, '>': 1.0}
  kwargs_positive:
    top_p: -1.0
  kwargs_negative:
    top_p: 0.5
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "top_p must be in (0, 1], got {self.top_p}."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: vllm_top_k_must_be_valid
  engine: vllm
  invariant_under_test: SamplingParams._verify_args flags top_k invalid
  severity: error
  native_type: vllm.SamplingParams
  miner_source:
    path: sampling.py
    method: _verify_args
    line_at_scan: 40
  match:
    engine: vllm
    fields:
      vllm.sampling.top_k: {present: true, not_equal: -1}
  kwargs_positive:
    top_k: 0
  kwargs_negative:
    top_k: -1
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "top_k must be -1 (disable), or at least 1, got {self.top_k}."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: vllm_top_k_must_be_int
  engine: vllm
  invariant_under_test: SamplingParams._verify_args flags top_k not int
  severity: error
  native_type: vllm.SamplingParams
  miner_source:
    path: sampling.py
    method: _verify_args
    line_at_scan: 45
  match:
    engine: vllm
    fields:
      vllm.sampling.top_k: {present: true, type_is_not: [int]}
  kwargs_positive:
    top_k: "a"
  kwargs_negative:
    top_k: -1
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "top_k must be an integer, got {type(self.top_k).__name__}"
  added_by: llm_miner
  added_at: '2026-05-25'

- id: vllm_min_p_must_be_in_range
  engine: vllm
  invariant_under_test: SamplingParams._verify_args flags min_p out of range
  severity: error
  native_type: vllm.SamplingParams
  miner_source:
    path: sampling.py
    method: _verify_args
    line_at_scan: 50
  match:
    engine: vllm
    fields:
      vllm.sampling.min_p: {'<': 0.0, '>': 1.0}
  kwargs_positive:
    min_p: -1.0
  kwargs_negative:
    min_p: 0.5
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "min_p must be in [0, 1], got {self.min_p}."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: vllm_max_tokens_must_be_at_least_1
  engine: vllm
  invariant_under_test: SamplingParams._verify_args flags max_tokens < 1
  severity: error
  native_type: vllm.SamplingParams
  miner_source:
    path: sampling.py
    method: _verify_args
    line_at_scan: 55
  match:
    engine: vllm
    fields:
      vllm.sampling.max_tokens: {'<': 1}
  kwargs_positive:
    max_tokens: 0
  kwargs_negative:
    max_tokens: 2
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "max_tokens must be at least 1, got {self.max_tokens}."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: vllm_min_tokens_must_be_non_negative
  engine: vllm
  invariant_under_test: SamplingParams._verify_args flags min_tokens < 0
  severity: error
  native_type: vllm.SamplingParams
  miner_source:
    path: sampling.py
    method: _verify_args
    line_at_scan: 60
  match:
    engine: vllm
    fields:
      vllm.sampling.min_tokens: {'<': 0}
  kwargs_positive:
    min_tokens: -1
  kwargs_negative:
    min_tokens: 2
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "min_tokens must be greater than or equal to 0, got {self.min_tokens}."
  added_by: llm_miner
  added_at: '2026-05-25'

- id: vllm_min_tokens_must_be_less_than_or_equal_to_max_tokens
  engine: vllm
  invariant_under_test: SamplingParams._verify_args flags min_tokens > max_tokens
  severity: error
  native_type: vllm.SamplingParams
  mine
...<truncated>...
```
