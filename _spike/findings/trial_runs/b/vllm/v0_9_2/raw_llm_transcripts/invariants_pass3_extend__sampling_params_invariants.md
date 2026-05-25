# invariants_pass3_extend extraction transcript: sampling_params_invariants

- chunk_description: SamplingParams __post_init__ + _verify_args + _verify_greedy_sampling
- expected_namespaces: ['vllm.sampling']
- attempts: 1
- elapsed_sec: 259.06
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser doing PASS 3 (extension) of multi-pass
invariant extraction. PASS 1 already extracted N invariants from
vllm v0.9.2 for ONE chunk of source. Your job is to
identify ADDITIONAL invariants PASS 1 missed.

INPUT 1 - PASS 1 EXTRACTED INVARIANTS (already known; do NOT re-emit):

invariants:
- id: vllm_n_must_be_at_least_1
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.n:
        <: 1
  invariant_under_test: SamplingParams._verify_args flags n < 1
- id: vllm_best_of_must_be_greater_than_or_equal_to_n
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.best_of:
        <: vllm.sampling.n
  invariant_under_test: SamplingParams._verify_args flags best_of < n
- id: vllm_presence_penalty_must_be_in_range
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.presence_penalty:
        <: -2.0
        '>': 2.0
  invariant_under_test: SamplingParams._verify_args flags presence_penalty out of
    range
- id: vllm_frequency_penalty_must_be_in_range
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.frequency_penalty:
        <: -2.0
        '>': 2.0
  invariant_under_test: SamplingParams._verify_args flags frequency_penalty out of
    range
- id: vllm_repetition_penalty_must_be_greater_than_zero
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.repetition_penalty:
        <=: 0.0
  invariant_under_test: SamplingParams._verify_args flags repetition_penalty <= 0.0
- id: vllm_temperature_must_be_non_negative
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.temperature:
        <: 0.0
  invariant_under_test: SamplingParams._verify_args flags temperature < 0.0
- id: vllm_top_p_must_be_in_range
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.top_p:
        <: 0.0
        '>': 1.0
  invariant_under_test: SamplingParams._verify_args flags top_p out of range
- id: vllm_top_k_must_be_non_negative_integer
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.top_k:
        <: -1
        type_is_not:
        - int
  invariant_under_test: SamplingParams._verify_args flags top_k < -1 or not integer
- id: vllm_min_p_must_be_in_range
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.min_p:
        <: 0.0
        '>': 1.0
  invariant_under_test: SamplingParams._verify_args flags min_p out of range
- id: vllm_max_tokens_must_be_greater_than_zero
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.max_tokens:
        <: 1
  invariant_under_test: SamplingParams._verify_args flags max_tokens < 1
- id: vllm_min_tokens_must_be_greater_than_or_equal_to_zero
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.min_tokens:
        <: 0
  invariant_under_test: SamplingParams._verify_args flags min_tokens < 0
- id: vllm_min_tokens_must_be_less_than_or_equal_to_max_tokens
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.min_tokens:
        '>': vllm.sampling.max_tokens
  invariant_under_test: SamplingParams._verify_args flags min_tokens > max_tokens
- id: vllm_logprobs_must_be_non_negative
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.logprobs:
        <: 0
  invariant_under_test: SamplingParams._verify_args flags logprobs < 0
- id: vllm_prompt_logprobs_must_be_non_negative
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.prompt_logprobs:
        <: 0
  invariant_under_test: SamplingParams._verify_args flags prompt_logprobs < 0
- id: vllm_truncate_prompt_tokens_must_be_greater_than_or_equal_to_one
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.truncate_prompt_tokens:
        <: 1
  invariant_under_test: SamplingParams._verify_args flags truncate_prompt_tokens <
    1
- id: vllm_stop_token_ids_must_contain_only_integers
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.stop_token_ids:
        present: true
        type_is_not:
        - int
  invariant_under_test: SamplingParams._verify_args flags stop_token_ids contains
    non-integers
- id: vllm_stop_cannot_contain_empty_string
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.stop:
        present: true
        not_equal: ''
  invariant_under_test: SamplingParams._verify_args flags stop contains empty string
- id: vllm_stop_strings_are_only_supported_when_detokenize_is_true
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.stop:
        present: true
      vllm.sampling.detokenize: false
  invariant_under_test: SamplingParams._verify_args flags stop and detokenize=False
- id: vllm_best_of_must_equal_n_to_use_output_kind_delta
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.best_of:
        present: true
        not_equal: vllm.sampling.n
      vllm.sampling.output_kind: DELTA
  invariant_under_test: SamplingParams._verify_args flags best_of != n and output_kind=DELTA


INPUT 2 - PASS 2 FLAGGED ISSUES (for context; consider whether they hint
at related misses):

flagged:
- id: vllm_top_k_must_be_non_negative_integer
  reason: source allows -1 as disabled, but invariant does not
  fix: correct_predicate:range
- id: vllm_stop_strings_are_only_supported_when_detokenize_is_true
  reason: source also checks for presence of stop strings, but invariant does not
  fix: correct_kwargs_positive


INPUT 3 - THE SOURCE PASS 1 READ:

=== CONTEXT ===
vllm.SamplingParams enforces validation via `_verify_args` called from `__post_init__`. The `_verify_args` body contains the bulk of the `if not <cond>: raise ValueError(...)` checks. `__post_init__` also adds dormant warning paths (e.g. temperature below threshold). Use namespace `vllm.sampling`.

=== SOURCE: SamplingParams.__post_init__ ===
    def __post_init__(self) -> None:
        # how we deal with `best_of``:
        # if `best_of`` is not set, we default to `n`;
        # if `best_of`` is set, we set `n`` to `best_of`,
        # and set `_real_n`` to the original `n`.
        # when we return the result, we will check
        # if we need to return `n` or `_real_n` results
        if self.best_of:
            if self.best_of < self.n:
                raise ValueError(
                    f"best_of must be greater than or equal to n, "
                    f"got n={self.n} and best_of={self.best_of}.")
            if not self._real_n:
                self._real_n = self.n
                self.n = self.best_of

        if 0 < self.temperature < _MAX_TEMP:
            logger.warning(
                "temperature %s is less than %s, which may cause numerical "
                "errors nan or inf in tensors. We have maxed it out to %s.",
                self.temperature, _MAX_TEMP, _MAX_TEMP)
            self.temperature = max(self.temperature, _MAX_TEMP)

        if self.seed == -1:
            self.seed = None

        if self.stop is None:
            self.stop = []
        elif isinstance(self.stop, str):
            self.stop = [self.stop]

        if self.stop_token_ids is None:
            self.stop_token_ids = []

        if self.bad_words is None:
            self.bad_words = []

        if self.logprobs is True:
            self.logprobs = 1

        if self.prompt_logprobs is True:
            self.prompt_logprobs = 1

        # Number of characters to hold back for stop string evaluation
        # until sequence is finished.
        if self.stop and not self.include_stop_str_in_output:
            self.output_text_buffer_length = max(len(s) for s in self.stop) - 1

        self._verify_args()

        if self.temperature < _SAMPLING_EPS:
            # Zero temperature means greedy sampling.
            self.top_p = 1.0
            self.top_k = 0
            self.min_p = 0.0
            self._verify_greedy_sampling()

        # eos_token_id is added to this by the engine
        self._al
...<truncated>...
```

### Response

```
invariants:
- id: vllm_temperature_must_be_maxed_out_for_numerical_stability
  engine: vllm
  library: vllm
  invariant_under_test: temperature is less than _MAX_TEMP, which may cause numerical errors nan or inf in tensors.
  severity: warning
  native_type: vllm.SamplingParams
  miner_source:
    path: SamplingParams.__post_init__
    method: __post_init__
    line_at_scan: 10
  match:
    engine: vllm
    fields:
      vllm.sampling.temperature: {'<': _MAX_TEMP}
  kwargs_positive:
    temperature: 0.5
  kwargs_negative:
    temperature: 1.0
  expected_outcome:
    outcome: warning
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: 'temperature %s is less than %s, which may cause numerical errors nan or inf in tensors. We have maxed it out to %s.'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
- id: vllm_n_must_be_one_for_greedy_sampling
  engine: vllm
  library: vllm
  invariant_under_test: n must be 1 when using greedy sampling.
  severity: error
  native_type: vllm.SamplingParams
  miner_source:
    path: SamplingParams._verify_greedy_sampling
    method: _verify_greedy_sampling
    line_at_scan: 5
  match:
    engine: vllm
    fields:
      vllm.sampling.n: {'>': 1}
  kwargs_positive:
    n: 2
  kwargs_negative:
    n: 1
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: 'n must be 1 when using greedy sampling, got %s.'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
- id: vllm_stop_token_ids_must_contain_only_integers
  engine: vllm
  library: vllm
  invariant_under_test: stop_token_ids must contain only integers.
  severity: error
  native_type: vllm.SamplingParams
  miner_source:
    path: SamplingParams._verify_args
    method: _verify_args
    line_at_scan: 35
  match:
    engine: vllm
    fields:
      vllm.sampling.stop_token_ids: {present: true, type_is_not: [int]}
  kwargs_positive:
    stop_token_ids: ['a', 'b']
  kwargs_negative:
    stop_token_ids: [1, 2]
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: 'stop_token_ids must contain only integers, got %s.'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
- id: vllm_stop_cannot_contain_empty_string
  engine: vllm
  library: vllm
  invariant_under_test: stop cannot contain an empty string.
  severity: error
  native_type: vllm.SamplingParams
  miner_source:
    path: SamplingParams._verify_args
    method: _verify_args
    line_at_scan: 40
  match:
    engine: vllm
    fields:
      vllm.sampling.stop: {present: true, not_equal: ''}
  kwargs_positive:
    stop: ['']
  kwargs_negative:
    stop: ['a', 'b']
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: 'stop cannot contain an empty string.'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
```
