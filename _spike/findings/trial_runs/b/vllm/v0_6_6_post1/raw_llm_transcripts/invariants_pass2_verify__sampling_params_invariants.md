# invariants_pass2_verify extraction transcript: sampling_params_invariants

- chunk_description: SamplingParams __post_init__ + _verify_args + _verify_greedy_sampling
- expected_namespaces: ['vllm.sampling']
- attempts: 1
- elapsed_sec: 232.57
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser doing PASS 2 (verification) of multi-pass
invariant extraction. PASS 1 already extracted invariants from
vllm v0.6.6.post1 for ONE chunk of source. Your job is to
REVIEW each emitted invariant against the source and flag any that
look WRONG.

INPUT 1 - PASS 1 EXTRACTED INVARIANTS (the candidate list):

invariants:
- id: vllm_n_must_be_int
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.n:
        type_is_not:
        - int
  invariant_under_test: SamplingParams._verify_args flags n not an int
- id: vllm_n_must_be_at_least_1
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.n:
        <: 1
  invariant_under_test: SamplingParams._verify_args flags n < 1
- id: vllm_presence_penalty_must_be_in_range
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.presence_penalty:
        not_in:
        - -2
        - -1.5
        - -1
        - -0.5
        - 0
        - 0.5
        - 1
        - 1.5
        - 2
  invariant_under_test: SamplingParams._verify_args flags presence_penalty not in
    range [-2, 2]
- id: vllm_frequency_penalty_must_be_in_range
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.frequency_penalty:
        not_in:
        - -2
        - -1.5
        - -1
        - -0.5
        - 0
        - 0.5
        - 1
        - 1.5
        - 2
  invariant_under_test: SamplingParams._verify_args flags frequency_penalty not in
    range [-2, 2]
- id: vllm_repetition_penalty_must_be_in_range
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.repetition_penalty:
        not_in:
        - 0.1
        - 0.5
        - 1
        - 1.5
        - 2
  invariant_under_test: SamplingParams._verify_args flags repetition_penalty not in
    range (0, 2]
- id: vllm_temperature_must_be_non_negative
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.temperature:
        <: 0
  invariant_under_test: SamplingParams._verify_args flags temperature < 0
- id: vllm_top_p_must_be_in_range
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.top_p:
        not_in:
        - 0.1
        - 0.5
        - 1
  invariant_under_test: SamplingParams._verify_args flags top_p not in range (0, 1]
- id: vllm_top_k_must_be_int_and_in_range
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.top_k:
        type_is_not:
        - int
  invariant_under_test: SamplingParams._verify_args flags top_k not an int or out
    of range [-1, inf)
- id: vllm_top_k_must_be_in_range
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.top_k:
        <: -1
  invariant_under_test: SamplingParams._verify_args flags top_k out of range [-1,
    inf)
- id: vllm_min_p_must_be_in_range
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.min_p:
        not_in:
        - 0
        - 0.5
        - 1
  invariant_under_test: SamplingParams._verify_args flags min_p not in range [0, 1]
- id: vllm_max_tokens_must_be_at_least_1
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.max_tokens:
        <: 1
  invariant_under_test: SamplingParams._verify_args flags max_tokens < 1
- id: vllm_min_tokens_must_be_non_negative
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
        '>': 1
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
- id: vllm_truncate_prompt_tokens_must_be_greater_than_or_equal_to_1
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.truncate_prompt_tokens:
        <: 1
  invariant_under_test: SamplingParams._verify_args flags truncate_prompt_tokens <
    1
- id: vllm_stop_cannot_contain_empty_string
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.stop:
        present: true
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
        not_equal: 1
      vllm.sampling.output_kind: DELTA
  invariant_under_test: SamplingParams._verify_args flags best_of != n and output_kind=DELTA


INPUT 2 - THE SOURCE PASS 1 READ:

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
        else:
            self.seed = self.seed

        if self.stop is None:
            self.stop = []
        elif isinstance(self.stop, str):
            self.stop = [self.stop]
        else:
            self.stop = list(self.stop)

        if self.stop_token_ids is None:
            self.stop_token_ids = []
        else:
            self.stop_token_ids = list(self.stop_token_ids)

        if self.bad_words is None:
            self.bad_words = []
        else:
            self.bad_words = list(self.bad_words)

        self.logprobs = 1 if self.logprobs is True else self.logprobs
        self.prompt_logprobs = (1 if self.prompt_logprobs is True else
                                self.prompt_logprobs)

        # Number of characters to hold back for stop string evaluation
        # until sequence is finished.
        if self.stop and not self.include_stop_str_in_output:
            self.output_text_buffer_length = max(len(s) for s in self.stop) - 1

        self._verify_args()

        if self.temperature < _SAMPLING_EPS:
            # Zero temperature means greedy sampling.
            self.top_p = 1.0
            self.top_k = -1
            self.min_p = 0.0
            self._verify_greedy_sampling()
        # eos_token_id is added
...<truncated>...
```

### Response

```
confirmed:
- vllm_n_must_be_int
- vllm_n_must_be_at_least_1
- vllm_presence_penalty_must_be_in_range
- vllm_frequency_penalty_must_be_in_range
- vllm_repetition_penalty_must_be_in_range
- vllm_temperature_must_be_non_negative
- vllm_top_p_must_be_in_range
- vllm_top_k_must_be_int_and_in_range
- vllm_top_k_must_be_in_range
- vllm_min_p_must_be_in_range
- vllm_max_tokens_must_be_at_least_1
- vllm_min_tokens_must_be_non_negative
- vllm_min_tokens_must_be_less_than_or_equal_to_max_tokens
- vllm_logprobs_must_be_non_negative
- vllm_prompt_logprobs_must_be_non_negative
- vllm_truncate_prompt_tokens_must_be_greater_than_or_equal_to_1

flagged:
- id: vllm_stop_cannot_contain_empty_string
  reason: Source checks for `any(not stop_str for stop_str in self.stop)`, not just presence of empty string.
  fix: correct_predicate:not_in
- id: vllm_best_of_must_equal_n_to_use_output_kind_delta
  reason: Source checks for `self.best_of != self._real_n` and `self.output_kind == RequestOutputKind.DELTA`, but invariant does not account for `_real_n`.
  fix: correct_predicate:not_equal
```
