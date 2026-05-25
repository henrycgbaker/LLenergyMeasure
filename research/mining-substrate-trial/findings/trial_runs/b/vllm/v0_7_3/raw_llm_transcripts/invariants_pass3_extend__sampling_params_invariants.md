# invariants_pass3_extend extraction transcript: sampling_params_invariants

- chunk_description: SamplingParams __post_init__ + _verify_args + _verify_greedy_sampling
- expected_namespaces: ['vllm.sampling']
- attempts: 1
- elapsed_sec: 17.86
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser doing PASS 3 (extension) of multi-pass
invariant extraction. PASS 1 already extracted N invariants from
vllm v0.7.3 for ONE chunk of source. Your job is to
identify ADDITIONAL invariants PASS 1 missed.

INPUT 1 - PASS 1 EXTRACTED INVARIANTS (already known; do NOT re-emit):

invariants:
- id: vllm_n_must_be_int
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.n:
        present: true
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
- id: vllm_repetition_penalty_must_be_in_range
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.repetition_penalty:
        <: 0.0
        '>': 2.0
  invariant_under_test: SamplingParams._verify_args flags repetition_penalty out of
    range
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
- id: vllm_top_k_must_be_valid
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.top_k:
        <: -1
        ==: 0
  invariant_under_test: SamplingParams._verify_args flags top_k invalid
- id: vllm_top_k_must_be_int
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.top_k:
        present: true
        type_is_not:
        - int
  invariant_under_test: SamplingParams._verify_args flags top_k not an int
- id: vllm_min_p_must_be_in_range
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.min_p:
        <: 0.0
        '>': 1.0
  invariant_under_test: SamplingParams._verify_args flags min_p out of range
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
        '>': 10
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
  invariant_under_test: SamplingParams._verify_args flags stop containing empty string
- id: vllm_stop_strings_require_detokenize_true
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.stop:
        present: true
      vllm.sampling.detokenize: false
  invariant_under_test: SamplingParams._verify_args flags stop strings without detokenize=True
- id: vllm_best_of_must_equal_n_for_delta_output_kind
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.best_of:
        present: true
        not_equal: 10
      vllm.sampling.output_kind: DELTA
  invariant_under_test: SamplingParams._verify_args flags best_of != n for output_kind=DELTA
- id: vllm_n_must_be_1_for_greedy_sampling
  severity: error
  match: null
  invariant_under_test: SamplingParams._verify_greedy_sampling flags n > 1 for greedy
    sampling


INPUT 2 - PASS 2 FLAGGED ISSUES (for context; consider whether they hint
at related misses):

(no flags)

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
        # eos_token_id is added to this by the engine
        self._all_stop_token_ids = set(self.stop_toke
...<truncated>...
```

### Response

```
invariants:
- id: vllm_best_of_must_be_greater_than_or_equal_to_n
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.best_of:
        <: vllm.sampling.n
  invariant_under_test: SamplingParams._verify_args flags best_of < n
- id: vllm_temperature_must_not_cause_numerical_errors
  severity: warning
  match:
    engine: vllm
    fields:
      vllm.sampling.temperature:
        <: _MAX_TEMP
  invariant_under_test: SamplingParams.__post_init__ flags temperature below threshold
- id: vllm_stop_token_ids_must_not_contain_empty_string
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.stop_token_ids:
        present: true
        not_in: [""]
  invariant_under_test: SamplingParams._verify_args flags stop containing empty string
- id: vllm_bad_words_must_not_contain_empty_string
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.bad_words:
        present: true
        not_in: [""]
  invariant_under_test: SamplingParams._verify_args flags bad_words containing empty string
```
