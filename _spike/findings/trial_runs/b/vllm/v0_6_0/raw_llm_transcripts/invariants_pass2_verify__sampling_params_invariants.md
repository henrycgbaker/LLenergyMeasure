# invariants_pass2_verify extraction transcript: sampling_params_invariants

- chunk_description: SamplingParams __post_init__ + _verify_args + _verify_greedy_sampling
- expected_namespaces: ['vllm.sampling']
- attempts: 1
- elapsed_sec: 409.98
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser doing PASS 2 (verification) of multi-pass
invariant extraction. PASS 1 already extracted invariants from
vllm v0.6.0 for ONE chunk of source. Your job is to
REVIEW each emitted invariant against the source and flag any that
look WRONG.

INPUT 1 - PASS 1 EXTRACTED INVARIANTS (the candidate list):

invariants:
- id: vllm_n_lt_1
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.n:
        <: 1
  invariant_under_test: SamplingParams._verify_args flags n < 1
- id: vllm_best_of_lt_n
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.best_of:
        <: vllm.sampling.n
  invariant_under_test: SamplingParams._verify_args flags best_of < n
- id: vllm_presence_penalty_out_of_range
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.presence_penalty:
        <: -2.0
        '>': 2.0
  invariant_under_test: SamplingParams._verify_args flags presence_penalty out of
    range
- id: vllm_frequency_penalty_out_of_range
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.frequency_penalty:
        <: -2.0
        '>': 2.0
  invariant_under_test: SamplingParams._verify_args flags frequency_penalty out of
    range
- id: vllm_repetition_penalty_out_of_range
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.repetition_penalty:
        <: 0.0
        '>': 2.0
  invariant_under_test: SamplingParams._verify_args flags repetition_penalty out of
    range
- id: vllm_temperature_lt_0
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.temperature:
        <: 0.0
  invariant_under_test: SamplingParams._verify_args flags temperature < 0
- id: vllm_top_p_out_of_range
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.top_p:
        <: 0.0
        '>': 1.0
  invariant_under_test: SamplingParams._verify_args flags top_p out of range
- id: vllm_top_k_out_of_range
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.top_k:
        <: -1
        '>': 0
  invariant_under_test: SamplingParams._verify_args flags top_k out of range
- id: vllm_top_k_not_int
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.top_k:
        type_is_not:
        - int
  invariant_under_test: SamplingParams._verify_args flags top_k not int
- id: vllm_min_p_out_of_range
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.min_p:
        <: 0.0
        '>': 1.0
  invariant_under_test: SamplingParams._verify_args flags min_p out of range
- id: vllm_max_tokens_lt_1
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.max_tokens:
        <: 1
  invariant_under_test: SamplingParams._verify_args flags max_tokens < 1
- id: vllm_min_tokens_lt_0
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.min_tokens:
        <: 0
  invariant_under_test: SamplingParams._verify_args flags min_tokens < 0
- id: vllm_min_tokens_gt_max_tokens
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.min_tokens:
        '>': vllm.sampling.max_tokens
  invariant_under_test: SamplingParams._verify_args flags min_tokens > max_tokens
- id: vllm_logprobs_lt_0
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.logprobs:
        <: 0
  invariant_under_test: SamplingParams._verify_args flags logprobs < 0
- id: vllm_prompt_logprobs_lt_0
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.prompt_logprobs:
        <: 0
  invariant_under_test: SamplingParams._verify_args flags prompt_logprobs < 0
- id: vllm_truncate_prompt_tokens_lt_1
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.truncate_prompt_tokens:
        <: 1
  invariant_under_test: SamplingParams._verify_args flags truncate_prompt_tokens <
    1
- id: vllm_stop_empty_string
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.stop:
        present: true
  invariant_under_test: SamplingParams._verify_args flags stop contains empty string
- id: vllm_stop_without_detokenize
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.stop:
        present: true
      vllm.sampling.detokenize: false
  invariant_under_test: SamplingParams._verify_args flags stop without detokenize
- id: vllm_best_of_gt_1_in_greedy_sampling
  severity: error
  match:
    engine: vllm
    fields:
      vllm.sampling.best_of:
        '>': 1
  invariant_under_test: SamplingParams._verify_greedy_sampling flags best_of > 1 in
    greedy sampling
- id: vllm_temperature_lt_max_temp
  severity: warning
  match:
    engine: vllm
    fields: vllm
  invariant_under_test: SamplingParams.__post_init__ flags temperature < _MAX_TEMP


INPUT 2 - THE SOURCE PASS 1 READ:

=== CONTEXT ===
vllm.SamplingParams enforces validation via `_verify_args` called from `__post_init__`. The `_verify_args` body contains the bulk of the `if not <cond>: raise ValueError(...)` checks. `__post_init__` also adds dormant warning paths (e.g. temperature below threshold). Use namespace `vllm.sampling`.

=== SOURCE: SamplingParams.__post_init__ ===
    def __post_init__(self) -> None:
        self.best_of = self.best_of or self.n
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
        self.logprobs = 1 if self.logprobs is True else self.logprobs
        self.prompt_logprobs = (1 if self.prompt_logprobs is True else
                                self.prompt_logprobs)

        # Number of characters to hold back for stop string evaluation
        # until sequence is finished.
        if self.stop and not self.include_stop_str_in_output:
            self.output_text_buffer_length = max(len(s) for s in self.stop) - 1

        self._verify_args()
        if self.use_beam_search:
            self._verify_beam_search()
        else:
            self._verify_non_beam_search()
            if self.temperature < _SAMPLING_EPS:
                # Zero temperature means greedy sampling.
                self.top_p = 1.0
                self.top_k = -1
                self.min_p = 0.0
                self._verify_greedy_sampling()
        # eos_token_id is added to this by the engine
        self._all_stop_token_ids = set(self.stop_token_ids)


=== SOURCE: SamplingParams._verify_args ===
    def _verify_args(self) -> None:
        if self.n < 1:
            raise ValueError(f"n must be at least 1, got {self.n}.")
        assert isinstance(self.best_of, int)
        if self.best_of < self.n:
            raise ValueError(f"best_of must be greater than or equal to n, "
                             f"got n={self.n} and best_of={self.best_of}.")
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError("presence_penalty must be in [-2, 2], got "
                             f"{self.presence_penalty}.")
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError("frequency_penalty must be in [-2, 2], got "
                             f"{self.frequency_penalty}.")
        if not 0.0 < self.repetition_penalty <= 2.0:
            raise ValueError("repetition_penalty must be in (0, 2], got "
                             f"{self.repetition_penalty}.")
    
...<truncated>...
```

### Response

```
confirmed:
- vllm_n_lt_1
- vllm_best_of_lt_n
- vllm_presence_penalty_out_of_range
- vllm_frequency_penalty_out_of_range
- vllm_repetition_penalty_out_of_range
- vllm_temperature_lt_0
- vllm_top_p_out_of_range
- vllm_top_k_out_of_range
- vllm_top_k_not_int
- vllm_min_p_out_of_range
- vllm_max_tokens_lt_1
- vllm_min_tokens_lt_0
- vllm_min_tokens_gt_max_tokens
- vllm_logprobs_lt_0
- vllm_prompt_logprobs_lt_0
- vllm_truncate_prompt_tokens_lt_1
- vllm_stop_empty_string
- vllm_stop_without_detokenize

flagged:
- id: vllm_best_of_gt_1_in_greedy_sampling
  reason: Source has `if self.best_of > 1` but invariant says `best_of '>': 1`.
  fix: correct_predicate:exact
- id: vllm_temperature_lt_max_temp
  reason: Severity is warning, but source does not have a corresponding warning.
  fix: drop
```
