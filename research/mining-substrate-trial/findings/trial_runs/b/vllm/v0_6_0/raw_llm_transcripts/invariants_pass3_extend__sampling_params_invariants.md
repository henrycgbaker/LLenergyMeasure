# invariants_pass3_extend extraction transcript: sampling_params_invariants

- chunk_description: SamplingParams __post_init__ + _verify_args + _verify_greedy_sampling
- expected_namespaces: ['vllm.sampling']
- attempts: 1
- elapsed_sec: 236.82
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser doing PASS 3 (extension) of multi-pass
invariant extraction. PASS 1 already extracted N invariants from
vllm v0.6.0 for ONE chunk of source. Your job is to
identify ADDITIONAL invariants PASS 1 missed.

INPUT 1 - PASS 1 EXTRACTED INVARIANTS (already known; do NOT re-emit):

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


INPUT 2 - PASS 2 FLAGGED ISSUES (for context; consider whether they hint
at related misses):

(no flags)

INPUT 3 - THE SOURCE PASS 1 READ:

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
            raise ValueError("repetition_penalty must b
...<truncated>...
```

### Response

```
invariants:
- id: vllm_temperature_set_to_max_temp_if_lt_max_temp
  engine: vllm
  library: vllm
  invariant_under_test: SamplingParams.__post_init__ sets temperature to _MAX_TEMP if less than _MAX_TEMP
  severity: warning
  native_type: vllm.SamplingParams
  miner_source:
    path: sampling_params.py
    method: __post_init__
    line_at_scan: 10
  match:
    engine: vllm
    fields:
      vllm.sampling.temperature:
        <: _MAX_TEMP
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
- id: vllm_seed_set_to_none_if_eq_minus_one
  engine: vllm
  library: vllm
  invariant_under_test: SamplingParams.__post_init__ sets seed to None if equal to -1
  severity: dormant
  native_type: vllm.SamplingParams
  miner_source:
    path: sampling_params.py
    method: __post_init__
    line_at_scan: 15
  match:
    engine: vllm
    fields:
      vllm.sampling.seed:
        <: -1
  kwargs_positive:
    seed: -1
  kwargs_negative:
    seed: 0
  expected_outcome:
    outcome: dormant_announced
    emission_channel: none
    normalised_fields: []
  message_template: ''
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
- id: vllm_stop_token_ids_set_to_list_if_not_none
  engine: vllm
  library: vllm
  invariant_under_test: SamplingParams.__post_init__ sets stop_token_ids to list if not None
  severity: dormant
  native_type: vllm.SamplingParams
  miner_source:
    path: sampling_params.py
    method: __post_init__
    line_at_scan: 25
  match:
    engine: vllm
    fields:
      vllm.sampling.stop_token_ids:
        present: true
  kwargs_positive:
    stop_token_ids: [1, 2]
  kwargs_negative:
    stop_token_ids: None
  expected_outcome:
    outcome: dormant_announced
    emission_channel: none
    normalised_fields: []
  message_template: ''
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
- id: vllm_logprobs_set_to_one_if_true
  engine: vllm
  library: vllm
  invariant_under_test: SamplingParams.__post_init__ sets logprobs to 1 if True
  severity: dormant
  native_type: vllm.SamplingParams
  miner_source:
    path: sampling_params.py
    method: __post_init__
    line_at_scan: 30
  match:
    engine: vllm
    fields:
      vllm.sampling.logprobs:
        <: true
  kwargs_positive:
    logprobs: True
  kwargs_negative:
    logprobs: False
  expected_outcome:
    outcome: dormant_announced
    emission_channel: none
    normalised_fields: []
  message_template: ''
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
- id: vllm_prompt_logprobs_set_to_one_if_true
  engine: vllm
  library: vllm
  invariant_under_test: SamplingParams.__post_init__ sets prompt_logprobs to 1 if True
  severity: dormant
  native_type: vllm.SamplingParams
  miner_source:
    path: sampling_params.py
    method: __post_init__
    line_at_scan: 35
  match:
    engine: vllm
    fields:
      vllm.sampling.prompt_logprobs:
        <: true
  kwargs_positive:
    prompt_logprobs: True
  kwargs_negative:
    prompt_logprobs: False
  expected_outcome:
    outcome: dormant_announced
    emission_channel: none
    normalised_fields: []
  message_template: ''
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
- id: vllm_output_text_buffer_length_set_if_stop_and_not_include_stop_str_in_output
  engine: vllm
  library: vllm
  invariant_under_test: SamplingParams.__post_init__ sets output_text_buffer_length if stop and not include_stop_str_in_output
  severity: dormant
  native_type: vllm.SamplingParams
  miner_source:
    path: sampling_params.py
    method: __post_init__
    line_at_scan: 40
  match:
    engine: vllm
    fields:
      vllm.sampling.stop:
        present: true
      vllm.sampling.include_stop_str_in_output:
        <: false
  kwargs_positive:
    stop: ['stop']
    include_stop_str_in_output: False
  kwargs_negative:
    stop: []
    include_stop_str_in_output: True
  expected_outcome:
    outcome: dormant_announced
    emission_channel: none
    normalised_fields: []
  message_template: ''
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
- id: vllm_top_p_set_to_one_if_temperature_lt_sampling_eps
  engine: vllm
  library: vllm
  invariant_under_test: SamplingParams.__post_init__ sets top_p to 1 if temperature less than _SAMPLING_EPS
  severity: dormant
  native_type: vllm.SamplingParams
  miner_source:
    path: sampling_params.py
    method: __post_init__
    line_at_scan: 50
  match:
    engine: vllm
    fields:
      vllm.sampling.temperature:
        <: _SAMPLING_EPS
  kwargs_positive:
    temperature: 0.1
  kwargs_negative:
    temperature: 1.0
  expected_outcome:
    outcome: dormant_announced
    emission_channel: none
    normalised_fields: []
  message_template: ''
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
- id: vllm_top_k_set_to_minus_one_if_temperature_lt_sampling_eps
  engine: vllm
  library: vllm
  invariant_under_test: SamplingParams.__post_init__ sets top_k to -1 if temperature less than _SAMPLING_EPS
  severity: dormant
  native_type: vllm.SamplingParams
  miner_source:
    path: sampling_params.py
    method: __post_init__
    line_at_scan: 55
  match:
    engine: vllm
    fields:
      vllm.sampling.temperature:
        <: _SAMPLING_EPS
  kwargs_positive:
    temperature: 0.1
  kwargs_negative:
    temperature: 1.0
  expected_outcome:
    outcome: dormant_announced
    emission_channel: none
    normalised_fields: []
  message_template: ''
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
- id: vllm_min_p_set_to_zero_if_temperature_lt_sampling_eps
  engine: vllm
  library: vllm
  invariant_under_test: SamplingParams.__post_init__ sets min_p to 0 if temperature less than _SAMPLING_EPS
  severity: dormant
  native_type: vllm.SamplingParams
  miner_source:
    path: sampling_params.py
    method: __post_init__
    line_at_scan: 60
  match:
    engine: vllm
    fields:
      vllm.sampling.temperature:
        <: _SAMPLING_EPS
  kwargs_positive:
    temperature: 0.1
  kwargs_negative:
    temperature: 1.0
  expected_outcome:
    outcome: dormant_announced
    emission_channel: none
    normalised_fields: []
  message_template: ''
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
- id: vllm_all_stop_token_ids_set_if_stop
  engine: vllm
  library: vllm
  invariant_under_test: SamplingParams.__post_init__ sets _all_stop_token_ids if stop
  severity: dormant
  native_type: vllm.SamplingParams
  miner_source:
    path: sampling_params.py
    method: __post_init__
    line_at_scan: 65
  match:
    engine: vllm
    fields:
      vllm.sampling.stop:
        present: true
  kwargs_positive:
    stop: ['stop']
  kwargs_negative:
    stop: []
  expected_outcome:
    outcome: dormant_announced
    emission_channel: none
    normalised_fields: []
  message_template: ''
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
- id: vllm_best_of_must_be_int
  engine: vllm
  library: vllm
  invariant_under_test: SamplingParams._verify_args checks if best_of is an integer
  severity: error
  native_type: vllm.SamplingParams
  miner_source:
    path: sampling_params.py
    method: _verify_args
    line_at_scan: 10
  match:
    engine: vllm
    fields:
      vllm.sampling.best_of:
        type_is_not:
          - int
  kwargs_positive:
    best_of: 'a'
  kwargs_negative:
    best_of: 1
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "best_of must be an integer, got %s."
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
- id: vllm_best_of_must_be_ge_n
  engine: vllm
  library: vllm
  invariant_under_test: SamplingParams._verify_args checks if best_of is greater than or equal to n
  severity: error
  native_type: vllm.SamplingParams
  miner_source:
    path: sampling_params.py
    m
...<truncated>...
```
