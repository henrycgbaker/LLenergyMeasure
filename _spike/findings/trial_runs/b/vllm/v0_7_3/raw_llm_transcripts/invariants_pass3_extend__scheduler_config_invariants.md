# invariants_pass3_extend extraction transcript: scheduler_config_invariants

- chunk_description: vllm.SchedulerConfig __post_init__ + _verify_args
- expected_namespaces: ['vllm']
- attempts: 1
- elapsed_sec: 63.41
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
- id: vllm_max_num_batched_tokens_lt_max_model_len
  severity: error
  match:
    engine: vllm
    fields:
      vllm.max_num_batched_tokens:
        <: 1
      vllm.chunked_prefill_enabled: false
  invariant_under_test: SchedulerConfig._verify_args flags max_num_batched_tokens
    < max_model_len when not chunked_prefill_enabled
- id: vllm_max_num_batched_tokens_lt_max_num_seqs
  severity: error
  match:
    engine: vllm
    fields:
      vllm.max_num_batched_tokens:
        <: 1
  invariant_under_test: SchedulerConfig._verify_args flags max_num_batched_tokens
    < max_num_seqs
- id: vllm_num_lookahead_slots_lt_0
  severity: error
  match:
    engine: vllm
    fields:
      vllm.num_lookahead_slots:
        <: 0
  invariant_under_test: SchedulerConfig._verify_args flags num_lookahead_slots < 0
- id: vllm_num_scheduler_steps_lt_1
  severity: error
  match:
    engine: vllm
    fields:
      vllm.num_scheduler_steps:
        <: 1
  invariant_under_test: SchedulerConfig._verify_args flags num_scheduler_steps < 1
- id: vllm_max_num_partial_prefills_lt_1
  severity: error
  match:
    engine: vllm
    fields:
      vllm.max_num_partial_prefills:
        <: 1
  invariant_under_test: SchedulerConfig._verify_args flags max_num_partial_prefills
    < 1
- id: vllm_max_long_partial_prefills_lt_1_or_gt_max_num_partial_prefills
  severity: error
  match:
    engine: vllm
    fields:
      vllm.max_long_partial_prefills:
        <: 1
      vllm.max_num_partial_prefills:
        '>=': 1
  invariant_under_test: SchedulerConfig._verify_args flags max_long_partial_prefills
    < 1 or > max_num_partial_prefills
- id: vllm_max_num_partial_prefills_gt_1_without_chunked_prefill_enabled
  severity: error
  match:
    engine: vllm
    fields:
      vllm.max_num_partial_prefills:
        '>': 1
      vllm.chunked_prefill_enabled: false
  invariant_under_test: SchedulerConfig._verify_args flags max_num_partial_prefills
    > 1 without chunked_prefill_enabled
- id: vllm_long_prefill_token_threshold_gt_max_model_len
  severity: error
  match:
    engine: vllm
    fields:
      vllm.long_prefill_token_threshold:
        '>': 1
  invariant_under_test: SchedulerConfig._verify_args flags long_prefill_token_threshold
    > max_model_len


INPUT 2 - PASS 2 FLAGGED ISSUES (for context; consider whether they hint
at related misses):

flagged:
- id: vllm_max_num_partial_prefills_lt_1
  reason: Source checks for `max_num_partial_prefills < 1` but allows it to be equal
    to 1.
  fix: correct_predicate:not_equal_or_less_than
- id: vllm_max_long_partial_prefills_lt_1_or_gt_max_num_partial_prefills
  reason: Source checks for `max_long_partial_prefills < 1 or max_long_partial_prefills
    > max_num_partial_prefills` but allows it to be equal to max_num_partial_prefills.
  fix: correct_predicate:not_equal_or_less_than_and_not_greater_than
- id: vllm_max_num_partial_prefills_gt_1_without_chunked_prefill_enabled
  reason: Source checks for `max_num_partial_prefills > 1 and not chunked_prefill_enabled`
    but allows it to be equal to 1 without chunked_prefill_enabled.
  fix: correct_predicate:not_equal_or_greater_than_and_not_equal


INPUT 3 - THE SOURCE PASS 1 READ:

=== CONTEXT ===
vllm.SchedulerConfig validates `max_num_batched_tokens` vs `max_num_seqs` and the `policy` enum. Use namespace `vllm`.

=== SOURCE: SchedulerConfig.__post_init__ ===
    def __post_init__(self) -> None:
        if self.max_num_batched_tokens is None:
            if self.enable_chunked_prefill:
                if self.num_scheduler_steps > 1:
                    # Multi-step Chunked-Prefill doesn't allow prompt-chunking
                    # for now. Have max_num_batched_tokens set to max_model_len
                    # so we don't reject sequences on account of a short
                    # max_num_batched_tokens.
                    self.max_num_batched_tokens = max(self.max_model_len, 2048)
                else:
                    # This value is chosen to have a balance between ITL
                    # and TTFT. Note it is not optimized for throughput.
                    self.max_num_batched_tokens = 2048
            else:
                # If max_model_len is too short, use 2048 as the default value
                # for higher throughput.
                self.max_num_batched_tokens = max(self.max_model_len, 2048)

            if self.runner_type == "pooling":
                # Choose specific value for higher throughput
                self.max_num_batched_tokens = max(
                    self.max_num_batched_tokens,
                    _POOLING_MODEL_MAX_NUM_BATCHED_TOKENS,
                )
            if self.is_multimodal_model:
                # The value needs to be at least the number of multimodal tokens
                self.max_num_batched_tokens = max(
                    self.max_num_batched_tokens,
                    _MULTIMODAL_MODEL_MAX_NUM_BATCHED_TOKENS,
                )

        self.max_num_encoder_input_tokens = self.max_num_batched_tokens
        self.encoder_cache_size = self.max_num_batched_tokens

        if self.enable_chunked_prefill:
            logger.info(
                "Chunked prefill is enabled with max_num_batched_tokens=%d.",
                self.max_num_batched_tokens)

        self.chunked_prefill_enabled = self.enable_chunked_prefill
        if self.max_num_partial_prefills > 1:
            if self.long_prefill_token_threshold == 0:
                self.long_prefill_token_threshold = int(self.max_model_len *
                                                        0.04)

            logger.info(
                "Concurrent partial prefills enabled with "
                "max_num_partial_prefills=%d, max_long_partial_prefills=%d, "
                "long_prefill_token_threshold=%d",
                self.max_num_partial_prefills, self.max_long_partial_prefills,
                self.long_prefill_token_threshold)

        self._verify_args()


=== SOURCE: SchedulerConfig._verify_args ===
    def _verify_args(self) -> None:
        if (self.max_num_batched_tokens < self.max_model_len
                and not self.chunked_prefill_enabled):
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) is "
                f"smaller than max_model_len ({self.max_model_len}). "
                "This effectively limits the maximum sequence length to "
                "max_num_batched_tokens and makes vLLM reject longer "
                "sequences. Please increase max_num_batched_tokens or "
                "decrease max_model_len.")

        if self.max_num_batched_tokens < self.max_num_seqs:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) must "
                "be greater than or equal to max_num_seqs "
                f"({self.max_num_seqs}).")

        if self.num_lookahead_slots < 0:
            raise ValueError(
                "num_lookahead_slots "
                f"({self.num_lookahead_slots}) must be greater than or "
                "equal to 0.")

        if self.num_scheduler_steps < 1:
            raise ValueError(
                "num_scheduler_steps "
                f"({self.num_scheduler_steps}) must be greater than or "
                "equal to 1.")

        if self.max_num_partial_prefills < 1:
            raise ValueError(
                f"max_num_partial_prefills ({self.max_num_partial_prefills}) "
                "must be greater than or equal to 1.")
        elif self.max_num_partial_prefills > 1:
            if not self.chunked_prefill_enabled:
                raise ValueError("Chunked prefill must be enabled to set "
                             
...<truncated>...
```

### Response

```
invariants:
- id: vllm_max_num_batched_tokens_set_to_max_model_len_when_chunked_prefill_enabled_and_num_scheduler_steps_gt_1
  engine: vllm
  library: vllm
  invariant_under_test: SchedulerConfig sets max_num_batched_tokens to max_model_len when chunked prefill is enabled and num_scheduler_steps > 1
  severity: dormant
  native_type: vllm.SchedulerConfig
  miner_source:
    path: SchedulerConfig.py
    method: __post_init__
    line_at_scan: 10
  match:
    engine: vllm
    fields:
      vllm.enable_chunked_prefill: true
      vllm.num_scheduler_steps: {'>': 1}
  kwargs_positive:
    enable_chunked_prefill: True
    num_scheduler_steps: 2
  kwargs_negative:
    enable_chunked_prefill: False
    num_scheduler_steps: 1
  expected_outcome:
    outcome: dormant_announced
    emission_channel: logger_info
    normalised_fields: []
  message_template: 'Chunked prefill is enabled with max_num_batched_tokens=%d.'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: vllm_max_num_batched_tokens_set_to_2048_when_enable_chunked_prefill_false_and_max_model_len_lt_2048
  engine: vllm
  library: vllm
  invariant_under_test: SchedulerConfig sets max_num_batched_tokens to 2048 when chunked prefill is disabled and max_model_len < 2048
  severity: dormant
  native_type: vllm.SchedulerConfig
  miner_source:
    path: SchedulerConfig.py
    method: __post_init__
    line_at_scan: 15
  match:
    engine: vllm
    fields:
      vllm.enable_chunked_prefill: false
      vllm.max_model_len: {'<': 2048}
  kwargs_positive:
    enable_chunked_prefill: False
    max_model_len: 1024
  kwargs_negative:
    enable_chunked_prefill: True
    max_model_len: 4096
  expected_outcome:
    outcome: dormant_announced
    emission_channel: none
    normalised_fields: []
  message_template: ''
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: vllm_max_num_batched_tokens_set_toPooling_model_max_num_batched_tokens_when_runner_type_pooling
  engine: vllm
  library: vllm
  invariant_under_test: SchedulerConfig sets max_num_batched_tokens to _POOLING_MODEL_MAX_NUM_BATCHED_TOKENS when runner_type is 'pooling'
  severity: dormant
  native_type: vllm.SchedulerConfig
  miner_source:
    path: SchedulerConfig.py
    method: __post_init__
    line_at_scan: 20
  match:
    engine: vllm
    fields:
      vllm.runner_type: 'pooling'
  kwargs_positive:
    runner_type: 'pooling'
  kwargs_negative:
    runner_type: 'other'
  expected_outcome:
    outcome: dormant_announced
    emission_channel: none
    normalised_fields: []
  message_template: ''
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: vllm_max_num_batched_tokens_set_to_multimodal_model_max_num_batched_tokens_when_is_multimodal_model_true
  engine: vllm
  library: vllm
  invariant_under_test: SchedulerConfig sets max_num_batched_tokens to _MULTIMODAL_MODEL_MAX_NUM_BATCHED_TOKENS when is_multimodal_model is True
  severity: dormant
  native_type: vllm.SchedulerConfig
  miner_source:
    path: SchedulerConfig.py
    method: __post_init__
    line_at_scan: 25
  match:
    engine: vllm
    fields:
      vllm.is_multimodal_model: true
  kwargs_positive:
    is_multimodal_model: True
  kwargs_negative:
    is_multimodal_model: False
  expected_outcome:
    outcome: dormant_announced
    emission_channel: none
    normalised_fields: []
  message_template: ''
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: vllm_long_prefill_token_threshold_set_to_max_model_len_times_0_04_when_long_prefill_token_threshold_eq_0_and_max_num_partial_prefills_gt_1
  engine: vllm
  library: vllm
  invariant_under_test: SchedulerConfig sets long_prefill_token_threshold to max_model_len * 0.04 when long_prefill_token_threshold is 0 and max_num_partial_prefills > 1
  severity: dormant
  native_type: vllm.SchedulerConfig
  miner_source:
    path: SchedulerConfig.py
    method: __post_init__
    line_at_scan: 30
  match:
    engine: vllm
    fields:
      vllm.long_prefill_token_threshold: 0
      vllm.max_num_partial_prefills: {'>': 1}
  kwargs_positive:
    long_prefill_token_threshold: 0
    max_num_partial_prefills: 2
  kwargs_negative:
    long_prefill_token_threshold: 10
    max_num_partial_prefills: 1
  expected_outcome:
    outcome: dormant_announced
    emission_channel: logger_info
    normalised_fields: []
  message_template: 'Concurrent partial prefills enabled with max_num_partial_prefills=%d, max_long_partial_prefills=%d, long_prefill_token_threshold=%d'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
```
