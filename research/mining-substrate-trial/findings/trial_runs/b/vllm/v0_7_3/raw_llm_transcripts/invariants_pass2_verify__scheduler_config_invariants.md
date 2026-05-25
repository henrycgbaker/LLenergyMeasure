# invariants_pass2_verify extraction transcript: scheduler_config_invariants

- chunk_description: vllm.SchedulerConfig __post_init__ + _verify_args
- expected_namespaces: ['vllm']
- attempts: 1
- elapsed_sec: 15.56
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser doing PASS 2 (verification) of multi-pass
invariant extraction. PASS 1 already extracted invariants from
vllm v0.7.3 for ONE chunk of source. Your job is to
REVIEW each emitted invariant against the source and flag any that
look WRONG.

INPUT 1 - PASS 1 EXTRACTED INVARIANTS (the candidate list):

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


INPUT 2 - THE SOURCE PASS 1 READ:

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
                                 "max_num_partial_prefills > 1.")

            if self.long_prefill_token_threshold > self.max_model_len:
                raise ValueError(
                    "long_prefill_token_threshold "
                    f"({self.long_prefill_token_threshold}) cannot be greater "
                    f"than the max_model_len ({self.max_model_len}).")

        if (self.max_long_partial_prefills
                < 1) or (self.max_long_partial_prefills
                         > self.max_num_partial_prefills):
            raise ValueError(
                f"max_long_partial_prefills ({self.max_long_partial_prefills}) "
                "must be greater than or equal to 1 and less than or equal to "
                f"max_num_partial_prefills ({self.max_num_partial_prefills}).")




OUTPUT FORMAT: a YAML document with TWO sections:

confirmed:
- <id-of-pass1-invariant>  # one ID pe
...<truncated>...
```

### Response

```
confirmed:
- vllm_max_num_batched_tokens_lt_max_model_len
- vllm_max_num_batched_tokens_lt_max_num_seqs
- vllm_num_lookahead_slots_lt_0
- vllm_num_scheduler_steps_lt_1
- vllm_long_prefill_token_threshold_gt_max_model_len

flagged:
- id: vllm_max_num_partial_prefills_lt_1
  reason: Source checks for `max_num_partial_prefills < 1` but allows it to be equal to 1.
  fix: correct_predicate:not_equal_or_less_than
- id: vllm_max_long_partial_prefills_lt_1_or_gt_max_num_partial_prefills
  reason: Source checks for `max_long_partial_prefills < 1 or max_long_partial_prefills > max_num_partial_prefills` but allows it to be equal to max_num_partial_prefills.
  fix: correct_predicate:not_equal_or_less_than_and_not_greater_than
- id: vllm_max_num_partial_prefills_gt_1_without_chunked_prefill_enabled
  reason: Source checks for `max_num_partial_prefills > 1 and not chunked_prefill_enabled` but allows it to be equal to 1 without chunked_prefill_enabled.
  fix: correct_predicate:not_equal_or_greater_than_and_not_equal
```
