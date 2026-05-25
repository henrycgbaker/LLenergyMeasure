# invariants_pass2_verify extraction transcript: scheduler_config_invariants

- chunk_description: vllm.SchedulerConfig __post_init__ + _verify_args
- expected_namespaces: ['vllm']
- attempts: 1
- elapsed_sec: 56.63
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser doing PASS 2 (verification) of multi-pass
invariant extraction. PASS 1 already extracted invariants from
vllm v0.9.2 for ONE chunk of source. Your job is to
REVIEW each emitted invariant against the source and flag any that
look WRONG.

INPUT 1 - PASS 1 EXTRACTED INVARIANTS (the candidate list):

invariants:
- id: vllm_max_num_batched_tokens_lt_max_model_len
  severity: error
  match:
    engine: vllm
    fields:
      vllm.chunked_prefill_enabled: false
      vllm.max_num_batched_tokens:
        <: 8192
  invariant_under_test: SchedulerConfig._verify_args flags max_num_batched_tokens
    < max_model_len when not chunked_prefill_enabled
- id: vllm_max_num_batched_tokens_lt_max_num_seqs
  severity: error
  match:
    engine: vllm
    fields:
      vllm.max_num_batched_tokens:
        <: 128
  invariant_under_test: SchedulerConfig._verify_args flags max_num_batched_tokens
    < max_num_seqs
- id: vllm_max_num_batched_tokens_gt_max_num_seqs_times_max_model_len
  severity: warning
  match:
    engine: vllm
    fields:
      vllm.max_num_batched_tokens:
        '>': 1048576
  invariant_under_test: SchedulerConfig._verify_args flags max_num_batched_tokens
    > max_num_seqs * max_model_len
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
        '>': 2
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
        '>': 8192
  invariant_under_test: SchedulerConfig._verify_args flags long_prefill_token_threshold
    > max_model_len


INPUT 2 - THE SOURCE PASS 1 READ:

=== CONTEXT ===
vllm.SchedulerConfig validates `max_num_batched_tokens` vs `max_num_seqs` and the `policy` enum. Use namespace `vllm`.

=== SOURCE: SchedulerConfig.__post_init__ ===
    def __post_init__(self) -> None:
        if self.max_model_len is None:
            self.max_model_len = 8192

        if self.max_num_seqs is None:
            self.max_num_seqs = 128

        if self.max_num_batched_tokens is None:
            if self.enable_chunked_prefill:
                if self.num_scheduler_steps > 1:
                    # Multi-step Chunked-Prefill doesn't allow prompt-chunking
                    # for now. Have max_num_batched_tokens set to max_model_len
                    # so we don't reject sequences on account of a short
                    # max_num_batched_tokens.
                    self.max_num_batched_tokens = max(
                        self.max_model_len, DEFAULT_MAX_NUM_BATCHED_TOKENS)
                else:
                    self.max_num_batched_tokens = (
                        DEFAULT_MAX_NUM_BATCHED_TOKENS)
            else:
                # If max_model_len is too short, use
                # DEFAULT_MAX_NUM_BATCHED_TOKENS as the default value
                # for higher throughput.
                self.max_num_batched_tokens = max(
                    self.max_model_len, DEFAULT_MAX_NUM_BATCHED_TOKENS)

            if self.runner_type == "pooling":
                # Choose specific value for higher throughput
                self.max_num_batched_tokens = max(
                    self.max_num_batched_tokens,
                    POOLING_MODEL_MAX_NUM_BATCHED_TOKENS,
                )
            if self.is_multimodal_model:
                # The value needs to be at least the number of multimodal tokens
                self.max_num_batched_tokens = max(
                    self.max_num_batched_tokens,
                    MULTIMODAL_MODEL_MAX_NUM_BATCHED_TOKENS,
                )

            # When using default settings,
            # Ensure max_num_batched_tokens does not exceed model limit.
            # Some models (e.g., Whisper) have embeddings tied to max length.
            self.max_num_batched_tokens = min(
                self.max_num_seqs * self.max_model_len,
                self.max_num_batched_tokens)

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


=== SOURCE: SchedulerConfig._verify_args ===
    def _verify_args(self) -> Self:
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

        if self.max_num_batched_tokens > self.max_num_seqs * self.max_model_len:
            logger.warning(
                "max_num_batched_tokens (%d) exceeds max_num_seqs "
                "* max_model_len (%d). This may lead to unexpected behavior.",
                self.max_num_batched_tokens,
                self.max_num_seqs * self.max_model_len)

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
      
...<truncated>...
```

### Response

```
confirmed:
- vllm_max_num_batched_tokens_lt_max_model_len
- vllm_max_num_batched_tokens_lt_max_num_seqs
- vllm_max_num_batched_tokens_gt_max_num_seqs_times_max_model_len
- vllm_num_lookahead_slots_lt_0
- vllm_num_scheduler_steps_lt_1
- vllm_max_num_partial_prefills_lt_1

flagged:
- id: vllm_max_long_partial_prefills_lt_1_or_gt_max_num_partial_prefills
  reason: Source checks for `< 1 or > max_num_partial_prefills`, but invariant only checks `> 2`.
  fix: correct_predicate:range
- id: vllm_max_num_partial_prefills_gt_1_without_chunked_prefill_enabled
  reason: Severity is error, but source raises ValueError with a different message.
  fix: correct_severity:warning
- id: vllm_long_prefill_token_threshold_gt_max_model_len
  reason: Source checks for `> max_model_len`, but invariant only checks `> 8192`.
  fix: correct_predicate:exact
```
