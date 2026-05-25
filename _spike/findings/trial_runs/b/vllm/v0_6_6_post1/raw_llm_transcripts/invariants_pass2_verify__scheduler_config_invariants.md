# invariants_pass2_verify extraction transcript: scheduler_config_invariants

- chunk_description: vllm.SchedulerConfig __post_init__ + _verify_args
- expected_namespaces: ['vllm']
- attempts: 1
- elapsed_sec: 50.17
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
- id: vllm_max_num_batched_tokens_lt_max_model_len
  severity: error
  match:
    engine: vllm
    fields:
      vllm.chunked_prefill_enabled: false
      vllm.max_num_batched_tokens:
        <: 1
  invariant_under_test: SchedulerConfig._verify_args flags max_num_batched_tokens
    < max_model_len when chunked_prefill_enabled=False
- id: vllm_max_num_batched_tokens_lt_max_num_seqs
  severity: error
  match:
    engine: vllm
    fields:
      vllm.max_num_batched_tokens:
        <: 1
  invariant_under_test: SchedulerConfig._verify_args flags max_num_batched_tokens
    < max_num_seqs
- id: vllm_num_lookahead_slots_lt_zero
  severity: error
  match:
    engine: vllm
    fields:
      vllm.num_lookahead_slots:
        <: 0
  invariant_under_test: SchedulerConfig._verify_args flags num_lookahead_slots < 0
- id: vllm_num_scheduler_steps_lt_one
  severity: error
  match:
    engine: vllm
    fields:
      vllm.num_scheduler_steps:
        <: 1
  invariant_under_test: SchedulerConfig._verify_args flags num_scheduler_steps < 1


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

        if self.enable_chunked_prefill:
            logger.info(
                "Chunked prefill is enabled with max_num_batched_tokens=%d.",
                self.max_num_batched_tokens)

        self.chunked_prefill_enabled = self.enable_chunked_prefill
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




OUTPUT FORMAT: a YAML document with TWO sections:

confirmed:
- <id-of-pass1-invariant>  # one ID per line, no further detail needed.

flagged:
- id: <id-of-pass1-invariant>
  reason: <one-line: what looks wrong>
  fix: <one of: "drop", "correct_severity:<new-severity>", "correct_predicate:<new-kind>", "correct_kwargs_positive">

RULES:
1. Every PASS 1 invariant MUST appear in EITHER `confirmed` OR `flagged`
   (not both, not neither). If you're unsure, place in `confirmed` (the
   bar for flagging is "obviously wrong against the source").
2. Flag reasons must be CONCRETE - cite the source line that contradicts
   the invariant, or note the specific shape mismatch.
3. NO markdown code fences. NO commentary outside the YAML.
4. First character must be `c` (from `confirmed:`).

CRITERIA FOR FLAGGING:
- Severity wrong: source has `raise ValueError` but invariant says
  `severity: dormant`; or source has `minor_issues[...] = ...` but
  invariant says `severity: error`.
- Predicate wrong: source has `if X.field not in {a, b}: raise`
  but invariant emits `predicate_kind: exact`.
- Kwargs_positive wrong: source's predicate is "value < 0 raises" but
  invariant's kwargs_positive shows `field: 1` (which would NOT trigger).
- Hallucinated: invariant references a field name that does NOT appear
  in the source.

Emit the YAML now:

```

### Response

```
confirmed:
- vllm_max_num_batched_tokens_lt_max_model_len
- vllm_max_num_batched_tokens_lt_max_num_seqs
- vllm_num_scheduler_steps_lt_one

flagged:
- id: vllm_num_lookahead_slots_lt_zero
  reason: Source checks for `num_lookahead_slots < 0` but invariant has `< 1`.
  fix: correct_predicate:range
```
