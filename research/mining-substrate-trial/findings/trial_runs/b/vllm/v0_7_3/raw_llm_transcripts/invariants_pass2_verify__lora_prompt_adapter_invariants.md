# invariants_pass2_verify extraction transcript: lora_prompt_adapter_invariants

- chunk_description: vllm.{LoRAConfig, PromptAdapterConfig, TokenizerPoolConfig}.__post_init__
- expected_namespaces: ['vllm']
- attempts: 1
- elapsed_sec: 9.50
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
- id: vllm_max_lora_rank_not_in_allowlist
  severity: error
  match:
    engine: vllm
    fields:
      vllm.max_lora_rank:
        present: true
        not_in:
        - 8
        - 16
        - 32
        - 64
        - 128
        - 256
  invariant_under_test: LoRAConfig.__post_init__ flags max_lora_rank not in allowlist
- id: vllm_lora_extra_vocab_size_not_in_allowlist
  severity: error
  match:
    engine: vllm
    fields:
      vllm.lora_extra_vocab_size:
        present: true
        not_in:
        - 0
        - 256
        - 512
  invariant_under_test: LoRAConfig.__post_init__ flags lora_extra_vocab_size not in
    allowlist
- id: vllm_max_loras_lt_1
  severity: error
  match:
    engine: vllm
    fields:
      vllm.max_loras:
        <: 1
  invariant_under_test: LoRAConfig.__post_init__ flags max_loras < 1
- id: vllm_max_cpu_loras_lt_max_loras
  severity: error
  match:
    engine: vllm
    fields:
      vllm.max_cpu_loras:
        <: vllm.max_loras
  invariant_under_test: LoRAConfig.__post_init__ flags max_cpu_loras < max_loras
- id: vllm_max_prompt_adapters_lt_1
  severity: error
  match:
    engine: vllm
    fields:
      vllm.max_prompt_adapters:
        <: 1
  invariant_under_test: PromptAdapterConfig.__post_init__ flags max_prompt_adapters
    < 1
- id: vllm_max_prompt_adapter_token_eq_0
  severity: error
  match:
    engine: vllm
    fields:
      vllm.max_prompt_adapter_token:
        ==: 0
  invariant_under_test: PromptAdapterConfig.__post_init__ flags max_prompt_adapter_token
    == 0
- id: vllm_pool_type_not_in_allowlist
  severity: error
  match:
    engine: vllm
    fields:
      vllm.pool_type:
        present: true
        not_in:
        - ray
  invariant_under_test: TokenizerPoolConfig.__post_init__ flags pool_type not in allowlist
- id: vllm_extra_config_not_dict
  severity: error
  match:
    engine: vllm
    fields:
      vllm.extra_config:
        present: true
        type_is_not:
        - dict
  invariant_under_test: TokenizerPoolConfig.__post_init__ flags extra_config not a
    dict


INPUT 2 - THE SOURCE PASS 1 READ:

=== CONTEXT ===
Three small dataclasses share namespace `vllm`. Each `__post_init__` emits ValueError on bad shapes (max_loras<1, max_cpu_loras<max_loras, max_prompt_adapters<1, max_prompt_adapter_token==0, pool_type not in {ray}, extra_config not a dict).

=== SOURCE: LoRAConfig.__post_init__ ===
    def __post_init__(self):
        # Setting the maximum rank to 256 should be able to satisfy the vast
        # majority of applications.
        possible_max_ranks = (8, 16, 32, 64, 128, 256)
        possible_lora_extra_vocab_size = (0, 256, 512)
        if self.max_lora_rank not in possible_max_ranks:
            raise ValueError(
                f"max_lora_rank ({self.max_lora_rank}) must be one of "
                f"{possible_max_ranks}.")
        if self.lora_extra_vocab_size not in possible_lora_extra_vocab_size:
            raise ValueError(
                f"lora_extra_vocab_size ({self.lora_extra_vocab_size}) "
                f"must be one of {possible_lora_extra_vocab_size}.")
        if self.max_loras < 1:
            raise ValueError(f"max_loras ({self.max_loras}) must be >= 1.")
        if self.max_cpu_loras is None:
            self.max_cpu_loras = self.max_loras
        elif self.max_cpu_loras < self.max_loras:
            raise ValueError(
                f"max_cpu_loras ({self.max_cpu_loras}) must be >= "
                f"max_loras ({self.max_loras})")


=== SOURCE: PromptAdapterConfig.__post_init__ ===
    def __post_init__(self):

        if self.max_prompt_adapters < 1:
            raise ValueError(f"max_prompt_adapters "
                             f"({self.max_prompt_adapters}) must be >= 1.")
        if self.max_prompt_adapter_token == 0:
            raise ValueError("max_prompt_adapter_token must be set.")
        if self.max_cpu_prompt_adapters is None:
            self.max_cpu_prompt_adapters = self.max_prompt_adapters


=== SOURCE: TokenizerPoolConfig.__post_init__ ===
    def __post_init__(self):
        if self.pool_type not in ("ray", ) and not isinstance(
                self.pool_type, type):
            raise ValueError(f"Unknown pool type: {self.pool_type}")
        if not isinstance(self.extra_config, dict):
            raise ValueError("extra_config must be a dictionary.")




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
- vllm_max_lora_rank_not_in_allowlist
- vllm_lora_extra_vocab_size_not_in_allowlist
- vllm_max_loras_lt_1
- vllm_max_cpu_loras_lt_max_loras
- vllm_max_prompt_adapters_lt_1
- vllm_max_prompt_adapter_token_eq_0

flagged:
- id: vllm_pool_type_not_in_allowlist
  reason: Source allows pool_type to be a type, not just "ray".
  fix: correct_predicate:not_in_or_isinstance
- id: vllm_extra_config_not_dict
  reason: Source checks for isinstance(self.extra_config, dict), not just type.
  fix: correct_predicate:type_is
```
