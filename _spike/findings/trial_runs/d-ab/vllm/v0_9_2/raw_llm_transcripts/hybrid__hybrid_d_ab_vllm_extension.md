# hybrid extraction transcript: hybrid_d_ab_vllm_extension

- chunk_description: Hybrid d-ab on vllm: (a) output + source -> extension
- expected_namespaces: ['vllm.*']
- attempts: 1
- elapsed_sec: 143.91
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser working as a SECOND PASS on top of a
deterministic miner's output. The deterministic miner has already
extracted invariants from vllm v0.9.2; your job is to
find what it MISSED, find what looks SPURIOUS, and diagnose WHY it
missed what it missed.

INPUT 1 - DETERMINISTIC MINER'S OUTPUT (this is what (a) found):

schema_version: 1.0.0
engine: vllm
engine_version: 0.7.3
mined_at: '2026-05-25'
invariants:
- id: vllm_samplingparams_dormant_seed_eq_neg1
  engine: vllm
  library: vllm
  invariant_under_test: 'SamplingParams.__post_init__: marks dormant when seed == -1'
  severity: dormant
  native_type: vllm.SamplingParams
  miner_source:
    path: sampling_params.py
    method: __post_init__
    line_at_scan: 311
  match:
    engine: vllm
    fields:
      vllm.sampling.seed: -1
  kwargs_positive:
    seed: -1
  kwargs_negative:
    seed: 0
  expected_outcome:
    outcome: dormant_silent
    emission_channel: none
    normalised_fields:
    - seed
  message_template: null
  references:
  - sampling_params.py:311 (vllm.SamplingParams.__post_init__)
  added_by: static_miner
  added_at: '2026-04-27'
- id: vllm_samplingparams_raises_best_of_lt_ref_n
  engine: vllm
  library: vllm
  invariant_under_test: 'SamplingParams.__post_init__: raises when best_of present True AND best_of <
    @n'
  severity: error
  native_type: vllm.SamplingParams
  miner_source:
    path: sampling_params.py
    method: __post_init__
    line_at_scan: 296
  match:
    engine: vllm
    fields:
      vllm.sampling.best_of:
        present: true
        <: '@n'
  kwargs_positive:
    best_of: 1
    n: 2
  kwargs_negative:
    best_of: 2
    n: 2
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: best_of must be greater than or equal to n, got n={n} and best_of={best_of}.
  references:
  - sampling_params.py:296 (vllm.SamplingParams.__post_init__)
  added_by: static_miner
  added_at: '2026-05-13T22:55:33+02:00'
- id: vllm_samplingparams_raises_logprobs_lt_0
  engine: vllm
  library: vllm
  invariant_under_test: 'SamplingParams._verify_args: raises when logprobs present True AND logprobs < 0'
  severity: error
  native_type: vllm.SamplingParams
  miner_source:
    path: sampling_params.py
    method: _verify_args
    line_at_scan: 391
  match:
    engine: vllm
    fields:
      vllm.sampling.logprobs:
        present: true
        <: 0
  kwargs_positive:
    logprobs: -1
  kwargs_negative:
    logprobs: 0
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: logprobs must be non-negative, got {self.logprobs}.
  references:
  - sampling_params.py:391 (vllm.SamplingParams._verify_args)
  added_by: static_miner
  added_at: '2026-05-25'
- id: vllm_samplingparams_raises_max_tokens_lt_1
  engine: vllm
  library: vllm
  invariant_under_test: 'SamplingParams._verify_args: raises when max_tokens present True AND max_tokens < 1'
  severity: error
  native_type: vllm.SamplingParams
  miner_source:
    path: sampling_params.py
    method: _verify_args
    line_at_scan: 381
  match:
    engine: vllm
    fields:
      vllm.sampling.max_tokens:
        present: true
        <: 1
  kwargs_positive:
    max_tokens: 0
  kwargs_negative:
    max_tokens: 1
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: max_tokens must be at least 1, got {self.max_tokens}.
  references:
  - sampling_params.py:381 (vllm.SamplingParams._verify_args)
  added_by: static_miner
  added_at: '2026-05-25'
- id: vllm_samplingparams_raises_min_tokens_gt_ref_max_tokens
  engine: vllm
  library: vllm
  invariant_under_test: 'SamplingParams._verify_args: raises when max_tokens present True AND min_tokens
    > @max_tokens'
  severity: error
  native_type: vllm.SamplingParams
  miner_source:
    path: sampling_params.py
    method: _verify_args
    line_at_scan: 388
  match:
    engine: vllm
    fields:
      vllm.sampling.max_tokens:
        present: true
      vllm.sampling.min_tokens:
        '>': '@max_tokens'
  kwargs_positive:
    max_tokens: 1
    min_tokens: 2
  kwargs_negative:
    max_tokens: 1
    min_tokens: 1
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: min_tokens must be less than or equal to max_tokens={max_tokens}, got {min_tokens}.
  references:
  - sampling_params.py:388 (vllm.SamplingParams._verify_args)
  added_by: static_miner
  added_at: '2026-04-27'
- id: vllm_samplingparams_raises_min_tokens_lt_0
  engine: vllm
  library: vllm
  invariant_under_test: 'SamplingParams._verify_args: raises when min_tokens < 0'
  severity: error
  native_type: vllm.SamplingParams
  miner_source:
    path: sampling_params.py
    method: _verify_args
    line_at_scan: 385
  match:
    engine: vllm
    fields:
      vllm.sampling.min_tokens:
        <: 0
  kwargs_positive:
    min_tokens: -1
  kwargs_negative:
    min_tokens: 0
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: min_tokens must be greater than or equal to 0, got {min_tokens}.
  references:
  - sampling_params.py:385 (vllm.SamplingParams._verify_args)
  added_by: static_miner
  added_at: '2026-04-27'
- id: vllm_samplingparams_raises_n_lt_1
  engine: vllm
  library: vllm
  invariant_under_test: 'SamplingParams._verify_args: raises when n < 1'
  severity: error
  native_type: vllm.SamplingParams
  miner_source:
    path: sampling_params.py
    method: _verify_args
    line_at_scan: 357
  match:
    engine: vllm
    fields:
      vllm.sampling.n:
        <: 1
  kwargs_positive:
    n: 0
  kwargs_negative:
    n: 1
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: n must be at least 1, got {n}.
  references:
  - sampling_params.py:357 (vllm.SamplingParams._verify_args)
  added_by: static_miner
  added_at: '2026-04-27'
- id: vllm_samplingparams_raises_n_not_type_int
  engine: vllm
  library: vllm
  invariant_under_test: 'SamplingParams._verify_args: raises when n type_is_not int'
  severity: error
  native_type: vllm.SamplingParams
  miner_source:
    path: sampling_params.py
    method: _verify_args
    line_at_scan: 354
  match:
    engine: vllm
    fields:
      vllm.sampling.n:
        type_is_not: int
  kwargs_positive:
    n: x
  kwargs_negative:
    n: 1
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: n must be an int, but is of type {type(self.n)}
  references:
  - sampling_params.py:354 (vllm.SamplingParams._verify_args)
  added_by: static_miner
  added_at: '2026-04-27'
- id: vllm_samplingparams_raises_prompt_logprobs_lt_0
  engine: vllm
  library: vllm
  invariant_under_test: 'SamplingParams._verify_args: raises when prompt_logprobs present True AND
    prompt_logprobs < 0'
  severity: error
  native_type: vllm.SamplingParams
  miner_source:
    path: sampling_params.py
    method: _verify_args
    line_at_scan: 394
  match:
    engine: vllm
    fields:
      vllm.sampling.prompt_logprobs:
        present: true
        <: 0
  kwargs_positive:
    prompt_logprobs: -1
  kwargs_negative:
    prompt_logprobs: 0
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: prompt_logprobs must be non-negative, got {self.prompt_logprobs}.
  references:
  - sampling_params.py:394 (vllm.SamplingParams._verify_args)
  added_by: static_miner
  added_at: '2026-05-25'
- id: vllm_samplingparams_raises_temperature_lt_0p0
  engine: vllm
  library: vllm
  invariant_under_test: 'SamplingParams._verify_args: raises when temperature < 0.0'
  severity: error
  native_type: vllm.SamplingParams
  miner_source:
    path: sampling_params.py
    method: _verify_args
    line_at_scan: 368
  match:
    engine: vllm
    fields:
      vllm.sampling.temperature:
        <: 0.0
  kwargs_positive:

...<truncated>...
```

### Response

```
added_by_llm_verifier:
  - id: vllm_samplingparams_raises_temperature_lt_0p1
    engine: vllm
    library: vllm
    invariant_under_test: 'SamplingParams._verify_args: raises when temperature < 0.1'
    severity: error
    native_type: vllm.SamplingParams
    miner_source:
      path: sampling_params.py
      method: _verify_args
      line_at_scan: 368
    match:
      engine: vllm
      fields:
        vllm.sampling.temperature:
          <: 0.1
    kwargs_positive:
      temperature: -1.0
    kwargs_negative:
      temperature: 0.1
    expected_outcome:
      outcome: error
      emission_channel: none
      normalised_fields: []
    message_template: temperature must be non-negative, got {temperature}.
    references:
      - sampling_params.py:368 (vllm.SamplingParams._verify_args)
    added_by: llm_verifier
    flagged_for_review: true

  - id: vllm_loraconfig_raises_max_cpu_loras_lt_ref_max_loras
    engine: vllm
    library: vllm
    invariant_under_test: 'LoRAConfig.__post_init__: raises when max_cpu_loras present True AND max_cpu_loras < @max_loras'
    severity: error
    native_type: vllm.config.LoRAConfig
    miner_source:
      path: config.py
      method: __post_init__
      line_at_scan: 2198
    match:
      engine: vllm
      fields:
        vllm.engine.lora.max_cpu_loras:
          present: true
          <: '@max_loras'
    kwargs_positive:
      max_lora_rank: 8
      max_loras: 2
      max_cpu_loras: 1
    kwargs_negative:
      max_lora_rank: 8
      max_loras: 1
      max_cpu_loras: 2
    expected_outcome:
      outcome: error
      emission_channel: none
      normalised_fields: []
    message_template: max_cpu_loras ({self.max_cpu_loras}) must be >= max_loras ({self.max_loras})
    references:
      - config.py:2198 (vllm.config.LoRAConfig.__post_init__)
    added_by: llm_verifier
    flagged_for_review: true

flagged_spurious_in_deterministic:
  - id: vllm_samplingparams_raises_max_tokens_lt_1
    reason: The invariant is too restrictive, as max_tokens can be set to 0 for certain use cases.

missed_diagnosis:
  - id: vllm_samplingparams_raises_temperature_lt_0p1
    why_missed: AST limitation, the deterministic miner missed this invariant because it only checks for exact matches in the source code and does not account for implicit comparisons.
```
