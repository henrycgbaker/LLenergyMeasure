# hybrid extraction transcript: hybrid_d_ab_vllm_extension

- chunk_description: Hybrid d-ab on vllm: (a) output + source -> extension
- expected_namespaces: ['vllm.*']
- attempts: 3
- elapsed_sec: 433.19
- failure_modes: ['parse_failure_after_retries']
- schema_errors: []
- parsed: no

## Attempt 1

### Prompt

```
You are a code analyser working as a SECOND PASS on top of a
deterministic miner's output. The deterministic miner has already
extracted invariants from vllm v0.7.3; your job is to
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
  - id: vllm_samplingparams_dormant_temperature_lt_max_temp
    engine: vllm
    library: vllm
    invariant_under_test: 'SamplingParams.__post_init__: marks dormant when temperature < _MAX_TEMP'
    severity: dormant
    native_type: vllm.SamplingParams
    miner_source:
      path: sampling_params.py
      method: __post_init__
      line_at_scan: 311
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
      outcome: dormant_silent
      emission_channel: none
      normalised_fields:
      - temperature
    message_template: null
    references:
    - sampling_params.py:311 (vllm.SamplingParams.__post_init__)
    added_by: llm_verifier
    flagged_for_review: true

  - id: vllm_modelconfig_raises_unknown_tokenizer_mode
    engine: vllm
    library: vllm
    invariant_under_test: 'ModelConfig._verify_tokenizer_mode: raises when tokenizer_mode not_in ["auto", "slow", "mistral", "custom"]'
    severity: error
    native_type: vllm.ModelConfig
    miner_source:
      path: config.py
      method: _verify_tokenizer_mode
      line_at_scan: 1087
    match:
      engine: vllm
      fields:
        vllm.tokenizer_mode:
          not_in:
          - auto
          - slow
          - mistral
          - custom
    kwargs_positive:
      tokenizer_mode: invalid
    kwargs_negative:
      tokenizer_mode: auto
    expected_outcome:
      outcome: error
      emission_channel: none
      normalised_fields: []
    message_template: Unknown tokenizer mode: {self.tokenizer_mode}. Must be either 'auto', 'slow', 'mistral' or 'custom'.
    references:
    - config.py:1087 (vllm.ModelConfig._verify_tokenizer_mode)
    added_by: llm_verifier
    flagged_for_review: true

  - id: vllm_modelconfig_raises_unknown_quantization_method
    engine: vllm
    library: vllm
    invariant_under_test: 'ModelConfig._verify_quantization: raises when quantization not_in QUANTIZATION_METHODS'
    severity: error
    native_type: vllm.ModelConfig
    miner_source:
      path: config.py
      method: _verify_quantization
      line_at_scan: 1087
    match:
      engine: vllm
      fields:
        vllm.quantization:
          not_in: QUANTIZATION_METHODS
    kwargs_positive:
      quantization: invalid
    kwargs_negative:
      quantization: fp8
    expected_outcome:
      outcome: error
      emission_channel: none
      normalised_fields: []
    message_template: Unknown quantization method: {self.quantization}. Must be one of {QUANTIZATION_METHODS}.
    references:
    - config.py:1087 (vllm.ModelConfig._verify_quantization)
    added_by: llm_verifier
    flagged_for_review: true

flagged_spurious_in_deterministic:
  - id: vllm_samplingparams_raises_best_of_lt_ref_n
    reason: The invariant is too restrictive, as it does not account for the case where best_of is set to n.

missed_diagnosis:
  - id: vllm_modelconfig_unknown_tokenizer_mode
    why_missed: AST limitation - the deterministic miner did not correctly parse the tokenizer_mode validation logic.
  - id: vllm_modelconfig_unknown_quantization_method
    why_missed: The deterministic miner missed this invariant due to a limitation in its handling of complex conditional statements.
```

## Attempt 2

### Prompt

```
You are a code analyser working as a SECOND PASS on top of a
deterministic miner's output. The deterministic miner has already
extracted invariants from vllm v0.7.3; your job is to
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
  - id: vllm_modelconfig_raises_unknown_tokenizer_mode
    engine: vllm
    library: vllm
    invariant_under_test: 'ModelConfig._verify_tokenizer_mode: raises when tokenizer_mode not_in ["auto", "slow", "mistral", "custom"]'
    severity: error
    native_type: vllm.ModelConfig
    miner_source:
      path: config.py
      method: _verify_tokenizer_mode
      line_at_scan: 1087
    match:
      engine: vllm
      fields:
        vllm.tokenizer_mode:
          not_in:
            - auto
            - slow
            - mistral
            - custom
    kwargs_positive:
      tokenizer_mode: __vllm_static_synth__
    kwargs_negative:
      tokenizer_mode: auto
    expected_outcome:
      outcome: error
      emission_channel: none
      normalised_fields: []
    message_template: Unknown tokenizer mode: {self.tokenizer_mode}. Must be either 'auto', 'slow', 'mistral' or 'custom'.
    references:
      - config.py:1087 (vllm.ModelConfig._verify_tokenizer_mode)
    added_by: llm_verifier
    flagged_for_review: true

  - id: vllm_modelconfig_raises_unknown_quantization_method
    engine: vllm
    library: vllm
    invariant_under_test: 'ModelConfig._verify_quantization: raises when quantization not_in QUANTIZATION_METHODS'
    severity: error
    native_type: vllm.ModelConfig
    miner_source:
      path: config.py
      method: _verify_quantization
      line_at_scan: 1087
    match:
      engine: vllm
      fields:
        vllm.quantization:
          not_in: QUANTIZATION_METHODS
    kwargs_positive:
      quantization: __vllm_static_synth__
    kwargs_negative:
      quantization: fp8
    expected_outcome:
      outcome: error
      emission_channel: none
      normalised_fields: []
    message_template: Unknown quantization method: {self.quantization}. Must be one of QUANTIZATION_METHODS.
    references:
      - config.py:1087 (vllm.ModelConfig._verify_quantization)
    added_by: llm_verifier
    flagged_for_review: true

  - id: vllm_modelconfig_raises_unknown_cache_dtype
    engine: vllm
    library: vllm
    invariant_under_test: 'CacheConfig._verify_cache_dtype: raises when cache_dtype not_in ["auto", "fp8", "fp8_e4m3", "fp8_e5m2"]'
    severity: error
    native_type: vllm.CacheConfig
    miner_source:
      path: config.py
      method: _verify_cache_dtype
      line_at_scan: 1087
    match:
      engine: vllm
      fields:
        vllm.cache_dtype:
          not_in:
            - auto
            - fp8
            - fp8_e4m3
            - fp8_e5m2
    kwargs_positive:
      cache_dtype: __vllm_static_synth__
    kwargs_negative:
      cache_dtype: auto
    expected_outcome:
      outcome: error
      emission_channel: none
      normalised_fields: []
    message_template: Unknown kv cache dtype: {self.cache_dtype}
    references:
      - config.py:1087 (vllm.CacheConfig._verify_cache_dtype)
    added_by: llm_verifier
    flagged_for_review: true

  - id: vllm_schedulerconfig_raises_max_num_batched_tokens_lt_ref_max_model_len_and_enable_chunked_prefill_is_false
    engine: vllm
    library: vllm
    invariant_under_test: 'SchedulerConfig._verify_args: raises when max_num_batched_tokens < @max_model_len AND enable_chunked_prefill is False'
    severity: error
    native_type: vllm.SchedulerConfig
    miner_source:
      path: config.py
      method: _verify_args
      line_at_scan: 1575
    match:
      engine: vllm
      fields:
        vllm.max_num_batched_tokens:
          <: '@max_model_len'
        vllm.enable_chunked_prefill: false
    kwargs_positive:
      max_num_batched_tokens: 100
      max_model_len: 200
      enable_chunked_prefill: false
    kwargs_negative:
      max_num_batched_tokens: 200
      max_model_len: 200
      enable_chunked_prefill: false
    expected_outcome:
      outcome: error
      emission_channel: none
      normalised_fields: []
    message_template: max_num_batched_tokens ({self.max_num_batched_tokens}) is smaller than max_model_len ({self.max_model_len}).
    references:
      - config.py:1575 (vllm.SchedulerConfig._verify_args)
    added_by: llm_verifier
    flagged_for_review: true

  - id: vllm_schedulerconfig_raises_max_num_batched_tokens_lt_ref_max_num_seqs
    engine: vllm
    library: vllm
    invariant_under_test: 'SchedulerConfig._verify_args: raises when max_num_batched_tokens < @max_num_seqs'
    severity: error
    native_type: vllm.SchedulerConfig
    miner_source:
      path: config.py
      method: _verify_args
      line_at_scan: 1585
    match:
      engine: vllm
      fields:
        vllm.max_num_batched_tokens:
          <: '@max_num_seqs'
    kwargs_positive:
      max_num_batched_tokens: 1
      max_num_seqs: 2
    kwargs_negative:
      max_num_batched_tokens: 2
      max_num_seqs: 2
    expected_outcome:
      outcome: error
      emission_channel: none
      normalised_fields: []
    message_template: max_num_batched_tokens ({self.max_num_batched_tokens}) must be greater than or equal to max_num_seqs ({self.max_num_seqs}).
    references:
      - config.py:1585 (vllm.SchedulerConfig._verify_args)
    added_by: llm_verifier
    flagged_for_review: true

  - id: vllm_schedulerconfig_raises_max_num_partial_prefills_lt_1
    engine: vllm
    library: vllm
    invariant_under_test: 'SchedulerConfig._verify_args: raises when max_num_partial_prefills < 1'
    severity: error
    native_type: vllm.SchedulerConfig
    miner_source:
      path: config.py
      method: _verify_args
      line_at_scan: 1604
    match:
      engine: vllm
      fields:
        vllm.max_num_partial_prefills:
          <: 1
    kwargs_positive:
      max_num_partial_prefills: 0
    kwargs_negative:
      max_num_partial_prefills: 1
    expected_outcome:
      outcome: error
      emission_channel: none
      normalised_fields: []
    message_template: max_num_partial_prefills ({max_num_partial_prefills}) must be greater than or equal to 1.
    references:
      - config.py:1604 (vllm.SchedulerConfig._verify_args)
    added_by: llm_verifier
    flagged_for_review: true

flagged_spurious_in_deterministic:
  - id: vllm_samplingparams_raises_best_of_lt_ref_n
    reason: The invariant is not specific enough, as it does not consider the case where best_of is equal to n.

missed_diagnosis:
  - id: vllm_modelconfig_unknown_tokenizer_mode
    why_missed: The deterministic miner missed this invariant because it did not properly handle the `not_in` predicate in the `ModelConfig._verify_tokenizer_mode` method.
```

## Attempt 3

### Prompt

```
You are a code analyser working as a SECOND PASS on top of a
deterministic miner's output. The deterministic miner has already
extracted invariants from vllm v0.7.3; your job is to
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
  - id: vllm_modelconfig_raises_unknown_tokenizer_mode
    engine: vllm
    library: vllm
    invariant_under_test: 'ModelConfig._verify_tokenizer_mode: raises when tokenizer_mode not_in ["auto", "slow", "mistral", "custom"]'
    severity: error
    native_type: vllm.ModelConfig
    miner_source:
      path: config.py
      method: _verify_tokenizer_mode
      line_at_scan: 1087
    match:
      engine: vllm
      fields:
        vllm.tokenizer_mode:
          not_in:
            - auto
            - slow
            - mistral
            - custom
    kwargs_positive:
      tokenizer_mode: unknown
    kwargs_negative:
      tokenizer_mode: auto
    expected_outcome:
      outcome: error
      emission_channel: none
      normalised_fields: []
    message_template: Unknown tokenizer mode: {self.tokenizer_mode}. Must be either 'auto', 'slow', 'mistral' or 'custom'.
    references:
      - config.py:1087 (vllm.ModelConfig._verify_tokenizer_mode)
    added_by: llm_verifier
    flagged_for_review: true

  - id: vllm_modelconfig_raises_unknown_quantization_method
    engine: vllm
    library: vllm
    invariant_under_test: 'ModelConfig._verify_quantization: raises when quantization not_in QUANTIZATION_METHODS'
    severity: error
    native_type: vllm.ModelConfig
    miner_source:
      path: config.py
      method: _verify_quantization
      line_at_scan: 1097
    match:
      engine: vllm
      fields:
        vllm.quantization:
          not_in:
            - fp8
            - marlin
            - modelopt
            - gptq_marlin_24
            - gptq_marlin
            - awq_marlin
            - fbgemm_fp8
            - compressed_tensors
            - compressed-tensors
            - experts_int8
            - quark
    kwargs_positive:
      quantization: unknown
    kwargs_negative:
      quantization: fp8
    expected_outcome:
      outcome: error
      emission_channel: none
      normalised_fields: []
    message_template: Unknown quantization method: {self.quantization}. Must be one of QUANTIZATION_METHODS.
    references:
      - config.py:1097 (vllm.ModelConfig._verify_quantization)
    added_by: llm_verifier
    flagged_for_review: true

  - id: vllm_modelconfig_raises_cuda_graph_not_supported
    engine: vllm
    library: vllm
    invariant_under_test: 'ModelConfig._verify_cuda_graph: raises when model_type in MODEL_NOT_SUPPORT_CUDA_GRAPH and not enforce_eager'
    severity: error
    native_type: vllm.ModelConfig
    miner_source:
      path: config.py
      method: _verify_cuda_graph
      line_at_scan: 1107
    match:
      engine: vllm
      fields:
        vllm.hf_config.model_type:
          in:
            - mllama
        vllm.enforce_eager:
          eq: false
    kwargs_positive:
      hf_config:
        model_type: mllama
      enforce_eager: false
    kwargs_negative:
      hf_config:
        model_type: other
      enforce_eager: true
    expected_outcome:
      outcome: error
      emission_channel: none
      normalised_fields: []
    message_template: CUDA graph is not supported for {self.hf_config.model_type} yet, fallback to the eager mode.
    references:
      - config.py:1107 (vllm.ModelConfig._verify_cuda_graph)
    added_by: llm_verifier
    flagged_for_review: true

  - id: vllm_modelconfig_raises_bnb_quantization_not_supported
    engine: vllm
    library: vllm
    invariant_under_test: 'ModelConfig._verify_bnb_config: raises when quantization is bitsandbytes and not enforce_eager'
    severity: error
    native_type: vllm.ModelConfig
    miner_source:
      path: config.py
      method: _verify_bnb_config
      line_at_scan: 1117
    match:
      engine: vllm
      fields:
        vllm.quantization:
          eq: bitsandbytes
        vllm.enforce_eager:
          eq: false
    kwargs_positive:
      quantization: bitsandbytes
      enforce_eager: false
    kwargs_negative:
      quantization: other
      enforce_eager: true
    expected_outcome:
      outcome: error
      emission_channel: none
      normalised_fields: []
    message_template: CUDA graph is not supported on BitAndBytes 8bit yet, fallback to the eager mode.
    references:
      - config.py:1117 (vllm.ModelConfig._verify_bnb_config)
    added_by: llm_verifier
    flagged_for_review: true

flagged_spurious_in_deterministic:
  - id: vllm_samplingparams_raises_best_of_lt_ref_n
    reason: The invariant is too restrictive, as it does not account for the case where best_of is set to n.

missed_diagnosis:
  - id: vllm_modelconfig_raises_unknown_tokenizer_mode
    why_missed: AST limitation, as the deterministic miner did not correctly parse the tokenizer_mode field in the ModelConfig class.
```
