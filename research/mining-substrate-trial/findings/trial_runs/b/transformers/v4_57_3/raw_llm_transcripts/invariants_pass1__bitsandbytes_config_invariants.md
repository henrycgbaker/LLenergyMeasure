# invariants_pass1 extraction transcript: bitsandbytes_config_invariants

- chunk_description: BitsAndBytesConfig.__init__ type checks (load_in_4bit, bnb_4bit_*, llm_int8_*)
- expected_namespaces: ['transformers']
- attempts: 1
- elapsed_sec: 189.79
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser extracting validation invariants from
transformers library v4.57.3. An "invariant" is a rule the
library checks at runtime - typically `if <predicate>: raise ValueError(...)`
or `if <predicate>: minor_issues[...] = ...` (which surfaces as a
warning).

You will be shown ONE CHUNK of validation source. Extract every
invariant visible in this chunk.

OUTPUT FORMAT: YAML document matching EXACTLY this shape:

invariants:
- id: <snake_case_unique_id>
  engine: transformers
  library: transformers
  invariant_under_test: <one-line: what the library checks here>
  severity: <error|dormant|warning>
  native_type: transformers.GenerationConfig
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: <validate|__init__>
    line_at_scan: <approximate line number if visible>
  match:
    engine: transformers
    fields:
      transformers.<field>: <value or predicate>
  kwargs_positive:
    <field>: <value that TRIGGERS the invariant>
  kwargs_negative:
    <field>: <value that does NOT trigger>
  expected_outcome:
    outcome: <error|dormant_announced|warning>
    emission_channel: <none|logger_warning_once|logger_warning>
    normalised_fields: []
  message_template: '<the exact error/warning string from source, with {} placeholders preserved>'
  added_by: llm_miner
  added_at: '2026-05-25'

INVARIANT TYPES TO EXTRACT (one per `if ... :` block typically):

1. ERROR (raises ValueError at construction or validate()):
   - Field value not in allowed enum -> severity: error, predicate: not_in
   - Field type mismatch (e.g. `not isinstance(x, T)`) -> severity: error, predicate: type_is_not
   - Field value out of range (e.g. <= 0) -> severity: error, predicate: gt/lt
   - Cross-field invalid combo (e.g. num_return_sequences > num_beams) -> severity: error

2. DORMANT (logs warning, parameter silently ignored or normalised):
   - Sampling-only param set when do_sample=False -> severity: dormant
   - Beam-only param set when num_beams=1 -> severity: dormant
   - Cache-related dormancy -> severity: dormant

3. WARNING (logs, execution continues with user value):
   - pad_token_id < 0 -> severity: warning
   - Other non-blocking minor_issues entries -> severity: warning

PREDICATE FORMS for the `match.fields` block (use the EXACT keys shown):
- Exact value:         `transformers.field: value`
- Not in list:         `transformers.field: {present: true, not_in: [a, b]}`
- Not equal:           `transformers.field: {present: true, not_equal: value}`
- Greater than:        `transformers.field: {'>': value}`
- Less than:           `transformers.field: {'<': value}`
- Greater or equal:    `transformers.field: {'>=': value}`
- Less or equal:       `transformers.field: {'<=': value}`
- Type not in:         `transformers.field: {present: true, type_is_not: [TypeName]}`
- Presence (any value):`transformers.field: {present: true}`

CRITICAL RULES:
1. Return ONLY the YAML document. NO markdown code fences (no ```yaml).
   NO commentary, no preamble. First character must be `i` (from
   `invariants:`).
2. Extract ONLY invariants VISIBLE in the source below. Do not invent.
3. Use snake_case_with_engine_prefix for `id` (e.g. `transformers_cache_implementation_not_in_allowlist`).
4. Each `if <cond>: raise / minor_issues[...] = ` block = ONE invariant.
5. Set `severity: error` when the source has `raise ValueError(...)`.
   Set `severity: dormant` when the source assigns to `minor_issues`
   AND comments / context indicate the parameter is silently ignored.
   Set `severity: warning` when the source assigns to `minor_issues`
   AND there's no silent-ignore semantics.
6. For `kwargs_positive`: provide a concrete dict that WOULD trigger
   the invariant (so a downstream validator can replay it).
7. For `kwargs_negative`: provide a concrete dict that would NOT
   trigger (so the negative case is checkable).
8. `message_template`: the EXACT f-string literal from `raise` /
   `minor_issues[...]`. Preserve `{}` placeholders. Do NOT
   substitute concrete values.

FEW-SHOT EXAMPLES (from transformers v4.57.3 reference):

Example 1 (ERROR, enum violation):
Source: ``if self.early_stopping not in {True, False, "never"}: raise ValueError(f"`early_stopping` must be a boolean or 'never', but is {self.early_stopping}.")``
Emit:
- id: transformers_early_stopping_not_in_allowlist
  engine: transformers
  invariant_under_test: GenerationConfig.validate flags `early_stopping` not in allowlist
  severity: error
  match:
    engine: transformers
    fields:
      transformers.sampling.early_stopping: {present: true, not_in: [true, false, "never"]}
  kwargs_positive:
    early_stopping: "invalid"
  kwargs_negative:
    early_stopping: true
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "`early_stopping` must be a boolean or 'never', but is {self.early_stopping}."

Example 2 (DORMANT, sampling-only when greedy):
Source (under `if self.do_sample is False:`):
``if self.temperature is not None and self.temperature != 1.0:
    minor_issues["temperature"] = greedy_wrong_parameter_msg.format(...)``
Emit:
- id: transformers_temperature_set_when_do_sample_false
  engine: transformers
  invariant_under_test: GenerationConfig.validate flags `temperature` set when do_sample=False
  severity: dormant
  match:
    engine: transformers
    fields:
      transformers.sampling.do_sample: false
  kwargs_positive:
    do_sample: false
    temperature: 0.5
  kwargs_negative:
    do_sample: true
    temperature: 0.5
  expected_outcome:
    outcome: dormant_announced
    emission_channel: logger_warning_once
    normalised_fields: []
  message_template: "`do_sample` is set to `False`. However, `{flag_name}` is set to `{flag_value}` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `{flag_name}`."

Example 3 (CROSS-FIELD ERROR):
Source: ``if self.num_return_sequences > self.num_beams: raise ValueError(...)``
Emit:
- id: transformers_num_return_sequences_gt_num_beams
  engine: transformers
  invariant_under_test: GenerationConfig.validate flags num_return_sequences > num_beams
  severity: error
  match:
    engine: transformers
    fields:
      transformers.sampling.num_return_sequences: {'>': 1}
  kwargs_positive:
    num_beams: 2
    num_return_sequences: 3
  kwargs_negative:
    num_beams: 5
    num_return_sequences: 2
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "`num_return_sequences` ({self.num_return_sequences}) has to be smaller or equal to `num_beams` ({self.num_beams})."

=== CONTEXT ===
BitsAndBytesConfig.__init__ has 8+ `if not isinstance(...): raise TypeError(...)` style type checks. Each is a separate invariant with severity=error, predicate=type_is_not. The namespace should be `transformers` (NOT transformers.sampling, since these are engine_params).

=== SOURCE: BitsAndBytesConfig (full) ===
@dataclass
class BitsAndBytesConfig(QuantizationConfigMixin):
    """
    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `bitsandbytes`.

    This replaces `load_in_8bit` or `load_in_4bit`therefore both options are mutually exclusive.

    Currently only supports `LLM.int8()`, `FP4`, and `NF4` quantization. If more methods are added to `bitsandbytes`,
    then more arguments will be added to this class.

    Args:
        load_in_8bit (`bool`, *optional*, defaults to `False`):
            This flag is used to enable 8-bit quantization with LLM.int8().
        load_in_4bit (`bool`, *optional*, defaults to `False`):
            This flag is used to enable 4-bit quantization by replacing the Linear layers with FP4/NF4 layers from
            `bitsandbytes`.
        llm_int8_threshold (`float`, *optional*, defaults to 6.0):
            This corresponds to the outlier threshold for outlier detection as described in `
...<truncated>...
```

### Response

```
invariants:
- id: transformers_load_in_4bit_and_load_in_8bit_both_true
  engine: transformers
  library: transformers
  invariant_under_test: BitsAndBytesConfig.__init__ flags load_in_4bit and load_in_8bit both True
  severity: error
  native_type: transformers.BitsAndBytesConfig
  miner_source:
    path: transformers/quantization/configuration_bitsandbytes.py
    method: __init__
    line_at_scan: 34
  match:
    engine: transformers
    fields:
      transformers.load_in_4bit: true
      transformers.load_in_8bit: true
  kwargs_positive:
    load_in_4bit: true
    load_in_8bit: true
  kwargs_negative:
    load_in_4bit: false
    load_in_8bit: true
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "load_in_4bit and load_in_8bit are both True, but only one can be used at the same time"
  added_by: llm_miner
  added_at: '2026-05-25'

- id: transformers_bnb_4bit_compute_dtype_not_string_or_torch_dtype
  engine: transformers
  library: transformers
  invariant_under_test: BitsAndBytesConfig.__init__ flags bnb_4bit_compute_dtype not string or torch.dtype
  severity: error
  native_type: transformers.BitsAndBytesConfig
  miner_source:
    path: transformers/quantization/configuration_bitsandbytes.py
    method: __init__
    line_at_scan: 55
  match:
    engine: transformers
    fields:
      transformers.bnb_4bit_compute_dtype: {present: true, type_is_not: [str, torch.dtype]}
  kwargs_positive:
    bnb_4bit_compute_dtype: 123
  kwargs_negative:
    bnb_4bit_compute_dtype: "float32"
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "bnb_4bit_compute_dtype must be a string or a torch.dtype"
  added_by: llm_miner
  added_at: '2026-05-25'

- id: transformers_bnb_4bit_quant_storage_not_string_or_torch_dtype
  engine: transformers
  library: transformers
  invariant_under_test: BitsAndBytesConfig.__init__ flags bnb_4bit_quant_storage not string or torch.dtype
  severity: error
  native_type: transformers.BitsAndBytesConfig
  miner_source:
    path: transformers/quantization/configuration_bitsandbytes.py
    method: __init__
    line_at_scan: 73
  match:
    engine: transformers
    fields:
      transformers.bnb_4bit_quant_storage: {present: true, type_is_not: [str, torch.dtype]}
  kwargs_positive:
    bnb_4bit_quant_storage: 123
  kwargs_negative:
    bnb_4bit_quant_storage: "float32"
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "bnb_4bit_quant_storage must be a string or a torch.dtype"
  added_by: llm_miner
  added_at: '2026-05-25'

- id: transformers_bnb_4bit_quant_storage_invalid_string
  engine: transformers
  library: transformers
  invariant_under_test: BitsAndBytesConfig.__init__ flags bnb_4bit_quant_storage invalid string
  severity: error
  native_type: transformers.BitsAndBytesConfig
  miner_source:
    path: transformers/quantization/configuration_bitsandbytes.py
    method: __init__
    line_at_scan: 81
  match:
    engine: transformers
    fields:
      transformers.bnb_4bit_quant_storage: {present: true, not_in: ["float16", "float32", "int8", "uint8", "float64", "bfloat16"]}
  kwargs_positive:
    bnb_4bit_quant_storage: "invalid"
  kwargs_negative:
    bnb_4bit_quant_storage: "float32"
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "`bnb_4bit_quant_storage` must be a valid string (one of 'float16', 'float32', 'int8', 'uint8', 'float64', 'bfloat16') "
  added_by: llm_miner
  added_at: '2026-05-25'

- id: transformers_load_in_4bit_not_boolean
  engine: transformers
  library: transformers
  invariant_under_test: BitsAndBytesConfig.__init__ flags load_in_4bit not boolean
  severity: error
  native_type: transformers.BitsAndBytesConfig
  miner_source:
    path: transformers/quantization/configuration_bitsandbytes.py
    method: __init__
    line_at_scan: 104
  match:
    engine: transformers
    fields:
      transformers.load_in_4bit: {present: true, type_is_not: [bool]}
  kwargs_positive:
    load_in_4bit: "true"
  kwargs_negative:
    load_in_4bit: true
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "load_in_4bit must be a boolean"
  added_by: llm_miner
  added_at: '2026-05-25'

- id: transformers_load_in_8bit_not_boolean
  engine: transformers
  library: transformers
  invariant_under_test: BitsAndBytesConfig.__init__ flags load_in_8bit not boolean
  severity: error
  native_type: transformers.BitsAndBytesConfig
  miner_source:
    path: transformers/quantization/configuration_bitsandbytes.py
    method: __init__
    line_at_scan: 113
  match:
    engine: transformers
    fields:
      transformers.load_in_8bit: {present: true, type_is_not: [bool]}
  kwargs_positive:
    load_in_8bit: "true"
  kwargs_negative:
    load_in_8bit: true
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "load_in_8bit must be a boolean"
  added_by: llm_miner
  added_at: '2026-05-25'

- id: transformers_llm_int8_threshold_not_float
  engine: transformers
  library: transformers
  invariant_under_test: BitsAndBytesConfig.__init__ flags llm_int8_threshold not float
  severity: error
  native_type: transformers.BitsAndBytesConfig
  miner_source:
    path: transformers/quantization/configuration_bitsandbytes.py
    method: __init__
    line_at_scan: 122
  match:
    engine: transformers
    fields:
      transformers.llm_int8_threshold: {present: true, type_is_not: [float]}
  kwargs_positive:
    llm_int8_threshold: "6.0"
  kwargs_negative:
    llm_int8_threshold: 6.0
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "llm_int8_threshold must be a float"
  added_by: llm_miner
  added_at: '2026-05-25'

- id: transformers_llm_int8_skip_modules_not_list_of_strings
  engine: transformers
  library: transformers
  invariant_under_test: BitsAndBytesConfig.__init__ flags llm_int8_skip_modules not list of strings
  severity: error
  native_type: transformers.BitsAndBytesConfig
  miner_source:
    path: transformers/quantization/configuration_bitsandbytes.py
    method: __init__
    line_at_scan: 131
  match:
    engine: transformers
    fields:
      transformers.llm_int8_skip_modules: {present: true, type_is_not: [list]}
  kwargs_positive:
    llm_int8_skip_modules: "module1"
  kwargs_negative:
    llm_int8_skip_modules: ["module1", "module2"]
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "llm_int8_skip_modules must be a list of strings"
  added_by: llm_miner
  added_at: '2026-05-25'

- id: transformers_llm_int8_enable_fp32_cpu_offload_not_boolean
  engine: transformers
  library: transformers
  invariant_under_test: BitsAndBytesConfig.__init__ flags llm_int8_enable_fp32_cpu_offload not boolean
  severity: error
  native_type: transformers.BitsAndBytesConfig
  miner_source:
    path: transformers/quantization/configuration_bitsandbytes.py
    method: __init__
    line_at_scan: 140
  match:
    engine: transformers
    fields:
      transformers.llm_int8_enable_fp32_cpu_offload: {present: true, type_is_not: [bool]}
  kwargs_positive:
    llm_int8_enable_fp32_cpu_offload: "true"
  kwargs_negative:
    llm_int8_enable_fp32_cpu_offload: true
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "llm_int8_enable_fp32_cpu_offload must be a boolean"
  added_by: llm_miner
  added_at: '2026-05-25'

- id: transformers_llm_int8_has_fp16_weight_not_boolean
  engine: transformers
  library: transformers
  invariant_under_test: BitsAndBytesConfig.__init__ flags llm_int8_has_fp16_weight not boolean
  severity: error
  native_type: transformers.BitsAndBytesConfig
  miner_source:
    path: transformers/quantization/configuration_bitsandb
...<truncated>...
```
