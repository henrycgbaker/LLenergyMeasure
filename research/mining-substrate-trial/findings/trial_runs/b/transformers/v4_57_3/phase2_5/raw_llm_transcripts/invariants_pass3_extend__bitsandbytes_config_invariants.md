# invariants_pass3_extend extraction transcript: bitsandbytes_config_invariants

- chunk_description: BitsAndBytesConfig.__init__ type checks (load_in_4bit, bnb_4bit_*, llm_int8_*)
- expected_namespaces: ['transformers']
- attempts: 1
- elapsed_sec: 97.89
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser doing PASS 3 (extension) of multi-pass
invariant extraction. PASS 1 already extracted N invariants from
transformers v4.57.3 for ONE chunk of source. Your job is to
identify ADDITIONAL invariants PASS 1 missed.

INPUT 1 - PASS 1 EXTRACTED INVARIANTS (already known; do NOT re-emit):

invariants:
- id: transformers_load_in_4bit_and_load_in_8bit_both_true
  severity: error
  match:
    engine: transformers
    fields:
      transformers.load_in_4bit: true
      transformers.load_in_8bit: true
  invariant_under_test: BitsAndBytesConfig.__init__ flags load_in_4bit and load_in_8bit
    both True
- id: transformers_bnb_4bit_compute_dtype_not_string_or_torch_dtype
  severity: error
  match:
    engine: transformers
    fields:
      transformers.bnb_4bit_compute_dtype:
        present: true
        type_is_not:
        - str
        - torch.dtype
  invariant_under_test: BitsAndBytesConfig.__init__ flags bnb_4bit_compute_dtype not
    string or torch.dtype
- id: transformers_bnb_4bit_quant_storage_not_string_or_torch_dtype
  severity: error
  match:
    engine: transformers
    fields:
      transformers.bnb_4bit_quant_storage:
        present: true
        type_is_not:
        - str
        - torch.dtype
  invariant_under_test: BitsAndBytesConfig.__init__ flags bnb_4bit_quant_storage not
    string or torch.dtype
- id: transformers_bnb_4bit_quant_storage_invalid_string
  severity: error
  match:
    engine: transformers
    fields:
      transformers.bnb_4bit_quant_storage:
        present: true
        not_in:
        - float16
        - float32
        - int8
        - uint8
        - float64
        - bfloat16
  invariant_under_test: BitsAndBytesConfig.__init__ flags bnb_4bit_quant_storage invalid
    string
- id: transformers_load_in_4bit_not_boolean
  severity: error
  match:
    engine: transformers
    fields:
      transformers.load_in_4bit:
        present: true
        type_is_not:
        - bool
  invariant_under_test: BitsAndBytesConfig.__init__ flags load_in_4bit not boolean
- id: transformers_load_in_8bit_not_boolean
  severity: error
  match:
    engine: transformers
    fields:
      transformers.load_in_8bit:
        present: true
        type_is_not:
        - bool
  invariant_under_test: BitsAndBytesConfig.__init__ flags load_in_8bit not boolean
- id: transformers_llm_int8_threshold_not_float
  severity: error
  match:
    engine: transformers
    fields:
      transformers.llm_int8_threshold:
        present: true
        type_is_not:
        - float
  invariant_under_test: BitsAndBytesConfig.__init__ flags llm_int8_threshold not float


INPUT 2 - PASS 2 FLAGGED ISSUES (for context; consider whether they hint
at related misses):

flagged:
- id: transformers_bnb_4bit_quant_type_not_string_or_torch_dtype
  reason: This invariant is not present in the source code.
  fix: drop


INPUT 3 - THE SOURCE PASS 1 READ:

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
            This corresponds to the outlier threshold for outlier detection as described in `LLM.int8() : 8-bit Matrix
            Multiplication for Transformers at Scale` paper: https://huggingface.co/papers/2208.07339 Any hidden states value
            that is above this threshold will be considered an outlier and the operation on those values will be done
            in fp16. Values are usually normally distributed, that is, most values are in the range [-3.5, 3.5], but
            there are some exceptional systematic outliers that are very differently distributed for large models.
            These outliers are often in the interval [-60, -6] or [6, 60]. Int8 quantization works well for values of
            magnitude ~5, but beyond that, there is a significant performance penalty. A good default threshold is 6,
            but a lower threshold might be needed for more unstable models (small models, fine-tuning).
        llm_int8_skip_modules (`list[str]`, *optional*):
            An explicit list of the modules that we do not want to convert in 8-bit. This is useful for models such as
            Jukebox that has several heads in different places and not necessarily at the last position. For example
            for `CausalLM` models, the last `lm_head` is kept in its original `dtype`.
        llm_int8_enable_fp32_cpu_offload (`bool`, *optional*, defaults to `False`):
            This flag is used for advanced use cases and users that are aware of this feature. If you want to split
            your model in different parts and run some parts in int8 on GPU and some parts in fp32 on CPU, you can use
            this flag. This is useful for offloading large models such as `google/flan-t5-xxl`. Note that the int8
            operations will not be run on CPU.
        llm_int8_has_fp16_weight (`bool`, *optional*, defaults to `False`):
            This flag runs LLM.int8() with 16-bit main weights. This is useful for fine-tuning as the weights do not
            have to be converted back and forth for the backward pass.
        bnb_4bit_compute_dtype (`torch.dtype` or str, *optional*, defaults to `torch.float32`):
            This sets the computational type which might be different than the input type. For example, inputs might be
            fp32, but computation can be set to bf16 for speedups.
        bnb_4bit_quant_type (`str`,  *optional*, defaults to `"fp4"`):
            This sets the quantization data type in the bnb.nn.Linear4Bit layers. Options are FP4 and NF4 data types
            which are specified by `fp4` or `nf4`.
        bnb_4bit_use_double_quant (`bool`, *optional*, defaults to `False`):
            This flag is used for nested quantization where the quantization constants from the first quantization are
            quantized again.
        bnb_4bit_quant_storage (`torch.dtype` or str, *optional*, defaults to `torch.uint8`):
            This sets the storage type to pack the quantized 4-bit params.
        kwargs (`dict[str, Any]`, *optional*):
            Additional parameters from which to initialize the configuration object.
    """

    def __init__(
        self,
        load_in_8bit=False,
        load_in_4bit=False,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=None,
        llm_int8_enable_fp32_cpu_offload=False,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=None,
        bnb_4bit_quant_type="fp4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_storage=None,
        **kwargs,
    ):
        self.quant_method = QuantizationMethod.BITS_AND_BYTES

        if load_in_4bit and load_in_8bit:
            raise ValueError("load_in_4bit and load_in_8bit are both True, but only one can be used at the same time")

        self._load_in_8bit = load_in_8bit
        self._load_in_4bit = load_in_4bit
        self
...<truncated>...
```

### Response

```
invariants:
- id: transformers_llm_int8_skip_modules_not_list_of_strings
  engine: transformers
  library: transformers
  invariant_under_test: BitsAndBytesConfig.__init__ flags llm_int8_skip_modules not list of strings
  severity: error
  native_type: transformers.BitsAndBytesConfig
  miner_source:
    path: transformers/quantization_bitsandbytes.py
    method: __init__
    line_at_scan: 123
  match:
    engine: transformers
    fields:
      transformers.llm_int8_skip_modules:
        present: true
        type_is_not:
        - list
  kwargs_positive:
    llm_int8_skip_modules: "not a list"
  kwargs_negative:
    llm_int8_skip_modules: ["module1", "module2"]
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: 'llm_int8_skip_modules must be a list of strings'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: transformers_bnb_4bit_quant_type_not_fp4_or_nf4
  engine: transformers
  library: transformers
  invariant_under_test: BitsAndBytesConfig.__init__ flags bnb_4bit_quant_type not fp4 or nf4
  severity: error
  native_type: transformers.BitsAndBytesConfig
  miner_source:
    path: transformers/quantization_bitsandbytes.py
    method: __init__
    line_at_scan: 145
  match:
    engine: transformers
    fields:
      transformers.bnb_4bit_quant_type:
        present: true
        not_in:
        - fp4
        - nf4
  kwargs_positive:
    bnb_4bit_quant_type: "not fp4 or nf4"
  kwargs_negative:
    bnb_4bit_quant_type: "fp4"
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: 'bnb_4bit_quant_type must be a string (one of "fp4" or "nf4")'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: transformers_bnb_4bit_use_double_quant_not_boolean
  engine: transformers
  library: transformers
  invariant_under_test: BitsAndBytesConfig.__init__ flags bnb_4bit_use_double_quant not boolean
  severity: error
  native_type: transformers.BitsAndBytesConfig
  miner_source:
    path: transformers/quantization_bitsandbytes.py
    method: __init__
    line_at_scan: 155
  match:
    engine: transformers
    fields:
      transformers.bnb_4bit_use_double_quant:
        present: true
        type_is_not:
        - bool
  kwargs_positive:
    bnb_4bit_use_double_quant: "not a boolean"
  kwargs_negative:
    bnb_4bit_use_double_quant: True
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: 'bnb_4bit_use_double_quant must be a boolean'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: transformers_llm_int8_enable_fp32_cpu_offload_not_boolean
  engine: transformers
  library: transformers
  invariant_under_test: BitsAndBytesConfig.__init__ flags llm_int8_enable_fp32_cpu_offload not boolean
  severity: error
  native_type: transformers.BitsAndBytesConfig
  miner_source:
    path: transformers/quantization_bitsandbytes.py
    method: __init__
    line_at_scan: 165
  match:
    engine: transformers
    fields:
      transformers.llm_int8_enable_fp32_cpu_offload:
        present: true
        type_is_not:
        - bool
  kwargs_positive:
    llm_int8_enable_fp32_cpu_offload: "not a boolean"
  kwargs_negative:
    llm_int8_enable_fp32_cpu_offload: True
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: 'llm_int8_enable_fp32_cpu_offload must be a boolean'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

- id: transformers_llm_int8_has_fp16_weight_not_boolean
  engine: transformers
  library: transformers
  invariant_under_test: BitsAndBytesConfig.__init__ flags llm_int8_has_fp16_weight not boolean
  severity: error
  native_type: transformers.BitsAndBytesConfig
  miner_source:
    path: transformers/quantization_bitsandbytes.py
    method: __init__
    line_at_scan: 175
  match:
    engine: transformers
    fields:
      transformers.llm_int8_has_fp16_weight:
        present: true
        type_is_not:
        - bool
  kwargs_positive:
    llm_int8_has_fp16_weight: "not a boolean"
  kwargs_negative:
    llm_int8_has_fp16_weight: True
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: 'llm_int8_has_fp16_weight must be a boolean'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'
```
