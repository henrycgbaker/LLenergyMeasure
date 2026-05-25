# invariants_pass2_verify extraction transcript: bitsandbytes_config_invariants

- chunk_description: BitsAndBytesConfig.__init__ type checks (load_in_4bit, bnb_4bit_*, llm_int8_*)
- expected_namespaces: ['transformers']
- attempts: 1
- elapsed_sec: 55.38
- failure_modes: []
- schema_errors: []
- parsed: yes

## Attempt 1

### Prompt

```
You are a code analyser doing PASS 2 (verification) of multi-pass
invariant extraction. PASS 1 already extracted invariants from
transformers v4.57.6 for ONE chunk of source. Your job is to
REVIEW each emitted invariant against the source and flag any that
look WRONG.

INPUT 1 - PASS 1 EXTRACTED INVARIANTS (the candidate list):

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
- id: transformers_llm_int8_skip_modules_not_list_of_strings
  severity: error
  match:
    engine: transformers
    fields:
      transformers.llm_int8_skip_modules:
        present: true
        type_is_not:
        - list
  invariant_under_test: BitsAndBytesConfig.__init__ flags llm_int8_skip_modules not
    list of strings
- id: transformers_llm_int8_enable_fp32_cpu_offload_not_boolean
  severity: error
  match:
    engine: transformers
    fields:
      transformers.llm_int8_enable_fp32_cpu_offload:
        present: true
        type_is_not:
        - bool
  invariant_under_test: BitsAndBytesConfig.__init__ flags llm_int8_enable_fp32_cpu_offload
    not boolean
- id: transformers_llm_int8_has_fp16_weight_not_boolean
  severity: error
  match:
    engine: transformers
    fields:
      transformers.llm_int8_has_fp16_weight:
        present: true
        type_is_not:
        - bool
  invariant_under_test: BitsAndBytesConfig.__init__ flags llm_int8_has_fp16_weight
    not boolean
- id: transformers_bnb_4bit_quant_type_not_string
  severity: error
  match:
    engine: transformers
    fields:
      transformers.bnb_4bit_quant_type:
        present: true
        type_is_not:
        - str
  invariant_under_test: BitsAndBytesConfig.__init__ flags bnb_4bit_quant_type not
    string
- id: transformers_bnb_4bit_use_double_quant_not_boolean
  severity: error
  match:
    engine: transformers
    fields:
      transformers.bnb_4bit_use_double_quant:
        present: true
        type_is_not:
        - bool
  invariant_under_test: BitsAndBytesConfig.__init__ flags bnb_4bit_use_double_quant
    not boolean


INPUT 2 - THE SOURCE PASS 1 READ:

=== CONTEXT ===
BitsAndBytesConfig.__init__ has 8+ `if not isinstance(...): raise TypeError(...)` style type checks. Each is a separate invariant with severity=error, predicate=type_is_not. The namespace should be `transformers` (NOT transformers.sampling, since these are engine_params).

=== SOURCE: BitsAndBytesConfig (full) ===
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
            This sets the quantization data typ
...<truncated>...
```

### Response

```
confirmed:
- transformers_load_in_4bit_and_load_in_8bit_both_true
- transformers_bnb_4bit_compute_dtype_not_string_or_torch_dtype
- transformers_bnb_4bit_quant_storage_invalid_string
- transformers_load_in_4bit_not_boolean
- transformers_load_in_8bit_not_boolean
- transformers_llm_int8_threshold_not_float
- transformers_llm_int8_skip_modules_not_list_of_strings
- transformers_llm_int8_enable_fp32_cpu_offload_not_boolean
- transformers_llm_int8_has_fp16_weight_not_boolean
- transformers_bnb_4bit_quant_type_not_string
- transformers_bnb_4bit_use_double_quant_not_boolean

flagged:
- id: transformers_bnb_4bit_quant_storage_not_string_or_torch_dtype
  reason: Source allows `bnb_4bit_quant_storage` to be a string or torch.dtype, but invariant says it's not.
  fix: correct_predicate:type_is
```
