# Bakeoff B: Local LLM (llama3.1:70b) as Knowledge Extractor

> **CORRECTION (2026-05-25)** — the verdict line below and the "Invariants recall: 0%" scoring are misleading. The invariants call DID produce ~20 valid invariants in the raw output (`bakeoff_B_invariants_raw.txt`, 20 entries via `grep -c '^- id:'`); the scoring harness failed to parse them because the LLM wrapped the YAML in markdown code fences (```yml ... ```). True invariants recall once parsing is fixed is estimated at 40-60%. Schema recall (52.3%) is correctly scored.
>
> Corrected verdict: **MEDIUM** — viable with proper LLM-side setup (chunking, structured output, retry-on-parse-error). Not "not worth pursuing" — worth pursuing AT scale with the engineering investments described in the empirical-trial plan (`.planning/mining-substrate-empirical-trial.md` § "Per-strategy infrastructure needs / (b) Pure OSS LLM").
>
> See `research/mining-substrate-trial/findings/mining_strategy_bakeoff.md` § "Synthesis" for the reconciled assessment.

---

**Original verdict (UNCORRECTED — see above):** LOW - NOT WORTH PURSUING - low overlap; handwritten machinery has more value than feared

Schema recall: 52.3% | Invariants recall: 0.0% | Total wall-clock: 2085s

## Environment

| Item | Value |
|------|-------|
| Model | llama3.1:70b (Q4_K_M, ~42GB) |
| Ollama endpoint | http://localhost:11434 |
| Hardware | A100-40GB |
| Schema call wall-clock | 637s |
| Invariants call wall-clock | 1448s |
| Total wall-clock | 2085s |
| Schema prompt chars | 35,467 |
| Invariants prompt chars | 19,025 |

## Schema Scoring

| Metric | Value |
|--------|-------|
| Ground truth fields | 107 |
| LLM-produced fields | 61 |
| Intersection (overlap) | 56 |
| Recall (GT coverage) | 52.3% |
| Precision | 91.8% |
| Jaccard | 50.0% |
| Type accuracy (overlapping) | 57.1% (32/56) |
| Spurious fields | 5 (8.2% of LLM output) |

### Missed fields (in GT, not in LLM output)

- `engine_params.attn_implementation`
- `engine_params.bnb_4bit_compute_dtype`
- `engine_params.bnb_4bit_quant_storage`
- `engine_params.bnb_4bit_quant_type`
- `engine_params.bnb_4bit_use_double_quant`
- `engine_params.device_map`
- `engine_params.device_mesh`
- `engine_params.dtype`
- `engine_params.from_flax`
- `engine_params.from_tf`
- `engine_params.ignore_mismatched_sizes`
- `engine_params.llm_int8_enable_fp32_cpu_offload`
- `engine_params.llm_int8_has_fp16_weight`
- `engine_params.llm_int8_skip_modules`
- `engine_params.llm_int8_threshold`
- `engine_params.load_in_4bit`
- `engine_params.load_in_8bit`
- `engine_params.max_memory`
- `engine_params.offload_buffers`
- `engine_params.offload_folder`
- `engine_params.output_loading_info`
- `engine_params.pretrained_model_name_or_path`
- `engine_params.quantization_config`
- `engine_params.state_dict`
- `engine_params.tp_plan`
- `engine_params.tp_size`
- `engine_params.use_safetensors`
- `engine_params.variant`
- `engine_params.weights_only`
- `sampling_params._from_model_config`
- ... and 21 more

### Spurious fields (in LLM output, not in GT)

- `engine_params._commit_hash`
- `engine_params._from_auto`
- `engine_params.adapter_kwargs`
- `engine_params.resume_download`
- `engine_params.use_auth_token`

## Invariants Scoring

**PARSE ERROR**: `while scanning for the next token
found character '`' that cannot start any token
  in "<unicode string>", line 1, column 1:
    ```yml
    ^`

Raw LLM output (first 2000 chars):
```
```yml
schema_version: 1.0.0
engine: transformers
engine_version: 4.57.3
mined_at: '2026-05-24T00:00:00+00:00'
invariants:
- id: early_stopping_enum_violation
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.validate()
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: validate
    line_at_scan: 24
  match:
    engine: transformers
    fields:
      transformers.sampling.early_stopping: {not_in: [True, False, "never"]}
  kwargs_positive:
    early_stopping: "invalid"
  kwargs_negative:
    early_stopping: True
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "`early_stopping` must be a boolean or 'never', but is {self.early_stopping}."
  references:
  - GenerationConfig.validate()
  added_by: llm_miner
  added_at: '2026-05-24'

- id: max_new_tokens_range_violation
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.validate()
  severity: error
  native_type: transformers.GenerationConfig
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: validate
    line_at_scan: 28
  match:
    engine: transformers
    fields:
      transformers.sampling.max_new_tokens: {'<=' : 0}
  kwargs_positive:
    max_new_tokens: 0
  kwargs_negative:
    max_new_tokens: 1
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "`max_new_tokens` must be greater than 0, but is {self.max_new_tokens}."
  references:
  - GenerationConfig.validate()
  added_by: llm_miner
  added_at: '2026-05-24'

- id: pad_token_id_range_violation
  engine: transformers
  library: transformers
  invariant_under_test: GenerationConfig.validate()
  severity: warning
  native_type: transformers.GenerationConfig
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: val
```
## Prompts Used

### Schema prompt (truncated to 3000 chars)
```
You are a code analyser. Your task is to extract the parameter schema for the HuggingFace Transformers library (v4.57.3) engine.

You will be given Python source code for:
1. AutoModelForCausalLM.from_pretrained (engine params)
2. BitsAndBytesConfig (quantisation config fields that also surface as engine params)
3. GenerationConfig.__init__ (sampling params - all kwargs)
4. GenerationConfig.__doc__ (docstring describing fields)
5. CompileConfig, WatermarkingConfig, SynthIDTextWatermarkingConfig (nested config classes)

OUTPUT: Return a JSON object matching EXACTLY this schema:

{
  "engine": "transformers",
  "engine_version": "4.57.3",
  "engine_params": {
    "<param_name>": {
      "type": "<json_schema_type>",   // one of: "string","integer","number","boolean","array","object","null", or omit if unknown
      "default": <value_or_null>,     // omit if no default
      "description": "<brief>",       // optional
      "enum": [<values>]              // only if constrained to a set of values
    }
  },
  "sampling_params": {
    "<param_name>": {
      "type": "<json_schema_type>",
      "default": <value_or_null>,
      "description": "<brief>",
      "enum": [<values>]
    }
  },
  "$defs": {
    "CompileConfig": {
      "type": "object",
      "properties": {
        "<field>": {"type": "...", "default": ...}
      }
    }
  }
}

Rules:
- engine_params come from AutoModelForCausalLM.from_pretrained signature + BitsAndBytesConfig fields
- sampling_params come from GenerationConfig.__init__ signature (all kwargs except self, **kwargs)
- $defs should contain CompileConfig properties
- For fields with no type annotation, omit "type" key but include "default" if known
- For None defaults, set "default": null
- Return ONLY valid JSON - no commentary, no markdown code blocks

=== SOURCE: AutoModelForCausalLM.from_pretrained ===
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike[str]], *model_args, **kwargs):
        config = kwargs.pop("config", None)
        trust_remote_code = kwargs.get("trust_remote_code")
        kwargs["_from_auto"] = True
        hub_kwargs_names = [
            "cache_dir",
            "force_download",
            "local_files_only",
            "proxies",
            "resume_download",
            "revision",
            "subfolder",
            "use_auth_token",
            "token",
        ]
        hub_kwargs = {name: kwargs.pop(name) for name in hub_kwargs_names if name in kwargs}
        code_revision = kwargs.pop("code_revision", None)
        commit_hash = kwargs.pop("_commit_hash", None)
        adapter_kwargs = kwargs.pop("adapter_kwargs", None)

        token = hub_kwargs.pop("token", None)
        use_auth_token = hub_kwargs.pop("use_auth_token", None)
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
              
...
```

### Invariants prompt (truncated to 3000 chars)
```
You are a code analyser. Your task is to extract validation invariants from HuggingFace Transformers GenerationConfig.

You will be given:
1. GenerationConfig.validate() source - the primary validation method
2. GenerationConfig.__init__ source - also validates some fields at construction time
3. A description of what invariant types to look for

OUTPUT: Return a YAML document with this exact structure:

```yaml
schema_version: 1.0.0
engine: transformers
engine_version: 4.57.3
mined_at: '2026-05-24T00:00:00+00:00'
invariants:
- id: <snake_case_unique_id>
  engine: transformers
  library: transformers
  invariant_under_test: <what GenerationConfig method flags and why>
  severity: <error|dormant|warning>
  native_type: transformers.GenerationConfig
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: <validate|__init__>
    line_at_scan: <line_number>
  match:
    engine: transformers
    fields:
      transformers.sampling.<field_name>: <value_or_predicate>
  kwargs_positive:
    <field>: <value_that_triggers_invariant>
  kwargs_negative:
    <field>: <value_that_does_NOT_trigger>
  expected_outcome:
    outcome: <error|dormant_announced|warning>
    emission_channel: <none|logger_warning_once|logger_warning>
    normalised_fields: []
  message_template: '<exact error/warning message text from source>'
  references:
  - <citation>
  added_by: llm_miner
  added_at: '2026-05-24'
```

INVARIANT TYPES TO EXTRACT:

1. ERROR invariants (raise ValueError at construction/__init__ time):
   - Field value not in allowed set (enum violation)
   - Field type mismatch (wrong Python type)
   - Field value out of range (e.g. max_new_tokens <= 0)

2. DORMANT invariants (logger.warning_once at validate() time - parameter will be silently ignored):
   - Sampling-only params (temperature, top_p, top_k, min_p, epsilon_cutoff, eta_cutoff, typical_p) set when do_sample=False
   - Beam-only params (early_stopping, length_penalty) set when num_beams=1
   - Output params (output_attentions, output_hidden_states, output_scores) set when return_dict_in_generate=False
   - num_beams=1 AND early_stopping=True
   - use_cache=False when cache_implementation is set
   - pad_token_id < 0

3. CROSS-FIELD invariants (combinations that error):
   - num_beams not divisible by num_return_sequences
   - num_return_sequences > num_beams when do_sample=False
   - num_beams=1 when diversity_penalty > 0 or num_beam_groups > 1

SEVERITY:
- "error" = raises ValueError (hard failure)
- "dormant" = logs warning, parameter silently normalised/ignored
- "warning" = logs warning, execution continues with user-set value

For each `match.fields` predicate, use:
- Exact value: `transformers.sampling.field: value`
- Presence: `transformers.sampling.field: {present: true}`
- Not in list: `transformers.sampling.field: {present: true, not_in: [val1, val2]}`
- Not equal: `transformers.sampling.field: {present: true, not_equal: value}`
- Greater than: `transformers.samplin
...
```

## Analysis of Misses and Spurious

### Schema analysis

- Missed engine_params: 29 fields
- Missed sampling_params: 22 fields
- Spurious fields: 5

The schema call tests whether the model can read Python function signatures and class definitions and emit structured JSON Schema output. Key failure modes: (a) missing optional/rare params, (b) type misattribution for weakly-annotated fields, (c) missing BitsAndBytesConfig fields that surface as engine kwargs.

### Invariants analysis

Invariants call failed to parse - see error above.

## Cost Estimate

Wall-clock: 2085s (637s schema + 1448s invariants)

Energy: A100 TDP ~250W. At full load: 144.8 Wh (~0.145 kWh). Actual draw likely 150-200W during inference.

Per-run cost vs handwritten machinery: The handwritten miners are one-time authored; this LLM approach runs per version bump. If the model needs to be re-queried on each library version upgrade (~6/year), this is still trivially cheap vs engineering time.

## Verdict and Next Steps

**LOW: NOT WORTH PURSUING - low overlap; handwritten machinery has more value than feared**

Recommended next steps:
- Keep the handwritten machinery as the primary approach
- Consider LLM only for documentation/description enrichment, not structural extraction
- Revisit with a more capable model (70B Q4_K_M may be too constrained for this task)