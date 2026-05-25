# Locked invariants-extraction prompt (Phase 2)

**Locked at:** Phase 2 calibration round 1 (initial baseline lock).
**Source:** `research/mining-substrate-trial/scripts/strategies/prompts.py` constant
`INVARIANTS_PROMPT_TEMPLATE`.
**Used by:** strategies (b), (c), and the extension portion of
(d-ab) / (d-ac).

## Chunking instructions

Chunks are produced by `_spike.scripts.strategies.transformers_chunker.invariants_chunks()`:

1. **generation_config_init_invariants**: GenerationConfig.__init__
   source (type checks at construction) + CompileConfig source
   (referenced from init checks). Companion-inline pattern.
2. **validate_section_NN_<label>**: GenerationConfig.validate() body
   split by the `# 1.x` / `# 2.x` comment markers into ~12 logical
   sections. Each section is sent independently to AVOID the long-
   signature truncation bake-off B suffered.

Sections include: decoding attributes, cache attributes, performance
attributes, watermarking, sampling-only-when-greedy detection,
beam-only-when-greedy detection, num_return_sequences cross-field,
cache cross-field, etc.

Each chunk targets <2k tokens of source (~10k chars max). Validate()
sections are typically 500-2000 chars each.

## Output format

YAML envelope. No JSON Schema validation (YAML structure is harder
to validate; we rely on post-parse shape checks in the merger).

The merger deduplicates by `(namespace, native_field,
predicate_kind)` tuple AND by `id`. Last-write-wins; ambiguous
entries are logged as observations.

## Full prompt template

```
You are a code analyser extracting validation invariants from
{engine} library v{engine_version}. An "invariant" is a rule the
library checks at runtime - typically `if <predicate>: raise ValueError(...)`
or `if <predicate>: minor_issues[...] = ...` (which surfaces as a
warning).

You will be shown ONE CHUNK of validation source. Extract every
invariant visible in this chunk.

OUTPUT FORMAT: YAML document matching EXACTLY this shape:

invariants:
- id: <snake_case_unique_id>
  engine: {engine}
  library: {engine}
  invariant_under_test: <one-line: what the library checks here>
  severity: <error|dormant|warning>
  native_type: {engine}.GenerationConfig
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: <validate|__init__>
    line_at_scan: <approximate line number if visible>
  match:
    engine: {engine}
    fields:
      transformers.sampling.<field>: <value or predicate>
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
- Exact value:         `transformers.sampling.field: value`
- Not in list:         `transformers.sampling.field: {present: true, not_in: [a, b]}`
- Not equal:           `transformers.sampling.field: {present: true, not_equal: value}`
- Greater than:        `transformers.sampling.field: {'>': value}`
- Less than:           `transformers.sampling.field: {'<': value}`
- Greater or equal:    `transformers.sampling.field: {'>=': value}`
- Less or equal:       `transformers.sampling.field: {'<=': value}`
- Type not in:         `transformers.sampling.field: {present: true, type_is_not: [TypeName]}`
- Presence (any value):`transformers.sampling.field: {present: true}`

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
  ... etc.

Example 2 (DORMANT, sampling-only when greedy):
Source: ``if self.temperature is not None and self.temperature != 1.0: minor_issues["temperature"] = greedy_wrong_parameter_msg.format(...)``
Emit:
- id: transformers_temperature_set_when_do_sample_false
  ... etc.

Example 3 (CROSS-FIELD ERROR):
Source: ``if self.num_return_sequences > self.num_beams: raise ValueError(...)``
Emit:
- id: transformers_num_return_sequences_gt_num_beams
  ... etc.

{source}

Emit the YAML now:
```

(Full prompt with all three example bodies is in
`research/mining-substrate-trial/scripts/strategies/prompts.py` constant
`INVARIANTS_PROMPT_TEMPLATE`.)

## Behavioural notes

- Source-section ordering is preserved per chunk; the merger
  concatenates outputs across the 13 chunks then deduplicates.
- `kwargs_positive` / `kwargs_negative` provide a runtime-validation
  hook: a downstream verifier can call `GenerationConfig(**positive).validate()`
  to confirm the invariant ACTUALLY fires (Phase 3 capability).
- `severity` taxonomy is binary in the source (raise vs minor_issues)
  but split three ways for downstream consumers - the prompt's
  classification rule maps each source-form to a severity bucket.
- Code-fence stripping is identical to the schema prompt - the parser
  handles ```yaml``` / ```yml``` /  bare ``` fences transparently.
- No JSON Schema validation: YAML's looser shape doesn't lend
  itself to a Pydantic-style validator. The merger does shape checks
  inline (missing `match.fields` -> entry dropped with observation).
