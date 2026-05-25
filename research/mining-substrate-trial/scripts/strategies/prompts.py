"""Prompt templates for strategy (b)/(c)/(d) LLM extraction.

Locked at end of Phase 2 per the empirical-trial plan; saved in
``research/mining-substrate-trial/findings/phase2_locked_prompts/`` for archival.

Two task prompts:
- **Schema prompt**: extracts fields + types + defaults. JSON output.
- **Invariants prompt**: extracts predicates + severity + message
  templates. YAML output.

Plus hybrid prompts (used by ``hybrid_extractor.py``):
- **Hybrid extension prompt**: extend (a)'s output with what (a) missed.
- **Hybrid diagnose prompt**: explain WHY (a) missed what it missed.

Key design decisions, encoded into prompt rules:
- "Return ONLY raw JSON/YAML; no markdown fences" - Bake-off B's #1 failure.
- "Skip fields starting with `_` or in [blocklist]" - internal plumbing.
- "Only emit fields visible in the source provided" - don't hallucinate.
- "If unsure of type, omit the type key (don't guess)" - precision over recall.
- Few-shot examples drawn from the v4_57_3 reference.

ALL EXAMPLES BELOW are from transformers v4_57_3 - the calibration cell.
For other engines, the few-shot examples should be regenerated from
that engine's reference (or kept generic).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Schema prompt
# ---------------------------------------------------------------------------

SCHEMA_JSON_SCHEMA = {
    "type": "object",
    "required": ["engine", "engine_version", "chunk_fields"],
    "properties": {
        "engine": {"type": "string"},
        "engine_version": {"type": "string"},
        "chunk_name": {"type": "string"},
        "chunk_fields": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "namespace": {
                        "type": "string",
                        "enum": [
                            # Cross-engine top-level namespaces.
                            "engine_params",
                            "sampling_params",
                            # transformers $defs (Phase 2 calibration).
                            "$defs.CompileConfig",
                            "$defs.WatermarkingConfig",
                            "$defs.SynthIDTextWatermarkingConfig",
                            "$defs.BitsAndBytesConfig",
                            # vllm $defs (Phase 3 Prep-1 - companion + nested
                            # config classes referenced from EngineArgs /
                            # ModelConfig / SamplingParams sources).
                            "$defs.GuidedDecodingParams",
                            "$defs.ModelConfig",
                            "$defs.CacheConfig",
                            "$defs.SchedulerConfig",
                            "$defs.ParallelConfig",
                            "$defs.LoRAConfig",
                            "$defs.PromptAdapterConfig",
                            "$defs.TokenizerPoolConfig",
                            "$defs.LoadConfig",
                            "$defs.DecodingConfig",
                            "$defs.ObservabilityConfig",
                            # tensorrt $defs (Phase 3 Prep-1 - nested Pydantic
                            # configs referenced from TrtLlmArgs / BaseLlmArgs
                            # sources).
                            "$defs.CalibConfig",
                            "$defs.KvCacheConfig",
                            "$defs.PeftCacheConfig",
                            "$defs.DynamicBatchConfig",
                            "$defs.BuildCacheConfig",
                            "$defs.BatchingType",
                            "$defs.CapacitySchedulerPolicy",
                            "$defs.ContextChunkingPolicy",
                            "$defs.LookaheadDecodingConfig",
                            "$defs.MedusaDecodingConfig",
                            "$defs.EagleDecodingConfig",
                            "$defs.NGramDecodingConfig",
                            "$defs.MTPDecodingConfig",
                        ],
                    },
                    "type": {
                        "type": ["string", "array", "null"],
                    },
                    "default": {},  # any
                    "description": {"type": "string"},
                    "enum": {"type": "array"},
                    "anyOf": {"type": "array"},
                },
                "required": ["namespace"],
            },
        },
    },
}
"""JSON Schema for one chunk's schema output. Per-chunk extraction;
the strategy merges across chunks."""


SCHEMA_PROMPT_TEMPLATE = """You are a code analyser extracting the parameter schema for the
{engine} library, version {engine_version}.

You will be shown ONE CHUNK of source code. Extract the parameter
schema for the fields visible in this chunk. Other chunks are mined
separately and merged later - do not include fields from outside
this chunk.

OUTPUT FORMAT: a single JSON object matching EXACTLY this shape:

{{
  "engine": "{engine}",
  "engine_version": "{engine_version}",
  "chunk_name": "{chunk_name}",
  "chunk_fields": {{
    "<field_name>": {{
      "namespace": "<one of: engine_params, sampling_params, $defs.CompileConfig, ...>",
      "type": "<one of: string, integer, number, boolean, array, object, null>",
      "default": <value_or_null>,
      "description": "<brief one-liner>",
      "enum": [<values>],
      "anyOf": [{{"type": "..."}}, ...]
    }}
  }}
}}

EXPECTED NAMESPACES FOR THIS CHUNK: {namespaces}
(Other namespaces are extracted from other chunks. If you see fields
that belong to other namespaces, ignore them in this chunk.)

CRITICAL RULES:
1. Return ONLY the JSON document. NO markdown code fences (no ```).
   NO commentary, no preamble, no postamble. The first character of
   your response must be `{{`.
2. Extract ONLY fields VISIBLE in the source below. Do not invent or
   hallucinate fields. If a field is referenced but its source is not
   shown, omit it.
3. Skip internal-plumbing fields: any name starting with `_`
   (e.g. `_commit_hash`, `_from_auto`) and these explicit names:
   `adapter_kwargs`, `model_kwargs`, `torch_dtype`.
4. For fields with NO clear type annotation, OMIT the "type" key. Do
   NOT guess. Only set "type" when the source explicitly annotates.
5. For Optional[X] / Union[X, None] / X | None: set "type" to X and
   "default" to null if applicable. If multiple non-null types use
   "anyOf": [{{"type": "X"}}, {{"type": "Y"}}, {{"type": "null"}}].
6. For typing.Union[A, B] without None: use "anyOf" not "type".
7. For defaults that are None, use null. For defaults that are
   complex objects, use null (the schema only carries simple defaults).
8. For Sphinx-documented kwargs (pulled by name from a `kwargs.pop(...)`
   call) the docstring usually documents the type and default - read
   it and emit accordingly. If the docstring is shown, USE IT.

FEW-SHOT EXAMPLES (from transformers v4.57.3 reference catalogue):

Example 1 (engine_params, simple bool):
  Source: `force_download: bool = False,`
  Emit: `"force_download": {{"namespace": "engine_params", "type": "boolean", "default": false}}`

Example 2 (engine_params, Optional Union with PathLike):
  Source: `cache_dir: Optional[Union[str, os.PathLike]] = None,`
  Emit: `"cache_dir": {{"namespace": "engine_params", "anyOf": [{{"type": "string"}}, {{"type": "object", "description": "PathLike"}}, {{"type": "null"}}], "default": null}}`

Example 3 (engine_params, BitsAndBytesConfig field with default):
  Source: `bnb_4bit_compute_dtype=None,` (in BitsAndBytesConfig.__init__)
  Emit: `"bnb_4bit_compute_dtype": {{"namespace": "engine_params", "default": null, "description": "BitsAndBytesConfig quantisation field"}}`

Example 4 (sampling_params with enum from validate()):
  Source: GenerationConfig docstring mentions `cache_implementation (str, *optional*)` and validate() checks `not in ALL_CACHE_IMPLEMENTATIONS`.
  Emit: `"cache_implementation": {{"namespace": "sampling_params", "type": "string"}}`

Example 5 ($defs entry for CompileConfig):
  Source: CompileConfig dataclass with `fullgraph: bool = True`
  Emit: `"fullgraph": {{"namespace": "$defs.CompileConfig", "type": "boolean", "default": true}}`

Example 6 (sampling_params, unannotated default-None - common in GenerationConfig.__init__):
  Source: `temperature = kwargs.pop("temperature", None)` and docstring `temperature (float, *optional*, defaults to 1.0)`
  Emit: `"temperature": {{"namespace": "sampling_params", "type": "number", "default": 1.0}}`

CRITICAL: every field name MUST appear in the chunk_fields object.
Do NOT nest by namespace at the top level - the "namespace" key
inside each field's object is the namespace marker.

{source}

Emit the JSON now:
"""


# ---------------------------------------------------------------------------
# Invariants prompt
# ---------------------------------------------------------------------------


INVARIANTS_PROMPT_TEMPLATE = """You are a code analyser extracting validation invariants from
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
      {field_namespace}.<field>: <value or predicate>
  kwargs_positive:
    <field>: <value that TRIGGERS the invariant>
  kwargs_negative:
    <field>: <value that does NOT trigger>
  expected_outcome:
    outcome: <error|dormant_announced|warning>
    emission_channel: <none|logger_warning_once|logger_warning>
    normalised_fields: []
  message_template: '<the exact error/warning string from source, with {{}} placeholders preserved>'
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
- Exact value:         `{field_namespace}.field: value`
- Not in list:         `{field_namespace}.field: {{present: true, not_in: [a, b]}}`
- Not equal:           `{field_namespace}.field: {{present: true, not_equal: value}}`
- Greater than:        `{field_namespace}.field: {{'>': value}}`
- Less than:           `{field_namespace}.field: {{'<': value}}`
- Greater or equal:    `{field_namespace}.field: {{'>=': value}}`
- Less or equal:       `{field_namespace}.field: {{'<=': value}}`
- Type not in:         `{field_namespace}.field: {{present: true, type_is_not: [TypeName]}}`
- Presence (any value):`{field_namespace}.field: {{present: true}}`

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
   `minor_issues[...]`. Preserve `{{}}` placeholders. Do NOT
   substitute concrete values.

FEW-SHOT EXAMPLES (from transformers v4.57.3 reference):

Example 1 (ERROR, enum violation):
Source: ``if self.early_stopping not in {{True, False, "never"}}: raise ValueError(f"`early_stopping` must be a boolean or 'never', but is {{self.early_stopping}}.")``
Emit:
- id: transformers_early_stopping_not_in_allowlist
  engine: transformers
  invariant_under_test: GenerationConfig.validate flags `early_stopping` not in allowlist
  severity: error
  match:
    engine: transformers
    fields:
      transformers.sampling.early_stopping: {{present: true, not_in: [true, false, "never"]}}
  kwargs_positive:
    early_stopping: "invalid"
  kwargs_negative:
    early_stopping: true
  expected_outcome:
    outcome: error
    emission_channel: none
    normalised_fields: []
  message_template: "`early_stopping` must be a boolean or 'never', but is {{self.early_stopping}}."

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
  message_template: "`do_sample` is set to `False`. However, `{{flag_name}}` is set to `{{flag_value}}` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `{{flag_name}}`."

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
      transformers.sampling.num_return_sequences: {{'>': 1}}
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
  message_template: "`num_return_sequences` ({{self.num_return_sequences}}) has to be smaller or equal to `num_beams` ({{self.num_beams}})."

{source}

Emit the YAML now:
"""


# ---------------------------------------------------------------------------
# Hybrid (d-ab / d-ac) prompts
# ---------------------------------------------------------------------------


HYBRID_EXTENSION_PROMPT_TEMPLATE = """You are a code analyser working as a SECOND PASS on top of a
deterministic miner's output. The deterministic miner has already
extracted invariants from {engine} v{engine_version}; your job is to
find what it MISSED, find what looks SPURIOUS, and diagnose WHY it
missed what it missed.

INPUT 1 - DETERMINISTIC MINER'S OUTPUT (this is what (a) found):

{deterministic_output}

INPUT 2 - ENGINE SOURCE (the same source the deterministic miner read):

{source}

OUTPUT FORMAT: YAML document with THREE sections:

added_by_llm_verifier:
- # invariants the deterministic miner missed. Same shape as the
  # per-invariant entries it emits, plus `added_by: llm_verifier`
  # and `flagged_for_review: true`. Use the same predicate-form
  # vocabulary (`not_in`, `present`, `type_is_not`, etc.).

flagged_spurious_in_deterministic:
- id: <existing-invariant-id-from-deterministic-output>
  reason: <one-line: why this looks spurious or wrong>
  # NOTE: do not auto-resolve; just flag for human review.

missed_diagnosis:
- id: <a-missed-invariant-name>
  why_missed: <one-line: AST limitation, kwargs.pop pattern, etc.>

RULES:
1. ONLY emit invariants the deterministic output does NOT already
   contain. Check by `id` and by the (field, predicate_kind) tuple.
2. For each emitted invariant, set `added_by: llm_verifier` and
   `flagged_for_review: true` in the YAML body.
3. Be conservative: it's better to under-emit than to add spurious.
   The human reviewer will see your flagged_for_review entries.
4. NO markdown code fences. NO commentary outside the YAML.

Emit the YAML now:
"""


# ---------------------------------------------------------------------------
# Multi-pass refinement prompts (Phase 2.5)
# ---------------------------------------------------------------------------
#
# Phase 2 shipped extract-only (the single-shot SCHEMA + INVARIANTS prompts
# above). The plan ("Multi-pass refinement: extract -> verify -> extend")
# specified a 3-pass pipeline. Phase 2.5 adds VERIFY + EXTEND prompts; the
# initial extract pass continues to use INVARIANTS_PROMPT_TEMPLATE locked
# at round 3 of Phase 2 calibration.
#
# Why two prompts (not one): empirically, asking an LLM to do BOTH
# verification and extension in a single response splits its attention
# across both tasks. Two focused passes per chunk give the LLM clearer
# objectives.
#
# These prompts are PER-CHUNK (run against the same chunked source as the
# extract pass), not per-output (hybrid_extension is per-output). Multi-
# pass is INTERNAL to strategy (b); hybrid_extension is the (d) cell's
# bridge between (a) and (b)/(c).


INVARIANTS_VERIFY_PROMPT_TEMPLATE = """You are a code analyser doing PASS 2 (verification) of multi-pass
invariant extraction. PASS 1 already extracted invariants from
{engine} v{engine_version} for ONE chunk of source. Your job is to
REVIEW each emitted invariant against the source and flag any that
look WRONG.

INPUT 1 - PASS 1 EXTRACTED INVARIANTS (the candidate list):

{pass1_invariants}

INPUT 2 - THE SOURCE PASS 1 READ:

{source}

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
- Predicate wrong: source has `if X.field not in {{a, b}}: raise`
  but invariant emits `predicate_kind: exact`.
- Kwargs_positive wrong: source's predicate is "value < 0 raises" but
  invariant's kwargs_positive shows `field: 1` (which would NOT trigger).
- Hallucinated: invariant references a field name that does NOT appear
  in the source.

Emit the YAML now:
"""


INVARIANTS_EXTEND_PROMPT_TEMPLATE = """You are a code analyser doing PASS 3 (extension) of multi-pass
invariant extraction. PASS 1 already extracted N invariants from
{engine} v{engine_version} for ONE chunk of source. Your job is to
identify ADDITIONAL invariants PASS 1 missed.

INPUT 1 - PASS 1 EXTRACTED INVARIANTS (already known; do NOT re-emit):

{pass1_invariants}

INPUT 2 - PASS 2 FLAGGED ISSUES (for context; consider whether they hint
at related misses):

{pass2_flags}

INPUT 3 - THE SOURCE PASS 1 READ:

{source}

OUTPUT FORMAT: YAML document with the SAME shape as the original
invariants prompt's output:

invariants:
- id: <snake_case_unique_id>
  engine: {engine}
  library: {engine}
  invariant_under_test: <one-line: what the library checks here>
  severity: <error|dormant|warning>
  native_type: <e.g. {engine}.GenerationConfig or {engine}.EngineArgs>
  miner_source:
    path: <file path>
    method: <validate|__init__|_verify_args|...>
    line_at_scan: <approximate line number if visible>
  match:
    engine: {engine}
    fields:
      {field_namespace}.<field>: <value or predicate>
  kwargs_positive:
    <field>: <value that TRIGGERS the invariant>
  kwargs_negative:
    <field>: <value that does NOT trigger>
  expected_outcome:
    outcome: <error|dormant_announced|warning>
    emission_channel: <none|logger_warning_once|logger_warning>
    normalised_fields: []
  message_template: '<the exact error/warning string from source>'
  added_by: llm_miner_pass3
  added_at: '2026-05-25'

PREDICATE FORMS (use the EXACT keys shown):
- Exact value:         `{field_namespace}.field: value`
- Not in list:         `{field_namespace}.field: {{present: true, not_in: [a, b]}}`
- Greater than:        `{field_namespace}.field: {{'>': value}}`
- Less than:           `{field_namespace}.field: {{'<': value}}`
- Type not in:         `{field_namespace}.field: {{present: true, type_is_not: [TypeName]}}`
- Presence:            `{field_namespace}.field: {{present: true}}`

CRITICAL RULES FOR PASS 3:
1. Return ONLY the YAML document. NO markdown code fences. NO commentary.
   First character must be `i` (from `invariants:`).
2. ONLY emit invariants that are NOT already in PASS 1's list (check by
   ID and by (field, predicate_kind) tuple).
3. Look ESPECIALLY for these PASS 1 blind spots:
   a. PER-FIELD invariants where PASS 1 emitted ONE entry covering many
      fields. Example: PASS 1 emitted `transformers_temperature_set_when_do_sample_false`
      but the source has SIMILAR if-blocks for top_p, top_k, min_p,
      typical_p, epsilon_cutoff, eta_cutoff. Emit ONE per field PASS 1
      missed.
   b. Multi-clause `if A and B and C:` invariants where PASS 1 keyed on
      only A. Emit one invariant per (A, B, C) tuple if each is independent.
   c. Type-check invariants `if not isinstance(x, T): raise` that PASS 1
      may have skipped (these often follow a similar pattern repeated
      per field).
   d. Less-common predicates: `<=`, `>=`, `not_equal`, `present`. PASS 1
      may have collapsed these to `exact`.
4. If PASS 1 covered every visible invariant, emit `invariants: []` (empty
   list) - do NOT fabricate.
5. Use snake_case_with_engine_prefix for `id`.
6. Set `added_by: llm_miner_pass3` (NOT `llm_miner`) so the multi-pass
   bookkeeping can track pass-3 contributions.

Emit the YAML now:
"""


HYBRID_DIAGNOSE_PROMPT_TEMPLATE = """You are a code analyser diagnosing GAPS in a deterministic miner.

The deterministic miner produced output X for {engine} v{engine_version}.
A reference catalogue has output Y (the ground truth). The reference
contains items the deterministic miner did NOT surface.

INPUT 1 - DETERMINISTIC OUTPUT:
{deterministic_output}

INPUT 2 - REFERENCE OUTPUT (ground truth):
{reference_output}

INPUT 3 - SOURCE THE DETERMINISTIC MINER READ:
{source}

Produce a YAML document listing each reference item the deterministic
miner missed, with a ONE-LINE diagnosis of WHY:

diagnoses:
- id: <reference-invariant-id>
  why_missed: <root-cause: AST walker limitation? kwargs.pop pattern?
    cross-method invariant? docstring-only specification?>
  fix_suggestion: <terse: "extend walker to recognise X" / "walker
    correctly skipped; this is documentation-only">

RULES:
- Limit each diagnosis to ONE LINE for `why_missed` and ONE LINE for
  `fix_suggestion`.
- NO markdown code fences. NO commentary outside the YAML.

Emit the YAML now:
"""
