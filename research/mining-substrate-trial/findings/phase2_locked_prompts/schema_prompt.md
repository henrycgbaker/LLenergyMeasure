# Locked schema-extraction prompt (Phase 2)

**Locked at:** Phase 2 calibration round 1 (initial baseline lock).
**Source:** `research/mining-substrate-trial/scripts/strategies/prompts.py` constant
`SCHEMA_PROMPT_TEMPLATE`.
**Used by:** strategies (b), (c), and the schema portion of (d-ab) /
(d-ac).

The prompt below is the FULL text the runner sends to the LLM, with
the `{engine}`, `{engine_version}`, `{chunk_name}`, `{namespaces}`,
and `{source}` placeholders filled in.

## Chunking instructions

Chunks are produced by `_spike.scripts.strategies.transformers_chunker.schema_chunks()`:

1. **from_pretrained_engine_params**: PreTrainedModel.from_pretrained
   signature + docstring (truncated to 8000 chars - skips bulk of
   body; the kwarg-pop bodies' first 3000 chars are included so the
   LLM sees the `_commit_hash`, `adapter_kwargs` plumbing it must
   filter out).
2. **bitsandbytes_compile_configs**: BitsAndBytesConfig source +
   CompileConfig + WatermarkingConfig - inlined COMPANION classes per
   Bake-off B lesson (LLM doesn't follow imports).
3. **generation_config_sampling_params**: GenerationConfig.__init__
   source + docstring (the docstring is what documents the
   None-default kwargs the static walker can't type).

Each chunk targets <15k chars (~3.5k tokens) - fits comfortably in
the 32k ctx window with output headroom.

## JSON Schema (used for retry-on-validation-failure)

```json
{
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
              "engine_params",
              "sampling_params",
              "$defs.CompileConfig",
              "$defs.WatermarkingConfig",
              "$defs.SynthIDTextWatermarkingConfig",
              "$defs.BitsAndBytesConfig"
            ]
          },
          "type": {"type": ["string", "array", "null"]},
          "default": {},
          "description": {"type": "string"},
          "enum": {"type": "array"},
          "anyOf": {"type": "array"}
        },
        "required": ["namespace"]
      }
    }
  }
}
```

## Full prompt template

```
You are a code analyser extracting the parameter schema for the
{engine} library, version {engine_version}.

You will be shown ONE CHUNK of source code. Extract the parameter
schema for the fields visible in this chunk. Other chunks are mined
separately and merged later - do not include fields from outside
this chunk.

OUTPUT FORMAT: a single JSON object matching EXACTLY this shape:

{
  "engine": "{engine}",
  "engine_version": "{engine_version}",
  "chunk_name": "{chunk_name}",
  "chunk_fields": {
    "<field_name>": {
      "namespace": "<one of: engine_params, sampling_params, $defs.CompileConfig, ...>",
      "type": "<one of: string, integer, number, boolean, array, object, null>",
      "default": <value_or_null>,
      "description": "<brief one-liner>",
      "enum": [<values>],
      "anyOf": [{"type": "..."}, ...]
    }
  }
}

EXPECTED NAMESPACES FOR THIS CHUNK: {namespaces}
(Other namespaces are extracted from other chunks. If you see fields
that belong to other namespaces, ignore them in this chunk.)

CRITICAL RULES:
1. Return ONLY the JSON document. NO markdown code fences (no ```).
   NO commentary, no preamble, no postamble. The first character of
   your response must be `{`.
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
   "anyOf": [{"type": "X"}, {"type": "Y"}, {"type": "null"}].
6. For typing.Union[A, B] without None: use "anyOf" not "type".
7. For defaults that are None, use null. For defaults that are
   complex objects, use null (the schema only carries simple defaults).
8. For Sphinx-documented kwargs (pulled by name from a `kwargs.pop(...)`
   call) the docstring usually documents the type and default - read
   it and emit accordingly. If the docstring is shown, USE IT.

FEW-SHOT EXAMPLES (from transformers v4.57.3 reference catalogue):

Example 1 (engine_params, simple bool):
  Source: `force_download: bool = False,`
  Emit: `"force_download": {"namespace": "engine_params", "type": "boolean", "default": false}`

Example 2 (engine_params, Optional Union with PathLike):
  Source: `cache_dir: Optional[Union[str, os.PathLike]] = None,`
  Emit: `"cache_dir": {"namespace": "engine_params", "anyOf": [{"type": "string"}, {"type": "object", "description": "PathLike"}, {"type": "null"}], "default": null}`

Example 3 (engine_params, BitsAndBytesConfig field with default):
  Source: `bnb_4bit_compute_dtype=None,` (in BitsAndBytesConfig.__init__)
  Emit: `"bnb_4bit_compute_dtype": {"namespace": "engine_params", "default": null, "description": "BitsAndBytesConfig quantisation field"}`

Example 4 (sampling_params with enum from validate()):
  Source: GenerationConfig docstring mentions `cache_implementation (str, *optional*)` and validate() checks `not in ALL_CACHE_IMPLEMENTATIONS`.
  Emit: `"cache_implementation": {"namespace": "sampling_params", "type": "string"}`

Example 5 ($defs entry for CompileConfig):
  Source: CompileConfig dataclass with `fullgraph: bool = True`
  Emit: `"fullgraph": {"namespace": "$defs.CompileConfig", "type": "boolean", "default": true}`

Example 6 (sampling_params, unannotated default-None - common in GenerationConfig.__init__):
  Source: `temperature = kwargs.pop("temperature", None)` and docstring `temperature (float, *optional*, defaults to 1.0)`
  Emit: `"temperature": {"namespace": "sampling_params", "type": "number", "default": 1.0}`

CRITICAL: every field name MUST appear in the chunk_fields object.
Do NOT nest by namespace at the top level - the "namespace" key
inside each field's object is the namespace marker.

{source}

Emit the JSON now:
```

## Behavioural notes

- The LLM frequently emits ```json``` code fences despite rule 1. The
  parser (`llm_extractor.parse_json_block`) strips them transparently.
  We do NOT retry on "fences present" - it's a recoverable parse.
- The few-shot examples include both PRIMITIVE-type and ANYOF cases
  to set the schema-shape expectation.
- "Skip _internal" is reinforced by the post-filter
  (`filter_internal_plumbing`) - the prompt rule is the FIRST line of
  defence; the filter is the SECOND.
- Per-chunk MERGE: outputs are merged across chunks by the executor;
  same-namespace duplicates last-write-win. Cross-chunk dupes are
  rare (each chunk has a clear scope).
