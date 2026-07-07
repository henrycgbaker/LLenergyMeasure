# Schema discovered format

Format specification for the JSON Schema artefact emitted by the
schema-discovery pipeline: top-level envelope, parameter-section shape,
limitations records, schema-version history, and example.

The artefact is produced by `scripts/engine_producers/` running
inside an engine's Docker image. For how to run and troubleshoot schema
discovery, see [schema refresh](/contributing/schema-refresh).
For the runtime-loader side that reads these files at import time, see
[parameter discovery](/explanation/architecture/parameter-discovery) and
[parameter curation](/explanation/architecture/parameter-curation).

---

## File locations

```
  src/llenergymeasure/engines/{engine}/
  └── schema.discovered.json    Discovered schema artefact (committed)

  src/llenergymeasure/config/
  └── schema_loader.py          Runtime consumer (SchemaLoader)
```

One file per engine. The runtime `SchemaLoader` reads them via
`importlib.resources`, which works in both editable installs and
installed wheels. Repeated loads are cached per engine.

---

## Top-level envelope

```json
{
  "schema_version": "1.0.0",
  "engine": "transformers",
  "engine_version": "4.57.3",
  "engine_commit_sha": null,
  "image_ref": "llenergymeasure:transformers-4.57.3",
  "base_image_ref": "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
  "discovered_at": "2026-05-08T02:36:55Z",
  "discovery_method": "inspect.signature(from_pretrained) + GenerationConfig().to_dict()",
  "discovery_limitations": [ /* see below */ ],
  "engine_params":   { /* {field_name: {type, default, ...}} */ },
  "sampling_params": { /* {field_name: {type, default, ...}} */ }
}
```

Field order in the file is stable: introspectors emit keys in the order
above so JSON diffs stay readable.

---

## Field reference

### `schema_version`

Envelope schema version. Major version must equal
`SUPPORTED_MAJOR_VERSION` in
`src/llenergymeasure/config/schema_loader.py`. Major bumps are breaking
and `SchemaLoader` raises `UnsupportedSchemaVersionError`. Minor bumps
add envelope keys; downstream loaders are forward-compatible.

### `engine`

Canonical engine identifier. Must match an entry in the `Engine` enum
(`src/llenergymeasure/config/ssot.py`). Used as the dispatch key by the
runtime loader and the dictionary key under which the file is keyed
when multiple schemas are loaded.

### `engine_version`

Library version the schema was discovered against. Sourced from the
SSOT (`engine_versions/{engine}/current.yaml`) at discovery time. Renovate-driven
bumps re-fire discovery so this field tracks the pinned upstream version
on `main`.

### `engine_commit_sha`

Engine commit SHA, when discovery records it (currently `null` for all
three engines; reserved for future use when a discovery target ships
from a git ref rather than a versioned release).

### `image_ref`

Container image reference recorded in the envelope. Defaults to the
Dockerfile FROM tag for the transformers introspector (which has a
first-party `docker/Dockerfile.transformers`); set explicitly via
`--image-ref` for vLLM and TensorRT-LLM (which run inside upstream
images supplied by the workflow).

### `base_image_ref`

Base image the engine container is built on. Distinct from `image_ref`
when discovery is run inside a multi-stage Dockerfile (the transformers
case, where `image_ref = llenergymeasure:transformers-<VER>` and
`base_image_ref = pytorch/pytorch:...`). `null` when the upstream image
is the runtime image directly (the vLLM and TensorRT-LLM cases).

### `discovered_at`

ISO 8601 timestamp of the discovery run. Honours the
`LLENERGY_DISCOVERY_FROZEN_AT` environment variable: when set, the
introspector pins the timestamp to that value (typically the author
date of the most recent commit touching any input path) so CI re-runs
do not produce a fresh wallclock timestamp on every invocation. Without
the freeze, every CI run would emit a 2-line diff and re-fire the path
filter.

### `discovery_method`

Short human-readable description of how the schema was extracted.
Examples seen in the current corpus:

| Engine | `discovery_method` |
|---|---|
| transformers | `inspect.signature(from_pretrained) + GenerationConfig().to_dict()` |
| vllm | `dataclasses.fields(EngineArgs) + msgspec.json.schema(SamplingParams)` |
| tensorrt | `TrtLlmArgs.model_json_schema() + dataclasses.fields(SamplingParams)` |

The string is informational; consumers do not parse it.

### `discovery_limitations`

A list of records documenting fields that discovery could not recover.
Surfaced here rather than silently dropped so reviewers see exactly
where the introspector's reach ends.

```json
[
  {
    "section": "engine_params",
    "fields": [
      "AutoModelForCausalLM.from_pretrained.**model_args",
      "AutoModelForCausalLM.from_pretrained.**kwargs"
    ],
    "reason": "from_pretrained accepts **kwargs; kwargs are not in the signature (documented kwargs live in the class docstring only)"
  },
  {
    "section": "sampling_params",
    "fields": ["max_new_tokens", "min_new_tokens"],
    "reason": "GenerationConfig has no type annotations; None defaults yield type='unknown'"
  }
]
```

| Sub-field | Type | Meaning |
|---|---|---|
| `section` | string | One of `engine_params`, `sampling_params`. The parameter section the limitation applies to. |
| `fields` | list[string] | Field paths or qualified names that discovery did not recover. May be empty when the limitation is stated in general terms. |
| `reason` | string | Human-readable explanation. Used by reviewers, not by machines. |

### `engine_params`

Object keyed by engine-parameter name. Values are descriptor objects
with at minimum `type` and `default`; some engines additionally surface
`description` and `deprecated` (notably TRT-LLM, which exposes a
Pydantic schema with these fields).

```json
"engine_params": {
  "model": {
    "type": "string",
    "default": null,
    "description": "The path to the model checkpoint or the model name from the Hugging Face Hub.",
    "deprecated": false
  },
  "tokenizer_mode": {
    "type": "Literal['auto', 'slow']",
    "default": "auto",
    "description": "The mode to initialize the tokenizer.",
    "deprecated": false
  }
}
```

| Sub-field | Type | Notes |
|---|---|---|
| `type` | string | Compact string rendering of the type annotation. Handles `None`, `Optional[X]`, `X \| None`, `Union`, `Literal[...]`, generics, and forward refs. Falls back to `str(annotation)` when the annotation is unrecognised; falls back to `"unknown"` when the annotation is missing entirely (the transformers `GenerationConfig` case for fields whose default is `None`). |
| `default` | JSON value | Default value, JSON-coerced. Enums render as their `.name`; types render as `__name__`; sets render as sorted lists. Anything else falls back to `str(value)` so the output stays deterministic. |
| `description` | string (optional) | Per-field description, when discovery can recover one (TRT-LLM Pydantic schema). Absent for engines whose introspection target carries only a class docstring (vLLM, transformers). |
| `deprecated` | boolean (optional) | Deprecation flag, when discovery can recover one. |

### `sampling_params`

Same shape as `engine_params`. Holds the sampling-side parameters
extracted from the engine's sampling-config class (transformers
`GenerationConfig`, vLLM `SamplingParams`, TRT-LLM `SamplingParams`).

The split between `engine_params` and `sampling_params` reflects the
two distinct config classes each engine exposes: one for engine
construction (model loading, parallelism, scheduler), one for per-call
sampling (temperature, top-k, beams). The runtime config models in
`src/llenergymeasure/config/engine_configs.py` mirror this split.

---

## Per-engine variations

The three engines surface different richness because their native
introspection targets differ. The envelope is the same; the per-field
descriptor is richer for some.

| Engine | `engine_params` source | `sampling_params` source | Per-field `description` available |
|---|---|---|---|
| transformers | `inspect.signature(AutoModelForCausalLM.from_pretrained)` + `inspect.signature(PreTrainedModel.from_pretrained)` | `GenerationConfig().to_dict()` | No (no class docstring per field) |
| vllm | `dataclasses.fields(EngineArgs)` | `msgspec.json.schema(SamplingParams)` | No (vLLM `EngineArgs` has only a class docstring) |
| tensorrt | `TrtLlmArgs.model_json_schema()` (Pydantic) | `dataclasses.fields(SamplingParams)` | Yes for `engine_params` (Pydantic schema carries descriptions); no for `sampling_params` |

`discovery_limitations` documents these variations explicitly per file.

---

## How the file is consumed

`src/llenergymeasure/config/schema_loader.py` parses each engine's
`schema.discovered.json` into a `DiscoveredSchema` dataclass:

```python
from llenergymeasure.config.schema_loader import SchemaLoader

loader = SchemaLoader()
schema = loader.load("vllm")
schema.engine_version       # "0.7.3"
schema.engine_params["dtype"]["default"]  # "auto"
```

The loader has three direct consumers:

- The drift checker (`scripts/check_pydantic_matches_discovered.py`)
  flags Pydantic fields in `engine_configs.py` that have no
  corresponding entry in the discovered schema.
- The doc generators (`scripts/generate_curation_doc.py`,
  `scripts/generate_schema_doc.py`) build the
  `docs/reference/engines/{schema,curation}-{engine}.md` digests from
  the loaded schemas.
- The runtime parameter-discovery layer reads the schema at
  config-validation time to align user fields with the engine's actual
  parameter surface.

For the full runtime data flow, see [parameter discovery](/explanation/architecture/parameter-discovery).

---

## Schema-version history

| Version | Changes |
|---|---|
| `1.0.0` | Initial release. Top-level envelope: `schema_version`, `engine`, `engine_version`, `engine_commit_sha`, `image_ref`, `base_image_ref`, `discovered_at`, `discovery_method`, `discovery_limitations`, `engine_params`, `sampling_params`. `SUPPORTED_MAJOR_VERSION = 1` in the loader. |

---

## See also

- [Invariants corpus format](/reference/invariants-corpus-format) - the parallel format spec for the validation-rules artefact
- [Architecture overview](/explanation/architecture/architecture-overview) - how the discovered schema fits the engine-knowledge products
- [Parameter curation](/explanation/architecture/parameter-curation) - how the discovered schema relates to the hand-authored Pydantic config models
- [Parameter discovery](/explanation/architecture/parameter-discovery) - runtime config validation
- [Schema refresh (operations guide)](/contributing/schema-refresh) - manual refresh procedure
- [Per-engine schema digests](/reference/engines/schema-transformers) - auto-generated reference rendered from these files
