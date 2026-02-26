# Peer Research: Config Introspection & Auto-Discovery

> Generated 2026-02-26. Peer evidence for preservation audit item P-03.

---

## Evidence Per Tool

### 1. Pydantic v2 (Built-in Introspection)

**Does it support auto-discovery of all config fields?** Yes — comprehensively. Pydantic v2
provides two complementary introspection APIs:

1. **`model_fields`** — a `dict[str, FieldInfo]` mapping every field name to its `FieldInfo`
   object, available as a class attribute on every `BaseModel` subclass. No custom code needed.

2. **`model_json_schema()`** — generates a complete JSON Schema (Draft 2020-12) dictionary
   from the model, including all nested `$defs`, `$ref` cross-references, constraints, enums,
   defaults, and descriptions.

**What can it extract?**

| Attribute | Available via `FieldInfo` | Available via `model_json_schema()` |
|-----------|--------------------------|-------------------------------------|
| Field name | `model_fields.keys()` | `properties` keys |
| Type annotation | `field_info.annotation` | `type`, `anyOf` |
| Default value | `field_info.default` | `default` |
| Description | `field_info.description` | `description` |
| Constraints (ge/le/gt/lt) | `field_info.metadata` list (annotated-types objects) | `minimum`, `maximum`, `exclusiveMinimum`, etc. |
| Enum/Literal values | `get_args(annotation)` when `get_origin() is Literal` | `enum` array |
| Required vs optional | `field_info.is_required()` | `required` array |
| Title | `field_info.title` | `title` |
| Examples | `field_info.examples` | `examples` |
| Alias | `field_info.alias` | via `by_alias` mode |
| Deprecated | `field_info.deprecated` | N/A |
| Nested models | check `hasattr(annotation, 'model_fields')` | `$ref` / `$defs` |

**Constraints in `metadata`:** In Pydantic v2, numeric constraints (ge, le, gt, lt,
multiple_of) and string constraints (min_length, max_length, pattern) are stored as
`annotated_types` objects in `field_info.metadata`. Iterating this list to extract constraint
values requires ~5-10 lines of reflection code, as our v1.x `_extract_param_metadata()` does.
This is the only part of introspection that requires manual iteration — everything else is
direct attribute access.

**Test value generation:** Not built-in. Pydantic does not generate representative test
values from constraints. Our v1.x code that derives `[ge, midpoint, le]` for bounded ints
or `[False, True]` for bools is genuinely custom logic. However, the *inputs* to that logic
(constraints, type, default) are all available from Pydantic's built-in API without custom
reflection.

**Custom code needed:** For pure field discovery (names, types, defaults, descriptions,
constraints), approximately **0 custom LOC** — Pydantic v2's API provides everything directly.
For *interpreting* that metadata into test values, CLI rules, or doc tables, custom code is
needed (our ~300 LOC does this interpretation). The distinction matters: the *reflection* is
free; the *downstream consumption* requires domain logic.

**Key observation:** Pydantic v2's `model_json_schema()` already does 80% of what our
`get_params_from_model()` does. Our function adds: (a) test value generation, (b) flattened
dotted-path keys, (c) backend-aware prefix routing, (d) constraint metadata merged into a
single dict. Of these, only (a) is non-trivial custom logic.

**Sources:**
- [Pydantic v2 Fields documentation](https://docs.pydantic.dev/latest/concepts/fields/)
- [Pydantic v2 JSON Schema documentation](https://docs.pydantic.dev/latest/concepts/json_schema/)
- [Pydantic v2 Models documentation](https://docs.pydantic.dev/latest/concepts/models/)
- [Pydantic FieldInfo API (DeepWiki)](https://deepwiki.com/pydantic/pydantic/2.2-field-system)

---

### 2. Hydra / OmegaConf (Structured Configs)

**Does it support auto-discovery of all config fields?** Partially. Hydra uses OmegaConf's
`structured()` function to convert dataclass definitions into configuration objects with
runtime type validation. OmegaConf knows the schema — types, defaults, optionality — because
the dataclass definition *is* the schema.

**What can it extract?**

| Attribute | Supported | Mechanism |
|-----------|-----------|-----------|
| Field name | Yes | `dataclasses.fields()` |
| Type | Yes | `field.type` annotation |
| Default | Yes | `field.default` / `field.default_factory` |
| Description | **No** (open issue) | `field(metadata={"help": "..."})` — convention only |
| Constraints (min/max) | **No** | OmegaConf does range validation only via custom resolvers |
| Enum values | Partial | Python `Enum` types supported; no Literal introspection |
| Required vs optional | Yes | `MISSING` sentinel |

**CLI help generation:** This is a notable gap. Hydra issue
[#633](https://github.com/facebookresearch/hydra/issues/633) requests auto-generation of
`--help` text from dataclass field descriptions. It remains open with "wishlist" priority.
The workaround is manual `hydra.help.template` override — i.e., maintaining help text
separately from the config definition, exactly the anti-pattern SSOT introspection avoids.

**Is introspection used for test generation?** No. Hydra/OmegaConf does not generate test
values or feed config metadata into test infrastructure. Test suites for Hydra-based projects
(e.g., optimum-benchmark, NeMo) maintain separate test fixtures.

**Custom code needed:** OmegaConf provides runtime type checking automatically. For anything
beyond that (descriptions in help text, constraint extraction, test value generation), you
need custom code or additional libraries (hydra-zen provides some dynamic config generation
but not introspection for downstream consumers).

**Key finding:** Hydra validates configs at runtime (types match the dataclass) but does not
support the SSOT introspection pattern of deriving docs, tests, and CLI help from the same
typed config definition. This is a frequently-requested but unimplemented feature.

**Sources:**
- [Hydra Structured Config intro](https://hydra.cc/docs/tutorials/structured_config/intro/)
- [Hydra issue #633: help messages from field metadata](https://github.com/facebookresearch/hydra/issues/633)
- [OmegaConf Structured Configs](https://omegaconf.readthedocs.io/en/2.1_branch/structured_config.html)

---

### 3. Typer / pydantic-typer / pydantic-settings CLI

**Does Typer auto-generate CLI params from type annotations?** Yes — this is Typer's core
design. Function parameters with type annotations become CLI options/arguments. `Annotated`
metadata (`typer.Option(help="...")`) populates `--help` text. However, Typer operates on
*function signatures*, not on Pydantic model fields.

**Pydantic model support:** Not built into Typer. Third-party extensions bridge this gap:

- **pydantic-typer** (pypae): Introspects Pydantic model fields and generates CLI options
  with dot-notation for nested models (`--person.name`, `--person.pet.species`). Extracts
  types, defaults, and descriptions from `FieldInfo`. Supports `pydantic_typer.Typer` as a
  drop-in replacement. Small library (~500 LOC).

- **pydantic-cli** (mpkocher): Converts Pydantic models to full CLI tools. Supports loading
  from JSON files. Older project, less actively maintained.

- **pydantic-settings v2 CLI support:** As of pydantic-settings v2, there is first-party
  CLI support that auto-generates CLI arguments from `BaseSettings` model fields. This
  extracts field types, defaults, descriptions, and constraints directly. Provides
  `CliPositionalArg`, `CliImplicitFlag`, `CliSubCommand`, and `CliMutuallyExclusiveGroup`
  annotations. Minimal custom code — just `cli_parse_args=True` in `SettingsConfigDict`.

**What can these extract?**

| Attribute | pydantic-typer | pydantic-settings CLI |
|-----------|---------------|----------------------|
| Field name | Yes | Yes |
| Type | Yes | Yes |
| Default | Yes | Yes |
| Description | Yes (from Field) | Yes (from Field + docstring) |
| Constraints | Validation only | Validation only |
| Nested models | Yes (dot notation) | Yes |
| Enum/Literal | Yes | Yes |

**Is introspection used for test/doc generation?** No. These tools use model introspection
exclusively for CLI argument generation. They do not feed the same metadata into test
parametrisation or documentation pipelines.

**Key finding:** The pattern of auto-generating CLI parameters from typed model fields is
well-established and used by multiple libraries. However, none of them extend this to also
generate tests or docs from the same source — that multi-consumer SSOT pattern is our
specific contribution.

**Sources:**
- [pydantic-typer GitHub](https://github.com/pypae/pydantic-typer)
- [pydantic-settings CLI support (DeepWiki)](https://deepwiki.com/pydantic/pydantic-settings/4-cli-support)
- [pydantic-cli GitHub](https://github.com/mpkocher/pydantic-cli)
- [Typer feature request #111](https://github.com/fastapi/typer/issues/111)

---

### 4. FastAPI (OpenAPI Schema Auto-Generation)

**Does it support auto-discovery of all config fields?** Yes — this is the canonical example
of the pattern. FastAPI introspects Pydantic models used as request/response types and
auto-generates a complete OpenAPI 3.1 specification. This is the closest peer precedent for
our SSOT introspection approach.

**What does it extract?**

| Attribute | Extracted | Used for |
|-----------|-----------|----------|
| Field name | Yes | Path/query/body parameters |
| Type | Yes | OpenAPI `type` / `format` |
| Default | Yes | `default` in schema |
| Description | Yes | Parameter description in docs |
| Constraints (ge/le/etc.) | Yes | `minimum`, `maximum`, etc. in schema |
| Enum/Literal | Yes | `enum` array |
| Nested models | Yes | `$ref` / `components.schemas` |
| Required vs optional | Yes | `required` array |
| Examples | Yes | `examples` in schema |

**How much custom code is needed?** Zero for the introspection itself. FastAPI calls
`model_json_schema()` internally and merges results across all routes. The entire OpenAPI
spec generation is ~200 LOC in FastAPI's internals (`openapi/utils.py`), but this is
framework code — users write zero introspection code.

**Is introspection used for multiple consumers?** Yes — this is the key parallel:

1. **Interactive docs** (Swagger UI, ReDoc) — auto-generated from schema
2. **Request validation** — Pydantic validates inputs against the same models
3. **Client SDK generation** — third-party tools (e.g., Speakeasy, openapi-generator) consume
   the schema to generate typed clients
4. **Testing** — Schemathesis consumes the OpenAPI schema to auto-generate test cases

This is a four-consumer SSOT: one model definition produces docs + validation + SDKs + tests.
Our three-consumer SSOT (tests + docs + CLI validation) is a subset of this pattern.

**Key finding:** FastAPI validates the SSOT pattern at scale. Millions of production APIs use
"define the model once, derive everything else" as their core architecture. Our introspection
module implements the same pattern in a CLI/library context rather than a web API context.

**Sources:**
- [FastAPI OpenAPI schema generation (DeepWiki)](https://deepwiki.com/fastapi/fastapi/3.1-openapi-schema-generation)
- [FastAPI Features](https://fastapi.tiangolo.com/features/)
- [Pydantic JSON Schema for FastAPI (Orchestra)](https://www.getorchestra.io/guides/pydantic-json-schema-a-comprehensive-guide-for-fastapi-users)

---

### 5. attrs (`fields()` Introspection)

**Does it support auto-discovery of all config fields?** Yes. `attrs.fields(MyClass)` returns
a tuple of `Attribute` objects with full metadata. `attrs.has()` checks if a class is
attrs-decorated. The `__attrs_attrs__` dunder provides the same information.

**What can it extract?**

| Attribute | Supported | Accessor |
|-----------|-----------|----------|
| Field name | Yes | `attr.name` |
| Type | Yes | `attr.type` (annotation) |
| Default | Yes | `attr.default` |
| Description | **No** (no built-in description field) | N/A |
| Constraints | Via validators only | `attr.validator` |
| Enum/Literal | Via type annotation | `get_args()` |
| Required vs optional | Yes | `attr.default is NOTHING` |
| Metadata | Yes | `attr.metadata` (arbitrary dict) |

**Field transformers:** attrs provides a `field_transformer` hook that receives all fields
before class finalisation. This enables automatic addition of converters or validators based
on type, which is a form of type-driven auto-configuration. However, this is a *build-time*
hook, not runtime introspection for downstream consumers.

**Used by benchmark/ML tools?** Not prominently. attrs is used internally by some ML
libraries (e.g., cattrs for serialisation), but no major ML benchmark framework uses
`attrs.fields()` as an SSOT for test/doc generation. The ML ecosystem has largely
standardised on Pydantic or dataclasses for configuration.

**Key finding:** attrs has strong introspection (`fields()` is cleaner than
`dataclasses.fields()`), but lacks built-in `description` support and is not widely used
as a config introspection SSOT in the ML benchmark space.

**Sources:**
- [attrs How Does It Work](https://www.attrs.org/en/stable/how-does-it-work.html)
- [attrs API Reference](https://www.attrs.org/en/stable/api.html)

---

### 6. Python dataclasses (`dataclasses.fields()`)

**Does it support auto-discovery of all config fields?** Yes, but with less metadata than
Pydantic or attrs. `dataclasses.fields(MyClass)` returns a tuple of `Field` objects.

**What can it extract?**

| Attribute | Supported | Notes |
|-----------|-----------|-------|
| Field name | Yes | `field.name` |
| Type | Yes | `field.type` (string annotation in some cases) |
| Default | Yes | `field.default` / `field.default_factory` |
| Description | **No** | No built-in field description |
| Constraints | **No** | No validation — dataclasses are data holders only |
| Enum/Literal | Via `get_type_hints()` | Manual type inspection needed |
| Required vs optional | Yes | `field.default is MISSING` |
| Metadata | Yes | `field.metadata` (arbitrary `MappingProxy`) |
| init/repr/compare flags | Yes | `field.init`, `field.repr`, `field.compare` |

**How do tools use this?** Hydra/OmegaConf is the primary consumer of
`dataclasses.fields()` for config introspection (see section 2). lm-eval-harness uses a
dataclass-based `TaskConfig` — fields are accessed programmatically for config display
(`--show_config`) and YAML loading, but there is no systematic introspection that derives
tests or docs from the dataclass schema.

**Custom code needed for parity with Pydantic:** Significant. To get descriptions, you need
`field(metadata={"description": "..."})` — convention-based, not standardised. To get
constraints, you need a validation layer (Pydantic, beartype, or manual). To get Literal
options, you need `typing.get_args()` + `get_origin()`. Essentially, building SSOT
introspection on raw dataclasses means reimplementing half of what Pydantic provides
out of the box.

**Key finding:** Dataclasses provide the structural introspection (`fields()`) but lack the
validation metadata (constraints, descriptions) that makes SSOT useful. This is precisely
why Pydantic dominates configuration in Python ML tools — it bundles structure + validation
+ metadata in one definition.

**Sources:**
- [Python dataclasses documentation](https://docs.python.org/3/library/dataclasses.html)
- [lm-evaluation-harness task.py](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/api/task.py)

---

### 7. pytest (Auto-Parametrise from Schema)

**Does pytest support auto-generating test cases from model schemas?** Not natively. pytest's
`@pytest.mark.parametrize` requires explicit value lists. However, several libraries bridge
the gap:

**Hypothesis + Pydantic:**
- Pydantic v1 shipped a Hypothesis plugin (`pydantic.hypothesis_plugin`) that registered
  strategies for all Pydantic types. `st.builds(MyModel)` would auto-generate valid model
  instances respecting constraints.
- **Pydantic v2 dropped this integration.** The plugin was removed in v2.0 "in favor of
  studying a different mechanism." As of Feb 2026, it has not been reinstated.

**hypothesis-jsonschema:**
- `from_schema(json_schema)` generates Hypothesis strategies from any JSON Schema.
- Since Pydantic's `model_json_schema()` produces valid JSON Schema, the pipeline is:
  `model_json_schema() -> from_schema() -> @given(...)`.
- This supports all JSON Schema constraints: min/max, enum, pattern, required fields.
- However, this generates *valid schema instances*, not *meaningful test values*. A valid
  `batch_size` might be `2147483647` — technically valid but not a useful test case.

**Schemathesis:**
- Built on Hypothesis. Generates API test cases from OpenAPI schemas.
- `@schema.parametrize()` creates one test per API operation, with ~100 generated examples.
- Designed for web API testing, not config parameter testing.

**pydantic-fixturegen:**
- Generates deterministic fixtures from Pydantic v2 models. Newer library (2025+).
- Creates pytest fixtures that satisfy model validation.

**Our approach vs these tools:** Our introspection generates *domain-meaningful* test values
(e.g., `batch_size: [1, 4, 8]`, `temperature: [0.35, 0.7, 1.05]`), not arbitrary valid
values. This is a deliberate choice — we want to test real-world configurations, not random
valid ones. hypothesis-jsonschema solves a different problem (fuzzing for edge cases).

**Key finding:** No standard tool auto-generates *meaningful* test parametrisation from
Pydantic models. hypothesis-jsonschema generates *valid* data; our introspection generates
*representative* data. These are complementary, not competing.

**Sources:**
- [Pydantic v2 Hypothesis integration](https://docs.pydantic.dev/latest/integrations/hypothesis/)
- [hypothesis-jsonschema PyPI](https://pypi.org/project/hypothesis-jsonschema/)
- [Schemathesis pytest integration](https://schemathesis.readthedocs.io/en/latest/explanations/pytest/)
- [pydantic-fixturegen GitHub](https://github.com/CasperKristiansson/pydantic-fixturegen)

---

### 8. Sphinx / mkdocstrings (Doc Generation from Models)

**autodoc-pydantic (Sphinx):**
Sphinx's built-in `autodoc` does not understand Pydantic models — it treats them as regular
classes and misses field descriptions, validators, and constraints.
`autodoc_pydantic` is a dedicated Sphinx extension that:

- Extracts field names, types, defaults, descriptions, constraints from Pydantic models
- Documents validators and links them to their associated fields
- Shows JSON Schema output
- Shows model Config class settings
- Uses a `ModelInspector` that introspects `model_fields` and validator metadata

This is a direct parallel to our `generate_config_docs.py` script — both introspect Pydantic
models and produce documentation tables. `autodoc_pydantic` does this generically for any
model; our script adds domain context (backend grouping, preset display).

**griffe-pydantic (mkdocstrings):**
For mkdocstrings (MkDocs), the `griffe-pydantic` extension provides equivalent functionality:

- Extracts fields, defaults, constraints, validators, aliases
- Renders models as "pydantic-model" sections with Config/Fields/Validators subsections
- Stores metadata in the `extra` attribute of documentation objects

**Custom code needed:** Zero for generic model documentation. Both extensions work
out-of-the-box once configured. Custom code is only needed for domain-specific formatting
(e.g., grouping by backend, generating YAML examples).

**Key finding:** The doc generation community has converged on Pydantic model introspection
as the standard approach. Two independent ecosystems (Sphinx, MkDocs) each have dedicated
extensions that do exactly what our `generate_config_docs.py` does — confirming the pattern
is standard practice.

**Sources:**
- [autodoc-pydantic](https://autodoc-pydantic.readthedocs.io/en/stable/)
- [griffe-pydantic](https://mkdocstrings.github.io/griffe-pydantic/)
- [Pydantic Documentation Integrations](https://docs.pydantic.dev/latest/integrations/documentation/)

---

### 9. JSON Schema (Schema-Driven Validation & Test Generation)

**Relevance:** JSON Schema is the *interchange format* that makes multi-consumer introspection
portable. Pydantic's `model_json_schema()` produces it; downstream tools consume it.

**Schema-driven validation:**
- `jsonschema` library validates arbitrary JSON against a schema. Supports Draft 2020-12.
- In a Pydantic-based project, you rarely need `jsonschema` directly — Pydantic validates
  during construction. But the generated schema can be exported for external validators
  (e.g., YAML pre-validation before Pydantic parsing).

**Schema-driven test generation:**
- `hypothesis-jsonschema.from_schema()` generates Hypothesis strategies from JSON Schema.
- Constraints (`minimum`, `maximum`, `enum`, `pattern`) are respected.
- Nested `$ref`/`$defs` are resolved.
- This is the most rigorous schema-to-tests pipeline available.

**Schema-driven documentation:**
- JSON Schema is the input format for OpenAPI docs, Swagger UI, Redoc, and numerous doc
  generators. Pydantic's schema output is directly consumable by all of these.

**Key finding:** JSON Schema serves as the universal interchange format for the SSOT pattern.
Rather than building custom introspection, a project can: (1) define Pydantic models,
(2) export JSON Schema via `model_json_schema()`, (3) feed that schema into off-the-shelf
tools for validation, testing, and documentation. Our custom introspection module does
steps (1-2) via Pydantic's API and adds domain-specific logic on top.

**Sources:**
- [Python jsonschema library](https://github.com/python-jsonschema/jsonschema)
- [hypothesis-jsonschema](https://github.com/python-jsonschema/hypothesis-jsonschema)

---

## Summary Table

| Tool / Pattern | Field Discovery | Types | Defaults | Descriptions | Constraints | Enum/Literal | Test Gen | Doc Gen | CLI Gen | Custom LOC Needed |
|---------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|---|
| **Pydantic v2** (built-in) | Yes | Yes | Yes | Yes | Yes | Yes | No | No | No | 0 (reflection); ~50-100 for interpretation |
| **Hydra/OmegaConf** | Yes | Yes | Yes | No* | No | Partial | No | No | No* | Moderate |
| **Typer** | Func sigs only | Yes | Yes | Yes | No | Yes | No | No | Yes | 0 (for functions) |
| **pydantic-settings CLI** | Yes | Yes | Yes | Yes | Validation only | Yes | No | No | Yes | ~5 LOC config |
| **pydantic-typer** | Yes | Yes | Yes | Yes | Validation only | Yes | No | No | Yes | ~5 LOC config |
| **FastAPI** | Yes | Yes | Yes | Yes | Yes | Yes | Via Schemathesis | Yes (OpenAPI) | N/A | 0 |
| **attrs** | Yes | Yes | Yes | No | Via validators | Via type hints | No | No | No | Moderate |
| **dataclasses** | Yes | Yes | Yes | No | No | Via type hints | No | No | No | High |
| **hypothesis-jsonschema** | Consumes schema | Yes | Yes | N/A | Yes | Yes | Yes (random) | No | No | ~5 LOC |
| **autodoc-pydantic** | Yes | Yes | Yes | Yes | Yes | Yes | No | Yes | No | 0 (Sphinx config) |
| **griffe-pydantic** | Yes | Yes | Yes | Yes | Yes | Yes | No | Yes | No | 0 (MkDocs config) |
| **Our v1.x introspection** | Yes | Yes | Yes | Yes | Yes | Yes | Yes (domain-meaningful) | Yes | Yes | ~300 LOC |

*Hydra: descriptions available via convention (`metadata={"help": ...}`) but not auto-surfaced to `--help` (open issue #633). CLI overrides work but help text requires manual template.

---

## Recommendation

### The pattern is standard practice — preserve it

The SSOT introspection pattern (define typed config once, derive tests/docs/CLI from it) is
validated by multiple major frameworks:

- **FastAPI** is the canonical example at scale — millions of APIs use
  "Pydantic model → OpenAPI schema → docs + validation + SDK generation + test generation"
- **autodoc-pydantic** and **griffe-pydantic** independently implement the docs-from-model
  pattern for the two dominant Python doc ecosystems
- **pydantic-settings CLI** and **pydantic-typer** implement the CLI-from-model pattern
- **hypothesis-jsonschema** implements the tests-from-schema pattern

Our introspection module is the **multi-consumer** version: one module feeds tests, docs,
*and* CLI validation. This is rarer in peer tools (most implement only one consumer), but
FastAPI proves the multi-consumer approach works at scale.

### What should change in v2.0

1. **Lean harder on Pydantic v2's built-in API.** Our `_extract_param_metadata()` reimplements
   some of what `model_fields` + `FieldInfo` already provides. With the v2.0 config refactor
   (single `ExperimentConfig` with optional backend sections), the new introspection should
   use `model_json_schema()` as the primary discovery mechanism and add domain logic on top,
   rather than walking `model_fields` and manually unwrapping `Optional`/`Literal` types.

2. **Keep the test value generation.** No peer tool generates *domain-meaningful* test values
   from config schemas. hypothesis-jsonschema generates *valid* random data;
   pydantic-fixturegen generates *deterministic* fixtures. Neither produces "test
   `batch_size` at `[1, 4, 8]`" — that is custom domain logic worth preserving. Expected
   LOC: ~80-100 (the type-to-test-values mapping logic).

3. **Consider adopting autodoc-pydantic or griffe-pydantic** instead of our custom
   `generate_config_docs.py`. The community tools are maintained, support Pydantic v2, and
   handle edge cases (validators, nested models, aliases) that our ~100 LOC script does not.
   Our script adds backend-grouping and preset display, which could be a post-processing
   step on top of a standard tool.

4. **Consider pydantic-settings CLI** for the `llem config` command. It auto-generates CLI
   arguments from model fields with zero custom introspection code, handles nested models,
   and provides mutually-exclusive groups — features we would otherwise build manually.

5. **Preserve the constraint metadata functions** (`get_mutual_exclusions()`,
   `get_param_skip_conditions()`, `get_special_test_models()`,
   `get_params_requiring_gpu_capability()`). These encode domain knowledge that no generic
   tool can provide. They should remain as a thin layer on top of Pydantic's built-in
   introspection. Expected LOC: ~100-120 (unchanged from v1.x, these are data, not logic).

### Estimated v2.0 introspection size

| Component | v1.x LOC | v2.0 estimate | Rationale |
|-----------|----------|---------------|-----------|
| Field discovery (`get_params_from_model`) | ~80 | ~30 | Delegate to `model_json_schema()` |
| Test value generation | ~60 | ~80 | Preserve; slightly expand for new config shape |
| Domain constraint metadata | ~120 | ~120 | Data tables — no change in pattern |
| Backend routing (`get_backend_params` etc.) | ~40 | ~20 | Simpler with single `ExperimentConfig` |
| **Total** | **~300** | **~250** | Net reduction from leaning on Pydantic v2 API |

The pattern is sound. The v2.0 rewrite should reduce custom reflection code by delegating
to `model_json_schema()`, while preserving the domain-specific layers (test values,
constraint metadata, backend routing) that no off-the-shelf tool provides.
