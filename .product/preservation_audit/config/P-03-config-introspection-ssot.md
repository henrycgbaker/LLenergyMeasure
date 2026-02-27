# P-03: Config Introspection SSOT

**Module**: `src/llenergymeasure/config/introspection.py`
**Risk Level**: HIGH
**Decision**: Keep — v2.0
**Planning Gap**: Not mentioned in any planning document; the config refactor risks silently breaking it.

---

## What Exists in the Code

**Primary file**: `src/llenergymeasure/config/introspection.py` (~300 lines)
**Key functions**:
- `get_backend_params(backend: str) -> dict[str, dict]` (line 196) — extracts ALL parameter metadata from a backend Pydantic model via reflection
- `get_shared_params() -> dict[str, dict]` (line 235) — universal params (decoder, precision, streaming, max tokens)
- `get_params_from_model(model_class, prefix, include_nested) -> dict` (line 131) — recursive introspection into nested Pydantic models
- `_extract_param_metadata(field_name, field_info) -> dict` (line 31) — per-field metadata: type, default, description, constraints, test values
- `_get_custom_test_values() -> dict` (line 175) — manual overrides for params that need special test values (e.g., `vllm.max_model_len`, `trt.max_input_len`)

This is the SSOT that the test suite, documentation generator, and CLI validation layer all read from. It uses Pydantic's reflection API (`model_fields`, `FieldInfo`, `get_origin()`, `get_args()`) to derive parameter metadata without any manual registry.

## Why It Matters

Enables zero-maintenance test generation — new config fields are auto-discovered, no manual test lists needed. The doc generator derives the full backend parameter capability table from it. Without it, both break silently. The `CLAUDE.md` notes it as a hard dependency; no planning document does.

## Planning Gap Details

`decisions/config-architecture.md` discusses the composition decision at length but does not mention that `introspection.py` exists or that it must survive the refactor. The new composition model changes the Pydantic model hierarchy (single `ExperimentConfig` with optional backend sections instead of `BaseConfig + BackendConfig` subclasses). The introspection functions that traverse model hierarchies via `get_params_from_model()` must be rewritten to work with the new structure.

## Recommendation for Phase 5

Add to `decisions/config-architecture.md`:

> **Config Introspection SSOT**: `config/introspection.py` must be explicitly ported as part of the
> config refactor. All downstream consumers depend on it:
> - Test suite: `get_param_test_values()`, `get_param_skip_conditions()`
> - Doc generator: `get_backend_capabilities()` (produces markdown capability tables)
> - CLI validation: `get_validation_rules()`
>
> After porting to the composition model, verify that `get_backend_params('pytorch')`,
> `get_backend_params('vllm')`, `get_backend_params('tensorrt')`, and `get_shared_params()`
> all return correct results. The Pydantic reflection API (`model_fields`, `FieldInfo`) is
> version-sensitive — test against the pinned Pydantic version.
