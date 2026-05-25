# Phase 3 Consumer Audit: introspection.py Consumers

**Audit Purpose:** Assess how 8 downstream consumers depend on `src/llenergymeasure/config/introspection.py` return value shapes, in preparation for Phase 3 rewrite (data-driven YAML-based parameter metadata instead of Pydantic Literal introspection).

**Reference:** 
- `src/llenergymeasure/config/introspection.py` — SSOT module providing:
  - `get_engine_params(engine: str) -> dict[str, dict[str, Any]]` — paths like `"transformers.batch_size"` → metadata dicts with keys: `path`, `name`, `default`, `type_str`, `options`, `test_values`, `engine_support`, `constraints`
  - `get_shared_params() -> dict[str, dict[str, Any]]` — same shape, universal params
  - `get_swept_field_paths(experiments) -> set[str]` — dotted field paths that vary across experiments
  - `get_display_label(field_info, field_name) -> str` — display label from json_schema_extra
  - `get_field_role(field_info) -> str | None` — 'workload' or 'experimental' role metadata

---

## Consumer 1: `src/llenergymeasure/config/grid.py`

**File:** `/home/h.baker@hertie-school.lan/workspace/llenergymeasure/src/llenergymeasure/config/grid.py`

**Imports:**
```python
from llenergymeasure.config.introspection import (
    get_display_label,
    get_field_role,
    get_swept_field_paths,
)
```

**Call Sites & Usage:**

1. **Line 442:** `get_swept_field_paths(experiments)`
   - **Context:** Building preflight panel for study display
   - **Usage:** Returns `set[str]` of dotted field paths; consumer annotates fields with "+" marker if path in swept_paths
   - **Return shape accessed:** Set of path strings (e.g., `"task.dataset.n_prompts"`, `"transformers.engine_params.dtype"`)
   - **Code snippet:**
     ```python
     swept_paths = get_swept_field_paths(experiments)
     # ... later ...
     is_swept = ds_path in swept_paths  # e.g., ds_path = "task.dataset.n_prompts"
     ```

2. **Line 499:** `get_field_role(ds_fi)` 
   - **Context:** Filtering workload fields from DatasetConfig for display
   - **Usage:** Reads role metadata to filter on role == "workload"
   - **Return shape accessed:** String value ("workload") or None
   - **Code snippet:**
     ```python
     ds_role = get_field_role(ds_fi)
     if ds_role != "workload":
         continue
     ```

3. **Line 509:** `get_display_label(ds_fi, ds_field)`
   - **Context:** Display label for DatasetConfig fields in panel
   - **Usage:** Returns display string for field name
   - **Return shape accessed:** String (e.g., "N Prompts" for n_prompts)
   - **Code snippet:**
     ```python
     label = get_display_label(ds_fi, ds_field)
     workload_rows.append((label, val_str, is_decl, is_swept))
     ```

4. **Line 517:** `get_display_label(fi, field_name)` for task fields
   - **Same as above for TaskConfig fields**

5. **Line 522-524:** `get_display_label(energy_fi, "energy_sampler")` for measurement field
   - **Same usage pattern**

**Sensitivity Assessment:**

- **Path-format sensitivity:** NO — `get_swept_field_paths()` returns opaque path strings. Consumer iterates dotted paths via `in` membership tests. Works with any path format.
- **Metadata-sensitive:** NO — `get_field_role()` and `get_display_label()` read Pydantic FieldInfo objects (not return values from introspection that would change). These helpers are stable.
- **Shape-agnostic:** YES — This consumer is robust to Phase 3 changes. It accesses swept paths as a set of strings and metadata display labels from FieldInfo, not from parameter metadata dicts.

**Classification:** `shape-agnostic`

---

## Consumer 2: `src/llenergymeasure/api/_impl.py`

**File:** `/home/h.baker@hertie-school.lan/workspace/llenergymeasure/src/llenergymeasure/api/_impl.py`

**Imports:**
```python
from llenergymeasure.config.introspection import get_swept_field_paths
```

**Call Sites & Usage:**

1. **Line 450-454:** `get_swept_field_paths(study.experiments)` in resolution log building
   - **Context:** Building per-experiment resolution logs showing which fields were swept vs CLI-overridden
   - **Usage:** Returns set of swept field paths
   - **Return shape accessed:** `set[str]` of dotted paths
   - **Code snippet:**
     ```python
     swept_fields = get_swept_field_paths(study.experiments)
     seen_hashes: set[str] = set()
     for exp in study.experiments:
         h = compute_declared_config_hash(exp)
         if h in seen_hashes:
             continue
         seen_hashes.add(h)
         resolution_logs[h] = build_resolution_log(
             exp.model_dump(),
             cli_overrides=cli_overrides,
             swept_fields=swept_fields,  # PASSED TO build_resolution_log
         )
     ```

**Sensitivity Assessment:**

- **Path-format sensitivity:** YES (conditional) — `swept_fields` is passed to `build_resolution_log()` which likely iterates over paths and checks them in field comparison logic. If path format changes (e.g., `transformers.batch_size` → `transformers.engine_params.batch_size`), the resolver logic that checks `if field_path in swept_fields` would fail to match.
- **Metadata-sensitive:** NO — only accesses the set itself, not field metadata.
- **Shape-agnostic:** NO — depends on path string format consistency.

**Classification:** `path-sensitive` (passed downstream to resolution log builder)

---

## Consumer 3: `src/llenergymeasure/infra/version_handshake.py`

**File:** `/home/h.baker@hertie-school.lan/workspace/llenergymeasure/src/llenergymeasure/infra/version_handshake.py`

**Imports:**
```python
from llenergymeasure.config.introspection import get_experiment_config_schema
```

**Call Sites & Usage:**

1. **Line 110:** `get_experiment_config_schema()` in fingerprint computation
   - **Context:** Computing SHA-256 fingerprint of ExperimentConfig schema for Docker image validation
   - **Usage:** Returns Pydantic JSON schema dict; serialized to JSON and hashed
   - **Return shape accessed:** Full JSON schema structure with `$defs`, `properties`, etc.
   - **Code snippet:**
     ```python
     payload = json.dumps(
         get_experiment_config_schema(),
         sort_keys=True,
         separators=(",", ":"),
     ).encode("utf-8")
     return hashlib.sha256(payload).hexdigest()
     ```

**Sensitivity Assessment:**

- **Path-format sensitivity:** NO — function reads entire JSON schema object, not individual paths.
- **Metadata-sensitive:** NO — consumes the full schema as a blob; only uses it for hashing.
- **Shape-agnostic:** YES — Phase 3 won't change `ExperimentConfig.model_json_schema()` itself, only how parameter metadata is sourced at runtime.

**Classification:** `shape-agnostic`

---

## Consumer 4: `tests/unit/config/test_config_introspection.py`

**File:** `/home/h.baker@hertie-school.lan/workspace/llenergymeasure/tests/unit/config/test_config_introspection.py`

**Imports:**
```python
from llenergymeasure.config.introspection import (
    get_all_params,
    get_display_label,
    get_engine_params,
    get_experiment_config_schema,
    get_field_role,
    get_param_test_values,
    get_shared_params,
    get_swept_field_paths,
    get_validation_rules,
    list_all_param_paths,
)
```

**Call Sites & Usage (selective key tests):**

1. **Line 35:** `get_engine_params("transformers")` → asserts `"transformers.batch_size" in params`
   - **Checks:** Dict keys are param paths like `"transformers.batch_size"`; each value has `"engine_support"` key
   - **Test 40-45:** Validates param metadata structure: `"engine_support" in meta`
   - **Sensitivity:** PATH and METADATA — test expects exact path `"transformers.batch_size"` (flat, old style). If renamed to nested path, test fails.

2. **Line 49:** `get_engine_params("vllm")` → asserts `"vllm.engine.max_num_seqs" in params`
   - **Sensitivity:** PATH — expects nested path format (already supports this).

3. **Line 56-62:** `get_engine_params("tensorrt")` — asserts nested paths like `"tensorrt.quant_config.quant_algo"` and `"tensorrt.kv_cache_config.free_gpu_memory_fraction"`
   - **Sensitivity:** PATH — hard-coded path expectations.

4. **Line 91:** `get_shared_params()` — asserts `"dataset.n_prompts" in params` and keys include `"engine_support"`
   - **Sensitivity:** PATH and METADATA — expects `engine_support` field presence.

5. **Line 162-165:** `get_param_test_values("transformers.dtype")` → expects `{"float32", "float16", "bfloat16"}` returned
   - **Sensitivity:** METADATA — test explicitly checks `test_values` content for Literal-typed fields. If dtype becomes `str + extra='allow'` with empty `test_values`, test fails.

6. **Line 254:** `exp.transformers.engine_params.dtype == dt` for dtype param
   - **Test indirectly validates param paths work with nested structures**

7. **Line 346-354:** `get_swept_field_paths([exp1, exp2])` asserts `"transformers.engine_params.dtype" in result`
   - **Sensitivity:** PATH — expects exact path format.

8. **Line 359-364:** `get_swept_field_paths([exp1, exp2])` asserts `"task.dataset.n_prompts" in result`
   - **Sensitivity:** PATH — expects exact path format.

9. **Line 378:** `get_display_label()` and `get_field_role()` — test FieldInfo helpers
   - **Sensitivity:** METADATA — tests that role annotation reads work (not shape-dependent).

**Sensitivity Assessment:**

- **Path-format sensitivity:** YES (strong) — Many tests hard-code old-style flat paths (`transformers.batch_size`, `transformers.dtype`) and expect them as dict keys. Phase 3 path rewrite would break these assertions.
- **Metadata-sensitive:** YES (strong) — Tests assert `test_values` presence and content for Literal-typed fields. If Literal-typed fields become `str + extra='allow'`, their `test_values` become empty, breaking L162-165.
- **Shape-agnostic:** NO

**Classification:** `tests-only` (unit tests; failures are expected during Phase 3 as reference implementations change, so failures themselves validate the migration)

---

## Consumer 5: `scripts/runtime-test-orchestrator.py`

**File:** `/home/h.baker@hertie-school.lan/workspace/llenergymeasure/scripts/runtime-test-orchestrator.py`

**Imports:**
```python
from llenergymeasure.config.introspection import (
    get_engine_params,
    get_shared_params,
)
```

**Call Sites & Usage:**

1. **Line 128-157:** `get_engine_params(engine)` and `get_shared_params()`
   - **Context:** Discover all test-able parameters via introspection; used by runtime test orchestrator
   - **Usage:** Iterates `{param_path: meta}` dicts; reads `meta["test_values"]` for each param
   - **Return shape accessed:**
     ```python
     for param_path, meta in engine_params.items():
         test_values = meta.get("test_values", [])
         if test_values:
             params[param_path] = test_values
     ```
   - **Code snippet shows filter:**
     ```python
     for param_path, meta in shared.items():
         test_values = meta.get("test_values", [])
         if test_values:
             shared_test_values[param_path] = test_values
     ```

**Sensitivity Assessment:**

- **Path-format sensitivity:** YES — Consumer stores `{param_path: [test_values]}` and later uses param_path in test case creation (L467). If paths change format, test routing to engines breaks. 
- **Metadata-sensitive:** YES (strong) — Directly reads `meta["test_values"]`. If Literal-typed fields lose `test_values` (become empty list), those params will be skipped entirely (L150 `if test_values:`). The QUICK_PARAMS filter (L64-68) hard-codes old paths.
- **Shape-agnostic:** NO

**Sensitivity Failures if Phase 3 lands:**
- Missing `test_values` for dtype (if it becomes `str + extra='allow'`) → dtype won't be tested at all
- Path format change → test cases routed incorrectly or param discovery fails

**Classification:** `path-sensitive` + `metadata-sensitive` → reclassify as `metadata-sensitive` (primary pain point is missing test_values)

---

## Consumer 6: `scripts/check_pydantic_matches_discovered.py`

**File:** `/home/h.baker@hertie-school.lan/workspace/llenergymeasure/scripts/check_pydantic_matches_discovered.py`

**Imports:**
```python
from llenergymeasure.config.introspection import get_engine_params
```

**Call Sites & Usage:**

1. **Line 254:** `get_engine_params(engine)` in `_get_pydantic_leaves()`
   - **Context:** Drift-checking script; compares Pydantic models to discovered engine schemas
   - **Usage:** Iterates `{param_path: meta}` and extracts `meta["name"]` (leaf field name)
   - **Return shape accessed:**
     ```python
     for _path, meta in params.items():
         leaf_name = meta["name"]
         result[leaf_name] = all_props.get(leaf_name, {})
     ```

**Sensitivity Assessment:**

- **Path-format sensitivity:** YES (weak) — Script uses `meta["name"]` (the leaf field name extracted from path), not the full path itself. If the path structure changes but `meta["name"]` is populated correctly, this still works. However, if Phase 3 changes what "name" means (e.g., from `"batch_size"` to a different leaf extraction logic), it breaks.
- **Metadata-sensitive:** YES (conditional) — Relies on `meta["name"]` field presence and correctness. If Phase 3 changes field extraction, breakage occurs.
- **Shape-agnostic:** NO

**Classification:** `metadata-sensitive` (depends on `meta["name"]` field structure)

---

## Consumer 7: `scripts/generate_invalid_combos_doc.py`

**File:** `/home/h.baker@hertie-school.lan/workspace/llenergymeasure/scripts/generate_invalid_combos_doc.py`

**Imports:**
```python
from llenergymeasure.config.introspection import (
    get_capability_matrix_markdown,
    get_runtime_limitations,
    get_streaming_constraints,
    get_validation_rules,
)
```

**Call Sites & Usage:**

1. **Line 45:** `get_validation_rules()`
   - **Usage:** Iterates list of dicts with keys `['engine', 'combination', 'reason', 'resolution']`; renders Markdown table
   - **Return shape accessed:** List of dicts with fixed string keys

2. **Line 64:** `get_streaming_constraints()`
   - **Usage:** Iterates list of dicts with keys `['engine', 'parameter', 'behaviour', 'impact']`; renders Markdown table

3. **Line 98:** `get_runtime_limitations()`
   - **Usage:** Iterates list of dicts with keys `['engine', 'parameter', 'limitation', 'resolution']`; renders Markdown table

4. **Line 109:** `get_capability_matrix_markdown()`
   - **Usage:** Returns pre-formatted Markdown string; inserted directly into output

**Sensitivity Assessment:**

- **Path-format sensitivity:** NO — Functions return high-level rule/constraint/limitation dicts and pre-rendered Markdown. No low-level path structure dependency.
- **Metadata-sensitive:** NO — Accesses fixed dict keys (`engine`, `combination`, etc.) that are stable API of introspection module.
- **Shape-agnostic:** YES — Phase 3 changes parameter metadata sourcing, not the validation rules, streaming constraints, or capability matrix logic.

**Classification:** `shape-agnostic`

---

## Consumer 8: `scripts/generate_curation_doc.py`

**File:** `/home/h.baker@hertie-school.lan/workspace/llenergymeasure/scripts/generate_curation_doc.py`

**Imports:**
```python
from llenergymeasure.config.introspection import get_engine_params
```

**Call Sites & Usage:**

1. **Line 36:** `get_engine_params(engine)` in `_get_curated_names()`
   - **Context:** Determines which discovered parameters are curated (exposed) by llem
   - **Usage:** Iterates `{param_path: meta}` and extracts `meta["name"]` (leaf field names)
   - **Return shape accessed:**
     ```python
     params = get_engine_params(engine)
     return {meta["name"] for meta in params.values()}
     ```

**Sensitivity Assessment:**

- **Path-format sensitivity:** YES (weak) — Uses `meta["name"]` extraction, same as consumer 6. If "name" field changes, breaks.
- **Metadata-sensitive:** YES — Relies on `meta["name"]` field presence and correctness.
- **Shape-agnostic:** NO

**Classification:** `metadata-sensitive` (same dependency as consumer 6 on `meta["name"]` field)

---

## Summary Table

| Consumer | File | Primary Sensitivity | Classification | Failure Mode if Phase 3 Lands |
|----------|------|---------------------|-----------------|-------------------------------|
| 1 | `config/grid.py` | None (path set passed opaquely) | `shape-agnostic` | None (robust to rewrite) |
| 2 | `api/_impl.py` | Path format (passed downstream) | `path-sensitive` | Resolution log builder fails to match swept fields if paths change |
| 3 | `infra/version_handshake.py` | None (consumes full schema blob) | `shape-agnostic` | None (schema fingerprint stable) |
| 4 | `tests/unit/config/test_config_introspection.py` | Path format + test_values metadata | `tests-only` | Test assertions on hard-coded paths and `test_values` fail (expected during migration) |
| 5 | `scripts/runtime-test-orchestrator.py` | test_values metadata (Literal fields lose it) | `metadata-sensitive` | Parameters with empty `test_values` are skipped; dtype won't be tested |
| 6 | `scripts/check_pydantic_matches_discovered.py` | meta["name"] field structure | `metadata-sensitive` | Drift detection fails if "name" extraction logic changes |
| 7 | `scripts/generate_invalid_combos_doc.py` | None (stable rule/constraint APIs) | `shape-agnostic` | None (validation rules/limitations stable) |
| 8 | `scripts/generate_curation_doc.py` | meta["name"] field structure | `metadata-sensitive` | Curation doc generation fails if "name" extraction logic changes |

---

## Key Findings

1. **Path-sensitive consumer(s):** 1 — `api/_impl.py` (passed downstream to resolution log builder)

2. **Metadata-sensitive consumers:** 3 — `runtime-test-orchestrator.py` (critical: missing `test_values` for Literal-typed params like `dtype`), `check_pydantic_matches_discovered.py` (meta["name"]), `generate_curation_doc.py` (meta["name"])

3. **Critical risk:** Literal-typed fields (`dtype`, quantization enums, etc.) that become `str + extra='allow'` lose their `test_values` list, breaking runtime test orchestration and potentially breaking unit tests.

4. **Phase 3 rewrite implications:**
   - If bundled YAML param metadata includes `options` and `test_values` fields for all param types, no change needed.
   - If new schema drops `test_values` for non-Literal types, test orchestration breaks.
   - If "name" field extraction logic changes, drift-check and curation doc scripts break.
   - If path format changes (e.g., nested per-option-A migration), resolution log builder fails to match swept fields.

