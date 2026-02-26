# Schema Migration Design

**Last updated**: 2026-02-25
**Source decisions**: [../decisions/result-schema-migration.md](../decisions/result-schema-migration.md)
**Status**: DRAFT — fields confirmed; implementation details need review

---

## Strategy: Additive Only in v2.x; No Migration Functions

- Every result file includes `schema_version: str`
- Old files loaded by newer tool: missing fields → `None`, no error
- No migration functions in v2.x (schema is still evolving)
- Warn when aggregating results across schema versions
- Breaking changes (field renames, type changes) deferred to v3.0 if ever needed

---

## Schema Version Map

| Tool version | Schema version | New fields added |
|---|---|---|
| v2.0 (core) | `"2.0"` | Base fields (energy, throughput, latency, FLOPs, warmup, config_hash, measurement_methodology, steady_state_window, environment_snapshot) |
| v2.0 (Zeus) | `"2.0"` | Zeus energy fields, baseline_power_w, thermal_time_series |
| v2.0 (Docker) | `"2.0"` | `docker_image_digest` |

<!-- TODO: The schema version map needs to be pinned against actual implementation.
     The field additions listed above are best-guess based on versioning-roadmap.md
     but have not been formally decided. Confirm which fields are added in each milestone
     before implementation begins. Note: all are v2.0 milestones, not separate versions. -->

---

## `schema_version` in ExperimentResult

```python
# src/llenergymeasure/domain/results.py

class ExperimentResult(BaseModel):
    schema_version: str = "2.0"   # set to current version on creation

    # ... all other fields ...
```

**Writing**: `schema_version` is set to the current tool version's schema string on creation.
**Reading**: Pydantic loads whatever `schema_version` is stored in the file — no transformation.
Missing fields from older schema versions load as `None` (Pydantic default).

```python
# Loader — no migration, just permissive reading
@classmethod
def from_json(cls, path: str | Path) -> "ExperimentResult":
    data = json.loads(Path(path).read_text())
    # Pydantic handles missing fields → None automatically
    return cls.model_validate(data)
```

---

## Cross-Version Aggregation Warning

When a `StudyResult` aggregates `ExperimentResult` files across schema versions (e.g. from
an interrupted study resumed after a version upgrade), warn the user:

```python
def _check_schema_consistency(results: list[ExperimentResult]) -> None:
    versions = {r.schema_version for r in results}
    if len(versions) > 1:
        import warnings
        warnings.warn(
            f"Aggregating results across schema versions: {sorted(versions)}. "
            "Some fields may be missing in older results (shown as None). "
            "Re-run older experiments to get consistent schema versions.",
            stacklevel=2,
        )
```

---

## Backwards-Compatible Loading Example

```python
# A v2.0 result file loaded by a v2.1 tool
# v2.0 file contents:
{
    "schema_version": "2.0",
    "energy_total_j": 312.4,
    "tokens_per_second": 847.0
    # ... no zeus_energy fields (v2.1 addition)
}

# After loading with v2.1 tool:
result = ExperimentResult.from_json("old_result.json")
assert result.schema_version == "2.0"
assert result.energy_total_j == 312.4
assert result.zeus_energy_j is None      # v2.1 field — missing in old file → None
assert result.baseline_power_w is None   # v2.1 field — missing in old file → None
```

User-visible warning (printed to stderr when loading an old file):
```
Warning: Result file schema_version="2.0"; current schema is "2.1".
  Missing fields loaded as None: zeus_energy_j, baseline_power_w, thermal_time_series
  To rerun with updated schema: llem run experiment.yaml
```

<!-- TODO: Decide whether the warning should be per-file or only when aggregating across
     versions. A single old file loaded silently might be the right behaviour (user just
     wants to inspect it). The warning is most useful when mixing versions in a study. -->

---

## No Migration Functions in v2.x

Re-running the experiment is the correct migration path. Fabricating field values from
old data would produce misleading results — the new fields (e.g., `zeus_energy_j`) are
genuinely measured, not derivable from old measurements.

If v3.0 makes a breaking change (field renames, type changes):
- Migration functions will be added at that point
- A `llem results migrate <file>` command may be added
- Not designed now — premature until the schema stabilises

---

## StudyResult Schema Version

`StudyResult` has its own `schema_version` separate from `ExperimentResult.schema_version`.

<!-- TODO: The designs/result-schema.md currently has `schema_version: str = "1.0"` for
     StudyResult, but this decision doc says v2.0 tool uses schema_version "2.0".
     INCONSISTENCY — resolve: should StudyResult schema_version track the tool version
     (matching ExperimentResult) or be independent? Recommendation: match tool version
     for consistency. Update result-schema.md. -->

---

## Related

- [../decisions/result-schema-migration.md](../decisions/result-schema-migration.md): Decision rationale
- [result-schema.md](result-schema.md): ExperimentResult and StudyResult field definitions
- [study-resume.md](study-resume.md): Cross-version schema issues in study resume
