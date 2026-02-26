# Result Schema Migration

**Status:** Accepted
**Date decided:** 2026-02-19
**Last updated:** 2026-02-26
**Research:** N/A

## Decision

`schema_version` field on every result file. Missing fields in older files load as `None` (no error). No migration functions in v2.x — re-run to get new fields. Warn when aggregating across schema versions.

---

## Context

> **Updated (2026-02-26):** v2.0/v2.1/v2.2 field split collapsed — all result fields ship at v2.0 per NEEDS_ADDRESSING.md decision #50.

LLenergyMeasure ships all result fields at v2.0, but the schema will evolve in future major
versions (v3.0+). Without a versioning strategy, tools loading old result files would fail
silently or hard-crash when expected fields are absent. The schema also needs to remain stable
enough that researchers can aggregate results across runs without unexpected data loss.

Constraints: (1) the schema is still evolving in v2.x — heavy migration machinery would be
premature; (2) result files are on-disk JSON — users may have files from multiple tool versions;
(3) re-running experiments is acceptable (and preferred) when new fields are needed.

## Considered Options

### Sub-decision 1: How to identify schema version

| Option | Pros | Cons |
|--------|------|------|
| **`schema_version` field on every result file** | Explicit, machine-readable, no field-presence inspection required. | One extra field per file (negligible). |
| Infer version from field presence | No extra field. | Fragile — fields may be absent for other reasons; complex detection logic. |
| Tool version as proxy for schema version | Simple — one source of truth. | Tool version ≠ schema version once backports exist; tight coupling. |

### Sub-decision 2: Backwards compatibility for missing fields

| Option | Pros | Cons |
|--------|------|------|
| **Missing fields in older files load as `None`, no error** | Always readable. Tools degrade gracefully. Users can load old files and get partial data. | Silent data loss — aggregation tools may not notice absent fields. |
| Hard error on schema mismatch | Forces users to migrate explicitly. | Breaks all tooling on every minor version bump; unworkable during active development. |
| Auto-migrate on load | Transparent to users. | Cannot fabricate measured values (e.g. `zeus_energy_j`) — migration would invent data. |

### Sub-decision 3: Migration functions in v2.x

| Option | Pros | Cons |
|--------|------|------|
| **No migration functions in v2.x** | Zero complexity while schema evolves. Encourages re-running over interpolating. | Users cannot upgrade old files to new schema without re-running. |
| Write migration functions per minor version | Users keep old files and get new fields. | Cannot fabricate measured values — migration would produce invalid data. Schema still evolving so migrations written now may be wrong. |

### Sub-decision 4: Aggregating results across schema versions

| Option | Pros | Cons |
|--------|------|------|
| **Warn when aggregating results across schema versions** | User is informed that fields may differ; can make conscious choice. | Warning may be noisy if user intentionally mixes versions. |
| Silently aggregate (best-effort) | No interruption. | User may not notice that half their aggregate has `None` for new fields. |
| Hard error on cross-version aggregation | Forces homogeneity. | Too strict — cross-version comparison is a legitimate use case. |

## Decision

We will add a `schema_version: str` field to every result file. Old files loaded by a newer
tool will have missing fields populated as `None` with no error raised. No migration functions
will be written for v2.x additive changes. When aggregating across schema versions, the tool
will emit a warning listing which fields will be `None` for older files.

Rationale: re-running the experiment is the correct way to obtain new measured fields —
interpolating or fabricating values from old data would compromise scientific integrity. The
warning-on-aggregation approach gives users the information they need without blocking a
legitimate use case.

## Consequences

Positive:
- Result files are always loadable regardless of tool version mismatch.
- Schema version is explicitly machine-readable — no fragile field-presence detection.
- No complexity burden during active v2.x schema evolution.

Negative / Trade-offs:
- Users cannot upgrade old result files without re-running experiments.
- Cross-version aggregation silently produces `None` for new fields (mitigated by warning).

Neutral / Follow-up decisions triggered:
- Migration functions will be reconsidered at v3.0 if a breaking schema change (field renames,
  type changes) is introduced. All result fields ship at v2.0, so no v2.x additive migration
  is needed.
- The upgrade warning format is specified below; CLI/library must implement it consistently.

## Schema Version Map

> **Updated (2026-02-26):** CI fields moved from v2.1 to v2.0 per NEEDS_ADDRESSING.md decision #50 — all result fields ship at v2.0.

| Tool version | Schema version | Fields |
|-------------|----------------|--------|
| v2.0 | `"2.0"` | `measurement_config_hash`, `measurement_methodology`, `steady_state_window`, `schema_version`, `zeus_energy_j`, `baseline_power_w`, `thermal_time_series`, `warmup_excluded_samples`, `environment_snapshot`, bootstrap CI fields, `docker_image_digest` (see designs/result-schema.md) |

> **Superseded (2026-02-26):** The previous version of this table split fields across v2.0, v2.1, and v2.2. Per NEEDS_ADDRESSING.md decision #50 and #61/#62 (v2.2 elimination), all fields now ship at v2.0. The original v2.1/v2.2 rows were:
>
> | v2.1 | `"2.1"` | `schema_version` on ExperimentResult, `zeus_energy_j`, `baseline_power_w`, `thermal_time_series`, `warmup_excluded_samples`, `environment_snapshot`, bootstrap CI fields |
> | v2.2 | `"2.2"` | `docker_image_digest` |

**Source**: `designs/result-schema.md` — all fields are v2.0.

## Upgrade Warning

> **Superseded (2026-02-26):** The original example below showed a v2.0 → v2.1 upgrade scenario. Since all fields now ship at v2.0 (decision #50), this specific scenario no longer applies. The warning mechanism remains valid for future schema changes (v2.0 → v3.0+). Updated example follows.

When loading an older schema version file with a newer tool:
```
Warning: Result file schema_version="2.0"; current schema is "3.0".
  Missing fields: [hypothetical_v3_field_a], [hypothetical_v3_field_b] → None
  To rerun with updated schema: llem run experiment.yaml
```

## Migration Path

No automatic migration. The user re-runs the experiment if they need the new fields.
This is intentional — re-running ensures the new fields are actually measured, not
interpolated or fabricated from old data.

Since all result fields ship at v2.0, there are no v2.x additive schema changes requiring
migration. If v3.0 makes a breaking change (field renames, type changes), migration
functions will be added at that point.

## Related

- [../designs/result-schema.md](../designs/result-schema.md) — full field definitions per version
- [../designs/schema-migration.md](../designs/schema-migration.md) — implementation design for schema versioning
- [study-resume.md](study-resume.md) — `study_manifest.json` schema versioning
