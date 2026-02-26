# Backward Compatibility and API Stability

**Status:** Accepted
**Date decided:** 2026-02-19
**Last updated:** 2026-02-25
**Research:** N/A

## Decision

Stable API = `__init__.py` exports only (SemVer-guaranteed from v2.0.0). One minor version deprecation window (deprecated in v2.1 → removed in v2.2). Pre-v2.0 code has no stability guarantee. Protocol classes (`EnergyBackend`, `InferenceBackend`) are opt-in stable API for custom backend implementors.

---

## Context

At v2.0.0, llenergymeasure establishes its first stability baseline. Researchers and practitioners who integrate the library into scripts or CI pipelines need to know which parts of the API they can depend on, and how long deprecated features will be supported before removal. Without an explicit policy, every release risks breaking downstream code with no warning.

The codebase has a large internal surface (`orchestration/`, `core/`, `config/`, `cli/`) that must remain free to refactor without version ceremonies. The public surface must be narrow and explicit. Peer tools (httpx, SQLAlchemy, Pydantic, lm-eval) all define stability by `__init__.py` exports — this is the established industry convention for Python libraries.

---

## L1 — What Counts as Stable API

### Considered Options

| Option | Pros | Cons |
|--------|------|------|
| **`__init__.py` exports only — Chosen** | Industry standard (httpx, SQLAlchemy, Pydantic, lm-eval all use this); forces clean architectural thinking; frees internal modules to refactor | Users importing internal modules are surprised when they break; requires documentation |
| All public modules stable | Maximum stability for power users | Internal modules need freedom to refactor; creates huge stability surface that constrains all refactoring |
| Stability by documentation annotation | Flexible | Ambiguous; no machine-enforceable boundary; leads to drift |

### Decision

We will treat `llenergymeasure/__init__.py` exports as the **sole stable, SemVer-guaranteed
API** from v2.0.0 onwards. All other modules are internal and may change without notice.

Rationale: httpx, SQLAlchemy, Pydantic, and lm-eval all define stability this way.
Internal modules need freedom to refactor without version ceremonies. A narrow public
surface forces clean architectural thinking about what truly belongs in the public API.

### Stable exports (v2.0.0 surface)

```python
from llenergymeasure import (
    run_experiment,      # function
    run_study,           # function
    ExperimentConfig,    # Pydantic model
    StudyConfig,         # Pydantic model
    ExperimentResult,    # Pydantic model
    StudyResult,         # Pydantic model
)
from llenergymeasure.exceptions import (
    LLEMError,           # base exception (stable)
    ConfigError,         # stable
    BackendError,        # stable
    PreFlightError,      # stable
    ExperimentError,     # stable
    StudyError,          # stable
)
```

**Not stable (internal, may change without notice):**
- `llenergymeasure.orchestration.*` — `ExperimentOrchestrator`, `StudyRunner`
- `llenergymeasure.core.*` — inference backends, energy backends, metrics
- `llenergymeasure.config.*` — internal config loading, SSOT dicts
- `llenergymeasure.cli.*` — CLI implementation
- Any module not explicitly listed in `__init__.py`

Users who import internal modules do so at their own risk. Internal APIs can change in any
release, including patch versions. This is documented in the API reference.

**Exception — Protocol classes**: `EnergyBackend` and `InferenceBackend` in
`llenergymeasure.core` are available as **opt-in stable API** for custom backend
implementors — documented as stable in v2.0, accessed via
`from llenergymeasure.core import EnergyBackend`. These are an exception to the
"internal only" rule for `core/` because custom backends are a first-class use case.

### Consequences

Positive: Internal modules remain free to refactor without version ceremonies.
Negative / Trade-offs: Users who discover and use internal modules are surprised when they break.
Neutral: Requires clear documentation of the stable/internal boundary in the API reference.

---

## L2 — Deprecation Policy

### Considered Options

| Option | Pros | Cons |
|--------|------|------|
| **One minor version deprecation window — Chosen** | Keeps API surface clean; users are developers who update frequently; avoids dead code accumulation | Aggressive for users who update infrequently (mitigated: target audience is researchers/developers) |
| Two minor version window | More time to migrate | Accumulates dead code; confuses newcomers about the intended API |
| No deprecation (just remove in major) | Maximum cleanliness | Breaking changes in minor versions would be unexpected and violate SemVer |

### Decision

A deprecated feature raises `DeprecationWarning` in the release that deprecates it and is
**removed in the next minor version**.

```
v2.0.0   Feature X introduced
v2.1.0   Feature X deprecated → DeprecationWarning on use, documented in CHANGELOG
v2.2.0   Feature X removed → ImportError / AttributeError
```

Rationale: v2.x is a research tool — users are developers who update frequently. Long
deprecation windows accumulate dead code and confuse newcomers about the intended API.
Major version boundaries (v2 → v3) are breaking by default — no deprecation ceremony
needed. If a feature needs to stay longer due to user feedback, extend on a case-by-case
basis.

**Deprecation warning format:**
```python
import warnings

warnings.warn(
    "run_campaign() is deprecated as of v2.1 and will be removed in v2.2. "
    "Use run_study() instead.",
    DeprecationWarning,
    stacklevel=2,
)
```

**CHANGELOG format for deprecations:**
```
## v2.1.0

### Deprecated (removed in v2.2.0)
- `run_campaign()` → use `run_study()` instead
- `ExperimentConfig.model_name` field → renamed to `ExperimentConfig.model`
```

### Consequences

Positive: API surface remains clean; deprecated code removed within two releases.
Negative / Trade-offs: Users who skip minor versions may encounter removals without seeing the deprecation warning.
Neutral: Each extension of the deprecation window requires an explicit case-by-case decision.

---

## L3 — Pre-v2.0 Stability

### Decision

**Pre-v2.0.0 code has no stability guarantee.** The current codebase on the
`planning/strategic-reset` branch is pre-v2.0. Any class, function, or field may change
or disappear without notice or deprecation warning.

This is documented in:
- `README.md`: "v2.0.0 is the stability baseline. Pre-release versions may change without notice."
- `pyproject.toml`: Version pinned to `2.0.0.dev0` until release

**The v2.0.0 release is the commitment point.** After that tag, the stability contract applies.

### Consequences

Positive: Development speed is unconstrained before v2.0.
Negative / Trade-offs: Early adopters of pre-release versions may encounter breaking changes.
Neutral: Pre-release versions must be clearly marked in `pyproject.toml` and README.

---

## SemVer Interpretation for this Project

| Version type | Meaning |
|---|---|
| Patch (`2.0.0` → `2.0.1`) | Bug fixes only. No new features. No API changes. |
| Minor (`2.0.x` → `2.1.0`) | New features, additive API changes. Deprecations with one-version warning. |
| Major (`2.x.x` → `3.0.0`) | Breaking changes. v3.0 = lm-eval integration — expected to change metric APIs. |

**Additive changes in minor versions (always safe):**
- New optional parameters with defaults (e.g., `run_study(..., output_dir=...)`)
- New fields on result models with defaults (e.g., `ExperimentResult.baseline_power_w`)
- New exception subclasses under `LLEMError`
- New optional extras (`[zeus]`, `[codecarbon]`)

**Breaking changes require major version:**
- Removing or renaming exported names
- Changing required parameters
- Changing field types or removing fields from Pydantic models
- Changing exception hierarchy (though new subclasses are additive)

---

## Related

- [versioning-roadmap.md](versioning-roadmap.md): Version sequence and scope
- [../designs/library-api.md](../designs/library-api.md): Complete public API surface definition
- [error-handling.md](error-handling.md): Exception hierarchy (stable from v2.0)
- [release-process.md](release-process.md): How releases are cut and tagged
