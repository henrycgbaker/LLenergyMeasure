# utils/ - Cross-cutting Utilities

Shared foundation helpers. Layer 0 in the six-layer architecture.

## Purpose

Provides the foundation all other layers import: the exception hierarchy,
environment-variable helpers, formatting/IO utilities, and Python-version shims.

## Modules

| Module | Description |
|--------|-------------|
| `exceptions.py` | Exception hierarchy rooted at `LLEMError` |
| `env_config.py` | Canonical `LLEM_*` env-var constants and their reader helpers |
| `security.py` | `trust_remote_code_enabled()` env-var gate for HuggingFace `trust_remote_code` |
| `formatting.py` | Number / name / byte formatting helpers |
| `io.py` | Filesystem IO helpers (`load_json`) |
| `compat.py` | Python-version compatibility shims (e.g. `StrEnum` backport) |
| `__init__.py` | Package marker |

## exceptions.py

```python
from llenergymeasure.utils.exceptions import (
    LLEMError,           # base
    ConfigError,         # invalid or missing config
    EngineError,        # inference engine failures
    PreFlightError,      # pre-flight check failures
    ExperimentError,     # experiment execution errors
    StudyError,          # study orchestration errors
    DockerError,         # Docker container dispatch
    DockerPreFlightError,  # Docker pre-flight check (inherits PreFlightError)
)
```

`DockerError` carries structured fields: `fix_suggestion` and `stderr_snippet` for actionable error messages.

## security.py

```python
from llenergymeasure.utils.security import trust_remote_code_enabled

# Opt-in for HuggingFace trust_remote_code via LLEM_TRUST_REMOTE_CODE.
# Unset / 0 / false means the HF default (False).
if trust_remote_code_enabled():
    ...
```

## Layer constraints

- Layer 0 - base layer; no imports from other llenergymeasure layers
- Can be imported by all layers above
- Do not add logic here that belongs in a higher layer (domain, config, etc.)
