# utils/ - Cross-cutting Utilities

Shared exceptions and security utilities. Layer 0 in the six-layer architecture.

## Purpose

Provides the foundation all other layers import: exception hierarchy and filesystem security helpers.

## Modules

| Module | Description |
|--------|-------------|
| `exceptions.py` | Exception hierarchy rooted at `LLEMError` |
| `security.py` | Path safety and experiment ID sanitisation |
| `formatting.py` | Number / name formatting helpers |
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
