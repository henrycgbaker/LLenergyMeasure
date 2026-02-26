# N-X08: Security — Path Sanitisation

**Module**: `src/llenergymeasure/security.py`
**Risk Level**: LOW
**Decision**: Keep — v2.0
**Planning Gap**: Not mentioned in any planning document. Security concerns around file path handling are an implementation-level detail, but the specific functions here are called directly by `StateManager` and any omission in the rebuild would introduce a path-traversal vulnerability in experiment state handling.

---

## What Exists in the Code

**Primary file(s)**: `src/llenergymeasure/security.py`
**Key classes/functions**:
- `validate_path(path, must_exist, allow_relative)` (line 9) — resolves path to absolute, optionally checks existence, optionally rejects relative paths; returns resolved `Path` or raises `ConfigurationError`
- `is_safe_path(base_dir, target_path)` (line 34) — checks `target_resolved.is_relative_to(base_resolved)` using Python 3.9+ `Path.is_relative_to()` (not string prefix — which would pass `/foo/bar_malicious` as a child of `/foo/bar`); handles `OSError`/`ValueError` by returning `False`; returns `bool`
- `sanitize_experiment_id(experiment_id)` (line 56) — replaces any char that is not alphanumeric, underscore, hyphen, or dot with `_`; raises `ConfigurationError` if empty; returns sanitised string
- `check_env_for_secrets(env_vars)` (line 81) — checks presence of env vars without exposing their values; returns `{var: bool}`

Total: 91 lines.

**Call sites in current codebase**:
- `state/experiment_state.py` lines 16, 298, 281: `StateManager` imports and uses both `is_safe_path` and `sanitize_experiment_id` for every state file read/write operation
- No other direct call sites found in the files read for this audit; the module is small and targeted

## Why It Matters

The `is_safe_path()` function prevents path-traversal attacks where a maliciously crafted experiment ID (e.g., `../../etc/passwd`) could cause state files to be written outside the `.state/` directory. The use of `Path.is_relative_to()` (not string prefix matching) is the correct implementation — a string prefix check on `/state/` would accept `/state_malicious/` as a valid path. The `sanitize_experiment_id()` function is the last defence before an experiment ID is used as a filename component — it converts any dangerous character to `_`.

These are not hypothetical concerns: experiment IDs can be influenced by model names, config names, or user-provided strings, any of which could contain path-separator characters. The mitigation is small (91 lines), correct, and load-bearing for `StateManager`.

## Planning Gap Details

No planning document references:
- `security.py`
- Path-traversal prevention
- Experiment ID sanitisation
- `check_env_for_secrets()`

The `designs/architecture.md` module layout does not include `security.py` as a module in the v2.0 structure (the layout shows `config/`, `core/`, `domain/`, `orchestration/`, `results/`, `state/`, `cli/`, `study/`, `datasets/`). This risks the module being omitted during the Phase 5 rebuild. The module should be explicitly listed at package root level alongside `protocols.py` and `resilience.py`.

## Recommendation for Phase 5

Carry `security.py` forward unchanged at `src/llenergymeasure/security.py`. The file is 91 lines and requires no changes for v2.0.

Ensure `StateManager` in the rebuilt `state/machine.py` imports and uses both:
```python
from llenergymeasure.security import is_safe_path, sanitize_experiment_id
```

The `validate_path()` function is not currently called in the files audited here — check whether it is used elsewhere in `config/loader.py` or `results/repository.py` to confirm it should be retained.

Add `security.py` explicitly to the module layout in `designs/architecture.md` as a root-level utility alongside `protocols.py`.
