# N-X14: Backend Detection

**Module**: `src/llenergymeasure/config/backend_detection.py`
**Risk Level**: LOW
**Decision**: Keep — v2.0
**Planning Gap**: The `llem config` command design in `decisions/cli-ux.md` and `designs/observability.md` shows backend detection output ("pytorch: installed", "vllm: not installed → pip install...") but does not reference `backend_detection.py` or specify that `get_available_backends()` is the function powering it.

---

## What Exists in the Code

**Primary file(s)**: `src/llenergymeasure/config/backend_detection.py`
**Key classes/functions**:

- `KNOWN_BACKENDS: list[str] = ["pytorch", "vllm", "tensorrt"]` (line 5) — ordered list of all supported backends

- `is_backend_available(backend)` (line 8) — per-backend availability check:
  - `pytorch`: `import torch`
  - `vllm`: `import vllm`
  - `tensorrt`: `import tensorrt_llm`
  - Returns `False` on `ImportError`, `OSError` (for TensorRT-LLM which has native library deps), or any other exception
  - The broad exception catch (`Exception`) on line 29 is intentional — TensorRT-LLM can raise non-standard errors during import if CUDA libraries are missing

- `get_available_backends()` (line 35) — `[b for b in KNOWN_BACKENDS if is_backend_available(b)]`; returns list preserving `KNOWN_BACKENDS` order

- `get_backend_install_hint(backend)` (line 44) — returns human-readable install instruction:
  - `pytorch`: `"pip install llenergymeasure"`
  - `vllm`: `"Docker recommended — see docs/deployment.md"`
  - `tensorrt`: `"Docker recommended — see docs/deployment.md"`
  - Default: `f"pip install llenergymeasure[{backend}]"`

Total: 59 lines.

The vLLM and TensorRT hints correctly reflect the v2.0 deployment model (Docker-first for those backends) — these are not stale.

## Why It Matters

`get_available_backends()` is the system-level query that drives the `llem config` output and the zero-config interactive backend selector. Without it, `llem config` would need inline import-checking logic, and `llem run --model X` (with no backend specified) would have no way to determine which backends are available to offer as choices. The `OSError` catch in `is_backend_available()` is non-obvious but critical — TensorRT-LLM fails at import time with an `OSError` (not `ImportError`) when CUDA shared libraries are missing, which happens on systems where the package is installed but CUDA is not.

The `get_backend_install_hint()` function encodes the deployment model in a single place: if the deployment recommendation changes (e.g., pip-installable vLLM becomes standard), this function is the only place to update.

## Planning Gap Details

`decisions/cli-ux.md` `llem config` example output (lines 79–88) shows:
```
vllm       not installed → pip install llenergymeasure[vllm]
```
But the actual hint returned by `get_backend_install_hint("vllm")` is "Docker recommended — see docs/deployment.md" — not a pip install hint. The planning doc example and the code disagree on the install message for vLLM and TensorRT.

This needs to be reconciled: either update the planning doc to match the Docker-first message, or update the function to match the planning doc's pip-install message.

`designs/observability.md` `llem config` output (lines 207–221) shows:
```
vllm       not installed → pip install llenergymeasure[vllm]
tensorrt   not installed → pip install llenergymeasure[tensorrt]
```
Again, uses pip install hints — not Docker hints. The current code returns Docker hints. The planning docs appear to have been written before the Docker-first deployment decision was made (or the decision was made after the observability design was written).

## Recommendation for Phase 5

Carry `backend_detection.py` forward unchanged into `config/backend_detection.py`.

Resolve the discrepancy between planning docs and code on the install hints. Two options:

**Option A**: Update `get_backend_install_hint()` to match planning docs (pip install hints). This aligns CLI output with what the docs say. But it may confuse users who try `pip install llenergymeasure[vllm]` and find it doesn't install a working vLLM (Docker is needed).

**Option B**: Update planning docs to match code (Docker hints). More technically accurate for vLLM/TensorRT.

Recommendation: **Option B** — the Docker-first message is more accurate. The `llem config` output should say `"Docker recommended → docker pull ..."` not `"pip install llenergymeasure[vllm]"`. Update `designs/observability.md` and `decisions/cli-ux.md` example outputs to match.

Also: `KNOWN_BACKENDS` order matters — the first available backend is used as the default in zero-config runs. Currently `["pytorch", "vllm", "tensorrt"]` means PyTorch is preferred. This ordering should be documented as a deliberate choice in `decisions/cli-ux.md`.
