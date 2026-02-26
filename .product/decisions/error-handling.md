# Error Handling: Exit Codes, Exception Hierarchy, CLI vs Library Surface

**Status:** Accepted
**Date decided:** 2026-02-19
**Last updated:** 2026-02-25
**Research:** N/A

## Decision

Exit codes: 0 (success), 1 (error), 2 (usage), 130 (interrupted). Exception hierarchy: `LLEMError` → `ConfigError`, `BackendError`, `PreFlightError`, `ExperimentError`, `StudyError`. Pydantic `ValidationError` passes through unchanged (not wrapped). CLI catches exceptions and maps to exit codes; library callers catch typed exceptions directly.

---

## Context

LLenergyMeasure has two distinct error surfaces: the CLI (`llem run`, `llem config`) and the
library API (Python code calling `run_experiment()` / `run_study()`). These surfaces have
different requirements. CLI callers inspect exit codes and stderr text. Library callers catch
exceptions programmatically and inspect `.errors()` or `.message`.

Three related questions required decisions:

- **K1**: What exit codes should the CLI emit?
- **K2**: What exception hierarchy should the library expose?
- **K3**: How should Pydantic `ValidationError` be handled — wrapped or passed through?

---

## K1 — CLI Exit Codes

### Considered Options

| Option | Pros | Cons |
|--------|------|------|
| **Two codes only: 0 / 1 (chosen)** | Matches lm-eval, every ML benchmark tool; shell scripts check `$? -ne 0`; simple contract | Cannot distinguish partial from total study failure by exit code alone |
| Granular codes: 0 / 1 / 3 / 4 | Allows scripts to branch on partial vs total failure | No ML benchmarking tool uses this; scripts should parse `study_summary.json` instead |
| Docker-style: 0 / 1 / 125 / 126 / 127 | Container-specific errors clearly signalled | Those extra codes are container-specific (no container = no process); not applicable to business logic |
| Cargo-style: 0 / 1 / 101 | Distinguishes panic from user error | `101` is Rust-specific (panic = crash); we have exception hierarchy instead |

### Decision

We will use four exit codes only:

| Code | Meaning | When |
|------|---------|------|
| `0` | Success | All experiments ran and completed |
| `1` | Error / user error | Bad config, missing backend, pre-flight failure, partial study failure, all experiments failed |
| `2` | Usage error | Wrong CLI flags / args (Typer handles this automatically) |
| `130` | Interrupted | User pressed Ctrl+C / SIGINT received |

Rationale: No ML benchmarking tool uses granular exit codes for partial failure — it is always
`0/1`. Adding codes `3` or `4` would create a bespoke contract with no industry precedent.
Shell scripts that need to distinguish partial failure should parse the `study_summary.json`
`failed_count` field, not the exit code.

**Partial study failure exits `1`** — some experiments ran, some failed. A clear stderr summary
is sufficient. Parsers read the output JSON, not the exit code, to determine how many experiments
succeeded.

### Consequences

Positive: Simple, conventional, compatible with any CI system or shell idiom.
Negative / Trade-offs: Scripts cannot branch on partial vs total failure by exit code alone — must
parse JSON output. This is the correct separation of concerns.
Neutral: `2` is emitted automatically by Typer for usage errors; no custom handling needed.

---

## K2 — Library Exception Hierarchy

### Considered Options

| Option | Pros | Cons |
|--------|------|------|
| **Custom hierarchy rooted at `LLEMError` (chosen)** | Broad-catch and specific-catch both possible; instructive messages enforced by convention; familiar pattern (httpx, SQLAlchemy) | Requires defining and maintaining exception classes |
| Plain Python exceptions (`ValueError`, `RuntimeError`) | Zero boilerplate | Cannot broad-catch all llem errors; callers must catch `Exception`; no semantic distinction |
| Single flat `LLEMError` only | Simple | Cannot distinguish `ConfigError` from `BackendError` programmatically |
| lm-eval pattern (no hierarchy, plain exceptions) | Simple | No structured catch; our tool is library-first, lm-eval is CLI-first |

### Decision

We will expose a custom exception hierarchy rooted at `LLEMError`. Enables broad-catch
(`except LLEMError`) or specific catch (`except ConfigError`) in user code. Critical for library
users who want to handle errors programmatically without catching `Exception`.

```python
# llenergymeasure/exceptions.py

class LLEMError(Exception):
    """Base exception for all LLenergyMeasure errors."""

class ConfigError(LLEMError):
    """Invalid or missing configuration."""
    # Raised when: YAML is malformed, required field missing, field value invalid

class BackendError(LLEMError):
    """Backend runtime error."""
    # Raised when: backend not installed, backend fails to load model, CUDA error

class PreFlightError(LLEMError):
    """Pre-flight validation failed."""
    # Raised when: model not accessible, HF_TOKEN missing, insufficient VRAM (--dry-run)

class ExperimentError(LLEMError):
    """Single experiment execution failed."""
    # Raised from StudyRunner when a subprocess experiment fails
    # Carries: config dict, exception type, error message, traceback
    config: dict
    cause: Exception

class StudyError(LLEMError):
    """Study-level error (not a single-experiment failure)."""
    # Raised when: study config invalid, no valid experiments generated,
    # runner unavailable (Docker not running), all experiments failed
    failed: list[ExperimentError]
    succeeded: int
```

Peer references:
- Pydantic: `ValidationError` with `.errors()` list — each error has `loc`, `msg`, `type`
- httpx: `HTTPError` → `TransportError` → `ConnectError` (hierarchy, all instructive)
- SQLAlchemy: `SQLAlchemyError` root, `IntegrityError`, `OperationalError` specialisations
- lm-eval: Uses plain Python exceptions without hierarchy (our approach is an improvement)

**All error messages must be instructive.** Every exception message should tell the user:
1. What went wrong (clearly stated)
2. How to fix it (exact command or action where possible)

Rationale: "vllm not found" is not instructive. "Install it with: pip install llenergymeasure[vllm]" is.

### Consequences

Positive: Library users can catch errors at the right granularity. Enforces a discipline of
instructive error messages.
Negative / Trade-offs: Exception classes must be maintained alongside new error conditions.
Neutral: Exception hierarchy is internal; the stable API surface is `__init__.py` exports.

---

## K3 — Pydantic `ValidationError` Passthrough

### Considered Options

| Option | Pros | Cons |
|--------|------|------|
| **Passthrough — let `ValidationError` propagate unchanged (chosen)** | Already informative (`.errors()` with loc/msg/type); familiar to Python users; CLI layer can format it | Library users receive a Pydantic type, not an `LLEMError` — breaks strict hierarchy |
| Wrap in `ConfigError` | Strict hierarchy — everything is an `LLEMError` | Loses Pydantic's structured `.errors()` list; wrapping degrades the error information |

### Decision

Pydantic `ValidationError` is not caught and re-raised as a `ConfigError`. It propagates
unchanged from `ExperimentConfig` and `StudyConfig` construction.

Rationale: Pydantic's `ValidationError` is already informative (`ValidationError.errors()` gives
field, message, type for each failure), familiar to Python users, and is the industry-standard
config validation error. The CLI layer catches `ValidationError` and formats it for terminal
output (coloured, with suggested fixes where applicable). Library users get the raw
`ValidationError` — they can inspect `.errors()` programmatically.

**Exception:** We raise `ConfigError` (not `ValidationError`) for non-Pydantic config errors:
- YAML parse errors (malformed YAML syntax, not schema errors)
- File not found errors (`config.yaml` path does not exist)
- Cross-config validation failures (e.g., `base:` reference points to non-existent file)

### Consequences

Positive: Library users get the richest possible error information from Pydantic.
Negative / Trade-offs: `ValidationError` is not a subclass of `LLEMError`, so broad-catch
`except LLEMError` will not catch Pydantic validation errors. This is a known and acceptable
asymmetry — documented in the library API.
Neutral: The CLI catches both `ValidationError` and `LLEMError` and formats both consistently.

---

## CLI Error Formatting

The CLI layer catches all errors and formats them consistently. Library users get raw exceptions.

```
# Pre-flight failure (PreFlightError):
Pre-flight failed: 2 issues found

  ✗ vllm      not installed → pip install llenergymeasure[vllm]
  ✗ Llama-3-70B  gated model — no HF_TOKEN → export HF_TOKEN=<your_token>

# Config error (ValidationError from Pydantic):
Config error in experiment.yaml:

  ✗ backend    'xpu' is not valid. Valid: ['pytorch', 'vllm', 'tensorrt']
  ✗ precision  'fp64' is not valid for pytorch. Valid: ['fp32', 'fp16', 'bf16']

# Study partial failure (StudyError):
Study complete: 9/12 experiments ran, 3 failed

  Failed:
  ✗ llama-3.1-8b pytorch fp32 batch=32   RuntimeError: CUDA out of memory
  ✗ llama-3.1-8b pytorch fp32 batch=16   RuntimeError: CUDA out of memory
  ✗ llama-3.1-8b tensorrt fp32           TensorRT: fp32 is not supported
  See results/batch-size-effects_2026-02-18T14-30/study_summary.json for details
```

**Principle:** Every CLI error output must include the exact command or action to resolve it,
where deterministic. "not installed" → install command. "gated model" → export command.
"OOM" → suggest reducing batch size. Vague messages like "backend error" are not acceptable.

---

## Related

- [cli-ux.md](cli-ux.md): Error message examples in pre-flight design
- [../designs/library-api.md](../designs/library-api.md): Library API surface and error expectations
- [../designs/study-yaml.md](../designs/study-yaml.md): Three-layer validation (L1/L2/L3) in study execution
