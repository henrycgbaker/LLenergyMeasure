# Phase 4: PyTorch Backend and Pre-flight - Research

**Researched:** 2026-02-26
**Domain:** PyTorch/Transformers inference backend, CUDA environment detection, pre-flight validation
**Confidence:** HIGH (all findings verified from existing codebase + requirements)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Pre-flight checks:**
- Flat list of all failures collected into a single `PreFlightError` (not grouped by category)
- Boundary: Pydantic handles schema validation (types, enums, missing fields); pre-flight handles runtime checks (backend installed? model accessible? CUDA available?)
- Essential checks only: backend installed, model accessible (HF hub reachable, gated model token), CUDA available. No VRAM estimation — unreliable across quantisation/batch sizes
- Always runs before every experiment — no opt-out, no `--skip-preflight`
- Always raises `PreFlightError` on failure — no warn mode. Library users wrap in `try/except PreFlightError`

**Environment snapshot:**
- Full `pip freeze` output — the entire environment, not filtered to llem deps
- If conda is detected, also include `conda list` output
- Captured before inference starts (before model loading) — this is the starting state, not runtime state
- GPU memory usage is a measurement result, not an environment property

**PyTorch runner shape:**
- Rewrite from scratch using v1.x code as reference only — not an incremental adaptation
- Direct `ExperimentConfig` acceptance: `run(config: ExperimentConfig) -> ExperimentResult`
- Shared Protocol/ABC defines the contract for all backends (PyTorch, vLLM, TRT-LLM)
- Runner reads `config.model` (shared) + `config.pytorch.*` (backend-specific) directly — no adapter layer
- Matches peer pattern: lm-eval `LM` subclass, Optimum-Benchmark direct config

**Error messages:**
- Generic but helpful fix suggestions: "CUDA out of memory. Try: reduce batch_size, use precision=fp16, or use a smaller model"
- No VRAM calculations in error messages — estimates are unreliable
- Text instructions only for model access errors — no URLs (terminal inconsistency)
- No partial results on failure — a failed experiment's measurements are invalid
- `BackendError` for inference runtime failures (CUDA errors, model load, inference crashes)
- `ExperimentError` wraps `BackendError` with study context (which config, which iteration)

### Claude's Discretion

- Whether to record CUDA version detection source alongside the resolved version
- model_kwargs regression test (likely unnecessary given rewrite)
- Exact pre-flight check ordering and parallelism
- Loading skeleton / progress display during model download

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| CM-01 | PyTorch inference backend (local) | Rewrite `core/backends/pytorch.py` using `transformers.AutoModelForCausalLM` + new `ExperimentConfig` contract |
| CM-04 | `InferenceBackend` Protocol in `core/backends/protocol.py` | Define Protocol with `run(config: ExperimentConfig) -> ExperimentResult`; new module path from architecture.md |
| CM-05 | Backend default: `pytorch` when multiple installed | Detection logic: try-import each backend, return first available; hardcode `pytorch` as baseline |
| CM-06 | P0 fix: PyTorch `model_kwargs` bug (L375) | Root cause confirmed: `_build_model_kwargs()` builds kwargs but `loader.load(config)` ignores them. Fix: pass kwargs directly in rewritten backend |
| CM-29 | Pre-flight checks: GPU available, backend installed, dataset accessible | Collect all failures into `failures: list[str]`, raise single `PreFlightError` |
| CM-30 | Pre-flight failure → `PreFlightError`. All failures reported at once | `PreFlightError` already in `exceptions.py`; pre-flight collects list then raises once |
| CM-31 | GPU persistence mode: pre-flight warning (not blocking error) | NVML `nvmlDeviceGetPersistenceMode()` → emit warning via stdlib `logging.warning()` |
| CM-32 | `EnvironmentSnapshot` auto-captured at experiment start: Python version, CUDA version, driver, GPU names/VRAM, pip freeze, tool version | New `EnvironmentSnapshot` model + `collect_environment_snapshot()` function; extends existing `EnvironmentMetadata` pattern |
| CM-33 | CUDA version: multi-source detection (torch → version.txt → nvcc → None) | Verified detection chain against existing `core/environment.py` which uses NVML only; must add torch/version.txt/nvcc sources |
| CM-34 | Thermal throttle detection (carry-forward from v1.x) | `core/power_thermal.py` exists in v1.x — carry forward and adapt to new contract |
</phase_requirements>

---

## Summary

Phase 4 implements the core measurement pipeline's entry point: a rewritten PyTorch inference backend that accepts the v2.0 `ExperimentConfig` directly, pre-flight validation that catches all configuration errors before GPU allocation, and an `EnvironmentSnapshot` that fully characterises the measurement context.

The phase is primarily a **rewrite, not an extension**. The v1.x `PyTorchBackend` class (`core/inference_backends/pytorch.py`, ~1000 lines) is heavily coupled to v1.x types (`config.model_name`, `config.fp_precision`, `loader.load(config)` without kwargs, Accelerate `BackendRuntime`), and contains the P0 `model_kwargs` bug at L375 where kwargs are built but silently dropped. The decision to rewrite from scratch rather than adapt is correct — the v1.x class is not structurally compatible with the v2.0 `ExperimentConfig` contract.

The target architecture from `designs/architecture.md` places the new backend at `core/backends/pytorch.py` (note: different path from v1.x `core/inference_backends/pytorch.py`) and the protocol at `core/backends/protocol.py`. `EnvironmentSnapshot` belongs in `domain/environment.py` (augmenting the existing `EnvironmentMetadata` model), and pre-flight logic belongs in `orchestration/preflight.py`. The `_run()` stub in `_api.py` is the implementation target for this phase — it currently raises `NotImplementedError`.

**Primary recommendation:** Write `orchestration/preflight.py` first (pure logic, no GPU), then `domain/environment.py` additions, then `core/backends/protocol.py`, then `core/backends/pytorch.py`, and finally wire `_run()` in `_api.py`. This order enables incremental testing without a GPU.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `transformers` | `>=4.40` | `AutoModelForCausalLM.from_pretrained()`, `AutoTokenizer`, generation | The v2.0 `[pytorch]` extra; already in pyproject.toml |
| `torch` | `>=2.1` | `torch.cuda.is_available()`, `torch.inference_mode()`, precision dtypes | Part of `[pytorch]` extra |
| `pynvml` (nvidia-ml-py) | `>=11.0` | CUDA version, driver version, persistence mode, GPU name | Already a base dependency (INF-02) |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `subprocess` (stdlib) | — | `nvcc --version`, `pip freeze`, `conda list` | CUDA version fallback; pip/conda environment capture |
| `sys` (stdlib) | — | Python version, platform | Always — `sys.version_info`, `platform.python_version()` |
| `importlib.util` | — | `importlib.util.find_spec("torch")` for backend availability check | Pre-flight backend-installed check |
| `huggingface_hub` | — | `HfApi().model_info()` for gated model access check | Pre-flight model-accessible check |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `importlib.util.find_spec()` | `try: import torch` | `find_spec` does not import the module; avoids heavy torch init during pre-flight when torch is absent |
| `subprocess pip freeze` | `importlib.metadata.packages_distributions()` | `pip freeze` output is what researchers actually paste; the stdlib approach loses editable install markers and extras |
| `HfApi().model_info()` for access check | `requests.get(hub_url)` | HfApi handles auth headers, redirect, and 401/403 correctly |

---

## Architecture Patterns

### Recommended Module Structure (v2.0 target)

```
src/llenergymeasure/
├── core/
│   └── backends/              ← NEW module path (from designs/architecture.md)
│       ├── __init__.py
│       ├── protocol.py        ← InferenceBackend Protocol (CM-04)
│       └── pytorch.py         ← Rewritten PyTorch backend (CM-01, CM-06)
├── domain/
│   └── environment.py         ← AUGMENT: add EnvironmentSnapshot (CM-32)
└── orchestration/
    └── preflight.py           ← NEW: pre-flight validation (CM-29, CM-30, CM-31)
```

**Note:** The v1.x code lives at `core/inference_backends/pytorch.py`. The v2.0 target path is `core/backends/pytorch.py`. These are different directories — the rewrite creates the new module; the old module is not modified in this phase (it will be deleted later when v1.x dead code is purged).

### Pattern 1: Protocol-First Backend Contract (CM-04)

**What:** A `typing.Protocol` with `runtime_checkable` that defines the single method backends must implement.

**When to use:** Enables duck-typing checks (`isinstance(backend, InferenceBackend)`) without inheritance.

```python
# src/llenergymeasure/core/backends/protocol.py
from typing import Protocol, runtime_checkable
from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.domain.experiment import ExperimentResult

@runtime_checkable
class InferenceBackend(Protocol):
    """Contract for all inference backends."""

    @property
    def name(self) -> str: ...

    def run(self, config: ExperimentConfig) -> ExperimentResult: ...
```

**Rationale:** The v1.x `InferenceBackend` Protocol in `core/inference_backends/protocols.py` uses `initialize(config, runtime)` + `run_inference(prompts, config)` — a two-step API that requires `BackendRuntime` (an Accelerate abstraction). The v2.0 design simplifies to a single `run(config)` method. The backend owns its entire lifecycle internally, matching lm-eval's `LM` subclass pattern where the subclass takes config at `__init__` and exposes a single callable interface.

### Pattern 2: Collect-All Pre-flight (CM-29, CM-30)

**What:** All checks run regardless of prior failures; a single `PreFlightError` is raised with the complete failure list.

**When to use:** Always — the CONTEXT.md decision is unambiguous.

```python
# src/llenergymeasure/orchestration/preflight.py
from llenergymeasure.exceptions import PreFlightError

def run_preflight(config: ExperimentConfig) -> None:
    """Run all pre-flight checks. Raises PreFlightError listing ALL failures."""
    failures: list[str] = []

    if not _check_cuda_available():
        failures.append("CUDA not available → install CUDA or use a GPU machine")

    if not _check_backend_installed(config.backend):
        failures.append(f"{config.backend} not installed → pip install llenergymeasure[{config.backend}]")

    model_issue = _check_model_accessible(config.model)
    if model_issue:
        failures.append(model_issue)

    # Warnings (non-blocking)
    _check_persistence_mode_warning()

    if failures:
        raise PreFlightError(
            f"Pre-flight failed: {len(failures)} issue(s) found\n"
            + "\n".join(f"  ✗ {f}" for f in failures)
        )
```

**Pattern note:** CONTEXT.md specifies the display format exactly:
```
Pre-flight failed: 2 issues found
  ✗ vllm      not installed → pip install llenergymeasure[vllm]
  ✗ Llama-3-70B  gated model — no HF_TOKEN → export HF_TOKEN=<your_token>
```

### Pattern 3: Multi-Source CUDA Version Detection (CM-33)

**What:** Try four sources in order; return the first that succeeds.

```python
def detect_cuda_version() -> str | None:
    """Detect CUDA version via multi-source fallback."""
    # Source 1: torch.version.cuda (most reliable when torch is installed)
    try:
        import torch
        if torch.version.cuda:
            return torch.version.cuda
    except ImportError:
        pass

    # Source 2: /usr/local/cuda/version.txt or version.json
    for path in ["/usr/local/cuda/version.txt", "/usr/local/cuda/version.json"]:
        version = _parse_cuda_version_file(path)
        if version:
            return version

    # Source 3: nvcc --version
    try:
        import subprocess
        result = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return _parse_nvcc_output(result.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Source 4: None — CUDA not found
    return None
```

**Note on existing code:** The v1.x `core/environment.py` uses `pynvml.nvmlSystemGetCudaDriverVersion()` to get the CUDA *driver* version (what the kernel supports). `torch.version.cuda` returns the CUDA *runtime* version (what PyTorch was compiled against). These are different values. CM-33 specifies `torch → version.txt → nvcc → None` — these all detect the runtime/installed toolkit version, not the driver maximum. The existing NVML approach detects a different thing and should be kept separately as `cuda_driver_max_version`.

### Pattern 4: EnvironmentSnapshot vs EnvironmentMetadata

**What:** The existing `EnvironmentMetadata` in `domain/environment.py` captures GPU/CUDA/thermal/CPU/container info via NVML. CM-32 requires additional fields: Python version, pip freeze, tool version. The design choice is whether to extend `EnvironmentMetadata` or create a new `EnvironmentSnapshot` model.

**Recommendation:** Create `EnvironmentSnapshot` as a new top-level model that *embeds* `EnvironmentMetadata`:

```python
# domain/environment.py (additions)

class EnvironmentSnapshot(BaseModel):
    """Full environment snapshot for experiment reproducibility (CM-32)."""

    # Hardware + CUDA (existing EnvironmentMetadata)
    hardware: EnvironmentMetadata

    # Software stack
    python_version: str                    # platform.python_version() → "3.11.5"
    pip_freeze: str                        # subprocess pip freeze output
    conda_list: str | None = None          # subprocess conda list output (if conda detected)
    tool_version: str                      # llenergymeasure.__version__

    # CUDA detection provenance (Claude's discretion — recommended: YES)
    cuda_version: str | None              # resolved CUDA toolkit version
    cuda_version_source: str | None       # "torch" | "version.txt" | "nvcc" | None
```

**Rationale for `cuda_version_source`:** The four detection sources can return different values (torch compiled against CUDA 12.1, system has CUDA 12.4). Recording the source makes debugging environment mismatches tractable. Overhead is zero (one string field). Recommended: include it.

**Integration point:** `ExperimentResult` in `domain/experiment.py` currently has `environment: EnvironmentMetadata | None`. This field should be updated to `environment_snapshot: EnvironmentSnapshot | None` in Phase 4, or kept as is with a separate `environment_snapshot` field. Since the v2.0 `ExperimentResult` schema (Phase 6) will be the final arbiter of field names, Phase 4 can use `environment_snapshot: EnvironmentSnapshot | None` as the new field, leaving the existing `environment: EnvironmentMetadata | None` as a v1.x relic (it is `None` for all existing Phase 3 stubs anyway).

### Pattern 5: Rewritten PyTorch Backend Shape

**What:** A clean, self-contained class with `__init__(self)` + `run(self, config: ExperimentConfig) -> ExperimentResult`.

```python
# src/llenergymeasure/core/backends/pytorch.py

class PyTorchBackend:
    """PyTorch/Transformers inference backend for LLM efficiency measurement."""

    name = "pytorch"

    def run(self, config: ExperimentConfig) -> ExperimentResult:
        """Run a complete inference experiment: pre-flight + snapshot + inference + result."""
        # 1. Environment snapshot (before model loading)
        snapshot = collect_environment_snapshot()

        # 2. Model load
        model, tokenizer = self._load_model(config)

        # 3. Warmup
        self._run_warmup(model, tokenizer, config)

        # 4. Measurement
        result = self._run_measurement(model, tokenizer, config)

        # 5. Cleanup
        self._cleanup(model)

        return self._build_result(config, result, snapshot)

    def _load_model(self, config: ExperimentConfig):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        kwargs = self._model_load_kwargs(config)
        tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(config.model, **kwargs)
        return model, tokenizer

    def _model_load_kwargs(self, config: ExperimentConfig) -> dict:
        """Build AutoModelForCausalLM.from_pretrained() kwargs from ExperimentConfig.

        The P0 fix: kwargs are built AND passed to from_pretrained() here.
        No separate loader.load(config) call that drops the kwargs.
        """
        kwargs: dict = {
            "torch_dtype": _precision_to_dtype(config.precision),
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if config.pytorch:
            if config.pytorch.attn_implementation:
                kwargs["attn_implementation"] = config.pytorch.attn_implementation
            if config.pytorch.load_in_4bit:
                kwargs["load_in_4bit"] = True
            if config.pytorch.load_in_8bit:
                kwargs["load_in_8bit"] = True
        # passthrough_kwargs passed through directly (CM-06 fix: these are NOT dropped)
        if config.passthrough_kwargs:
            kwargs.update(config.passthrough_kwargs)
        return kwargs
```

**P0 bug root cause:** In v1.x, `_build_model_kwargs()` at L131 builds the dict, but `initialize()` at L375 calls `loader.load(config)` without passing the kwargs — there is a `TODO` comment confirming this. The rewrite eliminates the intermediate `loader.load()` call entirely; kwargs are built inline and passed directly to `AutoModelForCausalLM.from_pretrained()`.

### Anti-Patterns to Avoid

- **Separate loader class:** The v1.x `HuggingFaceModelLoader` abstraction is what caused the P0 bug — it accepted `config` but not extra kwargs. The rewrite calls `from_pretrained()` directly.
- **Accelerate BackendRuntime coupling:** The v1.x backend requires an `Accelerator` instance in `BackendRuntime`. The v2.0 single-experiment in-process execution (STU-05) does not need Accelerate. Use `device_map="auto"` directly.
- **Suppress-and-continue on pre-flight:** All pre-flight failures must be collected and raised, not logged and skipped.
- **CUDA version from NVML only:** NVML returns the *driver's* maximum supported CUDA version, not the runtime/toolkit version. CM-33 specifies `torch → version.txt → nvcc → None` chain.
- **Loguru in the new backend:** The v1.x codebase uses `loguru`. The STATE.md decision is "base package uses stdlib logging only; loguru is not a base dependency." New code in Phase 4 must use `import logging; logger = logging.getLogger(__name__)`.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Precision dtype mapping | Custom `if/elif` chain | `{"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[config.precision]` | One dict, no branching, same as existing v1.x `get_torch_dtype()` |
| Model availability on HuggingFace Hub | Custom HTTP requests | `huggingface_hub.HfApi().model_info(config.model)` | Handles auth, gating (401/403), private repos correctly |
| Backend installed check | `try: import torch; import transformers` | `importlib.util.find_spec("torch") is not None` | Does not trigger module import/init; fast in pre-flight |
| pip package list | Custom package inspection | `subprocess.run(["pip", "freeze"])` | Exact output researchers expect; includes extras markers |
| CUDA device count | Custom NVML iteration | `torch.cuda.device_count()` | Simpler when torch is installed (which it always is for pytorch backend) |

**Key insight:** Pre-flight runs before model loading and before any GPU allocation. It must be fast and avoid importing heavy modules. Use `importlib.util.find_spec()` for availability checks, not `try: import`.

---

## Common Pitfalls

### Pitfall 1: The P0 model_kwargs Bug Pattern
**What goes wrong:** Build a kwargs dict, then pass `config` to a loader that ignores the dict.
**Why it happens:** An intermediate abstraction (`HuggingFaceModelLoader`) that doesn't accept extra kwargs was introduced, and the call site had a `TODO` left unresolved.
**How to avoid:** In the rewrite, call `AutoModelForCausalLM.from_pretrained()` directly with all kwargs. No intermediate loader class.
**Warning signs:** Any `_build_*_kwargs()` method whose result is not verifiably passed to the underlying library call.

### Pitfall 2: CUDA Version Source Confusion
**What goes wrong:** `torch.version.cuda` returns "12.1" (compiled against); `nvmlSystemGetCudaDriverVersion()` returns 12040 (driver supports up to 12.4); `nvcc --version` returns "12.4" (installed toolkit). Three different answers.
**Why it happens:** CUDA has three distinct version concepts: driver API max, runtime (toolkit), and torch compiled-against.
**How to avoid:** CM-33 specifies `torch → version.txt → nvcc` chain, which all target the *runtime/toolkit* version. Document what is being measured. Keep `driver_version` from NVML as a separate field.
**Warning signs:** A single `cuda_version` field populated from NVML without noting it is the driver maximum.

### Pitfall 3: loguru in New Code
**What goes wrong:** New Phase 4 code imports `from loguru import logger` — violates the STATE.md decision that base package uses stdlib logging only.
**Why it happens:** v1.x codebase pervasively uses loguru; new code naturally copies the pattern.
**How to avoid:** New modules use `import logging; logger = logging.getLogger(__name__)`.
**Warning signs:** Any `from loguru import logger` in newly created files.

### Pitfall 4: `ExperimentResult` Field Name Mismatch
**What goes wrong:** Phase 4 populates `environment: EnvironmentMetadata` (v1.x field name) instead of the v2.0 `environment_snapshot: EnvironmentSnapshot`. When Phase 6 (Results) finalises the schema, there is a mismatch.
**Why it happens:** The existing `domain/experiment.py` has `environment: EnvironmentMetadata | None` from v1.x. The v2.0 schema uses `EnvironmentSnapshot`.
**How to avoid:** Phase 4 adds `environment_snapshot: EnvironmentSnapshot | None` as a new field alongside the existing `environment` field. The old field stays `None`; only the new field is populated. Phase 6 reconciles.

### Pitfall 5: Pre-flight Imports Triggering Model Load
**What goes wrong:** Importing `torch` at module level in `preflight.py` means the CUDA context initialises when pre-flight runs, defeating the purpose of checking CUDA availability before allocation.
**Why it happens:** Convenience — it is simpler to have `import torch` at the top.
**How to avoid:** Pre-flight checks that test torch availability must use `importlib.util.find_spec("torch")` at the check level. If torch is confirmed available, then `import torch` inside the check function is safe.

### Pitfall 6: `device_map="auto"` With Multi-GPU and No Accelerate
**What goes wrong:** `device_map="auto"` on a multi-GPU machine distributes model layers across GPUs without any measurement coordination — energy is measured per-GPU but results are aggregated inconsistently.
**Why it happens:** "auto" is the simplest setting and works for the common single-GPU case.
**How to avoid:** For Phase 4 (M1 single-experiment), `device_map="auto"` is correct — single GPU is the target. Note in code that multi-GPU via `device_map="auto"` is Phase 5+ territory (when `num_processes > 1` or `pytorch.num_processes > 1`).

---

## Code Examples

### Pre-flight with collect-all pattern

```python
# src/llenergymeasure/orchestration/preflight.py
import importlib.util
import logging

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.exceptions import PreFlightError

logger = logging.getLogger(__name__)


def run_preflight(config: ExperimentConfig) -> None:
    """Run all pre-flight checks. Raises PreFlightError listing ALL failures."""
    failures: list[str] = []

    # Check 1: CUDA available
    if not _check_cuda_available():
        failures.append("CUDA not available → install CUDA or use a GPU machine")

    # Check 2: Backend installed
    if not _check_backend_installed(config.backend):
        failures.append(
            f"{config.backend} not installed → "
            f"pip install llenergymeasure[{config.backend}]"
        )

    # Check 3: Model accessible
    model_issue = _check_model_accessible(config.model)
    if model_issue:
        failures.append(model_issue)

    # Non-blocking: GPU persistence mode warning
    _warn_if_persistence_mode_off()

    if failures:
        msg = f"Pre-flight failed: {len(failures)} issue(s) found\n"
        msg += "\n".join(f"  \u2717 {f}" for f in failures)
        raise PreFlightError(msg)


def _check_cuda_available() -> bool:
    """Check CUDA is available without importing torch at module level."""
    if importlib.util.find_spec("torch") is None:
        return False
    import torch
    return torch.cuda.is_available()


def _check_backend_installed(backend: str) -> bool:
    """Check the required backend package is installed."""
    package_map = {"pytorch": "transformers", "vllm": "vllm", "tensorrt": "tensorrt_llm"}
    package = package_map.get(backend, backend)
    return importlib.util.find_spec(package) is not None


def _check_model_accessible(model_id: str) -> str | None:
    """Check model is accessible on HuggingFace Hub. Returns error string or None."""
    if importlib.util.find_spec("huggingface_hub") is None:
        return None  # Can't check without huggingface_hub; skip
    try:
        from huggingface_hub import HfApi
        HfApi().model_info(model_id)
        return None
    except Exception as e:
        err_str = str(e)
        if "401" in err_str or "403" in err_str or "gated" in err_str.lower():
            return (
                f"{model_id} gated model — no HF_TOKEN "
                "→ export HF_TOKEN=<your_token>"
            )
        if "404" in err_str:
            return f"{model_id} not found on HuggingFace Hub"
        # Local path — skip hub check
        return None


def _warn_if_persistence_mode_off() -> None:
    """Warn (not error) if GPU persistence mode is not enabled."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mode = pynvml.nvmlDeviceGetPersistenceMode(handle)
        pynvml.nvmlShutdown()
        if mode == pynvml.NVML_FEATURE_DISABLED:
            logger.warning(
                "GPU persistence mode is off. "
                "First experiment may have higher latency. "
                "Enable: sudo nvidia-smi -pm 1"
            )
    except Exception:
        pass  # NVML unavailable or persistence mode query unsupported — not a blocker
```

### EnvironmentSnapshot collection

```python
# src/llenergymeasure/domain/environment.py (additions)
import logging
import platform
import subprocess
import sys

logger = logging.getLogger(__name__)


class EnvironmentSnapshot(BaseModel):
    """Full software+hardware environment snapshot (CM-32)."""

    # Hardware snapshot (existing model)
    hardware: EnvironmentMetadata

    # Software stack
    python_version: str
    pip_freeze: str
    conda_list: str | None = None
    tool_version: str

    # CUDA detection (CM-33)
    cuda_version: str | None
    cuda_version_source: str | None  # "torch" | "version_txt" | "nvcc" | None


def collect_environment_snapshot() -> EnvironmentSnapshot:
    """Capture full environment state before experiment starts."""
    from llenergymeasure import __version__
    from llenergymeasure.core.environment import collect_environment_metadata

    hardware = collect_environment_metadata()
    cuda_version, cuda_source = detect_cuda_version_with_source()

    return EnvironmentSnapshot(
        hardware=hardware,
        python_version=platform.python_version(),
        pip_freeze=_capture_pip_freeze(),
        conda_list=_capture_conda_list(),
        tool_version=__version__,
        cuda_version=cuda_version,
        cuda_version_source=cuda_source,
    )


def detect_cuda_version_with_source() -> tuple[str | None, str | None]:
    """Detect CUDA toolkit version. Returns (version, source) pair."""
    # Source 1: torch.version.cuda
    if importlib.util.find_spec("torch") is not None:
        import torch
        if torch.version.cuda:
            return torch.version.cuda, "torch"

    # Source 2: /usr/local/cuda/version.txt or version.json
    from pathlib import Path
    for path_str in ["/usr/local/cuda/version.txt", "/usr/local/cuda/version.json"]:
        p = Path(path_str)
        if p.exists():
            text = p.read_text()
            # "CUDA Version 12.4" or JSON {"cuda": "12.4"}
            import re
            m = re.search(r"(\d+\.\d+)", text)
            if m:
                return m.group(1), "version_txt"

    # Source 3: nvcc --version
    try:
        result = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            import re
            m = re.search(r"release (\d+\.\d+)", result.stdout)
            if m:
                return m.group(1), "nvcc"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return None, None


def _capture_pip_freeze() -> str:
    """Capture full pip freeze output."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True, timeout=30
        )
        return result.stdout if result.returncode == 0 else ""
    except Exception as e:
        logger.debug(f"pip freeze failed: {e}")
        return ""


def _capture_conda_list() -> str | None:
    """Capture conda list if conda is detected."""
    import shutil
    if shutil.which("conda") is None:
        return None
    try:
        result = subprocess.run(
            ["conda", "list"], capture_output=True, text=True, timeout=30
        )
        return result.stdout if result.returncode == 0 else None
    except Exception:
        return None
```

### _run() implementation skeleton

```python
# src/llenergymeasure/_api.py (Phase 4 replaces the NotImplementedError stub)

def _run(study: StudyConfig) -> StudyResult:
    """Internal runner — always receives StudyConfig, returns StudyResult."""
    from llenergymeasure.orchestration.preflight import run_preflight
    from llenergymeasure.core.backends.pytorch import PyTorchBackend

    results = []
    for config in study.experiments:
        run_preflight(config)
        backend = PyTorchBackend()
        result = backend.run(config)
        results.append(result)

    return StudyResult(experiments=results, name=study.name)
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `AutoModelForCausalLM.from_pretrained(..., low_cpu_mem_usage=True)` | `device_map="auto"` (implies accelerate dispatch) | transformers ~4.20 | `device_map="auto"` automatically enables `low_cpu_mem_usage` and handles multi-GPU; no need to set separately |
| `torch.half()` precision cast after load | `torch_dtype=torch.float16` in `from_pretrained` | transformers ~4.15 | Direct dtype in load avoids CPU→GPU copy at wrong precision |
| CUDA version from NVML driver API | `torch.version.cuda` as primary | Always separate | `torch.version.cuda` reflects what PyTorch can actually use; NVML driver version is the ceiling, not the actual runtime |
| BetterTransformer (`model.to_bettertransformer()`) | SDPA via `attn_implementation="sdpa"` | torch 2.0 / transformers 4.36 | BetterTransformer is deprecated in favour of native PyTorch SDPA; the v2.0 PyTorchConfig uses `attn_implementation` |

**Deprecated/outdated (v1.x-specific):**
- `HuggingFaceModelLoader` class: custom loader abstraction that caused the P0 bug — eliminated in rewrite
- `BackendRuntime` / Accelerate coupling in single-experiment path: not needed for in-process single-GPU (STU-05)
- `loguru` in new code: replaced with stdlib `logging` per STATE.md decision
- `config.model_name`: renamed to `config.model` in v2.0 `ExperimentConfig`
- `config.fp_precision`: renamed to `config.precision` in v2.0

---

## Codebase Integration Points

These are the exact files that Phase 4 touches or creates:

| Action | File | What Changes |
|--------|------|-------------|
| **Create** | `src/llenergymeasure/core/backends/__init__.py` | New module |
| **Create** | `src/llenergymeasure/core/backends/protocol.py` | `InferenceBackend` Protocol (CM-04) |
| **Create** | `src/llenergymeasure/core/backends/pytorch.py` | Rewritten PyTorch backend (CM-01, CM-06) |
| **Create** | `src/llenergymeasure/orchestration/preflight.py` | Pre-flight logic (CM-29, CM-30, CM-31) |
| **Augment** | `src/llenergymeasure/domain/environment.py` | Add `EnvironmentSnapshot`, `collect_environment_snapshot()` (CM-32, CM-33) |
| **Augment** | `src/llenergymeasure/domain/experiment.py` | Add `environment_snapshot: EnvironmentSnapshot | None` field to `ExperimentResult` |
| **Replace** | `src/llenergymeasure/_api.py` — `_run()` | Replace `NotImplementedError` stub with real implementation |
| **Carry-forward** | `src/llenergymeasure/core/power_thermal.py` | Thermal throttle detection — adapt to new contract (CM-34) |

**Do not touch:**
- `src/llenergymeasure/core/inference_backends/pytorch.py` (v1.x — dead code, leave for later purge)
- `src/llenergymeasure/config/models.py` (v2.0 `ExperimentConfig` — complete from Phase 2)
- `src/llenergymeasure/exceptions.py` (`PreFlightError` already defined)

---

## Open Questions

1. **Where does `_run()` call pre-flight?**
   - What we know: The CONTEXT.md says "Always runs before every experiment." The `_run()` function iterates over `study.experiments`.
   - What's unclear: Should pre-flight run once for the whole study (before any model loading) or once per experiment config?
   - Recommendation: Once per `ExperimentConfig` in the iteration — each config may differ (different model, different backend). This is consistent with "always runs before every experiment."

2. **`ExperimentResult` field for environment snapshot — `environment` vs `environment_snapshot`?**
   - What we know: Existing `domain/experiment.py` has `environment: EnvironmentMetadata | None` from v1.x. The v2.0 requirement uses "EnvironmentSnapshot" (CM-32). Phase 6 will finalise the ExperimentResult schema.
   - What's unclear: Whether to rename the field now or add a new field alongside.
   - Recommendation: Add new field `environment_snapshot: EnvironmentSnapshot | None = None` alongside the existing `environment` field. The existing field stays `None` in all Phase 4 outputs. Phase 6 reconciles by removing the old field and making the new one required.

3. **Model accessibility check for local paths**
   - What we know: `config.model` can be a HuggingFace Hub model ID (e.g. `"gpt2"`) or a local filesystem path (e.g. `"/models/llama-3.1-8b"`).
   - What's unclear: Should pre-flight check local paths exist?
   - Recommendation: If `config.model` starts with `/` or `./` or `~`, check `Path(config.model).exists()` instead of calling HfApi. Covers both cases.

4. **`pip freeze` timeout**
   - What we know: `pip freeze` on large environments can take 5–15 seconds.
   - Recommendation: 30-second timeout as shown in code example; log warning if it times out, return empty string rather than failing the experiment.

---

## Sources

### Primary (HIGH confidence)

- Existing codebase: `src/llenergymeasure/core/inference_backends/pytorch.py` — L375 P0 bug confirmed at `loader.load(config)` without kwargs pass-through
- Existing codebase: `src/llenergymeasure/exceptions.py` — `PreFlightError` confirmed defined
- Existing codebase: `src/llenergymeasure/config/models.py` — `ExperimentConfig`, `PyTorchConfig`, `passthrough_kwargs` all confirmed
- Existing codebase: `src/llenergymeasure/domain/environment.py` — `EnvironmentMetadata` confirmed; `EnvironmentSnapshot` does not yet exist
- Existing codebase: `src/llenergymeasure/_api.py` — `_run()` confirmed as `NotImplementedError` stub
- `.product/designs/architecture.md` — confirmed target module paths (`core/backends/`, `orchestration/preflight.py`)
- `.product/REQUIREMENTS.md` — CM-01 through CM-34 phase requirements confirmed
- `.planning/STATE.md` — stdlib logging decision confirmed; STU-05 in-process single-experiment confirmed

### Secondary (MEDIUM confidence)

- PyTorch `device_map="auto"` behaviour with `from_pretrained`: standard HuggingFace accelerate docs pattern — implicitly enables `low_cpu_mem_usage`
- `torch.version.cuda` vs NVML driver version distinction: well-documented PyTorch/CUDA distinction from community practice

### Tertiary (LOW confidence)

- HuggingFace Hub gated model response codes (401 vs 403): based on general HTTP convention and HF Hub API patterns — validate against `huggingface_hub` library error types in implementation

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already in pyproject.toml extras or base deps
- Architecture: HIGH — target paths confirmed from `designs/architecture.md`; P0 bug root cause confirmed from source inspection
- Pitfalls: HIGH — loguru/stdlib decision from STATE.md; P0 pattern from L375 inspection; CUDA version source distinction from code inspection
- Pre-flight pattern: HIGH — CONTEXT.md decisions are explicit; `PreFlightError` confirmed in exceptions.py

**Research date:** 2026-02-26
**Valid until:** 2026-03-26 (stable domain — extend if Phase 4 planning is delayed)
