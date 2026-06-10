# engines/ - Inference Engine Plugins

Thin inference engine plugins implementing the `EnginePlugin` protocol. Layer 2 in the six-layer architecture.

## Purpose

Each engine owns only inference: load model, run warmup, run inference, clean up. The `MeasurementHarness` (layer 3) owns everything else: energy tracking, FLOPs estimation, CUDA sync, result assembly.

## Layout

Z-engines structure: each engine is a sub-package owning its plugin code AND
its runtime data (validation invariants + discovered schema). SOTA Python
pattern; mirrors sphinx (`themes/`), spaCy (`languages/`), scikit-learn
(`datasets/`).

```
engines/
  __init__.py          # get_engine(name) factory, detect_default_engine()
  protocol.py          # EnginePlugin protocol + InferenceOutput dataclass
  probe_adapter.py     # Adapter helpers
  _observed.py         # Observed-runtime-data extraction (effective params, metrics)
  _cuda.py             # CUDA memory and warmup helpers
  _errors.py           # OOM and import error helpers

  transformers/
    __init__.py        # Re-exports TransformersEngine from plugin
    plugin.py          # HuggingFace Transformers engine
    invariants.proposed.yaml    # Mined proposed invariants
    invariants.validated.yaml   # Validation-confirmed observations
    schema.discovered.json      # Introspected parameter schema
    _staging/          # Per-engine miner staging (gitkeep'd)

  vllm/                # Same shape: vLLM engine (Docker-only)
  tensorrt/            # Same shape: TensorRT-LLM engine (NGC Docker)
```

The runtime data files (`invariants.*.yaml`, `schema.discovered.json`) are
loaded by `llenergymeasure.config.engine_invariants.EngineInvariantsLoader` and
`llenergymeasure.config.SchemaLoader` respectively. CI workflows
(`update-engine-{invariants,schemas}.yml`) regenerate them via Renovate-driven
library bumps and commit-back. See `INVARIANTS_README.md` for the corpus
schema specification.

## EnginePlugin protocol

```python
from llenergymeasure.engines.protocol import EnginePlugin, InferenceOutput

class MyEngine:
    @property
    def name(self) -> str: ...

    def load_model(self, config: ExperimentConfig) -> Any: ...
    def warmup(self, config: ExperimentConfig, model: Any) -> WarmupResult: ...
    def run_inference(self, config: ExperimentConfig, model: Any) -> InferenceOutput: ...
    def cleanup(self, model: Any) -> None: ...
    def check_hardware(self, config: ExperimentConfig) -> list[str]: ...
```

`check_hardware()` returns a list of error strings (empty means compatible). Called via `engines.probe_adapter.build_config_probe()` at preflight to catch host-GPU mismatches (e.g., FP8 on A100, SM below the engine's floor). Framework-invariant validation (library-semantics) lives in the validated corpus consumed by `ExperimentConfig._apply_invariants`.

`InferenceOutput` carries the minimal data the harness needs:

```python
InferenceOutput(
    elapsed_time_sec=...,  # engine-measured (overridden by harness perf_counter)
    input_tokens=512,
    output_tokens=256,
    peak_memory_mb=14000.0,
    model_memory_mb=12000.0,
    batch_times=[...],
    extras={"hf_model": model},  # optional, e.g. for FLOPs estimation
)
```

## Engine factory

```python
from llenergymeasure.engines import get_engine, detect_default_engine

engine = get_engine("pytorch")   # PyTorchEngine
engine = get_engine("vllm")      # VLLMEngine
engine = get_engine("tensorrt")  # TensorRTEngine

default = detect_default_engine()  # "pytorch" if transformers installed, etc.
```

Priority for auto-detection: pytorch > tensorrt > vllm.

## Engine code runs in Docker

Engines have no host install path. Each engine runs inside its own Docker
image, built from the SSOT in `engine_versions/{engine}.yaml`:

| Engine | Required package | Image source |
|---------|-----------------|---------------|
| `transformers` | `transformers` | `docker/Dockerfile.transformers` (first-party) |
| `vllm` | `vllm` | `vllm/vllm-openai:<version>` (upstream; project source bind-mounted at run time) |
| `tensorrt` | `tensorrt_llm` | `nvcr.io/nvidia/tensorrt-llm/release:<version>` (NGC upstream; project source bind-mounted at run time) |

See [docs/development.md](../../../docs/development.md) for the build/run
pattern. Host imports of these libraries fail by design.

## Layer constraints

- Layer 2 - may import from layers 0-1 only
- Can import from: `config/`, `domain/`, `device/`, `utils/`, `energy/`, `datasets/`, `infra/`
- Cannot import from: `harness/`, `study/`, `api/`, `cli/`, `results/`

## Related

- See `../harness/` for the measurement lifecycle that drives these engines
- See `../config/README.md` for `TransformersConfig`, `VLLMConfig`, `TensorRTConfig`
- See `../api/preflight.py` for engine pre-flight checks
