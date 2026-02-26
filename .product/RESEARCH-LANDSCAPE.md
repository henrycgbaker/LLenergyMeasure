# Industry Landscape Research

**Date**: 2026-02-17
**Phase**: 04.5 Strategic Reset
**Purpose**: Inform product vision and architecture decisions with industry evidence

---

## 1. Architecture Patterns

### Library-First is Universal

Every successful ML tool follows the same architectural pattern:

```
Python Library (core)
    |
    +--- CLI (thin wrapper)
    |
    +--- Server/API (HTTP layer on library)
    |
    +--- Web UI (bundled with server)
```

**Evidence:**
- **MLflow**: `mlflow` package is importable. `mlflow server` starts tracking UI. `mlflow run` is CLI.
- **W&B**: `wandb` package. `wandb.init()` starts tracking. CLI is thin wrapper.
- **ClearML**: `clearml` package. `clearml-init` configures. Server is separate deployment.
- **lm-eval-harness**: `lm_eval` package. `lm-eval` CLI wraps `lm_eval.simple_evaluate()`.
- **Zeus**: `zeus` package. `zeus.monitor.ZeusMonitor` class. No CLI at all.

**Implication for LLenergyMeasure**: Currently CLI-first — modules are importable but
there's no defined public API. Need to establish what `import llenergymeasure` exposes
as a clean, documented interface.

### URI-Based Store Routing (MLflow Pattern)

MLflow uses URI strings to transparently switch between local and remote storage:
- `mlflow.set_tracking_uri("./mlruns")` — local filesystem
- `mlflow.set_tracking_uri("http://server:5000")` — remote server
- `mlflow.set_tracking_uri("databricks")` — managed service

Same code, different storage backends. This is relevant for future central DB integration.

### Dependency Injection / Protocol Pattern

LLenergyMeasure already uses this well (ExperimentOrchestrator with protocol-based
components). This is ahead of many tools that use concrete class hierarchies.

---

## 2. Energy Measurement Tools

### Zeus (ML.ENERGY, Carnegie Mellon)

**Repository**: github.com/ml-energy/zeus
**Scope**: GPU energy measurement library
**Architecture**: Pure Python library, no CLI

**Key Design:**
```python
from zeus.monitor import ZeusMonitor
monitor = ZeusMonitor(gpu_indices=[0, 1, 2, 3])
monitor.begin_window("inference")
# ... run inference ...
result = monitor.end_window("inference")
print(f"Energy: {result.total_energy} J, Time: {result.time} s")
```

**Strengths over LLenergyMeasure:**
- Direct NVML access (not via CodeCarbon wrapper) — ~10ms overhead
- CPU + DRAM energy via Intel RAPL
- AMD GPU support (ROCm SMI)
- Apple Silicon support (powermetrics)
- NVIDIA Jetson support
- CO2 emissions calculation (electricity carbon intensity)
- Pareto frontier analysis (energy vs performance tradeoffs)

**Weaknesses vs LLenergyMeasure:**
- No LLM-specific metrics (no TTFT, ITL, streaming latency)
- No FLOPs estimation
- No inference backend orchestration
- No config system or experiment management
- No multi-backend comparison capability
- No campaign/grid search support

**ML.ENERGY Ecosystem:**
- Zeus (measurement library) → ML.ENERGY Benchmark (standardised tests) → ML.ENERGY Leaderboard (results display)
- Leaderboard runs ALL benchmarks on their own hardware (A40 GPUs)
- Does NOT accept user-submitted results

### CodeCarbon

**Scope**: Carbon emissions tracking
**Architecture**: Library + dashboard
**Relationship to LLenergyMeasure**: Currently used as energy measurement backend

**Limitations:**
- Wraps pynvml (adds overhead vs direct NVML)
- Reports energy in kWh and carbon, not raw joules
- Sometimes reports `duration_sec=0.0` (bug we already work around)
- No CPU/DRAM energy measurement on all platforms

### Carbontracker

**Scope**: DNN training carbon footprint prediction
**Status**: Less actively maintained than Zeus or CodeCarbon
**Not relevant**: Training-focused, not inference

### Eco2AI

**Scope**: CO2 emissions from ML
**Status**: Academic project, limited maintenance
**Not relevant**: Training-focused, simpler than CodeCarbon

---

## 3. LLM Benchmarking / Evaluation Tools

### lm-eval-harness (EleutherAI)

**Repository**: github.com/EleutherAI/lm-evaluation-harness
**Scope**: LLM evaluation framework (accuracy, not efficiency)
**Architecture**: Library-first, importable

**Key Design:**
```python
import lm_eval
results = lm_eval.simple_evaluate(
    model="hf",
    model_args="pretrained=meta-llama/Llama-2-7b-hf",
    tasks=["hellaswag", "mmlu"],
    batch_size=8,
)
```

**Relevant Patterns:**
- Task registry system for extensible benchmarks
- Model wrapper abstraction (supports HF, vLLM, GGUF, API models)
- Results as structured dicts, serialisable to JSON
- CLI is thin: `lm-eval --model hf --tasks hellaswag --batch_size 8`
- Only 3 top-level CLI commands

**Not relevant**: Measures accuracy/quality, not energy/efficiency

### HuggingFace LLM-Perf Leaderboard

**Scope**: LLM inference performance benchmarking
**Key insight**: Runs benchmarks centrally on specific hardware (A100, H100)
**Model**: Pre-computed results only, no user submissions
**Metrics**: Latency, throughput, memory — but NOT energy

### Artificial Analysis

**Scope**: LLM API performance tracking
**Model**: Tests API endpoints, not local inference
**Not relevant**: Different use case (API benchmarking vs local measurement)

---

## 4. Experiment Tracking / MLOps

### MLflow

**Architecture**: Most relevant architectural reference
**Key patterns we could adopt:**
- URI-based tracking store (local ↔ remote transparency)
- Experiment → Run → Metrics/Artifacts hierarchy
- Autologging integration (automatic metric capture)
- REST API for programmatic access
- 5 top-level CLI commands only

### Weights & Biases

**Architecture**: Cloud-first, client library
**Key insight**: Heavy vendor lock-in, but excellent UX for results visualisation
**Relevant**: Their results explorer UI is what our future web platform could aspire to

### ClearML

**Architecture**: Self-hostable alternative to W&B
**Key insight**: Open-source server + agents model
**Relevant**: Shows how self-hosted ML platforms work

---

## 5. Deployment Patterns

### pip extras for backends
**Universal pattern**: `pip install package[optional]`
- LLenergyMeasure already does this correctly
- Industry confirms: one base install, extras for GPU-specific backends

### Docker usage
- Docker for **isolation and reproducibility**, not as primary deployment
- Researchers prefer bare-metal for accurate benchmarking (container overhead)
- Docker images for CI/CD and multi-backend comparison
- LLenergyMeasure's Docker support is appropriate but shouldn't be the primary path

### Leaderboard trust model
- **Every credible leaderboard** runs benchmarks on their own hardware
- None accept user-submitted results (reproducibility + trust)
- HuggingFace Open LLM Leaderboard: runs on HF infrastructure
- ML.ENERGY Leaderboard: runs on their A40 GPUs
- HF LLM-Perf: runs on their A100/H100
- **Implication**: Future central DB should be curator-verified, not crowd-sourced

---

## 6. Competitive Positioning

### What LLenergyMeasure uniquely offers (no other tool does all four):
1. **Energy measurement** during LLM inference (Zeus does energy, but not LLM-specific)
2. **LLM-specific metrics** — TTFT, ITL, streaming latency, FLOPs estimation
3. **Multi-backend orchestration** — PyTorch, vLLM, TensorRT-LLM comparison
4. **SSOT config introspection** — Auto-discovered parameters from Pydantic models

### Closest competitors by dimension:
| Dimension | Closest Tool | Gap |
|-----------|-------------|-----|
| Energy | Zeus | Zeus better at raw measurement; we add LLM context |
| LLM metrics | lm-eval-harness | They do accuracy; we do efficiency |
| Orchestration | MLflow | They do experiment tracking; we do experiment execution |
| Config | None | SSOT introspection is genuinely unique |

### Integration opportunities:
- **Zeus as energy backend**: Replace CodeCarbon with Zeus for better accuracy + broader hardware
- **lm-eval integration**: Run quality benchmarks alongside efficiency benchmarks
- **MLflow as tracking store**: Optional MLflow integration for users who already use it

---

## 7. Key Recommendations

### Architecture (confirmed by research):
1. **Library-first**: Extract clean public API from current module structure
2. **CLI as thin wrapper**: Reduce to 7-9 commands, each calling library functions
3. **No premature API/server**: Clean layers first, HTTP later when web platform comes

### Energy measurement:
4. **Consider Zeus integration**: Better accuracy, broader hardware support
5. **Keep CodeCarbon as fallback**: For users without Zeus dependencies

### Campaign system:
6. **Simplify or remove for CLI**: Grid generation → config command. Resume → minimal state.
7. **Don't build web campaign on CLI campaign code**: Architecturally incompatible

### Results & storage:
8. **Version the results schema now**: Needed regardless of web timeline
9. **Local-first storage**: URI-based routing can come later (MLflow pattern)
10. **Leaderboard is separate product**: Don't conflate with CLI results

### Deployment:
11. **pip install primary**: Docker for multi-backend comparison only
12. **HPC support deferred**: SLURM/K8s/Apptainer post-v2.0
