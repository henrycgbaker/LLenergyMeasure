# P-19: GPU Utilisation Sampling

**Module**: `src/llenergymeasure/core/gpu_utilisation.py`
**Risk Level**: MEDIUM
**Decision**: Keep — v2.0
**Planning Gap**: Not explicitly described in any design document. `energy-backends.md` references the NVML single-session owner pattern for energy backends but does not mention that `GPUUtilisationSampler` is a separate concurrent NVML user that could conflict.

---

## What Exists in the Code

**Primary file(s)**: `src/llenergymeasure/core/gpu_utilisation.py`
**Key classes/functions**:
- `UtilisationSample` (line 21) — dataclass with three fields: `timestamp: float` (via `time.perf_counter()`), `sm_utilisation: float | None` (0–100, percent), `memory_bandwidth: float | None` (0–100, percent)
- `GPUUtilisationSampler` (line 29) — context manager class; starts a daemon `threading.Thread` on entry, stops it on exit
- `GPUUtilisationSampler._sample_loop()` (line 84) — calls `pynvml.nvmlDeviceGetUtilizationRates()` at the configured interval; stores both `util.gpu` (SM) and `util.memory` (memory bandwidth) per sample
- `GPUUtilisationSampler.get_mean_utilisation()` (line 137) — returns mean SM utilisation or `None` if no samples
- `GPUUtilisationSampler.get_mean_memory_bandwidth()` (line 146) — returns mean memory bandwidth or `None`
- `GPUSamplerResult` (line 167) — convenience dataclass for extracting results from sampler; has `from_sampler()` classmethod

The sampler uses `start_new_session=False` (threading, not multiprocessing). The default interval is 100ms. It gracefully handles `ImportError` (pynvml not installed) and NVML init failures by simply collecting zero samples rather than raising — making the whole system degrade cleanly on non-NVML systems or when vLLM's CUDA context conflicts. The `is_available` property (line 162) indicates whether pynvml was successfully initialised.

## Why It Matters

SM utilisation and memory bandwidth utilisation are critical secondary metrics for understanding how efficiently the GPU is being used during inference. A model running at 800 tokens/second with 40% SM utilisation tells a very different story from one at 800 tokens/second with 95% SM utilisation — the former has GPU headroom; the latter is at capacity. These metrics, collected into `GPUUtilisationMetrics` within `ExtendedEfficiencyMetrics`, feed directly into the tool's core mission of characterising implementation-parameter effects on efficiency. Without this sampler, those fields in the result schema are permanently null.

## Planning Gap Details

- `designs/architecture.md` — names `core/` as containing "inference engine, energy backends, metrics" but does not enumerate the GPU sampler specifically
- `designs/energy-backends.md` (line 62–63) — acknowledges that `PowerThermalSampler` and `GPUUtilisationSampler` both use `pynvml` directly, and notes "concurrent access needs validation testing" — but provides no resolution and defers it
- `designs/observability.md` — mentions Rich for display; does not describe how GPU utilisation data flows from the sampler to the display layer
- `decisions/architecture.md` — confirms NVML single-session owner for energy backends; does not address the concurrent use with the utilisation sampler

The `core/CLAUDE.md` internal note (not a planning doc) correctly documents this, including the CUDA context conflict warning with vLLM. The planning docs do not.

## Recommendation for Phase 5

Carry `GPUUtilisationSampler` forward unchanged into `core/gpu_utilisation.py` in the v2.0 structure. The implementation is clean, handles all failure modes, and is already wired into `ExtendedEfficiencyMetrics` via `GPUUtilisationMetrics.sm_utilisation_mean` and `memory_bandwidth_utilisation`.

One action required: resolve the NVML concurrent access question flagged in `energy-backends.md`. The sampler and the energy backend both call `pynvml.nvmlInit()` and `pynvml.nvmlShutdown()`. In practice, multiple NVML sessions are supported by the driver — but this needs an explicit test before v2.0 ships with Zeus, which owns the session differently. Document the resolution in `energy-backends.md`.

The `GPUSamplerResult.from_sampler()` classmethod (line 176) provides a clean extraction point — use it in the orchestrator to decouple the sampler lifecycle from the metrics collection step.
