# P-14: Warmup Iteration Config

**Module**: `src/llenergymeasure/config/models.py`
**Risk Level**: MEDIUM
**Decision**: Keep — v2.0
**Planning Gap**: `designs/experiment-config.md` mentions warmup as a concept but provides no YAML examples or field documentation for the warmup config surface.

---

## What Exists in the Code

**Primary file**: `src/llenergymeasure/config/models.py`
**Key class**: `WarmupConfig` (line 379)
**Fields**:
- `enabled: bool = True` (line 387)
- `convergence_detection: bool = True` (line 388) — CV-based auto-stop vs fixed iterations
- `cv_threshold: float = 0.05` (line 392) — 5% coefficient of variation to declare convergence
- `max_prompts: int = 50` (line 398) — hard cap on warmup iterations
- `window_size: int = 5` (line 404) — rolling window for CV computation
- `min_prompts: int = 5` (line 410) — minimum iterations before convergence check

**Energy backend integration** (`core/energy_backends/codecarbon.py`): `warm_up()` function reads `WarmupConfig` and runs the model until the CV threshold is met or `max_prompts` is exhausted.

## Why It Matters

Warmup is critical for measurement validity — GPU/CPU caches, JIT compilation, and CUDA graph construction all affect the first N inference steps. Without controlled warmup, the measurement window includes GPU ramp-up artefacts. The CV-threshold approach is methodologically correct: it runs warmup until power/latency variance stabilises, rather than a fixed N iterations that may be too few for some hardware.

## Planning Gap Details

`designs/experiment-config.md` acknowledges `warmup` as a concept (mentioning `steady_state_window` in the result schema) but provides no YAML example and no field documentation. A Phase 5 implementor will need to reinvent this unless the existing `WarmupConfig` is explicitly referenced. The companion `WarmupResult` (N-C03) tracks convergence outcomes.

## Recommendation for Phase 5

Add a `warmup:` subsection to `designs/experiment-config.md`:

```yaml
warmup:
  enabled: true
  convergence_detection: true   # false = fixed max_prompts iterations
  cv_threshold: 0.05            # 5% coefficient of variation to declare convergence
  max_prompts: 50               # hard cap regardless of convergence
  window_size: 5                # rolling window for CV computation
  min_prompts: 5                # minimum runs before convergence check starts
```

> Implementation: `config/models.py`, `WarmupConfig`. The warmup runner is in
> `core/energy_backends/codecarbon.py`, `warm_up()`. Warmup outcomes are tracked in
> `WarmupResult` (see N-C03) and embedded in `RawProcessResult.warmup_result`.
