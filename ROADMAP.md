# LLenergyMeasure Roadmap

> Research-focused efficiency measurement for LLM inference.
> Complements MLPerf by providing energy, FLOPs, and quality-efficiency analysis.

## Vision

A comprehensive tool for ML researchers to understand the **true cost** of LLM inference—energy, compute, and quality tradeoffs—with publication-grade rigour.

---

## Current State (v2.x)

### What We Have

| Category | Capabilities |
|----------|-------------|
| **Energy** | Total energy (J), GPU/CPU/RAM breakdown, CO₂ emissions via CodeCarbon |
| **Performance** | Throughput (tok/s), TTFT, ITL, latency distributions |
| **Compute** | FLOPs estimation (calflops → architecture → parameter fallback), peak memory |
| **Backends** | PyTorch, vLLM, TensorRT-LLM with backend-specific optimisations |
| **Statistical** | Multi-cycle experiments, confidence intervals, late aggregation |
| **Infrastructure** | Docker, MIG awareness, experiment resumption, config inheritance |
| **Provenance** | Full parameter provenance tracking (source of every config value) |

### Known Limitations

- Energy readings include idle power (15-30% overestimation)
- Prefill and decode phases blended together
- No power profile over time
- FLOPs are estimated, not measured
- No output quality metrics
- vLLM ITL is proportionally estimated, not truly measured

---

## v3.0 — Measurement Precision

> **Theme:** Get the fundamentals right with a clean schema break.

### 3.0.1 Prefill/Decode Phase Split

Separate metrics at the first generated token boundary.

**Deliverables:**
- [ ] Split timing metrics: `prefill_time_ms`, `decode_time_ms`
- [ ] Split energy metrics: `prefill_energy_j`, `decode_energy_j`
- [ ] Split throughput: `prefill_tokens_per_sec`, `decode_tokens_per_sec`
- [ ] Update results schema with phase-level breakdown
- [ ] Aggregate statistics per phase in multi-cycle experiments

**Why:** Prefill is compute-bound (parallel), decode is memory-bound (sequential). Blending them hides where inefficiency lies. Researchers need phase-level visibility.

### 3.0.2 Time-Series Power Collection

Sample power, memory, and utilisation over time.

**Deliverables:**
- [ ] Background sampling thread (configurable interval, default 100ms)
- [ ] Collect: power (W), memory (GB), GPU utilisation (%), timestamp
- [ ] Store time-series in results JSON
- [ ] Correlate with phase boundaries (prefill/decode transition visible)
- [ ] Optional: thermal state tracking for throttling detection

**Why:** Enables power profile analysis, phase visualisation, and anomaly detection. Essential for the web app analysis layer.

**Schema:**
```json
{
  "time_series": {
    "interval_ms": 100,
    "samples": [
      {"t": 0.0, "power_w": 85, "memory_gb": 2.1, "gpu_util": 15},
      {"t": 0.1, "power_w": 340, "memory_gb": 8.4, "gpu_util": 95},
      ...
    ]
  }
}
```

### 3.0.3 Baseline Power Subtraction

Measure idle GPU power and subtract from readings.

**Deliverables:**
- [ ] `--measure-baseline` flag to capture idle power before experiment
- [ ] Automatic baseline measurement option in config
- [ ] Store baseline in results: `baseline_power_w`
- [ ] Report both raw and baseline-corrected energy
- [ ] Document methodology for reproducibility

**Why:** Current readings overestimate by 15-30% due to idle power. Baseline subtraction gives true inference energy cost.

### 3.0.4 Profiler Integration (Optional Mode)

Use `torch.profiler` for ground-truth FLOPs.

**Deliverables:**
- [ ] `--profile` CLI flag to enable profiling mode
- [ ] `profiling.enabled` config option
- [ ] Extract actual FLOPs from CUDA kernel execution
- [ ] Compare against estimation, report discrepancy
- [ ] Store profiler summary in results
- [ ] Clear documentation: profiling adds overhead, use for validation not energy benchmarks

**Why:** Current FLOPs are estimated. Profiler gives ground truth for validation. Overhead (~10-30%) means it's a separate mode, not default.

**Config:**
```yaml
profiling:
  enabled: true
  record_shapes: true
  with_flops: true
```

### 3.0.5 Environment Metadata

Capture system state for reproducibility.

**Deliverables:**
- [ ] CUDA version, driver version
- [ ] GPU thermal state at start/end
- [ ] Power limit settings
- [ ] CPU governor, frequency scaling state
- [ ] Container/VM detection
- [ ] Store in results under `environment` key

**Why:** Results without environment context aren't reproducible. Researchers need to know if thermal throttling or power limits affected measurements.

### 3.0.6 Schema Migration

Clean break with migration support.

**Deliverables:**
- [ ] New results schema version (v3)
- [ ] Schema version field in all results
- [ ] Migration script for v2 → v3 results
- [ ] Documentation of breaking changes
- [ ] Deprecation warnings for old schema access patterns

---

## v3.1 — Quality Metrics

> **Theme:** Efficiency without quality context is incomplete.

### 3.1.1 Perplexity Computation

Measure output quality via perplexity.

**Deliverables:**
- [ ] Perplexity on generated outputs (model's own generations)
- [ ] Perplexity on held-out eval sets (WikiText-2, C4)
- [ ] Config option to select mode: `quality.mode: generated | eval_set | both`
- [ ] Store in results: `quality.perplexity`, `quality.eval_dataset`
- [ ] Handle edge cases (empty outputs, OOM on long sequences)

**Config:**
```yaml
quality:
  enabled: true
  mode: both  # generated | eval_set | both
  eval_dataset: wikitext-2
  max_eval_samples: 1000
```

**Why:** Perplexity is the standard research metric for language model quality. Enables quality-efficiency tradeoff analysis.

### 3.1.2 Pareto Frontier Analysis

Identify optimal quality-efficiency tradeoffs.

**Deliverables:**
- [ ] Compute Pareto frontier across experiments
- [ ] Metrics: quality (perplexity) vs efficiency (tokens/joule, tokens/FLOP)
- [ ] CLI command: `results pareto --x-metric tokens_per_joule --y-metric perplexity`
- [ ] Export Pareto-optimal configurations
- [ ] Integration with web app for visualisation

**Why:** Researchers need to find the best models for their quality-efficiency constraints. Pareto analysis answers "which model gives best quality per joule?"

### 3.1.3 External Quality Scores (Optional)

Accept user-provided quality metrics.

**Deliverables:**
- [ ] Config option to specify external quality file
- [ ] Support common formats (JSON, CSV)
- [ ] Merge external scores into results
- [ ] Enable Pareto analysis with custom metrics (BLEU, ROUGE, accuracy)

**Why:** Perplexity is a proxy. Task-specific metrics (BLEU for translation, accuracy for classification) matter for real applications.

---

## v3.2 — Ecosystem Integration

> **Theme:** Fit into existing ML research workflows.

### 3.2.1 MLflow Integration

Log experiments to MLflow tracking.

**Deliverables:**
- [ ] `--mlflow` flag to enable logging
- [ ] Config section for MLflow settings (tracking URI, experiment name)
- [ ] Log parameters: model, backend, batch size, etc.
- [ ] Log metrics: energy, throughput, latency, quality
- [ ] Log artifacts: full results JSON, config
- [ ] Support both local and remote MLflow servers

**Config:**
```yaml
integrations:
  mlflow:
    enabled: true
    tracking_uri: http://localhost:5000
    experiment_name: llm-efficiency
```

### 3.2.2 Weights & Biases Integration

Push results to W&B.

**Deliverables:**
- [ ] `--wandb` flag to enable logging
- [ ] Config section for W&B settings (project, entity)
- [ ] Log same parameters/metrics as MLflow
- [ ] Support W&B Tables for structured results
- [ ] Enable W&B comparison features

**Config:**
```yaml
integrations:
  wandb:
    enabled: true
    project: llm-efficiency
    entity: my-team
```

### 3.2.3 Integration Selection

User chooses preferred platform.

**Deliverables:**
- [ ] Support enabling one or both integrations
- [ ] Consistent metric naming across platforms
- [ ] Documentation comparing MLflow vs W&B for different use cases
- [ ] Environment variable overrides for CI/CD

---

## v3.3 — Analysis Features

> **Theme:** Extract insights from experiment data.

### 3.3.1 Correlation Analysis

Automatically compute correlation between parameter values and energy/efficiency metrics across experiments.

```bash
$ lem analyse correlations --experiments exp1,exp2,exp3,...

Parameter Correlations with Energy (J/token):
  vllm.speculative.num_tokens:     r=-0.72 (p<0.01)  # more speculation = lower energy
  batching.batch_size:             r=-0.45 (p<0.05)  # larger batches = lower energy
  decoder.temperature:             r=+0.12 (p=0.31)  # no significant effect
```

**Requirements:**
- Multiple experiments with varied parameters
- Statistical analysis module (scipy/statsmodels)
- Sufficient sample sizes for significance

### 3.3.2 Multi-Experiment Comparison Dashboard

Visual comparison of experiments highlighting parameter differences and their effects.

**Features:**
- Side-by-side metric comparison
- Parameter diff highlighting
- Trend visualisation across experiment series
- Export to publication-ready formats

### 3.3.3 Automated Parameter Sweep

Given a base config, automatically vary parameters to find energy-optimal configurations.

```bash
$ lem sweep config.yaml \
    --vary batching.batch_size=1,2,4,8 \
    --vary vllm.speculative.num_tokens=3,5,7
```

**Features:**
- Cartesian product of parameter variations
- Optional early stopping for poor configurations
- Pareto frontier identification (throughput vs energy)
- Automatic best-config selection

### 3.3.4 NormalisedMetrics Population

Wire up the existing `NormalisedMetrics` model in `domain/metrics.py` to enable cross-backend efficiency comparisons with precision-adjusted FLOPs.

**Features:**
- Precision-adjusted FLOPs (INT4 ops vs FP16 ops)
- Backend-normalised throughput
- Energy efficiency per effective FLOP
- Cross-hardware normalisation

---

## Future — Architecture Expansion

> **Theme:** Beyond Transformers.

### State-Space Models (SSM)

Support for Mamba, RWKV, Jamba, and hybrid architectures.

**Considerations:**
- [ ] FLOPs estimation for SSM layers (different from attention)
- [ ] No prefill/decode distinction in pure SSM (linear complexity)
- [ ] Hybrid models (Jamba) need mixed handling
- [ ] Memory patterns differ significantly
- [ ] Benchmark datasets may need adaptation

**Target Models:**
- Mamba (state-space)
- RWKV (linear attention)
- Jamba (hybrid Mamba + Transformer)
- StripedHyena
- Griffin

**Timeline:** After v3.x stabilises, based on research community adoption.

---

## Out of Scope (Handled Elsewhere)

| Feature | Location |
|---------|----------|
| Visualisation & dashboards | Web app (separate branch) |
| Comparative analysis UI | Web app |
| Cost estimation ($/token) | Analysis layer |
| Real-time monitoring | Web app |
| Public leaderboards | Future consideration |

---

## Known High-Impact Parameters

Parameters that empirically show large energy effects (for documentation/guidance):

| Parameter | Typical Effect | Notes |
|-----------|---------------|-------|
| `vllm.speculative.method` | -30-50% energy | Speculative decoding reduces total compute |
| `vllm.speculative.num_tokens` | Variable | Optimal varies by model/task |
| `vllm.enable_prefix_caching` | -10-30% for repeated prefixes | Cache hit rate dependent |
| `pytorch.torch_compile` | -5-15% after warmup | Compilation overhead on first run |
| `batching.batch_size` | Higher = better efficiency | Diminishing returns, memory constrained |
| `pytorch.attn_implementation` | flash_attention_2 most efficient | Hardware dependent |

This is guidance only—users should measure actual effects for their specific workloads.

---

## Implementation Principles

1. **Measurement first** — Get accurate data before building analysis
2. **Optional complexity** — Profiling, quality metrics are opt-in
3. **Backwards compatibility within major versions** — Only break at v3.0
4. **Research rigour** — Statistical validity, reproducibility, clear methodology
5. **Minimal dependencies** — Don't bloat the core tool

---

## Priority Order

```
v3.0.1 Prefill/Decode Split     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━▶
v3.0.2 Time-Series Power        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━▶
v3.0.3 Baseline Power           ━━━━━━━━━━━━━━━━━━━━━▶
v3.0.4 Profiler Integration     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━▶
v3.0.5 Environment Metadata     ━━━━━━━━━━━━━━━▶
v3.0.6 Schema Migration         ━━━━━━━━━━▶
v3.1.x Quality Metrics          ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━▶
v3.2.x Ecosystem Integration    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━▶
v3.3.x Analysis Features        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━▶
Future SSM/Hybrid Support       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━▶
```

---

## Contributing

When implementing roadmap items:

1. Create feature branch: `feature/v3.0.1-prefill-decode-split`
2. Update relevant module CLAUDE.md with new functionality
3. Add tests for new metrics/features
4. Update CLI help and docs
5. Squash merge to main with conventional commit

---

## Changelog

| Date | Change |
|------|--------|
| 2026-01-21 | Added v3.3 Analysis Features section (correlation, sweep, comparison) |
| 2025-01-21 | Initial roadmap created from research-pm and research-scientist review |
