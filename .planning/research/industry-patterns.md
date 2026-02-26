# Industry Patterns

Validated norms from scanning lm-eval, MLflow, Optimum-Benchmark, CodeCarbon, Zeus,
ClearML, W&B, and ML.ENERGY. Sources: `01-lm-eval-harness.md`, `02-mlflow-architecture.md`,
`04-deployment-patterns.md`.

## Architecture: Library → CLI → Web

All credible tools ship in this order, never as monolith → split:
- **MLflow**: `mlflow` library (Fluent API + Client API) → `mlflow server` CLI → web UI
- **lm-eval**: `lm_eval` library (`simple_evaluate`) → `lm_eval` CLI → hosted leaderboard
- **CodeCarbon**: `codecarbon` library → CLI → dashboard
- **Zeus**: `zeus` library (ZeusMonitor) → no CLI — library-first by design

**Implication**: The library boundary must be established first, not later. Doing it in v3.0
after CLI complexity grows creates more debt, not less.

## Installation: pip Extras (No Default Backend)

lm-eval's pattern is the clearest example — base install has **zero** backends:
```bash
pip install lm_eval              # no backends
pip install lm_eval[hf]          # HuggingFace backend
pip install lm_eval[vllm]        # vLLM backend
pip install lm_eval[anthropic]   # Anthropic API
```

Optimum-Benchmark follows the same pattern. Explicit backend selection at install time is
the industry norm — not auto-detection or bundled defaults.

**Docker**: Used for conflicting deps (like vLLM + TensorRT-LLM), not as the primary
workflow. Bare-metal pip install remains primary for researchers (container overhead
contaminates benchmarks).

## Results: Local-First with Optional Remote

| Tool | Local storage | Remote sync |
|------|--------------|-------------|
| MLflow | Filesystem / SQLite | MLflow tracking server (URI-based routing) |
| lm-eval | JSON output files | EvaluationTracker → HF Hub (optional) |
| W&B | Local run cache | Cloud sync (opt-in configurable) |
| Zeus | In-process ZeusMonitor | ML.ENERGY leaderboard (curated, not crowdsourced) |

**URI-based routing** (MLflow pattern): `MLFLOW_TRACKING_URI=sqlite:///mlruns.db` vs
`MLFLOW_TRACKING_URI=http://remote-server:5000` — local → remote without code changes.

## Web Platforms: Start as Results Explorer

All credible ML leaderboards are **centrally run**, not crowdsourced:
- ML.ENERGY: Run their own benchmarks, publish as static JSON
- HF Open LLM Leaderboard: Run experiments on their infrastructure
- MLPerf: Submitters run on standardised hardware, results validated before publishing

**No leaderboard accepts raw user-submitted results directly.** The upload model (`llem results push`) should be framed as contributing to a curated archive, with results subject to validation or clear provenance metadata.

## Outbound Worker Model (ClearML-Agent Pattern)

For web platforms that need to run experiments, the dominant pattern is workers that
connect **outbound** to a central coordinator (not inbound SSH):

```
Central coordinator (web server)
    ↑ outbound connection
GPU worker (user's Docker container)
```

This is firewall-friendly — works on HPC clusters, university networks, corporate
environments. ClearML-Agent and W&B-Agent both use this pattern.
