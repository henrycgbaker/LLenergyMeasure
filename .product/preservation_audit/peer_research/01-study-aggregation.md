# Peer Research: Study Aggregation & Cross-Config Comparison

> Generated 2026-02-26. Peer evidence for preservation audit item P-02.

---

## Evidence Per Tool

### 1. lm-eval (EleutherAI/lm-evaluation-harness)

**Does it compute aggregate statistics across configs?** Yes, but within a predefined task
group hierarchy, not across arbitrary config fields. Per-task metrics (accuracy, BLEU, etc.)
are aggregated into group scores via each group's own `aggregate()` method (typically
weighted mean). Standard error is computed at the task level using bootstrap resampling
(configurable via `bootstrap_iters`, default 100,000 iterations). Group-level stderr uses
pooled sample stderr — combining task-level standard errors weighted by sample size — but
this was historically buggy (erroneously large stderr for MMLU groups, since fixed).

**Grouping support?** Hierarchical only. Groups and tags are predefined in task YAML
configs. There is no mechanism to group by arbitrary model config fields (e.g.,
`quantization`, `batch_size`). The grouping structure is `task -> subtask -> group` with
bottom-up collection via `_collect_groups_bottom_up()`. You cannot dynamically pivot by a
model parameter at query time.

**Statistical methods:** Bootstrap stderr (percentile method, up to 1000-draw chunks,
seeded RNG for reproducibility). `mean_stderr`, `sample_stddev`, `population_stddev` are
all available as metric aggregation functions. For expensive metrics (BLEU, CHRF, TER),
bootstrap iterations are capped at 100. No t-tests, no Bayesian methods, no effect size.

**Key finding:** lm-eval's aggregation answers "how does this model perform across tasks?"
not "how does this config compare to that config?" It does not produce cross-config
comparison tables. If you want to compare two model configurations, you run lm-eval twice
and diff the JSON outputs yourself.

**Sources:**
- [lm-evaluation-harness metrics.py](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/api/metrics.py)
- [lm-evaluation-harness evaluator_utils.py](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/evaluator_utils.py)
- [lm-evaluation-harness task guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md)

---

### 2. optimum-benchmark (HuggingFace)

**Does it compute aggregate statistics across configs?** No. Each benchmark run produces a
`BenchmarkReport` object (serialised to `benchmark_report.json`) containing single-run
metrics: latency, throughput, memory, energy. No built-in mechanism combines reports from
multiple runs or computes CIs.

**Grouping support?** None. Hydra's `--multirun` flag enables configuration sweeps
(Cartesian product of config parameters), but each sweep leg produces an independent report
in its own directory. There is no built-in tool to collect, group, or compare those reports.

**Comparison approach:** Fully delegated to users. An example script in
`examples/running-mistrals/report.py` demonstrates manual aggregation: it reads CSV files
from multiple sweep directories, merges them into a pandas DataFrame, then generates plots
grouped by quantisation scheme. This is an *example*, not part of the library API. The
LLM-Perf Leaderboard (which uses optimum-benchmark) performs its own aggregation externally.

**Statistical methods:** None built-in. The example `report.py` uses raw pandas operations
(no CIs, no bootstrap, no statistical tests).

**Key finding:** optimum-benchmark is a *runner*, not an *analyser*. It produces structured
JSON output and expects downstream tools (pandas, notebooks, leaderboard infrastructure) to
handle comparison.

**Sources:**
- [optimum-benchmark README](https://github.com/huggingface/optimum-benchmark)
- [Example report.py](https://github.com/huggingface/optimum-benchmark/blob/ef70214a33902d33896d4edd663e08480682c05f/examples/running-mistrals/report.py)
- [LLM-Perf Leaderboard](https://huggingface.co/spaces/optimum/llm-perf-leaderboard)

---

### 3. vLLM Benchmarks

**Does it compute aggregate statistics across configs?** Partially. `vllm bench sweep`
automates running the Cartesian product of serve-params and bench-params, producing per-run
JSON results. The `compare-json-results.py` script computes performance *ratios* between
two result files (output throughput, median TTFT, median TPOT) but does not compute CIs,
standard errors, or statistical tests. It is a reporting tool, not a statistical tool.

**Grouping support?** Yes, limited. The sweep plotting script (`plot.py`) and
`plot_pareto` subcommand support `--label-by` parameters (e.g.,
`max_concurrency`, `tensor_parallel_size`, `pipeline_parallel_size`) for visual grouping.
The Pareto chart identifies optimal throughput-vs-latency trade-offs across configurations.
However, this is visual — no numeric grouping API exists.

**Statistical methods:** None. Single-run descriptive statistics only (mean, median, p50,
p90, p95, p99 percentile latencies). No bootstrap, no CIs, no repeated-run aggregation.
The assumption is that each sweep point is a single benchmark run.

**Key finding:** vLLM's sweep infrastructure is the closest peer to our `campaign` concept:
it runs multiple configs, collects results, and provides basic comparison tooling (ratio
tables, Pareto charts). But the comparison is purely descriptive (ratios of single-run
metrics), with no statistical rigour (no repeated runs, no CIs). This is a tuning tool,
not a measurement tool.

**Sources:**
- [vLLM benchmark CLI docs](https://docs.vllm.ai/en/latest/benchmarking/cli/)
- [vLLM parameter sweeps docs](https://docs.vllm.ai/en/latest/benchmarking/sweeps/)
- [vLLM benchmarks directory](https://github.com/vllm-project/vllm/tree/main/benchmarks)

---

### 4. MLPerf (MLCommons)

**Does it compute aggregate statistics across configs?** Not directly. MLPerf is a
*specification and validation framework*, not an analysis tool. The LoadGen module generates
queries, measures end-to-end latency, and validates against SLA constraints. Results are
submitted as standardised JSON/log files and validated against rules.

**Grouping support?** Cross-submission comparison is handled externally via the
`cm4mlperf-results` repository (Collective Mind format) and the CK Playground. These tools
allow adding derived metrics (performance/Watt, performance/$) and filtering by hardware,
software stack, or submission. This is an external aggregation platform, not part of the
benchmark tool itself.

**Statistical methods:** MLPerf uses *statistical validity requirements*, not statistical
analysis. The LoadGen requires a 99% confidence interval that latency constraints hold,
determining the minimum number of queries needed (highly nonlinear). Percentile latencies
(p50, p90, p95, p97, p99, p99.9) are reported as raw values. There are no cross-submission
statistical tests — the community reviews results via peer validation, not automated
statistical comparison.

**Key finding:** MLPerf deliberately separates measurement (LoadGen) from analysis
(CK Playground). The measurement tool ensures statistical validity of individual
submissions; cross-submission comparison is a community activity, not an automated one.
The derived-metrics platform (performance/Watt) is the closest analogue to our grouping
function, but it is a separate tool.

**Sources:**
- [MLPerf Inference paper](https://arxiv.org/pdf/1911.02549)
- [cm4mlperf-results](https://github.com/mlcommons/cm4mlperf-results)
- [MLPerf Inference docs](https://docs.mlcommons.org/inference/)
- [MLPerf Submission Guide](https://docs.mlcommons.org/inference/submission/)

---

### 5. Zeus / ML.ENERGY Benchmark

**Does it compute aggregate statistics across configs?** Partially. Zeus itself is a
measurement library (GPU energy/power monitoring). The ML.ENERGY Benchmark (separate repo:
`ml-energy/benchmark`) runs configurations independently and produces per-config
`results.json` files with `steady_state_energy`, `output_throughput`, and
`steady_state_energy_per_token`.

The Benchmark's main analytical contribution is **Pareto frontier construction**: given a
user-specified latency target, it identifies the energy-optimal configuration satisfying
that constraint. This is a built-in cross-config comparison, but it is constraint-based
optimisation rather than statistical comparison.

**Grouping support?** The directory structure organises results by model, GPU, and runtime
parameters (hierarchical), but there is no arbitrary field-based grouping API.

**Statistical methods:** The ML.ENERGY Benchmark paper (arXiv:2505.06371) describes
per-request energy accounting during a "steady state" period. Notably, the paper does
*not* describe confidence intervals, statistical significance testing, or multiple-run
averaging. Measurements appear to be single runs. There is an `automated_validation`
script for sanity checks, but its details are undocumented.

**Key finding:** Zeus is the closest peer in terms of *what it measures* (energy per token,
throughput, energy-optimal configuration). But it does not replicate experiments or compute
CIs. The Pareto frontier is a valuable analysis primitive that we do not currently have.

**Sources:**
- [Zeus GitHub](https://github.com/ml-energy/zeus)
- [ML.ENERGY Benchmark GitHub](https://github.com/ml-energy/benchmark)
- [ML.ENERGY Benchmark paper](https://arxiv.org/html/2505.06371v1)
- [Zeus measuring energy docs](https://ml.energy/zeus/measure/)

---

### 6. llmperf (Ray/Anyscale)

**Does it compute aggregate statistics across configs?** No. Per-run output includes
descriptive statistics: mean, min, max, std, and quantiles (p25, p50, p75, p90, p95, p99)
for inter-token latency, TTFT, E2E latency, and throughput. All statistics are
*within-run* aggregations across requests, not across configs.

**Grouping support?** None. Each invocation benchmarks a single endpoint with fixed
parameters. The LLMPerf Leaderboard aggregates results across providers, but this is done
by a separate codebase (`llmperf-leaderboard`), not by llmperf itself.

**Comparison approach:** An `analyze-token-benchmark-results.ipynb` notebook is provided
for exploring single-run results (scatter plots of input tokens vs TTFT, latency
histograms). It uses pandas `read_json`, `mean()`, `plot.scatter()`, `plot.hist()`. No
cross-config comparison, no scipy imports, no statistical tests.

**Statistical methods:** Descriptive only (mean, std, percentiles). No CIs, no bootstrap,
no inferential statistics.

**Key finding:** llmperf is a single-endpoint benchmarking tool. Cross-config comparison is
entirely delegated to notebooks and the leaderboard.

**Sources:**
- [llmperf GitHub](https://github.com/ray-project/llmperf)
- [llmperf analysis notebook](https://github.com/ray-project/llmperf/blob/main/analyze-token-benchmark-results.ipynb)
- [llmperf-leaderboard](https://github.com/ray-project/llmperf-leaderboard)

---

### 7. W&B / MLflow (Experiment Tracking Platforms)

These are not benchmarking tools but provide the most mature cross-run comparison
infrastructure in the ML ecosystem.

#### Weights & Biases (W&B)

**Aggregate statistics:** Yes, built-in. Line plots support grouping with aggregation
functions: mean (default), median, min, max. Range visualisation supports min/max, std dev,
and std error. The Reports API exposes `groupby`, `groupby_aggfunc`, and
`groupby_rangefunc` programmatically.

**Grouping support:** Yes, arbitrary. Runs can be grouped by any config column
(`wandb.config` values), tags, or job type. Dynamic grouping: click "Group" in the UI and
select any column. Programmatic: `Runset(groupby=["config.group"])`. Multi-field grouping
is supported.

**Statistical methods:** Mean, median, min, max, std dev, std error as aggregation and range
functions. No bootstrap, no CIs (only std err bounds), no statistical tests. W&B is a
visualisation and tracking platform, not a statistics engine.

**Key observation:** W&B's grouping model is the gold standard for *arbitrary field-based
grouping* in ML experiment management. Our `aggregate_campaign_with_grouping()` function
is a library-level analogue of W&B's run grouping with bootstrap CIs instead of std err.

#### MLflow

**Aggregate statistics:** Limited. The Tracking UI shows the *last* tracked value for each
metric by default. A feature request (issue #7790) to allow mean, max, min, median
aggregation has been open since 2023 and is not yet implemented as a built-in feature.
Cross-experiment run comparison exists via `mlflow.search_runs()` and the comparison chart
view (parallel coordinates), but these are query/visualisation tools, not aggregation tools.

**Grouping support:** Runs can be queried and filtered by parameters using SQL-like
expressions. Visual comparison via chart views. No built-in arbitrary grouping with
aggregation.

**Statistical methods:** None built-in. The platform provides data access; statistics are
the user's responsibility.

**Key observation:** Even mature experiment tracking platforms do not provide built-in
statistical comparison (bootstrap, t-tests). They provide data access, grouping, and basic
descriptive aggregation (mean, std), then delegate inferential statistics to users.

**Sources:**
- [W&B grouping docs](https://docs.wandb.ai/guides/runs/grouping)
- [W&B line plot sampling](https://docs.wandb.ai/models/app/features/panels/line-plot/sampling)
- [W&B Reports API](https://docs.wandb.ai/guides/reports/edit-a-report/)
- [MLflow comparison docs](https://docs.databricks.com/aws/en/mlflow/visualize-runs)
- [MLflow aggregation feature request #7790](https://github.com/mlflow/mlflow/issues/7790)

---

### 8. scipy/statsmodels (Statistical Patterns for Benchmark Comparison)

This section captures the standard statistical methods used by researchers when they *do*
compare benchmark results, typically in notebooks or analysis scripts.

**Bootstrap CI (percentile method):** The most common approach for ML benchmark comparison.
`scipy.stats.bootstrap()` (added in scipy 1.7) supports BCa (bias-corrected and
accelerated), basic (reverse percentile), and percentile methods. Supports paired
resampling. The BCa method is generally recommended for small samples. Our `bootstrap_ci()`
uses the percentile method, which is the simplest but least accurate for small n.

**Paired comparisons:** When comparing two configs run on the same hardware under the same
conditions, a paired test is statistically more powerful. `scipy.stats.ttest_rel` (paired
t-test) or bootstrap with `paired=True` is appropriate. Our current grouping function does
not support paired comparison.

**Effect size:** Cohen's d or the common language effect size (CLES) quantifies *how much*
one config outperforms another, not just *whether* the difference is statistically
significant. Standard in A/B testing. Not used in any of the ML benchmark tools surveyed.

**Bayesian comparison:** PyMC-based Bayesian A/B testing produces posterior distributions
and probability statements ("P(config A is better) = 0.97"). More informative than
frequentist CIs for small samples. Not used in any peer tool.

**What researchers actually use:** Based on the peer tools surveyed, the overwhelming
pattern is: (1) descriptive statistics per config (mean, percentiles), optionally with (2)
bootstrap CIs, and then (3) manual visual comparison in notebooks. Nobody automates
pairwise statistical tests. Nobody computes effect sizes. The most sophisticated built-in
method (lm-eval's bootstrap stderr) is used for within-config uncertainty quantification,
not cross-config comparison.

**Sources:**
- [scipy.stats.bootstrap docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)
- [Bootstrap CI comparative study (2024)](https://arxiv.org/html/2404.12967v1)
- [Facebook bootstrapped library](https://github.com/facebookarchive/bootstrapped)
- [PyMC Bayesian A/B testing](https://www.pymc.io/projects/examples/en/latest/causal_inference/bayesian_ab_testing_introduction.html)

---

## Summary Table

| Tool | Built-in cross-config comparison? | Method | Arbitrary field grouping? | Repeated-run CIs? |
|------|----------------------------------|--------|--------------------------|-------------------|
| **lm-eval** | Within predefined groups only | Bootstrap stderr, pooled | No (hierarchical task groups) | Yes (bootstrap) |
| **optimum-benchmark** | No | N/A | No | No |
| **vLLM bench sweep** | Ratios + Pareto chart | Descriptive (single-run) | Visual only (`--label-by`) | No |
| **MLPerf** | External platform (CK) | Derived metrics (perf/Watt) | External (CK Playground) | No (validation only) |
| **Zeus/ML.ENERGY** | Pareto frontier | Constraint optimisation | No (directory hierarchy) | No |
| **llmperf** | No | N/A | No | No |
| **W&B** | Yes (UI + API) | Mean/median/std err | Yes (any config column) | Std err only (not bootstrap) |
| **MLflow** | Limited (query + visual) | Last value (no aggregation) | Filter only (no aggregation) | No |
| **Our v1.x** | **Yes (library)** | **Bootstrap CI (percentile)** | **Yes (dot-notation paths)** | **Yes** |

---

## Recommendation: Keep — v2.0

**Verdict: Incorporate.** The study aggregation and grouping functions should be preserved
and exposed in the v2.0 library API.

### Rationale

1. **No peer does what we do.** Of 8 tools surveyed, none provides a library-level function
   that (a) runs multiple cycles of the same config, (b) groups results by arbitrary config
   fields with dot-notation traversal, and (c) computes bootstrap CIs per group. This is a
   genuine differentiator. The closest peer (W&B) provides arbitrary grouping with std err,
   but it is a platform/UI feature, not a library API. lm-eval provides bootstrap stderr
   but only within predefined task hierarchies, not across arbitrary config fields.

2. **The alternative is worse for users.** If we delegate this to notebooks, every
   researcher using the tool must independently implement: (a) result loading, (b) field
   extraction with dot-notation, (c) grouping, (d) bootstrap resampling. The peer evidence
   shows this is exactly what happens with optimum-benchmark and llmperf — users write
   one-off analysis scripts. Given that our tool's core question is "which implementation
   choice is most efficient?", forcing users to answer it themselves undermines the tool's
   value proposition.

3. **The code is small and well-tested.** `aggregate_campaign_with_grouping()` is ~70 lines.
   `_extract_field_value()` is ~20 lines. `bootstrap_ci()` is ~40 lines. Total: ~130 lines
   of library code, with comprehensive unit tests covering multi-field grouping, nested
   paths, nonexistent fields, and CI computation. The maintenance burden is minimal.

4. **It belongs in the library, not the CLI.** The function operates on `list[ExperimentResult]`
   objects — it is a data transformation, not a presentation concern. Exposing it as a
   library function (not just a CLI feature) means notebook users and downstream tools can
   call it directly.

### Recommended changes for v2.0

- **Rename:** `aggregate_campaign_results()` -> `aggregate_study_results()`;
  `aggregate_campaign_with_grouping()` -> `aggregate_study_with_grouping()`
- **Expose in public API:** Include in `__init__.py` exports or as methods on `StudyResult`
- **Upgrade bootstrap method:** Consider BCa (bias-corrected and accelerated) instead of
  percentile — better for small samples (n=3-5 typical for energy measurement cycles).
  scipy's `scipy.stats.bootstrap` now supports this natively; our hand-rolled implementation
  could delegate to scipy.
- **Add to `StudyResult`:** The `StudyResult` model should carry aggregated statistics as
  a field, not require separate function calls. The P-02 audit item's proposal
  (`aggregated: dict | None`, `grouped: dict | None`) is sound.
- **Defer paired comparison and effect size to v2.1+:** No peer tool implements these.
  The added complexity is not justified until user demand materialises.
- **Defer Pareto frontier to v2.1+:** Zeus/ML.ENERGY and vLLM both use Pareto analysis for
  energy-latency trade-offs. This is a natural extension of our grouping function but
  requires a different API shape (multi-objective optimisation rather than per-group CIs).
