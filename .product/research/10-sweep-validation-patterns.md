# Sweep Validation Patterns for ML Experiment Grids

**Researched:** 2026-02-18
**Confidence:** MEDIUM — core framework APIs verified via official docs; practical "industry approach" section is MEDIUM/LOW, as published research group workflows are not well-documented publicly.

---

## Executive Summary

There is no sweep framework that natively enforces pre-flight parameter validity checks before GPU allocation. The ecosystem falls into two categories: (1) **constraint-aware HPO frameworks** (Optuna, ConfigSpace/SMAC, Ax) that handle infeasibility post-launch but with intelligent sampling that avoids repeating failures, and (2) **experiment orchestrators** (Hydra, W&B Sweeps, Ray Tune) that offer limited or no constraint syntax, requiring external pre-filtering or run-and-fail strategies.

For LLenergyMeasure's use case — filtering `(backend=tensorrt, precision=bfloat16)` before wasting GPU time — the practical answer is a **pre-flight validator layer written in the tool itself**, not a third-party sweep framework feature. ConfigSpace provides the cleanest declarative syntax for expressing forbidden combinations, but is academic tooling, not production-grade for general experiment grids.

---

## 1. Hydra Sweeps

### What Hydra Provides

Hydra's built-in sweeper does grid expansion over override syntax with no constraint awareness. Given:

```yaml
# conf/sweeper.yaml
hydra/sweeper: basic
hydra:
  sweeper:
    params:
      backend: pytorch,vllm,tensorrt
      precision: fp32,fp16,bf16,int8,int4
```

This generates every cross-product combination (15 configs) and launches them all. **Hydra has no built-in mechanism to declare "if backend=tensorrt then precision cannot be bf16".** Invalid combinations launch, fail, and mark the run as failed.

### Runtime Failure Handling

Failed runs are logged under `outputs/multirun/<date>/<time>/<run_id>/` with their full traceback. Hydra does not retry or adapt. The sweep continues to the next config regardless of how the previous one failed.

### Third-Party Filter: `hydra-filter-sweeper`

A community plugin ([PyPI: hydra-filter-sweeper](https://pypi.org/project/hydra-filter-sweeper/)) extends the basic sweeper with pre-generation filters. Configs are filtered **before** launch using Python expression evaluation against the config dict.

```yaml
defaults:
  - override hydra/sweeper: filter

hydra:
  sweeper:
    filters:
      - hydra_filter_sweeper.Expression:
          expr: "backend == 'tensorrt' and precision == 'bf16'"
      - hydra_filter_sweeper.Expression:
          expr: "backend == 'pytorch' and precision in ['int8', 'int4']"
      - hydra_filter_sweeper.Exists:
          path: "outputs/${backend}_${precision}/done.flag"
```

Any config where the expression evaluates to `True` is **excluded**. Multiple filters are OR'd — any match excludes the config.

**Assessment:** This is the closest thing to declarative constraint syntax in Hydra. The expression syntax is readable and maps directly to the LLenergyMeasure problem. Plugin is community-maintained with low bus factor. It works at config-generation time, so excluded configs never touch the GPU.

### Hydra Ax Sweeper

The official `hydra-ax-sweeper` plugin wraps Meta's Ax platform (Bayesian optimisation). It supports `parameter_constraints` at the Ax level (linear constraints on continuous parameters), but not categorical exclusion rules like "if backend=tensorrt exclude precision=bf16". Ax's constraint system is designed for numeric constraints, not conditional categorical logic.

**Sources:**
- [Hydra Multi-run](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/)
- [Hydra Filter Sweeper on PyPI](https://pypi.org/project/hydra-filter-sweeper/)
- [Ax Sweeper Plugin](https://hydra.cc/docs/plugins/ax_sweeper/)

---

## 2. Weights & Biases Sweeps

### Pre-Validation: None

W&B Sweeps has **no conditional parameter syntax** and no pre-validation of parameter combinations. This is a documented and frequently requested missing feature.

From official issue tracking (GitHub wandb/wandb issues #1487, #1863): conditional sweep configs are not supported. The W&B YAML config spec allows `values`, `min`/`max`, `distribution`, and `probabilities` per parameter, but parameters are fully independent. There is no syntax to express "parameter B's valid values depend on parameter A's chosen value."

### What W&B Does Instead

**Post-hoc failure collection.** Invalid configs launch, fail, log `wandb.finish(exit_code=1)`, and appear in the run table as failed. W&B's UI lets you filter failed runs and inspect their parameter combinations, making it easy to see which combos are systematically failing. The sweep controller (Bayesian or random) does not learn from constraint violations — it treats failed runs as having no metric value and ignores them in acquisition function updates.

**Early termination** (`early_terminate: hyperband`) is for stopping unpromising runs based on intermediate metric values — it is not constraint-based and does not apply here.

### Practical Workaround Pattern Used in Practice

```python
# At the start of every W&B sweep agent run:
import wandb

run = wandb.init()
config = wandb.config

# Pre-validate before any GPU allocation
if config.backend == "tensorrt" and config.precision not in ["fp16", "int8", "int4"]:
    wandb.log({"status": "skipped", "reason": "invalid_combination"})
    wandb.finish(exit_code=0)  # exit cleanly so sweep controller counts it
    exit(0)

# Proceed with actual experiment...
```

This is the documented community pattern — early exit with clean status before GPU is touched. The sweep controller logs the config as "finished" (not failed), and the combination is naturally de-prioritised in Bayesian sweeps because it has no metric signal.

**Sources:**
- [W&B Sweep Config Keys](https://docs.wandb.ai/models/sweeps/sweep-config-keys)
- [Conditional sweep config — W&B Community](https://community.wandb.ai/t/conditional-sweep-config/4017)
- [GitHub Issue: Sweeps conditional parameters #1487](https://github.com/wandb/wandb/issues/1487)

---

## 3. Optuna

### Constraint Handling: Post-Sampling, Pre-Objective

Optuna provides two distinct mechanisms:

#### 3a. Raise `TrialPruned` for Invalid Combinations

The standard pattern for excluding invalid combos is to raise at the start of the objective function, before any training:

```python
import optuna

def objective(trial):
    backend = trial.suggest_categorical("backend", ["pytorch", "vllm", "tensorrt"])
    precision = trial.suggest_categorical("precision", ["fp32", "fp16", "bf16", "int8", "int4"])

    # Constraint: TensorRT does not support bf16
    if backend == "tensorrt" and precision == "bf16":
        raise optuna.TrialPruned()

    # Constraint: PyTorch eager mode has no int4 support
    if backend == "pytorch" and precision == "int4":
        raise optuna.TrialPruned()

    # ... actual experiment
    return energy_joules
```

Pruned trials are stored with `PRUNED` state, not `FAILED`. The sampler (TPE, NSGA-II, etc.) learns to avoid pruned regions — this is the key difference from W&B's run-and-fail approach. After several pruned trials in a region, the sampler stops proposing configurations from that region.

**Important caveat:** This learning is sampler-dependent and requires enough pruned trials to shape the surrogate model. For small grids (< 50 trials), the sampler may not have enough signal. For large Bayesian sweeps over continuous spaces, this is highly effective.

#### 3b. `constraints_func` for Soft Constraints (Multi-Objective Only)

For the `NSGAIISampler` and `TPESampler`, a `constraints_func` can be registered at study creation time:

```python
def constraints_func(trial):
    # Return sequence of floats. Values > 0 = violated.
    violations = []
    backend = trial.user_attrs.get("backend")
    precision = trial.user_attrs.get("precision")

    if backend == "tensorrt" and precision == "bf16":
        violations.append(1.0)  # violated
    else:
        violations.append(-1.0)  # feasible

    return violations

sampler = optuna.samplers.NSGAIISampler(constraints_func=constraints_func)
study = optuna.create_study(sampler=sampler, direction="minimize")
```

**Limitation:** `constraints_func` is called *after* the trial runs — it receives a `FrozenTrial` with results, not during sampling. This means the GPU was still used for the infeasible trial. This API is designed for black-box constraints (e.g., energy budget exceeded) not for known-invalid categorical combinations. Use `raise TrialPruned()` instead for pre-flight invalidity.

#### 3c. Conditional Parameter Suggestion

For hierarchical/conditional spaces, Optuna supports suggesting parameters conditionally:

```python
def objective(trial):
    backend = trial.suggest_categorical("backend", ["pytorch", "vllm", "tensorrt"])

    # Only suggest precision values valid for this backend
    if backend == "tensorrt":
        precision = trial.suggest_categorical("precision", ["fp16", "int8", "int4"])
    elif backend == "vllm":
        precision = trial.suggest_categorical("precision", ["fp16", "bf16", "int8"])
    else:  # pytorch
        precision = trial.suggest_categorical("precision", ["fp32", "fp16", "bf16"])
```

This is the **most efficient pattern** — invalid combinations are never even sampled. However, it requires hard-coding constraint logic inside the objective function, which is less maintainable as the grid grows.

**Sources:**
- [Optuna 4.7.0 Trial Docs](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html)
- [Optuna FAQ](https://optuna.readthedocs.io/en/stable/faq.html)
- [Optuna NSGAIISampler](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.NSGAIISampler.html)

---

## 4. Framework with Native Constraint Declaration: ConfigSpace

**ConfigSpace** (from AutoML group at Freiburg, used by SMAC3, Auto-sklearn, etc.) is the most principled framework for declaring forbidden parameter combinations. It is not a sweep runner — it is a **search space DSL** that generates valid configurations.

### ForbiddenClause API

```python
from ConfigSpace import (
    ConfigurationSpace,
    CategoricalHyperparameter,
    ForbiddenEqualsClause,
    ForbiddenInClause,
    ForbiddenAndConjunction,
    EqualsCondition,
)

cs = ConfigurationSpace(seed=42)

backend = CategoricalHyperparameter("backend", ["pytorch", "vllm", "tensorrt"])
precision = CategoricalHyperparameter(
    "precision", ["fp32", "fp16", "bf16", "int8", "int4"]
)
batch_size = CategoricalHyperparameter("batch_size", [1, 4, 8, 16, 32])

cs.add([backend, precision, batch_size])

# Declare forbidden combinations
# TensorRT does not support fp32 or bf16
forbidden_trt_fp32 = ForbiddenAndConjunction(
    ForbiddenEqualsClause(backend, "tensorrt"),
    ForbiddenEqualsClause(precision, "fp32"),
)
forbidden_trt_bf16 = ForbiddenAndConjunction(
    ForbiddenEqualsClause(backend, "tensorrt"),
    ForbiddenEqualsClause(precision, "bf16"),
)
# PyTorch eager has no int4 support
forbidden_pytorch_int4 = ForbiddenAndConjunction(
    ForbiddenEqualsClause(backend, "pytorch"),
    ForbiddenEqualsClause(precision, "int4"),
)

cs.add([forbidden_trt_fp32, forbidden_trt_bf16, forbidden_pytorch_int4])

# Generate valid configs — ConfigSpace guarantees no forbidden combo is returned
configs = cs.sample_configuration(n=100)
# All 100 configs are guaranteed constraint-compliant
```

### Key Properties

- **Sampling always returns valid configs** — forbidden combinations are rejected at sample time, not run-time
- Supports `ForbiddenEqualsClause`, `ForbiddenInClause`, `ForbiddenAndConjunction`
- Supports conditional hyperparameters (child param only active if parent equals X)
- Used by SMAC3, Auto-sklearn, HyperSweeper (Hydra plugin)
- Can enumerate all valid grid configs (for fixed sweep), or sample (for Bayesian search)

### Limitation for LLenergyMeasure

ConfigSpace is research-oriented and its documentation/API is geared towards AutoML use cases. It works well as a constraint-declaration layer but adds a dependency on a niche library. There is no native integration with vLLM, TensorRT, or energy measurement — constraints must be manually coded from domain knowledge.

**Sources:**
- [ConfigSpace Forbidden Clauses](https://automl.github.io/ConfigSpace/latest/reference/forbiddens/)
- [ConfigSpace Conditions](https://automl.github.io/ConfigSpace/latest/api/ConfigSpace/conditions/)
- [ConfigSpace Guide](https://automl.github.io/ConfigSpace/latest/guide/)
- [GitHub: automl/ConfigSpace](https://github.com/automl/ConfigSpace)

---

## 5. VRAM Estimation for LLM Experiments

### Estimation Formulae (verified, MEDIUM confidence)

**Weight memory:**
```
weight_vram_gb = (num_parameters * bytes_per_dtype) / 1e9
```

Bytes per dtype: fp32=4, fp16=2, bf16=2, int8=1, int4=0.5

**KV cache memory (approximation):**
```
kv_cache_gb = (2 * num_layers * hidden_size * seq_len * batch_size * bytes_per_dtype) / 1e9
```

**Total estimate:**
```
total_vram_gb = weight_vram_gb + kv_cache_gb + activation_overhead
```
Where `activation_overhead` is typically +10-20% of weight_vram_gb.

**Example:** Llama-3 70B at fp16, batch=1, seq_len=2048
- Weight VRAM: 70e9 * 2 / 1e9 = 140 GB
- KV cache: 2 * 80 * 8192 * 2048 * 1 * 2 / 1e9 approx 4.3 GB
- Overhead: ~14 GB
- Total estimate: ~158 GB (requires 2x H100 80GB)

### Tools and Libraries

#### HuggingFace Accelerate `estimate-memory`

**Confidence: HIGH** — This is the most useful publicly-available API for pre-flight VRAM checking.

```bash
# CLI
accelerate estimate-memory meta-llama/Llama-3-70b --dtypes float16 int8
```

```python
# Python API (requires accelerate >= 0.28)
from accelerate.commands.estimate import create_empty_model
from accelerate import init_empty_weights
from accelerate.utils import compute_module_sizes

with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(model_id)

sizes = compute_module_sizes(model, dtype="float16")
total_gb = sizes[""] / 1e9
```

**Caveats:**
- Estimates weight memory only — does not account for KV cache or activations
- Accurate to within a few percent for weight loading
- Does not integrate with vLLM or TensorRT (transformers models only)
- Known issue: estimate-memory underestimates for some 70B+ models (GitHub issue #3379)

#### vLLM Internal Profiling (not externally exposed)

vLLM runs an internal `profile_run` during engine initialisation to determine how much VRAM remains after loading weights, then sizes the KV cache accordingly. This is **not an externally-callable API** — it is part of vLLM's `LLMEngine.__init__()` flow. The only way to access it is to instantiate a full vLLM LLM object, which loads the model into GPU memory. This defeats the purpose of pre-flight checking.

vLLM does log memory stats to stdout:
```
INFO:    # GPU blocks: 2867, # CPU blocks: 2048
INFO:    Maximum concurrency for 32768 tokens per request: 5.59x
```

This is not a programmatic API — there is no external Python callable for vLLM's VRAM estimation.

#### Web-Based Calculators (LOW confidence for programmatic use)

- [vram.asmirnov.xyz](https://vram.asmirnov.xyz/) — formula-based, no KV cache modelling
- [HuggingFace Space: NyxKrage/LLM-Model-VRAM-Calculator](https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator) — web UI only
- [apxml.com/tools/vram-calculator](https://apxml.com/tools/vram-calculator) — web UI, includes KV cache

None expose a Python API.

#### Roll-Your-Own Pre-flight Check Pattern

For LLenergyMeasure specifically, the most practical approach is a purpose-built validator:

```python
from dataclasses import dataclass

DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "int4": 0.5}


@dataclass
class VRAMEstimate:
    weight_gb: float
    kv_cache_gb: float
    overhead_gb: float

    @property
    def total_gb(self) -> float:
        return self.weight_gb + self.kv_cache_gb + self.overhead_gb


def estimate_vram(
    num_params: int,
    num_layers: int,
    hidden_size: int,
    precision: str,
    seq_len: int,
    batch_size: int,
    overhead_factor: float = 0.15,
) -> VRAMEstimate:
    dtype_bytes = DTYPE_BYTES[precision]
    weight_gb = (num_params * dtype_bytes) / 1e9
    kv_cache_gb = (2 * num_layers * hidden_size * seq_len * batch_size * dtype_bytes) / 1e9
    overhead_gb = weight_gb * overhead_factor
    return VRAMEstimate(weight_gb, kv_cache_gb, overhead_gb)


def will_fit(estimate: VRAMEstimate, available_vram_gb: float) -> bool:
    return estimate.total_gb < available_vram_gb * 0.90  # 10% safety margin
```

This is sufficient for pre-flight grid filtering. The key insight from vLLM's approach: allocating 90-95% of available VRAM is the practical upper bound.

**Sources:**
- [HuggingFace Accelerate: Model Memory Estimator](https://huggingface.co/docs/accelerate/en/usage_guides/model_size_estimator)
- [vLLM: Conserving Memory](https://docs.vllm.ai/en/latest/configuration/conserving_memory/)
- [vLLM: GPU Memory Calculation](https://docs.vllm.ai/projects/vllm-omni/en/latest/configuration/gpu_memory_utilization/)
- [How vLLM profile_run works — Discussion #10110](https://github.com/vllm-project/vllm/discussions/10110)
- [Accelerate estimate-memory issue #2980](https://github.com/huggingface/accelerate/issues/2980)
- [Accelerate estimate-memory issue #3379](https://github.com/huggingface/accelerate/issues/3379)

---

## 6. Practical Industry Approach for 1000+ Experiment Sweeps

**Confidence: MEDIUM** — Based on community discussions, open-source tooling patterns, and GitHub issue threads. Published research groups do not typically document their exact sweep management workflows.

### What Actually Happens

Research at scale (1000+ configurations) almost universally uses a combination of:

#### 6a. Offline Pre-filtering / Grid Curation

The most common documented approach in research codebases: generate the full cross-product first, then **manually curate or programmatically filter** the list before queuing. This is done in a pre-processing script, not inside the sweep framework.

```python
# generate_sweep_configs.py
import itertools

ALL_CONFIGS = list(itertools.product(
    ["pytorch", "vllm", "tensorrt"],
    ["fp32", "fp16", "bf16", "int8", "int4"],
    [1, 4, 8, 16, 32],
))


def is_valid(backend, precision, batch_size):
    # Hard backend constraints
    if backend == "tensorrt" and precision in ["fp32", "bf16"]:
        return False
    if backend == "pytorch" and precision in ["int8", "int4"]:
        return False
    # VRAM pre-check
    est = estimate_vram(model_params, num_layers, hidden_size, precision, 2048, batch_size)
    if not will_fit(est, available_gpu_vram_gb):
        return False
    return True


valid_configs = [c for c in ALL_CONFIGS if is_valid(*c)]
# Queue valid_configs to SLURM / Ray / W&B
```

This pattern appears in Hydra-based research repos (e.g., facebookresearch/how-to-autorl), SLURM job arrays, and W&B sweep scripts.

#### 6b. Run-and-Collect-Failures (the W&B standard)

For Bayesian sweeps over continuous spaces where the constraint boundary is unknown a priori, the approach is: run everything, mark failures, post-hoc analysis to identify the constraint boundary. This is explicitly what W&B's sweep dashboard is built for.

**When this is acceptable:**
- Bayesian search (sampler naturally avoids bad regions after a few failures)
- Cheap experiments (few minutes per trial, CPU-only, etc.)
- Unknown constraint boundaries

**When this is not acceptable:**
- Grid sweeps (every combination runs once — no sampler to "learn")
- Expensive experiments (1+ hour GPU time per trial)
- Known constraint boundaries (categorical combos based on documented backend limits)

For LLenergyMeasure, experiments are expensive and the constraints are known. Run-and-collect is not appropriate.

#### 6c. Defensive Early Exit Pattern (universal)

Regardless of how the sweep is managed, serious ML codebases add a validation guard at the very start of each experiment run — before any GPU initialisation:

```python
def validate_experiment_config(config: ExperimentConfig) -> None:
    """Raise ConfigurationError for known-invalid combos. Runs before GPU init."""

    PRECISION_SUPPORT = {
        "tensorrt": {"fp16", "int8", "int4"},
        "vllm": {"fp16", "bf16", "int8", "awq", "gptq"},
        "pytorch": {"fp32", "fp16", "bf16"},
    }

    supported = PRECISION_SUPPORT.get(config.backend, set())
    if config.precision not in supported:
        raise ConfigurationError(
            f"Backend '{config.backend}' does not support precision '{config.precision}'. "
            f"Supported: {sorted(supported)}"
        )
```

This is the "defence in depth" layer — even if pre-filtering has bugs, individual runs fail fast and informatively rather than after wasting GPU time loading a model.

#### 6d. SLURM-Based Skip Pattern

In HPC environments with SLURM job arrays, a common pattern is to generate a manifest of all valid configs, write them to a JSON/YAML file, and submit array jobs that index into the manifest. Invalid configs are simply not written to the manifest.

```bash
# job_array.sh
#SBATCH --array=0-149  # exactly len(valid_configs)

CONFIG_IDX=$SLURM_ARRAY_TASK_ID
python run_experiment.py --config-idx $CONFIG_IDX --config-file valid_configs.json
```

This avoids framework-level constraint support entirely — the constraint is applied at manifest-generation time.

---

## Summary: Design Implications for LLenergyMeasure

### The Core Problem Decomposed

The sweep validation problem splits into three independent concerns:

| Concern | Responsibility | Recommended Mechanism |
|---------|---------------|-----------------------|
| Known categorical invalidity | Tool pre-flight validator | `validate_experiment_config()`, runs before GPU init |
| VRAM insufficiency | Tool pre-flight estimator | `estimate_vram()`, formula-based, fast |
| Runtime failures (unexpected) | Sweep orchestration layer | Run fails, logged, skipped in aggregation |

### Key Recommendations

1. **Do not depend on sweep framework constraint features.** No major framework (W&B, Hydra basic, Ray Tune) provides declarative categorical constraint syntax. ConfigSpace does but it is niche academic tooling.

2. **Build a `validate_experiment_config()` function in the tool.** This is the standard industry pattern. It runs before any GPU allocation, raises a clear error, and the caller (CLI, sweep loop, etc.) skips the config. Approximately 50 lines of Python — not worth a ConfigSpace dependency.

3. **Build a `estimate_vram()` utility.** Using the formulae above, this gives a fast pre-flight VRAM check. The HuggingFace Accelerate CLI is useful for one-off checks but does not expose a clean API for programmatic grid filtering. vLLM has no external estimator API.

4. **Consider `hydra-filter-sweeper` if using Hydra as the runner.** It provides YAML-declarable expression filters that map naturally to the constraint syntax needed. However, it is community-maintained with limited long-term guarantees.

5. **Use Optuna's conditional suggestion pattern** (not `constraints_func`) for Bayesian optimisation. Suggest precision values conditionally based on backend — this eliminates invalid combos from the sampler's space entirely without touching the GPU.

6. **The SLURM manifest pattern is the most robust for large grids.** Generate valid configs offline, write manifest, submit array jobs indexed into manifest. No sweep framework dependency, no runtime waste.

---

## Sources Index

- [Hydra Multi-run Documentation](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/)
- [Hydra Filter Sweeper — PyPI](https://pypi.org/project/hydra-filter-sweeper/)
- [Hydra Ax Sweeper Plugin](https://hydra.cc/docs/plugins/ax_sweeper/)
- [W&B Sweep Config Keys](https://docs.wandb.ai/models/sweeps/sweep-config-keys)
- [W&B GitHub: Conditional parameters #1487](https://github.com/wandb/wandb/issues/1487)
- [W&B Community: Conditional sweep config](https://community.wandb.ai/t/conditional-sweep-config/4017)
- [Optuna 4.7.0 Documentation](https://optuna.readthedocs.io/en/stable/)
- [Optuna NSGAIISampler](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.NSGAIISampler.html)
- [ConfigSpace — Forbidden Clauses](https://automl.github.io/ConfigSpace/latest/reference/forbiddens/)
- [ConfigSpace — Conditions](https://automl.github.io/ConfigSpace/latest/api/ConfigSpace/conditions/)
- [GitHub: automl/ConfigSpace](https://github.com/automl/ConfigSpace)
- [HuggingFace Accelerate: Model Memory Estimator](https://huggingface.co/docs/accelerate/en/usage_guides/model_size_estimator)
- [vLLM: Conserving Memory](https://docs.vllm.ai/en/latest/configuration/conserving_memory/)
- [vLLM profile_run discussion](https://github.com/vllm-project/vllm/discussions/10110)
- [Ray Tune Search Spaces](https://docs.ray.io/en/latest/tune/tutorials/tune-search-spaces.html)
- [GitHub: facebookresearch/how-to-autorl](https://github.com/facebookresearch/how-to-autorl)
