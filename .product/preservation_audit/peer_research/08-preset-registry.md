# Peer Research: Preset Registry & Default Configurations
> Generated 2026-02-26. Peer evidence for preservation audit item N-X07.

## Evidence Per Tool

### 1. lm-eval (EleutherAI lm-evaluation-harness)

**Does it ship named presets?** No named "preset" concept. Instead, it ships ~200 **task YAML configs** that each encode opinionated defaults for a specific evaluation (dataset, metrics, num_fewshot, prompt template, filters). Task groups (e.g. `mmlu`, `hellaswag`) bundle related sub-tasks with aggregate metrics. These are functionally presets, but the abstraction is "task" not "preset".

**Discovery mechanism:** Built-in task registry scanned from `lm_eval/tasks/` directory. Custom tasks via `--include_path`. List all tasks: `lm-eval --tasks list`.

**Zero-config experience:** There is no zero-config run. Both `--model` and `--tasks` are **required** arguments. Minimal invocation:
```
lm-eval --model hf --model_args pretrained=gpt2 --tasks hellaswag
```
The task YAML then provides all remaining defaults (num_fewshot=0, batch_size=1, dataset split, metric list, prompt template). The user never needs to specify evaluation parameters beyond model + task.

**Opinionated or minimal?** Opinionated per-task. Each task YAML encodes research-community-validated defaults (e.g. MMLU uses 5-shot, HellaSwag uses 0-shot). These are not arbitrary — they match published paper methodologies.

**Override individual values?** Yes. CLI flags override task YAML values: `--num_fewshot 5`, `--batch_size auto`. The task YAML is a default layer, not a ceiling.

**Key defaults:** `num_fewshot=0` (global default, overridden per-task YAML), `batch_size=1`, `repeats=1`.

> Source: [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), [task_guide.md](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md), [Architecture analysis](https://slyracoon23.github.io/blog/posts/2025-03-21_eleutherai-evaluation-methods.html)

---

### 2. optimum-benchmark (Hugging Face)

**Does it ship named presets?** No named presets in the llem sense. Ships **~87 composable YAML config files** in `tests/configs/`, split into underscore-prefixed base templates (`_cpu_.yaml`, `_pytorch_.yaml`, `_bert_.yaml`, `_inference_.yaml`) and concrete compositions (`cuda_inference_pytorch_bnb.yaml`). These are test fixtures, not user-facing presets, but they demonstrate the pattern.

**Discovery mechanism:** Hydra config groups. Users specify `--config-dir` and `--config-name` explicitly — there is no automatic registry or `list` command. Configs compose via Hydra's `defaults:` list.

**Zero-config experience:** None. Both `--config-dir` and `--config-name` are mandatory (Hydra requirement). No default config is loaded without explicit specification.

**Opinionated or minimal?** Opinionated at the dataclass level. The `InferenceConfig` dataclass ships production-reasonable defaults:

| Field | Default |
|-------|---------|
| `iterations` | 10 |
| `duration` | 10 (seconds) |
| `warmup_runs` | 10 |
| `latency` | True |
| `energy` | False |
| `memory` | False |

Users override these in YAML or via Hydra CLI overrides (`scenario.warmup_runs=20`).

**Override individual values?** Yes. Hydra CLI override syntax: `backend.model=gpt2 scenario.warmup_runs=5`. Any field in any config group is individually overridable.

> Source: [optimum-benchmark](https://github.com/huggingface/optimum-benchmark), [tests/configs](https://github.com/huggingface/optimum-benchmark/tree/main/tests/configs), [scenarios/inference/config.py](https://github.com/huggingface/optimum-benchmark/blob/main/optimum_benchmark/scenarios/inference/config.py)

---

### 3. vLLM bench

**Does it ship named presets?** No. All parameters are argparse flags with hardcoded defaults. No named configurations, no preset files, no config groups.

**Discovery mechanism:** N/A — there is nothing to discover. `vllm bench serve --help` lists all flags.

**Zero-config experience:** Functional with just a running server endpoint. Key defaults make the tool usable immediately:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `num_prompts` | 1000 | |
| `request_rate` | inf | All requests sent at t=0 (stress test) |
| `random_input_len` | 1024 | tokens |
| `random_output_len` | 128 | tokens |
| `dataset_name` | "random" | Synthetic data |
| `best_of` | 1 | |
| `seed` | 0 | Deterministic |
| `burstiness` | 1.0 | Poisson process |
| `percentile_metrics` | "ttft,tpot,itl" | |
| `metric_percentiles` | "99" | 99th percentile |

**Opinionated or minimal?** Opinionated. The defaults produce a meaningful stress-test benchmark (1000 prompts, infinite request rate, 1024 input tokens). A user running with zero overrides gets a real throughput measurement.

**Override individual values?** Yes, all via CLI flags. No layered config — purely flat argparse.

> Source: [vllm bench serve docs](https://docs.vllm.ai/en/v0.9.0/api/vllm/benchmarks/serve.html), [benchmark_serving.py](https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_serving.py)

---

### 4. Hydra (Meta / OmegaConf)

**Does it ship named presets?** Hydra is a framework, not a benchmark tool. It provides the **config group** primitive that other tools use to implement presets. A config group is a named directory containing YAML options (e.g. `db/mysql.yaml`, `db/postgresql.yaml`).

**Discovery mechanism:** File-system convention. Config groups are subdirectories under the config path. The `defaults:` list in the primary YAML selects which option from each group to load:
```yaml
defaults:
  - db: mysql        # loads db/mysql.yaml
  - server: apache   # loads server/apache.yaml
  - _self_
```

**Zero-config experience:** Hydra itself requires a config file. But the defaults list means a tool built on Hydra can have a fully specified default configuration that requires zero user input.

**Opinionated or minimal?** Depends on the tool author. Hydra provides the mechanism; the tool author decides whether defaults are opinionated or empty.

**Override individual values?** Yes — this is Hydra's core value proposition. Any field in any composed config is overridable via CLI: `db=postgresql db.timeout=20`. The `defaults:` list selects the base config; CLI overrides modify individual fields on top.

**Key pattern for presets:** A tool could ship `preset/quick-test.yaml`, `preset/benchmark.yaml` as a config group, then use `defaults: [preset: quick-test]` as the default, with users overriding via `preset=benchmark` or individual fields.

> Source: [Hydra defaults](https://hydra.cc/docs/tutorials/basic/your_first_app/defaults/), [config groups](https://hydra.cc/docs/tutorials/basic/your_first_app/config_groups/), [defaults list](https://hydra.cc/docs/advanced/defaults_list/)

---

### 5. MLPerf Inference

**Does it ship named presets?** Yes — **scenarios** are exactly named presets with fixed structural requirements. Four scenarios: `SingleStream`, `MultiStream`, `Server`, `Offline`. These are not user-definable; they are standardised by the MLCommons specification.

**Discovery mechanism:** Specification document + LoadGen library. Scenarios are enum values compiled into the LoadGen binary. The user selects a scenario; LoadGen enforces all constraints.

**Zero-config experience:** None at the tool level. The user must select a scenario and provide a model. But within a scenario, all parameters are prescribed:

| Scenario | Duration | Samples/Query | Metric | Latency Constraint |
|----------|----------|---------------|--------|-------------------|
| SingleStream | 600s | 1 | 90th-%-ile latency | None |
| MultiStream | 600s | 8 | 99th-%-ile latency | 99th-%-ile |
| Server | 600s | 1 | Max QPS at target latency | Benchmark-specific |
| Offline | 600s+ | min 24,576 | Throughput | None |

**Opinionated or minimal?** Maximally opinionated. The scenarios exist precisely to eliminate configuration variance across submissions. Duration, sample counts, latency percentiles, and metrics are all fixed by the specification.

**Override individual values?** No. The LoadGen must be compiled from the tagged approved revision without alteration. Users cannot change scenario parameters and still produce a valid submission.

> Source: [MLPerf inference rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc), [LoadGen settings](https://github.com/mlcommons/inference/blob/master/loadgen/test_settings_internal.cc), [MLCommons benchmarks](https://mlcommons.org/benchmarks/inference-datacenter/)

---

### 6. pytest

**Does it ship named presets?** No named presets. Ships **opinionated discovery defaults** baked into the tool (files matching `test_*.py`, classes matching `Test*`, functions matching `test_*`). Project-level configuration via `pytest.ini`, `pyproject.toml [tool.pytest.ini_options]`, or `tox.ini`.

**Discovery mechanism:** File-system hierarchy search. pytest searches upward from invocation directory for config files (`pytest.toml`, `pytest.ini`, `pyproject.toml`, `setup.cfg`). First match wins.

**Zero-config experience:** Excellent. Running bare `pytest` with no config file:
- Discovers `test_*.py` files recursively from the current directory
- Runs all `test_*` functions and `Test*` classes
- Reports pass/fail with default verbosity
- No configuration file needed whatsoever

**Opinionated or minimal?** Opinionated discovery conventions (the `test_` prefix pattern), minimal execution defaults. The `addopts` mechanism in config files lets projects bake in default flags (e.g. `addopts = "-ra -q"` makes every `pytest` invocation use those flags).

**Override individual values?** Yes. CLI flags override config file values. Config file values override built-in defaults. Three-layer precedence: built-in < config file < CLI.

**Key analogy for our tool:** `addopts` is the pattern — sensible defaults baked into the tool, overridable per-project via config, overridable per-invocation via CLI.

> Source: [pytest configuration](https://docs.pytest.org/en/stable/reference/customize.html)

---

### 7. Docker Compose (profiles)

**Does it ship named presets?** Profiles are a **service grouping mechanism**, not configuration presets. A profile is a tag on services; activating a profile starts tagged services alongside always-on services.

**Discovery mechanism:** Defined inline in `compose.yaml` via the `profiles:` attribute on each service. No registry, no separate files. Profile names follow `[a-zA-Z0-9][a-zA-Z0-9_.-]+`.

**Zero-config experience:** Services without `profiles:` attribute always start. `docker compose up` starts all untagged services. Core services should never be tagged with a profile.

**Opinionated or minimal?** The pattern is opinionated about one thing: **core services have no profile (always on), optional services are profile-gated**. This maps well to "default config always works; presets add/modify on top".

**Override individual values?** Profiles are binary (on/off). You cannot partially apply a profile. But you can combine profiles (`--profile frontend --profile debug`).

**Key analogy for our tool:** The Docker Compose pattern of "untagged = always on, tagged = opt-in" maps to "Pydantic defaults = always active, presets = override layer".

> Source: [Docker Compose profiles](https://docs.docker.com/compose/how-tos/profiles/), [Profiles reference](https://docs.docker.com/reference/compose-file/profiles/)

---

### 8. llmperf (Ray)

**Does it ship named presets?** No. All configuration via argparse flags with hardcoded defaults. No named presets, no config files, no profiles.

**Discovery mechanism:** N/A. `--help` is the only discovery mechanism.

**Zero-config experience:** Requires `--model` only (and appropriate API key env var). Defaults produce a meaningful benchmark:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `mean_input_tokens` | 550 | |
| `stddev_input_tokens` | 150 | |
| `mean_output_tokens` | 150 | |
| `stddev_output_tokens` | 80 | |
| `num_concurrent_requests` | 10 | |
| `max_num_completed_requests` | 10 | |
| `timeout` | 90s | |
| `llm_api` | "openai" | |

**Opinionated or minimal?** Opinionated. The defaults (550 input tokens, 150 output, 10 concurrent requests) represent a realistic API benchmarking scenario. A user providing only `--model` gets a meaningful result.

**Override individual values?** Yes, all via CLI flags. Flat argparse, no layering.

> Source: [llmperf](https://github.com/ray-project/llmperf), [token_benchmark_ray.py](https://github.com/ray-project/llmperf/blob/main/token_benchmark_ray.py)

---

### Supplementary: HuggingFace inference-benchmarker

Noteworthy because it is the only tool found that ships **named benchmark profiles** resembling our preset concept:

**Profiles:** `chat`, `code-generation`, `classification`, `fixed-length` — each encoding a workload pattern with specific prompt distributions.

**Modes:** `sweep` (default, auto-detects max throughput), `rate` (fixed QPS), `throughput` (constant virtual users).

This is the closest peer analogy to our v1.x preset registry.

> Source: [inference-benchmarker](https://github.com/huggingface/inference-benchmarker)

---

## Summary Table

| Tool | Named Presets? | Discovery | Zero-Config? | Opinionated Defaults? | Override Fields? |
|------|---------------|-----------|-------------|----------------------|-----------------|
| **lm-eval** | No (task YAMLs serve as presets) | Task registry directory | No (`--model` + `--tasks` required) | Yes (per-task) | Yes (CLI flags) |
| **optimum-benchmark** | No (Hydra config files) | Hydra config groups | No (config-dir + config-name required) | Yes (dataclass defaults) | Yes (Hydra CLI) |
| **vLLM bench** | No | `--help` | Near-zero (server endpoint only) | Yes (1000 prompts, inf rate) | Yes (CLI flags) |
| **Hydra** | Framework provides config groups | File-system directories | Depends on tool author | Depends on tool author | Yes (core feature) |
| **MLPerf** | Yes (4 fixed scenarios) | Specification + LoadGen enum | No (scenario + model required) | Maximally (all params fixed) | No (spec-locked) |
| **pytest** | No (convention-based defaults) | Config file hierarchy search | Yes (bare `pytest` works) | Yes (discovery conventions) | Yes (3-layer precedence) |
| **Docker Compose** | Profiles (service groups, not configs) | Inline `profiles:` attribute | Yes (untagged services always start) | Yes (core = always-on pattern) | No (profiles are binary) |
| **llmperf** | No | `--help` | Near-zero (`--model` only) | Yes (550/150 tokens, 10 concurrent) | Yes (CLI flags) |
| **HF inference-benchmarker** | Yes (4 named profiles) | Built-in | Near-zero | Yes (workload-specific) | Yes (CLI flags) |

---

## Recommendation

### Pattern observed across peers

**No peer tool uses a named preset registry like our v1.x `PRESETS` dict.** The universal pattern is one of three approaches:

1. **Opinionated dataclass/argparse defaults** (vLLM bench, llmperf, optimum-benchmark) — the tool has an opinion baked into field defaults. Zero-config works because every field has a sensible default. No registry, no names, no `_meta`.

2. **Task/config YAML files** (lm-eval, optimum-benchmark via Hydra) — each "preset" is a full YAML file in a known directory. Discovery is file-system-based. Composition and override is structural (Hydra merge, YAML inheritance).

3. **Fixed scenarios** (MLPerf) — a small number of named, spec-locked configurations for reproducibility. Users cannot modify individual fields.

### What this means for v2.0

Our v1.x preset registry (10 named dicts in `constants.py` with `_meta`) combines patterns 1 and 2 in a non-standard way. The `_meta` self-documentation pattern is novel — no peer does this.

**The peer-validated path for v2.0 is approach (1): opinionated Pydantic defaults.**

- `ExperimentConfig` fields already have defaults via Pydantic. If those defaults are *good* (research-validated values for warmup, batch size, token counts, precision), then `llem run --model X` works with zero additional configuration. The Pydantic model *is* the preset.
- Backend-specific presets (`vllm-throughput`, `vllm-speculative`, etc.) encode backend-tuning knowledge. This knowledge should live in **backend config section defaults** or as **example YAML files** in a `configs/examples/` directory — not as an in-code registry. This matches the optimum-benchmark and lm-eval patterns.
- The `quick-test` / `benchmark` / `throughput` distinction maps to different **default value sets**. If needed, these could be YAML files rather than Python dicts (discoverable via file system, editable by users, composable via YAML merge). But the simplest approach is: make the Pydantic defaults the "benchmark" preset (the common case), and let `quick-test` be a shipped example YAML.

**Concrete recommendation:**

1. **Drop the in-code `PRESETS` registry.** Replace with opinionated Pydantic defaults on `ExperimentConfig` that produce a meaningful benchmark run with just `--model`.
2. **Ship 2-3 example YAML files** in `configs/examples/` (e.g. `quick-test.yaml`, `throughput.yaml`, `low-latency.yaml`) for common alternative configurations.
3. **Drop `_meta` / `get_preset_metadata()`.** No peer tool self-documents presets this way. Example YAMLs can have comments. `llem run --help` describes the defaults.
4. **Backend-specific tuning knowledge** (prefix caching, chunked prefill, enforce_eager) belongs in backend config section documentation and example YAMLs, not in a preset registry.
5. **Keep the timeout/constant values** (`DEFAULT_WARMUP_RUNS`, `GRACEFUL_SHUTDOWN_TIMEOUT_SEC`, etc.) — these are not presets, they are operational constants. Every peer tool has these as hardcoded defaults or dataclass field defaults.
