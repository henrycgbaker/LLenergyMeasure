# Automated Codebase Analysis Report

**Date**: 2026-02-05
**Tool**: llenergymeasure
**Analysis Tools**: vulture 2.14, deadcode 2.4.1, ruff (C901 complexity)

---

## Executive Summary

Automated analysis reveals significant dead code accumulation and complexity hotspots. Industry comparison shows our CLI surface is 3-4x larger than comparable tools, with substantial feature overlap and over-engineering in campaign/grid orchestration.

**Key Findings**:
- **287 dead code findings** from vulture (60%+ confidence)
- **274 dead code findings** from deadcode
- **32 functions exceeding complexity threshold** (C901 > 10)
- **13 total CLI commands** vs industry norm of 2-4
- **0 unused imports** (ruff F401)
- **Stub/TODO count**: 3 TODO comments, 30+ pass statements (mostly error handling), 35+ ellipsis (protocols)

---

## 1. Dead Code Detection Results

### 1.1 Vulture Findings (287 items, 60%+ confidence)

Both vulture and deadcode identified similar patterns. Cross-referencing shows high agreement (>95%), indicating these are genuine dead code candidates.

#### CLI Module (cli/)

| File | Finding | Type | Confidence | Severity | Notes |
|------|---------|------|------------|----------|-------|
| cli/config.py | `config_list` | function | 60% | **INVESTIGATE** | Listed in CLI help, should be used |
| cli/config.py | `config_callback` | function | 60% | **INVESTIGATE** | Typer callback pattern |
| cli/config.py | `config_validate` | function | 60% | **INVESTIGATE** | Listed in CLI help |
| cli/config.py | `config_show` | function | 60% | **INVESTIGATE** | Listed in CLI help |
| cli/config.py | `config_new` | function | 60% | **INVESTIGATE** | Listed in CLI help |
| cli/config.py | `config_generate_grid` | function | 60% | **INVESTIGATE** | Listed in CLI help |
| cli/results.py | `results_callback` | function | 60% | **INVESTIGATE** | Typer callback |
| cli/results.py | `results_list` | function | 60% | **INVESTIGATE** | Listed in CLI help |
| cli/results.py | `results_show` | function | 60% | **INVESTIGATE** | Listed in CLI help |
| cli/display/results.py | `show_parameter_provenance` | function | 60% | REMOVE | Utility function, unused |
| cli/display/summaries.py | `display_non_default_summary` | function | 60% | REMOVE | Display variant, unused |
| cli/experiment.py | `frame`, `signum` | variables | 100% | SIMPLIFY | Signal handler args not used |
| cli/lifecycle.py | `frame`, `signum` | variables | 100% | SIMPLIFY | Signal handler args not used |
| cli/schedule.py | `frame`, `signum` | variables | 100% | SIMPLIFY | Signal handler args not used |

**Analysis**: The CLI functions flagged as "unused" are actually registered via Typer decorators (e.g., `@config_app.command("list")`). Vulture doesn't detect decorator-based registration. These are **FALSE POSITIVES** for actual dead code, but indicate **registration may not be wired correctly** if they're truly unreachable.

**Action Required**: Manual verification that Typer apps are registered in main CLI. If registration is correct, these are false positives. If not registered, they're genuinely dead.

#### Config Module (config/)

| File | Finding | Type | Confidence | Severity | Notes |
|------|---------|------|------------|----------|-------|
| backend_configs.py | `validate_quantization` | method | 60% | REMOVE | Pydantic validator, never called |
| backend_configs.py | `flash_version`, `multiple_profiles`, `enable_chunked_context`, `enable_kv_cache_reuse`, `max_num_tokens`, `num_draft_tokens`, `extra_build_args`, `extra_runtime_args` | fields | 60% | **REMOVE** | TensorRT fields, never accessed |
| backend_detection.py | `get_available_backends` | function | 60% | REMOVE | Detection utility, unused |
| campaign_config.py | `validate_time_format` (2x) | methods | 60% | REMOVE | Validators never triggered |
| campaign_config.py | `validate_backends` | method | 60% | REMOVE | Validator never triggered |
| campaign_config.py | `gpu_memory_threshold_pct`, `restart_on_failure`, `max_restarts`, `restart_container` | fields | 60% | **REMOVE** | Health check fields, never read |
| campaign_config.py | `validate_interval`, `validate_total_duration` | methods | 60% | REMOVE | Schedule validators unused |
| campaign_config.py | `configs_dir` | field | 60% | REMOVE | Path field, never accessed |
| campaign_config.py | `validate_config_paths`, `validate_experiment_sources`, `get_config_names` | methods | 60% | REMOVE | CampaignConfig methods unused |
| introspection.py | `get_mutual_exclusions`, `get_backend_specific_params`, `get_special_test_models`, `get_params_requiring_gpu_capability`, `get_streaming_constraints`, `get_streaming_incompatible_tests`, `get_capability_matrix_markdown` | functions | 60% | **REMOVE** | SSOT introspection functions, never called |
| introspection.py | `get_campaign_params`, `get_campaign_grid_params`, `get_campaign_health_check_params`, `get_validation_rules` | functions | 60% | **REMOVE** | Campaign introspection, never called |
| models.py | `LatencySimulation` | class | 60% | **REMOVE** | Domain model, never instantiated |
| models.py | `expand_day_aliases`, `validate_schedule_has_timing`, `apply_preset`, `resolve_builtin_alias`, `validate_window_size`, `ensure_gpus_list` | methods | 60% | REMOVE | Various config methods unused |
| naming.py | `get_canonical_name`, `get_cli_flag_for_param`, `is_deprecated_cli_flag`, `get_all_deprecated_cli_flags` | functions | 60% | **REMOVE** | Naming utilities, entire module likely dead |
| provenance.py | `get_provenance`, `to_summary_dict` | methods | 60% | REMOVE | Provenance tracking unused |
| quantization.py | `weight_only`, `backend_method`, `bits` (property), `validate_calibration_requirements` | members | 60% | **REMOVE** | Quantization config fields/methods unused |
| speculative.py | `num_speculative_tokens`, `draft_tensor_parallel_size`, `ngram_min`, `ngram_max`, `validate_draft_model_requirement`, `uses_draft_model` | members | 60% | **REMOVE** | Speculative decoding config, entire feature unused |
| validation.py | `migration_hint` | field | 60% | REMOVE | Migration field unused |

**Analysis**: Config module has extensive staged/incomplete features:
- **TensorRT fields**: 8 fields defined but never accessed (dead weight in schema)
- **Campaign health checks**: 4 fields for health monitoring, never implemented
- **Introspection SSOT**: 11 functions for parameter metadata, never called (contradicts SSOT claim in docs)
- **Speculative decoding**: Entire config class unused (feature stub)
- **Quantization**: Config exists but fields unused
- **Naming module**: Entire module (4 functions) unused

#### Constants Module (constants.py)

| Finding | Type | Severity | Notes |
|---------|------|----------|-------|
| `DEFAULT_WARMUP_RUNS`, `DEFAULT_SAMPLING_INTERVAL_SEC`, `DEFAULT_ACCELERATE_PORT` | constants | REMOVE | Never referenced |
| `DEFAULT_MAX_NEW_TOKENS`, `DEFAULT_TEMPERATURE`, `DEFAULT_TOP_P` | constants | REMOVE | Never used (Pydantic defaults inline) |
| `DEFAULT_BARRIER_TIMEOUT_SEC`, `DEFAULT_FLOPS_TIMEOUT_SEC`, `DEFAULT_GPU_INFO_TIMEOUT_SEC`, `DEFAULT_SIGKILL_WAIT_SEC` | constants | REMOVE | Timeout constants unused |
| `get_preset_metadata`, `get_preset_config` | functions | REMOVE | Preset utilities unused |
| `is_cli_flag_deprecated`, `get_deprecation_info` | functions | REMOVE | Deprecation system unused |

**Analysis**: 14 dead constants/functions. Suggests constants were defined for future use but Pydantic model defaults replaced them.

#### Core Module (core/)

| File | Finding | Type | Severity | Notes |
|------|---------|------|----------|-------|
| baseline.py | `invalidate_baseline_cache` | function | REMOVE | Cache invalidation never called |
| compute_metrics.py | `current_reserved_bytes`, `max_reserved_bytes`, `gpu_utilization_percent`, `cpu_memory_bytes` | fields | **REMOVE** | Compute metrics fields, never accessed |
| energy_backends/codecarbon.py | `get_raw_data` | method | REMOVE | CodeCarbon backend method unused |
| extended_metrics.py | `tokens_per_gb_vram`, `model_memory_utilisation`, `kv_cache_memory_ratio`, `sm_utilisation_samples`, `batch_utilisation`, `e2e_latency_median_ms`, `e2e_latency_p99_ms`, `e2e_latency_samples` | attributes | **REMOVE** | Extended metrics, 8 fields never accessed |
| flops.py | `_timeout_sec` | attribute | REMOVE | Timeout field unused |
| gpu_info.py | `parent_gpus` (property) | property | REMOVE | MIG parent tracking unused |
| gpu_info.py | `pending_mode` | variable | REMOVE | GPU mode variable unused |

**Analysis**: Extended metrics module has 8 dead fields. Suggests over-designed metric collection with many fields never exported/used.

#### Domain Module (domain/)

| File | Finding | Type | Severity | Notes |
|------|---------|------|----------|-------|
| metrics.py | 20+ fields | fields | **REMOVE** | See detailed list below |
| model_info.py | `bits`, `is_bnb`, `flops_reduction_factor`, `revision`, `num_layers`, `torch_dtype`, `parameters_billions`, `is_quantized`, `from_hf_config` | members | **REMOVE** | ModelInfo fields/methods unused |

**Domain/metrics.py dead fields** (20 items):
- `target_cv`, `flops_per_token`
- `efficiency_tokens_per_joule`, `efficiency_flops_per_watt` (properties)
- `tokens_per_gb_vram`, `model_memory_utilisation`, `kv_cache_memory_ratio`
- `sm_utilisation_samples`, `memory_bandwidth_utilisation`
- `batch_utilisation`
- `e2e_latency_median_ms`, `e2e_latency_p99_ms`, `e2e_latency_samples`
- `request_count`, `excluded_tokens`, `streaming_mode`, `warmup_requests_excluded`
- `measurement_method` (property)
- `ttft_min_ms`, `ttft_max_ms`

**Analysis**: Domain models have extensive dead fields. Suggests over-designed schemas with many fields defined "just in case" but never populated/consumed.

#### Exceptions Module (exceptions.py)

| Finding | Type | Severity | Notes |
|---------|------|----------|-------|
| `ModelLoadError`, `InferenceError`, `EnergyTrackingError`, `DistributedError`, `BackendTimeoutError`, `BackendConfigError` | exception classes | **REMOVE** | 6 exception types, never raised |

**Analysis**: Exception hierarchy defined but never used. Generic exceptions or inline error handling used instead.

#### Orchestration Module (orchestration/)

| File | Finding | Type | Severity | Notes |
|------|---------|------|----------|-------|
| campaign.py | `warmup_completed`, `completed_at`, `current_cycle`, `current_config_index`, `warmup_in_progress`, `progress_fraction`, `config_names`, `run_warmup`, `wait_config_gap`, `wait_cycle_gap`, `_last_cycle_complete_time`, `should_health_check`, `should_cold_start` | members | **REMOVE** | CampaignRunner fields/methods, 12 dead items |
| container.py | `get_status`, `restart_service` | methods | REMOVE | ContainerManager methods unused |
| context.py | `elapsed_time` | property | REMOVE | ExperimentContext property unused |
| launcher.py | `launch_experiment_accelerate`, `run_from_config` | functions | **REMOVE** | Accelerate launcher functions, never called |
| manifest.py | `completed_at`, `retry_count`, `created_at`, `progress_fraction`, `get_by_status`, `check_config_changed` | members | REMOVE | CampaignManifest fields/methods unused |

**Analysis**: Campaign orchestration has 12 dead fields/methods in CampaignRunner alone. Suggests over-designed state machine with many states/transitions never reached. Accelerate launcher functions completely unused (2 functions).

#### Results Module (results/)

| File | Finding | Type | Severity | Notes |
|------|---------|------|----------|-------|
| aggregation.py | `found_processes`, `missing_indices`, `duplicate_indices` | fields | REMOVE | Aggregation validation fields unused |
| bootstrap.py | `metric_name` | variable | SIMPLIFY | Loop variable unused (100% confidence) |
| exporters.py | `export_aggregated`, `export_raw`, `export_json` | methods | REMOVE | Exporter methods unused |
| repository.py | `has_raw`, `delete_experiment` | methods | REMOVE | Repository methods unused |

#### State Module (state/)

| File | Finding | Type | Severity | Notes |
|------|---------|------|----------|-------|
| experiment_state.py | `completed_at`, `last_updated`, `processes_failed`, `can_aggregate`, `mark_completed`, `mark_failed`, `is_pending`, `can_transition_to`, `delete` | members | **REMOVE** | ExperimentState fields/methods, 9 dead items |

**Analysis**: State module has extensive dead state management fields. Suggests over-designed state machine.

#### Top-Level Modules

| File | Finding | Type | Severity | Notes |
|------|---------|------|----------|-------|
| logging.py | `setup_logging_for_verbosity`, `get_logger`, `is_backend_filtering_active` | functions | REMOVE | Logging utilities unused |
| progress.py | `_current`, `exc_type`, `exc_val`, `exc_tb` | members | SIMPLIFY | Progress tracker context manager vars unused |
| resilience.py | `retry_on_error`, `cleanup_gpu_memory`, `safe_cleanup` | functions | **REMOVE** | Resilience utilities, entire module unused |
| security.py | `validate_path`, `check_env_for_secrets` | functions | **REMOVE** | Security utilities, entire module unused |

**Analysis**: Entire modules unused:
- **resilience.py**: 3 functions, never called
- **security.py**: 2 functions, never called

### 1.2 Deadcode Findings (274 items)

Deadcode results show 95%+ overlap with vulture. Key differences:
- Deadcode provides **error codes** (DC01=variable, DC02=function, DC04=method, DC05=attribute, DC08=property)
- Both tools agree on CLI functions being "unused" (Typer decorator limitation)
- Both tools agree on 10+ entire modules/classes being unused (introspection.py functions, speculative.py, resilience.py, security.py)

**High-confidence overlap** (both tools flagged):
- config/introspection.py: 11 functions
- config/speculative.py: entire class (7 members)
- config/naming.py: 4 functions
- exceptions.py: 6 exception classes
- resilience.py: 3 functions
- security.py: 2 functions
- orchestration/launcher.py: 2 functions (accelerate)
- domain/metrics.py: 20+ fields
- domain/model_info.py: 9 members

### 1.3 Unused Imports

**Ruff F401 check**: 0 unused imports found.

**Analysis**: Code is well-maintained for import cleanup. No accumulation of unused imports.

---

## 2. Complexity Hotspots (Ruff C901)

**32 functions exceed complexity threshold (C901 > 10)**. Ruff default threshold is 10; industry best practice is 10-15.

### 2.1 Critical Complexity (>30)

| File | Function | Complexity | Severity | Notes |
|------|----------|------------|----------|-------|
| cli/campaign.py | `campaign_cmd` | **62** | CRITICAL | Main campaign entry point |
| cli/experiment.py | `experiment_cmd` | **71** | CRITICAL | Main experiment entry point |
| cli/schedule.py | `schedule_experiment_cmd` | **38** | CRITICAL | Schedule command entry |
| core/inference_backends/vllm.py | `_build_engine_kwargs` | **36** | CRITICAL | vLLM config builder |
| config/generate_grid.py | `config_generate_grid` | **32** | CRITICAL | Grid generation logic |
| display/summaries.py | `display_config_summary` | **32** | CRITICAL | Config display formatting |
| orchestration/runner.py | `run` | **33** | CRITICAL | ExperimentOrchestrator.run |

**Analysis**: 7 functions with complexity 30+. These are the highest-risk refactoring targets. CLI entry points (`campaign_cmd`, `experiment_cmd`) are massive god functions handling validation, config loading, orchestration, display, and error handling.

### 2.2 High Complexity (20-30)

| File | Function | Complexity |
|------|----------|------------|
| cli/init_cmd.py | `init_cmd` | 20 |
| core/inference_backends/pytorch.py | `_build_generation_kwargs` | 20 |
| core/inference_backends/shared.py | `create_precision_metadata` | 18 |
| config/introspection.py | `_extract_param_metadata` | 18 |
| orchestration/grid.py | `expand_campaign_grid` | 19 |

**Analysis**: 5 functions with complexity 20-30. Config building and parameter introspection logic has high branching.

### 2.3 Moderate Complexity (11-19)

| File | Function | Complexity | Notes |
|------|----------|------------|-------|
| cli/batch.py | `batch_run_cmd` | 14 | Batch orchestration |
| cli/campaign.py | `_run_campaign_loop` | 16 | Campaign execution loop |
| cli/campaign.py | `_run_single_experiment` | 13 | Single experiment runner |
| cli/campaign.py | `_apply_cli_overrides` | 12 | CLI override logic |
| cli/campaign.py | `_display_campaign_ci_summary` | 12 | CI display formatting |
| cli/config.py | `config_list` | 13 | Config listing with grouping |
| cli/config.py | `config_new` | 14 | Interactive config builder |
| cli/resume.py | `resume_cmd` | 16 | Resume discovery logic |
| cli/display/summaries.py | `show_effective_config` | 12 | Config display |
| config/loader.py | `validate_config` | 15 | Config validation |
| config/loader.py | `load_config_with_provenance` | 12 | Config loading with provenance |
| core/model_loader.py | `load_model_tokenizer` | 12 | Model loading branching |
| core/power_thermal.py | `_sample_loop` | 12 | Power sampling loop |
| core/warmup.py | `warmup_until_converged` | 15 | Warmup convergence logic |
| core/inference_backends/pytorch.py | `_run_streaming_inference` | 11 | PyTorch streaming |
| core/inference_backends/tensorrt.py | `_run_streaming_inference` | 17 | TensorRT streaming |
| core/inference_backends/tensorrt.py | `_run_batch_with_ttft_estimation` | 12 | TensorRT batch+TTFT |
| core/inference_backends/vllm.py | `_create_sampling_params` | 12 | vLLM sampling params |
| core/inference_backends/vllm.py | `_run_streaming_inference` | 15 | vLLM streaming |
| orchestration/launcher.py | `launch_experiment_accelerate` | 11 | Accelerate launcher (DEAD CODE) |

**Analysis**: 20 functions with complexity 11-19. Most are CLI commands and backend parameter builders. High branching due to parameter validation and conditional logic.

### 2.4 Complexity Summary

| Complexity Range | Count | Severity | Action |
|------------------|-------|----------|--------|
| 60-71 | 2 | CRITICAL | Refactor immediately |
| 30-40 | 5 | CRITICAL | Refactor in audit phase |
| 20-30 | 5 | HIGH | Consider simplification |
| 11-19 | 20 | MODERATE | Monitor, refactor if touched |

**Total**: 32 functions exceeding threshold

**Hotspots by module**:
- **cli/campaign.py**: 6 functions (complexity 12-62)
- **cli/**: 15 functions total
- **core/inference_backends/**: 8 functions (backend config builders)
- **config/**: 4 functions
- **orchestration/**: 3 functions

---

## 3. Stub and TODO Detection

### 3.1 Pass Statements (30 items)

Most `pass` statements are in **error handling blocks** (expected pattern for "ignore this error"):

| File | Context | Count | Assessment |
|------|---------|-------|------------|
| cli/init_cmd.py | Keyboard interrupt handlers | 4 | KEEP (intentional) |
| core/power_thermal.py | NVML exception handling | 6 | KEEP (error suppression) |
| core/gpu_info.py | GPU detection error handling | 3 | KEEP (error suppression) |
| security.py, resilience.py | Dead modules | 2 | REMOVE (modules unused) |
| domain/metrics.py | Empty Protocol class | 1 | KEEP (protocol stub) |
| orchestration/runner.py | Error suppression | 1 | KEEP |
| core/baseline.py | Error suppression | 1 | KEEP |
| others | Various error handling | 12 | KEEP |

**Analysis**: 2 `pass` statements in dead modules (security.py, resilience.py). Rest are intentional error suppression. No functionality stubs found.

### 3.2 NotImplementedError (1 item)

| File | Context | Line | Assessment |
|------|---------|------|------------|
| core/parallelism.py | `_validate_distributed_setup()` | 542 | **REMOVE** (dead code branch) |

**Analysis**: Single `NotImplementedError` in validation logic for an unsupported distributed mode. Dead code path.

### 3.3 Ellipsis (35 items)

Most ellipsis (`...`) are in **Protocol definitions** (type hints for structural subtyping):

| File | Count | Context |
|------|-------|---------|
| core/inference_backends/protocols.py | 10 | Protocol method stubs (expected) |
| protocols.py | 11 | Protocol method stubs (expected) |
| core/parallelism.py | 5 | Protocol method stubs (expected) |
| core/prompts.py | 1 | Protocol method stub (expected) |
| README files, code comments | 8 | Documentation (ignore) |

**Analysis**: All ellipsis are in Protocol class definitions or documentation. **No functionality stubs found**.

### 3.4 TODO Comments (3 items)

| File | Line | TODO | Severity | Assessment |
|------|------|------|----------|------------|
| config/introspection.py | 807 | "In future, could use AST parsing to extract these automatically" | LOW | Idea note, not blocker |
| cli/experiment.py | 593 | "Actual resume logic" | **HIGH** | Resume feature stub |
| core/inference_backends/pytorch.py | 375 | "Pass model_kwargs to loader when supported" | MEDIUM | Parameter passing incomplete |

**Analysis**:
- **1 HIGH priority TODO**: Resume logic in experiment.py (feature incomplete)
- **2 LOW-MEDIUM**: Future ideas, not blocking

---

## 4. Industry CLI Surface Comparison

### 4.1 Our Tool CLI Structure

**Main commands (13)**:
1. `experiment` — Run single experiment
2. `aggregate` — Aggregate per-process results
3. `datasets` — List datasets
4. `presets` — List presets
5. `gpus` — Show GPU topology
6. `doctor` — Diagnostic checks
7. `batch` — Batch run configs
8. `schedule` — Scheduled experiments
9. `campaign` — Multi-config campaigns
10. `init` — Initialize project
11. `resume` — Resume campaigns
12. `config` — Config management (5 subcommands)
13. `results` — Results inspection (2 subcommands)

**Subcommands**:
- `config`: list, validate, show, new, generate-grid (5)
- `results`: list, show (2)

**Total surface**: 13 main + 7 subcommands = **20 CLI entry points**

### 4.2 Industry Tool Comparison

#### lm-eval-harness (EleutherAI)

**Repository**: https://github.com/EleutherAI/lm-evaluation-harness

**CLI Surface**:
- **Main command**: `lm_eval` or `lm-eval` (single entry point)
- **Primary mode**: `lm_eval --model <type> --model_args <args> --tasks <tasks>`
- **Config**: JSON files or CLI args (no config subcommands)
- **Results**: JSON/CSV output files (no results CLI)

**Count**: **1 main command** (all via flags)

**Architecture**:
- Simple CLI with many flags (--model, --tasks, --output_path, --batch_size, etc.)
- No campaign/grid/sweep built-in (users script it)
- No result aggregation CLI (files are output directly)
- No init/setup wizard

**Comparison**:
| Aspect | lm-eval | Our Tool | Assessment |
|--------|---------|----------|------------|
| Main commands | 1 | 13 | **Over-engineered** |
| Config management | CLI args/files | 5-command subsystem | **Over-engineered** |
| Campaign/grid | External scripting | Built-in | **Over-engineered** |
| Result inspection | File output | 2-command subsystem | **Over-engineered** |
| Init wizard | None | Built-in | **Nice-to-have** |

#### vLLM Benchmarks

**Repository**: https://github.com/vllm-project/vllm/tree/main/benchmarks

**CLI Surface**:
- **Scripts**: `benchmark_serving.py`, `benchmark_throughput.py`, `benchmark_latency.py` (3 separate scripts)
- **Usage**: `python benchmark_throughput.py --model <model> --input-len 128 --output-len 128`
- **Config**: CLI args only (no YAML configs)
- **Results**: Stdout/JSON files (no CLI)

**Count**: **3 scripts** (not a CLI tool, Python scripts)

**Architecture**:
- Separate Python scripts for different benchmark types
- No package entry point (run via `python`)
- No config files (all flags)
- No campaign orchestration
- No result management

**Comparison**:
| Aspect | vLLM Benchmarks | Our Tool | Assessment |
|--------|-----------------|----------|------------|
| Entry points | 3 scripts | 1 CLI + 13 commands | **Over-engineered** |
| Config system | Flags only | YAML + presets + validation | **Over-engineered** |
| Campaign | Manual scripting | Built-in orchestrator | **Over-engineered** |

#### nanoGPT (Karpathy)

**Repository**: https://github.com/karpathy/nanoGPT

**CLI Surface**:
- **Scripts**: `train.py`, `sample.py`, `prepare.py` (3 scripts)
- **Usage**: `python train.py config/train_gpt2.py`
- **Config**: Python files (config as code)
- **No CLI tool** (educational codebase)

**Count**: **3 Python scripts**

**Architecture**:
- Minimalist: Python scripts + config files
- No CLI framework
- No orchestration
- Educational focus

**Comparison**: Not directly comparable (different domain), but demonstrates "minimal viable tool" philosophy.

#### LLMPerf (Ray)

**Repository**: https://github.com/ray-project/llmperf

**CLI Surface**:
- **Main command**: `llmperf` (single entry)
- **Usage**: `llmperf --model <endpoint> --num-requests 100 --output-path results.json`
- **Config**: CLI args + optional config file
- **Results**: JSON files (no inspection CLI)

**Count**: **1 main command**

**Architecture**:
- Single command with many flags
- Targets hosted API endpoints (different use case)
- No local model loading
- No campaign orchestration (users loop manually)

**Comparison**:
| Aspect | LLMPerf | Our Tool | Assessment |
|--------|---------|----------|------------|
| Commands | 1 | 13 | **Over-engineered** |
| Config | Flags + optional file | Full config subsystem | **Over-engineered** |
| Orchestration | Manual | Built-in | **Over-engineered** |

#### GenAI-Perf (NVIDIA Triton)

**Repository**: https://github.com/triton-inference-server/perf_analyzer

**CLI Surface**:
- **Main command**: `perf_analyzer` (single entry)
- **Usage**: `perf_analyzer -m <model> -u <url> --concurrency-range 1:4`
- **Config**: CLI args (no files)
- **Results**: JSON/CSV output (no CLI)

**Count**: **1 main command**

**Architecture**:
- Single CLI with extensive flags
- Targets inference servers (Triton)
- No config files
- No campaign/grid built-in

**Comparison**:
| Aspect | GenAI-Perf | Our Tool | Assessment |
|--------|------------|----------|------------|
| Commands | 1 | 13 | **Over-engineered** |
| Config | Flags only | YAML subsystem | **Over-engineered** |

### 4.3 Industry Comparison Summary

| Tool | Entry Points | Config Approach | Campaign/Grid | Result CLI | Init Wizard |
|------|-------------|-----------------|---------------|------------|-------------|
| **lm-eval-harness** | 1 command | CLI args/JSON | External | None | None |
| **vLLM Benchmarks** | 3 scripts | Flags | External | None | None |
| **nanoGPT** | 3 scripts | Python files | N/A | None | None |
| **LLMPerf** | 1 command | Flags + file | External | None | None |
| **GenAI-Perf** | 1 command | Flags | External | None | None |
| **Our Tool** | 13 commands | YAML + presets | **Built-in** | 2 commands | **Built-in** |

**Industry norm**: **1-3 entry points**, flags/files for config, **no built-in campaign orchestration**, **no result inspection CLI**.

### 4.4 CLI Comparison Analysis

| Aspect | Industry Norm | Our Tool | Verdict |
|--------|---------------|----------|---------|
| **Total commands** | 1-3 | 13 | **3-4x larger** |
| **Config management** | Flags or files | Dedicated subsystem (5 commands) | **Over-engineered** |
| **Campaign/grid** | User scripts | Built-in orchestrator | **Over-engineered** |
| **Result inspection** | File output | Dedicated subsystem (2 commands) | **Over-engineered** |
| **Diagnostics** | None | `lem doctor` | **Valuable unique feature** |
| **Init wizard** | None | `lem init` | **Valuable unique feature** |
| **Dataset/preset listing** | N/A | Dedicated commands | **Minor utility** |
| **GPU topology** | N/A | `lem gpus` | **Minor utility** |
| **Batch/schedule** | User scripts | Built-in (2 commands) | **Over-engineered** |

**Recommendations**:
1. **Keep unique value**: `doctor`, `init`, `experiment` (core)
2. **Remove over-engineered**: `batch`, `schedule`, `campaign`, `resume`, `aggregate`
3. **Simplify config**: Merge 5 subcommands into 2-3 (validate, new, show)
4. **Simplify results**: Merge into `experiment` output or single `results` command
5. **Target**: Reduce from 13 commands to **4-6 commands** (60% reduction)

---

## 5. Key Takeaways

### 5.1 Dead Code Categories

| Category | Severity | Count | Examples |
|----------|----------|-------|----------|
| **Entire modules unused** | CRITICAL | 3 | resilience.py, security.py, naming.py |
| **Entire classes unused** | CRITICAL | 8 | LatencySimulation, SpeculativeConfig, 6 exception classes |
| **Config feature stubs** | HIGH | 50+ | TensorRT fields, campaign health checks, introspection functions |
| **Domain model dead fields** | HIGH | 30+ | Extended metrics, model info fields |
| **Orchestration dead state** | MEDIUM | 20+ | Campaign/manifest fields, accelerate launcher |
| **Utility functions unused** | MEDIUM | 15+ | Various helpers across modules |

**Total dead code estimate**: **150+ items** (functions, classes, fields, constants)

### 5.2 Complexity Hotspots

- **2 critical functions** (complexity 60-71): `campaign_cmd`, `experiment_cmd`
- **5 critical functions** (complexity 30-40): Grid generation, config display, orchestrator
- **32 total functions** exceeding threshold
- **Concentrated in CLI** (15 functions) and **backend config builders** (8 functions)

### 5.3 CLI Surface

- **13 main commands vs industry 1-3** (3-4x larger)
- **Over-engineered orchestration**: campaign, batch, schedule, resume
- **Over-engineered config**: 5 subcommands vs industry "just use files"
- **Unique value preserved**: doctor, init, core experiment runner

### 5.4 Stub/Incomplete Features

- **1 HIGH TODO**: Resume logic incomplete
- **Speculative decoding**: Config exists, feature unused
- **Quantization**: Config exists, feature unused
- **Health checks**: Campaign fields exist, feature unimplemented
- **Introspection SSOT**: 11 functions exist, never called (contradicts docs)

---

## 6. Recommendations for Audit

### 6.1 Immediate Actions (Phase 4)

1. **Verify CLI registration** — Confirm whether `config list/show/new` and `results list/show` are actually dead or false positives
2. **Remove dead modules** — Delete resilience.py, security.py, naming.py (3 entire modules)
3. **Remove dead exceptions** — Delete 6 exception classes never raised
4. **Remove dead config features** — Delete speculative.py, quantization stub fields, TensorRT unused fields
5. **Remove dead introspection** — Delete 11 unused SSOT functions (contradicts docs)
6. **Document incomplete features** — Flag resume TODO, quantization/speculative stubs

### 6.2 Refactoring Priorities (Phase 5)

1. **Refactor god functions** — Split `campaign_cmd` (62) and `experiment_cmd` (71) into smaller functions
2. **Simplify CLI surface** — Remove/merge campaign, batch, schedule, aggregate, resume commands
3. **Consolidate config commands** — Merge 5 config subcommands into 2-3
4. **Remove extended metrics dead fields** — Delete 8 unused metric fields
5. **Simplify campaign orchestration** — Remove 12 dead CampaignRunner fields/methods

### 6.3 Architecture Decisions Needed

1. **Campaign orchestration** — Keep or remove? Industry doesn't have it.
2. **Config subsystem** — Simplify to match industry (flags + files)?
3. **Result inspection CLI** — Keep or rely on file output?
4. **Batch/schedule commands** — Remove in favor of user scripting?

---

## Appendix A: Tool Versions

- **vulture**: 2.14
- **deadcode**: 2.4.1
- **ruff**: Latest (C901 complexity check)
- **Analysis date**: 2026-02-05

## Appendix B: Methodology

1. Ran vulture with 60% confidence threshold (balance false positives vs coverage)
2. Cross-referenced vulture with deadcode for high-confidence findings
3. Ruff complexity analysis with default threshold (10)
4. Manual grep for stub patterns (pass, NotImplementedError, ellipsis, TODO)
5. Industry research via GitHub repositories and documentation
6. CLI surface comparison via `--help` output

## Appendix C: False Positive Notes

**Typer decorator registration**: Vulture/deadcode flag CLI commands as unused because decorator-based registration (`@app.command()`) doesn't show direct calls. **Manual verification required** to confirm these are genuinely wired vs genuinely dead.

**Protocol methods**: Ellipsis in Protocol classes are expected (structural subtyping). Not dead code.

**Error suppression**: `pass` in exception handlers is intentional. Not dead code.
