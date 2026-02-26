# Codebase Concerns

**Analysis Date:** 2026-02-05

## Tech Debt

**Docker GPU Passthrough System (CRITICAL):**
- Issue: Three separate backends all fail with CUDA driver initialization errors in Docker containers. PyTorch hangs when using `accelerate launch` with data_parallel, vLLM worker processes can't see GPUs with `tensor_parallel_size=2`, TensorRT routes to wrong container and reports backend unavailable. The common thread is `NVIDIA_VISIBLE_DEVICES` and CUDA env vars not being propagated correctly through docker-compose.yml.
- Files: `docker-compose.yml`, `docker/Dockerfile.pytorch`, `docker/Dockerfile.vllm`, `docker/Dockerfile.tensorrt`, `src/llenergymeasure/orchestration/launcher.py`, `src/llenergymeasure/cli/campaign.py`, `src/llenergymeasure/cli/experiment.py`
- Impact: Docker execution is completely broken for all three backends. Multi-GPU parallelism fails. Users cannot run experiments in containers, which defeats the isolation/reproducibility benefits of Docker.
- Fix approach: Audit entire GPU environment variable flow from docker-compose.yml → entrypoint.sh → launcher.py → subprocess env. Ensure `NVIDIA_VISIBLE_DEVICES` (for container runtime) and `CUDA_VISIBLE_DEVICES` (for CUDA inside container) are set consistently. Add validation that GPUs are visible before launching inference. Consider integration test that validates GPU access in all three backend containers.

**CUDA Initialization Ordering:**
- Issue: vLLM and TensorRT backends manage their own CUDA contexts but orchestration layer (`ExperimentContext.create()`, launcher.py) has multiple code paths that call `torch.cuda.is_available()` or create `torch.device("cuda:0")` BEFORE backend initialization. This breaks CUDA fork safety.
- Files: `src/llenergymeasure/orchestration/context.py` lines 123-161, `src/llenergymeasure/orchestration/launcher.py` lines 692-755, `src/llenergymeasure/core/distributed.py` lines 41-44
- Impact: CUDA driver init failures in Docker. vLLM and TensorRT may fail with cryptic multiprocessing errors.
- Fix approach: Strictly enforce `orchestrator_may_call_cuda` check (already in RuntimeCapabilities). NO torch.cuda.* calls before backend.initialize() when False. Use environment detection only (CUDA_VISIBLE_DEVICES string parsing, no torch imports). Add assertion at top of vLLM/TensorRT initialize() that detects CUDA already initialized and fails fast.

**Multiple Detection Systems with Unclear Boundaries:**
- Issue: Four separate detection/setup modules added across 10+ phases: `docker_detection.py`, `backend_detection.py`, `env_setup.py`, and container strategy selection in campaign.py. Unclear which handles what, possible redundancy.
- Files: `src/llenergymeasure/config/docker_detection.py`, `src/llenergymeasure/config/backend_detection.py`, `src/llenergymeasure/config/env_setup.py`, `src/llenergymeasure/cli/campaign.py` lines 1271-1277
- Impact: Maintenance burden, risk of conflicting logic, unclear entry points for debugging environment issues.
- Fix approach: Document clear responsibility boundaries: docker_detection = "are we in container?", backend_detection = "is backend importable?", env_setup = ".env generation", container strategy = "ephemeral vs persistent". Consider unifying docker_detection + backend_detection into single `runtime_detection.py` if overlap found. Add architecture diagram showing how these modules interact.

**1754-Line campaign.py File:**
- Issue: Campaign command file is 1754 lines, handling grid expansion, Docker dispatch, manifest management, progress display, and execution orchestration in one module.
- Files: `src/llenergymeasure/cli/campaign.py`
- Impact: Hard to maintain, test, and reason about. Changes in one area risk breaking others.
- Fix approach: Extract modules: `campaign_grid.py` (grid expansion), `campaign_docker.py` (container dispatch), `campaign_progress.py` (display), `campaign_execution.py` (execution loop). Keep campaign.py as thin orchestrator importing from these modules.

**Launcher.py Launch Mode Complexity:**
- Issue: `launcher.py` has three launch modes (direct, torchrun, accelerate) with conditional logic spread across `_get_launch_mode()`, `_build_launch_command()`, `launch_experiment_accelerate()`, and environment setup. Backend capabilities used for some decisions, hardcoded backend names for others.
- Files: `src/llenergymeasure/orchestration/launcher.py` lines 143-429
- Impact: Adding new backends requires touching multiple functions. Launch mode logic not centralized.
- Fix approach: Refactor to strategy pattern: `DirectLauncher`, `TorchrunLauncher`, `AccelerateLauncher` classes. Backend returns launcher instance via capabilities. Eliminates conditional logic. Centralizes environment setup per launcher type.

**Incomplete TODOs in Production Code:**
- Issue: Two TODOs found in production paths: `cli/experiment.py:593` "TODO: Actual resume logic" and `core/inference_backends/pytorch.py:375` "TODO: Pass model_kwargs to loader when supported". First is in resume flow (may be incomplete), second is feature gap.
- Files: `src/llenergymeasure/cli/experiment.py` line 593, `src/llenergymeasure/core/inference_backends/pytorch.py` line 375
- Impact: Resume functionality may not work correctly (experiment.py TODO). PyTorch backend missing model_kwargs passthrough (limited model customization).
- Fix approach: Either implement resume logic or remove `--resume` flag. For model_kwargs, determine if blocker and implement or document as won't-fix. Add lint rule to fail CI on "TODO" in production paths.

**Config Introspection Module (SSOT) Complexity:**
- Issue: `config/introspection.py` is 807 lines implementing Single Source of Truth for parameter metadata. Has 15+ introspection functions. Research shows comparable tools (lm-eval-harness, vLLM) use simple static dicts/YAML for parameter metadata, not runtime introspection.
- Files: `src/llenergymeasure/config/introspection.py`
- Impact: High cognitive overhead for new contributors. Pre-commit hooks require understanding AST parsing to modify parameters. Debugging parameter metadata requires tracing through Pydantic model structure.
- Fix approach: Trade-off decision needed. SSOT prevents config drift (parameter added to model but forgotten in tests/docs). Simpler approach: static parameter registry + linter to check sync. Evaluate if research-grade correctness justifies complexity vs industry norm (static lists). Document decision rationale.

**Ephemeral vs Persistent Container Strategy:**
- Issue: Campaign defaults to ephemeral containers (`docker run --rm`) per Phase 02.2, but persistent mode exists with confirmation dialogs. Adds conditional logic in campaign execution.
- Files: `src/llenergymeasure/cli/campaign.py` container strategy selection, user config persistence
- Impact: Two execution paths to maintain. Research tools (vLLM benchmarks, lm-eval-harness) don't have container strategy options - they just run containers.
- Fix approach: Simplify to always ephemeral (`--rm`). If user wants persistent containers, they can use `docker compose up -d` manually and run commands inside. Removes user-facing complexity. Update docs to show persistent pattern as manual Docker workflow.

**Large Display Summary Module:**
- Issue: `cli/display/summaries.py` (804 lines) handles all config/state summary rendering in one module
- Files: `src/llenergymeasure/cli/display/summaries.py`
- Impact: Hard to test display logic, mixing data preparation and Rich output formatting
- Fix approach: Separate data collection (config summaries) from rendering (Rich panels). Extract to `summary_data.py` (data models) + `summary_render.py` (Rich formatters).

**Large Monolithic Backend Files:**
- Issue: Inference backend files (pytorch.py, vllm.py, tensorrt.py) are 1000+ lines each, mixing initialization, configuration building, execution, and error handling
- Files: `src/llenergymeasure/core/inference_backends/pytorch.py` (1149 lines), `src/llenergymeasure/core/inference_backends/vllm.py` (1006 lines), `src/llenergymeasure/core/inference_backends/tensorrt.py` (1168 lines)
- Impact: Difficult to test individual concerns, high cognitive load for maintenance, risk of bugs in config building or error paths
- Fix approach: Extract per backend: `{backend}_config_builder.py` (generation kwargs building), `{backend}_lifecycle.py` (init/cleanup), keep backend file for execution only. Reduces each to ~400 lines.

## Known Bugs

**PyTorch Backend Hangs in Docker with Data Parallel (CRITICAL):**
- Symptoms: When using PyTorch backend with `num_processes > 1` in Docker, accelerate launch starts but processes hang after "CUDA driver initialization failed", then falls back to CPU, then hangs indefinitely.
- Files: PyTorch backend execution in Docker containers, `src/llenergymeasure/core/inference_backends/pytorch.py`, `docker-compose.yml` GPU passthrough, `src/llenergymeasure/orchestration/launcher.py`
- Trigger: `docker compose run pytorch lem experiment configs/pytorch_example.yaml` with `pytorch.num_processes: 2` in config
- Workaround: Run locally (not in Docker) or use `num_processes: 1`

**vLLM Tensor Parallel Fails in Docker (CRITICAL):**
- Symptoms: vLLM with `tensor_parallel_size=2` spawns worker processes that fail with "CUDA driver initialization failed: unknown error". Multiproc executor workers cannot see GPUs.
- Files: vLLM backend, `src/llenergymeasure/core/inference_backends/vllm.py` lines 222-258 (tensor parallel size detection), `docker-compose.yml` lines 132-154 (vllm service)
- Trigger: `docker compose run vllm lem experiment configs/vllm_example.yaml` with `vllm.tensor_parallel_size: 2`
- Workaround: Use `tensor_parallel_size: 1` or run locally

**TensorRT Container Routing (CRITICAL):**
- Symptoms: When running TensorRT backend, experiment routes to wrong Docker container (base instead of tensorrt). Backend reports "not available" error even though TensorRT image built.
- Files: Container dispatch logic, `src/llenergymeasure/cli/campaign.py` or backend detection, `docker-compose.yml` lines 179-201 (tensorrt service)
- Trigger: `docker compose run tensorrt lem experiment configs/tensorrt_example.yaml`
- Workaround: Manual invocation with explicit container name

**Experiment Resume Incomplete:**
- Symptoms: `--resume` flag accepted by CLI but TODO comment in code suggests actual resume logic not implemented (line 593 of experiment.py).
- Files: `src/llenergymeasure/cli/experiment.py` line 593
- Trigger: Using `lem experiment config.yaml --resume`
- Workaround: None - feature may not work. State tracking exists but checkpoint loading missing.

**pynvml Thread Safety with vLLM:**
- Symptoms: GPU utilisation sampler may fail silently when vLLM initialises CUDA context, returning empty samples
- Files: `src/llenergymeasure/core/gpu_utilisation.py`, `src/llenergymeasure/core/inference_backends/vllm.py`
- Trigger: Running vLLM backend with GPU utilisation sampling enabled
- Workaround: Sampling gracefully handles failures - returns empty sample list which extended metrics treats as N/A; not a crash but metric unavailability
- Root cause: vLLM's CUDA context management conflicts with pynvml initialization in background thread

**Memory Peak Statistics Reset Timing:**
- Symptoms: GPU memory metrics may not accurately reflect inference-only peak if model loading peak is higher
- Files: `src/llenergymeasure/core/compute_metrics.py`
- Trigger: When peak memory during model load exceeds peak during inference
- Workaround: Use reported peak_memory_mb which is most recent peak since reset
- Root cause: `torch.cuda.reset_peak_memory_stats()` called before inference, not accounting for model load peak in final metrics

**Incomplete Process Detection Logic:**
- Symptoms: Aggregation may proceed with partial results if some processes fail silently
- Files: `src/llenergymeasure/results/aggregation.py` lines 42-112
- Trigger: Process crashes without writing completion marker, or marker file corruption
- Workaround: Validation reports missing/duplicate processes but aggregation can continue if user overrides
- Root cause: Marker files written atomically but result files may exist without markers if process crashes between result write and marker write

## Security Considerations

**Privileged Docker Mode for Energy Metrics:**
- Risk: docker-compose.yml uses `privileged: true` for all containers to enable NVML energy readings. This grants container full host access.
- Files: `docker-compose.yml` line 36
- Current mitigation: Documented in docker-compose.yml comments. Required for CodeCarbon NVML access.
- Recommendations: Research capabilities-based alternative. Try `--cap-add=SYS_ADMIN` or device-specific permissions instead of full privileged mode. Document security tradeoff in deployment docs. Consider making privileged mode opt-in via environment variable.

**HF_TOKEN in Environment Variables:**
- Risk: HuggingFace token passed via environment variable in docker-compose.yml. Could be logged or exposed in process listings.
- Files: `docker-compose.yml` line 52, `src/llenergymeasure/cli/experiment.py` lines 675-677 (subprocess env propagation)
- Current mitigation: Token sourced from host environment, not hardcoded in compose file. Subprocess environment, not command line args.
- Recommendations: Use Docker secrets or file-based token mounting for production deployments. Document in deployment guide. Add warning in logs when HF_TOKEN detected in environment.

**Config File Path Traversal Prevention:**
- Risk: Config inheritance via `_extends` field could load arbitrary files if path not validated
- Files: `src/llenergymeasure/config/loader.py` lines 70-100
- Current mitigation: Uses `Path.resolve()` and cycle detection; no explicit boundary checking
- Recommendations: Add explicit check that inherited configs stay within allowed directory (e.g., same dir or subdirs only); warn on unusual paths

**YAML Deserialization:**
- Risk: YAML parsing with `yaml.safe_load()` is used, but still processes YAML/JSON without strict schema validation before Pydantic
- Files: `src/llenergymeasure/config/loader.py` line 57
- Current mitigation: `yaml.safe_load()` used (not `yaml.load()`), Pydantic validates after loading
- Recommendations: Add file size limits to prevent billion laughs attacks; consider pre-validating YAML structure

**Experiment ID Sanitization:**
- Risk: Experiment IDs used in filesystem paths; malformed IDs could cause issues
- Files: `src/llenergymeasure/security.py` lines 56-78
- Current mitigation: Alphanumeric + underscore/hyphen/dot allowed, non-conforming chars replaced with underscore
- Recommendations: Ensure all places creating directories/files use sanitised IDs; add tests for edge cases like "../" or special chars

## Performance Bottlenecks

**Sequential Campaign Execution:**
- Problem: Campaign runs experiments sequentially, one at a time. For large grids (e.g. 10 configs × 3 cycles = 30 experiments), total time scales linearly.
- Files: `src/llenergymeasure/orchestration/campaign.py` execution loop
- Cause: Simple for loop over experiments, no parallelization
- Improvement path: Parallel execution with GPU allocation (e.g. 4 GPUs → run 4 experiments in parallel). Requires resource manager and process coordination. Consider whether research tool needs this complexity vs users scripting their own parallelism.

**Grid Expansion at Campaign Start:**
- Problem: For large parameter grids, expansion happens synchronously at campaign start. Config file grows linearly with grid size.
- Files: `src/llenergymeasure/orchestration/grid.py`
- Cause: Grid expansion generates full Cartesian product upfront, writes all configs to manifest
- Improvement path: Lazy evaluation - generate configs on-demand during execution. Saves memory and startup time for large grids. Tradeoff: manifest doesn't show full plan upfront.

**Single-Threaded Result Aggregation:**
- Problem: Aggregation reads all process result files, computes statistics, writes aggregated result. For large datasets (e.g. 1000 prompts × 8 processes), I/O-bound.
- Files: `src/llenergymeasure/results/aggregation.py`
- Cause: Sequential file reads, in-memory accumulation
- Improvement path: Parallel file reading, streaming aggregation. Minimal gain unless datasets very large (>10k prompts).

**GPU Memory Tracking Overhead:**
- Problem: `get_memory_stats()` called potentially many times during inference to track peak memory
- Files: `src/llenergymeasure/core/compute_metrics.py`
- Cause: CUDA memory API calls have overhead; sampling on every batch adds up
- Improvement path: Cache memory stats, update only periodically (e.g., per 10 batches); or use single max_memory_allocated() call post-inference instead of continuous tracking

**Late Aggregation Full Dataset Load:**
- Problem: Aggregation loads all raw process results into memory at once for extended metrics computation
- Files: `src/llenergymeasure/results/aggregation.py` lines 115+
- Cause: Latency samples and GPU utilisation samples collected as lists in memory
- Improvement path: For very large experiments, stream results and compute running statistics; or use memory-mapped numpy arrays for latency data

**File I/O for Completion Markers:**
- Problem: Each process writes a separate marker file after result save - can cause filesystem thrashing in high-process counts
- Files: `src/llenergymeasure/orchestration/runner.py` lines 44-78
- Cause: Atomic write via temp + rename for each process marker
- Improvement path: Batch marker writes; or use single completion log file with process indices

## Fragile Areas

**GPU Environment Variable Propagation (CRITICAL):**
- Files: `docker-compose.yml` lines 53-58, `src/llenergymeasure/orchestration/launcher.py` lines 364-382, `src/llenergymeasure/cli/campaign.py` lines 1271-1277, `src/llenergymeasure/cli/experiment.py` lines 675-677
- Why fragile: Three separate locations set CUDA_VISIBLE_DEVICES and NVIDIA_VISIBLE_DEVICES. Order matters (container runtime vs CUDA inside container). MIG UUIDs vs integer indices handled differently in each location. Currently broken in Docker.
- Safe modification: Document the full flow: docker-compose.yml sets initial NVIDIA_VISIBLE_DEVICES → launcher.py propagates to subprocess env → CUDA code sees remapped indices. Add integration test that validates GPU visibility in containers. Centralize GPU env var logic into single utility function.
- Test coverage: No integration test for Docker GPU passthrough. Unit tests mock GPU detection.

**Backend Initialization Order:**
- Files: `src/llenergymeasure/core/inference_backends/vllm.py`, `src/llenergymeasure/core/inference_backends/tensorrt.py`, `src/llenergymeasure/orchestration/context.py` lines 123-161
- Why fragile: vLLM and TensorRT require NO torch.cuda.* calls before their initialize(). Capabilities declare this via `orchestrator_may_call_cuda=False`, but code has multiple paths (local vs Docker, single-process vs multi-process) that conditionally call CUDA functions.
- Safe modification: Strict gating on `orchestrator_may_call_cuda` check. Add assertion at top of vLLM/TensorRT initialize() that detects CUDA already initialized (via env var or torch internal state) and fails fast.
- Test coverage: No test validates CUDA init order. RuntimeCapabilities tested in isolation, not integration paths.

**Streaming Latency with TextIteratorStreamer Threading:**
- Files: `src/llenergymeasure/core/inference_backends/pytorch.py` lines 648-770
- Why fragile: Uses background thread with TextIteratorStreamer, complex synchronisation between main thread and streamer thread for token timestamp collection, race condition potential if generation fails midway
- Safe modification: Always wrap streamer code in try/finally; ensure thread joins even on exception; test with various generation failure modes
- Test coverage: Limited - needs tests for early stopping, max_length exceeded, and error cases

**vLLM Multiprocessing Configuration:**
- Files: `src/llenergymeasure/cli/experiment.py` lines 635-643, `src/llenergymeasure/core/inference_backends/vllm.py` lines 140+
- Why fragile: Multiple environment variable hacks to work around vLLM multiprocessing issues (V1 multiprocessing disable, spawn method, torch.compile disable). Changes to vLLM versions may break these workarounds
- Safe modification: Test with vLLM version updates before deploying; document exact versions these hacks are for; consider feature detection instead of version hardcoding
- Test coverage: Integration tests exist but only for specific vLLM versions

**GPU Topology Detection and Validation:**
- Files: `src/llenergymeasure/core/gpu_info.py` (482 lines), called from `src/llenergymeasure/cli/experiment.py` lines 599-605
- Why fragile: Complex GPU topology detection (MIG instances, nvlink, etc.) with multiple fallback paths; warnings-only approach means misconfigured GPUs silently proceed
- Safe modification: Add strict mode for GPU validation; test on various GPU configurations before changes; validate CUDA_VISIBLE_DEVICES matches detected topology
- Test coverage: Limited to basic cases - needs testing with MIG, nvlink, mixed GPU types

**Results Aggregation State Machine:**
- Files: `src/llenergymeasure/results/aggregation.py` lines 115-200+, `src/llenergymeasure/state/experiment_state.py` lines 30-47
- Why fragile: Complex validation logic for process completion with multiple edge cases (missing indices, duplicates, markers); aggregation can proceed with partial data if validation is overridden
- Safe modification: Add comprehensive logging at each validation step; unit test each validation condition independently; consider making aggregation strict by default
- Test coverage: Basic completeness checks exist, but edge cases like partial writes or concurrent aggregation attempts not tested

**Config Inheritance with Deep Merge:**
- Files: `src/llenergymeasure/config/loader.py` lines 20-36, 70-120
- Why fragile: Deep merge logic for config inheritance can silently drop values if merge order is wrong; cycle detection prevents infinite loops but doesn't prevent logical errors
- Safe modification: Add logging of merge operations; validate final config structure; test with complex inheritance chains
- Test coverage: Basic inheritance works, but edge cases like merging lists or special config sections not thoroughly tested

**Extended Metrics Null Handling:**
- Files: `src/llenergymeasure/core/extended_metrics.py`, `src/llenergymeasure/results/aggregation.py`
- Why fragile: Extended metrics schema always present but with null values when unavailable; downstream consumers must check for None carefully
- Safe modification: Add type guards; document null semantics; consider using sentinel values instead of None for clarity
- Test coverage: Should test aggregation with various combinations of unavailable metrics

**Config Provenance Tracking:**
- Files: `src/llenergymeasure/config/loader.py`, parameter_provenance dict propagated through CLI → context → results
- Why fragile: Parameter provenance tracked by name-based dictionary lookups. If parameter renamed or moved between config sections (e.g. batch_size moved from shared to backend-specific), provenance breaks.
- Safe modification: Add schema version to provenance. Validate that tracked parameters still exist in current config model.
- Test coverage: Provenance tested for happy path, not for config schema evolution.

**Manifest Resumption:**
- Files: `src/llenergymeasure/orchestration/manifest.py`, campaign resume logic
- Why fragile: Manifest JSON stores experiment state (pending, completed, failed). If experiment is killed mid-inference, state may be incorrect. Resume assumes manifest reflects reality.
- Safe modification: Add manifest validation on resume - check that "completed" experiments actually have result files, mark as "pending" if missing.
- Test coverage: Resumption tested for clean interrupts, not for partial writes or corrupted state.

## Scaling Limits

**Single-Node Campaign Execution:**
- Current capacity: Campaign runs on single host, limited by local GPU count
- Limit: Cannot scale campaign beyond local GPU resources. 100-config grid on 8-GPU server still sequential.
- Scaling path: Multi-node campaign orchestration (e.g. Ray, Dask, or Kubernetes Jobs). Major architectural change - campaigns become distributed system. Consider if research tool needs this vs users running multiple campaign instances on different machines.

**Result File Storage:**
- Current capacity: Per-process JSON result files, aggregated on-demand
- Limit: 10,000 experiments × 8 processes = 80,000 JSON files in results directory. File system performance degrades. Inode limits on some filesystems.
- Scaling path: Database backend for results (SQLite for local, PostgreSQL for multi-user). Requires result schema versioning and migration strategy. OR implement result sharding (subdirectories per GPU or process range).

**Docker Image Size:**
- Current capacity: TensorRT image includes CUDA 12.4, TensorRT-LLM, and dependencies = ~15GB
- Limit: Large images slow pulls, increase storage costs, slower iteration
- Scaling path: Multi-stage builds already used, but can optimize further. Use distroless base images, minimize layer count. Tradeoff: complexity vs size.

**Extended Metrics Late Aggregation Dataset Size:**
- Current capacity: Aggregation collects all per-request latencies and GPU samples in memory
- Limit: Large experiments (1000+ requests × 100+ processes) = millions of float values in memory
- Scaling path: Streaming aggregation with running statistics; or memory-mapped numpy arrays for sample data

**State File Lookups:**
- Current capacity: StateManager iterates all state files to find by config hash
- Limit: With 1000+ experiments, linear scan becomes slow
- Scaling path: Index state files by config hash; use simple SQLite for state store instead of JSON files

## Dependencies at Risk

**vLLM and TensorRT-LLM Version Conflicts:**
- Risk: Both require different PyTorch versions (vLLM wants 2.8+, pytorch backend uses 2.5.x, TensorRT wants specific CUDA bindings). Cannot coexist in same environment.
- Impact: Users cannot install both backends in one Python environment. Must use separate Docker images or virtualenvs.
- Migration plan: Already addressed via separate Docker images (`docker/Dockerfile.vllm`, `docker/Dockerfile.tensorrt`). Document clearly that vLLM + TensorRT are mutually exclusive in local installs. Recommend Docker for these backends.

**vLLM Backend Stability:**
- Risk: vLLM has frequent API changes; multiple version-specific workarounds in codebase (multiprocessing flags, cuda init)
- Impact: Updates may break inference or cause version conflicts
- Migration plan: Pin to LTS vLLM version; maintain compatibility matrix in docs; consider abstracting vLLM init behind stable interface. Document exact vLLM version workarounds are for.

**TensorRT-LLM Build Complexity:**
- Risk: Requires CUDA 12.x and specific GPU types (Ampere+ compute capability >= 8.0); build process fragile, multi-stage Docker build necessary
- Impact: Setup friction; compatibility issues across environments. V100, T4, RTX 20xx NOT supported.
- Migration plan: Document exact build requirements; consider pre-built wheel distribution; monitor for lighter TensorRT alternatives. Clearly document GPU compatibility in README.

**accelerate 0.x API Stability:**
- Risk: Accelerate is 0.x version (currently 0.26.1). API may break between minor versions.
- Impact: PyTorch backend parallelism may break on accelerate updates
- Migration plan: Pin accelerate to tested version range in pyproject.toml. When breaking changes occur, add compatibility shims or bump major version.

**CodeCarbon Energy Measurement:**
- Risk: Requires system tools (intel-rapl, nvidia-ml) that may not be available; measurements vary by environment
- Impact: Energy metrics may be missing or inaccurate in some environments
- Migration plan: Graceful degradation already in place; document limitations per environment type

**Pydantic V2 Migration:**
- Risk: Codebase uses Pydantic V2 patterns (`model_dump()`, `ConfigDict`). Some dependencies may still require V1.
- Impact: Dependency conflicts if libraries require `pydantic<2.0`
- Migration plan: Already on V2. Monitor for dependency conflicts. Use `pydantic-settings` for env var integration if needed. Keep Pydantic v2 as hard requirement.

## Missing Critical Features

**Docker GPU Passthrough Working:**
- Problem: All three backends fail in Docker with CUDA driver errors. Docker execution completely broken.
- Blocks: Isolated/reproducible experiments, multi-user environments, deployment
- Priority: CRITICAL

**Resume/Checkpoint System:**
- Problem: Multi-cycle experiments with failures cannot resume cleanly; must re-run from start. TODO comment suggests incomplete implementation.
- Blocks: Statistical robustness testing (multi-cycle resumption after transient failures)
- Priority: Medium - workaround exists (manual restart) but efficiency loss

**No Baseline Measurements:**
- Problem: Tool measures model inference but provides no idle GPU baseline or system overhead measurements. Cannot isolate model energy from background GPU power.
- Blocks: Accurate energy attribution, comparison across different hardware setups
- Priority: Medium

**No Multi-Node GPU Support:**
- Problem: vLLM and TensorRT support multi-node tensor parallelism but framework only handles single-node execution
- Blocks: Scaling to models requiring >8 GPUs (e.g. 70B+ models across 4 nodes with 8 GPUs each)
- Priority: Low (most research is single-node)

**No Result Database:**
- Problem: All results stored as JSON files. No structured query interface, no aggregation across campaigns, no historical comparison.
- Blocks: Meta-analysis, trend detection, experiment comparison UI
- Priority: Low (research workflows use notebooks for analysis)

**Backend-Specific Metric Collection:**
- Problem: Some backend-native metrics (vLLM speculative decoding acceptance rate, TensorRT quantization stats) not collected or exposed
- Blocks: Deep backend performance analysis
- Priority: Low - nice-to-have for advanced analysis

**Distributed Multi-GPU Aggregation:**
- Problem: Aggregation is single-process only; large result sets load all into one process
- Blocks: Scaling to 10,000+ request experiments across many GPUs
- Priority: Low - most experiments smaller, can be optimized later

## Test Coverage Gaps

**Docker GPU Passthrough (CRITICAL):**
- What's not tested: GPU visibility inside Docker containers, CUDA environment variable propagation, multi-GPU container execution
- Files: No integration tests in `tests/integration/` for Docker GPU access
- Risk: Docker execution completely broken (as currently observed) with no test detection. All three backends fail.
- Priority: CRITICAL

**Backend CUDA Initialization Order (CRITICAL):**
- What's not tested: Validation that vLLM/TensorRT backends don't have CUDA pre-initialized by orchestration layer
- Files: No test validates `orchestrator_may_call_cuda=False` enforcement
- Risk: CUDA fork errors in production, hard to debug. Currently breaking vLLM/TensorRT in Docker.
- Priority: CRITICAL

**Untested Subprocess Signal Handling:**
- What's not tested: SIGINT/SIGTERM handling during vLLM/TensorRT subprocess execution, graceful shutdown timing
- Files: `src/llenergymeasure/cli/experiment.py` lines 325-375, 788-810
- Risk: Ungraceful shutdown could leave GPU memory occupied; timeout may not trigger correctly
- Priority: High

**vLLM Backend Multiprocessing Edge Cases:**
- What's not tested: vLLM with >2 GPU parallelism, tensor + pipeline parallelism combinations, distributed executor failure modes
- Files: `src/llenergymeasure/core/inference_backends/vllm.py`
- Risk: Crashes or hangs in multi-GPU setups. Currently failing in Docker with tensor_parallel_size=2.
- Priority: High

**Results Aggregation Partial Failures:**
- What's not tested: Aggregation with some processes missing, duplicate process results, corrupted marker files
- Files: `src/llenergymeasure/results/aggregation.py`
- Risk: Incomplete aggregation proceeding silently if validation overridden
- Priority: High

**Campaign Resumption with Corrupt State:**
- What's not tested: Resume when manifest says "completed" but result files missing, or when process killed mid-write
- Files: `tests/` has no campaign resumption integration tests
- Risk: Resume fails silently or produces incorrect results
- Priority: Medium

**Multi-GPU Parallelism Edge Cases:**
- What's not tested: What happens when `num_processes > len(gpus)`, when MIG UUIDs used instead of indices, when CUDA_VISIBLE_DEVICES already set on host
- Files: `tests/unit/` has basic parallelism tests, no edge cases
- Risk: Misconfigured experiments run on wrong GPUs or fail cryptically
- Priority: Medium

**Config Provenance After Schema Changes:**
- What's not tested: What happens to parameter_provenance dict when config schema evolves (field renamed, moved to different section, deleted)
- Files: No tests for config schema evolution
- Risk: Provenance tracking breaks silently, results missing metadata
- Priority: Medium

**Config Inheritance Complex Cases:**
- What's not tested: Deep inheritance chains (A → B → C), conflicting overrides in inheritance, list merging behavior
- Files: `src/llenergymeasure/config/loader.py`
- Risk: Silent config corruption in complex inheritance scenarios
- Priority: Medium

**Extended Metrics Null Scenarios:**
- What's not tested: All combinations of unavailable extended metrics (e.g., pynvml + no streaming + no energy), downstream handling of nulls
- Files: `src/llenergymeasure/core/extended_metrics.py`, `src/llenergymeasure/results/aggregation.py`
- Risk: Downstream code assumes metric availability without checking, crashes on None
- Priority: Medium

**Thread Safety in Background Sampling:**
- What's not tested: Concurrent access to sampler._samples during active sampling, exception during sampling, multiple sampler instances
- Files: `src/llenergymeasure/core/gpu_utilisation.py`
- Risk: Race conditions in multi-threaded contexts; sampling corruption
- Priority: Medium

**Large-Scale Grid Expansion:**
- What's not tested: Grid expansion with 100+ configs, file system limits, memory usage for large Cartesian products
- Files: Grid expansion tested with small examples only
- Risk: Campaign startup fails or OOM for large parameter sweeps
- Priority: Low

**Streaming Latency with Slow Models:**
- What's not tested: TTFT/ITL measurement when model generation is very slow (>30s per token) or when streaming times out
- Files: Streaming tests use fast models only
- Risk: Timeout handling incorrect, metrics missing for slow generation
- Priority: Low

---

*Concerns audit: 2026-02-05*
