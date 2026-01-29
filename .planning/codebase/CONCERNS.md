# Codebase Concerns

**Analysis Date:** 2026-01-26

## Tech Debt

**Resume Logic Incomplete:**
- Issue: Resume functionality marked as TODO but not fully implemented
- Files: `src/llenergymeasure/cli/experiment.py` (line 583)
- Impact: Experiments that are interrupted cannot be truly resumed - only fresh starts are possible. State tracking infrastructure exists but actual resumption logic missing
- Fix approach: Implement checkpoint loading in orchestrator to skip completed processes and continue from current state

**Large Monolithic Backend Files:**
- Issue: Inference backend files (pytorch.py, vllm.py, tensorrt.py) are 1000+ lines each, mixing initialization, configuration building, execution, and error handling
- Files: `src/llenergymeasure/core/inference_backends/pytorch.py` (1149 lines), `src/llenergymeasure/core/inference_backends/vllm.py` (1006 lines), `src/llenergymeasure/core/inference_backends/tensorrt.py` (1168 lines)
- Impact: Difficult to test individual concerns, high cognitive load for maintenance, risk of bugs in config building or error paths
- Fix approach: Refactor backends to separate concerns: config building → separate builder classes, initialization → lifecycle manager, execution → separate runner, error handling → decorators or middleware

**Model Kwargs Manual Passing in PyTorch:**
- Issue: PyTorch backend has TODO comment about passing model_kwargs to loader when supported (line 369)
- Files: `src/llenergymeasure/core/inference_backends/pytorch.py` (line 369)
- Impact: Extra model configuration parameters can't flow through the loader, limiting control over model initialization
- Fix approach: Update model_loader to accept **model_kwargs parameter and pass through to AutoModelForCausalLM

**SSOT Introspection Module Hard to Maintain:**
- Issue: `config/introspection.py` (807 lines) contains hand-written mappings for parameter discovery via AST comment (line 763)
- Files: `src/llenergymeasure/config/introspection.py` (line 763)
- Impact: Parameter metadata must be manually synchronised across SSOT sources; missing parameters may not auto-discover; comment indicates future improvement needed
- Fix approach: Consider using Pydantic model introspection more robustly or generate metadata at build time via pre-commit hooks

**Large Display Summary Module:**
- Issue: `cli/display/summaries.py` (804 lines) handles all config/state summary rendering in one module
- Files: `src/llenergymeasure/cli/display/summaries.py`
- Impact: Hard to test display logic, mixing data preparation and Rich output formatting
- Fix approach: Separate data collection (config summaries) from rendering (Rich panels)

## Known Bugs

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
- Files: `src/llenergymeasure/results/aggregation.py` (lines 42-112)
- Trigger: Process crashes without writing completion marker, or marker file corruption
- Workaround: Validation reports missing/duplicate processes but aggregation can continue if user overrides
- Root cause: Marker files written atomically but result files may exist without markers if process crashes between result write and marker write

## Security Considerations

**Environment Variable Exposure in Subprocess:**
- Risk: HF_TOKEN propagated to child processes via subprocess_env - could be visible in process listing briefly
- Files: `src/llenergymeasure/cli/experiment.py` (lines 623-626)
- Current mitigation: Token passed only to subprocess environment (not command line args), subprocess runs as same user
- Recommendations: Consider using secure credential passing mechanisms (credentials file, keyring) for sensitive tokens in production; document that env vars visible during process startup

**Config File Path Traversal Prevention:**
- Risk: Config inheritance via `_extends` field could load arbitrary files if path not validated
- Files: `src/llenergymeasure/config/loader.py` (lines 70-100)
- Current mitigation: Uses `Path.resolve()` and cycle detection; no explicit boundary checking
- Recommendations: Add explicit check that inherited configs stay within allowed directory (e.g., same dir or subdirs only); warn on unusual paths

**YAML Deserialization:**
- Risk: YAML parsing with `yaml.safe_load()` is used, but still processes YAML/JSON without strict schema validation before Pydantic
- Files: `src/llenergymeasure/config/loader.py` (line 57)
- Current mitigation: `yaml.safe_load()` used (not `yaml.load()`), Pydantic validates after loading
- Recommendations: Add file size limits to prevent billion laughs attacks; consider pre-validating YAML structure

**Experiment ID Sanitization:**
- Risk: Experiment IDs used in filesystem paths; malformed IDs could cause issues
- Files: `src/llenergymeasure/security.py` (lines 56-78)
- Current mitigation: Alphanumeric + underscore/hyphen/dot allowed, non-conforming chars replaced with underscore
- Recommendations: Ensure all places creating directories/files use sanitised IDs; add tests for edge cases like "../" or special chars

## Performance Bottlenecks

**GPU Memory Tracking Overhead:**
- Problem: `get_memory_stats()` called potentially many times during inference to track peak memory
- Files: `src/llenergymeasure/core/compute_metrics.py`
- Cause: CUDA memory API calls have overhead; sampling on every batch adds up
- Improvement path: Cache memory stats, update only periodically (e.g., per 10 batches); or use single max_memory_allocated() call post-inference instead of continuous tracking

**Late Aggregation Full Dataset Load:**
- Problem: Aggregation loads all raw process results into memory at once for extended metrics computation
- Files: `src/llenergymeasure/results/aggregation.py` (lines 115+)
- Cause: Latency samples and GPU utilisation samples collected as lists in memory
- Improvement path: For very large experiments, stream results and compute running statistics; or use memory-mapped numpy arrays for latency data

**ThreadPoolExecutor Single Worker for Memory Calculation:**
- Problem: Timeout-protected GPU memory calls use ThreadPoolExecutor with single worker
- Files: `src/llenergymeasure/core/compute_metrics.py` (lines 116, 138)
- Cause: Adds context switch overhead for timeout mechanism
- Improvement path: Use signal-based timeout on Linux; or refactor to avoid timeout (pre-allocate GPU memory check)

**File I/O for Completion Markers:**
- Problem: Each process writes a separate marker file after result save - can cause filesystem thrashing in high-process counts
- Files: `src/llenergymeasure/orchestration/runner.py` (lines 44-78)
- Cause: Atomic write via temp + rename for each process marker
- Improvement path: Batch marker writes; or use single completion log file with process indices

## Fragile Areas

**Streaming Latency with TextIteratorStreamer Threading:**
- Files: `src/llenergymeasure/core/inference_backends/pytorch.py` (lines 648-770)
- Why fragile: Uses background thread with TextIteratorStreamer, complex synchronisation between main thread and streamer thread for token timestamp collection, race condition potential if generation fails midway
- Safe modification: Always wrap streamer code in try/finally; ensure thread joins even on exception; test with various generation failure modes
- Test coverage: Limited - needs tests for early stopping, max_length exceeded, and error cases

**vLLM Multiprocessing Configuration:**
- Files: `src/llenergymeasure/cli/experiment.py` (lines 635-643), `src/llenergymeasure/core/inference_backends/vllm.py` (lines 140+)
- Why fragile: Multiple environment variable hacks to work around vLLM multiprocessing issues (V1 multiprocessing disable, spawn method, torch.compile disable). Changes to vLLM versions may break these workarounds
- Safe modification: Test with vLLM version updates before deploying; document exact versions these hacks are for; consider feature detection instead of version hardcoding
- Test coverage: Integration tests exist but only for specific vLLM versions

**GPU Topology Detection and Validation:**
- Files: `src/llenergymeasure/core/gpu_info.py` (482 lines), called from `src/llenergymeasure/cli/experiment.py` (lines 599-605)
- Why fragile: Complex GPU topology detection (MIG instances, nvlink, etc.) with multiple fallback paths; warnings-only approach means misconfigured GPUs silently proceed
- Safe modification: Add strict mode for GPU validation; test on various GPU configurations before changes; validate CUDA_VISIBLE_DEVICES matches detected topology
- Test coverage: Limited to basic cases - needs testing with MIG, nvlink, mixed GPU types

**Results Aggregation State Machine:**
- Files: `src/llenergymeasure/results/aggregation.py` (lines 115-200+), `src/llenergymeasure/state/experiment_state.py` (lines 30-47)
- Why fragile: Complex validation logic for process completion with multiple edge cases (missing indices, duplicates, markers); aggregation can proceed with partial data if validation is overridden
- Safe modification: Add comprehensive logging at each validation step; unit test each validation condition independently; consider making aggregation strict by default
- Test coverage: Basic completeness checks exist, but edge cases like partial writes or concurrent aggregation attempts not tested

**Extended Metrics Null Handling:**
- Files: `src/llenergymeasure/core/extended_metrics.py`, `src/llenergymeasure/results/aggregation.py`
- Why fragile: Extended metrics schema always present but with null values when unavailable; downstream consumers must check for None carefully
- Safe modification: Add type guards; document null semantics; consider using sentinel values instead of None for clarity
- Test coverage: Should test aggregation with various combinations of unavailable metrics

**Config Inheritance with Deep Merge:**
- Files: `src/llenergymeasure/config/loader.py` (lines 20-36, 70-120)
- Why fragile: Deep merge logic for config inheritance can silently drop values if merge order is wrong; cycle detection prevents infinite loops but doesn't prevent logical errors
- Safe modification: Add logging of merge operations; validate final config structure; test with complex inheritance chains
- Test coverage: Basic inheritance works, but edge cases like merging lists or special config sections not thoroughly tested

## Scaling Limits

**Process Result File Accumulation:**
- Current capacity: Stores raw per-process results as separate JSON files; experiments with 100+ processes × multiple cycles = 100s of files
- Limit: Filesystem inode limits on some systems; directory listing performance degrades with many files
- Scaling path: Implement result sharding (subdirectories per GPU or process range); consider database backend for results instead of filesystem

**Extended Metrics Late Aggregation Dataset Size:**
- Current capacity: Aggregation collects all per-request latencies and GPU samples in memory
- Limit: Large experiments (1000+ requests × 100+ processes) = millions of float values in memory
- Scaling path: Streaming aggregation with running statistics; or memory-mapped numpy arrays for sample data

**State File Lookups:**
- Current capacity: StateManager iterates all state files to find by config hash
- Limit: With 1000+ experiments, linear scan becomes slow
- Scaling path: Index state files by config hash; use simple SQLite for state store instead of JSON files

## Dependencies at Risk

**vLLM Backend Stability:**
- Risk: vLLM has frequent API changes; multiple version-specific workarounds in codebase (multiprocessing flags, cuda init)
- Impact: Updates may break inference or cause version conflicts
- Migration plan: Pin to LTS vLLM version; maintain compatibility matrix in docs; consider abstracting vLLM init behind stable interface

**TensorRT-LLM Build Complexity:**
- Risk: Requires CUDA 12.x and specific GPU types; build process fragile, multi-stage Docker build necessary
- Impact: Setup friction; compatibility issues across environments
- Migration plan: Document exact build requirements; consider pre-built wheel distribution; monitor for lighter TensorRT alternatives

**CodeCarbon Energy Measurement:**
- Risk: Requires system tools (intel-rapl, nvidia-ml) that may not be available; measurements vary by environment
- Impact: Energy metrics may be missing or inaccurate in some environments
- Migration plan: Graceful degradation already in place; document limitations per environment type

**Pydantic v2 Migration Debt:**
- Risk: Currently using Pydantic v2; if dependencies require Pydantic v1, conflicts arise
- Impact: Locking on Pydantic v2; version pins needed
- Migration plan: Monitor dependency updates; keep Pydantic v2 as hard requirement in pyproject.toml

## Missing Critical Features

**Resume/Checkpoint System:**
- Problem: Multi-cycle experiments with failures cannot resume cleanly; must re-run from start
- Blocks: Statistical robustness testing (multi-cycle resumption after transient failures)
- Priority: Medium - workaround exists (manual restart) but efficiency loss

**Distributed Multi-GPU Aggregation:**
- Problem: Aggregation is single-process only; large result sets load all into one process
- Blocks: Scaling to 10,000+ request experiments across many GPUs
- Priority: Medium-Low - most experiments smaller, can be optimized later

**Result Export Streaming:**
- Problem: `results show` and exports load full aggregated result into memory
- Blocks: Interactive exploration of very large experiments
- Priority: Low - most users work with smaller experiments

**Backend-Specific Metric Collection:**
- Problem: Some backend-native metrics (vLLM speculative decoding acceptance rate, TensorRT quantization stats) not collected or exposed
- Blocks: Deep backend performance analysis
- Priority: Low - nice-to-have for advanced analysis

## Test Coverage Gaps

**Untested Subprocess Signal Handling:**
- What's not tested: SIGINT/SIGTERM handling during vLLM/TensorRT subprocess execution, graceful shutdown timing
- Files: `src/llenergymeasure/cli/experiment.py` (lines 325-375, 788-810)
- Risk: Ungraceful shutdown could leave GPU memory occupied; timeout may not trigger correctly
- Priority: High

**vLLM Backend Multiprocessing Edge Cases:**
- What's not tested: vLLM with >2 GPU parallelism, tensor + pipeline parallelism combinations, distributed executor failure modes
- Files: `src/llenergymeasure/core/inference_backends/vllm.py`
- Risk: Crashes or hangs in multi-GPU setups
- Priority: High

**Config Inheritance Complex Cases:**
- What's not tested: Deep inheritance chains (A → B → C), conflicting overrides in inheritance, list merging behavior
- Files: `src/llenergymeasure/config/loader.py`
- Risk: Silent config corruption in complex inheritance scenarios
- Priority: Medium

**Results Aggregation Partial Failures:**
- What's not tested: Aggregation with some processes missing, duplicate process results, corrupted marker files
- Files: `src/llenergymeasure/results/aggregation.py`
- Risk: Incomplete aggregation proceeding silently if validation overridden
- Priority: High

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

**Docker TensorRT Build Correctness:**
- What's not tested: TensorRT Docker image builds, engine caching, multi-stage build artifact correctness
- Files: `Dockerfile` (TensorRT section), `docker-compose.yml`
- Risk: Docker images built but untested; bad engines cached and reused
- Priority: Medium

---

*Concerns audit: 2026-01-26*
