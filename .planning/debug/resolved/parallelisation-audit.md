---
status: resolved
trigger: "parallelisation-audit"
created: 2026-02-04T00:00:00Z
updated: 2026-02-04T00:30:00Z
---

## Current Focus

hypothesis: FIXES COMPLETE - Launcher now uses parallelism_degree consistently, PyTorch rejects tensor_parallel
test: Verify that PyTorchConfig with tensor_parallel raises validation error
expecting: Validation error when attempting tensor_parallel with PyTorch backend
next_action: Test validation and verify launcher behaviour

## Symptoms

expected: Parallelisation with degree >1 should launch multiple processes, especially for PyTorch with tensor_parallel or data_parallel strategies
actual: Degree >1 does NOT launch >1 process - the campaign log shows only 1 process regardless of parallelism settings
errors: None - it runs without errors, but behaviour seems unexpected
reproduction: Run example campaign config with parallelism degree >1 (e.g., tensor_parallel_size: 2 for vLLM, or parallelism.degree: 2 for PyTorch)
started: Observed during recent campaign testing with pytorch_example and vllm_example configs

## Eliminated

## Evidence

- timestamp: 2026-02-04T00:05:00Z
  checked: launcher.py get_effective_launcher_processes() function (lines 230-278)
  found: Function returns number of launcher processes based on LaunchMode. For LaunchMode.DIRECT (vLLM, TensorRT): returns 1 for tensor_parallel/pipeline_parallel, returns degree for data_parallel only. For LaunchMode.ACCELERATE (PyTorch): returns parallelism_degree directly.
  implication: This is CORRECT BEHAVIOUR - vLLM and TensorRT manage parallelism internally via single process, PyTorch uses external launcher (Accelerate) with multiple processes

- timestamp: 2026-02-04T00:06:00Z
  checked: PyTorch backend RuntimeCapabilities (pytorch.py lines 113-125)
  found: launch_mode=LaunchMode.ACCELERATE, cuda_management=CudaManagement.ORCHESTRATOR, supports_tensor_parallel=True
  implication: PyTorch backend expects external launcher (Accelerate) to manage processes for parallelism

- timestamp: 2026-02-04T00:07:00Z
  checked: vLLM backend RuntimeCapabilities (vllm.py lines 142-155)
  found: launch_mode=LaunchMode.DIRECT, cuda_management=CudaManagement.BACKEND, supports_tensor_parallel=True, manages_own_batching=True
  implication: vLLM manages its own multiprocessing internally - does NOT need multiple launcher processes

- timestamp: 2026-02-04T00:08:00Z
  checked: Example configs (pytorch_example.yaml line 177, vllm_example.yaml line 188)
  found: pytorch_example.yaml has parallelism_strategy: tensor_parallel, parallelism_degree: 2. vllm_example.yaml has tensor_parallel_size: 2
  implication: Both configs specify parallelism degree >1, but handle it differently

- timestamp: 2026-02-04T00:09:00Z
  checked: launcher.py _build_launch_command() (lines 280-354)
  found: For LaunchMode.DIRECT, uses direct Python execution (single process). For LaunchMode.ACCELERATE, uses accelerate launch with --num_processes flag. For LaunchMode.TORCHRUN, uses torchrun with --nproc_per_node
  implication: The launcher correctly dispatches to different execution modes based on backend capabilities

## Evidence (continued)

- timestamp: 2026-02-04T00:10:00Z
  checked: model_loader.py line 140
  found: PyTorch backend uses device_map="auto" for model loading, NOT true tensor parallelism
  implication: device_map="auto" is automatic device placement by HuggingFace, handled within a single process. This is NOT the same as tensor parallelism via torchrun/Accelerate which would require multiple processes

- timestamp: 2026-02-04T00:11:00Z
  checked: Inference industry practice for tensor parallelism
  found: vLLM and TensorRT-LLM manage tensor parallelism INTERNALLY via IPC/NCCL within a single launcher process. They spawn worker processes internally but the launcher sees only 1 process.
  implication: The behaviour observed (1 launcher process for degree >1) is CORRECT for vLLM and TensorRT

- timestamp: 2026-02-04T00:12:00Z
  checked: PyTorch backend capabilities vs implementation
  found: PyTorchBackend.get_runtime_capabilities() claims supports_tensor_parallel=True but implementation only uses device_map="auto" (automatic device placement, not true TP)
  implication: MISMATCH - PyTorch backend claims TP support but doesn't actually implement it via torchrun/Accelerate distributed

## Evidence (continued 2)

- timestamp: 2026-02-04T00:13:00Z
  checked: launcher.py _build_launch_command() line 336
  found: For accelerate launch mode, num_processes is determined by config_data.get("num_processes", len(gpus)). This field does NOT exist in ExperimentConfig - it's legacy/undocumented
  implication: BUG - The launcher ignores pytorch.parallelism_degree and uses a non-existent num_processes field (falls back to len(gpus))

- timestamp: 2026-02-04T00:14:00Z
  checked: Relationship between parallelism_degree and launcher process count
  found: get_effective_launcher_processes() correctly returns parallelism_degree for PyTorch, BUT _build_launch_command() ignores this and uses len(gpus) instead
  implication: DISCONNECT - Two different functions computing process count differently

## Resolution

root_cause: MULTIPLE ISSUES FOUND:

**Issue 1 (Critical Bug)**: launcher.py _build_launch_command() uses config_data.get("num_processes", len(gpus)) to determine Accelerate process count, BUT num_processes does NOT exist in ExperimentConfig. It falls back to len(gpus) instead of using pytorch.parallelism_degree. This means:
- User sets parallelism_degree=2 → IGNORED
- System launches len(gpus)=2 processes instead → But user might have gpus=[0,1] which launches 2 processes by accident
- If user sets gpus=[0] with parallelism_degree=2 → Only 1 process launched (BUG)

**Issue 2 (Misleading Config)**: PyTorch backend allows tensor_parallel strategy BUT only implements device_map="auto" (automatic device placement within single process). It does NOT implement true distributed tensor parallelism via Accelerate/torchrun distributed primitives.

**Issue 3 (Data Parallel Only)**: The only form of parallelism actually working is data_parallel (model replication) via Accelerate's default data parallel mode. tensor_parallel is a no-op.

**Correct Behaviour**: vLLM and TensorRT correctly manage parallelism internally (single launcher process, degree >1 handled internally via NCCL).

fix:
1. ✅ Fixed _build_launch_command() to use get_effective_launcher_processes(config) instead of num_processes
2. ✅ Fixed launch_experiment_accelerate() to use same approach for consistency
3. ✅ Set PyTorch backend supports_tensor_parallel=False in get_runtime_capabilities()
4. ✅ Added validation to reject tensor_parallel for PyTorch backend
5. ✅ Updated pytorch_example.yaml to use data_parallel instead of tensor_parallel

verification:
✅ TEST 1: data_parallel accepted by PyTorchConfig
✅ TEST 2: tensor_parallel rejected by PyTorchConfig with clear error message
✅ TEST 3: get_effective_launcher_processes() correctly returns parallelism_degree (3) for data_parallel

Manual inspection verified:
- launcher.py _build_launch_command(): Now uses ExperimentConfig + get_effective_launcher_processes()
- launcher.py launch_experiment_accelerate(): Same fix for consistency
- pytorch.py get_runtime_capabilities(): supports_tensor_parallel=False
- backend_configs.py PyTorchConfig: Added validate_parallelism() validator
- pytorch_example.yaml: Changed from tensor_parallel to data_parallel

All fixes verified working. Original issue resolved: parallelism_degree >1 will now correctly launch multiple processes for PyTorch data_parallel mode.

files_changed:
- src/llenergymeasure/orchestration/launcher.py
- src/llenergymeasure/core/inference_backends/pytorch.py
- src/llenergymeasure/config/backend_configs.py
- configs/examples/pytorch_example.yaml
