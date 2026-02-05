---
phase: 03-gpu-routing-fix
plan: 03
subsystem: testing
tags: [unit-tests, gpu, validation, docker, parallelism, vllm, tensorrt, pytorch]

# Dependency graph
requires:
  - phase: 03-01
    provides: "GPU env var propagation to Docker containers"
  - phase: 03-02
    provides: "Fail-fast parallelism validation"
provides:
  - "Unit tests for parallelism constraint validation"
  - "Unit tests for Docker GPU env propagation"
  - "Manual verification of multi-GPU scenarios"
affects: [future regression testing, CI pipelines]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Unit tests verify arithmetic validation (tp_size <= len(config.gpus))"
    - "Unit tests verify Docker command construction with GPU env vars"

key-files:
  created:
    - "tests/unit/config/test_parallelism_validation.py"
    - "tests/unit/cli/test_docker_gpu_propagation.py"
  modified: []

key-decisions:
  - "config.gpus is declarative (specifies expected GPU count, not hardware probe)"
  - "Validation is arithmetic (checks tp_size <= len(config.gpus))"
  - "Container-side adapts to actual NVIDIA_VISIBLE_DEVICES from SLURM/runtime"
  - "HPC compatibility confirmed: GPUs not visible on host but available in containers"

patterns-established:
  - "GPU routing tests verify both host index propagation and container remapping"
  - "Parallelism tests cover all three backends (vLLM, TensorRT, PyTorch)"

# Metrics
duration: 8min
completed: 2026-02-04
---

# Phase 03 Plan 03: GPU Routing Unit Tests Summary

**Comprehensive unit tests for GPU routing fixes, plus manual verification confirming HPC compatibility**

## Performance

- **Duration:** 8 min
- **Started:** 2026-02-04T18:38:50Z
- **Completed:** 2026-02-04T18:46:16Z
- **Tasks:** 3 (2 auto + 1 checkpoint)
- **Files created:** 2

## Accomplishments

- Added 18 unit tests for parallelism constraint validation (vLLM, TensorRT, PyTorch)
- Added 16 unit tests for Docker GPU environment variable propagation
- Manual verification confirmed GPU routing works correctly in HPC environment
- Total: 34 new tests, all passing

## Task Commits

Each task was committed atomically:

1. **Task 1: Add parallelism validation unit tests** - `3d00b84` (test)
2. **Task 2: Add GPU env propagation unit tests** - `688f486` (test)
3. **Task 3: Manual verification checkpoint** - Approved by user

## Files Created

- `tests/unit/config/test_parallelism_validation.py` - 18 tests covering:
  - vLLM tensor_parallel_size vs gpus validation
  - vLLM pipeline parallel (TP * PP) validation
  - TensorRT tp_size and pp_size validation
  - PyTorch num_processes validation
  - Edge cases: empty gpus, non-matching backend, suggestions

- `tests/unit/cli/test_docker_gpu_propagation.py` - 16 tests covering:
  - NVIDIA_VISIBLE_DEVICES host GPU index propagation
  - CUDA_VISIBLE_DEVICES remapping to container indices (0,1,2,...)
  - Container context detection logic
  - Campaign context propagation
  - Docker command structure and format

## Decisions Made

- **config.gpus is declarative:** Specifies expected GPU count for the experiment, not a hardware probe. This works correctly in HPC environments where GPUs aren't visible on the host but become available inside containers via SLURM allocation.

- **Validation is arithmetic:** Checks `tp_size <= len(config.gpus)` without probing actual hardware. This allows validation to run on any machine (including login nodes without GPUs).

- **Container adapts to runtime:** Inside containers, the `_early_cuda_visible_devices_setup()` function detects whether NVIDIA_VISIBLE_DEVICES is set by the runtime and adapts CUDA_VISIBLE_DEVICES accordingly.

## Manual Verification Results

User confirmed GPU routing works correctly in their HPC environment:
- Parallelism validation catches invalid configs with clear error messages
- Valid configs pass validation without false positives
- Container-side GPU detection adapts to SLURM-allocated GPUs

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- Pre-existing test failures in test suite (unrelated to new tests) due to read-only /tmp in sandboxed environment
- All 34 new tests pass consistently

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 03 (GPU Routing Fix) is now complete
- All three plans delivered:
  - 03-01: GPU env var propagation to Docker containers
  - 03-02: Fail-fast parallelism validation
  - 03-03: Unit tests and manual verification
- GPU routing bug fully resolved with comprehensive test coverage

---
*Phase: 03-gpu-routing-fix*
*Completed: 2026-02-04*
