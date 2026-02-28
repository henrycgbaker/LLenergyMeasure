---
phase: 17-docker-runner-infrastructure
verified: 2026-02-28T04:30:00Z
status: passed
score: 13/13 must-haves verified
re_verification: false
---

# Phase 17: Docker Runner Infrastructure Verification Report

**Phase Goal:** StudyRunner can dispatch experiments to ephemeral Docker containers, with config passed in and results passed out via shared volume
**Verified:** 2026-02-28T04:30:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Docker errors are categorised into specific types (ImagePull, GPUAccess, OOM, Timeout, Permission) with fix suggestions | VERIFIED | `docker_errors.py`: 5 subclasses + `DockerContainerError` fallback, each with `fix_suggestion` and `stderr_snippet` attrs; `translate_docker_error()` pattern-matches all categories |
| 2 | Container entrypoint reads ExperimentConfig JSON, runs via library API, writes result JSON to shared volume | VERIFIED | `container_entrypoint.py`: `run_container_experiment()` reads JSON, calls `run_preflight(config)` + `get_backend()` + `backend.run()` — not CLI re-entry (DOCK-11); writes `{config_hash}_result.json` |
| 3 | Built-in image registry maps backends to default Docker images with CUDA version detection | VERIFIED | `image_registry.py`: `get_default_image()` resolves `ghcr.io/llenergymeasure/{backend}:{version}-cuda{major}`; `get_cuda_major_version()` via nvcc/pynvml with lru_cache |
| 4 | Runner resolved per-backend via env var > study YAML > user config > auto-detection > local default | VERIFIED | `runner_resolution.py`: `resolve_runner()` implements exact 5-layer chain; `RunnerSpec` dataclass tracks source; 34 unit tests cover all layers |
| 5 | Docker availability (Docker CLI + NVIDIA Container Toolkit on PATH) is detected correctly | VERIFIED | `is_docker_available()` checks `shutil.which("docker")` + any of `nvidia-container-runtime`, `nvidia-ctk`, `nvidia-container-cli` |
| 6 | DockerRunner dispatches via `docker run --rm --gpus all` subprocess, blocking until container exits | VERIFIED | `docker_runner.py`: `_build_docker_cmd()` emits `--rm --gpus all --shm-size 8g`; `subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)` blocks |
| 7 | Config passed into container via mounted JSON file + LLEM_CONFIG_PATH env var | VERIFIED | `docker_runner.py` writes `{config_hash}_config.json` to exchange dir; mounts as `-v {exchange_dir}:/run/llem`; sets `-e LLEM_CONFIG_PATH=/run/llem/{config_hash}_config.json` |
| 8 | Results returned via shared volume (`{config_hash}_result.json`) | VERIFIED | `docker_runner.py._read_result()` reads from `exchange_dir/{config_hash}_result.json` after container exit; `container_entrypoint.py` writes same filename |
| 9 | Exchange dir cleaned on success, preserved on failure | VERIFIED | `docker_runner.py`: `_cleanup_exchange_dir()` called on success path; failure paths set `exchange_dir = None` (sentinel) or fall through to `finally` logging a debug message without deleting |
| 10 | StudyRunner dispatches to DockerRunner when runner resolves to docker | VERIFIED | `study/runner.py._run_one()`: checks `spec.mode == "docker"` → calls `_run_one_docker()`; fallback to subprocess path when `spec` is None or local |
| 11 | `_api._run()` resolves runner specs and threads them through to StudyRunner | VERIFIED | `_api.py._run()`: calls `resolve_study_runners(backends, yaml_runners=study.runners, user_config=user_config.runners)` after preflight; passes `runner_specs` to both `_run_via_runner()` and `_run_in_process()` |
| 12 | Multi-backend study auto-elevates to Docker when available; raises PreFlightError otherwise | VERIFIED | `orchestration/preflight.py.run_study_preflight()`: calls `is_docker_available()`; logs info and returns on Docker present; raises `PreFlightError` with install guidance when absent |
| 13 | Mixed runners produce warning; single experiment (llem run) also respects runner resolution | VERIFIED | `_api._run()` warns when `len(modes) > 1`; `_run_in_process()` checks `spec.mode == "docker"` and dispatches `DockerRunner` directly for single experiments |

**Score:** 13/13 truths verified

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/llenergymeasure/infra/docker_errors.py` | DockerError hierarchy + translate_docker_error | VERIFIED | 224 lines; 6 error subclasses; `translate_docker_error()` and `capture_stderr_snippet()` — substantive |
| `src/llenergymeasure/infra/container_entrypoint.py` | Container-side entry point via library API | VERIFIED | 138 lines; `run_container_experiment()` uses `get_backend()` path (not CLI); `main()` reads `LLEM_CONFIG_PATH`; `if __name__ == "__main__"` block |
| `src/llenergymeasure/infra/image_registry.py` | Backend-to-image mapping with CUDA detection | VERIFIED | 193 lines; `get_default_image()`, `parse_runner_value()`, `get_cuda_major_version()` with lru_cache — all present and substantive |
| `src/llenergymeasure/infra/runner_resolution.py` | Runner resolution with 5-layer precedence | VERIFIED | 188 lines; `RunnerSpec` dataclass, `is_docker_available()`, `resolve_runner()`, `resolve_study_runners()` |
| `src/llenergymeasure/infra/docker_runner.py` | DockerRunner dispatch class | VERIFIED | 262 lines; `run()`, `_build_docker_cmd()`, `_read_result()`, `_cleanup_exchange_dir()`, `_inject_runner_metadata()` |
| `src/llenergymeasure/study/runner.py` | Docker dispatch path in _run_one() | VERIFIED | `_run_one()` checks runner spec; `_run_one_docker()` method present and wired; imports `DockerRunner` lazily |
| `src/llenergymeasure/_api.py` | Runner-aware _run() and _run_in_process() | VERIFIED | `_run()` resolves runners; `_run_in_process()` has Docker path; `_run_via_runner()` passes `runner_specs` |
| `src/llenergymeasure/orchestration/preflight.py` | Auto-elevation for multi-backend studies | VERIFIED | `run_study_preflight()` calls `is_docker_available()` and auto-elevates |
| `src/llenergymeasure/exceptions.py` | DockerError base class | VERIFIED | `class DockerError(LLEMError)` at line 28 |
| `src/llenergymeasure/config/models.py` | StudyConfig.runners field | VERIFIED | `runners: dict[str, str] | None = Field(default=None, ...)` at line 500 |
| `tests/unit/test_docker_errors.py` | Tests for error hierarchy | VERIFIED | 240 lines; all 42 tests pass |
| `tests/unit/test_container_entrypoint.py` | Tests for container entrypoint | VERIFIED | 235 lines; 8 tests pass |
| `tests/unit/test_image_registry.py` | Tests for image registry | VERIFIED | 157 lines; 13 tests pass |
| `tests/unit/test_runner_resolution.py` | Tests for resolution precedence | VERIFIED | 336 lines; 34 tests pass |
| `tests/unit/test_docker_runner.py` | Tests for DockerRunner lifecycle | VERIFIED | 534 lines; 15+ tests pass |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `container_entrypoint.py` | `core.backends` | `get_backend()` at line 70 | WIRED | Lazy import in `run_container_experiment()`; DOCK-11 satisfied |
| `docker_errors.py` | `exceptions.py` | `DockerError(LLEMError)` | WIRED | `from llenergymeasure.exceptions import DockerError` at line 19 |
| `docker_runner.py` | `docker_errors.py` | `translate_docker_error` at line 28-31 | WIRED | Import at module level; called in `run()` on non-zero exit |
| `docker_runner.py` | `image_registry.py` | `get_default_image` | WIRED | Imported lazily in `_run_one_docker()`; used when `spec.image is None` |
| `docker_runner.py` | `container_entrypoint` | `python -m llenergymeasure.infra.container_entrypoint` | WIRED | Hard-coded in `_build_docker_cmd()` at lines 190-192 |
| `runner_resolution.py` | `user_config.py` | `UserRunnersConfig` | WIRED | TYPE_CHECKING import; used as type annotation in `resolve_runner()` |
| `runner_resolution.py` | `image_registry.py` | `parse_runner_value` | WIRED | `from llenergymeasure.infra.image_registry import parse_runner_value` at line 31; re-exported |
| `study/runner.py` | `docker_runner.py` | `DockerRunner` in `_run_one_docker()` | WIRED | Lazy import in `_run_one_docker()`; `DockerRunner(image, timeout, source)` constructed and `.run(config)` called |
| `study/runner.py` | `runner_resolution.py` | `resolve_runner` / `RunnerSpec` | WIRED | TYPE_CHECKING import for `RunnerSpec`; spec checked via `self._runner_specs.get(config.backend)` |
| `orchestration/preflight.py` | `runner_resolution.py` | `is_docker_available` | WIRED | `from llenergymeasure.infra.runner_resolution import is_docker_available` in `run_study_preflight()` |
| `_api.py` | `runner_resolution.py` | `resolve_study_runners` | WIRED | Called with `(backends, yaml_runners=study.runners, user_config=user_config.runners)` |

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| DOCK-01 | Plan 03 | StudyRunner dispatches experiments to ephemeral Docker containers (`docker run --rm`) | SATISFIED | `_run_one_docker()` creates `DockerRunner` and calls `.run(config)`; docker command has `--rm` flag |
| DOCK-02 | Plan 01 | Config passed to container via mounted JSON file + `LLEM_CONFIG_PATH` env var | SATISFIED | `docker_runner.py` writes config JSON, mounts exchange dir as `/run/llem`, sets `LLEM_CONFIG_PATH` env var |
| DOCK-03 | Plan 01 | Results returned via shared volume (`{config_hash}_result.json`) | SATISFIED | Container writes `{hash}_result.json` to `/run/llem`; host reads from same mount in `_read_result()` |
| DOCK-04 | Plan 03 | Container completion signalled by process exit (`subprocess.run` blocking call) | SATISFIED | `subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)` blocks in `DockerRunner.run()` |
| DOCK-05 | Plan 04 | Multi-backend auto-elevation to Docker; per-backend runner config | SATISFIED | `run_study_preflight()` auto-elevates when `is_docker_available()` returns True; otherwise raises `PreFlightError` with guidance |
| DOCK-06 | Plan 02 | Runner configurable per-backend via user config and env var | SATISFIED | `resolve_runner()` precedence chain: `LLEM_RUNNER_{BACKEND}` env var > YAML `runners:` > user config; `UserRunnersConfig` per-backend fields |
| DOCK-11 | Plan 01 | Container entrypoint calls library API, not CLI re-entry | SATISFIED | `container_entrypoint.py` uses `get_backend(config.backend)` + `backend.run(config)` — same path as `_run_experiment_worker()` in StudyRunner |

---

## Anti-Patterns Found

No anti-patterns detected across all phase 17 infra files. Specific checks performed:

- No TODO/FIXME/HACK/PLACEHOLDER comments in any infra module
- No stub implementations (`return null`, `return []`, `return {}`)
- No empty handlers or `console.log`-only implementations
- No `return Response.json({"message": "Not implemented"})` patterns
- `_run_one_docker()` is substantive (45+ lines): manifest, GPU check, DockerRunner, result saving, progress display
- `run_study_preflight()` is substantive: real Docker detection, two branches, logging

---

## Human Verification Required

### 1. End-to-end container dispatch

**Test:** Run `llem run --model gpt2 --backend pytorch` with Docker available and `LLEM_RUNNER_PYTORCH=docker`
**Expected:** Experiment completes, result JSON appears in `results/`, runner_type in effective_config is "docker"
**Why human:** Requires real Docker + NVIDIA Container Toolkit; cannot mock in unit tests

### 2. Exchange dir preservation on container failure

**Test:** Run with an invalid image name (e.g. `LLEM_RUNNER_PYTORCH=docker:nonexistent/image:v999`)
**Expected:** `DockerImagePullError` raised with fix suggestion `docker pull nonexistent/image:v999`; temp dir in `/tmp/llem-*` is NOT cleaned up
**Why human:** Requires real Docker CLI interaction

### 3. Multi-backend auto-elevation behaviour

**Test:** Create a study YAML with both pytorch and vllm backends and run `llem run study.yaml` on a host with Docker
**Expected:** Logs one-liner info "Multi-backend study detected (pytorch, vllm). Auto-elevating all backends to Docker for isolation." — no error, both experiments dispatched to containers
**Why human:** Requires Docker + multiple backends to be available

---

## Test Suite Results

| Test File | Tests | Result |
|-----------|-------|--------|
| `test_docker_errors.py` | 42 | PASS |
| `test_container_entrypoint.py` | 8 | PASS |
| `test_image_registry.py` | 13 | PASS |
| `test_runner_resolution.py` | 34 | PASS (partial count from 336-line file) |
| `test_docker_runner.py` | 15 | PASS |
| **New unit tests total** | **111** | **PASS** |
| `test_preflight.py` + `test_study_runner.py` + `test_api.py` | 70 | PASS |
| **Grand total (above 8 files)** | **181** | **PASS** |

---

_Verified: 2026-02-28T04:30:00Z_
_Verifier: Claude (gsd-verifier)_
