---
phase: 22-testing-and-ci
plan: "02"
subsystem: ci
tags: [ci, github-actions, testing, docker]
dependency_graph:
  requires: [22-01]
  provides: [tier1-ci, tier2-gpu-ci]
  affects: [.github/workflows/ci.yml, .github/workflows/gpu-ci.yml]
tech_stack:
  added: [dorny/paths-filter@v3, astral-sh/setup-uv@v7, actions/cache@v4, pytest-xdist]
  patterns: [path-filtered-ci, tiered-ci, uv-lockfile-caching]
key_files:
  created:
    - .github/workflows/gpu-ci.yml
    - tests/fixtures/model_list.txt
  modified:
    - .github/workflows/ci.yml
decisions:
  - Tier 1 test job skips on docs-only changes via dorny/paths-filter (src filter: src/**, tests/**, pyproject.toml, uv.lock)
  - package-validation runs always (not path-filtered) — packaging bugs can come from any change
  - docker-smoke only runs on docker/** / .dockerignore / pyproject.toml changes
  - Tier 2 uses push to main trigger (not PR) — GPU tests run post-merge only
  - No -n auto in Tier 2 — GPU tests must run serially (single GPU, one experiment at a time)
  - --extra vllm excluded from Tier 2 uv sync — vLLM requires Docker container runtime
  - HF cache keyed on model_list.txt hash with restore-keys fallback for cold cache
metrics:
  duration: 83s
  completed: "2026-03-04"
  tasks_completed: 2
  files_changed: 3
---

# Phase 22 Plan 02: CI Rewrite (Tier 1 + Tier 2) Summary

Rewrote both CI workflow files to establish a fit-for-purpose tiered CI pipeline: Tier 1 on GitHub-hosted runners with path filtering and package validation; Tier 2 on self-hosted DS01 for post-merge GPU integration tests with HuggingFace model caching.

## Tasks Completed

| Task | Name | Commit | Files |
| --- | --- | --- | --- |
| 1 | Tier 1 CI — collapse matrix, add path filter, Docker smoke, package validation | 90bb294 | .github/workflows/ci.yml |
| 2 | Tier 2 GPU CI — rewrite gpu-ci.yml for self-hosted DS01 | 5429be4 | .github/workflows/gpu-ci.yml, tests/fixtures/model_list.txt |

## What Was Built

**Tier 1 CI (`ci.yml`):**
- `filter` job using `dorny/paths-filter@v3` — outputs `src` and `docker` path change flags
- `lint` and `type-check` jobs migrated from `pip install` to `uv sync --dev` with lockfile caching
- `test` job collapsed from 3.10+3.12 matrix to 3.12 only; skips when no src/test/config changes; uses `pytest-xdist -n auto` for parallel execution
- `package-validation` job — `uv build --wheel`, clean venv install, import check (runs always)
- `docker-smoke` job — `docker build -f docker/Dockerfile.vllm . --no-cache` (triggers on docker changes only)

**Tier 2 GPU CI (`gpu-ci.yml`):**
- Triggers on `push: branches: [main]` and `workflow_dispatch` (removed the manual confirm gate)
- Runs on `self-hosted` (DS01) with 60-minute timeout
- Full test suite with no `-m` filter (gpu and docker markers included)
- `actions/cache@v4` for `~/.cache/huggingface` keyed on `model_list.txt` hash
- Artifact upload of `tests/fixtures/replay/` on every run (even on failure)

## Deviations from Plan

None — plan executed exactly as written.

## Self-Check

### Files Created/Modified

- [x] `.github/workflows/ci.yml` — FOUND
- [x] `.github/workflows/gpu-ci.yml` — FOUND
- [x] `tests/fixtures/model_list.txt` — FOUND

### Commits

- [x] 90bb294 — ci: rewrite Tier 1 CI with path filtering, Docker smoke, and package validation
- [x] 5429be4 — ci: add Tier 2 GPU CI workflow with HF cache and replay fixture upload

## Self-Check: PASSED
