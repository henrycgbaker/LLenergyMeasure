# Phase 8: Testing and Integration - Context

**Gathered:** 2026-02-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Systematic test coverage for all M1 subsystems. GPU-free unit tests using protocol injection mocks, GPU integration tests confirming end-to-end correctness on a real model, and a CI workflow running both tiers. This phase validates everything built in Phases 1-7 — it does not add new features.

</domain>

<decisions>
## Implementation Decisions

### Test scope and coverage
- Start fresh — delete v1.x tests entirely, write v2.0 tests from scratch targeting new subsystems
- All subsystems get dedicated unit test files: config (schema, loader, introspection, user config), library API, PyTorch backend, energy measurement (NVML poller, FLOPs, warmup, baseline), results (schema, persistence, aggregation), CLI (run, config commands), infrastructure (protocols, state machine, resilience)
- Two tiers only: `tests/unit/` (GPU-free, fast) and `tests/integration/` (`@pytest.mark.gpu`). No e2e/ or runtime/ directories.
- No numeric coverage target — the 5 success criteria are the bar. Coverage follows naturally from testing all subsystems.

### Mock and fixture design
- Fake protocol classes implementing InferenceBackend, EnergyBackend etc. protocols — defined in `tests/fakes.py`. Injected via constructor/function args, not `unittest.mock.patch`.
- Layered conftest: `tests/conftest.py` for shared fixtures (sample configs, tmp dirs), `tests/fakes.py` for protocol fakes, per-directory conftest.py only if needed.
- `make_config(**overrides)` factory in conftest.py — returns valid ExperimentConfig with sensible defaults. Tests override only what matters for each test case.
- Schema-driven test generation where natural (use config introspection to generate edge-case configs — boundary values, all backends, all precisions). Hardcode specific regression values where clarity matters.

### GPU integration strategy
- Self-hosted runner on user's A100 machine (free, no GitHub GPU runner costs)
- Test model: gpt2 (124M) — matches M1 exit criterion (`llem run --model gpt2 --backend pytorch`)
- Full result validation: assert ExperimentResult has non-zero energy_total_j, valid timeseries path, tokens_per_second > 0, environment snapshot populated, schema_version '2.0'
- GPU tests run inside container (Docker with --gpus) since CUDA only available inside containers on this machine

### CI workflow design
- Two separate workflows:
  - `ci.yml`: unit tests on every PR/push (GitHub-hosted runner, fast, free)
  - `gpu-ci.yml`: GPU integration tests on merge to main + weekly + manual (self-hosted runner)
- Unit CI checks: pytest, ruff lint + format, mypy type checking, import validation (`from llenergymeasure import run_experiment, ExperimentConfig, ExperimentResult`)
- Python version matrix: 3.10 + 3.12
- Branch protection on main: unit CI must pass before merge. GPU CI is not required (runs post-merge).

### Claude's Discretion
- Whether to extract factory functions from fakes if repetition appears (start with plain fakes)
- Exact test file naming and organisation within each tier
- Self-hosted runner setup details (runner agent configuration, labels)
- Specific pytest marks and fixture scoping decisions
- Whether path-filtered CI triggers are worth the configuration complexity

</decisions>

<specifics>
## Specific Ideas

- Protocol injection fakes should be transparent — reading the fake class shows exactly what it returns, unlike MagicMock where behaviour is implicit
- The `make_config()` factory pattern: each test specifies only what it cares about, factory handles defaults
- GPU integration test should validate the complete pipeline: energy, timeseries, throughput, environment snapshot, schema version

</specifics>

<deferred>
## Deferred Ideas

- Version numbering: M1 should ship as v1.x (not v2.0). v2.0 should mean all backends (vLLM, TensorRT-LLM, Docker multi-backend) are complete. Affects Phase 1's `__version__` setting and overall product versioning strategy.

</deferred>

---

*Phase: 08-testing-and-integration*
*Context gathered: 2026-02-27*
