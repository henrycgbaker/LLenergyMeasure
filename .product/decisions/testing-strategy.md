# Testing Strategy

**Status:** Accepted
**Date decided:** 2026-02-19
**Last updated:** 2026-02-19
**Research:** N/A

## Decision

pytest. Two-tier: unit (no GPU) + integration (`@pytest.mark.gpu`). Mock backends via Protocol injection for unit tests. GPU CI on merge-to-main + weekly + manual + path-filtered PRs. Measurement accuracy testing out of scope; Hypothesis deferred.

---

## Context

LLenergyMeasure has two fundamentally different categories of logic to test:

1. **GPU-independent logic** — config parsing, sweep expansion, CLI argument handling,
   schema validation, filename generation, orchestration with mock backends. Fast, cheap,
   always runnable.
2. **GPU-dependent logic** — actual inference backend execution, NVML/Zeus energy measurement,
   end-to-end experiment runs. Slow, hardware-gated, scarce resource.

Mixing these in a single test suite either (a) makes CI slow and hardware-dependent, or
(b) excludes GPU tests entirely and gives no integration coverage. The decision space is
how to separate them and when to run each tier.

Additional constraint: GPU time is a scarce resource — the self-hosted runner is not
always-on. Running GPU tests on every PR wastes capacity on PRs that don't touch
GPU-dependent code.

Peers using the same split: lm-eval, Zeus, CodeCarbon all use `@pytest.mark.gpu` or
equivalent to gate hardware-dependent tests.

## Considered Options

### Sub-decision 1: Test framework

| Option | Pros | Cons |
|--------|------|------|
| **pytest** | Industry standard. Extensive plugin ecosystem (pytest-cov, pytest-asyncio, markers). All peer tools use it. | None relevant. |
| unittest | Standard library — no install. | Verbose boilerplate. Poor fixture support. No parametrize. No marker system. |
| Hypothesis (property-based) | Excellent for config grammar edge cases. | Not a replacement for pytest — additive, not alternative. Deferred (see below). |

### Sub-decision 2: Test tier separation

| Option | Pros | Cons |
|--------|------|------|
| **Two-tier: unit (no GPU) + integration (GPU required), separated by `@pytest.mark.gpu`** | Standard pattern (lm-eval, Zeus, CodeCarbon). `pytest -m "not gpu"` in standard CI. GPU tests run only when needed. | Two test suites to maintain. Marker must be applied consistently. |
| Single suite, skip GPU tests in CI | Simple structure. | No structural enforcement — GPU tests can accidentally run. Harder to reason about CI behaviour. |
| Three tiers (unit / functional / integration) | More granular. | Over-engineering for current scale. Two tiers map cleanly to the hardware boundary. |

### Sub-decision 3: Mock backend approach for unit tests

| Option | Pros | Cons |
|--------|------|------|
| **Protocol injection — `MockBackend` implements the `InferenceBackend` Protocol** | Tests orchestration logic without hardware. Same approach as existing `EnergyBackend Protocol`. Clean, type-safe. Deterministic fake results. | Mock divergence risk — `MockBackend` may drift from real backend Protocol over time. |
| Patch real backends with `unittest.mock` | No separate mock class needed. | Brittle — patches are tied to import paths. Hard to read. Doesn't leverage Protocol typing. |
| Integration tests only (no mocks) | Tests real behaviour. | Cannot run without GPU. Makes CI hardware-dependent for all tests. |

### Sub-decision 4: GPU CI trigger policy

| Option | Pros | Cons |
|--------|------|------|
| **GPU tests on: merge to main + weekly schedule + manual dispatch + path-filtered PRs** | GPU time used only when code that needs GPU validation changes. Weekly schedule catches dependency drift. | Path filtering adds CI config complexity. Some GPU-affecting PRs may slip through if paths are wrong. |
| GPU tests on every PR | Maximum coverage. | GPU time is a scarce resource. Most PRs (config, CLI, docs, schema) don't need GPU validation. |
| GPU tests only on manual dispatch | Conserves GPU time fully. | Regressions in GPU code may go undetected until manual run. |
| GPU tests on merge to main only | Reasonable balance. | No early detection for path-filtered PRs touching `core/`, `backends/`. |

### Sub-decision 5: Measurement accuracy testing

| Option | Pros | Cons |
|--------|------|------|
| **Out of scope for now** | No automated approach is feasible without known power draw ground truth. | No automated regression detection for measurement drift. |
| Manual calibration procedure (documented) | Human-verified accuracy. | Not automated — not a test in the traditional sense. May be documented later. |
| Property-based tests on energy calculation math | Tests the formula, not the measurement. | Can verify correctness of `co2_grams = energy_kwh × intensity` — not whether NVML readings are accurate. |

### Sub-decision 6: Property-based testing (Hypothesis)

| Option | Pros | Cons |
|--------|------|------|
| **Not adopted for now — revisit if config validation bugs emerge** | No added dependency or complexity now. | May miss edge cases in sweep grammar that parametrize doesn't cover. |
| Adopt from the start | Catches edge cases automatically. | Adds Hypothesis dependency. Learning curve. Overkill until config grammar is stable. |

## Decision

We will use pytest as the sole test framework. Tests are split into two tiers: `unit/`
(no GPU, always run in CI) and `integration/` (GPU required, gated with `@pytest.mark.gpu`).
Inference backends are mocked in unit tests via Protocol injection (`MockBackend` implements
`InferenceBackend` Protocol). Integration tests run on a self-hosted GPU runner, triggered
on merge to main, weekly schedule, manual dispatch, and PRs touching `core/`, `orchestration/`,
or `backends/`. Measurement accuracy testing is out of scope for now.

Property-based testing (Hypothesis) is not adopted but deferred — revisit if config
validation bugs emerge in the sweep grammar.

Rationale: the two-tier split at the GPU/no-GPU hardware boundary is the correct structural
boundary for this codebase (same as lm-eval, Zeus, CodeCarbon). Protocol injection for mocks
is cleaner than patching and leverages the existing Protocol design. Path-filtered GPU CI
ensures GPU time is spent only on PRs that affect GPU-dependent code.

## Consequences

Positive:
- `pytest -m "not gpu"` in standard CI is fast, hardware-free, and covers all config/
  schema/CLI/orchestration logic.
- GPU tests run only when code that needs validation changes — conserves scarce GPU time.
- Protocol-based mocks are type-safe and leverage existing design patterns.
- Weekly schedule catches dependency drift between releases.

Negative / Trade-offs:
- `@pytest.mark.gpu` must be applied consistently — missing markers silently include GPU
  tests in unit runs.
- `MockBackend` may drift from real `InferenceBackend` Protocol over time without enforcement.
- No automated measurement accuracy regression detection.

Neutral / Follow-up decisions triggered:
- `MockBackend` Protocol conformance should be tested or enforced via mypy.
- Hypothesis adoption remains open — revisit if sweep grammar edge-case bugs emerge.
- Measurement calibration procedure may be documented separately (not a test).

## Test Pyramid

```
tests/
  unit/                    ← no GPU; always run in CI
    test_config.py         ← Pydantic model validation, field renames, extra=forbid
    test_grid.py           ← sweep: → list[ExperimentConfig] expansion
    test_validation.py     ← SSOT backend compatibility constraints
    test_vram.py           ← VRAM estimation math
    test_result_schema.py  ← ExperimentResult / StudyResult field validation
    test_filenames.py      ← human-readable filename generation
    test_cli_parsing.py    ← Typer argument parsing (no actual runs)
    test_preflight.py      ← pre-flight logic with mock backends
  integration/             ← GPU required; gated with @pytest.mark.gpu
    test_pytorch_backend.py
    test_vllm_backend.py
    test_experiment_run.py ← end-to-end single experiment
    test_study_run.py      ← end-to-end study with subprocess isolation
    test_energy_backend.py ← NVML / Zeus energy measurement
```

## CI Trigger Policy

| Trigger | Test suite |
|---------|------------|
| Every push / PR | Unit tests only (`pytest -m "not gpu"`) — fast, no hardware |
| Merge to `main` | Unit + integration (GPU runner) |
| Weekly schedule (Sunday night) | Unit + integration (GPU runner) — catches dependency drift |
| Manual `workflow_dispatch` | Full suite on demand |
| PRs touching `core/`, `orchestration/`, `backends/` | Unit + integration (path-filtered trigger) |

**Rationale for not running GPU tests on every PR:** GPU time is a scarce resource. Most PRs touch
config, CLI, docs, or schema — none of which need GPU validation. Path filtering catches the
PRs that genuinely do.

## Mock Backend Pattern

```python
# tests/conftest.py
from llenergymeasure.core.backends import InferenceBackend  # Protocol

class MockBackend:
    """Deterministic fake backend for unit testing orchestration logic."""
    def load_model(self, config): pass
    def run_inference(self, prompts, config):
        return [MockResult(tokens=100, latency_ms=50.0)] * len(prompts)
    def unload_model(self): pass

@pytest.fixture
def mock_backend():
    return MockBackend()
```

## Not Included

- **Property-based testing (Hypothesis)** — considered but not adopted for now. May be useful
  for sweep grammar edge cases; revisit if config validation bugs emerge.
  **Rejected (2026-02-19):** Adds dependency and complexity before sweep grammar is stable.
- **Measurement accuracy tests** — out of scope. No automated approach is feasible without
  known power draw ground truth.

## Related

- [architecture.md](architecture.md) — `InferenceBackend` and `EnergyBackend` Protocol design
- [../designs/result-schema.md](../designs/result-schema.md) — `ExperimentResult` / `StudyResult` field definitions tested in `test_result_schema.py`
- [config-architecture.md](config-architecture.md) — `ExperimentConfig` sweep grammar tested in `test_grid.py`
