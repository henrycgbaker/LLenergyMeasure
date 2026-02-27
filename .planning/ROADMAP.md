# Roadmap: LLenergyMeasure — M1 (Core Single-Experiment)

## Overview

M1 restructures the existing v1.x codebase (~22,000 lines) into a library-first package with a
clean public API, rewritten config and results schemas, and a two-command CLI. Phases follow the
dependency order: foundation → config → API → backend → measurement → results → CLI → testing.
Every phase delivers a verifiable capability; the final phase confirms end-to-end correctness
with a real GPU.

## Milestones

- [x] **v1.x Foundation & Planning** — Phases 1–4.5 (shipped 2026-02-26)
- [ ] **M1 — Core Single-Experiment** — Phases 1–8 (current)

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3...): Planned M1 work
- Decimal phases (1.1, 2.1...): Urgent insertions (marked with INSERTED)

- [x] **Phase 1: Package Foundation** - Dead code removal, src/ layout, pyproject.toml, protocols, state machine, resilience carry-forwards (completed 2026-02-26)
- [ ] **Phase 2: Config System** - ExperimentConfig composition model, YAML loader, user config, SSOT introspection
- [x] **Phase 3: Library API** - `__init__.py` public API, `run_experiment()`, internal `_run(StudyConfig)`, API stability contract (completed 2026-02-26)
- [ ] **Phase 4: PyTorch Backend and Pre-flight** - PyTorch inference backend (P0 bug fix), InferenceBackend protocol, pre-flight checks, environment snapshot
- [ ] **Phase 4.1: PyTorch Parameter Audit** - INSERTED — Audit PyTorchConfig fields against upstream `transformers`/`torch` APIs; ensure all researcher-useful parameters are exposed
- [ ] **Phase 5: Energy Measurement** - NVML poller, Zeus optional, CodeCarbon optional, baseline power, warmup, FLOPs estimation, timeseries
- [ ] **Phase 6: Results Schema and Persistence** - ExperimentResult schema, EnergyBreakdown, persistence API, late aggregation, output layout, collision handling
- [ ] **Phase 7: CLI** - `llem run`, `llem config`, `llem --version`, plain text display, exit codes, error hierarchy
- [ ] **Phase 8: Testing and Integration** - Unit + integration test tiers, protocol mocks, GPU CI workflow, UAT against exit criteria

## Phase Details

### Phase 1: Package Foundation
**Goal**: The codebase is restructured into a clean src/ layout with dead code removed, build system configured, and all reusable infrastructure (protocols, state machine, resilience) in place — ready for new subsystems to be written on top.
**Depends on**: Nothing (first phase)
**Requirements**: INF-01, INF-02, INF-03, INF-04, INF-05, INF-06, INF-07, INF-08, INF-18, INF-19, INF-20
**Success Criteria** (what must be TRUE):
  1. `pip install -e ".[pytorch]"` installs cleanly from the new src/ layout with no legacy entry points
  2. `llem --help` is the only entry point — `lem` is absent from the installed package
  3. `import llenergymeasure` succeeds and `llenergymeasure.__version__` returns `"2.0.0"`
  4. The 1,524 lines of confirmed dead code are absent from the repository
  5. `protocols.py` defines the 5 DI interfaces and the 3-state state machine exists in `core/state.py`
**Plans**: 3 plans

Plans:
- [ ] 01-01-PLAN.md — Build system (Poetry to hatchling) and dead code removal
- [ ] 01-02-PLAN.md — Protocols, exceptions, resilience, and security carry-forwards
- [ ] 01-03-PLAN.md — 3-state machine and subprocess lifecycle carry-forward

---

### Phase 2: Config System
**Goal**: Researchers can express any single-experiment configuration as a YAML file or Python object, validate it, and load it — with clear errors on bad input and a user config file for persistent defaults.
**Depends on**: Phase 1
**Requirements**: CFG-01, CFG-02, CFG-03, CFG-04, CFG-05, CFG-06, CFG-07, CFG-08, CFG-09, CFG-10, CFG-17, CFG-18, CFG-19, CFG-20, CFG-21, CFG-22, CFG-23, CFG-24, CFG-25, CFG-26
**Success Criteria** (what must be TRUE):
  1. A valid experiment YAML loads into an `ExperimentConfig` without error; an invalid YAML raises `ConfigError` with file path context
  2. Pydantic `ValidationError` (bad field values) passes through unchanged — not wrapped in `ConfigError`
  3. `~/.config/llenergymeasure/config.yaml` is read on startup; a missing file applies all defaults with no error
  4. `config/introspection.py` returns the full `ExperimentConfig` JSON schema and per-field constraint metadata
  5. Cross-validators reject mismatched precision/backend and backend-section/backend-field combinations with clear messages
**Plans**: 4 plans

Plans:
- [ ] 02-01-PLAN.md — ExperimentConfig v2.0 schema (field renames, extra=forbid, cross-validators, backend configs)
- [ ] 02-02-PLAN.md — YAML loader (collect-all-errors, ConfigError, did-you-mean, CLI override merging)
- [ ] 02-03-PLAN.md — User config (XDG path, v2.0 schema, env var overrides)
- [ ] 02-04-PLAN.md — Introspection (v2.0 field names, backend_support metadata, JSON schema export)

---

### Phase 3: Library API
**Goal**: The package exports a stable, documented public API — `run_experiment()` and `run_study()` — with no union return types and a clear stability contract so downstream code can depend on it without breakage across minor versions.
**Depends on**: Phase 2 (config types must exist before API functions can be typed)
**Requirements**: LA-01, LA-03, LA-04, LA-06, LA-07, LA-08, LA-09, LA-10
**Success Criteria** (what must be TRUE):
  1. `from llenergymeasure import run_experiment, ExperimentConfig, ExperimentResult` all resolve without error
  2. `run_experiment(config)` returns exactly `ExperimentResult` — no union types, no `None`
  3. `run_experiment()` with no `output_dir` produces no disk writes (side-effect-free)
  4. Any name not in `__init__.py.__all__` raises `AttributeError` on direct import — internal modules are private
  5. `llenergymeasure.__version__ == "2.0.0"`
**Plans**: TBD

Plans:
- [ ] TBD

---

### Phase 4: PyTorch Backend and Pre-flight
**Goal**: PyTorch inference runs correctly end-to-end with the P0 model_kwargs bug fixed, pre-flight checks catch configuration errors before wasting GPU time, and the environment is fully snapshotted at experiment start.
**Depends on**: Phase 3 (Library API must exist; backend returns `ExperimentResult`)
**Requirements**: CM-01, CM-04, CM-05, CM-06, CM-29, CM-30, CM-31, CM-32, CM-33, CM-34
**Success Criteria** (what must be TRUE):
  1. `run_experiment(ExperimentConfig(model="gpt2", backend="pytorch"))` completes without error on a GPU machine
  2. The P0 `model_kwargs` bug (L375) is fixed — extra kwargs pass through to the model without `TypeError`
  3. Pre-flight reports all failures at once (not one at a time) and raises `PreFlightError` before any GPU allocation
  4. `result.environment_snapshot` contains Python version, CUDA version, driver version, GPU names, and pip freeze
  5. CUDA version is detected via multi-source fallback (torch → version.txt → nvcc → `None`)
**Plans**: TBD

Plans:
- [ ] TBD

---

### Phase 4.1: PyTorch Parameter Audit
**INSERTED** — Added 2026-02-26
**Goal**: Every tuneable PyTorch parameter that a researcher would reasonably want to control is exposed as an `ExperimentConfig` / `PyTorchConfig` field — no hidden knobs that require `passthrough_kwargs` for common use cases.
**Depends on**: Phase 4 (PyTorch backend must be running and testable)
**Requirements**: None (quality audit — no formal requirement IDs)
**Success Criteria** (what must be TRUE):
  1. `PyTorchConfig` fields cover all parameters accepted by `AutoModelForCausalLM.from_pretrained()` that affect inference behaviour (e.g., `torch_dtype`, `attn_implementation`, `device_map`, quantisation flags)
  2. `DecoderConfig` fields cover all `model.generate()` parameters that affect output (temperature, top_p, top_k, repetition_penalty, do_sample, etc.)
  3. Any v1.x parameters that were dropped have a documented rationale (intentional removal, not oversight)
  4. SSOT introspection (`config/introspection.py`) reflects the updated field set
  5. Cross-validators updated for any new field interactions
**Plans**: TBD
**Note**: Repeat this audit for vLLM and TensorRT-LLM backends when they are implemented in M3.

Plans:
- [ ] TBD

---

### Phase 5: Energy Measurement
**Goal**: Every experiment produces scientifically credible energy numbers — baseline-corrected, warmed-up, with correct measurement backend priority and a 1 Hz timeseries sidecar — so results can be cited in a research paper.
**Depends on**: Phase 4.1 (config fields must be complete before energy measurement wraps the backend)
**Requirements**: CM-11, CM-12, CM-13, CM-14, CM-15, CM-16, CM-17, CM-18, CM-19, CM-20, CM-21, CM-22, CM-23, CM-24, CM-25, CM-26, CM-27, CM-28
**Success Criteria** (what must be TRUE):
  1. `result.energy_adjusted_j` equals `energy_total_j` minus `(baseline_power_w × duration_sec)` — baseline subtraction is applied
  2. When Zeus is installed, it is selected over NVML; when only NVML is available, it is used; CodeCarbon is the fallback of last resort
  3. Warmup runs 5 full-length prompts by default before measurement begins; warmup tokens are excluded from FLOPs calculation
  4. `result.timeseries` references a `timeseries.parquet` file with 1 Hz GPU power samples from the measurement window
  5. `result.flops_result.method` and `.confidence` are populated; FLOPs uses the PaLM formula (2 × N_params × tokens)
**Plans**: TBD

Plans:
- [ ] TBD

---

### Phase 6: Results Schema and Persistence
**Goal**: Every experiment produces a complete, schema-versioned `ExperimentResult` written to a stable output directory — with collision-safe naming, a Parquet timeseries sidecar, and a round-trip-safe persistence API.
**Depends on**: Phase 5 (energy measurement produces the values that populate the result schema)
**Requirements**: RES-01, RES-02, RES-03, RES-04, RES-05, RES-06, RES-07, RES-08, RES-09, RES-10, RES-11, RES-12, RES-16, RES-17, RES-18, RES-19, RES-20, RES-21
**Success Criteria** (what must be TRUE):
  1. `result.schema_version == "2.0"` and `result.measurement_config_hash` is a 16-char hex string derived from the config (environment snapshot excluded)
  2. Output is written to `{name}_{timestamp}/result.json`; running twice with the same name produces `{name}_{timestamp}_1/` — never overwrites
  3. `ExperimentResult.from_json(path)` round-trips without data loss — all fields survive serialisation and deserialisation
  4. `result.measurement_warnings` is a list (empty or populated); `result.reproducibility_notes` is a non-empty string
  5. `results/aggregation.py` aggregates per-process raw files into a single `ExperimentResult` (PyTorch multi-GPU path)
**Plans**: TBD

Plans:
- [ ] TBD

---

### Phase 7: CLI
**Goal**: Researchers interact with the tool entirely through `llem run` and `llem config` — plain text output, no Rich dependency, correct exit codes on all error paths, and a `--dry-run` that validates without running.
**Depends on**: Phase 6 (CLI wraps `run_experiment()` and displays `ExperimentResult`)
**Requirements**: CLI-01, CLI-02, CLI-03, CLI-04, CLI-06, CLI-07, CLI-08, CLI-09, CLI-10, CLI-12, CLI-13, CLI-14
**Success Criteria** (what must be TRUE):
  1. `llem run --model gpt2 --backend pytorch` runs a full experiment and prints the result summary to stdout
  2. `llem run experiment.yaml` loads the YAML, validates, runs, and writes output — same result as the flag-based form
  3. `llem run --dry-run experiment.yaml` validates the config and estimates VRAM without running inference; exits 0
  4. `llem config` prints environment state (GPU, backends installed, user config path); `llem config --verbose` adds per-backend detail
  5. A `ConfigError` exits with code 2; a `PreFlightError` or `ExperimentError` exits with code 1; SIGINT exits with code 130
**Plans**: TBD

Plans:
- [ ] TBD

---

### Phase 8: Testing and Integration
**Goal**: The codebase has systematic test coverage — GPU-free unit tests using protocol injection mocks and GPU integration tests that confirm the M1 exit criteria end-to-end — with a CI workflow that runs both tiers.
**Depends on**: Phase 7 (all subsystems complete before integration testing)
**Requirements**: STU-05, INF-09, INF-10, INF-11, INF-12
**Success Criteria** (what must be TRUE):
  1. `pytest tests/unit/` passes on a machine without a GPU — no GPU calls in unit tests
  2. `pytest tests/integration/ -m gpu` on a GPU machine runs a real PyTorch experiment and asserts a valid `ExperimentResult`
  3. Protocol injection mocks replace real backends in unit tests — no `unittest.mock.patch` on internal modules
  4. `llem run --model gpt2 --backend pytorch` produces valid `ExperimentResult` JSON (M1 primary exit criterion)
  5. `llem run experiment.yaml`, `llem config`, and `llem --version` all behave correctly per their Phase 7 success criteria
**Plans**: TBD

Plans:
- [ ] TBD

---

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 4.1 → 5 → 6 → 7 → 8

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Package Foundation | 5/5 | Complete   | 2026-02-26 |
| 2. Config System | 4/4 | Complete   | 2026-02-26 |
| 3. Library API | 2/2 | Complete   | 2026-02-26 |
| 4. PyTorch Backend and Pre-flight | 0/? | Not started | - |
| 4.1. PyTorch Parameter Audit | 0/? | Not started | - |
| 5. Energy Measurement | 0/? | Not started | - |
| 6. Results Schema and Persistence | 0/? | Not started | - |
| 7. CLI | 0/? | Not started | - |
| 8. Testing and Integration | 0/? | Not started | - |

---

*Roadmap created: 2026-02-26 for M1 — Core Single-Experiment*
*Previous v1.x phases (1–4.5) archived in MILESTONES.md*
