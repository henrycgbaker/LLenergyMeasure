# Architecture Decisions

**Status:** Accepted
**Date decided:** 2026-02-17
**Last updated:** 2026-02-25
**Research:** `.planning/research/ARCHITECTURE.md` (architecture audit, 2026-02-25)

## Decision

| Sub-decision | Resolution |
|---|---|
| **A. Architecture pattern** | Library-first at v2.0. CLI is a thin wrapper. |
| **B. Module structure** | Single package `src/llenergymeasure/`. CLI and study modules inside. |
| **C. Config model** | Two config sources + auto-capture: (1) user config (machine/runners), (2) experiment/study YAML, plus environment snapshot (auto-captured, stored with results — not a config layer). |
| **D. Field placement** | "Varies by machine?" → user config. "Defines what I'm measuring?" → experiment/study YAML. Environment snapshot auto-captured at runtime. |
| **E. Runner resolution** | YAML is silent on runner. Precedence: env var > user config > CLI flag > default (local). |
| **F. Infrastructure metadata** | Auto-detect + store with results. Scientific record, not config. |
| **G. Energy vs CO2** | Independent concerns. Base = NVML polling. `[zeus]` = accurate energy. `[codecarbon]` = CO2. |
| **H. Library API surface** | `__init__.py` exports only: `run_experiment`, `run_study`, `ExperimentConfig`, `StudyConfig`, `ExperimentResult`, `StudyResult`. |
| **I. Subprocess isolation** | `multiprocessing.Process` per experiment. Clean GPU state between runs. |
| **J. Study module scope** | Ships with core at v2.0. Docker multi-backend deferred to v2.2. |

---

## Context

LLenergyMeasure v1.x was a CLI monolith — a Typer application with inference backends embedded
directly. Scaling to multi-backend experiments, study sweeps, and a future web platform required
a structural rethink. The v2.0 redesign needed to answer: what is the correct architectural
pattern for a measurement tool that is also a research library?

Five forces shaped the decision space:
1. Researchers want `import llenergymeasure` to script custom experiments — a CLI-only tool is a dead end
2. vLLM + TensorRT-LLM cannot coexist in the same process (CUDA/driver conflicts) — subprocess isolation is required
3. Infrastructure varies by machine; experiment definitions should be shareable — the config model must separate these
4. CO2 estimation and energy measurement are independent concerns that different users opt into differently
5. A future web platform requires a clean library boundary to call from, not a CLI to subprocess

---

## Sub-decision A — Architecture Pattern (Library-First)

### Considered Options

| Option | Pros | Cons |
|--------|------|------|
| **Library-first at v2.0 (chosen)** | Clean public API; CLI and future web are both thin clients; standard pattern (lm-eval, MLflow, CodeCarbon); prevents accumulation of further debt | More restructuring work upfront |
| Library-first deferred to v3.0 | Less immediate work | v3.0 restructure must fight against more accumulated CLI-centric coupling; web platform blocked |
| API-first (HTTP service at v2.0) | Web-native; externally accessible | Premature — HTTP adds operational overhead for a local measurement tool; library provides the boundary without the overhead |
| CLI monolith (no library layer) | Simplest in the short term | No Python API; web platform cannot call it; every automation requires shelling out |

### Decision

We will restructure as library-first at v2.0. `import llenergymeasure` is the primary interface;
CLI (`llem`) is a thin wrapper over the library. The future v4.0 web platform and any CLI
subcommands are both clients of the library, never the reverse.

Rationale: Industry norm. lm-eval, MLflow, CodeCarbon all ship library-first. Clean library
boundaries require doing this at v2.0 — the longer this is deferred, the more CLI-centric
coupling accumulates.

### Consequences

Positive: Stable Python API from v2.0; CLI and library can evolve in parallel; web platform has
a clean entry point.

Negative / Trade-offs: Restructuring upfront cost; requires clear `__init__.py` boundary
enforcement.

Neutral: HTTP API remains premature — the library layer provides the boundary without service
overhead. Deferred to v4.0+ web platform.

---

## Sub-decision B — Module Structure

### Considered Options

| Option | Pros | Cons |
|--------|------|------|
| **CLI inside package, study inside package (chosen)** | Standard pattern (lm-eval, MLflow); single release artifact; avoids cross-package dep management | |
| CLI as separate top-level package (`llenergymeasure-cli`) | Hard separation | Independent release cycle not needed; adds installation complexity for users |
| Study as separate top-level package (`llenergymeasure-study`) | Independent release cycle | No independent release rationale; adds installation complexity |

### Decision

`src/llenergymeasure/` layout is retained (industry standard; lm-eval, Optimum-Benchmark both
use `src/`). `cli/` stays inside `llenergymeasure/` — accessed via `llenergymeasure.cli` —
the lm-eval/MLflow pattern. `study/` goes inside `llenergymeasure/study/` — `from
llenergymeasure.study import StudyRunner`.

Separate top-level packages are only warranted with independent release cycles, which neither
CLI nor study needs.

### Consequences

Positive: Single pip install; no cross-package version pinning; standard discoverable layout.

Negative / Trade-offs: Everything in one package (acceptable for this tool's scope).

Neutral: See [designs/architecture.md](../designs/architecture.md) for the full module layout
tree and StudyRunner subprocess pseudocode.

---

## Sub-decision C — Config Model (Two Sources + Auto-Capture)

> **Updated (2026-02-25):** Renamed from "Three-Layer Config Model" to "Two Sources + Auto-Capture".
> No peer tool uses a named three-layer config model. The valid separation is two config sources
> (user config + experiment YAML) plus an auto-captured environment snapshot stored with results.
> "Layer 3" was output metadata, not configuration — renaming prevents implementers from treating
> it as a config concern requiring validation and precedence resolution.
> See `.planning/research/ARCHITECTURE.md` (Decision C audit).

### Considered Options

| Option | Pros | Cons |
|--------|------|------|
| **Two config sources + auto-capture (chosen, revised 2026-02-25)** | Clean separation; shareable experiment files; environment snapshot travels with results; matches peer practice (Nextflow, Snakemake, DVC, MLflow) | Two config surfaces requires mental model; must be clearly documented |
| Flat single config file | Simple | Infrastructure settings and experiment definition mix; files not shareable between machines |
| Three-layer model (original, renamed 2026-02-25) | Same separation as chosen, with explicit naming | "Layer 3" naming implies a config concern; no peer uses three explicit layers; over-engineered naming |

> **Superseded (2026-02-25):** The original "Three-Layer Config Model" naming. The underlying
> separation (user config, experiment YAML, environment snapshot) is unchanged — only the
> naming is simplified. "Layer 1/2/3" jargon replaced with descriptive names throughout.

### Decision

Two config sources plus auto-captured environment data. Reference pattern: Nextflow
(`nextflow.config` profiles), Snakemake (`--profile` directories), DVC (`~/.dvc/config`).

```
User Config (HOW to run — machine-local)
  Location:  ~/.config/llenergymeasure/config.yaml   ← user-local, never versioned
  Contents:  Backend runner mappings, execution defaults, energy backend, carbon intensity/PUE defaults
  Example:
    runners:
      pytorch: local
      vllm:    docker:llenergymeasure:vllm
      tensorrt: singularity:/images/llenergymeasure-tensorrt.sif

Experiment / Study YAML (WHAT to measure — portable, shareable)
  Location:  experiment.yaml / study.yaml   ← versioned, shareable
  Contents:  Model, dataset, backend, batch size, hyperparams
             YAML is SILENT on execution environment — no runner: field
             Runner selection is purely a user config / CLI / env var concern
             Both experiment.yaml and study.yaml are consumed by `llem run`
             (unified CLI — see experiment-study-architecture.md)

Environment Snapshot (WHAT you're measuring ON — auto-captured, stored with results)
  Location:  Auto-captured at runtime; NOT a config file
             NOT in experiment.yaml — ExperimentConfig is infrastructure-agnostic (decided 2026-02-20)
  Contents:
    Auto-detected:     GPU model, VRAM, count, TDP, CPU model, CUDA version, driver version
    User-specified defaults (user config):
                       grid_carbon_intensity_gco2_kwh, datacenter_pue
    Env var override:  LLEM_CARBON_INTENSITY, LLEM_DATACENTER_PUE
                       (highest precedence; one-off overrides without touching config files)
  Stored:    With results — must travel with published data for reproducibility

  NOTE (2026-02-20): Per-experiment override of infrastructure context removed from ExperimentConfig.
  experiment.yaml must be portable and infrastructure-agnostic — a study file shared between
  colleagues or institutions should not encode one researcher's datacenter PUE. Env vars serve
  the one-off override case (e.g. measuring on a cloud GPU in a different region) cleanly.

  NOTE (2026-02-25): datacenter_location removed. Carbon intensity lookup delegated to
  CodeCarbon ([codecarbon] optional extra). Base package retains only user-specified
  grid_carbon_intensity_gco2_kwh with simple arithmetic. See decisions/carbon-intensity.md.
```

### Consequences

Positive: Experiment files are fully portable across machines; environment snapshot is captured
as scientific record; user config adapts the portable study to the specific machine.

Negative / Trade-offs: Two config surfaces requires mental model; must be clearly documented.

Neutral: Env vars handle one-off overrides without polluting any config file.

---

## Sub-decision D — Field Placement (Elaboration of Decision C)

> **Updated (2026-02-25):** Simplified from "three questions" framework to a direct field
> placement table. The questions ("Does this vary between machines?", etc.) added cognitive
> overhead without preventing real mistakes — the table below is the SSOT. See
> `.planning/research/ARCHITECTURE.md` (Decision D audit).

### Considered Options

N/A — this is a downstream elaboration of Decision C (which config source owns each field),
not a separate structural choice between architectural options.

### Decision

Each field belongs in exactly one place. The field placement table is the canonical reference.

### Field Placement Table

| Field | Config source | Where defined | Rationale |
|-------|---------------|---------------|-----------|
| `runners.<backend>` | User config | `~/.config/llenergymeasure/config.yaml` | How a backend is executed varies by machine (local vs Docker vs Singularity) |
| `output.results_dir` | User config | user config | Output path is machine-local (HPC scratch vs laptop home dir) |
| `output.model_cache_dir` | User config | user config | Model weights storage is machine-local |
| `measurement.energy_backend` | User config | user config | Which energy backend is installed varies by machine |
| `measurement.carbon_intensity_gco2_kwh` | User config (default) / env snapshot (stored) | user config default; env var override; stored with result | Varies by physical location of the machine. Used for base-package CO2 arithmetic when CodeCarbon is not installed. |
| `measurement.datacenter_pue` | User config (default) / env snapshot (stored) | user config default; env var override; stored with result | Varies by physical facility |
| `ui.prompt`, `ui.verbosity` | User config | user config | User preference; CI vs interactive environment |
| `execution.config_gap_seconds` | User config | user config | Machine-local thermal recovery between experiments |
| `execution.cycle_gap_seconds` | User config | user config | Machine-local thermal recovery between cycles |
| `model` | Experiment YAML | experiment.yaml | Defines what is being measured |
| `backend` | Experiment YAML | experiment.yaml | Defines the inference implementation being evaluated |
| `batch_size`, `precision`, `n` | Experiment YAML | experiment.yaml | The parameters under study |
| `dataset` | Experiment YAML | experiment.yaml | Defines the workload |
| `warmup` | Experiment YAML | experiment.yaml | Part of the measurement protocol |
| `sweep:` / `experiments:` | Study YAML | study file | Defines the parameter space → included in `study_design_hash` |
| `execution: { n_cycles, cycle_order, [gap overrides] }` | Study YAML / metadata | study file | In study.yaml for portability; **excluded from** `study_design_hash`; stored as `measurement_protocol` in StudyResult |
| `gpu_model`, `gpu_vram_gb`, `cuda_version` | Environment snapshot | auto-captured at runtime | Hardware is not configurable — it IS the context |
| `carbon_intensity_gco2_kwh` (stored) | Environment snapshot | from user config default or env var override | Environmental fact at measurement time |
| `datacenter_pue` (stored) | Environment snapshot | from user config default or env var override | Environmental fact at measurement time |

### Override Precedence for Environment Snapshot Fields

Environment snapshot fields have no user config + experiment YAML overlap —
`ExperimentConfig` does not carry these fields. The precedence chain is linear:

```
1. User config default:  ~/.config/llenergymeasure/config.yaml  measurement.*
2. Env var:              LLEM_CARBON_INTENSITY, LLEM_DATACENTER_PUE
3. Stored in environment snapshot in result JSON at measurement time
```

Env vars are the correct mechanism for one-off overrides (e.g. a US cloud GPU run when
your user config is set to Germany). This keeps `experiment.yaml` infrastructure-agnostic.

### Design Principles

1. **User config is not the experiment.** It describes the machine, not the measurement.
   A study file should be shareable and reproducible on any machine. The user config is the
   adapter between the portable study and the specific machine it runs on.

2. **Env vars override user config, not experiment files.** This gives CI/Docker environments
   a way to configure the tool without writing config files, while keeping the experiment
   definition clean.

3. **Environment snapshot is append-only.** Once a result is stored with its environment
   snapshot, that context is immutable. You cannot "re-attribute" a measurement to a different
   PUE after the fact.

4. **Secrets are never in experiment/study YAML files.** API keys (`HF_TOKEN`) are supplied via environment variables or `huggingface-cli login` (see [access-control.md](access-control.md)). Any field matching known credential patterns in experiment.yaml raises a hard error.

### Consequences

Positive: Environment snapshot is captured as scientific record; study files are portable and
shareable; snapshot is immutable provenance.

Negative / Trade-offs: Env vars are the only one-off override mechanism for environment
snapshot fields — no per-experiment YAML override.

Neutral: Runner resolution follows the same portability principle (see Sub-decision E).

---

## Sub-decision E — Runner Resolution

### Considered Options

| Option | Pros | Cons |
|--------|------|------|
| **Runner in user config + CLI flag (chosen)** | study.yaml is silent on execution environment; shareable across machines with different runner setups | Requires user to understand the config layer |
| `runner:` field in study.yaml | Explicit per-study | study.yaml not portable (local vs Docker varies by machine); breaks colleague shareability |
| Always local (no Docker option) | Simple | Multi-backend studies require Docker — correctness requirement, not preference (CUDA driver conflicts) |

### Decision

YAML configs have no `runner:` field. Runner selection, in precedence order:

```
1. LLEM_RUNNER_<BACKEND> env var          ← highest
2. ~/.config/llenergymeasure/config.yaml runners.<backend>
3. CLI flag: llem run --runner docker     ← overrides all backends
4. Default: local (DECIDED 2026-02-19 — see open-questions.md Q3)
```

**Execution modes and user config:**
- Exploration (local): runner omitted from study.yaml; user config defaults to `local` for all backends
- Formal benchmark (Docker): user config maps each backend to its Docker service; study YAML is identical
- Multi-backend study: user config is required; study module dispatches to containers per backend automatically
- HPC (Singularity): user config maps to Singularity image paths (deferred to post-v2.0)

User config `runners:` entries are consulted for each backend individually. Multi-backend detection
drives Docker requirement check at study startup. Multi-backend study with no Docker → hard error
with guidance (not auto-degrade to local).

See [designs/user-config.md](../designs/user-config.md) for full resolution logic.

### Consequences

Positive: Experiment/study YAML is infrastructure-agnostic; portable between colleagues' machines.

Negative / Trade-offs: Multi-backend + no Docker → hard error. Clear failure is better than
silent degradation to local-only.

Neutral: N/A

---

## Sub-decision F — Infrastructure Metadata as Scientific Record

### Considered Options

| Option | Pros | Cons |
|--------|------|------|
| **Auto-detect + store with results (chosen)** | Complete provenance; no user burden for GPU model/CUDA info; context travels with data | Must detect reliably across GPU configurations |
| Require user to specify all context in config | Explicit | Burdensome; error-prone; users forget to update when hardware changes |
| Store context separately (sidecar file) | Separated concerns | Separated files can get separated from results; provenance broken |

### Decision

GPU model, VRAM, CPU, datacenter PUE, grid carbon intensity are part of the measurement, not
execution plumbing. Auto-detected where possible (nvidia-smi, `/proc/cpuinfo`). User-specified
via user config for location, PUE. Stored with every result and must travel with
published data.

### Consequences

Positive: Results are self-contained; reproducibility checking possible without external config files.

Negative / Trade-offs: Auto-detection can fail on unusual hardware configurations — must handle
gracefully.

Neutral: Environment snapshot is append-only — context is immutable after measurement.

---

## Sub-decision G — Energy Measurement Separation (CO2 vs Energy)

### Considered Options

| Option | Pros | Cons |
|--------|------|------|
| **CO2 decoupled from energy (chosen)** | Zeus measures joules/watts (energy); CodeCarbon estimates CO2 — independent concerns; users opt in to each separately | More extras to explain; two install paths |
| Bundle CO2 + energy in one backend | Single install | Forces CO2 estimation on all users; conflates measurement (energy) with estimation (CO2) |
| CO2 only, no raw energy | Simplest for non-technical users | Loses precision; researchers need joules/watts for scientific claims |

### Decision

CO2 estimation and energy measurement are independent concerns. Base package includes raw NVML
polling (via `nvidia-ml-py`, already a transitive dep). `[zeus]` for accurate NVML energy
measurement (~5% error vs ~10-15% for polling). `[codecarbon]` for CO2 estimation. Users opt
in to each separately.

**NVML single-session owner:** Only one NVML session can be active at a time. When Zeus is
installed and active, the base NVML poller must yield — they cannot both run simultaneously.
The energy backend layer enforces a single owner.

### Consequences

Positive: Scientific claim integrity — users choose their energy measurement method explicitly;
CO2 is an opt-in estimate, not an invisible default.

Negative / Trade-offs: Base install cannot estimate CO2 without `[codecarbon]`; must document
both extras.

Neutral: NVML single-owner enforcement is an implementation detail in the energy backend layer.

---

## Sub-decision H — Library API Surface

### Considered Options

| Option | Pros | Cons |
|--------|------|------|
| **`__init__.py` exports only as stable API (chosen)** | Clear boundary; everything else can change; one minor version deprecation window is manageable | Requires discipline to not accidentally add exports |
| All public modules as API | No boundary to maintain | Any refactor is a breaking change; unstable |
| No library API (CLI only) | Simpler | Researchers cannot script custom experiments; web platform has no entry point |
| HTTP API at v2.0 | Web-native | Premature — local measurement tool doesn't need HTTP overhead at this stage |

### Decision

Stable API = `__init__.py` exports only: `run_experiment`, `run_study`, `ExperimentConfig`, `StudyConfig`, `ExperimentResult`, `StudyResult`. All other classes (`ExperimentOrchestrator`, `StudyRunner`, backend Protocols) are internal — not exported. One minor version deprecation window before removing any export.

- `run_experiment(config) -> ExperimentResult` — single measurement. Internally wraps input into `StudyConfig(experiments=[config])`, calls `_run(StudyConfig)`, unwraps result.
- `run_study(config) -> StudyResult` — structured investigation. Calls `_run(StudyConfig)` directly.

Two public functions with unambiguous return types. Both delegate to a single internal `_run(StudyConfig) -> StudyResult` runner (Option C architecture). The CLI uses unified `llem run`; the library has two functions because library callers need type-safe return types.

> **Superseded (2026-02-25):** A brief period where the API was unified to `run()` with union
> return type `ExperimentResult | StudyResult`. Reverted after peer research found 0/10 tools
> use union return types and official Python typing guidance advises against them. The original
> split API was the correct design. See [experiment-study-architecture.md](experiment-study-architecture.md) Q3.

See [designs/library-api.md](../designs/library-api.md) for full API signatures, overload forms, and usage patterns.

### Consequences

Positive: Stable public API from v2.0; unambiguous return types; internals can evolve freely;
web platform has a clean entry point.

Negative / Trade-offs: API surface must be deliberately managed; anything in `__init__.py` is
a commitment with a deprecation obligation. Two public functions to maintain (mitigated by
both delegating to single internal `_run()`).

Neutral: HTTP API remains out of scope until v4.0+ web platform.

---

## Sub-decision I — Subprocess Isolation

### Considered Options

| Option | Pros | Cons |
|--------|------|------|
| **`multiprocessing.Process` per experiment (chosen)** | Clean GPU state between experiments; CUDA prevents shared state across process boundaries | Subprocess startup overhead per experiment (~1-3s) |
| In-process isolation (try/finally GPU cleanup) | No subprocess overhead | GPU state leaks between experiments; backends cannot be unloaded cleanly; CUDA globals persist across calls |
| Thread-based isolation | Low overhead | GPU context is shared across threads; no isolation between backends |
| Always Docker per experiment | Perfect isolation | Docker startup latency is too high for study sweeps (10+ seconds per experiment) |

### Decision

Every experiment runs in a fresh `multiprocessing.Process` (local) or ephemeral `docker run`
(Docker). Hard requirement for clean GPU state between experiments. See
[decisions/experiment-isolation.md](experiment-isolation.md).

### Consequences

Positive: Clean GPU state; eliminates memory leak and state accumulation between study sweep experiments.

Negative / Trade-offs: Subprocess startup overhead (~1-3s per experiment); acceptable for study
sweeps where experiments take minutes.

Neutral: Docker per-experiment is available for formal benchmark mode.

---

## Sub-decision J — Study Module Scope (v2.0 vs deferred)

### Considered Options

| Option | Pros | Cons |
|--------|------|------|
| **Study module ships with core at v2.0 (chosen)** | Sweeps are a genuine research use case; study functionality is the primary differentiator; architecture is already designed | Increases v2.0 scope |
| Defer study to v2.2 | Smaller v2.0 | Delays the primary research use case; study infrastructure is already designed |
| Study as separate package | Independent releases | Not warranted; no separate release cycle needed |

### Decision

Study module ships with core at v2.0 as `llenergymeasure/study/`. Study functionality is
accessed via the unified `llem run` command (YAML file determines scope — see
[experiment-study-architecture.md](experiment-study-architecture.md)). Single-backend studies
run locally; Docker multi-backend deferred to v2.2.

### Consequences

Positive: Full sweep capability from v2.0; primary differentiator available immediately.

Negative / Trade-offs: v2.0 scope includes study orchestration, subprocess management, manifest
persistence, and study result aggregation.

Neutral: HPC support (SLURM/Apptainer) deferred to post-v2.0. Docker multi-backend to v2.2.

---

## Tentative Decisions

| Decision | Rationale | Status |
|----------|-----------|--------|
| API-first premature for v2.0 | Good layering does not require HTTP API yet; add when web platform comes | Tentative |
| SGLang as next inference backend candidate | 23K+ stars, novel optimisations, no other benchmarking tool has SGLang + energy. Accelerated to v2.2 candidate. | Tentative — see versioning-roadmap.md |
| HPC support (SLURM/Apptainer) deferred to post-v2.0 | SLURM/K8s/Apptainer complexity not warranted until core is stable | Tentative |
| Zeus as additional optional energy backend (v2.1) | More accurate NVML measurement (~5% vs ~10-15%); AMD/Apple Silicon support. Our EnergyBackend Protocol already supports it. Plugin model | Tentative |
| lm-eval integration (v3.0) | Unique differentiator — quality-alongside-efficiency. No other tool combines these. | Tentative |
| Leaderboard deferred to v4.0 | Results DB archive is v2.4; public leaderboard is a web platform concern | Tentative |

---

## Related

- [experiment-study-architecture.md](experiment-study-architecture.md) — Option C architecture, unified `llem run`, library API `run_experiment()`/`run_study()`
- [cli-ux.md](cli-ux.md) — CLI command set (2 commands + 1 flag), execution profiles
- [carbon-intensity.md](carbon-intensity.md) — CO2 estimation delegated to CodeCarbon
- [access-control.md](access-control.md) — credential handling
- [../designs/architecture.md](../designs/architecture.md) — full module layout tree and StudyRunner subprocess pseudocode
- [../designs/library-api.md](../designs/library-api.md) — full API signatures, overload forms, and usage patterns
- [../designs/user-config.md](../designs/user-config.md) — full resolution logic for runners and environment snapshot
- [../designs/config-model.md](../designs/config-model.md) — SSOT for field placement, data flow, hash semantics
- [experiment-isolation.md](experiment-isolation.md) — subprocess isolation details
- [study-execution-model.md](study-execution-model.md) — measurement protocol placement, hashing semantics (Decision D superseded by experiment-study-architecture.md)
- [installation.md](installation.md) — extras and progressive disclosure
- [versioning-roadmap.md](versioning-roadmap.md) — implementation sequencing
