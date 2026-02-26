# User Config: `~/.config/llenergymeasure/config.yaml`

**Last updated**: 2026-02-25
**Status**: Confirmed — execution_profiles removed, execution gap fields added (2026-02-20)
**Referenced by**: [../decisions/architecture.md](../decisions/architecture.md) Layer 1

---

## Purpose

This file is user-local and never version-controlled. It holds concerns that belong
to the user's physical environment and machine, not the shareable study/experiment definition:

1. **Runners** — how each backend is executed (local, Docker image, Singularity path)
2. **Output paths** — where results and model weights are stored on this machine
3. **Measurement defaults** — energy backend preference, carbon intensity for this location, PUE
4. **UI preferences** — verbosity, interactive prompt behaviour (important for CI/HPC batch jobs)
5. **Execution gap defaults** — machine-local thermal recovery defaults (`config_gap_seconds`,
   `cycle_gap_seconds`). `n_cycles` and `cycle_order` are NOT here — they belong in `study.yaml`.
6. **Advanced tuning** — NVML polling interval and other measurement quality knobs

Keeping these here means the same `study.yaml` runs identically on a laptop (local PyTorch),
a workstation (Docker vLLM), and an HPC cluster (Singularity TRT) — without touching the study file.

**API keys (HF_TOKEN, LLEM_API_KEY) are NOT in this file.** They go in `.env` or shell
environment variables. This file never contains credentials. See access-control.md.

---

## Config File Location (XDG)

**Path**: `~/.config/llenergymeasure/config.yaml`
**Implementation**: resolved via `platformdirs.user_config_dir("llenergymeasure")`

### Decision: XDG Base Directory Specification

**Decided 2026-02-19. Rationale below.**

We follow the [XDG Base Directory Specification](https://specifications.freedesktop.org/basedir-spec/latest/)
rather than the legacy `~/.toolname/` dotdir pattern.

Modern Python tools use `$XDG_CONFIG_HOME` (defaulting to `~/.config` if unset):

| Tool | Config path |
|---|---|
| `uv` | `~/.config/uv/uv.toml` |
| `ruff` | `~/.config/ruff/` |
| `poetry` | `~/.config/pypoetry/config.toml` |
| `gh` CLI | `~/.config/gh/config.yml` |
| `hatch` | `~/.config/hatch/config.toml` |

The `~/.toolname/` pattern is pre-XDG (circa 2000s: `~/.ssh/`, `~/.aws/`). Modern dotfile managers,
package managers, and CI environments all understand `~/.config/` and can be configured to
exclude or include it cleanly.

**Implementation via `platformdirs`** (already a common transitive dep):
```python
from platformdirs import user_config_dir
from pathlib import Path

CONFIG_DIR = Path(user_config_dir("llenergymeasure"))
CONFIG_PATH = CONFIG_DIR / "config.yaml"
# Linux:   ~/.config/llenergymeasure/config.yaml
# macOS:   ~/Library/Application Support/llenergymeasure/config.yaml
# Windows: %APPDATA%\llenergymeasure\config.yaml
```

`platformdirs` gives correct XDG, macOS, and Windows paths for free — important for
future cross-platform portability.

---

## Full Schema

**Last updated**: 2026-02-25 — removed execution_profiles; added execution.config_gap_seconds / cycle_gap_seconds.

```yaml
# ~/.config/llenergymeasure/config.yaml
# User-local. Never commit this file. No secrets — hardware/environment config only.
# Missing file → all defaults apply; tool works out of the box.

# ─── OUTPUT PATHS ─────────────────────────────────────────────────────────────
# All paths are resolved relative to the working directory where llem is invoked,
# unless absolute. Override per-run with --output / --configs flags.

output:
  results_dir: ./results                 # default results output location
  configs_dir: ./configs                 # default config search path (tab completion + convention)
  model_cache_dir: ~/.cache/huggingface  # HuggingFace model weights cache
                                         # sets HF_HOME in subprocesses (important for HPC
                                         # where scratch filesystems hold large model weights)

# ─── RUNNERS ─────────────────────────────────────────────────────────────────
# Maps each backend to its execution environment.
# If omitted entirely: all backends default to `local`.
# Overridden per-invocation by CLI flag: llem run file.yaml --runner docker

runners:
  pytorch:  local                                   # local Python process (default)
  vllm:     docker:ghcr.io/llenergymeasure/vllm     # Docker image reference
  tensorrt: docker:ghcr.io/llenergymeasure/trt      # Docker image reference
  # tensorrt: singularity:/scratch/images/llem-trt.sif  # HPC — raises NotImplementedError in v2.0

# Runner value format:
#   local                       — run in current Python environment
#   docker:<image>              — ephemeral docker run per experiment
#   singularity:<image-path>    — Singularity/Apptainer (future, NotImplementedError in v2.0)


# ─── MEASUREMENT ─────────────────────────────────────────────────────────────
# Physical environment constants. Vary by machine location and facility.
# Set per-machine here; use env vars for one-off overrides (LLEM_CARBON_INTENSITY,
# LLEM_DATACENTER_PUE). experiment.yaml does NOT contain these fields — experiment
# configs are infrastructure-agnostic (decided 2026-02-20).
#
# For CO2 estimation without CodeCarbon, set carbon_intensity_gco2_kwh.
# With CodeCarbon installed, it handles location detection and lookup automatically.
# See decisions/carbon-intensity.md.

measurement:
  energy_backend: auto                   # auto | nvml | zeus
                                         # auto = zeus if installed, else nvml
  carbon_intensity_gco2_kwh: ~           # gCO2/kWh for local electricity grid (no default)
                                         # required for CO2 estimation without CodeCarbon
                                         # examples: DE=385, US=386, FR=58, GB=233, SG=400
                                         # source: IEA World Energy Statistics
  datacenter_pue: 1.0                    # Power Usage Effectiveness (1.0 = ideal, no overhead)
                                         # typical values: home GPU=1.0, cloud DC=1.1-1.2, HPC=1.4+


# ─── UI PREFERENCES ──────────────────────────────────────────────────────────

ui:
  prompt: true                           # false = non-interactive (CI pipelines, HPC batch jobs,
                                         # environments without stdin)
  verbosity: standard                    # quiet | standard | verbose


# ─── ADVANCED ────────────────────────────────────────────────────────────────
# Measurement quality tuning. Most users should leave these at defaults.

advanced:
  nvml_poll_interval_ms: 100             # NVML energy sampling frequency
                                         # lower = more temporal resolution, more CPU overhead
                                         # 100ms is a good balance; <10ms not recommended


# ─── EXECUTION ────────────────────────────────────────────────────────────────
# Machine-local thermal gap defaults. These vary by hardware:
#   - hot shared cluster needs long gaps; idle workstation may need 0s
# n_cycles and cycle_order are NOT here — they belong in study.yaml (portable, versioned).
# Gap overrides can be specified per-study in the execution: block if needed.

execution:
  config_gap_seconds: 60                 # thermal gap between experiments (machine default)
  cycle_gap_seconds: 300                 # thermal gap between full cycles (machine default)


# ─── REMOVED: execution_profiles (2026-02-20) ─────────────────────────────────
# Named execution presets (quick/standard/publication) in user config were rejected.
# Rationale: they bundled machine-local thermal settings (config_gap_seconds, cycle_gap_seconds)
# with study portability settings (n_cycles, cycle_order), making study files depend on a profile
# name that may not exist on a colleague's machine. No peer tool uses named profiles for
# statistical rigour presets. See decisions/study-execution-model.md Decision A.
#
# Migration:
#   Old: execution_profiles.quick → New: study.yaml execution: {n_cycles: 1} + --no-gaps CLI flag
#   Old: execution_profiles.standard → New: CLI effective default (3 cycles, interleaved)
#   Old: execution_profiles.publication → New: study.yaml execution: {n_cycles: 5, cycle_order: shuffled}
#   Old: LLEM_PROFILE env var → New: REMOVED (no equivalent needed)
```

---

## Behaviour When File is Missing

If `~/.config/llenergymeasure/config.yaml` does not exist:
- All backends default to `local` runner
- Execution gap defaults: `config_gap_seconds=60`, `cycle_gap_seconds=300` (Pydantic defaults)
- Results go to `./results/`, model cache to `~/.cache/huggingface`
- Energy backend: auto (Zeus if installed, else NVML)
- Carbon intensity: ~475 gCO2/kWh (global average), PUE: 1.0
- UI: interactive prompts enabled, standard verbosity
- No error, no warning — zero-config works out of the box

This is the correct behaviour for a new user who has just done `pip install llenergymeasure`.

---

## Precedence

All fields follow a common pattern: **built-in default → user config → env var → study file → CLI flag**.

```
Priority (highest last):

1. Built-in defaults (coded into Pydantic models)
   e.g. n_cycles=1, config_gap_seconds=60, carbon_intensity=None

2. User config file (~/.config/llenergymeasure/config.yaml)
   e.g. measurement.carbon_intensity_gco2_kwh: 350, execution.config_gap_seconds: 120

3. Environment variables (for CI/Docker where config file is unavailable)
   LLEM_RUNNER_PYTORCH, LLEM_RUNNER_VLLM, LLEM_RUNNER_TENSORRT
   LLEM_CARBON_INTENSITY, LLEM_DATACENTER_PUE
   LLEM_VERBOSITY, LLEM_NO_PROMPT
   Note: LLEM_PROFILE removed (2026-02-20) — execution profiles system removed
   Note: LLEM_DATACENTER_LOCATION removed (2026-02-25) — location lookup delegated to CodeCarbon

4. Study file execution: block (n_cycles, cycle_order, gap overrides — portable, versioned)
   e.g. execution: {n_cycles: 5, cycle_order: shuffled, config_gap_seconds: 120}

5. CLI flags (per-invocation, highest priority)
   --runner docker, --output ./my-results/, --cycles 1, --no-gaps, --order shuffled
```

**Detailed precedence by concern:**

For runner selection:
```
1. built-in: local for all backends
2. user config: runners.<backend>
3. LLEM_RUNNER_<BACKEND> env var
4. study file: (no runner: field — removed 2026-02-19)
5. CLI flag: llem run file.yaml --runner docker       ← highest
```

For execution settings:
```
1. Pydantic defaults:  n_cycles=1, cycle_order="sequential", config_gap_seconds=60, cycle_gap_seconds=300
2. User config:        execution.config_gap_seconds / execution.cycle_gap_seconds
   (n_cycles + cycle_order are NOT in user config — they belong in study.yaml)
3. Study file:         execution: { n_cycles, cycle_order, [gap overrides] }
4. CLI flag:           --cycles N, --no-gaps, --order X, --profile quick|publication  ← highest
```

> **Superseded 2026-02-20:** The previous chain included `execution_profiles.<name>` (step 2) and
> `LLEM_PROFILE` env var (step 3). Both removed — execution profiles system rejected.
> See decisions/study-execution-model.md Decision A.

For infrastructure context (carbon intensity, PUE):
```
1. User config default:  measurement.carbon_intensity_gco2_kwh / .datacenter_pue
2. Env var override:     LLEM_CARBON_INTENSITY / LLEM_DATACENTER_PUE
3. Stored as Layer 3 in result JSON at measurement time (immutable after)
```

> **Superseded 2026-02-20:** experiment.yaml previously had `carbon_intensity_gco2_kwh`,
> `datacenter_pue`, `datacenter_location` as per-experiment overrides (step 4). Removed —
> experiment.yaml must be infrastructure-agnostic. Env vars serve the one-off override case cleanly.
>
> **Superseded 2026-02-25:** `datacenter_location` removed from user config and env vars.
> Location-based carbon intensity lookup is delegated entirely to CodeCarbon.
> See decisions/carbon-intensity.md.
> See decisions/architecture.md § Override Precedence for Infrastructure Context.

### Decision: Environment Variable Override Layer

**Decided 2026-02-19. Rationale below.**

Every mature CLI tool in this space supports env var overrides:

| Tool | Env var pattern |
|---|---|
| MLflow | `MLFLOW_TRACKING_URI` |
| DVC | `DVC_REMOTE`, `DVC_CONFIG_DIR` |
| Nextflow | `NXF_EXECUTOR`, `NXF_WORK` |
| gh CLI | `GH_TOKEN`, `GH_HOST` |

Without env vars, CI pipelines and Docker containers cannot configure the tool without
writing a config file at runtime (fragile, requires write access to `~/.config/`).
With env vars, a CI step can set `LLEM_RUNNER_PYTORCH=local` and the tool works correctly
in any container without a mounted config file.

**Supported env vars (v2.0):**
```
LLEM_RUNNER_PYTORCH=local|docker:<image>
LLEM_RUNNER_VLLM=local|docker:<image>
LLEM_RUNNER_TENSORRT=local|docker:<image>
LLEM_CARBON_INTENSITY=<float>            # gCO2/kWh override
LLEM_DATACENTER_PUE=<float>             # PUE override
LLEM_NO_PROMPT=1                        # disable interactive prompts (CI mode)
```

> **Removed (2026-02-20):** `LLEM_PROFILE=quick|standard|publication|<custom-name>` —
> the execution profiles system was rejected. Use `--profile quick` CLI flag or set
> `execution:` fields directly in study.yaml.

Env vars sit above the config file and below study.yaml in the precedence chain — they
override environment defaults but can be further overridden by the study or CLI.

---

## Singularity: NotImplementedError in v2.0

**Decided 2026-02-19.**

Singularity/Apptainer runner support is planned for a future release (HPC deployment milestone).
The user config schema **accepts** `singularity:<path>` values to avoid forcing users
to change their config files in the future, but the v2.0 executor raises a clear error at
validation time (not runtime):

```python
# In UserConfig.validate_runners():
if runner_value.startswith("singularity:"):
    raise NotImplementedError(
        f"Singularity runner is not yet supported (runners.{backend} = '{runner_value}'). "
        "Use 'local' or 'docker:<image>' for now."
    )
```

This gives a clear, actionable error immediately at `llem run` startup — not mid-run.

---

## Config Validation

**Decided 2026-02-19.**

The user config is validated at tool startup, not lazily at experiment dispatch time.
Bad config must surface immediately with a precise error:

```
Error: ~/.config/llenergymeasure/config.yaml is invalid
  runners.pytorch: expected 'local', 'docker:<image>', or 'singularity:<path>'
  got: 'docker'  (missing image name after colon — did you mean 'docker:ghcr.io/...'?)
```

**Where validation occurs:**
- `llem config` — always parses and validates user config as part of env snapshot
- `llem run` — validates user config at startup before any experiment runs
- `llem run` — validates user config at startup (for runner selection)

If the file is missing entirely → zero-config defaults apply, no error.
If the file exists but is malformed YAML → clear parse error with line number.
If the file is valid YAML but schema-invalid → Pydantic validation error with field path.

**Implementation**: `UserConfig` is a Pydantic model. `load_user_config()` calls
`UserConfig.model_validate(yaml.safe_load(path))` and re-raises `ValidationError`
as a `UserConfigError` with a formatted message.

---

## Runner Selection Logic

**Decided 2026-02-19**: `runner:` field removed from study.yaml entirely (Option A).

study.yaml is silent on execution environment. Runner resolution, in precedence order:

```
1. LLEM_RUNNER_<BACKEND> env var            ← highest — CI/Docker/scripted use
2. ~/.config/llenergymeasure/config.yaml    ← user config per-backend
3. Default: local                           ← ⚠️ UNDECIDED — see below
```

CLI flag `llem run study.yaml --runner docker` overrides all of the above for all backends.

### Why `runner:` was removed from study.yaml

No peer tool embeds execution environment selection in the portable pipeline/study file
(Nextflow, Snakemake, DVC, lm-eval, Hydra — all unanimous). The study/pipeline definition
describes **what** to measure. **Where** to run it is a runtime concern.

Named runner profiles (`runner: my-hpc-cluster`) also removed — they pointed at a schema
section (`runner_profiles:`) that didn't exist, and the use case (switching between named
environment sets) is covered by: different `~/.config/llenergymeasure/config.yaml` files
per machine + `runner: auto` (now implicit).

---

## ⚠️ UNDECIDED: Defaults and User Config Setup

**Status**: Discussion captured 2026-02-19. Not yet decided. Needs research.
**See**: [../decisions/open-questions.md](../decisions/open-questions.md) Q3

### The core unresolved question

Should `local` be the default runner at all? Or should Docker always be the default
(or at minimum always required) for measurement-grade isolation?

The argument for Docker-as-default:
- Docker provides environment isolation (pinned deps, CUDA version, OS packages)
- Removes "works on my machine" variation from results
- multiprocessing.Process provides GPU state isolation but NOT environment reproducibility
- For a research measurement tool, environment reproducibility is arguably as important
  as GPU state isolation

The argument for local-as-default:
- Zero-config first use: `pip install llenergymeasure[pytorch]` → `llem run` just works
- Docker is a system dependency (not pip-managed), cannot be assumed
- Thermal isolation is the primary concern; Docker doesn't improve thermal measurement
- Peers (optimum-benchmark) use local multiprocessing for research-grade benchmarks

**This needs research before deciding.** See open-questions.md Q3.

### Current working assumption (subject to change)

Pending that research, the working design is:

**v2.0 (local single-backend initially, Docker multi-backend later in v2.0):**
- No user config → all backends → `local` (single-backend default)
- Multi-backend study → hard error with guidance ("Docker required, available as v2.0 milestone")
- Zero setup for new users for single-backend studies
- Docker multi-backend available as a later v2.0 milestone

**Docker multi-backend (v2.0 milestone):**
- Runner resolved from user config → env var → CLI flag → `local` fallback
- Multi-backend detected with no Docker configured → informative error with exact config to paste
- No auto-pulling of Docker images without explicit user intent

### TODO: Decide before Docker multi-backend design lock

- [ ] Research whether Docker isolation is materially better than process isolation for
      energy measurement reproducibility (environment variation vs thermal variation)
- [ ] Research what optimum-benchmark/AIEnergyScore/MLPerf actually observe in practice:
      do local process runs produce reproducible measurements across environments?
- [ ] Decide: is Docker the default for ALL experiments, or only for multi-backend?
- [ ] Decide: what happens when user runs multi-backend without user config —
      Option 1 (hard error + guidance) or Option 2 (auto-detect Docker + confirm)?
- [ ] Decide: default Docker images (auto-select by version + CUDA) — where is this logic?

### llem config as onboarding UX (working assumption)

`llem config` bridges the gap between "I have nothing configured" and "I know what to do".
When user config is missing, it shows what's available and what config would enable it:

```
llem config

  GPU:     NVIDIA A100 80GB  CUDA 12.4  ✓
  PyTorch: installed (2.5.1)            ✓
  vLLM:    not installed locally
           Docker image available: ghcr.io/llenergymeasure/vllm:2.2.0-cuda12  ✓

  User config: not found

  To enable vLLM via Docker, add to ~/.config/llenergymeasure/config.yaml:
    runners:
      vllm: docker:ghcr.io/llenergymeasure/vllm:2.2.0-cuda12
```

No init command. The tool tells you what to paste.

---

## Example: Laptop (Local PyTorch Only, Minimal Config)

```yaml
# ~/.config/llenergymeasure/config.yaml — minimal, laptop
runners:
  pytorch: local

output:
  results_dir: ~/llem-results

measurement:
  carbon_intensity_gco2_kwh: 225    # UK grid average

execution:
  config_gap_seconds: 0             # idle laptop — no thermal gaps needed
  cycle_gap_seconds: 0
```

```yaml
# batch-size-effects.yaml — portable, no runner field, no machine-specific constants
model: meta-llama/Llama-3.1-8B
backend: pytorch
sweep:
  batch_size: [1, 8, 16]

execution:
  n_cycles: 1                       # quick iteration — set directly, not via profile
```

## Example: HPC Cluster (Multi-Backend, Singularity — future)

```yaml
# ~/.config/llenergymeasure/config.yaml — HPC
output:
  results_dir: /scratch/username/llem-results
  model_cache_dir: /scratch/models    # large model weights on scratch filesystem

runners:
  pytorch: local
  vllm:    singularity:/scratch/images/llem-vllm.sif    # future Singularity support
  tensorrt: singularity:/scratch/images/llem-trt.sif    # future Singularity support

measurement:
  carbon_intensity_gco2_kwh: 350    # German grid average
  datacenter_pue: 1.4               # typical HPC facility overhead

ui:
  prompt: false                     # no stdin in batch job environment

execution:
  config_gap_seconds: 120           # hot shared cluster — longer thermal gaps needed
  cycle_gap_seconds: 600
```

```yaml
# backend-comparison.yaml — identical file on laptop AND HPC
# no runner field, no machine-specific constants
model: meta-llama/Llama-3.1-8B
experiments:
  - backend: pytorch
    pytorch: { batch_size: 8 }
  - backend: vllm
    vllm: { max_num_seqs: 256 }

execution:
  n_cycles: 5                       # publication rigour — set directly in study file
  cycle_order: shuffled
```

The study file is identical on laptop and HPC. Only `~/.config/llenergymeasure/config.yaml` differs.

## Example: CI Pipeline (Env Vars, No Config File)

```yaml
# .github/workflows/benchmark.yml
env:
  LLEM_RUNNER_PYTORCH: local
  LLEM_NO_PROMPT: "1"
  # Quick iteration: use --profile quick CLI flag in the workflow step, or set n_cycles: 1 in study yaml
  # LLEM_PROFILE removed (2026-02-20) — execution profiles system removed
```

No config file needed in CI. Env vars configure the tool for the pipeline environment.

---

## Peer Reference

| Tool | User-local config path | Purpose |
|---|---|---|
| `uv` | `~/.config/uv/uv.toml` | Cache, index settings |
| `gh` CLI | `~/.config/gh/config.yml` | Default host, editor |
| Nextflow | `~/.nextflow/config` | Named executor profiles (local, SLURM, K8s) |
| DVC | `~/.config/dvc/config` | Remote storage, cache settings |
| Docker Compose | `~/.docker/config.json` | Registry credentials |

The pattern is universal: portable pipeline definition + user-local execution environment.
Modern tools use `~/.config/<tool>/` (XDG); older tools use `~/.toolname/` (pre-XDG).
