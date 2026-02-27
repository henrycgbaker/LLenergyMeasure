# N-C06: User Config Implementation vs Design Gap Analysis

**Module**: `src/llenergymeasure/config/user_config.py`
**Risk Level**: MEDIUM
**Decision**: Keep — v2.0 (sync the design doc)
**Planning Gap**: The actual `UserConfig` implementation is fundamentally different from what `designs/user-config.md` specifies. The implementation uses a project-local `.lem-config.yaml` file; the design specifies XDG user-global `~/.config/llenergymeasure/config.yaml`. The schema structures are entirely different. A Phase 5 developer following the design doc would discard the existing implementation rather than carrying it forward.

---

## What Exists in the Code

**Primary file(s)**: `src/llenergymeasure/config/user_config.py`
**Key classes/functions**:
- `ThermalGapConfig` (line 17) — Pydantic BaseModel for gap timing settings
- `DockerConfig` (line 32) — Pydantic BaseModel for Docker execution settings
- `NotificationsConfig` (line 50) — Pydantic BaseModel for webhook notifications
- `UserConfig` (line 67) — top-level Pydantic BaseModel
- `load_user_config(config_path: Path | None = None) -> UserConfig` (line 96) — loads from `.lem-config.yaml` in cwd

### Full `UserConfig` Schema (actual implementation)

```python
class ThermalGapConfig(BaseModel):
    between_experiments: float = 60.0    # seconds between experiments within a cycle
    between_cycles: float = 300.0        # seconds between cycles

class DockerConfig(BaseModel):
    strategy: Literal["ephemeral", "persistent"] = "ephemeral"
    warmup_delay: float = 0.0            # seconds to wait after container start
    auto_teardown: bool = True           # auto-teardown after campaign (persistent only)

class NotificationsConfig(BaseModel):
    webhook_url: str | None = None
    on_complete: bool = True
    on_failure: bool = True

class UserConfig(BaseModel):
    verbosity: Literal["quiet", "normal", "verbose"] = "normal"
    thermal_gaps: ThermalGapConfig = ThermalGapConfig()
    docker: DockerConfig = DockerConfig()
    notifications: NotificationsConfig = NotificationsConfig()
    results_dir: str = "results"
```

**Loading behaviour** (line 96–121): `load_user_config()` defaults to `Path(".lem-config.yaml")` in the current working directory. If the file does not exist, returns `UserConfig()` with all defaults — no error. If the file exists but is invalid YAML, raises `ValueError`. If valid YAML but schema-invalid (Pydantic `ValidationError`), raises `ValueError` with the validation message. The function uses `yaml.safe_load` + `UserConfig.model_validate(data)`.

## Design vs Implementation: Field-by-Field Gap Analysis

### What is in the design but NOT in the implementation

The design (`designs/user-config.md`) specifies:

```yaml
runners:
  pytorch: local
  vllm: docker:ghcr.io/llenergymeasure/vllm
  tensorrt: docker:ghcr.io/llenergymeasure/trt

execution_profiles:
  quick:
    n_cycles: 1
    cycle_order: interleaved
    config_gap_seconds: 0
    cycle_gap_seconds: 0
  standard:
    n_cycles: 3
    cycle_order: interleaved
    config_gap_seconds: 60
    cycle_gap_seconds: 300
  publication:
    n_cycles: 5
    cycle_order: shuffled
    config_gap_seconds: 120
    cycle_gap_seconds: 600
```

None of these fields (`runners`, `execution_profiles`) exist in the actual `UserConfig` model. The design's schema is entirely about:
1. **Runner configuration** — which Docker image or local environment to use per backend
2. **Named execution profiles** — reusable presets for study orchestration settings

### What is in the implementation but NOT in the design

The implementation has fields the design does not mention at all:

| Field | Type | Default | Not in design? |
|---|---|---|---|
| `verbosity` | `Literal["quiet", "normal", "verbose"]` | `"normal"` | Not mentioned |
| `thermal_gaps.between_experiments` | `float` | `60.0` | Partially overlaps `config_gap_seconds` in design |
| `thermal_gaps.between_cycles` | `float` | `300.0` | Partially overlaps `cycle_gap_seconds` in design |
| `docker.strategy` | `Literal["ephemeral", "persistent"]` | `"ephemeral"` | Not mentioned |
| `docker.warmup_delay` | `float` | `0.0` | Not mentioned |
| `docker.auto_teardown` | `bool` | `True` | Not mentioned |
| `notifications.webhook_url` | `str | None` | `None` | Not mentioned |
| `notifications.on_complete` | `bool` | `True` | Not mentioned |
| `notifications.on_failure` | `bool` | `True` | Not mentioned |
| `results_dir` | `str` | `"results"` | Not mentioned |

### File location mismatch

The implementation reads from `.lem-config.yaml` in the **current working directory** (project-local). The design specifies `~/.config/llenergymeasure/config.yaml` (user-global, via XDG). These are different files with different scopes and different intentions.

The design's rationale for XDG is well-argued (runner configuration is per-machine, not per-project; the same study.yaml should work on laptop and HPC without modification). The implementation's project-local approach is simpler but fundamentally limits the design's portability goal.

### Naming discrepancy

The design uses `config_gap_seconds` and `cycle_gap_seconds` inside `execution_profiles`. The implementation uses `thermal_gaps.between_experiments` and `thermal_gaps.between_cycles`. These cover the same concept with different names and structure. The design's names (`config_gap_seconds`, `cycle_gap_seconds`) match the `StudyConfig.execution` block field names — this alignment is intentional in the design.

### Module docstring discrepancy

The file docstring (line 1–6) states: "This is a minimal implementation for Phase 2.2 (thermal gaps, docker strategy). Full user preferences system (lem init wizard) is Phase 2.3." This is inconsistent with the planning session decisions that eliminated `llem init` entirely.

## Why It Matters

The user config file is the integration point between the portable study definition (`study.yaml`) and the machine-specific execution environment (which Docker images are available, where results go, how verbose to be). Getting this wrong means either:
- Study YAMLs become machine-specific (they embed Docker image names or gap timings) — breaking the portability goal
- The tool has no way to configure machine-specific behaviour without changing code

The `notifications.webhook_url` field in the existing implementation is a genuine addition the design does not cover. Webhook notification at experiment completion is a real user need for long-running studies (hours-long benchmark runs). It should not be silently dropped.

The `results_dir` field in the existing implementation is also missing from the design. Without it, results always go to `"results/"` relative to cwd. The design doc has no section on results persistence configuration.

## Recommendation for Phase 5

**Implement the design's schema (`runners` + `execution_profiles`) at the XDG path**, but migrate the valuable fields from the existing implementation:

1. **Keep from existing implementation**:
   - `verbosity` — add to the new schema as a top-level field
   - `notifications` sub-model — add as optional top-level field
   - `results_dir` — add as optional top-level field (or remove if CLI `--output` flag covers this)

2. **Discard from existing implementation** (superseded by design):
   - `thermal_gaps.between_experiments` → replaced by `execution_profiles.<name>.config_gap_seconds`
   - `thermal_gaps.between_cycles` → replaced by `execution_profiles.<name>.cycle_gap_seconds`
   - `docker.strategy` / `docker.warmup_delay` / `docker.auto_teardown` → replaced by runner config + Docker execution design

3. **Implement from design** (not yet in code):
   - `runners: dict[str, str]` — backend → runner string (`local`, `docker:<image>`, `singularity:<path>`)
   - `execution_profiles: dict[str, ExecutionProfile]` — named profile presets
   - File location: `~/.config/llenergymeasure/config.yaml` via `platformdirs.user_config_dir("llenergymeasure")`
   - Env var override layer: `LLEM_RUNNER_PYTORCH`, `LLEM_RUNNER_VLLM`, `LLEM_RUNNER_TENSORRT`, `LLEM_PROFILE`
   - Singularity `NotImplementedError` guard at validation time (not runtime)

4. **Update the module docstring** to remove the Phase 2.2/2.3 references, which no longer match the planning timeline.
