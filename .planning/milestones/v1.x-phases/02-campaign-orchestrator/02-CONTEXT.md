# Phase 2 Context: Campaign Orchestrator

Decisions captured during discussion phase. These guide research and planning — downstream agents should not re-ask these questions.

## Architecture Decision

**Host orchestrator + long-running containers.** The orchestrator runs as native Python on the host. Backend containers (PyTorch, vLLM, TensorRT) run as long-running `docker compose` services, receiving experiment dispatch via `docker compose exec`. No Docker-in-Docker, no socket mounting.

Rationale: Simpler, more secure (no root-equivalent socket access), matches how MLflow/BentoML work, and researchers shouldn't debug container networking. GPU passthrough handled natively by `docker compose`. State management via direct filesystem access.

```
Host (lem CLI — native Python)
    │
    ├── docker compose up pytorch vllm -d     (auto-managed by orchestrator)
    │
    ├── docker compose exec pytorch lem experiment config1.yaml
    ├── docker compose exec vllm lem experiment config2.yaml
    │
    └── Results ← mounted volume (host filesystem, direct access)
```

## Campaign Definition UX

### Grid + Explicit List
Support both modes in a single campaign YAML:
- **Grid mode**: Define axes (models, backends, precision), tool generates cartesian product
- **Explicit list**: Hand-pick specific experiment configs (file references or inline)
- Both can coexist in one campaign — grid for sweeps, explicit for special cases

### Two-Level Grid Design
Backend parameters split into shared (universal) and backend-specific tiers:

```yaml
campaign:
  name: "energy-comparison"
  cycles: 5

  # Shared grid — expands across ALL backends listed
  grid:
    backends: [pytorch, vllm]
    models: [gpt2, meta-llama/Llama-3.1-8B]
    fp_precision: [float16, bfloat16]
    decoder:
      preset: deterministic

  # Backend-specific sections — their own grid axes
  pytorch:
    batch_size: [1, 4, 8]
  vllm:
    max_num_seqs: [1, 4, 8]
    gpu_memory_utilization: 0.9

  # Explicit experiments (file references)
  experiments:
    - config: configs/tensorrt_special_case.yaml
```

**Shared params safe to grid across backends**: `fp_precision`, `decoder.*` (temperature, top_p, top_k, repetition_penalty), `max_input_tokens`, `max_output_tokens`, `streaming`.

**Backend-specific params (NOT interchangeable)**: batching (batch_size vs max_num_seqs vs max_batch_size), quantisation (different enum values per backend), parallelism (different param names), memory control.

**No auto-mapping** between semantically similar params (e.g. batch_size → max_num_seqs). Users must specify backend-specific params explicitly in their backend section.

### Config Style
Support both inline and file references:
- **Inline**: Campaign YAML contains grid axes and shared settings
- **File references**: `experiments: [{config: path/to/config.yaml}]`
- Inline overrides can be applied on top of referenced configs

### Invalid Combination Handling
**Auto-filter with summary**: SSOT introspection filters invalid backend × param combinations. Show clear summary before running: "Generated 48 experiments (12 filtered as invalid: TensorRT+float32, ...). Proceed?"

### Two-Layer Validation
1. **`lem campaign validate campaign.yaml`** — Dry-run (no GPU needed). Catches invalid combos, shows experiment count, validates all configs
2. **Runtime validation** — Each experiment validated again before dispatch (safety net)

### Default Cycles
Default to 1 cycle (backwards compatible). CLI displays warning: "Single cycle: confidence intervals and robustness metrics require >= 3 cycles (--cycles 3)". Bootstrap CI computed when cycles > 1.

## Manifest & Resumption

### Manifest Format
**JSON file** (`campaign_manifest.json`) alongside results. Contains:
- Campaign metadata (name, creation time, config hash)
- Per-experiment entries: exp_id → config → backend → container → status → result_path → timestamps
- Human-readable and git-trackable

### Resumption Semantics
**Skip completed, ask about failed.** On `lem campaign resume`:
- Experiments with successful results: skipped automatically
- Failed/interrupted experiments: prompt user — retry, skip, or abort
- New experiments (added to campaign YAML since last run): queued automatically

### Partial Results
**Always available.** Each experiment saves results independently (existing late-aggregation pattern). Campaign completion is a status flag, not a gate on result access. Users can inspect/export partial results at any time.

### Per-Experiment Failure
**Log and continue.** Failed experiment marked as `failed` in manifest with error details. Campaign continues with remaining experiments. Summary at end shows all failures.

## Container Lifecycle

### Management
**Orchestrator manages full lifecycle:**
- `lem campaign run` auto-starts needed backend containers
- Dispatches experiments via `docker compose exec`
- Tears down containers on campaign completion (or leaves running if user prefers — flag?)
- User doesn't touch `docker compose` directly during normal workflow

### Health Checks
**Restart + retry.** Orchestrator detects unhealthy container (e.g. OOM, GPU error), restarts it, retries the failed experiment. If retry also fails, marks as failed and continues.

### Progress Display
**Live terminal + quiet mode:**
- Default: Rich progress bar (current experiment, backend, cycle, ETA, completed/total)
- `--quiet` flag: log to file only, print summary at end (for daemon/unattended runs)

## Daemon & Scheduling

### Thermal Gaps
Automatic thermal cooldown between experiments. Configurable gap duration (default: 60s). GPU temperature monitored — can wait until below threshold before next experiment.

### Time-Based Scheduling
Optional cron-like scheduling for overnight/unattended campaigns:
```yaml
schedule:
  enabled: true
  at: "02:00"           # Start time
  interval: "6h"        # Between cycles
  total_duration: "48h"  # Max campaign duration
```

### Notification
**Log + optional webhook.** Campaign writes to log file. Optional webhook URL in campaign config for Slack/Discord/email notification on completion or failure:
```yaml
notification:
  webhook_url: "https://hooks.slack.com/..."
  on: [complete, failure]
```

## Campaign Grid Validation Architecture

### Current State: Three-Tiered Validation

The codebase has existing validation at three levels:

**Tier 1 — Pydantic Validators (Programmatic, Blocking)**
Validators on `ExperimentConfig` and backend config models. Run automatically on config instantiation:
- Backend-config mismatch (vllm section with pytorch backend)
- TensorRT + float32 rejection
- 4-bit + 8-bit mutual exclusion
- Field constraints (ge/le/Literal) enforced by Pydantic itself

**Tier 2 — Introspection Constraints (SSOT Lists, Non-blocking)**
Functions in `config/introspection.py` used by tests and doc generators:
- `get_backend_capabilities()` — derived from Pydantic model field existence (programmatic)
- `get_mutual_exclusions()` — hardcoded param pairs
- `get_streaming_constraints()` — params affected by streaming=True
- `get_validation_rules()` — cross-backend rules (hardcoded strings)
- `get_param_skip_conditions()` — hardware/environment requirements

**Tier 3 — Static Documentation (Manual)**
Generated docs call Tier 2 functions. Weakest tier, not enforced.

### Decision: Pydantic-First Dry-Run Validation

Campaign grid validation uses **Pydantic-first dry-run instantiation**:

1. **Expand grid** into individual experiment config dicts
2. **Attempt `ExperimentConfig(**config_dict)`** for each (dry-run) — catches ALL Tier 1 violations automatically. Any new validators added to Pydantic models are inherited for free.
3. **Overlay Tier 2 checks** — `get_backend_capabilities()` and `get_param_skip_conditions()` flag experiments that are valid configs but likely to fail at runtime (needs Ampere GPU, needs pre-quantised model). These are **warnings**, not errors.
4. **Report summary**: "Generated 48 experiments (12 filtered as invalid: TensorRT+float32 × 4, ...). 6 have hardware warnings."

```
Campaign YAML Grid
       │
       ├──→ expand_grid() → List[dict]  (cartesian product)
       │
       ├──→ For each: ExperimentConfig(**dict)
       │    ├── Valid → queued
       │    └── ValidationError → filtered (with reason)
       │
       ├──→ get_backend_capabilities() → pre-filter unsupported combos
       ├──→ get_param_skip_conditions() → flag hardware-dependent combos
       │
       └──→ Summary report (valid / filtered / warned)
```

**Why this approach:**
- Self-maintaining — new Pydantic validators auto-participate
- No separate campaign validation module with duplicate rules
- Environmental constraints (GPU, libraries) stay as warnings via Tier 2
- Existing `lem config validate` already does single-config Pydantic validation; campaign validate extends this to grids

## IO Path Configuration

**Discussion point (not yet decided):** Currently only `io.results_dir` exists. Campaign config should handle paths for:
- Results directory
- Config directory (for file references)
- State/manifest directory
- Docker volume mapping

Consider a broader `io` section in campaign YAML with sensible defaults. To be resolved during planning.

## Deferred Ideas

None captured — all discussion stayed within Phase 2 scope.
