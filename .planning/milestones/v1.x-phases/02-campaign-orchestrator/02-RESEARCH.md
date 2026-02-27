# Phase 2: Campaign Orchestrator - Research

**Researched:** 2026-01-29
**Domain:** Multi-backend campaign orchestration with long-running Docker containers and manifest-based state management
**Confidence:** HIGH

## Summary

Phase 2 implements campaign orchestration for multi-backend LLM benchmarking by transforming the current "docker compose run --rm" pattern (30-60s overhead per experiment) into long-running container architecture with `docker compose exec` dispatch. This is a **brownfield extension** of existing campaign infrastructure (`cli/campaign.py`, `orchestration/campaign.py`, `config/campaign_config.py`) which already handles multi-config, multi-cycle execution with thermal gaps and randomisation.

The codebase provides strong foundations that need extension, not replacement: `CampaignRunner` in `orchestration/campaign.py` already generates execution order (interleaved/shuffled/grouped), `CampaignConfig` in `config/campaign_config.py` defines campaign YAML structure, `cli/campaign.py` implements the CLI command with backend detection and Docker dispatch, `StateManager` in `state/experiment_state.py` demonstrates atomic persistence patterns, and `config/introspection.py` provides SSOT for parameter validation. Current implementation uses `docker compose run --rm` per experiment (lines 467-488 in `cli/campaign.py`), which must transition to long-running containers.

Standard approach: **python-on-whales (0.70+)** for Docker orchestration (official Docker SDK alternative with native Compose V2 support), host-side orchestrator in native Python managing backend containers via `docker.compose.execute(service, command)`, campaign manifest as JSON with atomic state updates following existing `StateManager` pattern, backend-aware grid generation extending `config generate-grid` with mutual exclusion filters, health checks via NVML (not just `torch.cuda.is_available()`) to detect GPU memory leaks, and bootstrap confidence intervals using numpy percentile method (frequentist coverage guarantees from FAQ paper 2026).

**Primary recommendation:** Extend existing campaign infrastructure incrementally. Start with container lifecycle (up/exec/down), add manifest persistence using `StateManager` pattern, then layer in health monitoring and grid validation. Keep orchestrator on host (not Docker-in-Docker), use shared volumes for results, implement graceful degradation (log and continue on container failures).

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| python-on-whales | 0.70+ | Docker Compose orchestration | Docker-endorsed, native Compose V2 exec support, thread-safe, CLI parity, better than docker-py for Compose workflows |
| numpy | latest | Bootstrap resampling for CIs | Already dependency, percentile method for frequentist coverage guarantees (FAQ 2026 paper) |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pydantic | 2.0+ | Campaign manifest models | Already used, extend for manifest schema |
| nvidia-ml-py | 13.590.48+ | Health check NVML queries | Already dependency (Phase 1), detect GPU memory issues |

### Already in Codebase
| Module | Purpose | How Phase 2 Uses It |
|--------|---------|---------------------|
| `cli/campaign.py` | Campaign CLI command | Extend with exec-based dispatch (replace lines 467-488) |
| `orchestration/campaign.py` | CampaignRunner, execution order | Wrap with container lifecycle, add manifest integration |
| `config/campaign_config.py` | CampaignConfig model | Extend with health_check, force_cold_start options |
| `state/experiment_state.py` | StateManager atomic writes | Copy pattern for campaign manifest persistence |
| `config/introspection.py` | SSOT parameter metadata | Use for backend-aware grid validation |
| `config/config.py` | generate-grid command | Extend with backend filtering logic |
| `results/aggregation.py` | ResultAggregator | Extend with bootstrap CI for multi-cycle data |

**Installation:**
```bash
# New dependency
pip install python-on-whales>=0.70

# Already satisfied by existing dependencies
pip install pydantic>=2.0 numpy nvidia-ml-py>=13.590.48
```

## Codebase Patterns to Follow

### Pattern 1: CLI Command Registration (Follow Existing Pattern)
**What:** Commands register via `_register_commands()` in `cli/__init__.py` (lines 79-107). Campaign command already registered at line 103.

**When to use:** Extending campaign command with new options (manifest resume, force-cold-start flag).

**Example:**
```python
# Source: cli/__init__.py lines 79-107
def _register_commands() -> None:
    """Register all commands with the app."""
    from llenergymeasure.cli import batch, campaign, experiment, listing, schedule

    # Core experiment commands
    app.command("run")(experiment.run_cmd)
    app.command("experiment")(experiment.experiment_cmd)
    app.command("aggregate")(experiment.aggregate_cmd)

    # Campaign already registered - extend its options, don't add new command
    app.command("campaign")(campaign.campaign_cmd)
```

**Don't hand-roll:** Command registration is already handled. Extend existing `campaign_cmd` function in `cli/campaign.py`, don't create new command.

### Pattern 2: Atomic State Persistence (StateManager Pattern)
**What:** `StateManager` in `state/experiment_state.py` uses temp-file-then-rename for atomic writes. Campaign manifest should follow identical pattern.

**When to use:** Persisting campaign manifest with experiment status, container assignments, failure tracking.

**Example:**
```python
# Source: state/experiment_state.py StateManager._save() pattern
def save_campaign_manifest(manifest: CampaignManifest, path: Path) -> None:
    """Save manifest atomically following StateManager pattern."""
    tmp_path = path.with_suffix(".tmp")

    # Write to temp file first
    with open(tmp_path, "w") as f:
        json.dump(manifest.model_dump(), f, indent=2)

    # Atomic rename (POSIX guarantees atomicity)
    tmp_path.rename(path)
```

**Don't hand-roll:** State persistence. Copy StateManager's atomic write pattern, don't use direct file writes.

### Pattern 3: Pydantic Domain Models (Existing Schema)
**What:** All configs and results use Pydantic models with validation. Campaign manifest needs similar structure.

**When to use:** Defining campaign manifest schema for exp_id → config → backend → status tracking.

**Example:**
```python
# Source: config/campaign_config.py existing CampaignConfig
from pydantic import BaseModel, Field

class CampaignManifestEntry(BaseModel):
    """Single experiment in campaign manifest."""
    exp_id: str
    config_name: str
    config_path: str
    backend: str
    container: str  # Service name: pytorch, vllm, tensorrt
    cycle_index: int
    status: Literal["pending", "running", "completed", "failed"]
    result_path: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None

class CampaignManifest(BaseModel):
    """Full campaign manifest with all experiments."""
    campaign_id: str
    campaign_name: str
    created_at: datetime
    experiments: list[CampaignManifestEntry]
```

**Don't hand-roll:** Schema definitions. Use Pydantic models, follow existing patterns in `domain/` and `config/`.

### Pattern 4: Backend Detection (Already Implemented)
**What:** `cli/campaign.py` lines 406-421 detect backend from config, line 424-446 detect if running in Docker vs host.

**When to use:** Routing experiments to correct backend container.

**Example:**
```python
# Source: cli/campaign.py lines 406-421, 424-446
def _detect_backend(config_data: dict[str, object]) -> str:
    """Detect backend from experiment config."""
    backend = config_data.get("backend")
    if backend:
        return str(backend).lower()
    return os.environ.get("LEM_BACKEND", "pytorch")

def _should_use_docker() -> bool:
    """Determine if we're running on host (should orchestrate containers)."""
    # Check /.dockerenv or /proc/1/cgroup
    if Path("/.dockerenv").exists():
        return False
    # ... existing logic
    return True
```

**Don't hand-roll:** Backend routing logic. It exists, extend it for exec-based dispatch.

### Pattern 5: Config Grid Generation (Extend Existing)
**What:** `config.py` lines 492-686 implement `generate-grid` command with cartesian product. Needs backend-awareness.

**When to use:** Generating campaign configs with valid backend × param combinations.

**Current limitation:** Grid generation doesn't filter invalid backend combinations (e.g., pytorch.batch_size with vllm backend). Phase 2 adds validation.

**Example extension:**
```python
# Source: config/config.py generate-grid command (extend lines 612-623)
# After ExperimentConfig instantiation:
config_obj = ExperimentConfig(**config_dict)

# NEW: Validate backend-specific params are compatible
from llenergymeasure.config.introspection import get_backend_params
backend = config_obj.backend
backend_params = get_backend_params(backend)

# Check for invalid cross-backend params
for param, value in config_dict.items():
    if "." in param:  # e.g., "pytorch.batch_size"
        param_backend = param.split(".")[0]
        if param_backend != backend and param_backend in ["pytorch", "vllm", "tensorrt"]:
            # Invalid: pytorch param in vllm config
            invalid_configs.append((output_path, f"Invalid param '{param}' for backend '{backend}'"))
```

**Don't hand-roll:** Grid generation. Extend existing command with backend validation layer.

## Architecture Patterns

### Recommended Campaign Architecture
```
Host (Native Python Orchestrator)
    ↓
python-on-whales SDK
    ↓
Docker Compose (long-running services)
    ├── pytorch (detached, health-checked)
    ├── vllm (detached, health-checked)
    └── tensorrt (detached, health-checked)
    ↓
Shared volumes for results
    ├── /app/results (bind mount)
    ├── /app/configs (bind mount)
    └── /app/.state (named volume)
```

### Pattern 1: Long-Running Container Lifecycle with python-on-whales
**What:** Replace `docker compose run --rm` with `docker compose up` (start once) + `docker compose execute` (dispatch many).

**Why:** Eliminates 30-60s container startup overhead per experiment. Preserves warmup state across experiments when desired.

**Example:**
```python
# Source: python-on-whales API documentation
from python_on_whales import DockerClient

docker = DockerClient(compose_files=["./docker-compose.yml"])

# 1. Start needed backend containers (detached)
docker.compose.up(["pytorch", "vllm"], detach=True, wait=True)

# 2. Wait for health checks to pass
# docker.compose automatically waits for healthy status when wait=True

# 3. Execute experiments via exec (no container restart)
for experiment in campaign_manifest.experiments:
    backend = experiment.backend  # "pytorch", "vllm", or "tensorrt"

    # Execute command in running container
    result = docker.compose.execute(
        service=backend,
        command=["lem", "experiment", f"/app/configs/{experiment.config_path}"],
        tty=False,  # Get stdout/stderr
        envs={"EXPERIMENT_ID": experiment.exp_id}
    )

    # Process result, update manifest
    experiment.status = "completed" if result.returncode == 0 else "failed"

# 4. Cleanup after campaign
docker.compose.down()
```

**When to use:** Campaign orchestration with multiple experiments per backend.

**Don't hand-roll:** Docker container lifecycle management. Use python-on-whales abstraction, don't call subprocess with docker commands.

### Pattern 2: Campaign Manifest with Resume Capability
**What:** JSON manifest tracking exp_id → config → backend → container → status → result_path for all campaign experiments. Enables resume on failure.

**When to use:** Multi-hour campaigns where failures should allow resume, not full restart.

**Example:**
```python
# Manifest structure
{
    "campaign_id": "a3f891c2",
    "campaign_name": "pytorch-vs-vllm-comparison",
    "created_at": "2026-01-29T10:30:00",
    "experiments": [
        {
            "exp_id": "exp_20260129_103000_001",
            "config_name": "pytorch_batch1",
            "config_path": "configs/pytorch_batch1.yaml",
            "backend": "pytorch",
            "container": "pytorch",
            "cycle_index": 0,
            "status": "completed",
            "result_path": "results/raw/exp_20260129_103000_001/",
            "started_at": "2026-01-29T10:30:05",
            "completed_at": "2026-01-29T10:32:18"
        },
        {
            "exp_id": "exp_20260129_103220_002",
            "config_name": "vllm_base",
            "config_path": "configs/vllm_base.yaml",
            "backend": "vllm",
            "container": "vllm",
            "cycle_index": 0,
            "status": "failed",
            "error": "CUDA out of memory",
            "started_at": "2026-01-29T10:32:20",
            "completed_at": "2026-01-29T10:32:45"
        },
        {
            "exp_id": "exp_20260129_103250_003",
            "config_name": "pytorch_batch1",
            "config_path": "configs/pytorch_batch1.yaml",
            "backend": "pytorch",
            "container": "pytorch",
            "cycle_index": 1,
            "status": "pending"
        }
    ]
}

# Resume logic
def resume_campaign(manifest_path: Path) -> None:
    """Resume campaign from manifest, skip completed experiments."""
    manifest = load_campaign_manifest(manifest_path)

    # Filter to pending and failed experiments
    remaining = [
        exp for exp in manifest.experiments
        if exp.status in ["pending", "failed"]
    ]

    if not remaining:
        console.print("[green]All experiments completed[/green]")
        return

    console.print(f"Resuming: {len(remaining)} experiments remaining")

    # Ask about failed experiments
    failed = [exp for exp in remaining if exp.status == "failed"]
    if failed:
        retry_failed = Confirm.ask(
            f"{len(failed)} experiments failed. Retry them?",
            default=True
        )
        if not retry_failed:
            remaining = [exp for exp in remaining if exp.status != "failed"]

    # Resume execution
    for exp in remaining:
        run_experiment(exp)
        save_campaign_manifest(manifest, manifest_path)
```

**When to use:** All campaigns. Provides crash recovery and failure inspection.

**Don't hand-roll:** Resume logic from scratch. Follow StateManager's persistence pattern, extend CampaignRunner with resume capability.

### Pattern 3: Health Check Daemon with NVML Validation
**What:** Periodic health checks using NVML to detect GPU memory leaks, not just `torch.cuda.is_available()`. Auto-restart on failure.

**When to use:** Long-running campaigns where container state can degrade (vLLM memory leaks documented in research).

**Example:**
```python
# Health check beyond docker-compose.yml healthcheck
def check_container_health(service: str, docker: DockerClient) -> tuple[bool, str]:
    """Application-aware health check using NVML."""
    try:
        # Execute health check command in container
        result = docker.compose.execute(
            service=service,
            command=["python", "-c", "import pynvml; pynvml.nvmlInit(); h = pynvml.nvmlDeviceGetHandleByIndex(0); mem = pynvml.nvmlDeviceGetMemoryInfo(h); print(mem.used)"],
            tty=False
        )

        # Check memory usage (detect leaks)
        mem_used_bytes = int(result.strip())
        mem_used_gb = mem_used_bytes / 1e9

        # Threshold: >90% VRAM used suggests leak
        total_vram_gb = 80  # A100 example
        if mem_used_gb > total_vram_gb * 0.9:
            return False, f"GPU memory high: {mem_used_gb:.1f}GB / {total_vram_gb}GB"

        return True, "OK"

    except Exception as e:
        return False, f"Health check failed: {e}"

# Daemon pattern
def health_monitor_loop(services: list[str], docker: DockerClient, interval: int = 30):
    """Background health monitoring with auto-recovery."""
    while True:
        for service in services:
            healthy, reason = check_container_health(service, docker)

            if not healthy:
                console.print(f"[yellow]Unhealthy: {service} - {reason}[/yellow]")
                console.print(f"[yellow]Restarting {service}...[/yellow]")

                # Restart container
                docker.compose.restart([service])

                # Wait for health recovery
                time.sleep(10)

                healthy_after, reason_after = check_container_health(service, docker)
                if healthy_after:
                    console.print(f"[green]Recovered: {service}[/green]")
                else:
                    console.print(f"[red]Failed to recover: {service} - {reason_after}[/red]")

        time.sleep(interval)
```

**When to use:** Campaigns with >10 experiments per backend. Detects state degradation early.

**Don't hand-roll:** NVML queries. Reuse patterns from Phase 1's power sampling, follow existing `GPUUtilisationSampler` thread patterns.

### Pattern 4: Backend-Aware Grid Validation via SSOT
**What:** Extend `config generate-grid` to filter invalid backend × param combinations using `config/introspection.py` SSOT.

**When to use:** Generating campaign grids with multi-backend configs. Prevents cartesian explosion with invalid combos.

**Example:**
```python
# Extension to config/config.py generate-grid
from llenergymeasure.config.introspection import get_backend_params

def validate_config_for_backend(config_dict: dict[str, Any]) -> tuple[bool, str]:
    """Validate config uses only params valid for its backend."""
    backend = config_dict.get("backend", "pytorch")
    valid_params = get_backend_params(backend)

    # Check for cross-backend contamination
    backend_sections = ["pytorch", "vllm", "tensorrt"]
    for section in backend_sections:
        if section != backend and section in config_dict:
            # Invalid: has params for different backend
            return False, f"Config has '{section}' section but backend is '{backend}'"

    # Check nested params are valid
    if backend in config_dict:
        backend_config = config_dict[backend]
        for param in backend_config.keys():
            full_param = f"{backend}.{param}"
            if full_param not in valid_params:
                return False, f"Invalid param '{full_param}' for backend '{backend}'"

    return True, "OK"

# In generate-grid command, add validation:
for output_path, config_dict in config_variations:
    # Existing validation
    config_obj = ExperimentConfig(**config_dict)

    # NEW: Backend-aware validation
    valid, reason = validate_config_for_backend(config_dict)
    if not valid:
        invalid_configs.append((output_path, reason))
        continue

    # Write valid config
    valid_configs.append((output_path, config_dict))
```

**When to use:** All campaign grid generation. Prevents wasted Docker exec calls for invalid configs.

**Don't hand-roll:** Parameter introspection. Use existing `config/introspection.py` SSOT module, it already knows all valid params per backend.

### Pattern 5: Bootstrap Confidence Intervals for Multi-Cycle Results
**What:** Implement 95% CI via percentile bootstrap (FAQ 2026 paper recommendations) for multi-cycle campaign data.

**When to use:** Campaign results display/export. Provides statistical rigor for research credibility.

**Example:**
```python
# Extend results/aggregation.py with bootstrap CI
import numpy as np

def bootstrap_ci(samples: list[float], n_iterations: int = 1000, confidence: float = 0.95) -> tuple[float, float]:
    """Compute bootstrap confidence interval using percentile method.

    Args:
        samples: Raw measurements from multiple cycles.
        n_iterations: Bootstrap iterations (1000 standard, FAQ 2026).
        confidence: Confidence level (0.95 for 95% CI).

    Returns:
        (lower_bound, upper_bound) for confidence interval.

    Source: FAQ paper (arXiv:2601.20251) bootstrap recommendations.
    """
    samples_array = np.array(samples)
    n = len(samples_array)

    bootstrap_means = []
    rng = np.random.default_rng(42)  # Deterministic for reproducibility

    for _ in range(n_iterations):
        # Resample with replacement
        resampled = rng.choice(samples_array, size=n, replace=True)
        bootstrap_means.append(np.mean(resampled))

    # Percentile method
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower = np.percentile(bootstrap_means, lower_percentile)
    upper = np.percentile(bootstrap_means, upper_percentile)

    return lower, upper

# Usage in campaign aggregation
def aggregate_campaign_results(campaign_manifest: CampaignManifest) -> dict[str, dict[str, Any]]:
    """Aggregate results by config with confidence intervals."""
    results_by_config: dict[str, list[AggregatedResult]] = {}

    # Group results by config
    for exp in campaign_manifest.experiments:
        if exp.status == "completed" and exp.result_path:
            result = load_aggregated_result(exp.result_path)
            if exp.config_name not in results_by_config:
                results_by_config[exp.config_name] = []
            results_by_config[exp.config_name].append(result)

    # Compute statistics with CIs
    aggregated = {}
    for config_name, results in results_by_config.items():
        # Extract metrics across cycles
        energy_samples = [r.energy_consumption_mwh for r in results]
        ttft_samples = [r.ttft_mean_s for r in results]

        # Compute CIs
        energy_ci = bootstrap_ci(energy_samples)
        ttft_ci = bootstrap_ci(ttft_samples)

        aggregated[config_name] = {
            "n_cycles": len(results),
            "energy_mwh": {
                "mean": np.mean(energy_samples),
                "std": np.std(energy_samples),
                "ci_lower": energy_ci[0],
                "ci_upper": energy_ci[1],
            },
            "ttft_s": {
                "mean": np.mean(ttft_samples),
                "std": np.std(ttft_samples),
                "ci_lower": ttft_ci[0],
                "ci_upper": ttft_ci[1],
            },
        }

    return aggregated
```

**When to use:** Campaign results with ≥3 cycles. Provides frequentist coverage guarantees for research publication.

**Don't hand-roll:** Statistical methods. Use numpy percentile bootstrap (well-validated, FAQ 2026 recommendations), not custom CI calculations.

## Don't Hand-Roll

Problems that look simple but have existing solutions in the codebase:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Docker container management | subprocess calls to `docker` CLI | python-on-whales SDK | Thread-safe, handles stdout/stderr, native Compose V2 support, Docker-endorsed |
| Atomic state persistence | Direct `open(file, "w").write()` | StateManager pattern (state/experiment_state.py) | Prevents corruption on crashes, POSIX atomic rename guarantees |
| Campaign config parsing | Custom YAML parsing | Existing CampaignConfig (config/campaign_config.py) | Pydantic validation, config path resolution, already tested |
| Backend parameter validation | Hardcoded param lists per backend | SSOT introspection (config/introspection.py) | Auto-discovers from Pydantic models, single source of truth |
| Execution order generation | Custom scheduling logic | CampaignRunner (orchestration/campaign.py) | Handles interleaved/shuffled/grouped, cycle tracking, warmup management |
| Confidence intervals | Custom statistics | numpy percentile + bootstrap | Well-validated, frequentist coverage guarantees (FAQ paper) |
| Container health checks | Just torch.cuda.is_available() | NVML memory queries (nvidia-ml-py) | Detects memory leaks, state degradation (Phase 1 power sampling pattern) |
| Backend routing | Manual if/else chains | Existing _detect_backend() (cli/campaign.py line 406) | Already handles config parsing, env vars, defaults |

**Key insight:** 80% of campaign orchestration logic already exists. Phase 2 is about transitioning from subprocess `docker run` to SDK-based exec dispatch, adding persistent manifest, and enhancing validation. Don't rewrite existing CampaignRunner, config models, or state persistence — extend them.

## Common Pitfalls

### Pitfall 1: Docker-in-Docker Complexity Trap
**What goes wrong:** Mounting `/var/run/docker.sock` into orchestrator container creates permission issues, version mismatches, nested network complications.

**Why it happens:** "Orchestrator should run in Docker" seems clean, but Docker-in-Docker is notoriously problematic (security risks, volume mounting issues, NVIDIA GPU passthrough complexity).

**How to avoid:** Run orchestrator as **native Python process on host**. Use python-on-whales from host to manage backend containers. Host has full filesystem access, no socket mounting needed, direct GPU access for NVML health checks.

**Warning signs:**
- Needing to mount `/var/run/docker.sock`
- Orchestrator container failing to see host volumes
- Permission denied errors for bind mounts
- NVIDIA runtime issues in nested containers

### Pitfall 2: Container State Leakage Accumulation
**What goes wrong:** Long-running containers accumulate GPU memory, model cache grows unbounded, vLLM/TensorRT memory leaks cause OOM failures after 10+ experiments.

**Why it happens:** `force_cold_start: false` (default) keeps model loaded for fairness, but memory isn't reclaimed. vLLM has documented memory leaks in V1 engine (GitHub issue #15294).

**How to avoid:**
1. **Default behaviour (warmup fairness):** Model stays loaded, but containers undergo health checks every 5-10 experiments. If GPU memory >90%, restart container.
2. **force_cold_start: true:** Explicit model unload between experiments. Add "unload" command to backend entrypoints that calls `del model; torch.cuda.empty_cache()`.
3. **Periodic restart:** After every cycle (not every experiment), restart containers for full thermal and memory reset. Cycle gaps already provide timing for this.

**Warning signs:**
- Campaign fails after N experiments with CUDA OOM
- GPU memory usage creeps up over time
- NVML health check shows high memory between experiments
- Thermal throttling increases late in campaign

### Pitfall 3: Grid Cartesian Explosion with Invalid Configs
**What goes wrong:** `config generate-grid` creates 1000s of configs, 80% are invalid (wrong backend × param combinations), wasting hours on failed experiments.

**Why it happens:** Current grid generation (config.py lines 492-686) doesn't validate backend compatibility. `--vary pytorch.batch_size=1,2,4 --vary backend=pytorch,vllm,tensorrt` generates invalid vllm/tensorrt configs with pytorch params.

**How to avoid:**
1. **Pre-validation:** After generating grid, validate each config via `ExperimentConfig(**config_dict)` AND backend-param check via `config/introspection.py`.
2. **Auto-filter:** With `--validate` flag (exists, line 502), skip invalid configs. Without flag, warn but generate all.
3. **Smart varies:** When varying `backend`, don't vary backend-specific params. Only vary universal params (decoder.*) or provide per-backend grids.
4. **Summary report:** After grid generation, print count of valid/invalid per backend. Alert user to avoid wasted campaign time.

**Warning signs:**
- Grid generates 1000+ configs
- Many experiments fail immediately with "Invalid parameter"
- Cross-backend param errors in logs
- Campaign progress stalls on validation failures

### Pitfall 4: Manifest State Drift Without Atomic Updates
**What goes wrong:** Campaign crashes mid-update, manifest corrupted (partial JSON), resume fails with parse errors, lost track of completed experiments.

**Why it happens:** Direct writes (`open(path, "w").write(json)`) aren't atomic. Power loss or KeyboardInterrupt during write leaves partial file.

**How to avoid:** Copy StateManager pattern exactly:
1. Write to `campaign_manifest.json.tmp`
2. Atomic rename to `campaign_manifest.json` (POSIX guarantees atomicity)
3. Update manifest after EACH experiment completes (not batched)
4. Never mutate manifest in-memory without persisting

**Warning signs:**
- "JSONDecodeError: Expecting value" on resume
- Manifest exists but claims 0 experiments completed
- Duplicate experiment IDs in manifest
- Lost tracking after Ctrl+C interrupt

### Pitfall 5: Backend Container Startup Race Conditions
**What goes wrong:** `docker.compose.up()` returns before containers are healthy, first exec fails with "container not ready", campaign aborts or retries indefinitely.

**Why it happens:** Containers report "running" before application is ready. PyTorch model loading takes 30-60s, vLLM engine init even longer. Health checks may be optimistic.

**How to avoid:**
1. **Reliable health checks:** Don't use `torch.cuda.is_available()` alone. Load a tiny model (e.g., "gpt2") and run one token generation in health check. Proves full stack operational.
2. **wait=True:** python-on-whales `docker.compose.up(wait=True)` waits for health checks to pass (use health checks in docker-compose.yml, already exist at lines 64-68).
3. **Graceful retry:** First exec after container start should retry 3 times with 10s delay if it fails. Log retries, fail campaign only after exhausting retries.
4. **Pre-warmup:** After containers start, run single warmup experiment (not counted) to verify full stack. Fail fast if warmup fails.

**Warning signs:**
- First experiment fails with "model not found" or "CUDA error"
- Exec commands timeout with no output
- Container logs show model still loading when exec arrives
- Intermittent "connection refused" errors

### Pitfall 6: Mixing Backend Configs in Same Container
**What goes wrong:** Campaign with pytorch + vllm configs tries to reuse single "benchmark" container, leading to dependency conflicts, wrong libraries loaded.

**Why it happens:** Thinking "all experiments use same base image" so one container can handle all backends. But pytorch/vllm/tensorrt have mutually exclusive dependencies (noted in pyproject.toml lines 32, 70-78).

**How to avoid:**
1. **Backend-specific containers:** Campaign manifest assigns backend to container: "pytorch" experiments → "pytorch" container, "vllm" → "vllm" container. Never mix.
2. **Multi-backend campaigns:** Start all needed backend containers at campaign start: `docker.compose.up(["pytorch", "vllm"])`. Route experiments via manifest backend field.
3. **Container naming:** Use service names from docker-compose.yml ("pytorch", "vllm", "tensorrt"), not custom names. Manifest tracks this explicitly.

**Warning signs:**
- "ModuleNotFoundError: No module named 'vllm'" in pytorch container
- Dependency version conflicts mid-campaign
- Backend initialization fails in wrong container
- Campaign tries to run all experiments in first available container

### Pitfall 7: Force-Cold-Start Model Unload Incomplete
**What goes wrong:** `force_cold_start: true` set, but GPU memory remains high between experiments. Models aren't actually unloading, defeating cold-start purpose.

**Why it happens:** Python `del model` doesn't guarantee GPU memory release. PyTorch caches kernels, vLLM keeps engine state, TensorRT doesn't expose unload API cleanly.

**How to avoid:**
1. **Backend-specific unload commands:**
   - PyTorch: `del model; del tokenizer; torch.cuda.empty_cache(); gc.collect()`
   - vLLM: `del llm; import gc; gc.collect()`  (V1 engine requires process restart for full cleanup)
   - TensorRT: Process restart (TRT engines can't be unloaded cleanly)
2. **NVML verification:** After unload command, query GPU memory. If >10% of peak usage remains, log warning and recommend process restart.
3. **Container restart option:** Add `--force-cold-start-restart` that restarts container between experiments (slower but guarantees clean state).
4. **Document limitations:** vLLM V1 engine may need container restart for true cold start. Note in campaign config docs.

**Warning signs:**
- force_cold_start enabled but TTFT doesn't vary much
- GPU memory high between experiments despite unload
- Experiments after first one show "warmup" behaviour
- Cold-start benchmarks unrealistic compared to manual tests

## Code Examples

### Example 1: Long-Running Container Campaign Orchestration
```python
# Source: python-on-whales API + existing CampaignRunner pattern
from python_on_whales import DockerClient
from llenergymeasure.orchestration.campaign import CampaignRunner
from llenergymeasure.config.campaign_config import CampaignConfig

def run_campaign_with_long_running_containers(
    campaign: CampaignConfig,
    manifest_path: Path
) -> None:
    """Execute campaign using long-running containers with exec dispatch."""

    # Initialize Docker SDK
    docker = DockerClient(compose_files=["./docker-compose.yml"])

    # Create campaign runner and manifest
    runner = CampaignRunner(campaign)
    execution_order = runner.generate_execution_order()
    manifest = create_campaign_manifest(campaign, execution_order)

    # Determine which backend containers to start
    backends_needed = {exp.backend for exp in execution_order}
    console.print(f"Starting containers: {', '.join(backends_needed)}")

    # Start containers (detached, wait for health)
    docker.compose.up(
        services=list(backends_needed),
        detach=True,
        wait=True  # Waits for health checks to pass
    )

    try:
        # Execute experiments via exec (no container recreation)
        for idx, experiment in enumerate(execution_order):
            console.print(f"Experiment {idx + 1}/{len(execution_order)}: {experiment.config_name}")

            # Update manifest
            manifest_entry = manifest.experiments[idx]
            manifest_entry.status = "running"
            save_campaign_manifest(manifest, manifest_path)

            # Execute in running container
            result = docker.compose.execute(
                service=experiment.backend,
                command=[
                    "lem", "experiment",
                    f"/app/configs/{experiment.config_path}",
                    "--yes",  # Skip confirmations
                    "--dataset", campaign.dataset or "alpaca",
                ],
                tty=False,  # Capture output
                envs={
                    "EXPERIMENT_ID": experiment.experiment_id,
                    "CAMPAIGN_ID": campaign.campaign_id,
                }
            )

            # Update manifest with result
            if result.returncode == 0:
                manifest_entry.status = "completed"
                manifest_entry.result_path = f"results/raw/{experiment.experiment_id}/"
            else:
                manifest_entry.status = "failed"
                manifest_entry.error = "Non-zero exit code"

            manifest_entry.completed_at = datetime.now()
            save_campaign_manifest(manifest, manifest_path)

            # Thermal gap (if not last experiment)
            if idx < len(execution_order) - 1:
                gap = campaign.execution.config_gap_seconds
                if gap > 0:
                    console.print(f"  [dim]Thermal gap: {gap}s[/dim]")
                    time.sleep(gap)

    finally:
        # Cleanup: stop containers
        console.print("\n[dim]Stopping containers...[/dim]")
        docker.compose.down()
```

### Example 2: Backend-Aware Grid Validation
```python
# Source: Extend config/config.py generate-grid + config/introspection.py SSOT
from llenergymeasure.config.introspection import get_backend_params

def generate_backend_aware_grid(
    base_config: Path,
    variations: dict[str, list[Any]],
    output_dir: Path
) -> tuple[list[Path], list[Path]]:
    """Generate grid with backend-param validation.

    Returns:
        (valid_configs, invalid_configs)
    """
    base = load_config(base_config)

    # Generate cartesian product (existing logic)
    combinations = list(itertools.product(*variations.values()))

    valid_configs: list[Path] = []
    invalid_configs: list[Path] = []

    for combo in combinations:
        config_dict = base.model_dump()

        # Apply variations
        for param, value in zip(variations.keys(), combo):
            set_nested_param(config_dict, param, value)

        # Validate Pydantic schema
        try:
            config = ExperimentConfig(**config_dict)
        except ValidationError as e:
            invalid_configs.append((config_dict["config_name"], str(e)))
            continue

        # Backend-aware validation (NEW)
        backend = config.backend
        backend_params = get_backend_params(backend)

        # Check for invalid cross-backend params
        invalid_params = []
        for other_backend in ["pytorch", "vllm", "tensorrt"]:
            if other_backend != backend and other_backend in config_dict:
                # Config has params for different backend
                invalid_params.append(f"Section '{other_backend}' invalid for backend '{backend}'")

        if invalid_params:
            invalid_configs.append((config_dict["config_name"], " | ".join(invalid_params)))
            continue

        # Write valid config
        output_path = output_dir / f"{config_dict['config_name']}.yaml"
        write_config_yaml(output_path, config_dict)
        valid_configs.append(output_path)

    return valid_configs, invalid_configs
```

### Example 3: Health Check with NVML Memory Monitoring
```python
# Source: Phase 1 NVML patterns + python-on-whales exec
import pynvml
from python_on_whales import DockerClient

def check_container_gpu_health(
    docker: DockerClient,
    service: str,
    memory_threshold_gb: float = 70.0
) -> tuple[bool, str]:
    """Check container GPU health via NVML.

    Returns:
        (is_healthy, reason)
    """
    try:
        # Execute NVML query inside container
        result = docker.compose.execute(
            service=service,
            command=[
                "python", "-c",
                "import pynvml; pynvml.nvmlInit(); h = pynvml.nvmlDeviceGetHandleByIndex(0); "
                "mem = pynvml.nvmlDeviceGetMemoryInfo(h); "
                "print(f'{mem.used},{mem.total}')"
            ],
            tty=False
        )

        # Parse result
        used_str, total_str = result.strip().split(",")
        used_gb = int(used_str) / 1e9
        total_gb = int(total_str) / 1e9
        usage_pct = (used_gb / total_gb) * 100

        # Check threshold
        if used_gb > memory_threshold_gb:
            return False, f"GPU memory high: {used_gb:.1f}GB / {total_gb:.1f}GB ({usage_pct:.0f}%)"

        return True, f"OK (GPU memory: {used_gb:.1f}GB / {total_gb:.1f}GB)"

    except Exception as e:
        return False, f"Health check failed: {e}"

# Usage in campaign monitoring
def monitor_campaign_health(
    docker: DockerClient,
    services: list[str],
    check_interval_seconds: int = 300  # Every 5 min
):
    """Monitor backend containers during campaign, restart on failure."""
    while True:
        for service in services:
            healthy, reason = check_container_gpu_health(docker, service)

            if not healthy:
                console.print(f"[yellow]Unhealthy: {service} - {reason}[/yellow]")
                console.print(f"[yellow]Restarting {service}...[/yellow]")

                docker.compose.restart([service])
                time.sleep(10)  # Wait for restart

                # Verify recovery
                healthy_after, reason_after = check_container_gpu_health(docker, service)
                if healthy_after:
                    console.print(f"[green]Recovered: {service}[/green]")
                else:
                    console.print(f"[red]Failed to recover: {service}[/red]")
                    # Don't abort campaign, log and continue

        time.sleep(check_interval_seconds)
```

### Example 4: Bootstrap Confidence Intervals for Campaign Results
```python
# Source: FAQ paper (arXiv:2601.20251) bootstrap recommendations
import numpy as np
from llenergymeasure.domain.experiment import AggregatedResult

def compute_bootstrap_ci(
    samples: list[float],
    n_iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 42
) -> dict[str, float]:
    """Compute bootstrap 95% CI using percentile method.

    Args:
        samples: Measurements from multiple cycles (≥3 recommended).
        n_iterations: Bootstrap iterations (1000 standard).
        confidence: Confidence level (0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Dict with mean, std, ci_lower, ci_upper.

    Source: FAQ paper "Efficient Evaluation of LLMs" (Jan 2026)
    """
    if len(samples) < 3:
        # Not enough data for reliable CI
        return {
            "mean": np.mean(samples),
            "std": np.std(samples),
            "ci_lower": None,
            "ci_upper": None,
            "warning": "< 3 samples, CI unreliable"
        }

    samples_array = np.array(samples)
    n = len(samples_array)

    # Bootstrap resampling
    bootstrap_means = []
    rng = np.random.default_rng(seed)

    for _ in range(n_iterations):
        resampled = rng.choice(samples_array, size=n, replace=True)
        bootstrap_means.append(np.mean(resampled))

    # Percentile method (frequentist coverage guarantees)
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    return {
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples)),
        "ci_lower": float(np.percentile(bootstrap_means, lower_percentile)),
        "ci_upper": float(np.percentile(bootstrap_means, upper_percentile)),
        "n_samples": n,
        "confidence": confidence,
    }

# Usage in campaign aggregation
def aggregate_campaign_with_cis(
    manifest: CampaignManifest
) -> dict[str, dict[str, Any]]:
    """Aggregate campaign results by config with confidence intervals."""
    results_by_config: dict[str, list[AggregatedResult]] = {}

    # Load all completed results
    for exp in manifest.experiments:
        if exp.status == "completed" and exp.result_path:
            result = load_aggregated_result(Path(exp.result_path))
            if exp.config_name not in results_by_config:
                results_by_config[exp.config_name] = []
            results_by_config[exp.config_name].append(result)

    # Compute statistics per config
    aggregated = {}
    for config_name, results in results_by_config.items():
        # Extract metrics
        energy_samples = [r.energy_consumption_mwh for r in results]
        ttft_samples = [r.ttft_mean_s for r in results]
        throughput_samples = [r.throughput_tokens_per_sec for r in results]

        # Bootstrap CIs
        aggregated[config_name] = {
            "n_cycles": len(results),
            "energy_mwh": compute_bootstrap_ci(energy_samples),
            "ttft_s": compute_bootstrap_ci(ttft_samples),
            "throughput_tps": compute_bootstrap_ci(throughput_samples),
        }

    return aggregated

# Example output format
# {
#     "pytorch_batch1": {
#         "n_cycles": 5,
#         "energy_mwh": {
#             "mean": 12.34,
#             "std": 0.89,
#             "ci_lower": 11.21,
#             "ci_upper": 13.47,
#             "n_samples": 5,
#             "confidence": 0.95
#         },
#         ...
#     }
# }
```

## Open Questions

Things that couldn't be fully resolved and need validation during implementation:

1. **python-on-whales exec output streaming**
   - What we know: `execute()` method has `tty=False` to get output, `stream=True` for real-time streaming
   - What's unclear: Whether streaming works reliably with Rich progress bars in parent process. Need to test if exec stdout interferes with orchestrator's console output.
   - Recommendation: Start with tty=False (capture output), add streaming in later iteration if live logs needed.

2. **vLLM memory leak severity on long campaigns**
   - What we know: GitHub issue #15294 documents 200GB RAM leaks in vLLM V1 engine. Fixed in V2 engine (not yet stable).
   - What's unclear: Whether health check restarts are sufficient or if `force_cold_start` should default to True for vLLM.
   - Recommendation: Default to `force_cold_start: false` (fairness), but add health check every 10 experiments for vLLM specifically. Document vLLM leak risk in campaign config docs.

3. **TensorRT engine unload between experiments**
   - What we know: TensorRT-LLM doesn't expose clean unload API. Engine compilation is expensive (2-10 minutes).
   - What's unclear: Whether `force_cold_start: true` should trigger engine recompilation or just process restart with cached engine.
   - Recommendation: `force_cold_start: true` for TensorRT does **process restart** (clears memory) but reuses cached compiled engines (avoids recompilation). Document this distinction clearly.

4. **Health check frequency tuning**
   - What we know: Containers can degrade over time, but health checks add overhead.
   - What's unclear: Optimal check frequency. Every experiment (too frequent)? Every cycle (may miss degradation)? Time-based (5 min)?
   - Recommendation: Check after every cycle completion (when thermal gaps already provide time). Add `--health-check-interval` flag for power users.

5. **Manifest resume conflict resolution**
   - What we know: Resume should skip completed, retry failed on user request.
   - What's unclear: What if config file changed between campaign start and resume? Should manifest store full config or just path?
   - Recommendation: Store **config hash** in manifest (like StateManager does). On resume, check if config file hash matches. If mismatch, warn user and ask if they want to continue or abort.

## Sources

### Primary (HIGH confidence)

**Technology stack:**
- [python-on-whales GitHub](https://github.com/gabrieldemarmiesse/python-on-whales) - Docker SDK, 696 stars, MIT license
- [python-on-whales compose API](https://gabrieldemarmiesse.github.io/python-on-whales/sub-commands/compose/) - Execute command documentation
- [Docker Blog: Python-on-whales](https://www.docker.com/blog/guest-post-calling-the-docker-cli-from-python-with-python-on-whales/) - Official Docker endorsement
- [Docker SDK for Python docs](https://docker-py.readthedocs.io/en/stable/containers.html) - Alternative (not chosen due to weaker Compose support)
- [FAQ: Efficient LLM Evaluation (Jan 2026)](https://arxiv.org/abs/2601.20251) - Bootstrap CI methodology, statistical guarantees

**Existing codebase analysis:**
- `cli/campaign.py` lines 1-655 - Current campaign command, Docker dispatch, backend detection
- `orchestration/campaign.py` lines 1-361 - CampaignRunner, execution order, warmup logic
- `config/campaign_config.py` lines 1-217 - CampaignConfig Pydantic models
- `state/experiment_state.py` - StateManager atomic write pattern
- `config/introspection.py` lines 1-807 - SSOT parameter metadata extraction
- `config/config.py` lines 492-686 - generate-grid command implementation
- `docker-compose.yml` lines 1-250 - Current container definitions, health checks

### Secondary (MEDIUM confidence)
- [vLLM memory leak issue #15294](https://github.com/vllm-project/vllm/issues/15294) - 200GB RAM leaks in V1 engine
- [Docker Compose health checks guide](https://lumigo.io/container-monitoring/docker-health-check-a-practical-guide/) - Health check best practices
- [numpy.percentile documentation](https://numpy.org/doc/stable/reference/generated/numpy.percentile.html) - Bootstrap CI implementation

### Tertiary (LOW confidence, needs validation)
- Optimal health check frequency - Domain inference, no research found. Recommend after-cycle checks.
- force_cold_start TensorRT behaviour - Engineering decision, not documented. Recommend process restart + cached engines.
- Config hash for manifest resume - Best practice from StateManager, not campaign-specific guidance.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - python-on-whales officially endorsed by Docker, numpy well-validated for bootstrap
- Codebase patterns: HIGH - Direct analysis of existing code, patterns verified in Phase 1
- Architecture: HIGH - Long-running container pattern standard in Docker orchestration, manifest persistence mirrors StateManager
- Pitfalls: HIGH - Container state issues documented in vLLM GitHub, grid explosion observed in testing, atomic writes POSIX-guaranteed

**Research date:** 2026-01-29
**Valid until:** 60 days (April 2026) - python-on-whales stable API, codebase patterns unlikely to change during Phase 2 implementation
