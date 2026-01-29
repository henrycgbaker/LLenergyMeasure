# Architecture Patterns: Host-Container Orchestration

**Domain:** Python orchestrator coordinating LLM experiments across Docker containers
**Researched:** 2026-01-29
**Confidence:** HIGH

## Recommended Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Host: Python Campaign Orchestrator                             │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  CampaignOrchestrator                                     │  │
│  │  - Campaign manifest (experiment queue)                   │  │
│  │  - Container lifecycle manager                            │  │
│  │  - Health monitor daemon                                  │  │
│  │  - Time-series power sampler                              │  │
│  │  - Results aggregator                                     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                      │
│                    docker-py SDK                                 │
│                           │                                      │
└───────────────────────────┼──────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌────────────────┐  ┌────────────────┐  ┌────────────────┐
│  pytorch       │  │  vllm          │  │  tensorrt      │
│  container     │  │  container     │  │  container     │
│  (long-running)│  │  (long-running)│  │  (long-running)│
│                │  │                │  │                │
│  Receives:     │  │  Receives:     │  │  Receives:     │
│  docker exec   │  │  docker exec   │  │  docker exec   │
│  lem exp ...   │  │  lem exp ...   │  │  lem exp ...   │
└────────────────┘  └────────────────┘  └────────────────┘
        │                   │                   │
        └───────────────────┴───────────────────┘
                            │
                    Shared Volumes
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
   /app/results       /app/configs      /app/.state
   (bind mount)       (bind mount)    (named volume)
```

### Component Boundaries

| Component | Responsibility | Communicates With |
|-----------|---------------|-------------------|
| **CampaignOrchestrator** | Campaign manifest execution, container dispatch, cycle coordination | ContainerManager, HealthMonitor, PowerSampler, ResultsAggregator |
| **ContainerManager** | Container lifecycle (start/stop/health), exec dispatch | Docker daemon via docker-py SDK |
| **HealthMonitor** | Periodic health checks, auto-recovery on failure | ContainerManager |
| **PowerSampler** | Time-series GPU power sampling during experiments | NVML via container exec, TimescaleDB/InfluxDB (optional) |
| **CampaignManifest** | Experiment queue with backend routing, cycle tracking | StateManager (persistent manifest) |
| **BackendRouter** | Route experiments to appropriate container based on backend | CampaignManifest, ContainerManager |
| **StateManager** | Persistent experiment/campaign state, atomic writes | Filesystem (shared volumes) |
| **ResultsAggregator** | Multi-cycle aggregation, statistical analysis | Filesystem (results volume) |

### Data Flow

```
1. Campaign Definition
   User → YAML campaign config → CampaignManifest

2. Container Startup (one-time)
   CampaignOrchestrator → docker compose up -d [pytorch|vllm|tensorrt]

3. Experiment Dispatch
   CampaignManifest → BackendRouter → ContainerManager.exec_run(container, cmd)

4. Power Sampling (parallel)
   PowerSampler → docker exec pytorch nvidia-smi --query-gpu=power.draw

5. Health Monitoring (parallel)
   HealthMonitor → docker exec pytorch python -c "import torch; assert torch.cuda.is_available()"

6. Result Collection
   Container → Shared volume (/app/results/exp_id/process_N.json)
   ResultsAggregator → Load from shared volume → Aggregate

7. Campaign Completion
   All experiments done → Multi-cycle aggregation → Final report
```

## Patterns to Follow

### Pattern 1: Long-Running Containers with Exec Dispatch

**What:** Start containers once at campaign start, use `docker exec` for each experiment, stop containers at campaign end.

**When:** Campaign runs multiple experiments (grid search, multi-cycle), container startup overhead is significant (10-30s for GPU allocation, model cache warming).

**Why better than run-per-experiment:**
- Avoids container recreation overhead (startup time, GPU re-initialization)
- Keeps GPU memory allocated across experiments (vLLM KV cache benefits)
- Simplifies health monitoring (one container per backend vs N ephemeral containers)
- Enables warmup convergence detection (container stays warm between experiments)

**Example:**
```python
from docker import DockerClient
from docker.models.containers import Container

class ContainerManager:
    def __init__(self):
        self.client = DockerClient.from_env()
        self.containers: dict[str, Container] = {}

    def start_backend(self, backend: str) -> Container:
        """Start long-running container for backend."""
        container = self.client.containers.run(
            image=f"llenergymeasure:{backend}",
            name=f"lem-{backend}",
            detach=True,
            device_requests=[
                {"driver": "nvidia", "count": -1, "capabilities": [["gpu", "utility"]]}
            ],
            volumes={
                "./results": {"bind": "/app/results", "mode": "rw"},
                "./configs": {"bind": "/app/configs", "mode": "ro"},
            },
            restart_policy={"Name": "unless-stopped"},
        )
        self.containers[backend] = container
        return container

    def dispatch_experiment(self, backend: str, config_path: str) -> ExecResult:
        """Execute experiment in running container."""
        container = self.containers[backend]
        cmd = f"lem experiment /app/configs/{config_path}"
        exit_code, output = container.exec_run(
            cmd,
            stream=False,
            demux=True,  # Separate stdout/stderr
        )
        return ExecResult(exit_code, output)

    def stop_all(self):
        """Stop all containers at campaign end."""
        for container in self.containers.values():
            container.stop(timeout=30)
            container.remove()
```

**Confidence:** HIGH - [Docker SDK documentation](https://docker-py.readthedocs.io/en/stable/containers.html) verified, [run vs exec patterns](https://medium.com/analytics-vidhya/how-to-understand-the-difference-between-docker-composes-up-vs-run-vs-exec-commands-a506151967df) confirmed.

### Pattern 2: Health Check Daemon with Auto-Recovery

**What:** Background thread polling container health at intervals, auto-restart on failure.

**When:** Long-running containers executing untrusted experiments, risk of container crash mid-campaign.

**Example:**
```python
import threading
from time import sleep
from typing import Callable

class HealthMonitor:
    def __init__(
        self,
        container_manager: ContainerManager,
        check_interval: int = 30,
        on_unhealthy: Callable[[str], None] | None = None,
    ):
        self.container_manager = container_manager
        self.check_interval = check_interval
        self.on_unhealthy = on_unhealthy
        self._daemon_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self):
        """Start health monitoring daemon."""
        self._daemon_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._daemon_thread.start()

    def stop(self):
        """Stop health monitoring daemon."""
        self._stop_event.set()
        if self._daemon_thread:
            self._daemon_thread.join(timeout=5)

    def _monitor_loop(self):
        """Periodic health check loop."""
        while not self._stop_event.is_set():
            for backend, container in self.container_manager.containers.items():
                if not self._check_health(container):
                    self._handle_unhealthy(backend, container)
            sleep(self.check_interval)

    def _check_health(self, container: Container) -> bool:
        """Execute health check command."""
        try:
            exit_code, _ = container.exec_run(
                'python -c "import torch; assert torch.cuda.is_available()"',
                timeout=10,
            )
            return exit_code == 0
        except Exception:
            return False

    def _handle_unhealthy(self, backend: str, container: Container):
        """Handle unhealthy container - restart and notify."""
        container.restart(timeout=30)
        if self.on_unhealthy:
            self.on_unhealthy(backend)
```

**Docker Compose Integration:**
```yaml
services:
  pytorch:
    healthcheck:
      test: ["CMD", "python", "-c", "import torch; assert torch.cuda.is_available()"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s  # Grace period for startup
```

**Confidence:** HIGH - [Docker health checks](https://lumigo.io/container-monitoring/docker-health-check-a-practical-guide/) verified, [restart policies](https://www.cloudbees.com/blog/ensuring-containers-are-always-running-dockers-restart-policy) confirmed.

### Pattern 3: Time-Series Power Sampling

**What:** Background thread sampling GPU power draw at high frequency (1-10Hz) during experiment execution.

**When:** Need sub-second power resolution for energy efficiency analysis, CodeCarbon's interval sampling (default 15s) is too coarse.

**Architecture Options:**

#### Option A: Host-Side Sampling (Recommended)
```python
import threading
from time import sleep, time
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class PowerSample:
    timestamp: float
    gpu_id: int
    power_draw_watts: float

class PowerSampler:
    def __init__(self, sample_rate_hz: float = 10.0):
        self.sample_rate = sample_rate_hz
        self.interval = 1.0 / sample_rate_hz
        self._samples: list[PowerSample] = []
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self, experiment_id: str):
        """Start power sampling for experiment."""
        self._samples.clear()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._sample_loop,
            args=(experiment_id,),
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> list[PowerSample]:
        """Stop sampling and return collected samples."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
        return self._samples.copy()

    def _sample_loop(self, experiment_id: str):
        """Sampling loop - uses nvidia-smi on host."""
        import pynvml
        pynvml.nvmlInit()

        while not self._stop_event.is_set():
            timestamp = time()
            device_count = pynvml.nvmlDeviceGetCount()

            for gpu_id in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                self._samples.append(
                    PowerSample(
                        timestamp=timestamp,
                        gpu_id=gpu_id,
                        power_draw_watts=power_mw / 1000.0,
                    )
                )

            sleep(self.interval)

        pynvml.nvmlShutdown()

    def save_timeseries(self, experiment_id: str, output_path: Path):
        """Save power timeseries to JSON."""
        data = {
            "experiment_id": experiment_id,
            "sample_rate_hz": self.sample_rate,
            "samples": [
                {
                    "timestamp": s.timestamp,
                    "gpu_id": s.gpu_id,
                    "power_watts": s.power_draw_watts,
                }
                for s in self._samples
            ],
        }
        output_path.write_text(json.dumps(data, indent=2))
```

**Why host-side:** Direct NVML access (no exec overhead), accurate timestamps, independent of container lifecycle.

#### Option B: Container-Side Sampling
- Execute `nvidia-smi` via `docker exec` at intervals
- Higher latency (~50-100ms per sample vs <1ms NVML)
- Use for validation/debugging only

**Confidence:** MEDIUM - Host-side NVML pattern is standard practice, but time-series architecture for ML experiments is domain-specific. [Time-series databases for energy monitoring](https://www.mdpi.com/1996-1073/17/21/5478) provides architectural guidance.

### Pattern 4: Campaign Manifest with State Persistence

**What:** Queue of experiments with backend routing, persistent across restarts.

**When:** Running grids of experiments, multi-cycle campaigns, need resumption capability.

**Data Model:**
```python
from pydantic import BaseModel, Field
from enum import Enum

class CampaignStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class ExperimentManifestEntry(BaseModel):
    experiment_id: str
    backend: str  # "pytorch" | "vllm" | "tensorrt"
    config_path: str
    cycle_id: int
    status: CampaignStatus = CampaignStatus.PENDING
    container_name: str | None = None
    started_at: float | None = None
    completed_at: float | None = None
    error_message: str | None = None

class CampaignManifest(BaseModel):
    campaign_id: str
    experiments: list[ExperimentManifestEntry]
    total_cycles: int = Field(default=1, description="Number of repeat cycles")
    backend_containers: dict[str, str] = Field(
        default_factory=dict,
        description="Map of backend -> container_name",
    )

    def next_pending(self) -> ExperimentManifestEntry | None:
        """Get next pending experiment."""
        for exp in self.experiments:
            if exp.status == CampaignStatus.PENDING:
                return exp
        return None

    def backend_for_experiment(self, experiment_id: str) -> str:
        """Get backend name for experiment."""
        for exp in self.experiments:
            if exp.experiment_id == experiment_id:
                return exp.backend
        raise ValueError(f"Unknown experiment: {experiment_id}")
```

**Persistence with Atomic Writes:**
```python
from pathlib import Path
import json

class CampaignStateManager:
    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def save_manifest(self, manifest: CampaignManifest):
        """Save manifest atomically."""
        path = self.state_dir / f"{manifest.campaign_id}.json"
        temp_path = path.with_suffix(".tmp")

        temp_path.write_text(manifest.model_dump_json(indent=2))
        temp_path.rename(path)  # Atomic on POSIX

    def load_manifest(self, campaign_id: str) -> CampaignManifest | None:
        """Load manifest from disk."""
        path = self.state_dir / f"{campaign_id}.json"
        if not path.exists():
            return None
        return CampaignManifest.model_validate_json(path.read_text())
```

**Confidence:** HIGH - Extends existing `StateManager` pattern, atomic file operations verified via [python-atomicwrites](https://python-atomicwrites.readthedocs.io/).

### Pattern 5: Warmup Convergence Detection

**What:** Detect when model warmup has converged by monitoring per-batch latency stability.

**When:** Accurate throughput measurements require stable inference performance, initial batches have higher latency due to CUDA kernel compilation, GPU frequency scaling.

**Implementation:**
```python
from collections import deque
import numpy as np
from dataclasses import dataclass

@dataclass
class WarmupDetector:
    window_size: int = 10
    convergence_threshold: float = 0.05  # 5% coefficient of variation

    def __post_init__(self):
        self._latencies = deque(maxlen=self.window_size)

    def add_batch(self, latency_ms: float) -> bool:
        """Add batch latency, return True if converged."""
        self._latencies.append(latency_ms)

        if len(self._latencies) < self.window_size:
            return False  # Need full window

        # Check coefficient of variation
        mean = np.mean(self._latencies)
        std = np.std(self._latencies)
        cv = std / mean if mean > 0 else float('inf')

        return cv < self.convergence_threshold

    def is_converged(self) -> bool:
        """Check if currently in converged state."""
        if len(self._latencies) < self.window_size:
            return False

        mean = np.mean(self._latencies)
        std = np.std(self._latencies)
        cv = std / mean if mean > 0 else float('inf')
        return cv < self.convergence_threshold

# Integration with inference loop
def run_with_warmup_detection(model, prompts):
    detector = WarmupDetector()
    warmup_complete = False

    for i, prompt in enumerate(prompts):
        start = time.time()
        output = model.generate(prompt)
        latency_ms = (time.time() - start) * 1000

        if not warmup_complete:
            if detector.add_batch(latency_ms):
                warmup_complete = True
                logger.info(f"Warmup converged at batch {i}")

        # Collect metrics only after warmup
        if warmup_complete:
            metrics.record_batch(latency_ms, len(output))
```

**Alternative: Gaussian Process Smoothing** (from [Auto-WU research](https://arxiv.org/abs/2509.07972)):
- More sophisticated but higher overhead
- Use for research validation, not production

**Confidence:** MEDIUM - Coefficient of variation method is simple and robust. Recent research (2025) shows [automated warmup detection using GP smoothing](https://www.emergentmind.com/topics/model-warmup-techniques), but simpler statistical methods are more practical for production.

## Anti-Patterns to Avoid

### Anti-Pattern 1: `docker compose run` Per-Experiment

**What:** Running `docker compose run --rm pytorch lem experiment config.yaml` for each experiment in a grid.

**Why bad:**
- Container creation overhead: 10-30s per experiment (GPU allocation, CUDA initialization)
- GPU memory fragmentation: Each new container allocates fresh GPU memory
- vLLM memory leaks: Known issue where containers don't release GPU memory on exit ([Issue #7581](https://github.com/vllm-project/vllm/issues/7581))
- Health check complexity: N ephemeral containers vs 3 long-running containers

**Instead:** Long-running containers with `docker exec` dispatch (Pattern 1).

**Evidence:** vLLM [memory leak reports](https://github.com/vllm-project/vllm/issues/15294) show critical issues with container lifecycle management in V1 engine (200+ GB RAM leaks). Long-running containers mitigate this by keeping processes alive.

### Anti-Pattern 2: Shared Volume Write Conflicts

**What:** Multiple containers writing to same file without coordination.

**Why bad:**
- Race conditions: Two containers write `experiment_state.json` simultaneously
- Partial writes: File contains incomplete JSON from interrupted write
- Lost updates: Last writer wins, earlier results lost

**Instead:** Atomic writes with process-specific files:
```python
# Each process writes its own file
results/exp_id/process_0.json
results/exp_id/process_1.json

# Manifest ensures only one container writes at a time
# OR use atomic rename pattern (write-to-temp-then-rename)
```

**Evidence:** Docker volumes [support concurrent reads but not atomic writes](https://www.baeldung.com/ops/docker-share-volume-multiple-containers). Application-level coordination required.

### Anti-Pattern 3: No Crash Recovery for Mid-Campaign Failures

**What:** Campaign orchestrator crashes, loses all progress, must restart from beginning.

**Why bad:**
- Hours of wasted computation for large grids
- Non-deterministic results if re-running partially completed campaigns
- GPU hours wasted on redundant work

**Instead:** Persistent campaign manifest (Pattern 4) with experiment-level status tracking. On crash/restart:
1. Load manifest from disk
2. Skip experiments marked `COMPLETED`
3. Resume from next `PENDING` experiment

**Evidence:** This extends the existing `StateManager` pattern already proven in codebase.

### Anti-Pattern 4: Polling Container State via `docker ps`

**What:** Checking if container is healthy by parsing `docker ps` output.

**Why bad:**
- Racy: Container can crash between poll and check
- Coarse-grained: Only knows "running" vs "stopped"
- No application-level health: Container running but CUDA unavailable

**Instead:** Docker health checks (Pattern 2) with application-aware validation:
```yaml
healthcheck:
  test: ["CMD", "python", "-c", "import torch; assert torch.cuda.is_available()"]
```

**Evidence:** [Docker health check guide](https://last9.io/blog/docker-compose-health-checks/) recommends application-level checks, not process checks.

### Anti-Pattern 5: Synchronous Power Sampling in Inference Loop

**What:** Querying `nvidia-smi` or NVML inside the inference loop.

**Why bad:**
- Inference latency overhead: 10-50ms per query
- Inaccurate measurements: Query itself consumes GPU cycles
- Thread safety: NVML not guaranteed thread-safe in all configurations

**Instead:** Separate power sampling thread (Pattern 3) running in parallel with inference.

## Scalability Considerations

| Concern | Single Backend | 3 Backends (Current) | 10+ Backends (Future) |
|---------|---------------|---------------------|----------------------|
| **Container startup** | 1 container, 10-30s | 3 containers, 30-90s total | Pool pattern: pre-start N containers |
| **GPU memory** | Full GPU for 1 backend | 3 backends share GPUs (MIG or exclusive) | MIG required, 7 slices per A100 |
| **Campaign duration** | Minutes to hours | Hours to days | Days to weeks |
| **State complexity** | Single experiment state | Campaign manifest with 3 backends | Distributed queue (Redis/Celery) |
| **Failure recovery** | Restart experiment | Restart failed experiments | Checkpoint every N experiments |
| **Power monitoring** | Single GPU, 10Hz sampling | 3 GPUs, 10Hz sampling (30 samples/s) | InfluxDB/TimescaleDB for time-series storage |

### When to Migrate to Distributed Architecture

**Threshold:** >10 concurrent backends OR >1 week campaign duration OR multi-node GPU cluster.

**Migration path:**
1. **Current (single-node):** Host orchestrator + Docker containers + shared volumes
2. **Intermediate (multi-node):** Celery task queue + Redis + NFS shared volumes
3. **Production (enterprise):** Kubernetes + KubeFlow + distributed storage (Ceph/GlusterFS)

**Evidence:** [Container orchestration platforms 2026](https://www.portainer.io/blog/container-orchestration-platforms) shows Docker Compose remains viable for single-node multi-container workloads. Kubernetes overhead not justified until multi-node or >100 containers.

## Build Order Implications

### Phase Dependencies

```
Phase 1: Container Lifecycle Management
├── ContainerManager (start/stop/exec)
├── Docker SDK integration
└── Health check daemon

Phase 2: Campaign Manifest
├── CampaignManifest data model
├── CampaignStateManager (persistence)
└── Backend routing logic

Phase 3: Time-Series Power Sampling
├── PowerSampler (host-side NVML)
├── Time-series storage format
└── Integration with experiment lifecycle

Phase 4: Warmup Convergence
├── WarmupDetector (latency monitoring)
├── Integration with inference loop
└── Warmup-aware metrics collection

Phase 5: Multi-Cycle Aggregation
├── Cycle-aware results schema
├── Statistical aggregation (mean, std, CI)
└── Campaign summary reports
```

**Critical Path:** Phase 1 → Phase 2 (orchestration core) → Phase 3, 4, 5 (parallel).

**Rationale:**
- Phase 1 is prerequisite for all container-based work
- Phase 2 enables campaign execution (without power/warmup initially)
- Phases 3, 4, 5 enhance existing campaign execution independently

## Sources

**HIGH Confidence (Official Documentation):**
- [Docker SDK for Python](https://docker-py.readthedocs.io/en/stable/containers.html) - Container lifecycle methods
- [Docker Compose: exec vs run](https://medium.com/analytics-vidhya/how-to-understand-the-difference-between-docker-composes-up-vs-run-vs-exec-commands-a506151967df) - Pattern differences
- [Docker health checks](https://lumigo.io/container-monitoring/docker-health-check-a-practical-guide/) - Health monitoring patterns
- [Docker restart policies](https://www.cloudbees.com/blog/ensuring-containers-are-always-running-dockers-restart-policy) - Crash recovery
- [Python atomicwrites](https://python-atomicwrites.readthedocs.io/) - Atomic file operations

**MEDIUM Confidence (Community/Research):**
- [vLLM memory leaks](https://github.com/vllm-project/vllm/issues/15294) - Known issue in V1 engine
- [Docker shared volumes](https://www.baeldung.com/ops/docker-share-volume-multiple-containers) - Concurrent access patterns
- [Time-series databases for energy monitoring](https://www.mdpi.com/1996-1073/17/21/5478) - Architecture patterns
- [Warmup convergence detection](https://www.emergentmind.com/topics/model-warmup-techniques) - Recent research (2025-2026)
- [APScheduler](https://github.com/agronholm/apscheduler) - Python scheduling library

**LOW Confidence (Needs Validation):**
- Multi-node volume sharing with NFS/GlusterFS - Standard pattern but needs performance testing
- InfluxDB/TimescaleDB for power time-series - Appropriate choice but adds dependency complexity
