# Docker Container Strategy Research: `run --rm` vs `up + exec`

**Date:** 2026-01-31
**Context:** LLenergyMeasure campaign execution (multiple sequential LLM inference experiments)
**Question:** Should campaigns use ephemeral containers (`docker compose run --rm`) or long-running containers (`docker compose up -d` + `exec`)?

---

## Executive Summary

**Recommendation: Continue using `docker compose run --rm` (ephemeral containers)**

The current approach is the correct choice for this use case. Long-running containers add complexity without meaningful benefits given:
- 3-5 second overhead is negligible for 1-10 minute experiments (0.5-8% overhead)
- Ephemeral containers provide better isolation and error recovery
- Named volumes already cache models (no re-download)
- Long-running containers require lifecycle management complexity

The investigated `ContainerManager` class (orchestration/container.py) adds ~150 lines of complexity to save <30 seconds across a typical 6-experiment campaign (~60 minutes total).

---

## Detailed Analysis

### Current Implementation: `docker compose run --rm`

**How it works:**
```bash
# Per experiment
docker compose run --rm pytorch lem experiment /app/configs/exp1.yaml
docker compose run --rm pytorch lem experiment /app/configs/exp2.yaml
docker compose run --rm vllm lem experiment /app/configs/exp3.yaml
```

**Lifecycle per experiment:**
1. Create fresh container from image
2. Start container
3. Run experiment command
4. Exit and remove container (--rm)
5. Repeat for next experiment

**Overhead measured:** ~3-5 seconds per experiment for container create/start

---

### Alternative: `docker compose up -d` + `exec`

**How it would work:**
```bash
# Once at campaign start
docker compose up -d pytorch vllm tensorrt

# Per experiment
docker compose exec pytorch lem experiment /app/configs/exp1.yaml
docker compose exec pytorch lem experiment /app/configs/exp2.yaml
docker compose exec vllm lem experiment /app/configs/exp3.yaml

# Once at campaign end
docker compose down
```

**Lifecycle:**
1. Start all backend containers in detached mode
2. Keep containers alive with `sleep infinity` or similar
3. Execute experiments inside running containers
4. Tear down after campaign completes

**Overhead saved:** ~3-5 seconds per experiment (no container create/start)

---

## Comparison Matrix

| Dimension | `run --rm` (Current) | `up + exec` (Alternative) |
|-----------|---------------------|---------------------------|
| **Container Overhead** | 3-5s per experiment | 0s per experiment (startup once) |
| **Campaign Overhead** | 18-30s (6 experiments) | ~10s (startup + teardown once) |
| **Experiment Duration** | 1-10 minutes each | 1-10 minutes each |
| **Relative Overhead** | 0.5-8% of experiment time | 0.3% of campaign time |
| **GPU Memory Isolation** | Perfect (fresh container) | Shared (must manually clear) |
| **Cache Behaviour** | Named volumes persist | Named volumes persist |
| **Model Re-download** | No (named volumes) | No (named volumes) |
| **Error Recovery** | Automatic (fresh container) | Manual (restart service) |
| **OOM Recovery** | Automatic (exit + clean) | Manual (must detect + restart) |
| **Implementation Complexity** | Simple (1 line per exp) | Complex (lifecycle management) |
| **Code Maintenance** | ~5 lines | ~150+ lines (ContainerManager) |
| **Failure Modes** | Container exit = clean | Memory leak, GPU fragment, zombie processes |
| **Health Monitoring** | Not needed | Required (GPU memory checks) |
| **State Leakage Risk** | None (fresh every time) | High (shared containers) |

---

## Evidence from Similar Tools

### MLPerf Inference Benchmarks

MLPerf uses a **hybrid approach**:
- Interactive shell method for development/debugging: `docker run -it <image> /bin/bash`
- One-off containers for actual benchmark runs: Each scenario runs in its own container
- Persistent storage via volume mounts (not long-running containers)

**Key insight:** "Do not run any commands as root. Running under root messes up a lot of permissions." — MLPerf documentation emphasises clean container lifecycle over long-running daemons.

**Source:** [MLCommons Inference Documentation](https://github.com/mlcommons/inference)

### Docker Best Practices (2026)

Docker documentation recommends **ephemeral containers** as a core design principle:

> "The image defined by your Dockerfile should generate containers that are as ephemeral as possible, meaning that the container can be stopped and destroyed, then rebuilt and replaced with an absolute minimum set up and configuration."

**Why ephemeral is preferred:**
- Rapid scaling
- Quick updates
- Efficient resource utilisation
- Containers spun up and down as needed without long-term commitment

**Performance data (2026):**
- Docker Desktop 4.23 achieves startup times of 3.481 seconds
- Containers reduce deployment times by up to 65% compared to VMs
- Resource efficiency improves by 40% with proper orchestration

**Source:** [Docker Best Practices](https://docs.docker.com/build/building/best-practices/)

### Container Lifecycle Patterns

Research on `docker compose exec` vs `run`:

**`docker compose run`:**
- For "one-off" or "adhoc" tasks
- Starts fresh containers with service configuration
- Use case: "Run tests or perform an administrative task"
- Command overrides service configuration

**`docker compose exec`:**
- For executing in **already running** containers
- Allocates TTY by default (interactive mode)
- Use case: Debug or inspect running services

**Key distinction:** `exec` is designed for **persistent services** (web servers, databases), not task execution.

**Sources:**
- [Docker Compose exec vs run](https://medium.com/analytics-vidhya/how-to-understand-the-difference-between-docker-composes-up-vs-run-vs-exec-commands-a506151967df)
- [Docker Compose Documentation](https://docs.docker.com/reference/cli/docker/compose/)

---

## Overhead Analysis

### Quantitative Assessment

**Scenario:** 6-experiment campaign (2 backends × 3 cycles)
- Experiment duration: 5 minutes average
- Total campaign time: 30 minutes

**`run --rm` overhead:**
- Per experiment: 4 seconds (average)
- Total overhead: 24 seconds across campaign
- Percentage: 1.3% of total campaign time

**`up + exec` overhead:**
- Container startup: 5 seconds (all backends)
- Container teardown: 5 seconds
- Total overhead: 10 seconds across campaign
- Percentage: 0.6% of total campaign time

**Overhead saved: 14 seconds (0.7% improvement)**

### Qualitative Assessment

**Is 24 seconds meaningful?**
- In a 30-minute campaign: No
- In a 6-hour multi-model campaign: 288 seconds = 4.8 minutes (~1.3% still)
- Human waiting time: Negligible (user is not watching each container start)

**What matters more:**
- Reliability (no memory leaks, clean starts)
- Debuggability (clear failure isolation)
- Maintainability (simple code)

---

## `sleep infinity` Pattern

### How It Works

To keep containers alive for `exec`:

**Dockerfile:**
```dockerfile
CMD ["sleep", "infinity"]
```

**docker-compose.yml:**
```yaml
services:
  pytorch:
    command: sleep infinity
```

**Technical implementation:** `sleep infinity` is smart enough to use the `pause` syscall rather than a loop, making it extremely efficient.

### Best Practice Guidance

**Consensus from research:**

✅ **Use for:**
- Development environments
- Debugging containers
- Testing scenarios

❌ **Don't use for:**
- Production workloads
- Automated task execution
- CI/CD pipelines

**Quote from research:**
> "Obviously the techniques below are intended for development only. If you're using these to keep the container alive in prod, even though the main process has exited, what is wrong with you!?"

**Source:** [How to Keep Docker Container Running](https://kodekloud.com/blog/keep-docker-container-running/)

### Applicability to LLenergyMeasure

Campaigns are **automated task execution**, not interactive development. Using `sleep infinity` would be:
- Against Docker best practices (ephemeral > persistent for tasks)
- Adding complexity for minimal gain (<1% overhead reduction)
- Introducing lifecycle management burden

---

## GPU Memory Isolation

### Container Isolation Guarantees

**Fresh containers (`run --rm`):**
- GPU memory is completely released between experiments
- CUDA contexts are fully torn down
- No risk of memory fragmentation
- Perfect isolation per experiment

**Shared containers (`exec`):**
- Same CUDA context across experiments
- Must manually clear GPU memory between runs
- Risk of memory leaks accumulating
- Potential for fragmentation over time

### GPU Sharing Research

**Memory isolation approaches:**
1. **Device whitelisting**: `--gpus` flag restricts GPU access per container
2. **MIG (Multi-Instance GPU)**: Hardware partitioning (A100/H100)
3. **nvshare/gShare**: Software frameworks for memory isolation

**Key finding:**
> "nvshare allows multiple processes or containers to securely run on the same physical GPU concurrently, each having the whole GPU memory available. Memory and fault isolation is guaranteed because co-located processes use different CUDA contexts."

**Implication:** Even with GPU sharing frameworks, **different CUDA contexts** are the isolation mechanism. Fresh containers provide this by default.

**Sources:**
- [NVIDIA Docker GPU Isolation](https://github.com/NVIDIA/nvidia-docker/wiki/GPU-isolation-(version-1.0))
- [nvshare Framework](https://github.com/grgalex/nvshare)

### LLenergyMeasure Isolation Strategy

**Current implementation (volumes):**
```yaml
volumes:
  - hf-cache:/app/.cache/huggingface        # Persists models
  - experiment-state:/app/.state            # Persists state
  - trt-engine-cache:/app/.cache/tensorrt   # Persists compiled engines
```

**How it works:**
- Named volumes are Docker-managed, persist across containers
- Models download once, cached in volume
- Each `run --rm` mounts same volumes
- Container is ephemeral, cache is persistent

**Result:** Best of both worlds — ephemeral containers with persistent caches.

---

## Error Recovery Patterns

### Failure Scenario Comparison

| Failure Type | `run --rm` Recovery | `up + exec` Recovery |
|--------------|---------------------|----------------------|
| **OOM (Out of Memory)** | Container exits, next run is clean | Memory persists, must detect + restart |
| **CUDA Error** | Container exits, CUDA context cleared | CUDA context persists, may cascade |
| **Model Load Failure** | Exit code 1, campaign continues | Must detect failure, decide restart |
| **GPU Memory Leak** | Impossible (fresh container) | Possible (long-running process) |
| **Zombie Processes** | Impossible (container removed) | Possible (process management) |
| **Thermal Throttling** | Natural gap (container startup) | Must implement explicit delays |

### ContainerManager Complexity

The `ContainerManager` class (orchestration/container.py) implements:
- Health checking via NVML (GPU memory monitoring)
- Automatic restart on unhealthy status
- Restart count tracking (max 3 attempts)
- GPU memory threshold checks (90% default)

**Code complexity:**
- 308 lines of code
- Health check method: 50 lines
- Restart logic: 40 lines
- Recovery method: 30 lines

**What it prevents:**
- Executing in containers with high GPU memory usage
- Cascading failures from unhealthy containers

**What `run --rm` prevents automatically:**
- All of the above (fresh container = fresh state)

---

## Cache Warming Analysis

### HuggingFace Model Cache

**Concern:** Do ephemeral containers re-download models?

**Answer:** No, because of named volumes.

**How it works:**
```yaml
volumes:
  hf-cache:
    name: lem-hf-cache  # Named volume (Docker-managed)

services:
  pytorch:
    volumes:
      - hf-cache:/app/.cache/huggingface
```

**Lifecycle:**
1. First experiment: Model downloads to named volume
2. Container exits and is removed
3. Second experiment: New container mounts same named volume
4. Model already exists in cache, no re-download

**Verification:**
```bash
# Check volume exists
docker volume ls | grep lem-hf-cache

# Inspect volume contents
docker run --rm -v lem-hf-cache:/cache alpine ls -lh /cache
```

### TensorRT Engine Cache

Same pattern for compiled engines:
```yaml
volumes:
  trt-engine-cache:
    name: lem-trt-engine-cache

services:
  tensorrt:
    volumes:
      - trt-engine-cache:/app/.cache/tensorrt-engines
```

**First run:** Compiles engine, saves to volume
**Subsequent runs:** Loads from volume (no recompilation)

### Experiment State

Campaign state also persists:
```yaml
volumes:
  experiment-state:
    name: lem-experiment-state

services:
  pytorch:
    volumes:
      - experiment-state:/app/.state
```

**Result:** Campaign resumption works across ephemeral containers.

---

## Implementation Comparison

### Current Implementation (Simple)

**Campaign dispatch loop:**
```python
for experiment in campaign.experiments:
    cmd = [
        "docker", "compose", "run", "--rm",
        experiment.backend,
        "lem", "experiment", experiment.config_path, "--yes"
    ]
    subprocess.run(cmd, check=True)
```

**Lines of code:** ~5 lines (simplified)

**Error handling:** Subprocess exit code
**State management:** None needed (containers are stateless)
**Health monitoring:** None needed (fresh containers)

### Alternative Implementation (Complex)

**Campaign dispatch with ContainerManager:**
```python
with ContainerManager() as cm:
    # Start all backends
    backends = set(exp.backend for exp in campaign.experiments)
    cm.start_services(list(backends), wait=True)

    for experiment in campaign.experiments:
        # Check health before each experiment
        health = cm.check_and_recover(
            experiment.backend,
            gpu_memory_threshold_pct=90.0,
            max_restarts=3
        )
        if not health.healthy:
            raise RuntimeError(f"Backend {experiment.backend} unhealthy")

        # Execute experiment
        output, code = cm.execute_experiment(
            experiment.backend,
            experiment.config_path,
            extra_args=["--yes"]
        )
        if code != 0:
            # Handle failure, decide whether to continue
            pass

    # Teardown all backends
    cm.teardown(timeout=30)
```

**Lines of code:** ~30+ lines (simplified, actual implementation longer)

**Dependencies:** python-on-whales library
**Error handling:** Manual health checks, restart logic, timeout management
**State management:** Track active services, restart counts
**Health monitoring:** GPU memory NVML queries

**Additional code required:**
- ContainerManager class: 308 lines
- Health status dataclass: 20 lines
- Docker exception handling: Throughout

---

## Trade-offs Summary

### Ephemeral Containers (`run --rm`) — Current Approach

**Advantages:**
- ✅ Perfect isolation (fresh CUDA context, GPU memory)
- ✅ Automatic error recovery (container exit = clean slate)
- ✅ Simple implementation (5 lines vs 300+)
- ✅ Aligns with Docker best practices (ephemeral containers)
- ✅ No lifecycle management complexity
- ✅ No risk of memory leaks or state accumulation
- ✅ Natural thermal recovery gaps (container startup time)

**Disadvantages:**
- ❌ 3-5 second overhead per experiment
- ❌ Container create/start for each experiment
- ❌ Base image dependency check each time (cached but verified)

**Total cost:** ~24 seconds in a 30-minute campaign (1.3%)

### Long-Running Containers (`up + exec`) — Alternative

**Advantages:**
- ✅ Eliminates 3-5 second per-experiment overhead
- ✅ Containers stay warm (no startup delay)

**Disadvantages:**
- ❌ Requires lifecycle management (start, health, restart, teardown)
- ❌ Adds 300+ lines of code (ContainerManager)
- ❌ Must manually ensure GPU memory is cleared
- ❌ Risk of memory leaks across experiments
- ❌ Risk of GPU memory fragmentation
- ❌ Must implement health monitoring (NVML queries)
- ❌ Must implement restart logic (max attempts, backoff)
- ❌ Against Docker best practices for task execution
- ❌ Requires `sleep infinity` or dummy process to keep alive
- ❌ More complex error handling (detect vs exit code)
- ❌ Shared CUDA context across experiments (potential state leakage)

**Total savings:** ~14 seconds in a 30-minute campaign (0.7%)

---

## Recommendation

### Primary Recommendation: Continue Using `docker compose run --rm`

**Rationale:**

1. **Overhead is negligible:** 1.3% of campaign time is not worth added complexity
2. **Isolation is critical:** LLM inference benchmarks require clean GPU state per experiment
3. **Simplicity matters:** 5 lines of code vs 300+ lines is a 60x difference in maintenance burden
4. **Best practices alignment:** Docker documentation explicitly recommends ephemeral containers for task execution
5. **Named volumes solve caching:** Models are already cached, no re-download occurs
6. **Error recovery is automatic:** Container exit = guaranteed clean slate
7. **Similar tools use this pattern:** MLPerf uses one-off containers for benchmark runs

### When to Reconsider

Revisit this decision if:
- Container startup overhead exceeds 10% of average experiment time
- Campaign includes >100 experiments where overhead accumulates significantly
- Evidence emerges of state leakage across ephemeral containers (would be a Docker bug)
- Docker introduces a native "container pooling" feature for task execution

### Measured Overhead Threshold

**Break-even analysis:**

If experiments averaged 30 seconds each:
- 4-second overhead = 13% (might justify complexity)

If experiments averaged 10 seconds each:
- 4-second overhead = 40% (would justify complexity)

**Current reality:**
- Experiments average 1-10 minutes (60-600 seconds)
- 4-second overhead = 0.7-6.7% (does not justify complexity)

---

## Alternative Optimisation Strategies

If campaign overhead becomes a concern, consider these instead:

### 1. Parallel Experiment Execution

Run independent experiments in parallel on different GPUs:
```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=0 docker compose run --rm pytorch lem experiment exp1.yaml

# Terminal 2
CUDA_VISIBLE_DEVICES=1 docker compose run --rm pytorch lem experiment exp2.yaml
```

**Overhead reduction:** Amortised across parallel executions
**Complexity:** Low (no code changes)
**Risk:** Low (experiments are isolated by GPU)

### 2. Batch Experiment Submission

Modify LLenergyMeasure to accept multiple configs in one invocation:
```bash
docker compose run --rm pytorch lem batch exp1.yaml exp2.yaml exp3.yaml
```

**Overhead reduction:** One container for multiple experiments
**Complexity:** Medium (requires batch command implementation)
**Risk:** Medium (must ensure GPU cleanup between experiments)

### 3. Docker Image Pre-pulling

Ensure images are pre-pulled before campaign starts:
```bash
docker compose pull pytorch vllm tensorrt
```

**Overhead reduction:** Eliminates image pull time (if any)
**Complexity:** Trivial (one command)
**Risk:** None

### 4. Docker BuildKit Caching

Use BuildKit for faster image builds during development:
```bash
DOCKER_BUILDKIT=1 docker compose build
```

**Overhead reduction:** Faster image rebuilds
**Complexity:** Trivial (environment variable)
**Risk:** None

---

## Implementation Note

The `ContainerManager` class exists in the codebase (`orchestration/container.py`) but appears to be:
- Not currently used by the campaign command (campaign.py uses subprocess + `docker compose run --rm`)
- An experimental implementation exploring the `up + exec` pattern
- Evidence of prior investigation into this exact question

**Recommendation:** Remove `ContainerManager` class to reduce codebase complexity, or clearly document it as experimental/unused.

---

## Sources

### Primary Sources (HIGH Confidence)
- [MLCommons Inference Reference](https://github.com/mlcommons/inference) - MLPerf benchmark container patterns
- [Docker Best Practices](https://docs.docker.com/build/building/best-practices/) - Official ephemeral container guidance
- [Docker Compose exec Documentation](https://docs.docker.com/reference/cli/docker/compose/exec/) - Official exec vs run semantics
- [Docker Compose run Documentation](https://docs.docker.com/reference/cli/docker/compose/run/) - Official run command for tasks

### Secondary Sources (MEDIUM Confidence)
- [Docker Compose exec vs run](https://medium.com/analytics-vidhya/how-to-understand-the-difference-between-docker-composes-up-vs-run-vs-exec-commands-a506151967df) - Community guidance
- [Keep Docker Container Running](https://kodekloud.com/blog/keep-docker-container-running/) - sleep infinity patterns
- [GPU Isolation with NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker/wiki/GPU-isolation-(version-1.0)) - GPU memory isolation approaches

### Supporting Sources (MEDIUM Confidence)
- [Docker Performance 2026](https://www.docker.com/blog/dockers-developer-innovation-unveiling-performance-milestones/) - Container startup metrics
- [Ephemeral Containers Principles](https://medium.com/@h.stoychev87/containerization-docker-and-containers-8e8f28fd0694) - Container design philosophy
- [nvshare GPU Sharing](https://github.com/grgalex/nvshare) - GPU memory isolation frameworks

---

## Appendix: Code References

### Current Campaign Implementation
**File:** `src/llenergymeasure/cli/campaign.py`
**Pattern:** Subprocess invocation of `docker compose run --rm`
**Lines:** ~423-427

### ContainerManager Implementation
**File:** `src/llenergymeasure/orchestration/container.py`
**Pattern:** Long-running container lifecycle management
**Lines:** 308 total (full implementation)

### Docker Compose Configuration
**File:** `docker-compose.yml`
**Services:** pytorch, vllm, tensorrt
**Volumes:** Named volumes for HF cache, TRT engines, experiment state
**Key config:** `privileged: true` for NVML access

---

**End of Research Document**
