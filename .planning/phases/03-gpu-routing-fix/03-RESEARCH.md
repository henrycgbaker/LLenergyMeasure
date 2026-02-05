# Phase 3: GPU Routing Fix - Research

**Researched:** 2026-02-04
**Domain:** Docker GPU passthrough, NVIDIA Container Toolkit, CUDA device visibility
**Confidence:** HIGH

## Summary

This phase addresses a critical bug where the `gpus` field from experiment configuration is not properly propagated to Docker containers, causing GPU initialization failures in vLLM and TensorRT tensor parallelism scenarios. The core issue is a mismatch between the Docker Compose static `NVIDIA_VISIBLE_DEVICES=all` and the need for dynamic per-experiment GPU selection.

The research confirms:
1. NVIDIA Container Toolkit uses `NVIDIA_VISIBLE_DEVICES` to control which GPUs are mounted into containers, while `CUDA_VISIBLE_DEVICES` is a CUDA-level environment variable respected by applications inside the container
2. Docker Compose supports `device_ids` in the deploy specification to select specific GPUs, but this is static (compile-time), not dynamic (runtime)
3. The correct approach for dynamic GPU selection is to pass `NVIDIA_VISIBLE_DEVICES` via `docker compose run -e` flags

**Primary recommendation:** Propagate `config.gpus` to containers via `docker compose run -e NVIDIA_VISIBLE_DEVICES=0,1` at experiment dispatch time, while keeping `CUDA_VISIBLE_DEVICES` set inside the container to match the remapped device indices (always `0,1,...,n-1` after NVIDIA remapping).

## Standard Stack

The established tools/mechanisms for this domain:

### Core
| Component | Purpose | Why Standard |
|-----------|---------|--------------|
| NVIDIA Container Toolkit | Mounts GPUs into containers | Official NVIDIA solution for Docker GPU access |
| `NVIDIA_VISIBLE_DEVICES` | Controls which GPUs container sees | Container Toolkit's device selection env var |
| `CUDA_VISIBLE_DEVICES` | Application-level GPU filtering | Standard CUDA environment variable |
| Docker Compose `deploy.resources.reservations.devices` | Static GPU allocation | Compose specification for GPU resources |
| `docker compose run -e` | Runtime environment override | Passes env vars for per-invocation customisation |

### Environment Variables
| Variable | Layer | When Set | Purpose |
|----------|-------|----------|---------|
| `NVIDIA_VISIBLE_DEVICES` | Container Runtime | At container start | Determines which GPUs are mounted by nvidia-container-toolkit |
| `CUDA_VISIBLE_DEVICES` | Application | Inside container | Application-level GPU selection (after remapping) |
| `NVIDIA_DRIVER_CAPABILITIES` | Container Runtime | At container start | What capabilities to expose (compute, utility, etc.) |

## Architecture Patterns

### GPU Device Remapping

When `NVIDIA_VISIBLE_DEVICES=2,5` is set:
- Physical GPUs 2 and 5 are mounted into the container
- Inside the container, they become `cuda:0` and `cuda:1`
- Setting `CUDA_VISIBLE_DEVICES=0,1` inside the container refers to these remapped devices

```
Host GPU indices:  [0, 1, 2, 3, 4, 5, 6, 7]
                           |     |
NVIDIA_VISIBLE_DEVICES=2,5 |     |
                           v     v
Container sees:            [0]   [1]
                            |     |
Application uses:       cuda:0  cuda:1
```

### Current docker-compose.yml Pattern (PROBLEM)

```yaml
x-common: &common
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: all  # <-- Static: all GPUs mounted
            capabilities: [compute, utility]
  environment:
    - NVIDIA_VISIBLE_DEVICES=all  # <-- Static: redundant with count: all
    - CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}  # <-- From host env
```

**Problem:** This mounts all GPUs but relies on host's `CUDA_VISIBLE_DEVICES`. When the host has `CUDA_VISIBLE_DEVICES=""` (shared server setup), the container sees all GPUs but applications may fail to initialise correctly.

### Recommended Pattern (FIX)

**Option A: Dynamic NVIDIA_VISIBLE_DEVICES via docker compose run -e**

```python
# In _run_experiment_in_docker():
env_vars = {}
if gpus:
    # Pass config.gpus as NVIDIA_VISIBLE_DEVICES to container
    env_vars["NVIDIA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    # Inside container, CUDA sees devices as 0,1,2,... (remapped)
    env_vars["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(len(gpus)))

for key, value in env_vars.items():
    cmd.extend(["-e", f"{key}={value}"])
```

**Why this works:**
1. `NVIDIA_VISIBLE_DEVICES=0,1` passed via `-e` overrides the static `all` in compose file
2. Inside container, only GPUs 0,1 are visible (from host perspective)
3. Container's `CUDA_VISIBLE_DEVICES=0,1` (remapped indices) ensures applications use all visible GPUs
4. vLLM's `tensor_parallel_size=2` sees 2 GPUs and initialises correctly

**Option B: Use device_ids in compose (NOT recommended for dynamic selection)**

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ['0', '1']  # <-- Static, requires compose file modification
          capabilities: [compute, utility]
```

This is appropriate for static deployments but doesn't support per-experiment GPU selection.

### vLLM Worker Initialisation

vLLM spawns worker processes for tensor parallelism. Each worker needs to see all GPUs intended for the experiment:

```
Main Process           Worker 0            Worker 1
    |                     |                    |
    |--spawn------------->|                    |
    |                     |--spawn------------>|
    v                     v                    v
CUDA_VISIBLE_DEVICES  CUDA_VISIBLE_DEVICES  CUDA_VISIBLE_DEVICES
    =0,1                  =0,1                 =0,1
    |                     |                    |
    v                     v                    v
Uses cuda:0           Uses cuda:0          Uses cuda:1
(rank 0)              (rank 0)             (rank 1)
```

**Critical:** If `CUDA_VISIBLE_DEVICES` is not set correctly before vLLM initialises, workers may see different GPU counts than expected, causing "Tensor parallel size cannot be larger than available GPUs" errors.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Dynamic GPU selection | Custom Docker API wrapper | `docker compose run -e` flags | Compose handles env override correctly |
| Device remapping logic | Manual index translation | NVIDIA Container Toolkit | NVIDIA handles remapping transparently |
| GPU detection in container | `torch.cuda.device_count()` before backend init | `CUDA_VISIBLE_DEVICES` env var | Avoids CUDA context init issues with vLLM/TensorRT |
| Parallelism validation | Runtime GPU count checks | Config-time validation against `config.gpus` | Fail-fast before container launch |

**Key insight:** The NVIDIA Container Toolkit already handles device remapping. The application layer just needs to set `NVIDIA_VISIBLE_DEVICES` correctly at container launch time and trust that inside the container, devices appear as `0,1,2,...`.

## Common Pitfalls

### Pitfall 1: CUDA_VISIBLE_DEVICES vs NVIDIA_VISIBLE_DEVICES Confusion

**What goes wrong:** Setting `CUDA_VISIBLE_DEVICES` on the host thinking it controls container GPU access. The container still sees all GPUs because `NVIDIA_VISIBLE_DEVICES=all` in compose file takes precedence.

**Why it happens:** These are two different layers:
- `NVIDIA_VISIBLE_DEVICES` = what NVIDIA Container Toolkit mounts
- `CUDA_VISIBLE_DEVICES` = what CUDA library filters at application level

**How to avoid:**
- Use `NVIDIA_VISIBLE_DEVICES` (or `--gpus` flag) to control what's mounted
- Use `CUDA_VISIBLE_DEVICES` inside container only if further filtering is needed

**Warning signs:** "All GPUs visible" despite setting `CUDA_VISIBLE_DEVICES` on host

### Pitfall 2: Forgetting Device Index Remapping

**What goes wrong:** Setting `CUDA_VISIBLE_DEVICES=2,5` inside container after mounting only GPUs 2,5 via `NVIDIA_VISIBLE_DEVICES=2,5`. Container only has devices 0,1 (remapped), so CUDA can't find devices 2,5.

**Why it happens:** NVIDIA Container Toolkit remaps host indices to container indices (always 0-based).

**How to avoid:** After setting `NVIDIA_VISIBLE_DEVICES=X,Y`, set `CUDA_VISIBLE_DEVICES=0,1,...` (sequential from 0) inside container, or omit `CUDA_VISIBLE_DEVICES` entirely.

**Warning signs:** "CUDA driver initialization failed", "No CUDA-capable device detected"

### Pitfall 3: vLLM CUDA Context Initialisation Race

**What goes wrong:** Calling `torch.cuda.device_count()` or any `torch.cuda.*` function before vLLM initialises causes CUDA context to be created in main process. vLLM then forks workers which inherit the CUDA context incorrectly.

**Why it happens:** vLLM uses spawn/fork multiprocessing. Pre-initialised CUDA contexts cause issues.

**How to avoid:**
- Never call `torch.cuda.*` in orchestration code for vLLM/TensorRT backends
- Use environment variables for GPU detection instead of runtime calls
- Set `RuntimeCapabilities.cuda_management = BACKEND` in backend protocol

**Warning signs:** "Cannot re-initialize CUDA in forked subprocess", worker processes hang

### Pitfall 4: tensor_parallel_size > Visible GPUs

**What goes wrong:** Config specifies `tensor_parallel_size=4` but only 2 GPUs are visible due to `CUDA_VISIBLE_DEVICES` or `NVIDIA_VISIBLE_DEVICES` limiting access.

**Why it happens:** No early validation of parallelism constraints against actual GPU visibility.

**How to avoid:**
- Validate `tensor_parallel_size <= len(config.gpus)` at config load time
- Validate `tp_size <= len(config.gpus)` for TensorRT
- Provide clear error message before container launch

**Warning signs:** "Tensor parallel size (X) cannot be larger than the number of available GPUs (Y)"

### Pitfall 5: count vs device_ids Mutual Exclusivity

**What goes wrong:** Docker Compose build fails or behaves unexpectedly when both `count` and `device_ids` are specified in devices configuration.

**Why it happens:** Docker Compose specification requires exactly one of `count` or `device_ids`, not both.

**How to avoid:** Use `count: all` for "give me all GPUs" or `device_ids: ['0', '1']` for specific GPUs, never both.

**Warning signs:** Docker Compose validation errors on `docker compose up`

## Code Examples

### Correct Docker GPU Dispatch Pattern

```python
def _run_experiment_in_docker(
    config_path: Path | None,
    backend: str,
    gpus: list[int] | None = None,
    # ... other args
) -> int:
    """Run experiment in Docker container with correct GPU routing."""
    cmd = ["docker", "compose", "run", "--rm"]

    env_vars = {}

    if gpus:
        # NVIDIA_VISIBLE_DEVICES controls which GPUs are mounted
        env_vars["NVIDIA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
        # CUDA_VISIBLE_DEVICES inside container uses remapped indices (0-based)
        env_vars["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(len(gpus)))

    for key, value in env_vars.items():
        cmd.extend(["-e", f"{key}={value}"])

    cmd.append(backend)
    # ... rest of command
```

### Fail-Fast Parallelism Validation

```python
def validate_parallelism_constraints(config: ExperimentConfig) -> list[ConfigWarning]:
    """Validate parallelism settings against available GPUs."""
    warnings = []
    gpu_count = len(config.gpus) if config.gpus else 1

    # vLLM tensor parallelism
    if config.backend == "vllm" and config.vllm:
        tp = config.vllm.tensor_parallel_size
        if tp > gpu_count:
            warnings.append(ConfigWarning(
                field="vllm.tensor_parallel_size",
                message=f"tensor_parallel_size={tp} exceeds available GPUs ({gpu_count}). "
                        f"Set gpus: [0, 1, ..., {tp-1}] or reduce tensor_parallel_size.",
                severity="error"
            ))

    # TensorRT tensor parallelism
    if config.backend == "tensorrt" and config.tensorrt:
        tp = config.tensorrt.tp_size
        if tp > gpu_count:
            warnings.append(ConfigWarning(
                field="tensorrt.tp_size",
                message=f"tp_size={tp} exceeds available GPUs ({gpu_count}). "
                        f"Set gpus: [0, 1, ..., {tp-1}] or reduce tp_size.",
                severity="error"
            ))

    return warnings
```

### Environment Variable Propagation in launcher.py

```python
def _early_cuda_visible_devices_setup() -> None:
    """Set CUDA_VISIBLE_DEVICES from config.gpus before any CUDA init.

    This runs in the subprocess spawned by accelerate/torchrun.
    By this point, NVIDIA_VISIBLE_DEVICES has already been set by the
    container runtime, so we're working with remapped indices.
    """
    # Parse config
    gpus = config_data.get("gpus", [])
    if not gpus:
        return

    # Inside container: use 0-based indices (already remapped)
    # The container sees len(gpus) devices numbered 0 to len-1
    cuda_devices = ",".join(str(i) for i in range(len(gpus)))
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `count: all` + host `CUDA_VISIBLE_DEVICES` | `NVIDIA_VISIBLE_DEVICES` via `-e` flag | Docker 19.03+ | Proper per-container GPU isolation |
| Runtime GPU detection | Config-time validation | Best practice | Fail-fast, clearer errors |
| `--gpus` flag only | `device_ids` in compose or `-e NVIDIA_VISIBLE_DEVICES` | Compose v2+ | Better compose integration |

**Current best practice:**
- Use `NVIDIA_VISIBLE_DEVICES` for container GPU selection (runtime)
- Use `CUDA_VISIBLE_DEVICES` for application-level filtering within container
- Validate parallelism constraints at config time against `config.gpus`

## Open Questions

1. **MIG Instance Handling**
   - What we know: MIG instances use UUIDs like `MIG-abc123`
   - What's unclear: Does `NVIDIA_VISIBLE_DEVICES=MIG-abc,MIG-def` work correctly with docker compose run -e?
   - Recommendation: Test MIG path separately, may need special handling

2. **Multi-node Distributed Inference**
   - What we know: Current implementation is single-node only
   - What's unclear: How would GPU routing work across nodes?
   - Recommendation: Out of scope for Phase 3, note for future

## Sources

### Primary (HIGH confidence)
- [NVIDIA Container Toolkit - Docker Specialized](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html) - NVIDIA_VISIBLE_DEVICES documentation
- [Docker Compose GPU Support](https://docs.docker.com/compose/how-tos/gpu-support/) - device_ids vs count, compose GPU specification
- [Docker Compose Environment Variables](https://docs.docker.com/compose/how-tos/environment-variables/set-environment-variables/) - `-e` flag for runtime override

### Secondary (MEDIUM confidence)
- [vLLM Parallelism and Scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/) - tensor_parallel_size behaviour
- [NVIDIA Forums - NVIDIA_VISIBLE_DEVICES vs CUDA_VISIBLE_DEVICES](https://forums.developer.nvidia.com/t/nvidia-visible-devices-vs-cuda-visible-devices/172697) - Community explanation of difference

### Tertiary (LOW confidence)
- GitHub issues and forum posts on vLLM multi-GPU problems (anecdotal, but consistent with documented behaviour)

## Metadata

**Confidence breakdown:**
- GPU routing mechanism: HIGH - Official NVIDIA and Docker documentation
- vLLM worker behaviour: MEDIUM - Inferred from docs and source code
- Validation patterns: HIGH - Standard defensive programming practice
- MIG handling: LOW - Not thoroughly tested

**Research date:** 2026-02-04
**Valid until:** 2026-05-04 (Docker/NVIDIA Container Toolkit APIs are stable)
