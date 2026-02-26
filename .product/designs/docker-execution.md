# Docker Execution Design

**Status**: Complete — decisions confirmed 2026-02-19
**Last updated**: 2026-02-25
**Target version**: v2.0 (later milestone — Docker multi-backend studies)
**Source decisions**: [../decisions/docker-execution.md](../decisions/docker-execution.md)
**Note**: Docker is a later v2.0 milestone (after core single-backend local execution works).

---

## Core Architecture: Ephemeral `docker run` per Experiment

One container starts, runs one experiment, exits. `StudyRunner` (host) loops over experiments
calling `subprocess.run(["docker", "run", ...])` as a blocking call. Results written to a
mounted shared volume before the container exits.

```
StudyRunner (host)
│
│  for each experiment:
│    sleep(config_gap_seconds)          ← host manages thermal gap
│    subprocess.run([
│        "docker", "run",
│        "--rm",
│        "--gpus", "all",
│        "--volume", f"{results_dir}:/results:rw",
│        "--env", f"LLEM_CONFIG_PATH=/results/{config_id}.json",
│        "--shm-size", "8g",            ← required for vLLM (known P0 bug)
│        backend_image,
│    ])                                  ← blocking — returns when container exits
│    result = read_result_from_volume() ← container wrote result before exit
│
└── StudyResult
```

**Key invariant**: container startup and model load are NOT in the measurement window.
Measurement begins after the model is ready inside the container. Container startup ≡ latency,
not measurement error.

---

## Container Images

One image per backend. Shared base layer (CUDA + Python).

```
ghcr.io/llenergymeasure/base:{cuda_version}     ← CUDA + Python only (no ML backends)
ghcr.io/llenergymeasure/pytorch:{version}       ← base + torch + transformers
ghcr.io/llenergymeasure/vllm:{version}          ← base + vllm
ghcr.io/llenergymeasure/tensorrt:{version}      ← base + tensorrt-llm
```

**Version tag format**: `{llem_version}-cuda{cuda_version}` — e.g. `2.2.0-cuda12.4`

The entrypoint inside each backend image calls `ExperimentOrchestrator` directly via the
library API — not the `llem run` CLI. Config passed via env var → result written to mounted
volume. (Decision confirmed in decisions/docker-execution.md — library API is cleaner and
avoids needing the llem CLI installed in the container.)

---

## Host–Container Communication

**Config in**: mounted JSON file. Container reads `LLEM_CONFIG_PATH` env var → deserialises
`ExperimentConfig`.

```python
# StudyRunner — Docker dispatch
import json, subprocess, tempfile, shutil
from pathlib import Path

def _dispatch_docker(
    config: ExperimentConfig,
    backend_image: str,
    results_dir: Path,
) -> ExperimentResult:
    config_id = config.config_hash
    config_path = results_dir / f"{config_id}_config.json"
    config_path.write_text(config.model_dump_json())

    subprocess.run(
        [
            "docker", "run", "--rm",
            "--gpus", "all",
            "--volume", f"{results_dir}:/results:rw",
            "--env", f"LLEM_CONFIG_PATH=/results/{config_id}_config.json",
            "--shm-size", "8g",
            backend_image,
        ],
        check=True,   # raises CalledProcessError on non-zero exit
        timeout=_calculate_timeout(config),
    )

    result_path = results_dir / f"{config_id}_result.json"
    return ExperimentResult.from_json(result_path)
```

**Results out**: container writes `{config_hash}_result.json` to `/results/` before exit.
Completion signal is the container process exit (subprocess.run returns).

**Rejected alternatives**:
- `docker exec`: tight coupling to container internals, harder timeouts, weak isolation
- HTTP API inside container: only justified for persistent containers; not needed here
- stdin/stdout: fragile, hard to debug, breaks with backend verbose output

---

## TRT Engine Compilation: Disk Cache

TensorRT-LLM compiles engines keyed to exact config tuples:
`model × precision × tp_size × max_batch_size × max_seq_len → engine`

Cache strategy (cross-session, cross-study — compile once, reuse indefinitely):

```python
# Cache key: deterministic hash of compile-relevant config fields
def _trt_cache_key(config: ExperimentConfig) -> str:
    relevant = {
        "model": config.model,
        "precision": config.precision,
        "tp_size": config.tensorrt.tp_size if config.tensorrt else 1,
        "max_batch_size": config.tensorrt.max_batch_size if config.tensorrt else 1,
        "max_seq_len": config.tensorrt.max_seq_len if config.tensorrt else 2048,
    }
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:16]

CACHE_DIR = Path.home() / ".config" / "llenergymeasure" / "trt-engines"

# Mounted into each TRT container run:
# --volume ~/.config/llenergymeasure/trt-engines:/trt-cache:rw
```

**First run per unique config**: container compiles engine, writes to `/trt-cache/{hash}/`.
**Subsequent runs (same config)**: container reads cached engine from `/trt-cache/{hash}/`, skips compile.

**Implication**: TRT sweeps over unique configs compile one engine per unique config tuple —
unavoidable TRT-LLM constraint, not an llem architectural choice.

<!-- TODO: Define the exact set of config fields that determine the cache key. tp_size,
     max_batch_size, max_seq_len are obvious. What about builder_opt_level, quantization,
     speculative draft_model? Needs a dedicated spec before Docker milestone implementation. -->

---

## Error Handling

Container failures must not crash `StudyRunner`.

```python
# StudyRunner Docker path — error handling
try:
    subprocess.run(
        docker_cmd,
        check=True,
        timeout=_calculate_timeout(config),
    )
    result = ExperimentResult.from_json(results_dir / f"{config.config_hash}_result.json")
    study_result.add_result(result)

except subprocess.TimeoutExpired:
    # Container hung — kill it
    subprocess.run(["docker", "kill", container_name], check=False)
    study_result.add_failed(StudyFailed(
        config=config.model_dump(),
        exception_type="TimeoutError",
        error_message=f"Container exceeded timeout ({timeout}s)",
    ))

except subprocess.CalledProcessError as e:
    # Container exited with non-zero — OOM, CUDA error, unhandled exception
    # Container may have written partial result or error JSON
    error_path = results_dir / f"{config.config_hash}_error.json"
    exc_info = json.loads(error_path.read_text()) if error_path.exists() else None

    study_result.add_failed(StudyFailed(
        config=config.model_dump(),
        exception_type=exc_info["type"] if exc_info else "ContainerCrash",
        error_message=exc_info["message"] if exc_info else f"Exit code {e.returncode}",
    ))
```

Container writes `{config_hash}_error.json` (with `type`, `message`, `traceback`) on failure,
allowing structured error forwarding back to the host — mirroring the local Pipe pattern.

---

## Local vs Docker Symmetry

Both execution paths implement the same `StudyRunner` interface. Only the isolation primitive
differs:

| Concern | Local (v2.0) | Docker (v2.0 — later milestone) |
|---|---|---|
| Isolation | `multiprocessing.Process` | `docker run --rm` (ephemeral) |
| Config in | Function argument | Mounted JSON file + env var |
| Result out | `multiprocessing.Pipe` | Shared volume (JSON file) |
| Completion signal | `process.join()` | `subprocess.run()` returns |
| Timeout | `p.join(timeout=...)` | `subprocess.run(timeout=...)` |
| Error forwarding | Exception dict via Pipe | JSON error file in volume |

`StudyRunner` dispatches to `_dispatch_local()` or `_dispatch_docker()` based on the resolved
runner for each experiment's backend.

---

## What Is NOT Supported

**No persistent containers**: Ephemeral per experiment is the only supported model. Persistent
containers would require an HTTP API server inside every backend image, two orchestration code
paths, and a different result-retrieval mechanism. No peer tool offers this as a user toggle.

**No `container_lifecycle` toggle in study.yaml**: `runner: docker` selects Docker execution.
The lifecycle is always ephemeral.

**No `docker compose` for orchestration**: `docker run` directly, per experiment. Docker Compose
adds dependency and complexity for no gain over direct subprocess management.

---

## Runner Value Format (User Config)

From `~/.config/llenergymeasure/config.yaml`:

```yaml
runners:
  pytorch: local
  vllm:    docker:ghcr.io/llenergymeasure/vllm:2.2.0-cuda12.4
  tensorrt: docker:ghcr.io/llenergymeasure/tensorrt:2.2.0-cuda12.4
```

Parser:
```python
def _parse_runner(value: str) -> tuple[str, str | None]:
    if value == "local":
        return ("local", None)
    if value.startswith("docker:"):
        return ("docker", value.removeprefix("docker:"))
    if value.startswith("singularity:"):
        raise NotImplementedError(
            f"Singularity runner is not yet supported. Use 'local' or 'docker:<image>'."
        )
    raise ValueError(f"Unknown runner format: {value!r}")
```

---

## Image Publishing

Published to GitHub Container Registry (GHCR) via CI on each release.

```yaml
# .github/workflows/docker-publish.yml (sketch)
on:
  push:
    tags: ["v*"]

jobs:
  publish:
    strategy:
      matrix:
        backend: [pytorch, vllm, tensorrt]
    steps:
      - uses: docker/build-push-action@v5
        with:
          context: docker/${{ matrix.backend }}/
          tags: ghcr.io/llenergymeasure/${{ matrix.backend }}:${{ github.ref_name }}-cuda12.4
          push: true
```

<!-- TODO: Define the Dockerfile structure for each backend. What is in the base image vs
     per-backend? What CUDA version(s) to support? Does the base image install llenergymeasure
     itself, or does each backend image install it? Should images be built against pinned
     backend versions (e.g. vllm==0.6.x) or floating? This needs a concrete Dockerfile before
     Docker milestone implementation can begin. -->

---

## Peer References

| Tool | Local isolation | Docker isolation |
|---|---|---|
| **optimum-benchmark** | `multiprocessing.Process` per benchmark | Per backend family (CI only) |
| **AIEnergyScore** | N/A | Ephemeral `docker run` per experiment ← primary reference |
| **MLPerf inference** | Threaded in-process SUT | One container per backend run |
| **vLLM bench** | Long-running `subprocess.Popen` server | N/A (throughput tool) |

LLenergyMeasure follows AIEnergyScore for Docker orchestration.

---

## Related

- [../decisions/docker-execution.md](../decisions/docker-execution.md): Architectural decisions
- [experiment-isolation.md](experiment-isolation.md): Local subprocess isolation (v2.0)
- [study-yaml.md](study-yaml.md): `runner:` field, `cold_start:` field
- [user-config.md](user-config.md): Runner configuration (`runners:` section)
- [cli-commands.md](cli-commands.md): `llem run` command (unified CLI, handles both experiments and studies)
