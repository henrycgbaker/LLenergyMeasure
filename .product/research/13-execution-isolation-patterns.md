# Research: Multi-Backend Experiment Execution Isolation Patterns

**Researched**: 2026-02-18
**Purpose**: Inform `decisions/docker-execution.md` — specifically container lifecycle model,
host-container communication, and cold start semantics for `llem study` v2.2 Docker mode.
**Confidence**: HIGH for optimum-benchmark (source verified), MEDIUM for MLPerf/vLLM (source verified),
LOW for ML.ENERGY per-experiment orchestration (secondary sources only).

---

## Summary of Findings

The clearest pattern across all tools: **process isolation is the universal primitive**, not Docker.
Docker is used for *environment isolation* (conflicting deps, reproducibility) — not as the IPC
mechanism between orchestrator and worker. The communication patterns are:

1. **subprocess.Popen + HTTP healthcheck + SIGKILL** (vLLM bench sweep)
2. **multiprocessing.Process + Pipe IPC** (optimum-benchmark process launcher)
3. **subprocess.run(run_docker.sh) per experiment** (AIEnergyScore batch_runner)
4. **Same process, thread-based SUT** (MLPerf reference implementation)

None of the tools use `docker exec` to dispatch individual experiments into a running container.

---

## 1. Optimum-Benchmark (HuggingFace)

**Source**: https://github.com/huggingface/optimum-benchmark

### Architecture Overview

Optimum-benchmark's execution model is: **one `multiprocessing.Process` per experiment**.
The launcher system is the key design. Three launchers exist:

| Launcher | Isolation | When to Use |
|----------|-----------|-------------|
| `inline` | None (same process) | Debugging only |
| `process` | Fresh `multiprocessing.Process` per experiment | All real benchmarks |
| `torchrun` | `process` wrapper + `elastic_launch` for distributed ranks | Multi-GPU distributed |

### Process Launcher — Exact Code

```python
# optimum_benchmark/launchers/process/launcher.py

def launch(self, worker: Callable[..., BenchmarkReport], worker_args: List[Any]) -> BenchmarkReport:
    child_connection, parent_connection = Pipe()
    main_process_pid = os.getpid()
    isolated_process = Process(
        target=target,
        args=(worker, worker_args, child_connection, main_process_pid, self.logger),
        daemon=False
    )

    with ExitStack() as stack:
        if self.config.numactl:
            stack.enter_context(self.numactl_executable())

        isolated_process.start()

        if isolated_process.is_alive():
            sync_with_child(parent_connection)   # checkpoint 1
        else:
            raise RuntimeError("Could not synchronize with isolated process")

        if self.config.device_isolation:
            stack.enter_context(self.device_isolation(isolated_process.pid))

        if isolated_process.is_alive():
            sync_with_child(parent_connection)   # checkpoint 2
        else:
            raise RuntimeError("Could not synchronize with isolated process")

        while isolated_process.is_alive() and not parent_connection.poll():
            pass

        # ... receive response
        response = parent_connection.recv()

        # Large payloads (>1MB) go via temp file instead of Pipe
        if isinstance(response, str) and response.startswith(tempfile.gettempdir()):
            response = pickle.load(open(response, "rb"))
```

```python
# target() function — runs inside the child process
def target(worker, worker_args, child_connection, main_process_pid, logger):
    main_process = psutil.Process(main_process_pid)
    # Sync checkpoints (verify parent is still alive)
    sync_with_parent(child_connection)   # checkpoint 1
    # ...
    sync_with_parent(child_connection)   # checkpoint 2

    file_based_comm_threshold = int(os.environ.get("FILE_BASED_COMM_THRESHOLD", "1_000_000"))

    try:
        report = worker(*worker_args)   # <-- actual benchmark runs here
    except Exception:
        str_traceback = traceback.format_exc()
        # Send directly if <1MB, else write to /tmp and send path
        child_connection.send(str_traceback)
    else:
        report_dict = report.to_dict()
        if len(str(report_dict)) <= file_based_comm_threshold:
            child_connection.send(report_dict)
        else:
            temp_file_path = os.path.join(tempfile.gettempdir(), f"optimum_benchmark_{os.getpid()}.pkl")
            with open(temp_file_path, "wb") as f:
                pickle.dump(report_dict, f)
            child_connection.send(temp_file_path)
    finally:
        child_connection.close()
        exit(0)
```

**The `worker` function** passed to `launch()` is `Benchmark.run`:

```python
# optimum_benchmark/benchmark/base.py

@staticmethod
def launch(config: BenchmarkConfig):
    """Runs a benchmark using specified launcher configuration/logic"""
    launcher = launcher_factory(launcher_config)
    report = launcher.launch(worker=Benchmark.run, worker_args=[config])
    return report

@staticmethod
def run(config: BenchmarkConfig):
    """Runs inside the isolated child process"""
    backend = backend_factory(backend_config)    # e.g. vLLMBackend, PyTorchBackend
    scenario = scenario_factory(scenario_config)
    report = scenario.run(backend)
    return report
```

### Torchrun Launcher — Three-Layer Architecture

```
Main process
  └── isolated_process (multiprocessing.Process)
        └── elastic_launch (torch.distributed.launcher.api)
              ├── rank-0 worker process
              ├── rank-1 worker process
              └── rank-N worker process
```

The isolated_process exists solely to host `elastic_launch`, which then spawns `nproc_per_node`
rank worker processes. Results aggregate back through the same `Pipe()` mechanism.

### Multi-Backend Sweeps

Multi-backend sweeps use **Hydra `--multirun`** (`-m` flag). Each configuration in the sweep
gets a fresh `Process`. Backend selection is just another config parameter:

```bash
# Sweep across backends serially (default)
optimum-benchmark --config-dir examples --config-name pytorch_bert -m \
  backend.device=cpu,cuda

# Parallel execution via hydra plugin (optional)
# hydra/launcher=joblib
```

**Key constraint**: Backends must be installed in the same Python environment. When vLLM and
TensorRT-LLM conflict at pip level, they cannot be in the same sweep. This is the fundamental
reason optimum-benchmark uses **separate Dockerfiles per hardware platform** (not per backend):

```
docker/
  cpu/Dockerfile      # CPU-only backends
  cuda/Dockerfile     # PyTorch CUDA (GPTQModel included)
  rocm/Dockerfile     # AMD ROCm backends
```

Each Dockerfile installs one compatible set of backends. The CI runs separate workflows per
backend family (`test_cli_cuda_vllm.yaml`, `test_cli_cuda_tensorrt_llm.yaml`) using
`FORCE_SEQUENTIAL=1` to prevent test-level parallelism within each run.

### Device Isolation

`device_isolation: true` in launcher config spawns a separate monitoring process that watches
GPU device occupancy via `pynvml` (NVIDIA) or `amdsmi` (AMD). If a foreign process appears on
the target GPU, it can: `warn`, raise `error`, or `kill` the intruder. This is **GPU slot
isolation**, not process-level dependency isolation.

### vLLM Backend in optimum-benchmark

The vLLM backend instantiates the engine **in-process** (inside the child process spawned by
the process launcher). No sub-subprocess is started:

```python
# optimum_benchmark/backends/vllm/backend.py

def load_model_from_pretrained(self):
    if self.config.serving_mode == "offline":
        self.pretrained_model = LLMEngine.from_engine_args(EngineArgs(**self.vllm_kwargs))
    else:
        self.pretrained_model = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**self.vllm_kwargs))
```

vLLM's engine itself spawns its own internal worker subprocesses (tensor parallel workers, etc.)
but these are managed by vLLM, not by optimum-benchmark. The `process` launcher's child process
is the "host" for the entire vLLM engine.

---

## 2. MLPerf Inference

**Source**: https://github.com/mlcommons/inference
**Source**: https://github.com/mlcommons/inference/blob/master/loadgen/README.md

### Architecture: SUT Interface + LoadGen Library

MLPerf's architecture is fundamentally different from a study orchestrator. It measures a
**fixed, pre-selected backend** in a single invocation. There is no multi-backend orchestration
within a single run — cross-backend comparison is done by running the benchmark suite separately
for each backend and comparing submitted results.

```
┌─────────────────────────────────────────────┐
│  User's Benchmark Process (single binary)   │
│                                             │
│   ┌─────────┐     ┌──────────────────────┐  │
│   │ LoadGen │────▶│ SystemUnderTest (SUT)│  │
│   │ (C++ lib│     │ (user-implemented)   │  │
│   │ w/ pyb) │◀────│ - threads for workers│  │
│   └─────────┘     └──────────────────────┘  │
└─────────────────────────────────────────────┘
```

LoadGen is a C++ library with Python bindings. It issues queries to the SUT's `IssueQuery()`
method and receives responses via `QuerySamplesComplete()`. The SUT is **in-process**.

### Per-Backend Implementation

Each backend is a separate Python module (different file, same process model):

```python
# language/llama2-70b/main.py

parser.add_argument("--vllm", action="store_true", help="vllm mode")

if args.vllm:
    from SUT_API import SUT, SUTServer   # vLLM API-based SUT
else:
    from SUT import SUT, SUTServer       # PyTorch-based SUT
```

`SUT.py` (PyTorch backend) uses **threading**, not multiprocessing:

```python
# language/llama2-70b/SUT.py

self.query_queue = queue.Queue()
self.worker_threads = [None] * self.num_workers

# Model loaded in main thread
self.model = LlamaForCausalLM.from_pretrained(
    self.model_path,
    device_map="auto",
    low_cpu_mem_usage=True,
    torch_dtype=self.amp_dtype,
)

# Queries dispatched to worker threads via queue
```

Only **one backend runs per process invocation**. Cross-backend comparison is achieved by
running separate benchmarks (separate Docker containers, separate shell sessions) and submitting
results to MLCommons for aggregation.

### Docker Usage Pattern

MLPerf uses Docker as an **environment container** that wraps the entire benchmark. The Docker
lifecycle is: start container → run one backend benchmark to completion → container exits. No
orchestration of multiple backends within one container session:

```bash
# launch.sh (language/llama2-70b)
nvidia-docker run -it --rm --net=host --runtime=nvidia --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --cap-add=SYS_PTRACE --cap-add=SYS_ADMIN --cap-add=DAC_READ_SEARCH \
  --security-opt seccomp=unconfined \
  -w $PWD \
  --env-file `pwd`/.docker_env \
  ${MOUNT_FLAGS[*]} \
  llm/gpubringup \
  bash ./with_the_same_user
```

NVIDIA's submission infrastructure (from `inference_results_v3.0`) uses a Makefile-based system:

```bash
make build        # compile harness binaries inside container
make prebuild     # start container shell
make run_harness BENCHMARK=llama2-70b SCENARIO=Offline BACKEND=tensorrt  # run
```

Each `make run_harness` invocation runs one backend, one scenario. The container environment
stays consistent; the backend is a build-time or runtime selection.

### ml-commons automation tool (`mlcr`)

The `mlcr` CLI automates container launching:
```bash
# PyTorch reference implementation
mlcr run-mlperf,inference --model=llama2-70b-99 --implementation=reference \
  --framework=pytorch --category=datacenter --scenario=Offline \
  --execution_mode=test --device=cuda --quiet

# TensorRT (NVIDIA-specific)
mlcr run-mlperf,inference --model=llama2-70b-99 --implementation=nvidia \
  --framework=tensorrt --category=datacenter --scenario=Offline \
  --execution_mode=test --device=cuda --docker
```

Each `mlcr` invocation is a **complete, isolated benchmark run** for one backend. Cross-backend
comparison is a human-level operation (run A, then run B, compare submitted numbers).

**Key takeaway**: MLPerf does not have a "study orchestrator". It has one-shot benchmark
invocations per backend, with Docker for environment isolation.

---

## 3. vLLM Benchmark Suite

**Source**: https://github.com/vllm-project/vllm/tree/main/vllm/benchmarks/sweep

### Architecture: Long-Running Server + Multiple Client Runs

The vLLM `bench sweep serve` command implements the most directly relevant orchestration
pattern for LLenergyMeasure's `llem study` use case.

**Pattern**: For each serving configuration, start one server process → run all compatible
benchmark client configurations against it → stop server → next serving configuration.

```python
# vllm/benchmarks/sweep/serve.py (run_combs function)

for serve_comb in serve_params:
    with (
        run_server(
            serve_cmd,
            after_bench_cmd,
            show_stdout=show_stdout,
            serve_overrides=serve_comb,
            dry_run=dry_run,
            server_ready_timeout=server_ready_timeout,
        )
        if _comb_needs_server(serve_comb, bench_params, output_dir)
        else contextlib.nullcontext()
    ) as server:
        for bench_comb in bench_params:
            for run_number in range(num_runs):
                run_data = run_benchmark(server, bench_cmd, ...)
```

Server reuse: **the same server process handles multiple `bench_comb` iterations**. A new
server only starts for each new `serve_comb` (different model, tensor parallel size, etc.).

### Server Subprocess Management — Exact Code

```python
# vllm/benchmarks/sweep/server.py

def start(self):
    self._server_process = subprocess.Popen(
        self.server_cmd,
        start_new_session=True,          # new process group for clean kill
        stdout=None if self.show_stdout else subprocess.DEVNULL,
        env=os.environ | {"VLLM_SERVER_DEV_MODE": "1"},
    )

def wait_until_ready(self, timeout: int) -> None:
    start_time = time.monotonic()
    while not self.is_server_ready():
        if self._server_process.poll() is not None:
            raise RuntimeError(f"Server process crashed with return code {returncode}")
        if time.monotonic() - start_time > timeout:
            raise TimeoutError(f"Server failed to become ready within {timeout} seconds.")
        time.sleep(1)

def is_server_ready(self) -> bool:
    response = requests.get(f"{server_address}/health")
    return response.status_code == 200

def stop(self):
    server_process = self._server_process
    if server_process.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(server_process.pid), signal.SIGKILL)
```

**Communication pattern**: HTTP API (not `docker exec`, not `Pipe`). The orchestrator is a
client that POSTs requests to the server's HTTP endpoint. The server process is opaque — the
orchestrator only cares about the `/health` endpoint and the inference API.

### Parameter Sweep Grammar

```yaml
# serve_params.json — one entry per server configuration
[
  {"--model": "meta-llama/Llama-3.1-8B", "--tensor-parallel-size": "1"},
  {"--model": "meta-llama/Llama-3.1-8B", "--tensor-parallel-size": "2"}
]

# bench_params.json — one entry per client configuration
[
  {"--request-rate": "1"},
  {"--request-rate": "5"},
  {"--request-rate": "10"}
]
```

The sweep runs the Cartesian product: each serve_comb × each bench_comb. The server is started
**once per serve_comb** (not once per bench_comb). This means model loading happens once per
serving configuration, not once per benchmark run. This is the **long-running server** pattern.

### vLLM Sleep Mode (Relevant for Cold Start)

vLLM introduced sleep mode (2025) for in-process model weight offloading:

```python
# Level 1: offload weights to CPU, discard KV cache (~3-6s wake for large models)
llm.sleep(level=1)
llm.wake_up()

# Level 2: discard weights + KV cache, keep only buffers (reload from disk on wake)
llm.sleep(level=2)
llm.wake_up()
```

**Performance**: 18-200x faster than full server restart + model reload. Works with tensor
parallelism. This is directly relevant to `cold_start: true` semantics — sleep/wake_up is
the mechanism for process-level cold start measurement without container restart.

### No Docker in vLLM Sweep

The vLLM bench sweep does not use Docker. It assumes all backends are installed in the current
Python environment. Multi-backend comparison (vLLM vs. SGLang vs. TGI) is done by running
separate sweep invocations with different `--serve-cmd` values.

---

## 4. AIEnergyScore (HuggingFace)

**Source**: https://github.com/huggingface/AIEnergyScore

### Architecture: subprocess.run(docker_script) per Experiment

The most directly analogous tool to LLenergyMeasure's intended architecture. The `batch_runner.py`
orchestrator drives multiple experiments by invoking Docker **one container per experiment**:

```python
# batch_runner.py — _run_via_docker method

def _run_via_docker(self, config, run_dir, logger):
    cmd = [
        str(run_docker_script),
        "-n", str(self.num_prompts or 10),
        "--config-name", "text_generation",
    ]
    # Append Hydra-style overrides
    # e.g. backend.model=openai/gpt-oss-20b

    env = {
        **os.environ,
        "BENCHMARK_BACKEND": "pytorch",  # or "vllm", "optimum"
        "RESULTS_DIR": str(run_dir),
        "DOCKER_IMAGE": "ai_energy_score",
        "HF_HOME": str(hf_home),
    }

    result = subprocess.run(
        cmd,
        env=env,
        cwd=script_dir,
        capture_output=True,
        text=True,
    )
```

```bash
# run_docker.sh — invoked by batch_runner per experiment
docker run --gpus all --shm-size 1g \
  --user "$(id -u):$(id -g)" \
  ${VOLUME_MOUNTS} \
  ${ENV_VARS} \
  "${IMAGE_NAME}" \
  "${DOCKER_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
```

**Container lifecycle**: Ephemeral. Each experiment = one `docker run` → container exits → results
in mounted volume → `docker rm` cleanup. No reuse of container state between experiments.

### Explicit Cleanup Between Experiments

After each Docker run:

```python
def _cleanup_docker_containers(self, logger):
    result = subprocess.run(
        ["docker", "ps", "-a",
         "--filter", "ancestor=ai_energy_score",
         "--filter", "status=exited",
         "-q"],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode == 0 and result.stdout.strip():
        container_ids = result.stdout.strip().split("\n")
        for container_id in container_ids:
            subprocess.run(["docker", "rm", "-f", container_id], ...)
    time.sleep(0.5)
```

GPU memory is also cleared between runs for the vLLM backend (non-Docker path):
```python
torch.cuda.empty_cache()
gc.collect()
```

And a 2-second delay between model runs in the direct execution path.

### Submission Model

- Open-source models: HuggingFace portal auto-runs benchmarks centrally
- Proprietary models: Developer runs `run_docker.sh` locally, submits `GPU_ENERGY_WH.txt` +
  `GPU_ENERGY_SUMMARY.json` to submission portal

No central leaderboard collects raw measurements; contributors submit standardised output files.
This is a *decentralised collection* model (each contributor runs locally, submits results).

---

## 5. ML.ENERGY Leaderboard

**Source**: https://github.com/ml-energy/leaderboard
**Source**: https://github.com/ml-energy/zeus

### Architecture: Zeus Measurement Library + Centrally-Run Benchmarks

ML.ENERGY operates a **centrally-managed benchmark** (not contributor-submitted). They run all
experiments on their own hardware, publish results to a leaderboard. The benchmark suite is in a
separate repository (`ml-energy/benchmark`).

Zeus is the measurement library (not an orchestrator):

```python
# Zeus measurement window API
from zeus.monitor import ZeusMonitor

monitor = ZeusMonitor(gpu_indices=[0, 1])

monitor.begin_window("inference_run")
# ... run model inference ...
measurement = monitor.end_window("inference_run")
print(f"Energy: {measurement.total_energy} J")
```

Windows can be arbitrarily nested or overlapping. Zeus itself has no concept of "experiment
orchestration" — it is a measurement primitive that experiments call.

### Zeus Docker Constraints

Zeus requires:
- All GPUs mounted: `--gpus all`
- `SYS_ADMIN` capability for GPU power limit changes: `--cap-add=SYS_ADMIN`
- Intel RAPL metrics require sysfs access (disabled by Docker by default)

The `zeusd` daemon pattern solves the privilege problem: run `zeusd` with admin privileges as a
host service, expose a Unix Domain Socket, and the unprivileged container process calls zeusd
for NVML operations.

### Confidence Note

The ML.ENERGY per-experiment orchestration code is not public (it's in the separate `benchmark`
repo which is not fully open). The above is inferred from the leaderboard repo structure and Zeus
documentation. Confidence: LOW.

---

## 6. Dependency Isolation Reality: Why Docker Is Required

A key question for LLenergyMeasure: can vLLM, TensorRT-LLM, and PyTorch share a Python
environment?

**Answer: No for vLLM + TRT-LLM. Yes for PyTorch + vLLM in theory.**

Evidence from confirmed issues (2024-2025):

- TensorRT-LLM has strict `setuptools==70.3.0` requirements that conflict with many other packages
- TRT-LLM's CUDA version bakes in during build; switching CUDA versions requires rebuilding
- vLLM recommends "fresh new conda environment" for installation
- NVIDIA's own recommendation: use TRT-LLM's Docker image, not pip-mixed environments
- Installing TRT-LLM may uninstall/replace existing PyTorch (CUDA version mismatch)

**The only reliable multi-backend approach is one Python environment per backend**, which
means separate Docker images in practice.

However: **PyTorch + vLLM CAN coexist** (vLLM is built on PyTorch). The dependency boundary
is specifically between TRT-LLM and any other backend.

```
Isolation requirements by backend combination:
┌─────────────────────────────────────────────────────────┐
│ PyTorch + vLLM        → Same environment OK             │
│ PyTorch + TRT-LLM     → Separate environments required  │
│ vLLM + TRT-LLM        → Separate environments required  │
│ All three             → Separate environments required   │
└─────────────────────────────────────────────────────────┘
```

---

## Pattern Synthesis

### How the Ecosystem Handles Each Question

**1. Bare metal subprocess/multiprocessing vs Docker containers for process isolation**

| Tool | Mechanism | Docker? |
|------|-----------|---------|
| optimum-benchmark | `multiprocessing.Process` + `Pipe` IPC | No (CI uses Docker per backend family, not per experiment) |
| MLPerf reference | Same process, threaded SUT | Yes (one container per backend, per run) |
| vLLM bench sweep | `subprocess.Popen` for server process | No |
| AIEnergyScore | `subprocess.run(docker_script)` per experiment | Yes (ephemeral container per experiment) |
| ML.ENERGY | Unknown (private benchmark repo) | Yes (Zeus Docker image available) |

**2. How host orchestrator triggers experiment in isolated process/container**

| Tool | Communication | Pattern |
|------|--------------|---------|
| optimum-benchmark | `multiprocessing.Pipe` | Worker function passed by reference, result via Pipe |
| MLPerf reference | N/A (no multi-backend orchestrator) | One SUT per invocation |
| vLLM bench sweep | HTTP REST API | POST to `/generate`, GET `/health` for readiness |
| AIEnergyScore | CLI args via `subprocess.run` | Config as CLI flags to `run_docker.sh` |

**Nobody uses `docker exec` for experiment dispatch.** The pattern is always either:
- Pipe/queue IPC to a subprocess (optimum-benchmark)
- HTTP API to a server process (vLLM sweep)
- New container per experiment (AIEnergyScore, MLPerf)

**3. Whether experiments within a study reuse the same process or get fresh ones**

| Tool | Reuse? | Details |
|------|--------|---------|
| optimum-benchmark | **No** — fresh process per experiment | Core design principle; each `Benchmark.launch()` spawns new `Process` |
| MLPerf reference | N/A | Single run, no study concept |
| vLLM bench sweep | **Yes within serve_comb** — server reused for multiple bench_params | New server per new serving config |
| AIEnergyScore | **No** — new container per experiment | Ephemeral containers; explicit cleanup between runs |

**4. Conflicting Python-level dependencies (vLLM vs TRT-LLM)**

Universal answer: **separate environments** (i.e., separate containers). No tool
attempts to run vLLM and TRT-LLM in the same Python environment.

The mechanism differences:
- optimum-benchmark: separate Dockerfiles; user must install the right one
- MLPerf: separate Docker containers per backend submission
- AIEnergyScore: single `ai_energy_score` Docker image (PyTorch only; vLLM via external server)

---

## Implications for LLenergyMeasure v2.2 Docker Architecture

### What the Evidence Supports

**Container lifecycle decision**: Evidence strongly favours **long-running containers** (Option B
from `decisions/docker-execution.md`). The vLLM bench sweep pattern is most analogous: one
server per backend configuration, handle multiple experiment runs, then restart for the next
configuration. AIEnergyScore's ephemeral-per-experiment model incurs 10-30s startup overhead
per experiment, which directly confounds thermal gap measurement.

**Communication model decision**: Evidence strongly favours **HTTP API** over `docker exec`:
- vLLM already exposes an HTTP API (built-in)
- PyTorch backend can run a minimal FastAPI server (trivial to add)
- HTTP gives timeout semantics, progress streaming, health checks for free
- `docker exec` creates tight coupling to container internals and is harder to timeout
- No major tool uses `docker exec` for experiment dispatch

**Cold start decision**: vLLM sleep mode (level 1 or 2) is the right primitive for
process-level cold start measurement. Level 2 (discard weights, reload from disk) is
closest to `cold_start: true` semantics without requiring container restart. This validates
**Scenario A** (process-level unload/reload) from `decisions/docker-execution.md`.

**TRT-LLM engine caching**: MLPerf's approach (compile once per submission, store in shared
volume) validates the `~/.llenergymeasure/trt-engines/` cache strategy.

### Recommended Architecture for v2.2

Based on this research, the recommended execution pattern is:

```
Host orchestrator (llem study / StudyRunner)
│
├── docker run -d --name llem-pytorch  pytorch-image  # start long-running
├── docker run -d --name llem-vllm     vllm-image     # start long-running
└── docker run -d --name llem-trt      tensorrt-image # start long-running (+ TRT compile)
│
│  For each experiment in study:
│  ├── host sleeps config_gap_seconds            # exact thermal gap (host-controlled)
│  ├── POST http://localhost:{port}/run  {...}   # HTTP dispatch to container's API server
│  ├── poll GET http://localhost:{port}/status   # wait for completion
│  └── GET http://localhost:{port}/result        # retrieve ExperimentResult
│
└── docker stop + rm all containers when study complete
```

**If cold_start: true**:
```
For vLLM: POST /sleep {"level": 2}; POST /wake_up before experiment
For PyTorch: POST /unload; POST /load {"model": ...} before experiment
For TRT: POST /unload; POST /load {"engine_path": ...} before experiment
```

**Ports**: Each backend gets a fixed localhost port (`5001` PyTorch, `5002` vLLM, `5003` TRT).
The API server in each container is a thin FastAPI wrapper around the backend runner.

### What Not to Do

- **Do not use `docker exec`**: No peer tool does this. HTTP is cleaner, testable, and
  gives proper timeout semantics.
- **Do not use ephemeral containers per experiment** (AIEnergyScore model): startup time
  (10-30s vLLM, up to 60min TRT-LLM engine compile) makes `config_gap_seconds` uncontrollable.
- **Do not try to run vLLM + TRT-LLM in the same Python environment**: conflicting binary
  dependencies; separate containers are the correct solution.
- **Do not multiplex backends in one container**: different backends have different CUDA/driver
  requirements; one container per backend is the correct model.

---

## Source URLs

- Optimum-benchmark process launcher: https://github.com/huggingface/optimum-benchmark/blob/main/optimum_benchmark/launchers/process/launcher.py
- Optimum-benchmark Benchmark.launch: https://github.com/huggingface/optimum-benchmark/blob/main/optimum_benchmark/benchmark/base.py
- Optimum-benchmark vLLM backend: https://github.com/huggingface/optimum-benchmark/blob/main/optimum_benchmark/backends/vllm/backend.py
- Optimum-benchmark device isolation: https://github.com/huggingface/optimum-benchmark/blob/main/optimum_benchmark/launchers/device_isolation_utils.py
- Optimum-benchmark Dockerfile (CUDA): https://github.com/huggingface/optimum-benchmark/blob/main/docker/cuda/Dockerfile
- MLPerf inference repository: https://github.com/mlcommons/inference
- MLPerf llama2-70b main.py: https://github.com/mlcommons/inference/tree/master/language/llama2-70b
- MLPerf loadgen README: https://github.com/mlcommons/inference/blob/master/loadgen/README.md
- MLPerf NVIDIA submission: https://github.com/mlcommons/inference_results_v3.0/blob/main/closed/NVIDIA/README.md
- vLLM benchmarks sweep directory: https://github.com/vllm-project/vllm/tree/main/vllm/benchmarks/sweep
- vLLM server.py (Popen pattern): https://github.com/vllm-project/vllm/blob/main/vllm/benchmarks/sweep/server.py
- vLLM serve.py (run_combs): https://github.com/vllm-project/vllm/blob/main/vllm/benchmarks/sweep/serve.py
- vLLM sleep mode docs: https://docs.vllm.ai/en/latest/features/sleep_mode/
- AIEnergyScore repository: https://github.com/huggingface/AIEnergyScore
- AIEnergyScore batch_runner.py: https://github.com/huggingface/AIEnergyScore/blob/main/batch_runner.py
- AIEnergyScore run_docker.sh: https://github.com/huggingface/AIEnergyScore/blob/main/run_docker.sh
- ML.ENERGY leaderboard: https://github.com/ml-energy/leaderboard
- Zeus project: https://github.com/ml-energy/zeus
- TRT-LLM dependency conflict evidence: https://github.com/NVIDIA/TensorRT-LLM/issues/2587
