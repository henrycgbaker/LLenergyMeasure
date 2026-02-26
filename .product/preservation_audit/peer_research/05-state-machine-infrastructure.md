# Peer Research: State Machine Infrastructure

> Generated 2026-02-26. Peer evidence for preservation audit item N-X10.

## Evidence Per Tool

### 1. vLLM Bench Sweep

**Source**: [vLLM Parameter Sweeps docs](https://docs.vllm.ai/en/latest/benchmarking/sweeps/), [vLLM bench sweep serve CLI](https://docs.vllm.ai/en/latest/cli/bench/sweep/serve/)

**State persistence**: Filesystem. Each benchmark run produces a JSON result file named `{label}-{request_rate}qps-{model}-{timestamp}.json` in an output directory.

**Resume detection**: Output-file existence. The `--resume` flag takes the name of a previous output directory (a timestamp). When resuming, the sweep runner skips parameter combinations that already have output files present. There is no explicit state machine, state file, or database — the presence of a result file _is_ the state.

**Stale/orphaned cleanup**: None. If a run crashes mid-write, the partial output file either exists (treated as complete) or does not (re-run on resume). No PID check, no heartbeat, no timeout.

**Number of states**: Effectively 2 — "has output file" (done) or "no output file" (pending/needs run). No explicit state enum.

**State granularity**: Per-experiment (per parameter combination). No study-level state file.

**Key insight**: This is the simplest possible resume model. It works because benchmark result files are written atomically at the end of each run. The trade-off is that a run that crashes _during_ measurement is invisible — it simply has no output file and will be re-run. There is no way to distinguish "never started" from "started and crashed".

---

### 2. Hydra

**Source**: [Hydra multirun docs](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/), [Resume sweep feature request #1407](https://github.com/facebookresearch/hydra/issues/1407), [Re-run docs](https://hydra.cc/docs/1.2/experimental/rerun/)

**State persistence**: Filesystem — output directory structure. Each multirun creates `multirun/{date}/{time}/{job_num}/` with `.hydra/config.yaml`, `.hydra/overrides.yaml`, and `hydra.yaml` per job. No centralised state database.

**Resume detection**: **Not built in.** Hydra has no native resume-after-crash for multiruns. The maintainer (Omry Yadan) marked resume as "out of scope" for the basic sweeper, recommending custom sweeper plugins instead. Workarounds used by the community:
- Parse `.hydra/overrides.yaml` from completed job directories to identify what ran
- Log a sentinel value (e.g. "Job finished successfully") and parse logs
- Use Optuna sweeper plugin with persistent storage (the Optuna study itself tracks completion)
- Manually re-specify only remaining parameter combinations

An experimental "rerun" feature exists (v1.2+) but it replays a _specific_ job from saved config — it does not scan for incomplete jobs.

**Stale/orphaned cleanup**: None. No PID check, no heartbeat. If a job crashes, its output directory may be partially written with no indication of failure vs success unless the user's own code logs a completion marker.

**Number of states**: No explicit state machine. Jobs have a `JobStatus` property accessible from the launcher return value, but this is ephemeral (in-memory) — not persisted to disk.

**State granularity**: Per-job (per override combination). No sweep-level state file.

**Key insight**: Hydra deliberately avoids state management — it is a configuration framework, not an execution framework. Resume is pushed to plugins (Optuna, Ax) or user code. This is a conscious design boundary: Hydra composes configs, launchers execute them.

---

### 3. MLflow

**Source**: [MLflow RunStatus source](https://mlflow.org/docs/latest/_modules/mlflow/entities/run_status.html), [MLflow Backend Stores](https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/), [Filesystem deprecation notice](https://github.com/mlflow/mlflow/issues/18534)

**State persistence**: Database (SQLite or PostgreSQL) or filesystem (deprecated FileStore). In FileStore, each run gets `mlruns/{experiment_id}/{run_id}/meta.yaml` containing `run_id`, `experiment_id`, `user_id`, `status`, `start_time`, `end_time`, `lifecycle_stage`, `artifact_uri`. Database backend stores the same fields in a `runs` table.

**Resume detection**: By run ID. `mlflow.start_run(run_id=existing_id)` reopens a run. There is no config-hash-based matching — the caller must know the run ID. For "has this config already been run?" queries, users must search runs by logged parameters manually.

**Stale/orphaned cleanup**: **No automatic mechanism.** MLflow has no heartbeat, no PID check, no timeout. If a process crashes without calling `mlflow.end_run()`, the run remains in RUNNING state indefinitely. Cleanup is manual: users must call `mlflow.tracking.MlflowClient().set_terminated(run_id, status="FAILED")` or delete the run. The `lifecycle_stage` field (active/deleted) is for soft-delete, not staleness detection. FileStore moves deleted experiments to `.trash/`.

**Number of states**: 5 — `RUNNING`, `SCHEDULED`, `FINISHED`, `FAILED`, `KILLED`. Terminal states: `FINISHED`, `FAILED`, `KILLED`. `SCHEDULED` is for deferred execution (e.g. Databricks jobs). `KILLED` requires explicit user action.

**State granularity**: Per-run. Experiments are grouping containers, not stateful entities.

**Key insight**: MLflow is a _tracking_ system, not an execution system. It records what happened but does not orchestrate or resume runs. The lack of stale detection is a known pain point — issue #3228 documents that resuming a run does not properly reset state, and issue #3932 documents that status is not set correctly when the `with` context manager is not used. The 5-state machine is simple but the RUNNING-stuck problem is real and unsolved.

---

### 4. Weights & Biases (W&B)

**Source**: [W&B Run States](https://docs.wandb.ai/models/runs/run-states), [Resume docs](https://docs.wandb.ai/guides/runs/resuming), [Crash detection issue #1526](https://github.com/wandb/wandb/issues/1526)

**State persistence**: Cloud API (W&B servers). Run metadata is stored server-side. Local state cached in `wandb/` directory with run files, but the server is the source of truth.

**Resume detection**: By run ID + resume mode. Four modes:
- `resume="must"` — error if run ID not found
- `resume="allow"` — resume if found, create if not
- `resume="auto"` — automatic resume if restarted on same filesystem
- `resume="never"` — error if run ID already exists

No config-hash matching. The caller provides the run ID explicitly. `resume="auto"` uses a local `.wandb` directory marker to detect if a previous run existed in the same working directory.

**Stale/orphaned cleanup**: **Heartbeat-based.** The W&B client sends periodic heartbeats to the server. If heartbeats stop arriving (process crash, network loss, machine death), the server transitions the run to `Crashed` state. The heartbeat interval and timeout threshold are not publicly documented but are server-side. Known issue: false "Crashed" states when the process is still running but network is interrupted (issue #1526, #3405).

**Number of states**: 6 — `Pending`, `Running`, `Finished`, `Failed`, `Crashed`, `Killed`.
- `Pending`: scheduled but not started (sweeps, Launch jobs)
- `Running`: actively sending heartbeats
- `Finished`: explicit completion (exit code 0 or `wandb.finish()`)
- `Failed`: non-zero exit code
- `Crashed`: heartbeat timeout
- `Killed`: forcibly stopped (e.g. sweep cancellation)

**State granularity**: Per-run. Sweeps have separate status but do not control individual run states (pausing a sweep does not change running jobs' states; cancelling a sweep transitions running jobs to `Killed`).

**Key insight**: W&B is the only tool surveyed with automatic crash detection via heartbeat. However, it requires a server (cloud or self-hosted). The distinction between `Failed` (non-zero exit) and `Crashed` (heartbeat timeout) is unique and useful — most tools conflate these. The `Pending` state is also unique, reflecting W&B's role as a sweep/launch orchestrator.

---

### 5. Ray Tune

**Source**: [Ray Tune Trial source](https://docs.ray.io/en/latest/_modules/ray/tune/experiment/trial.html), [Fault tolerance docs](https://docs.ray.io/en/latest/tune/tutorials/tune-fault-tolerance.html), [Persistent storage docs](https://docs.ray.io/en/latest/tune/tutorials/tune-storage.html)

**State persistence**: Filesystem — `experiment_state-*.json` at experiment level, `trial_metadata.json` per trial. Checkpoints stored in `checkpoint_NNNNNN/` directories per trial. All persisted to a configurable storage path (local, NFS, S3, GCS). Trial state serialised/deserialised via `get_json_state()`/`from_json_state()`.

**Resume detection**: By experiment path. `Tuner.restore(path, resume_unfinished=True, resume_errored=True, restart_errored=False)` loads `experiment_state.json` and resumes trials based on their persisted state:
- `resume_unfinished=True` (default): re-run trials left in RUNNING state
- `resume_errored=True`: resume ERRORED trials from last checkpoint
- `restart_errored=True`: restart ERRORED trials from scratch
- TERMINATED trials are never re-run

No config-hash matching. Resume is path-based — "restore this experiment directory".

**Stale/orphaned cleanup**: **Implicit via restart.** When the experiment driver restarts and calls `Tuner.restore()`, trials in RUNNING state are treated as interrupted (since the driver is the one that would be running them). The `FailureConfig(max_failures=N)` parameter controls how many times a trial can fail and be re-scheduled. No PID check — Ray manages trial processes internally via its actor system. `_TemporaryTrialState` holds non-persistent state (hostname, PID, actor handle) that is explicitly _not_ restored on resume.

**Number of states**: 5 — `PENDING`, `RUNNING`, `PAUSED`, `TERMINATED`, `ERROR`. Transition path: PENDING -> RUNNING -> TERMINATED (success) or ERROR (failure). PAUSED is for PBT (Population-Based Training) trial suspension.

**State granularity**: Both per-trial (`trial_metadata.json`) and per-experiment (`experiment_state.json`). The experiment checkpoint is periodically saved (auto-tuned so <= 5% of time is spent checkpointing).

**Key insight**: Ray Tune is the most sophisticated resume system surveyed. It persists both experiment and trial state, supports partial resume (resume some trials, restart others), and handles the "trial was RUNNING when we crashed" case by design — any RUNNING trial found during restore is treated as interrupted. The 5-state machine with PAUSED is specific to Ray's actor model. The experiment-level periodic checkpoint is a pragmatic approach to balancing resume granularity against I/O overhead.

---

### 6. Optuna

**Source**: [Optuna TrialState](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.TrialState.html), [Optuna FAQ](https://optuna.readthedocs.io/en/stable/faq.html), [fail_stale_trials](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.fail_stale_trials.html), [RDBStorage](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.RDBStorage.html)

**State persistence**: Database — RDBStorage (PostgreSQL, MySQL) or JournalFileBackend (file-based). SQLite supported for single-process only (no concurrent access). State is per-trial in the `trials` table. Studies are grouping containers.

**Resume detection**: By study name. `optuna.load_study(study_name="my_study", storage="sqlite:///db.sqlite3")` loads the study and all its trials. New `study.optimize()` calls only create new trials — completed trials are not re-run. No config-hash matching — the storage _is_ the resume mechanism. If the process crashes, restarting `optimize()` with the same study name picks up where it left off (creates new trials, does not re-run COMPLETE ones).

**Stale/orphaned cleanup**: **Heartbeat-based** (experimental, since v2.9.0). Configuration:
```python
storage = optuna.storages.RDBStorage(
    url="sqlite:///db.sqlite3",
    heartbeat_interval=60,    # seconds between heartbeats
    grace_period=120,         # seconds before declaring stale
)
```
When a worker process dies, its trial's heartbeat stops updating. Other processes or `optuna.storages.fail_stale_trials(study)` transitions the trial from RUNNING to FAIL. A `RetryFailedTrialCallback` can automatically re-enqueue failed trials. Without heartbeat configured, killed trials remain RUNNING indefinitely (same problem as MLflow).

**Number of states**: 5 — `RUNNING` (0), `COMPLETE` (1), `PRUNED` (2), `FAIL` (3), `WAITING` (4). PRUNED is unique to Optuna — trials stopped early by the pruning algorithm. WAITING is for trials enqueued but not yet picked up by a worker.

**State granularity**: Per-trial. Studies are containers, not stateful entities.

**Key insight**: Optuna's heartbeat mechanism is the best-documented stale detection in the survey. It is explicit, configurable (interval + grace period), and has a dedicated API (`fail_stale_trials`). The trade-off is that it requires a database backend — the file-based JournalFileBackend does not support heartbeat. The PRUNED state is domain-specific (hyperparameter optimisation) and not relevant to our use case, but the WAITING state is interesting — it separates "queued" from "running", which our 3-state model does not.

---

### 7. Luigi / Airflow

#### Luigi

**Source**: [Luigi tasks docs](https://luigi.readthedocs.io/en/stable/tasks.html), [luigi/task_status.py](https://github.com/spotify/luigi/blob/master/luigi/task_status.py)

**State persistence**: **Output existence** (primary) + **scheduler in-memory** (ephemeral). Luigi's core completion model is the "target" pattern: a task is complete iff its `output()` target exists. This is checked by `complete()` which defaults to `self.output().exists()`. The scheduler tracks task status in memory for the current run but does not persist it. No state files, no database.

**Resume detection**: Output target existence. If a task's output file exists, it is skipped. If the output does not exist, the task (and its dependency chain) is re-run. This is identical in principle to Make's timestamp check but uses existence rather than modification time. No config hashing — task identity is determined by task class + parameters.

**Stale/orphaned cleanup**: **Scheduler timeout.** The scheduler has a `worker_disconnect_delay` (default 60s). If a worker stops heartbeating, its tasks are marked as failed after the timeout. For the execution itself, the `task_process_context` can be used to detect and clean up abandoned processes.

**Number of states**: 8 string constants — `PENDING`, `RUNNING`, `DONE`, `FAILED`, `BATCH_RUNNING`, `SUSPENDED`, `UNKNOWN`, `DISABLED`. However, the core model uses only 4: PENDING, RUNNING, DONE, FAILED. The others are for edge cases (DISABLED = too many failures, SUSPENDED = backward compat, BATCH_RUNNING = batch workers).

**State granularity**: Per-task. No workflow-level state.

#### Airflow

**Source**: [Airflow Tasks docs](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/tasks.html), [Airflow state module](https://airflow.apache.org/docs/apache-airflow/stable/_api/airflow/utils/state/index.html)

**State persistence**: Database (PostgreSQL, MySQL, SQLite). Task instance state stored in `task_instance` table. DAG run state stored in `dag_run` table.

**Resume detection**: By DAG run + task instance state. When a DAG run is retried or re-triggered, Airflow checks each task instance's state and only re-runs tasks that are not in a success/skipped state. No config hashing.

**Stale/orphaned cleanup**: **Heartbeat-based "zombie" detection.** The most sophisticated in the survey:
- Workers send heartbeats every `job_heartbeat_sec` (default 5s)
- Scheduler checks for zombies at a configurable interval
- Tasks are marked as zombie if no heartbeat received within `scheduler_zombie_task_threshold` seconds
- Zombie tasks are either failed or retried based on `retries` configuration
- The scheduler also has its own heartbeat (`scheduler_heartbeat_sec`) monitored by a health endpoint

**Number of states**: 13 — `none`, `scheduled`, `queued`, `running`, `success`, `restarting`, `failed`, `skipped`, `upstream_failed`, `up_for_retry`, `up_for_reschedule`, `deferred`, `removed`. Ideal flow: none -> scheduled -> queued -> running -> success.

**State granularity**: Per-task-instance (a specific task in a specific DAG run). DAG runs also have aggregate state.

**Key insight (Luigi)**: The "output target as state" pattern is the purest form of idempotent execution. No state machine needed — just "does the output exist?" This is elegant but limited: no way to distinguish "running" from "never started", and no partial progress tracking.

**Key insight (Airflow)**: 13 states is the most complex state machine in the survey, driven by Airflow's role as a production workflow orchestrator handling retries, scheduling, deferral, and branching. The zombie detection via heartbeat is production-proven at massive scale but requires the full Airflow scheduler infrastructure. The 5-second heartbeat interval is aggressive — appropriate for long-running workflows, possibly overkill for benchmarks.

---

### 8. Make / Bazel

#### Make

**Source**: [POSIX make spec](https://pubs.opengroup.org/onlinepubs/9699919799/utilities/make.html), [Content-based change detection with Make](https://andydote.co.uk/2022/09/19/make-content-hash/)

**State persistence**: **None** — Make has no state files. "State" is inferred entirely from the filesystem: a target is up-to-date iff its modification time is newer than all its prerequisites' modification times.

**Resume detection**: Timestamp comparison. If a target file exists and is newer than its sources, Make skips it. This is the original "output existence + freshness" pattern.

**Stale/orphaned cleanup**: None. If Make is interrupted, partially-written targets are left on disk. Since their timestamps will be newer than prerequisites, they appear "up to date" on the next run — a known problem. Make's `-j` (parallel) mode does not protect against this. The `.DELETE_ON_ERROR` special target can be used to delete targets on failure, but it is not default.

**Number of states**: Effectively 2 — "up to date" or "needs rebuild". No explicit state enum.

**Content-addressed alternative**: Make uses timestamps by default. Content-addressed builds require workarounds (e.g. [hashdeps](https://github.com/olipratt/hashdeps) which hashes file contents and only updates the timestamp when the hash changes). This is not native Make.

#### Bazel

**Source**: [Bazel remote caching](https://bazel.build/remote/caching), [How Bazel Works](https://sluongng.hashnode.dev/bazel-caching-explained-pt-1-how-bazel-works), [The Many Caches of Bazel](https://blog.engflow.com/2024/05/13/the-many-caches-of-bazel/)

**State persistence**: Two caches:
1. **Action Cache (AC)**: Maps action hash -> action result metadata. The action hash is computed from: command line, input file digests, environment variables, and other action metadata. This is the "config hash" analogy.
2. **Content-Addressable Store (CAS)**: Maps content hash -> file blob. Output files stored by their SHA-256 digest.

Both caches can be local (disk) or remote (gRPC server). The scheme is fully content-addressed — no timestamps.

**Resume detection**: Action hash lookup. Before executing an action, Bazel computes its hash from all inputs and checks the AC. If found, outputs are retrieved from CAS. If not found, the action executes and results are stored in both AC and CAS. This is pure content-addressed caching — the closest analogy to our `compute_config_hash()`.

**Stale/orphaned cleanup**: **Cache eviction only.** Local cache has a configurable size limit. Remote cache servers implement their own eviction policies (typically LRU). There is no concept of "stale" or "orphaned" — cache entries are either valid (hash matches) or absent (evicted or never cached). An action whose inputs change gets a new hash, so old cache entries are simply never looked up again.

**Number of states**: Effectively 2 — "cached" (hit) or "not cached" (miss/execute). No state machine. No concept of "running" or "failed" in the cache layer.

**Key insight**: Bazel's content-addressed model is the gold standard for config-hash-based resume. The critical design choice is that the hash includes _all_ inputs that affect the output, making cache hits guaranteed-correct (hermetic builds). Our `compute_config_hash()` is the same idea applied to experiment configs. The difference is that Bazel's hash is purely deterministic (same inputs = same hash = same outputs), while experiment results have inherent non-determinism (same config can produce different measurements). This means Bazel can _replace_ execution with cached results, while we can only _skip_ re-execution and keep old results.

---

## Summary Table

| Tool | Storage | Resume Mechanism | Stale Detection | States | Granularity |
|------|---------|-----------------|-----------------|--------|-------------|
| **vLLM sweep** | Files (result JSON) | Output file existence | None | 2 (done/pending) | Per-experiment |
| **Hydra** | Files (output dirs) | None built-in | None | ~4 (ephemeral) | Per-job |
| **MLflow** | DB or files (meta.yaml) | By run ID | None (manual cleanup) | 5 | Per-run |
| **W&B** | Cloud API | By run ID + mode | Heartbeat (server) | 6 | Per-run |
| **Ray Tune** | Files (JSON) | By experiment path | Implicit on restore | 5 | Per-trial + per-experiment |
| **Optuna** | DB (RDB/SQLite) | By study name | Heartbeat (configurable) | 5 | Per-trial |
| **Luigi** | Output targets | Output existence | Scheduler timeout | 4–8 | Per-task |
| **Airflow** | Database | By DAG run + task state | Heartbeat (zombie detection) | 13 | Per-task-instance |
| **Make** | None (filesystem timestamps) | Timestamp comparison | None | 2 | Per-target |
| **Bazel** | Action Cache + CAS | Content hash lookup | Cache eviction | 2 | Per-action |

---

## Recommendation

### State count: 3 states is well-supported

Our proposed reduction from 6 to 3 states (INITIALISING, MEASURING, DONE) aligns with the peer evidence. Most tools that actually execute work use 4-6 states, but the extra states serve purposes we do not need:

- SCHEDULED/PENDING/WAITING (MLflow, W&B, Optuna, Airflow) — for queued/deferred execution. We run experiments sequentially; no queue needed.
- PAUSED (Ray Tune) — for population-based training. Not relevant.
- PRUNED (Optuna) — for early stopping. Not relevant.
- KILLED (MLflow, W&B) — for explicit user cancellation. Our DONE + `failed=True` + `error_message` covers this.

The critical distinction is between "not done" and "done + succeeded vs failed". A `failed: bool` field on the DONE state (as proposed in N-X10) is cleaner than separate FAILED/INTERRUPTED states. W&B's distinction between Failed (exit code) and Crashed (heartbeat) is useful but can be captured in the error message rather than separate states.

### Config hash for resume: validated by Bazel and vLLM

Our `compute_config_hash()` approach is directly analogous to Bazel's action cache keying. Bazel proves this pattern works at enormous scale. vLLM sweep uses a simpler variant (output file existence per parameter combination) that achieves the same goal.

**Retain `compute_config_hash()`** — it is the right pattern. Ensure the hash includes all fields that affect measurement results (model, backend, all backend-specific parameters, precision, batch size, n, prompt source) and excludes all fields that do not (experiment ID, timestamps, output paths, metadata).

### Stale detection: PID check is adequate for our scope

The three approaches to stale detection in the survey:
1. **None** (Make, Hydra, vLLM, MLflow) — orphaned runs stay orphaned. Manual cleanup.
2. **PID check** (our v1.x) — check if the process is still alive. Local-only.
3. **Heartbeat** (W&B, Optuna, Airflow) — periodic signal to a central monitor. Works across machines.

Heartbeat is the most robust but requires either a server (W&B, Airflow) or a database (Optuna). For a local CLI tool, PID check is the right trade-off. It handles the common case (process killed, machine rebooted) and is zero-infrastructure.

**Retain `cleanup_stale()` with PID check.** It is simpler than heartbeat and sufficient for a single-machine tool. If we later add distributed execution (Docker multi-backend), we would need to revisit — but that is a future concern, not v2.0.

### Atomic writes: universal pattern, retain

Every tool that writes state files uses some form of atomic write (temp-then-rename). This is not optional infrastructure — it prevents corrupt state files. **Retain the atomic save pattern.**

### State file vs database: files are correct for our scope

Tools that use databases (Optuna, Airflow, MLflow) do so because they need concurrent access from multiple processes/machines. Our runner is single-process, single-machine. File-based state (JSON in `.state/`) is the right choice. vLLM sweep and Ray Tune both use file-based state successfully.

### StateManager and find_by_config_hash: retain both

`StateManager` provides load/save/find/cleanup — all needed infrastructure. `find_by_config_hash()` is the resume mechanism that makes `llem run study.yaml` idempotent after a crash. No peer tool provides this exact capability (most require explicit run IDs), which makes it a differentiator worth keeping.

### Summary of preservation recommendations

| Component | Recommendation | Peer justification |
|-----------|---------------|-------------------|
| 3-state enum | **Keep** (INITIALISING, MEASURING, DONE) | 4-6 is the norm; extra states serve needs we lack |
| `failed: bool` + `error_message` on DONE | **Keep** | W&B separates Failed/Crashed but most tools use a single terminal failure concept |
| `compute_config_hash()` | **Keep** | Bazel action cache validates content-addressed keying at scale |
| `StateManager` | **Keep** | File-based state is standard for single-machine tools (Ray Tune, vLLM) |
| `find_by_config_hash()` | **Keep** | Unique capability; closest peer is Bazel action cache lookup |
| `cleanup_stale()` (PID check) | **Keep** | Adequate for local execution; heartbeat only needed for distributed (Optuna, Airflow) |
| Atomic save (temp-then-rename) | **Keep** | Universal pattern across all tools that persist state |
| `ProcessProgress` / per-process tracking | **Drop** | v2.0 is single-process-per-experiment; no peer tool tracks sub-process state at this granularity |
| `completed_runs` / `failed_runs` / `total_runs` | **Drop** | Legacy batch tracking; study-level tracking belongs in `StudyManifest`, not `ExperimentState` |
