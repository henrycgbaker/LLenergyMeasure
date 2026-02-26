# Peer Research: Process Completeness Validation
> Generated 2026-02-26. Peer evidence for preservation audit item P-09.

## Evidence Per Tool

### 1. PyTorch Distributed (`torch.distributed`)

**Mechanism**: Collective barriers with timeouts.

PyTorch provides two synchronisation primitives relevant to completeness detection:

- **`barrier()`** — blocks every process until all ranks reach the call. If any process dies or hangs, the remaining processes block indefinitely until a backend-level timeout fires (default 10 min for NCCL, 30 min for Gloo). Barrier does not report *which* rank failed; it simply times out. A known footgun: calling `barrier()` under a conditional path that not all ranks enter causes a silent deadlock.

- **`monitored_barrier()`** (Gloo only) — rank 0 performs point-to-point send/recv with every other rank and actively reports which ranks did not respond within the timeout. The `wait_all_ranks=True` parameter collects *all* failed ranks before raising, rather than failing fast on the first. This is the closest PyTorch has to an explicit completeness check.

**Elastic Agent (`torchrun`)** adds a higher-level layer: an agent per node monitors local worker processes. When any worker fails, the agent kills *all* local workers, initiates a new rendezvous, and restarts everyone (up to `max_restarts`). If the agent itself enters an `UNKNOWN` state (partially applied action), it is expected to self-terminate rather than attempt recovery. Partial results are never accepted; the entire group restarts from the last checkpoint.

| Property | Value |
|----------|-------|
| Validates all completed before aggregating? | No aggregation concept; barriers gate forward progress |
| Mechanism | Collective barriers + timeouts; monitored_barrier reports failed ranks |
| Partial failure behaviour | Deadlock/timeout (barrier) or explicit error (monitored_barrier); elastic agent kills all + restarts |
| Strict vs lenient mode? | `wait_all_ranks` (True = collect all failures, False = fail-fast on first) |

**Sources**: [PyTorch Distributed docs](https://docs.pytorch.org/docs/stable/distributed.html), [Elastic Agent docs](https://docs.pytorch.org/docs/stable/elastic/agent.html), [torchrun docs](https://docs.pytorch.org/docs/stable/elastic/run.html)

---

### 2. HuggingFace Accelerate

**Mechanism**: `wait_for_everyone()` barrier + `gather_for_metrics()` with deduplication.

Accelerate wraps `torch.distributed` and adds two relevant operations:

- **`wait_for_everyone()`** — a barrier that blocks all processes until everyone arrives. Used before save/load operations. If any process diverges (e.g. early stopping on one GPU but not another), the others hang indefinitely until the collective timeout fires.

- **`gather_for_metrics()`** — gathers predictions/targets from all ranks and automatically removes duplicates introduced by padding the last batch for even distribution. This is the closest Accelerate has to a "validate completeness before aggregating" step: it ensures the gathered tensor has exactly the right number of samples, not more (from padding) and not fewer (from missing ranks).

Accelerate does not write marker files, check process indices, or provide a strict/non-strict mode. It trusts that `torch.distributed` collectives will either succeed (all ranks participate) or timeout.

| Property | Value |
|----------|-------|
| Validates all completed before aggregating? | Implicitly via collective gather; deduplicates padding artefacts |
| Mechanism | `torch.distributed` barrier + gather with padding removal |
| Partial failure behaviour | Hang until timeout → crash |
| Strict vs lenient mode? | No |

**Sources**: [Accelerate Accelerator API](https://huggingface.co/docs/accelerate/package_reference/accelerator), [Deferring Execution guide](https://huggingface.co/docs/accelerate/concept_guides/deferring_execution), [multi_process_metrics.py example](https://github.com/huggingface/accelerate/blob/main/examples/by_feature/multi_process_metrics.py)

---

### 3. DeepSpeed

**Mechanism**: Implicit all-rank participation in collective operations; checkpoint barriers.

DeepSpeed relies on NCCL/Gloo collectives for synchronisation and does not implement an explicit completeness validation layer. Key behaviours:

- **Checkpoint save/load**: All ranks *must* call save/load — not just rank 0 — because each rank holds its own optimizer state (ZeRO partitioning). If rank 0 calls `save_checkpoint()` alone, it hangs waiting for the collective. This is an implicit all-or-nothing barrier, but it detects failure via deadlock, not via an explicit health check.

- **Elastic training**: DeepSpeed supports elastic GPU counts via Universal Checkpoints (converting ZeRO-2/3 checkpoints for different parallelism configs). The elasticity config (`min_gpus`, `max_gpus`, `micro_batch_sizes`) defines the acceptable range of workers but does not provide worker health monitoring — that is delegated to the launcher (`torchrun`/`torch.distributed.elastic`).

- **Failure mode**: A `ProcessGroupNCCL` watchdog fires after a timeout (default 1800s) if any collective operation stalls. The result is a process crash, not a graceful degradation.

DeepSpeed has no equivalent of completion markers, completeness reports, or strict vs non-strict aggregation modes.

| Property | Value |
|----------|-------|
| Validates all completed before aggregating? | No explicit validation; relies on collective implicit barriers |
| Mechanism | NCCL collective all-or-nothing participation; checkpoint barrier |
| Partial failure behaviour | Deadlock → watchdog timeout → crash |
| Strict vs lenient mode? | No |

**Sources**: [DeepSpeed Checkpointing docs](https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html), [DeepSpeed Config JSON](https://www.deepspeed.ai/docs/config-json/), [Training API](https://deepspeed.readthedocs.io/en/latest/training.html)

---

### 4. Ray

**Mechanism**: Task-level exception wrapping, automatic retry, lineage reconstruction.

Ray is the only tool in this set that has a fully explicit, per-task completeness model with configurable retry and a clear partial-failure policy:

- **Failure detection**: When a task fails, Ray wraps the exception in a `RayTaskError` and stores it as the task's return value. Calling `ray.get()` on a failed task raises the wrapped exception. For actors, a `RayActorError` is raised.

- **Automatic retry**: `max_retries` (default 3 for tasks, 0 for actors). Set to `-1` for infinite retries. `retry_exceptions` can be configured to retry only specific exception types.

- **Lineage reconstruction**: If a completed result is lost (e.g. node failure after task finished), Ray re-executes the task's lineage to reconstruct the object, first checking other nodes for replicas.

- **No partial aggregation**: Ray does not aggregate partial results. A task either succeeds (result available), fails (exception raised), or is retried. There is no "best-effort" assembly of partial outcomes. The caller is responsible for handling `RayTaskError` exceptions.

- **Actor supervision**: Actors can be restarted up to `max_restarts` times. Tasks submitted to dead actors raise `RayActorError`. With `max_task_retries > 0`, actor tasks get at-least-once semantics.

| Property | Value |
|----------|-------|
| Validates all completed before aggregating? | Yes — every task has an explicit success/failure status; caller gates on `ray.get()` |
| Mechanism | Exception wrapping (`RayTaskError`), automatic retry, lineage reconstruction |
| Partial failure behaviour | Hard error (exception raised to caller); retries before surfacing |
| Strict vs lenient mode? | `max_retries=0` (strict, no retry) vs `max_retries=N/-1` (retry before failing) |

**Sources**: [Ray Task Fault Tolerance](https://docs.ray.io/en/latest/ray-core/fault_tolerance/tasks.html), [Ray Actor Fault Tolerance](https://docs.ray.io/en/latest/ray-core/fault_tolerance/actors.html), [Ray Object Fault Tolerance](https://docs.ray.io/en/latest/ray-core/fault_tolerance/objects.html)

---

### 5. MLPerf LoadGen

**Mechanism**: Post-run log validation with multi-criteria VALID/INVALID determination.

MLPerf LoadGen is the most directly analogous system to our P-09 implementation. It performs structured post-run validation of benchmark completeness before results can be submitted:

- **Minimum query count**: Each scenario has a required minimum (e.g. 24,576 for language models). Runs with fewer queries are marked `INVALID`.

- **Minimum duration**: 600 seconds for most scenarios. Runs shorter than this are `INVALID`.

- **Early stopping**: A statistical mechanism that allows runs with fewer queries to be valid, but applies a penalty (the computed percentile is slightly higher than the target). If early stopping criteria are not met, the result is `INVALID`.

- **Accuracy validation**: A separate test run uses each sample in the test library exactly once. Accuracy must meet benchmark-specific quality targets (e.g. 99% of FP32 baseline).

- **Submission checker** (`submission_checker.py`): A post-hoc automated tool that validates the entire submission package — checking log file presence, performance metrics, accuracy results, and compliance test outputs. Results that cannot be replicated are declared invalid.

- **Compliance audit**: Up to two submissions per round are audited by humans with hardware access over two days.

The key design: LoadGen records everything during the run (all traffic generated and received) for *later* analysis and verification. Validity is determined post-hoc from logs, not via real-time barriers.

| Property | Value |
|----------|-------|
| Validates all completed before aggregating? | Yes — multi-criteria post-run validation; results labelled VALID or INVALID |
| Mechanism | Log analysis: min queries, min duration, early stopping, accuracy check, submission checker |
| Partial failure behaviour | Hard INVALID — no partial acceptance; entire run must be redone |
| Strict vs lenient mode? | Two tiers: "Closed" (strict, all rules) vs "Open" (relaxed constraints on model modifications); but completeness checks apply to both |

**Sources**: [MLPerf Inference Rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc), [Submission Guide](https://docs.mlcommons.org/inference/submission/), [LoadGen README](https://github.com/mlcommons/inference/blob/master/loadgen/README.md)

---

### 6. Hydra

**Mechanism**: Per-job `JobReturn` dataclass with status enum; best-effort continuation.

Hydra's multirun mode executes a sweep of jobs (parameter combinations) and provides limited completeness tracking:

- **`JobReturn` dataclass**: Each completed job produces a `JobReturn` with a `status` field (`JobStatus.COMPLETED` or `JobStatus.FAILED`). The result is pickled to `.hydra/job_return.pickle` per job output directory.

- **Failure handling**: By default, Hydra continues executing remaining jobs even if earlier ones fail. Exceptions from failed jobs are only printed *after all runs complete*. There is a known issue ([#2284](https://github.com/facebookresearch/hydra/issues/2284)) that multirun does not clearly notify the user about exceptions during execution.

- **No completeness validation**: Hydra does not validate that all jobs in a sweep completed before producing aggregate output — because Hydra does not aggregate results at all. Each job writes independently to its own subdirectory (`multirun/<date>/<time>/<job_number>/`). The user is responsible for collecting and analysing outputs.

- **Callback hooks**: `on_multirun_end` fires after all jobs finish (regardless of individual success/failure), and `on_job_end` fires per job with a `JobReturn` parameter. These can be used to build custom completeness validation, but Hydra does not provide one.

| Property | Value |
|----------|-------|
| Validates all completed before aggregating? | No — no built-in aggregation; no sweep-level completeness check |
| Mechanism | Per-job `JobReturn` with status enum; callbacks for custom handling |
| Partial failure behaviour | Best-effort: remaining jobs continue; failures reported at end |
| Strict vs lenient mode? | No built-in toggle; behaviour is always best-effort |

**Sources**: [Hydra Multi-run docs](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/), [Hydra Callbacks](https://hydra.cc/docs/1.2/experimental/callbacks/), [Bug #2284 — multirun exception notification](https://github.com/facebookresearch/hydra/issues/2284)

---

### 7. Apache Spark / Dask

#### Apache Spark

**Mechanism**: DAGScheduler lineage tracking + per-task retry with configurable failure threshold.

Spark has the most mature distributed completeness model of any tool in this set:

- **Task retry**: Each task can fail up to `spark.task.maxFailures` (default 4) before the entire stage is aborted. Failures are tracked per task, not per executor — if three different executors die while running the same task, that counts as three failures for that task.

- **Stage abort**: When `maxFailures` is exceeded for any task, the `TaskSetManager` aborts the stage with an explicit error: `"Task [id] in stage [id] failed [maxTaskFailures] times; aborting job"`. Partial stage results are never aggregated.

- **Lineage reconstruction**: If an executor is lost after tasks completed, Spark recomputes the lost partitions from the DAG lineage. For `ShuffleMapStage`, all successfully-completed tasks on the failed executor are re-enqueued.

- **Non-determinism guard**: Newer Spark versions detect when a non-deterministic stage is partially recomputed and raise an exception rather than silently mixing results from different task attempts — this directly parallels our concern about partial results being silently aggregated.

- **Speculative execution**: Spark may run duplicate copies of slow tasks. If a speculative copy and the original both complete, only one result is kept. If a speculative copy is killed because the original finished, the kill is not counted as a failure.

| Property | Value |
|----------|-------|
| Validates all completed before aggregating? | Yes — all tasks in a stage must succeed; partial stages abort |
| Mechanism | Per-task failure counter, `maxFailures` threshold, DAG lineage recomputation |
| Partial failure behaviour | Retry up to threshold → abort stage → fail job |
| Strict vs lenient mode? | `spark.task.maxFailures` (default 4) — lower = stricter; 1 = fail on first error |

**Sources**: [Spark DAGScheduler internals](https://books.japila.pl/apache-spark-internals/scheduler/DAGScheduler/), [TaskSetManager internals](https://books.japila.pl/apache-spark-internals/scheduler/TaskSetManager/), [SPARK-51756 — non-deterministic partial retry](https://issues.apache.org/jira/browse/SPARK-51756)

#### Dask

**Mechanism**: Heartbeat-based worker monitoring + scheduler lineage recomputation.

- **Worker monitoring**: The scheduler detects worker death via missing heartbeats (timeout ~3 seconds). A "nanny" process per worker monitors memory usage and can restart the worker if it exceeds thresholds.

- **Result recomputation**: When a worker dies, the scheduler maintains a full history of how each result was produced and recomputes lost results on surviving workers. This is automatic and transparent to the user.

- **`allowed-failures`**: The config key `distributed.scheduler.allowed-failures` (default 3) controls how many workers can die running a task before the task is marked "bad" and a `KilledWorker` exception is raised. This is the closest Dask has to a strict/lenient mode.

- **No partial results**: A task either succeeds or fails. There is no mechanism to accept partial output from a killed worker. However, data sent via `scatter()` is not tracked by the scheduler and cannot be recomputed if lost — a known limitation.

| Property | Value |
|----------|-------|
| Validates all completed before aggregating? | Yes — implicitly via task state machine; all futures must resolve |
| Mechanism | Heartbeat monitoring, scheduler lineage, nanny process |
| Partial failure behaviour | Recompute on surviving workers → eventually `KilledWorker` if all retries exhausted |
| Strict vs lenient mode? | `allowed-failures` config (default 3); 0 = fail on first worker death |

**Sources**: [Dask Resilience docs](https://distributed.dask.org/en/stable/resilience.html), [Worker docs](https://distributed.dask.org/en/stable/worker.html), [allowed-failures issue #6078](https://github.com/dask/distributed/issues/6078)

---

### 8. vLLM

**Mechanism**: Minimal — largely delegated to Ray or multiprocessing; active health check is a known gap.

vLLM's multi-GPU handling is the least mature of the tools surveyed for completeness validation:

- **Multiprocessing executor**: The V1 `MultiprocExecutor` spawns worker subprocesses for tensor-parallel inference. However, the built-in `check_health()` method is effectively a no-op — it returns immediately and continues sending RPCs to a dead socket without errors ([Bug #19849](https://github.com/vllm-project/vllm/issues/19849)). If the EngineCore subprocess crashes (OOM, segfault), the parent process does not notice; it silently hangs.

- **Ray executor**: When using Ray as the distributed backend, worker failure detection is delegated to Ray's actor supervision. The `RayDistributedExecutor` can detect worker death via `RayActorError` and shut down the engine.

- **Termination management**: The multiproc executor has logic to *terminate* workers (send SIGTERM, wait, then SIGKILL), but this is for intentional shutdown, not failure detection.

- **No completeness validation for inference results**: vLLM does not validate that all tensor-parallel workers contributed to a response before returning it. The NCCL allreduce/allgather operations that combine partial results from workers are implicit — if a worker is dead, the collective hangs.

| Property | Value |
|----------|-------|
| Validates all completed before aggregating? | No — relies on NCCL collective semantics (hang if worker missing) |
| Mechanism | Delegated to Ray (if used); multiprocessing health check is a no-op |
| Partial failure behaviour | Silent hang (multiprocessing) or `RayActorError` (Ray backend) |
| Strict vs lenient mode? | No |

**Sources**: [vLLM MultiprocExecutor API](https://docs.vllm.ai/en/latest/api/vllm/v1/executor/multiproc_executor/), [Bug #19849 — health check no-op](https://github.com/vllm-project/vllm/issues/19849), [vLLM Multiprocessing design](https://docs.vllm.ai/en/stable/design/multiprocessing/)

---

## Summary Table

| Tool | Validates completeness? | Mechanism | Partial failure | Strict/lenient toggle |
|------|------------------------|-----------|-----------------|----------------------|
| **PyTorch Distributed** | Implicit (barriers) | `barrier()` / `monitored_barrier()` with timeout | Hang → timeout → crash; elastic agent kills all + restarts | `wait_all_ranks` on monitored_barrier |
| **Accelerate** | Implicit (gather) | `wait_for_everyone()` + `gather_for_metrics()` dedup | Hang → timeout → crash | No |
| **DeepSpeed** | No | Collective implicit barriers; checkpoint all-rank requirement | Hang → watchdog timeout → crash | No |
| **Ray** | **Yes — explicit** | `RayTaskError` wrapping, `max_retries`, lineage reconstruction | Hard error after retries exhausted | `max_retries` (0 = strict, N = retry, -1 = infinite) |
| **MLPerf LoadGen** | **Yes — explicit** | Post-run log validation: min queries, min duration, accuracy, submission checker | Hard INVALID; entire run redone | Closed (strict) vs Open (relaxed model rules, same completeness) |
| **Hydra** | No | Per-job `JobReturn` status; no sweep-level check | Best-effort continuation; failures reported post-sweep | No |
| **Spark** | **Yes — explicit** | Per-task failure counter, `maxFailures`, DAG lineage recompute | Retry → abort stage → fail job | `maxFailures` threshold (default 4) |
| **Dask** | **Yes — implicit** | Heartbeat monitoring, scheduler lineage, `allowed-failures` | Recompute → `KilledWorker` after retries | `allowed-failures` (default 3) |
| **vLLM** | No | Delegated to Ray/NCCL; multiprocessing health check is a no-op | Silent hang or `RayActorError` | No |

---

## Recommendation

Our v1.x implementation (`.completed_N` markers + 4-check `CompletenessReport` + strict/non-strict mode) is **well-aligned with industry practice and in some respects ahead of it**. Key observations:

1. **Explicit post-run validation is the right pattern.** The tools that handle completeness best — MLPerf LoadGen, Ray, and Spark — all use explicit, structured validation rather than relying on implicit barriers. Our marker-file + multi-check approach is closest to MLPerf's log-based post-hoc validation model, which is appropriate because we are a benchmarking tool, not a training framework.

2. **Most training frameworks (PyTorch, Accelerate, DeepSpeed) rely on barriers that hang on failure.** They detect incompleteness via timeout/deadlock, not via structured post-run checks. This is acceptable for synchronous collective operations but would be catastrophic for our use case where processes run independently and results are aggregated asynchronously.

3. **The strict/non-strict toggle is well-precedented.** Ray (`max_retries`), Spark (`maxFailures`), and Dask (`allowed-failures`) all provide configurable strictness. Our binary strict/non-strict is simpler but sufficient — we are not retrying failed processes, just deciding whether to aggregate partial data.

4. **Marker files are a pragmatic POSIX-level mechanism.** No peer tool uses exactly this pattern (most use in-memory state or collective operations), but our processes are POSIX subprocesses writing to a shared filesystem, not nodes in a distributed cluster with a message bus. Marker files are the correct primitive for our execution model. The closest analogue is MLPerf's log files that are validated post-hoc.

5. **The four-check design covers all failure modes.** Count match (process never started) + index contiguity (specific process failed) + no duplicates (race condition) + marker presence (crashed after partial write) is a complete set. No peer tool checks all four of these, but none of them need to — they operate in different execution models. For our POSIX-subprocess model, all four checks are necessary and sufficient.

**Preserve as-is for v2.0.** Document the marker-file protocol in `designs/experiment-isolation.md`. Ensure the orchestration layer (whoever replaces the v1.x process runner) is contractually required to write `.completed_N` after saving each raw result.
