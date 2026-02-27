# Pitfalls Research

**Domain:** Multi-experiment study execution — adding subprocess isolation, sweep grammar, and manifest checkpointing to an existing single-experiment LLM benchmarking tool
**Researched:** 2026-02-27
**Confidence:** HIGH for CUDA subprocess and IPC pitfalls (design documents + peer codebase verification); MEDIUM-HIGH for sweep explosion and thermal contamination; MEDIUM for manifest corruption patterns

---

> **Scope note:** This document covers pitfalls for the M2 milestone (adding `StudyRunner` and
> multi-experiment execution to an existing working M1 tool). Measurement methodology pitfalls
> (NVML accuracy, thermal floor calibration, FLOPs as a metric, bootstrap CIs) are covered in
> the pre-existing `.planning/research/PITFALLS.md` from the M1 research phase — not duplicated
> here. The two files are complementary.

---

## Critical Pitfalls

### CP-1: Using `fork` Instead of `spawn` for CUDA Subprocesses

**What goes wrong:**
Linux defaults to `fork` for `multiprocessing`. Forked children inherit the parent's CUDA driver state, including initialised contexts, memory allocations, and CUDA runtime handles. This produces either a hard crash (`RuntimeError: Cannot re-initialize CUDA in forked subprocess`) or, more dangerously, silent incorrect measurements where the child appears to run successfully but the CUDA context is in an undefined state.

The silent failure case is the dangerous one. The child process starts, loads the model, runs inference, and sends results back — but the underlying CUDA context is corrupted. Energy readings and latency measurements are invalid with no error signal.

**Why it happens:**
Linux's `fork` is so fast and familiar that developers reach for it instinctively. The global `multiprocessing.set_start_method('spawn')` fix is mentioned in PyTorch docs, but calling it globally breaks other libraries. The scoped `get_context('spawn')` pattern is less well known.

**How to avoid:**
Use `multiprocessing.get_context("spawn")` scoped to `StudyRunner` only — never call `set_start_method()` globally:

```python
# Correct: scoped to StudyRunner, does not affect other code
mp_ctx = multiprocessing.get_context("spawn")
p = mp_ctx.Process(target=_run_experiment_worker, args=(config, conn, queue))
```

Never:
```python
# Wrong: global mutation — breaks any other library using multiprocessing
multiprocessing.set_start_method("spawn")
```

**Warning signs:**
- `RuntimeError: Cannot re-initialize CUDA in forked subprocess`
- Child process runs successfully but energy readings are systematically lower than expected
- CUDA errors only appear after the second or third experiment in a study (first experiment in a fork often works by accident because the parent CUDA context was not yet initialised when fork happened)
- `CUDA error: initialization error` in worker stderr

**Phase to address:** StudyRunner implementation (M2, study execution phase). Test by running a 3-experiment study immediately after the runner is wired up — confirm each child gets a clean CUDA context by checking NVML memory before model load.

---

### CP-2: Pipe Deadlock — Sender Blocked, Parent at `p.join()`

**What goes wrong:**
Python's `multiprocessing.Pipe()` uses an OS pipe with a kernel buffer of approximately 64KB on Linux. If the child sends data that exceeds this buffer before the parent reads it, the child's `conn.send()` call blocks indefinitely waiting for the parent to drain the pipe. The parent, blocked at `p.join()`, never reads from the pipe. Both processes deadlock.

This is not a theoretical edge case. It manifests when:
- An exception traceback references many deeply nested frames (long tracebacks can exceed 64KB)
- A future `ExperimentResult` adds new fields (e.g., per-token latency arrays, time-series energy data)
- The child sends an error dict containing a large model state or configuration dump

The particularly insidious aspect: the deadlock only appears for certain error types (those with long stack traces), making it look like an intermittent hang rather than a systematic bug.

**Why it happens:**
The Pipe buffer limit is not documented prominently in the Python `multiprocessing` docs. Developers testing with simple configs see it work; edge cases with large payloads only appear under specific failure conditions.

**How to avoid:**
Two complementary mitigations:

1. **File-based IPC fallback for large payloads** — detect at send time:
```python
FILE_BASED_THRESHOLD = 1_000_000  # 1MB — matches optimum-benchmark pattern

serialised = result.model_dump_json()
if len(serialised) > FILE_BASED_THRESHOLD:
    tmp = Path(tempfile.mkstemp(suffix=".json")[1])
    tmp.write_text(serialised)
    conn.send({"__file__": str(tmp)})
else:
    conn.send(result)
```

2. **Reader thread** (for the exception dict path): if the exception dict is unexpectedly large, drain the pipe in a background thread concurrently with `p.join()`. The design docs note this as a future option if the 1MB threshold proves insufficient.

For error dicts specifically, cap the traceback string length: `"traceback": traceback.format_exc()[:8000]` prevents runaway tracebacks from exceeding the buffer.

**Warning signs:**
- Study hangs indefinitely on a specific experiment that failed (not a timeout — never reaches it)
- `p.join(timeout=...)` returns `None` (timeout fired) but `p.is_alive()` is `True` and the child process shows in `ps` but is not consuming CPU (blocked on pipe write)
- Hang only occurs when a specific error type is triggered (e.g. OOM, which tends to produce long stack traces)

**Phase to address:** StudyRunner implementation (M2). Test with a deliberately large error payload in the child process to verify the fallback works before relying on the timeout as the safety net.

---

### CP-3: SIGINT (Ctrl+C) Does Not Propagate Cleanly to Subprocess Tree

**What goes wrong:**
When the user presses `Ctrl+C` during a study, the terminal sends SIGINT to the foreground process group. If child processes are `daemon=False` (the correct setting for clean CUDA teardown), they are in the same process group and also receive SIGINT. What happens next is undefined:

- The child may be mid-CUDA-kernel and cannot handle the signal (CUDA signal handlers are not re-entrant)
- The child may partially write the manifest before dying, leaving it in a corrupt state
- The parent receives `KeyboardInterrupt` at `p.join()`, which propagates up the call stack — skipping the `progress_queue.put(None)` sentinel and consumer thread cleanup
- The Rich progress display may not clean up, leaving the terminal in a broken state

The most serious consequence: the manifest shows the experiment as `"running"` when in fact it was aborted mid-run. The resume logic (later milestone) will try to re-run it, but the result file referenced in adjacent experiments may be inconsistent.

**Why it happens:**
The interaction between Python signal handling, the multiprocessing process group, and CUDA's own signal handling is complex. Each layer (Python, CUDA runtime, the backend library) has its own signal handling assumptions that conflict under SIGINT.

**How to avoid:**
Wrap the inner study loop in a signal handler that:

1. Catches `KeyboardInterrupt` in the parent
2. Calls `p.kill()` (SIGKILL — not SIGTERM, which may be ignored by a CUDA-blocked child)
3. Sends the `None` sentinel to the progress queue
4. Waits for the consumer thread to exit
5. Marks the current manifest entry as `"failed"` with `error_type="UserInterrupted"`
6. Writes the manifest to disk before re-raising

```python
try:
    p.start()
    child_conn.close()
    p.join(timeout=timeout)
except KeyboardInterrupt:
    if p.is_alive():
        p.kill()
        p.join()
    progress_queue.put(None)  # sentinel always sent
    consumer.join()
    manifest.mark_failed(config_hash, cycle, StudyFailed(
        config=config.model_dump(),
        exception_type="UserInterrupted",
        error_message="Study interrupted by user (Ctrl+C)",
    ))
    raise  # re-raise to exit the study loop cleanly
```

Additionally, use `p.start()` inside the context, not before the try block — ensures the cleanup path is always reached.

**Warning signs:**
- `Ctrl+C` leaves a zombie child process visible in `ps aux`
- GPU memory remains allocated after the study exits (NVIDIA SMI shows model still in VRAM)
- Terminal has broken cursor state or missing newline after `Ctrl+C`
- `study_manifest.json` shows experiment as `"running"` after the tool exits

**Phase to address:** StudyRunner implementation (M2). Test `Ctrl+C` during the second experiment of a 3-experiment study. Confirm: manifest shows correct statuses, GPU VRAM is released, terminal is clean.

---

### CP-4: `daemon=True` on Child Processes Causes Dirty CUDA Teardown

**What goes wrong:**
Setting `daemon=True` on child processes (which is NOT the design choice, but a tempting simplification) means the child is killed immediately when the parent exits. For TensorRT-LLM specifically, this leaves GPU memory reservations orphaned — device memory is held until the next process that initialises CUDA on that GPU. On shared machines, this affects other users. Even for PyTorch and vLLM, abrupt CUDA teardown without calling `torch.cuda.empty_cache()` and letting the Python GC clean up model objects can leave the GPU in a state where the next CUDA initialisation is slower than normal.

**Why it happens:**
`daemon=True` is convenient: the parent does not need to call `p.join()` and child processes are guaranteed to not outlive the parent. Developers use it to avoid orphan processes.

**How to avoid:**
Use `daemon=False` (the default). `StudyRunner` always calls `p.join()` before returning — orphan processes do not occur in normal operation. For the abnormal case (parent crash without calling `p.join()`), the child becomes a genuine orphan, but it will complete its experiment and exit cleanly, releasing CUDA resources. This is the correct trade-off.

The design documentation (`designs/experiment-isolation.md`) documents this choice explicitly. Do not change it in implementation.

**Warning signs:**
- GPU VRAM not released after study exits (check `nvidia-smi`)
- Subsequent `llem run` invocations are slower than expected to initialise CUDA
- TRT-LLM engine compilation step is triggered again when it should have been cached (indicates the previous engine process did not complete its cleanup and left cache in an inconsistent state)

**Phase to address:** StudyRunner implementation (M2). Default is correct — pitfall is regressing to `daemon=True` during debugging. Add a comment in `StudyRunner` code explaining why `daemon=False` is required.

---

### CP-5: Sweep Grid Combinatorial Explosion Without Pre-Flight Warning

**What goes wrong:**
A researcher adds multiple sweep dimensions without realising the grid grows multiplicatively. The study YAML looks innocuous:

```yaml
sweep:
  precision: [fp16, bf16, int8]          # 3
  batch_size: [1, 2, 4, 8, 16, 32, 64]  # 7
  pytorch.num_threads: [1, 2, 4, 8]      # 4
  max_new_tokens: [50, 100, 200, 500]    # 4
```

This generates 3 × 7 × 4 × 4 = **336 experiments**. With `n_cycles: 3`, that is 1,008 individual subprocess invocations. With a 60-second thermal gap between each, the study takes 17+ hours. The tool silently starts running.

Worse: if a new sweep dimension is added by mistake (e.g., `vllm.max_num_seqs` accidentally added to a PyTorch-only study), it passes Pydantic validation (because `vllm` is a valid backend), creates a second per-backend grid, and doubles the experiment count unexpectedly.

**Why it happens:**
Researchers think of sweeps as "a few values per dimension" without multiplying. The grid expansion is implicit and non-obvious from the YAML alone.

**How to avoid:**
Show a pre-flight summary before any experiment runs:

```
Study: batch-size-sweep (336 experiments × 3 cycles = 1,008 runs)
  Estimated wall time: ~17h 30min (at 60s gap + ~2min per experiment)
  GPU memory check: all configs within available VRAM

  Skipping 0 invalid combinations.

  Continue? [y/N]
```

For non-interactive use (`--yes` / `-y` flag, or TTY detection), skip the prompt but always print the summary. The experiment count and estimated duration must always be visible before the first subprocess spawns.

Additionally, add a hard cap with a clear error:
```python
MAX_EXPERIMENTS_WITHOUT_CONFIRMATION = 100  # configurable in user config

if len(experiments) > MAX_EXPERIMENTS_WITHOUT_CONFIRMATION and not confirmed:
    raise ConfigError(
        f"Study would run {len(experiments)} experiments × {n_cycles} cycles = "
        f"{len(experiments) * n_cycles} total runs. "
        f"Pass --yes to confirm, or reduce the sweep grid."
    )
```

**Warning signs:**
- Study YAML has 3+ sweep dimensions with more than 3 values each
- `backend:` is a list combined with sweep dimensions (multiplies across backends)
- `n_cycles` > 1 combined with a large grid (n_cycles multiplies total runs)

**Phase to address:** Sweep grid expansion (`expand_grid()` function, M2). The pre-flight summary must be implemented alongside the grid expander — not deferred.

---

### CP-6: Manifest Corruption from Partial Writes During Interruption

**What goes wrong:**
`ManifestWriter._write()` calls `self.path.write_text(self.manifest.model_dump_json(indent=2))`. On Linux, `write_text()` is not atomic — it opens the file, truncates it, then writes. If the process is killed (SIGKILL, OOM killer, power failure) between the truncate and the completed write, the manifest file is either empty or contains partial JSON. When the resume logic (later milestone) reads the manifest, it gets a JSON parse error and cannot resume.

A secondary corruption vector: if two processes somehow both call `_write()` simultaneously (not expected in the current single-writer design, but possible in future parallel extensions), concurrent writes to the same file without locking corrupt it.

**Why it happens:**
`pathlib.Path.write_text()` is convenient but not atomic. Most developers do not think about the truncate-then-write window.

**How to avoid:**
Use atomic write-then-rename:

```python
def _write(self) -> None:
    content = self.manifest.model_dump_json(indent=2)
    tmp_path = self.path.with_suffix(".tmp")
    tmp_path.write_text(content)
    tmp_path.replace(self.path)  # atomic on POSIX (same filesystem)
```

`Path.replace()` is a POSIX rename, which is atomic on the same filesystem. The manifest is either the old version or the new version — never partial. The `.tmp` file is cleaned up automatically if the process exits after writing but before the rename (it is simply ignored on next startup, not mistaken for a valid manifest).

For the read side (resume logic), always validate the JSON before trusting it:
```python
try:
    manifest = StudyManifest.model_validate_json(path.read_text())
except (json.JSONDecodeError, pydantic.ValidationError) as e:
    raise StudyError(f"Manifest at {path} is corrupt: {e}") from e
```

**Warning signs:**
- `study_manifest.json` is 0 bytes after an interrupted study
- `json.JSONDecodeError` when attempting to resume
- Manifest file modification time is very close to the time the study was killed (partial write window)

**Phase to address:** `ManifestWriter` implementation (M2). Atomic write is two lines instead of one — use it from the start. Do not defer this as "we'll fix it when we implement resume."

---

### CP-7: Timeout Too Conservative Causes Spurious Experiment Failures

**What goes wrong:**
The timeout formula `max(config.n * 2, 600)` gives a minimum of 10 minutes regardless of the model or hardware. For a large model (Llama-3.1-70B) on a single GPU, model loading alone can take 5-8 minutes. If `n=50` (100-second estimate), the total timeout is 600 seconds. A 70B model loading at 6 minutes (360 seconds) leaves only 240 seconds for inference of 50 prompts. At typical 70B throughput (5-10 tokens/sec per prompt, 100 tokens output), 50 prompts × 100 tokens = 5,000 tokens / 7.5 tok/s = ~667 seconds. The experiment times out and is recorded as failed, with no result, when it would have completed given another 200 seconds.

**Why it happens:**
The timeout estimate of "2 seconds per prompt" is calibrated for 7B-13B models at batch_size=1 on an A100. It does not account for model loading time or larger models. The minimum of 600 seconds seems generous but is not.

**How to avoid:**
Make the timeout formula model-aware. Use a rough parameter count estimate from the model name or config:

```python
def _calculate_timeout(config: ExperimentConfig) -> int:
    # Generous base for model loading (larger models take longer)
    # No reliable way to know param count without downloading; use heuristic from name
    loading_seconds = 600  # minimum 10 min — covers 7B models
    if "70b" in config.model.lower() or "65b" in config.model.lower():
        loading_seconds = 1800  # 30 min for 70B
    elif "30b" in config.model.lower():
        loading_seconds = 1200  # 20 min

    # Inference time: 10s per prompt is very conservative
    inference_seconds = config.n * 10

    return loading_seconds + inference_seconds
```

Alternatively, make timeout configurable in `execution:` block (advanced users can override):
```yaml
execution:
  experiment_timeout_seconds: 3600  # override for large models
```

The cost of a spurious timeout is an experiment failure that must be re-run manually. The cost of no timeout is an indefinitely blocked study. Err on the side of generosity.

**Warning signs:**
- Studies with 70B+ parameter models show `TimeoutError` for experiments that should succeed
- Study completes with all experiments as `"failed"` with `TimeoutError` on a machine with sufficient VRAM
- Timeout fires shortly after `"started"` event is received from the worker (model loading was the bottleneck)

**Phase to address:** StudyRunner implementation (M2). The timeout formula must be reviewed before the first large-model study is run.

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| `multiprocessing.set_start_method("spawn")` globally | Simpler than `get_context` | Breaks other libraries using multiprocessing (torch.distributed, Zeus's own internals) | Never — always use `get_context` |
| Sending full `ExperimentResult` through Pipe without size check | Simpler code | Deadlock when result exceeds 64KB OS buffer — silent hang | Never — always check size |
| `path.write_text()` for manifest (non-atomic) | One line | Corrupt manifest on SIGKILL during write | Never for checkpoint files — always use atomic rename |
| `daemon=True` on worker processes | Automatic cleanup if parent crashes | Dirty CUDA teardown, orphaned GPU memory (especially TRT-LLM) | Never |
| Skipping the pre-flight experiment count display | Simpler implementation | Researcher unknowingly starts a 17-hour study | Acceptable only behind an explicit `--quiet` flag |
| Running `expand_grid()` at study execution time (not at YAML parse time) | Defers validation | Combinatorial explosion discovered after first model is loaded (wasted time, corrupt state if interrupted) | Never — expand at parse time per the design decision |
| Using `progress_queue.get()` with no timeout in the consumer thread | Simpler code | Consumer thread hangs forever if the sentinel is never sent (e.g. bug in SIGKILL path) | Never — use `queue.get(timeout=5)` with a loop |

---

## Integration Gotchas

Common mistakes when connecting the study runner to existing components.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| `ExperimentOrchestrator` in worker | Importing it at module level in worker file causes CUDA to initialise at import time on spawn | Import `ExperimentOrchestrator` inside `_run_experiment_worker()`, not at the module's top level |
| `ManifestWriter` and `StudyResult` | Conflating the two objects — merging their fields into one class | Keep them separate: manifest is a checkpoint (written incrementally), result is the final return value (written once at completion) |
| `expand_grid()` and Pydantic validation | Running `expand_grid()` after Pydantic validates the `StudyConfig` means invalid combos only discovered at runtime | Run `expand_grid()` during `StudyConfig.__init__` (or a pre-run validation step) so invalid combos are caught at YAML parse time with a clear error |
| Progress queue in worker | Worker starts sending events before the consumer thread is started | Start the consumer thread BEFORE calling `p.start()` — the worker can send events immediately on startup |
| `child_conn` in parent after `p.start()` | Parent holds both ends of the Pipe — parent's copy of child end keeps the pipe alive and may cause reads to block | Close `child_conn` in the parent immediately after `p.start()`: `child_conn.close()` |
| Thermal gap timing | Sleeping `config_gap_seconds` in the child process (inside the worker) rather than in the parent between experiments | Thermal gap must be in `StudyRunner` (parent), not in the worker. The child process is dead during the gap — sleeping there does nothing |
| NVML session ownership | Creating a new NVML session per experiment worker | Worker inherits no NVML state from parent (spawn). Worker must call `pynvml.nvmlInit()` at the start and `pynvml.nvmlShutdown()` at the end. No sharing across process boundary. |

---

## Performance Traps

Patterns that work at small scale but fail as study size grows.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Keeping completed `ExperimentResult` objects in parent memory during study | RAM grows linearly with experiment count; OOM on large studies | Write each result to disk immediately, store only the file path in `StudyResult.result_files` | Around 50-100 experiments if results contain time-series data |
| Manifest growing linearly in size | `model_dump_json()` serialises the entire manifest on every write; slows down as experiment count grows | Acceptable for v2.0 (studies are unlikely to exceed 500 experiments); for v3.0+, consider append-only manifest format | > 500 experiments (manifest approaches ~1MB; write latency becomes noticeable) |
| `p.join(timeout=timeout)` with no loop | One missed timeout leaves a zombie experiment and blocks the study | `p.join()` is correct for a single wait; just ensure `p.kill()` + `p.join()` follows immediately if `p.is_alive()` | N/A — design is correct; trap is implementing the join without the is_alive check |
| Starting all sweep experiments before any run (eager expansion) | Memory overhead for large grids; validation errors only appear at the end | Expand grid lazily (one row at a time) or eagerly but only hold one `ExperimentConfig` in memory at a time | > 10,000 experiments (rare but possible with 5+ sweep dimensions) |

---

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing critical pieces.

- [ ] **StudyRunner wired up:** confirm the worker uses `spawn` context, not the Linux default `fork` — check with `print(mp.current_process().daemon)` and `mp.get_start_method()` in the worker
- [ ] **SIGINT handled:** pressing Ctrl+C during experiment 2 of 3 should exit cleanly, leaving manifest with correct statuses (not all showing `"running"`)
- [ ] **Manifest atomic write:** verify by checking that `study_manifest.json` is never 0 bytes or partial JSON — kill the process mid-write and confirm the file is still valid
- [ ] **Pipe size safe:** test with a deliberately large error payload (e.g., an exception with a 100KB traceback) — confirm no hang
- [ ] **Pre-flight count displayed:** even for a 2-experiment study, the count and estimated time must be printed before anything runs
- [ ] **Thermal gap in parent, not worker:** inspect code to confirm `time.sleep(gap)` is called in `StudyRunner.run()`, not inside `_run_experiment_worker()`
- [ ] **Consumer thread sentinel always sent:** both the normal path AND the SIGKILL path must call `progress_queue.put(None)` — verify the finally/except blocks
- [ ] **Invalid combos skipped before GPU allocation:** Pydantic validation must run before any subprocess spawns, not inside the worker

---

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Fork instead of spawn (silent CUDA corruption) | HIGH | Stop the study, identify affected results by cross-checking energy readings (will be anomalously low or zero), delete and re-run the study from scratch after fixing the start method |
| Pipe deadlock (study hangs indefinitely) | MEDIUM | Kill the parent process (`kill -9 <pid>`), check manifest for last completed experiment, fix the payload size issue, re-run from last checkpoint (if resume is available) or from scratch |
| Sweep explosion (17-hour study started accidentally) | LOW | Kill the study (`Ctrl+C`), check manifest for completed experiments before interruption, run a filtered explicit `experiments:` list instead of the full grid |
| Manifest corruption (corrupt JSON) | LOW | The result files (per-experiment JSONs) are unaffected. Reconstruct the manifest by scanning the results directory for completed JSON files — their filenames contain the `config_hash`. Implement a `llem repair-manifest <results-dir>` command for this |
| Timeout miscalibration (spurious failures) | LOW | Check `ExperimentResult.duration_seconds` for successful experiments to calibrate; re-run failed experiments with `--experiment-timeout <seconds>` override |
| Dirty CUDA teardown (GPU memory not released) | MEDIUM | Run `nvidia-smi` to identify the orphaned process, `kill -9 <pid>` to force cleanup. On subsequent runs, CUDA will reinitialise correctly from a clean state |

---

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Fork vs spawn (CP-1) | StudyRunner core implementation | Confirm `mp.get_start_method()` returns `"spawn"` inside worker; check CUDA device count in worker vs parent match |
| Pipe deadlock (CP-2) | StudyRunner core implementation | Inject oversized payloads in tests; verify fallback to file-based IPC triggers correctly |
| SIGINT propagation (CP-3) | StudyRunner core implementation | Manual test: `Ctrl+C` during experiment 2 of 3; manifest must show exp-1 as completed, exp-2 as failed, exp-3 as pending |
| daemon=True regression (CP-4) | StudyRunner core implementation | Code review checklist item; confirmed by VRAM check after study exit |
| Sweep combinatorial explosion (CP-5) | Sweep grid expander + pre-flight | Run a study YAML with 4 dimensions × 5 values each; confirm pre-flight shows count and prompts for confirmation |
| Manifest corruption (CP-6) | ManifestWriter implementation | `kill -KILL <pid>` during manifest write; confirm file is valid JSON on next read |
| Timeout miscalibration (CP-7) | StudyRunner + timeout formula | Run a 70B model experiment to completion; confirm timeout is not triggered before results arrive |

---

## Sources

### Design Documents (Primary — this project)
- [`.product/designs/experiment-isolation.md`](../../.product/designs/experiment-isolation.md) — Full `StudyRunner` implementation pattern with rationale
- [`.product/decisions/experiment-isolation.md`](../../.product/decisions/experiment-isolation.md) — Fork vs spawn decision; daemon=False rationale; SIGKILL rationale
- [`.product/designs/study-yaml.md`](../../.product/designs/study-yaml.md) — Sweep grammar; grid expansion algorithm; invalid combo handling
- [`.product/designs/study-resume.md`](../../.product/designs/study-resume.md) — `ManifestWriter`; `StudyManifest` vs `StudyResult` distinction

### Peer Codebase Research
- [`.product/research/13-execution-isolation-patterns.md`](../../.product/research/13-execution-isolation-patterns.md) — Optimum-benchmark process launcher source; AIEnergyScore cleanup pattern; vLLM sweep server management
- [optimum-benchmark process launcher](https://github.com/huggingface/optimum-benchmark/blob/main/optimum_benchmark/launchers/process/launcher.py) — 1MB file-based IPC threshold; Pipe + sync checkpoint pattern; daemon=False

### Official Documentation
- [Python multiprocessing — Start Methods](https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods) — `get_context("spawn")` vs global `set_start_method`; fork-CUDA incompatibility
- [PyTorch: Best Practices for Multiprocessing](https://pytorch.org/docs/stable/notes/multiprocessing.html) — "CUDA runtime does not support fork"
- [POSIX rename(2) man page](https://man7.org/linux/man-pages/man2/rename.2.html) — Atomic rename guarantee on same filesystem (basis for write-then-rename pattern)

---
*Pitfalls research for: multi-experiment study execution with subprocess isolation, sweep grammar, and manifest checkpointing*
*Researched: 2026-02-27*
