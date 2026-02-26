# N-X05: Subprocess Lifecycle Management

**Module**: `src/llenergymeasure/cli/lifecycle.py`
**Risk Level**: MEDIUM
**Decision**: Keep — v2.0 (signal handling and process group management are required; launch command builders need review given architecture change)
**Planning Gap**: `designs/architecture.md` changes subprocess orchestration from the current `SubprocessRunner`/`accelerate launch` pattern to `multiprocessing.Process` + `Pipe` per experiment. The lifecycle.py patterns need to migrate to the new `StudyRunner` design, but no design document specifies how signal handling is handled in the new `multiprocessing` model.

---

## What Exists in the Code

**Primary file(s)**: `src/llenergymeasure/cli/lifecycle.py`
**Key classes/functions**:
- `SubprocessRunner` (line 31) — dataclass; fields: `state_manager: StateManager`, `shutdown_timeout_sec: int = GRACEFUL_SHUTDOWN_TIMEOUT_SEC` (from constants: 2 seconds)
- `SubprocessRunner.run(cmd, env, state)` (line 60) — launches subprocess with `start_new_session=True` (creates new process group), registers SIGINT/SIGTERM handlers, waits for completion, restores original handlers in `finally`
- `SubprocessRunner._handle_interrupt(signum, frame)` (line 105) — re-entrance guard (`_interrupt_in_progress`); sends `SIGTERM` to entire process group via `os.killpg(pgid, signal.SIGTERM)`; waits in 1-second increments up to `shutdown_timeout_sec`; falls back to `SIGKILL` on timeout; transitions `ExperimentState` to `INTERRUPTED`; raises `typer.Exit(130)`
- `build_pytorch_launch_cmd()` (line 161) — builds `accelerate.commands.launch` command list for multi-GPU PyTorch
- `build_vllm_launch_cmd()` (line 197) — builds direct `python -m module` command for vLLM (no accelerate)
- `build_subprocess_env()` (line 219) — constructs subprocess environment: propagates `LLM_ENERGY_VERBOSITY`, `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN`; sets vLLM-specific env vars (`VLLM_ENABLE_V1_MULTIPROCESSING=0`, `VLLM_WORKER_MULTIPROC_METHOD=spawn`, `TORCH_COMPILE_DISABLE=1`, `CUDA_VISIBLE_DEVICES`)

The process group pattern (`start_new_session=True` + `os.killpg`) ensures that when a multi-GPU `accelerate launch` command is interrupted, all worker processes are killed — not just the `accelerate` coordinator. Without this, worker processes would become orphaned, holding GPU memory and NVML sessions.

## Why It Matters

The signal handling and process group management are not optional — they are the mechanism that prevents GPU resource leaks when users press Ctrl+C during an experiment. Without `os.killpg`, interrupting an accelerate run leaves worker processes alive, holding GPU memory and potentially preventing the next experiment from starting. The 2-second graceful shutdown timeout followed by SIGKILL is the correct production pattern (not an arbitrary value — it gives backends time to flush buffers and release NVML). The vLLM-specific environment variables (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) address known vLLM v1 multiprocessing bugs and must be preserved for vLLM experiments to work correctly.

## Planning Gap Details

`designs/architecture.md` changes the execution model:
- v2.0: `ExperimentOrchestrator` runs in-process for `run_experiment()`, in a `multiprocessing.Process` for `run_study()`
- Current: `SubprocessRunner` launches an `accelerate launch` subprocess

This is a significant architecture change. The `build_pytorch_launch_cmd()` and `build_vllm_launch_cmd()` functions may be eliminated or repurposed in v2.0 — but the underlying concerns (signal handling, process group management, vLLM env vars) remain.

The `designs/architecture.md` `StudyRunner` pseudocode (lines 220–241) uses `multiprocessing.Process` + `Pipe` but does not address:
- How SIGINT/SIGTERM to the parent study process propagates to child `Process` objects
- Whether `start_new_session=True` is still needed for `multiprocessing.Process`
- How vLLM-specific env vars are set when the backend runs in a child process (not a subprocess)
- What happens to `CUDA_VISIBLE_DEVICES` in the multiprocessing spawn context

## Recommendation for Phase 5

The `SubprocessRunner` dataclass itself may not survive into v2.0 in its current form (it's designed for `subprocess.Popen`, not `multiprocessing.Process`). However, the signal handling *logic* must be migrated into `StudyRunner`.

Extract and port:
1. The SIGINT/SIGTERM handler pattern — wrap `multiprocessing.Process.join()` with signal handlers that call `process.terminate()` then `process.kill()` after timeout
2. The re-entrance guard (`_interrupt_in_progress`) — same pattern applies in the new model
3. The vLLM env vars from `build_subprocess_env()` — must be set before the child `multiprocessing.Process` is spawned (via `os.environ` mutation in parent before fork/spawn, or passed as explicit env dict to `Process`)
4. The `GRACEFUL_SHUTDOWN_TIMEOUT_SEC` constant — retain in `constants.py`

The `build_pytorch_launch_cmd()` function may still be needed if multi-GPU PyTorch continues to use `accelerate launch` even under the new architecture (the design is not explicit about this). Flag this as an open question for Phase 5 design.
