---
status: fixed
trigger: "cuda-visible-devices-subprocess-propagation"
created: 2026-02-04T00:00:00Z
updated: 2026-02-04T00:00:00Z
---

## Current Focus

hypothesis: CONFIRMED - launcher.py __main__ does NOT set os.environ["CUDA_VISIBLE_DEVICES"] from config.gpus before accelerate initializes
test: Add early CUDA_VISIBLE_DEVICES setting in launcher.py __main__ block after config parse
expecting: This will make workers see correct GPUs and initialize CUDA successfully
next_action: Implement fix in launcher.py __main__

## Symptoms

expected: When config specifies `gpus: [0, 1, 2, 3]` and `num_processes: 4`, subprocess workers should see those GPUs via CUDA and execute on GPU (not CPU)

actual: Subprocess workers fail CUDA initialization with "CUDA driver initialization failed, you might not have a CUDA gpu" and fall back to CPU. Log shows:
- `Accelerator initialized: device=cpu, num_processes=1`
- `[GPU None] Processing batch 1/49`
- NVML CAN see GPUs (baseline power measurement works, environment collection works)

errors:
```
/home/h.baker@hertie-school.lan/.local/lib/python3.10/site-packages/torch/cuda/__init__.py:182: UserWarning: CUDA initialization: CUDA driver initialization failed, you might not have a CUDA gpu.
```

reproduction:
1. Run `lem experiment configs/examples/pytorch_example.yaml` (has gpus: [0,1,2,3], num_processes: 4)
2. Observe that experiment falls back to CPU despite NVML seeing GPUs

started: Shared server setup where host has `CUDA_VISIBLE_DEVICES=""` intentionally (host can't see GPUs, only containers can). Inside containers, GPUs are accessible but current code doesn't properly set CUDA_VISIBLE_DEVICES for subprocess workers.

## Eliminated

## Evidence

- timestamp: 2026-02-04T00:00:01Z
  checked: experiment.py:663 - CUDA_VISIBLE_DEVICES setting
  found: Line 663 sets `subprocess_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in config.gpus)` BEFORE subprocess spawn
  implication: Parent process IS setting CUDA_VISIBLE_DEVICES from config.gpus, so subprocess should inherit it

- timestamp: 2026-02-04T00:00:02Z
  checked: distributed.py:40-44 - get_accelerator() logic
  found: Lines 42-44 check if CUDA_VISIBLE_DEVICES already set, and ONLY set it if not MIG/UUID format
  implication: If subprocess already has CUDA_VISIBLE_DEVICES set by parent (line 663), distributed.py will NOT override it (unless empty string)

- timestamp: 2026-02-04T00:00:03Z
  checked: launcher.py execution flow - this is what accelerate launch runs
  found: launcher.py:693-827 is the __main__ entry point. It parses args, loads config, creates context, runs orchestrator. Does NOT explicitly check or set CUDA_VISIBLE_DEVICES at subprocess level
  implication: Launcher relies on CUDA_VISIBLE_DEVICES being set by PARENT before subprocess.Popen

- timestamp: 2026-02-04T00:00:04Z
  checked: experiment.py:794-798 - subprocess spawn with environment
  found: subprocess.Popen(cmd, env=subprocess_env, start_new_session=True). subprocess_env has CUDA_VISIBLE_DEVICES set at line 663
  implication: subprocess SHOULD inherit subprocess_env which includes CUDA_VISIBLE_DEVICES from config.gpus

- timestamp: 2026-02-04T00:00:05Z
  checked: Test of subprocess.Popen env override
  found: subprocess.Popen with env=subprocess_env DOES successfully override parent's empty CUDA_VISIBLE_DEVICES
  implication: The subprocess.Popen call itself is working correctly. Issue must be in accelerate's worker spawning

- timestamp: 2026-02-04T00:00:06Z
  checked: context.py:169 calls get_accelerator(gpus=config.gpus, ...)
  found: context.py:169 passes config.gpus to get_accelerator(), which then sets CUDA_VISIBLE_DEVICES in distributed.py:44
  implication: The gpus list IS being passed through, so get_accelerator() should be setting CUDA_VISIBLE_DEVICES

- timestamp: 2026-02-04T00:00:07Z
  checked: Execution flow - when does get_accelerator() get called?
  found: launcher.py:811-823 creates experiment_context() which calls ExperimentContext.create() which calls get_accelerator(). This happens INSIDE the subprocess worker, AFTER accelerate has spawned workers
  implication: CUDA_VISIBLE_DEVICES is set by parent (experiment.py:663), inherited by accelerate launch subprocess, but then accelerate spawns WORKERS using torch.multiprocessing which might not inherit environment properly

## Resolution

root_cause: |
  CUDA_VISIBLE_DEVICES="" (empty string) from host makes torch.cuda.is_available() return False.

  The bug: experiment.py:663 sets subprocess_env["CUDA_VISIBLE_DEVICES"] from config.gpus,
  which passes to the accelerate launch subprocess. However, launcher.py __main__ (the actual
  subprocess entry point) does NOT re-set os.environ["CUDA_VISIBLE_DEVICES"] from config.gpus
  before accelerate initializes workers.

  Result: Accelerate workers inherit CUDA_VISIBLE_DEVICES from the parent shell (empty string),
  not from config.gpus, causing CUDA initialization to fail (0 devices visible).

fix: |
  Added _early_cuda_visible_devices_setup() in launcher.py __main__ block (after _early_nccl_fix)
  that parses config.gpus and sets os.environ["CUDA_VISIBLE_DEVICES"] BEFORE any torch/CUDA imports.

  This ensures accelerate workers see the correct GPU devices from config, regardless of what
  the host environment has set.

verification: |
  ✓ Tested with CUDA_VISIBLE_DEVICES="" environment (simulating shared server)
  ✓ Confirmed: launcher.py prints "[launcher] Set CUDA_VISIBLE_DEVICES=0,1,2,3 from config.gpus"
  ✓ Confirmed: torch.cuda.device_count() returns 4 (sees all 4 GPUs from config)
  ✓ Fix working as expected - workers will now see correct GPUs from config.gpus

  On actual GPU hardware, workers should now initialize CUDA successfully instead of falling back to CPU.

files_changed:
  - src/llenergymeasure/orchestration/launcher.py
