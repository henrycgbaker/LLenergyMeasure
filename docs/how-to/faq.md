---
title: Frequently asked questions
description: Common questions about running, interpreting, and citing llem.
---

# Frequently asked questions

A consolidated set of common questions that come up when researchers first run `llem`. Each answer links to the deeper reference / explanation page when you want more.

## Running experiments

### Why does my first `llem run` take so long?

Three one-time costs hit on first run:

1. **Model download** &mdash; HuggingFace weights pull on first reference (~500 MB for GPT-2; tens of GB for 7B+ models). Cached afterwards under `~/.cache/huggingface/`.
2. **Docker image pull** &mdash; the engine container (~5&ndash;15 GB) downloads from GHCR or the upstream registry.
3. **TensorRT-LLM compilation** &mdash; if you're on the TRT-LLM backend, the engine compiles from weights on first reference (5&ndash;15 minutes for a 7B model). The compiled engine is cached at `~/.cache/tensorrt_llm/` inside the container; subsequent runs of the same config skip compilation. See [How to: run with TensorRT-LLM](/how-to/run-with-tensorrt-llm).

### Why does my run hang?

Most common causes, in order of likelihood:

1. **Image pull** &mdash; the first run pulls a multi-GB Docker image. Check `docker pull` progress in another terminal.
2. **TRT-LLM engine build** &mdash; first run compiles, no progress shown for several minutes. Re-run with `--verbose` to see compilation logs.
3. **Stuck pre-flight check** &mdash; if Docker daemon is unreachable or the GPU is being used by another process, the pre-flight stalls. Run `llem doctor` to diagnose.
4. **Genuine stuck inference** &mdash; rare; raise `study_execution.experiment_timeout_seconds` if you have a model + workload combination that legitimately exceeds the 600-second default. See [How to: troubleshoot](/how-to/troubleshoot#stalls-or-hangs).

### Can I run llem without Docker?

Partially. The Transformers (PyTorch) engine runs in `local` mode without Docker if you set `runners: { transformers: local }` in your YAML. vLLM and TensorRT-LLM are Docker-only by design &mdash; their dependency closures are too divergent to install side-by-side on a host. See [Contributing &gt; Development](/contributing/development) for the rationale.

### Why are my results different on consecutive runs of the same config?

Several stochastic sources, in decreasing order of impact:

1. **Thermal state** &mdash; a hot GPU clocks down. Use the default `study_execution.experiment_gap_seconds: 10` and `cycle_gap_seconds: 30` (or longer) so each cell starts from a comparable thermal floor.
2. **Sampling RNG** &mdash; if `decoder.do_sample: true` and `random_seed` differs, generations vary. Pin `random_seed: 42`. See [Methodology &gt; Reproducibility](/methodology/methodology#reproducibility).
3. **System load** &mdash; other processes on the host affect baseline power and throttle the GPU. Run on a dedicated GPU when possible.
4. **NVML sampling jitter** &mdash; ~1 % run-to-run from sampling-instant variability. This is below the &plusmn;5 % NVML accuracy floor and not actionable.

For variance-aware comparison, set `study_execution.n_cycles: >= 3` and report median + IQR rather than single-run numbers.

## Interpreting results

### Why does adjusted energy sometimes look smaller than total energy?

That's the design intent. **Adjusted energy is total energy minus the baseline contribution** (`total_energy_j - baseline_power_w &times; total_inference_time_sec`). When the GPU was idle for part of the run, the baseline subtraction removes that contribution &mdash; leaving "energy attributable to inference work."

If adjusted energy comes out **negative** (rare but possible), the baseline measurement was higher than the inference's average power draw. Causes: a noisy baseline window, a thermal-throttled inference, or a mis-calibrated sampler. Treat this experiment as suspect and re-run with a longer `baseline.duration_seconds`.

See [Methodology &gt; Baseline power](/methodology/methodology#baseline-power) for the full reasoning.

### What does "FLOPs (roofline, medium)" mean?

`llem` reports a FLOPs *estimate*, not a measurement &mdash; instantaneous FLOPs are not directly observable during inference. The two qualifiers are:

- **Method** &mdash; how the estimate was computed. `roofline` uses the model's parameter count and a per-token forward-pass approximation. Future methods may include `cycle_count` (more accurate, requires Nsight integration) or `flop_counter` (PyTorch FlopCounterMode hook).
- **Confidence** &mdash; `low | medium | high`. `medium` reflects published roofline-method accuracy across architectures. `low` is emitted when the estimator hits an architecture it doesn't fully recognise.

Use `mj_per_tok_adjusted` (energy per token) or `total_inference_time_sec` (wall-clock) for cross-engine comparison; `flops_per_*` are reference metadata, useful for normalising across model sizes but not for headline efficiency claims. See [What we measure](/methodology/what-we-measure#flops).

### Why does my study report `unique_configurations` lower than `total_experiments`?

`llem` deduplicates measurement-equivalent configs before running. If your sweep generates two cells whose effective `ExperimentConfig` hashes are identical (e.g. a sampling parameter is `dormant` for the active engine), they collapse into one cell. `total_experiments` is the number actually *run* (after dedup &times; `n_cycles`); `unique_configurations` is the distinct configs.

To inspect what was deduplicated: read `_study-artefacts/equivalence_groups.json` in the study directory. To disable dedup: pass `--no-dedup` to `llem run`.

## Configuration

### Top-level `n: 50` doesn't work &mdash; what's the canonical YAML?

`n` is a CLI flag (`-n / --n-prompts`), not a YAML field. The YAML form is:

```yaml
task:
  dataset:
    source: aienergyscore
    n_prompts: 50
```

See [Reference &gt; Study config](/reference/study-config) for the full YAML schema.

### How do I sweep across implementation parameters without sweeping engines?

Engine-scoped sweep keys: `transformers.batch_size: [1, 4, 16]` only applies to Transformers experiments; the same config running under vLLM ignores it. Use the engine-prefixed form for any engine-specific axis. See the [Multi-backend study tutorial](/tutorials/multi-backend-study) for a worked example.

### Can I use a custom dataset (not `aienergyscore`)?

Yes. Set `task.dataset.source` to a path to a JSONL file:

```yaml
task:
  dataset:
    source: ./prompts.jsonl
    n_prompts: 100
```

Each line is a JSON object with at least a `prompt` field. Optional: `expected_output` for accuracy-based studies. The JSONL spec is light by design &mdash; if your study needs stricter typing, validate before passing.

## Hardware

### Does FP8 work on A100?

No. FP8 quantisation (per `tensorrt.quant_config.quant_algo: FP8`, `vllm.engine.kv_cache_dtype: fp8`) requires SM &geq; 8.9 (Ada Lovelace, Hopper). A100 is SM 8.0 (Ampere). The validation pipeline raises a clear `ConfigurationError` before engine init &mdash; you won't waste GPU time on it. Valid A100 quantisation: `INT8`, `W4A16_AWQ`, `W4A16_GPTQ`, `W8A16`. See [Reference &gt; Engine configuration](/reference/engines/configuration).

### Does it work on consumer GPUs (RTX 4090 / 4080)?

Yes for Transformers and vLLM. TensorRT-LLM compilation is supported on Ada Lovelace (SM 8.9). Consumer GPUs typically have less VRAM than datacentre cards, so larger models may need quantisation or smaller batch sizes. The reference invariants corpus catches the most common VRAM-vs-config-vs-batch-size mismatches before init.

### Multi-GPU?

Tensor parallel is supported via engine-native fields (`tensorrt.tensor_parallel_size`, `vllm.tensor_parallel_size`). `llem` measures aggregate energy across all visible GPUs. Set `CUDA_VISIBLE_DEVICES` to control which GPUs are used. Cross-node distributed is not currently supported.

## Citing and reporting

### How do I cite `llem` in a paper?

Until a formal release citation is published, cite the GitHub repository: `https://github.com/henrycgbaker/llenergymeasure` plus the version (`llem --version` or pin `llenergymeasure==0.9.0` in your reproducibility appendix). The full effective configuration is in every `result.json`'s `effective_config` block &mdash; share that JSON for full reproducibility.

### What should I include in a paper for measurement reproducibility?

Minimum reproducibility set:

1. The study YAML (or a hash + URL pointing to it)
2. The hardware: GPU model, host OS, NVIDIA driver version
3. The engine + library version (`engine_version` field in result.json)
4. `llem` version (`llenergymeasure_version` field)
5. `study_design_hash` from manifest.json
6. Per-experiment effective configs (or the directory structure they live in)

The shipped `manifest.json` plus the per-experiment `effective_config.json` files contain everything in items 3&ndash;6. See [Methodology &gt; Reproducibility](/methodology/methodology#reproducibility) for the full checklist.

## See also

- [How to: troubleshoot](/how-to/troubleshoot) &mdash; specific error message catalogue
- [Reference: results schema](/reference/results-schema) &mdash; every result.json field documented
- [Methodology](/methodology/methodology) &mdash; warmup, baseline, thermal management, reproducibility
