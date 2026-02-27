# Phase 7: CLI - Context

**Gathered:** 2026-02-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Researchers interact with the tool through `llem run` and `llem config` -- plain text output, correct exit codes on all error paths, `--dry-run` validation, and `--version` flag. The CLI is a thin wrapper around `run_experiment()` / `run_study()` from the library API. No new measurement, config, or results logic -- this phase only builds the terminal interface.

</domain>

<decisions>
## Implementation Decisions

### Result summary format
- Strictly raw metrics only -- no derived ratios (no J/token, no J/request)
- Grouped sections layout: Energy, Performance, Timing (not flat key:value)
- Include FLOPs estimate (with method/confidence) in the summary
- Include measurement warnings inline in the summary
- 3 significant figures for all numeric values (consistent, avoids false precision)
- All peer tools surveyed (AIEnergyScore, vLLM bench, lm-eval, Optimum-Benchmark, mlperf) show raw metrics only; derived ratios computed externally

### Dry-run presentation
- Single-experiment `--dry-run`: config echo + VRAM estimate (no pre-flight detail)
- VRAM breakdown shown: weights, KV cache, overhead as separate lines plus total
- `--dry-run` always exits 0 for valid configs (informational, not a gate). Exit 2 only for ConfigError.
- `--verbose --dry-run`: adds source annotations to config echo (e.g., 'bf16 (default)', 'pytorch (--backend)', '100 (experiment.yaml)')
- Standard `--dry-run`: values only, no source annotations
- Study-mode `--dry-run` grid preview already specified in decisions/cli-ux.md

### Error message style
- Guided errors (Rust/Elm style): what went wrong, where (file:line if applicable), and a fix suggestion
- Example: `ConfigError: unknown backend 'pytorh'\n  -> experiment.yaml, line 5\n  Did you mean: pytorch?\n  Valid backends: pytorch, vllm, tensorrt`
- Did-you-mean suggestions on ALL string enum fields (backend, precision, dataset aliases, etc.)
- Python stack traces hidden by default; shown with `--verbose`
- Pydantic ValidationError: wrapped with friendly header ("Config validation failed (N errors):") but Pydantic's own error messages pass through unchanged

### Progress display
- Standard tqdm bars for both warmup and measurement phases
- Warmup gets its own tqdm bar (replaced by measurement bar when warmup completes)
- Non-TTY / piped output: suppress all progress bars entirely, print final result summary only (tqdm auto-detects)
- NO_COLOR respected (tqdm handles natively)

### Claude's Discretion
- Experiment header line: model + backend always shown, plus any non-default parameters (Claude decides which params qualify as "non-default" worth showing)
- Exact tqdm format string customisation
- Exact indentation and spacing in output
- How --verbose subprocess events are formatted
- Internal error formatting utilities (difflib for did-you-mean, etc.)

</decisions>

<specifics>
## Specific Ideas

- Result summary mockup (grouped sections):
  ```
  Energy
    Total          312 J
    Baseline       45.2 W

  Performance
    Throughput     847 tok/s
    Latency TTFT   142 ms
    Latency ITL    28 ms

  Timing
    Duration       4m 32s
    Warmup         20 prompts (CV converged at 12)
  ```
- Dry-run mockup (single experiment):
  ```
  Config (resolved)
    Model          gpt2
    Backend        pytorch
    Precision      bf16
    Batch size     1
    Dataset        aienergyscore (100 prompts)
    Output         results/gpt2_pytorch_{timestamp}/

  VRAM estimate
    Weights        0.24 GB (bf16)
    KV cache       0.01 GB
    Overhead       0.04 GB
    Total          ~0.29 GB / 80 GB available   OK

  Config valid. Run without --dry-run to start.
  ```
- Error style follows Rust compiler pattern -- actionable, not just informative
- Peer research note: the research phase should investigate how peer CLI tools (lm-eval, vLLM bench, Optimum-Benchmark, mlflow, AIEnergyScore) handle each of these areas in their actual implementations

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope

</deferred>

---

*Phase: 07-cli*
*Context gathered: 2026-02-26*
