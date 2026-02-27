# Phase 4: PyTorch Backend and Pre-flight - Context

**Gathered:** 2026-02-26
**Status:** Ready for planning

<domain>
## Phase Boundary

PyTorch inference runs correctly end-to-end with the new ExperimentConfig/ExperimentResult contract. Pre-flight checks catch configuration errors before GPU allocation. Environment is fully snapshotted at experiment start. The P0 model_kwargs bug is eliminated by rewrite.

</domain>

<decisions>
## Implementation Decisions

### Pre-flight checks
- Flat list of all failures collected into a single `PreFlightError` (not grouped by category)
- Boundary: Pydantic handles schema validation (types, enums, missing fields); pre-flight handles runtime checks (backend installed? model accessible? CUDA available?)
- Essential checks only: backend installed, model accessible (HF hub reachable, gated model token), CUDA available. No VRAM estimation — unreliable across quantisation/batch sizes
- Always runs before every experiment — no opt-out, no `--skip-preflight`
- Always raises `PreFlightError` on failure — no warn mode. Library users wrap in `try/except PreFlightError`

### Environment snapshot
- Full `pip freeze` output — the entire environment, not filtered to llem deps
- If conda is detected, also include `conda list` output
- Captured before inference starts (before model loading) — this is the starting state, not runtime state
- GPU memory usage is a measurement result, not an environment property

### PyTorch runner shape
- Rewrite from scratch using v1.x code as reference only — not an incremental adaptation
- Direct `ExperimentConfig` acceptance: `run(config: ExperimentConfig) -> ExperimentResult`
- Shared Protocol/ABC defines the contract for all backends (PyTorch, vLLM, TRT-LLM)
- Runner reads `config.model` (shared) + `config.pytorch.*` (backend-specific) directly — no adapter layer
- Matches peer pattern: lm-eval `LM` subclass, Optimum-Benchmark direct config

### Error messages
- Generic but helpful fix suggestions: "CUDA out of memory. Try: reduce batch_size, use precision=fp16, or use a smaller model"
- No VRAM calculations in error messages — estimates are unreliable
- Text instructions only for model access errors — no URLs (terminal inconsistency)
- No partial results on failure — a failed experiment's measurements are invalid
- `BackendError` for inference runtime failures (CUDA errors, model load, inference crashes)
- `ExperimentError` wraps `BackendError` with study context (which config, which iteration)

### Claude's Discretion
- Whether to record CUDA version detection source alongside the resolved version
- model_kwargs regression test (likely unnecessary given rewrite)
- Exact pre-flight check ordering and parallelism
- Loading skeleton / progress display during model download

</decisions>

<specifics>
## Specific Ideas

- Pre-flight error display matches the format from error-handling.md decisions:
  ```
  Pre-flight failed: 2 issues found
    ✗ vllm      not installed → pip install llenergymeasure[vllm]
    ✗ Llama-3-70B  gated model — no HF_TOKEN → export HF_TOKEN=<your_token>
  ```
- Environment snapshot is captured once, at experiment start, and attached to `result.environment_snapshot`
- The runner Protocol should be simple enough that adding vLLM/TRT backends later is just implementing a new module with the same `run()` signature

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 04-pytorch-backend-pre-flight*
*Context gathered: 2026-02-26*
