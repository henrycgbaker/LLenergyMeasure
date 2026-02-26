# Warmup Strategy

**Status:** Accepted
**Date decided:** 2026-02-19
**Last updated:** 2026-02-25 (thermal floor → 60s configurable; warmup → full-length default; Docker gap auto-skip)
**Research:** [../research/14-flops-warmup-lora-multiGPU.md](../research/14-flops-warmup-lora-multiGPU.md)

---

## Context

LLM inference benchmarks suffer from transient first-call effects that inflate latency and energy measurements if not controlled. Four distinct sources of noise affect early runs:

1. **CUDA kernel compilation** — `torch.compile`, TRT engine load — one-time first-call cost
2. **CUDA context and memory allocation** — first-request GPU setup overhead
3. **Caching effects** — KV cache pool initialisation, attention mask compilation
4. **Thermal / frequency stabilisation** — GPU power state ramps up over first 10–30s

Effects 1–3 are transient and eliminated by discarding a fixed number of early runs. Effect 4 is a slower thermal process requiring a separate time-based floor for energy benchmarks. The warmup design must handle all four.

**Constraints:**
- Warmup must terminate in bounded time (convergence loops may not)
- Warmup tokens must not inflate FLOPs or energy metrics
- Cold-start measurement (deliberate first-call overhead) must be expressible and must bypass warmup entirely

---

## Considered Options

### W1: Warmup termination criterion

| Option | Pros | Cons |
|--------|------|------|
| **Fixed count (chosen)** | Bounded runtime; consistent across hardware; zero implementation complexity; matches all peer tools | Does not adapt to unusually noisy hardware |
| CV convergence (rolling window < 5%) | Theoretically adapts to hardware noise | No peer implementation (confirmed across lm-eval, optimum-benchmark, vLLM, Zeus, MLPerf, AIEnergyScore); unbounded runtime on noisy hardware (must be capped → becomes fixed count anyway); GPU latency CV is dominated by slow thermal drift, not fast convergence |

**Rejected: CV convergence (2026-02-19).** No peer LLM benchmarking tool implements it. The unbounded runtime problem means it must be capped regardless, making it equivalent to fixed count with extra complexity. The 30-second thermal floor (W3) addresses the underlying thermal concern more directly.

---

### W2: Output length during warmup runs

| Option | Pros | Cons |
|--------|------|------|
| Reduced output — 2 tokens max | Eliminates JIT/CUDA overhead quickly; minimal compute cost per warmup run; matches optimum-benchmark pattern | Does not warm KV cache at operational size; does not trigger vLLM continuous batching; insufficient thermal conditioning |
| **Full-length output (chosen, 2026-02-25)** | Identical memory pattern to measurement runs; warms KV cache + decode path + thermal state; methodologically correct for energy measurement | Higher warmup cost per run |
| Zero output (prefill only) | Fastest warmup possible | Does not warm up the decode path; incomplete warmup |

> **Superseded (2026-02-25):** 2-token warmup replaced with full-length warmup as default.
> `.planning/research/PITFALLS.md` (MP-3) showed that 2-token runs do not warm the KV cache
> at operational size, do not trigger vLLM's continuous batching scheduler, and provide only
> ~2–3 seconds of GPU compute — insufficient for thermal conditioning in energy benchmarks.
> Full-length warmup runs are the methodologically correct default for a tool measuring energy.
>
> **Rejected (2026-02-25): Reduced output — 2 tokens max.** Insufficient for energy measurement
> methodology. optimum-benchmark uses reduced output for speed benchmarks, not energy benchmarks.

---

### W3: Thermal stabilisation for energy benchmarks

| Option | Pros | Cons |
|--------|------|------|
| **Time-based floor — 60 seconds configurable (chosen, updated 2026-02-25)** | Matches MLPerf Power standard (60s minimum); configurable down to 30s for quick iteration | Adds wall-clock time per experiment |
| Time-based floor — 30 seconds (original, superseded) | Shorter wait time | Under-calibrated — no cited source; A100/H100 takes 45–90s to stabilise thermally |
| Fixed run count only | Simpler; no extra latency | Run count alone does not guarantee thermal steady state — depends on run duration and GPU model |
| No thermal floor | Zero overhead | Energy measurements on a non-steady GPU are systematically biased low |

The two mechanisms (fixed count + thermal floor) address distinct problems and are both required for energy benchmarks.

> **Superseded (2026-02-25):** 30-second thermal floor increased to 60 seconds (configurable).
> `.planning/research/PITFALLS.md` (CP-2) found: MLPerf Power mandates 60s minimum under load;
> A100/H100 thermal stabilisation takes 45–90s; the original 30s figure had no cited source.
> Now configurable: `thermal_floor_seconds: 60` (default); can be reduced to 30 for iteration.
> For publication-grade rigour, set `thermal_floor_seconds: 60` (or higher) in study YAML.
> Sources: [MLPerf Power (arXiv:2410.12032)](https://arxiv.org/html/2410.12032v2),
> [ML.ENERGY Benchmark (arXiv:2505.06371)](https://arxiv.org/html/2505.06371v1).
>
> **Rejected (2026-02-25): 30-second floor.** Under-calibrated per MLPerf research.

---

### W4: Default `n_warmup` value

| Option | Pros | Cons |
|--------|------|------|
| **n_warmup = 5 (chosen)** | Conservative but fast; DeepSpeed Profiler and Zeus use 5–10; covers CUDA compilation in one pass | Possibly insufficient on extremely cold hardware |
| n_warmup = 10 | AIEnergyScore default; more conservative | Doubles warmup time for marginal gain on typical hardware |
| n_warmup = 3 | Fast | Insufficient for CUDA context + KV cache initialisation on larger models |

---

### W5: Cold-start measurement support

| Option | Pros | Cons |
|--------|------|------|
| **`cold_start: true` → `n_warmup = 0` (chosen)** | Explicitly measures first-call overhead without warmup interference; semantically clear | Adds a special-case code path |
| No cold-start support | Simpler | Cannot measure cold-start overhead, which is a valid research question |

---

## Decision

We will use **fixed-count full-length warmup with a configurable thermal floor**:

- 3 warmup runs (default `warmup_runs: 3`) at full sequence length (same config as measurement runs)
- 60-second thermal floor (default `thermal_floor_seconds: 60`), configurable down to 30s for quick iteration
- For publication-grade rigour: set `thermal_floor_seconds >= 60` in study YAML `execution:` block
- `cold_start: true` in study YAML sets `n_warmup = 0`, bypassing warmup entirely
- Warmup tokens are excluded from all FLOPs and energy calculations
- Warmup duration (`WarmupResult.duration_sec`) is stored in `ExperimentResult` to enable audit of the thermal gap
- GPU temperature recorded at measurement start and end to detect thermal drift
- **Docker mode**: inter-experiment `gap_seconds` auto-skipped (container startup provides natural thermal reset)

**Rationale:** Full-length warmup runs are necessary for correct energy measurement — they warm KV cache at operational size, trigger backend schedulers (vLLM continuous batching), and provide sufficient thermal conditioning. The 60s thermal floor matches MLPerf Power (IEEE HPCA 2025). Docker containers provide natural inter-experiment gaps via startup + model load, making explicit gap_seconds redundant in Docker mode. CV convergence was rejected because no peer implements it and it introduces unbounded runtime.

---

## Consequences

**Positive:**
- Bounded warmup time — experiments are predictable in duration
- Complete elimination of CUDA kernel compilation, context allocation, and KV cache initialisation overhead
- Full-length warmup ensures KV cache, decode path, and thermal state are all at operational levels
- Thermal steady state guaranteed for energy measurements (60s floor, MLPerf-aligned)
- Consistent with peer tool practice — results are comparable
- Docker mode avoids wasted inter-experiment gaps

**Negative / Trade-offs:**
- Fixed count does not adapt to hardware variability; very noisy GPUs may still produce variable measurements
- 60-second floor adds wall-clock time per energy benchmark experiment (configurable down to 30s for iteration)
- Full-length warmup runs are slower than 2-token warmup — accepted trade-off for correctness
- Warmup cost is paid even for single-run experiments (unless `cold_start: true`)

**Neutral / Follow-up decisions triggered:**
- Warmup tokens excluded from FLOPs: see [decisions/flops-estimation.md](flops-estimation.md)
- `warmup_runs` config field: see [designs/experiment-config.md](../designs/experiment-config.md)
- `cold_start: bool` field: see [designs/study-yaml.md](../designs/study-yaml.md)
- Full implementation deferred to Phase 5 as `orchestration/warmup.py`

---

## Related

- [designs/experiment-config.md](../designs/experiment-config.md): `warmup_runs` config field
- [designs/study-yaml.md](../designs/study-yaml.md): `cold_start: bool` overrides warmup_runs to 0
- [decisions/flops-estimation.md](flops-estimation.md): Warmup tokens excluded from FLOPs
- [research/14-flops-warmup-lora-multiGPU.md](../research/14-flops-warmup-lora-multiGPU.md): §Warmup Strategy — peer tool survey
- NEEDS_ADDRESSING.md item 21
