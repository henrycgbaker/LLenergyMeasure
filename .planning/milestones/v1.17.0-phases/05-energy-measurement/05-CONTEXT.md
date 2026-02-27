# Phase 5: Energy Measurement - Context

**Gathered:** 2026-02-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Every experiment produces scientifically credible energy numbers — baseline-corrected, warmed-up, with correct measurement backend priority and a 1 Hz timeseries sidecar — so results can be cited in a research paper. Covers CM-11 through CM-28: energy backends (NVML, Zeus, CodeCarbon), baseline power, warmup, FLOPs estimation, and timeseries.

Note: Many implementation details are already locked in `.product/decisions/` (warmup-strategy.md, flops-estimation.md) and `.product/designs/` (energy-backends.md). This context captures the remaining gray-area decisions.

</domain>

<decisions>
## Implementation Decisions

### Energy backend wiring
- Default mode: `auto` — selects best available backend (Zeus > NVML > CodeCarbon)
- Explicit override supported: `energy.backend: auto | nvml | zeus | codecarbon | null` in ExperimentConfig
- `null` is a valid explicit choice — disables energy measurement intentionally (no warnings, energy fields are null)
- When user explicitly requests an unavailable backend: hard `ConfigError` with install guidance (e.g., `pip install llenergymeasure[zeus]`)
- When no energy backend is available at all (CPU-only machine): run inference, produce latency/throughput results, but energy fields are null; surface clearly in results that energy measurement was unavailable
- Backend conflicts/mutual exclusion enforced at selection time (only one backend owns the NVML session)

### Measurement quality signals
- Warnings are purely informational — never block the experiment
- Warning flags stored in `result.measurement_warnings` (list of strings) and displayed inline in CLI output
- Four warning flags: `short_measurement_duration` (<10s), `gpu_persistence_mode_off`, `thermal_drift_detected` (>10°C change), `nvml_low_sample_count` (<10 samples)
- No `--strict` mode — warnings annotate, researchers filter results themselves
- Thermal drift threshold: **configurable** with 10°C default (note: no peer citation — engineering judgement; confidence LOW; flagged for future empirical validation)
- Warning display format: inline with results, includes actionable remediation advice (e.g., "Run `nvidia-smi -pm 1` to enable persistence mode")

### Warmup-to-measurement handoff
- Thermal floor: pure time-based wait (sleep for `thermal_floor_seconds` after warmup) — no temperature probing. Simple, predictable, matches MLPerf Power. Already configurable: default 60s, minimum 30s.
- Docker mode: thermal floor still applies within each experiment (intra-experiment). Inter-experiment `gap_seconds` is auto-skipped (container startup provides natural thermal reset).
- WarmupResult captures everything: warmup run count, per-run latencies, total warmup duration, AND thermal floor wait duration. Full provenance for reproducibility.
- CV convergence (opt-in) is additive: run AT LEAST `n_warmup` runs, then continue until CV < target OR `max_warmup_prompts` hit. Fixed count is a minimum, not replaced.
- CLI output: plain-text status line updates as each phase completes: "Warmup: 5/5 runs complete (12.3s)" → "Thermal stabilisation: waiting 60s..." → "Measurement: starting"

### Energy result fields
- Always include all three: baseline power (`baseline_power_w`), raw energy (`energy_total_j`), and adjusted energy (`energy_adjusted_j`)
- Never omit any of the three — if baseline is unavailable, `baseline_power_w` is null and `energy_adjusted_j` is null, but `energy_total_j` is still populated

### Timeseries format and content
- Full telemetry columns: `timestamp_s`, `gpu_index`, `power_w`, `temperature_c`, `memory_used_mb`, `memory_total_mb`, `sm_utilisation_pct`, `throttle_reasons`
- Measurement window only — no warmup or thermal floor samples. Every row is a valid measurement sample.
- Long format for multi-GPU: one row per GPU per second (with `gpu_index` column), not wide format
- File location: same directory as result JSON — `{name}_{timestamp}/timeseries.parquet`
- Result JSON references timeseries via relative path

### Default confidence levels
- `n_warmup: 5` — HIGH (DeepSpeed 5–10, Zeus 10, AIEnergyScore 10)
- `thermal_floor_seconds: 60` — HIGH (MLPerf Power IEEE HPCA 2025 mandates 60s minimum)
- `thermal_drift_threshold: 10°C` — LOW (engineering judgement, no peer citation found; flagged for empirical validation in future milestone)
- `baseline_duration: 30s` — MEDIUM (similar to Verified Instruction-Level Energy paper 22–33s windows, but no direct spec)
- `sample_interval: 100ms` — HIGH (aligned with NVML hardware update period per NVIDIA docs)

### Claude's Discretion
- Internal module structure (where energy backend registry lives, how auto-selection is wired)
- NVML poller implementation details (thread vs async, error recovery)
- How baseline cache invalidation interacts with energy backend switching
- FLOPs implementation: carry forward existing `flops.py` or rewrite — as long as it implements the PaLM formula per the locked decision
- Exact warning message wording (as long as it includes remediation advice)

</decisions>

<specifics>
## Specific Ideas

- Thermal drift threshold and baseline duration defaults should document their confidence levels in code comments — researchers reading the source should know which defaults are empirically grounded vs engineering judgement
- Error messages for missing backends should mirror the install extras pattern: "Zeus not installed. Install with: pip install llenergymeasure[zeus]"
- When energy is unavailable, the result should make it obvious — not just null fields but a clear signal in the output

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 05-energy-measurement*
*Context gathered: 2026-02-26*
