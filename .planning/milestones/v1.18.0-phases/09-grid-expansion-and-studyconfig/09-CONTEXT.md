# Phase 9: Grid Expansion and StudyConfig - Context

**Gathered:** 2026-02-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Sweep YAML parsing and resolution into a flat `list[ExperimentConfig]`, cycle ordering logic, `study_design_hash` computation, and pre-flight count display. No subprocess execution, no manifest writing, no CLI wiring — those are Phases 10-12. This phase is pure data transformation: YAML in, resolved experiment list out.

</domain>

<decisions>
## Implementation Decisions

### Pre-flight display
- Count summary only: `Study [hash]: 4 configs x 3 cycles = 12 runs`
- No time estimates — just experiment count and total run count
- Hash displayed in pre-flight (git/Docker style — researchers reference it in papers and notebooks)
- Display ordering mode name only ("Order: interleaved"), not the full execution sequence
- `--dry-run` detailed config listing deferred to Phase 12 (CLI concern, noted for traceability)

### Invalid combination handling
- Count + per-skip log line with reason: "Skipping 3/15 (pytorch, fp32, batch=32): [Pydantic message]"
- Light wrapping: prepend which config failed, let Pydantic provide the "why" (matches Hydra pattern)
- Warn prominently if >50% of generated configs are invalid ("Most of your sweep is invalid — check your config")
- Hard error (exit 1) if ALL configs are invalid — "nothing to run" is an error, not a success. Matches pytest (exit 5), Snakemake, lm-eval, Hydra peer behaviour
- Skipped configs + validation reasons persisted in study metadata for post-hoc review

### Base config resolution
- `base:` field included in Phase 9 scope (optional DRY convenience)
- Hard error at parse time if base file doesn't exist
- Path resolution: relative to the study.yaml file's directory (matches Hydra, Docker Compose, Snakemake, Nextflow conventions)
- `base:` accepts experiment config files only — not study files. One level of inheritance, no chaining.

### Cycle ordering
- Shuffle seed: derived from `study_design_hash` by default (same study = same shuffle, reproducible, zero config)
- Optional `shuffle_seed:` field in `execution:` block for explicit override
- Shuffle seed is NOT part of `study_design_hash` (it's in execution block, excluded by design)

### Claude's Discretion
- Confirmation prompt threshold for large sweeps (whether to auto-proceed or pause above N experiments)
- `base:` field scoping details (how to detect experiment vs non-experiment YAML)
- Exact format of pre-flight display (spacing, colours, alignment)
- Internal structure of the grid expander (function signatures, module placement)

</decisions>

<specifics>
## Specific Ideas

- Pre-flight should feel like git/Docker — show the hash as a fingerprint researchers learn to reference
- Peer-grounded decisions throughout: every error handling and path resolution choice has peer codebase backing (Hydra, pytest, Snakemake, vLLM, Docker Compose)
- Thermal gaps between experiments are research-motivated (GPU thermal inertia biases energy measurements) but specific default values (60s/300s) are pragmatic estimates, not from a specific paper
- Warmup (per-experiment, software settling) and thermal gaps (inter-experiment, hardware thermal baseline) are complementary, not conflicting

</specifics>

<deferred>
## Deferred Ideas

- Documentation write-up phase for M1, M2, and M3 — add as future milestone phases
- `--dry-run` flag showing full resolved config list — Phase 12 (CLI integration)
- VRAM estimation in dry-run mode — Phase 12 or later (requires model metadata from HF cache)

</deferred>

---

*Phase: 09-grid-expansion-and-studyconfig*
*Context gathered: 2026-02-27*
