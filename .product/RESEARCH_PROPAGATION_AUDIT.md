# Propagation Audit: `.planning/research/` → `.product/`

**Scope**: All 12 files in `.planning/research/` cross-referenced against `.product/decisions/`, `.product/designs/`, `.product/NEEDS_ADDRESSING.md`, and `.product/research/` (files 01–16).

**Date**: 2026-02-25

---

## 1. Fully Propagated

Findings that exist in `.product/` with a citation, decision update, or explicit annotation.

---

### 1.1 Library API — Union Return Type Reverted to Split Functions

**Source**: `UNIFIED-API-RESEARCH.md`, `SUMMARY.md §1.1`, `ARCHITECTURE.md §H`, `DECISION-AUDIT.md §2.5 CI-1`

**Status**: FULLY PROPAGATED

The research recommended reverting from `llem.run() -> ExperimentResult | StudyResult` to two separate functions. This was accepted and is now documented in:

- `.product/decisions/experiment-study-architecture.md` Q3 (updated 2026-02-25): states `run_experiment(ExperimentConfig) -> ExperimentResult` + `run_study(StudyConfig) -> StudyResult`, with a `> **Superseded (2026-02-25):**` annotation explaining the union-return revert. Cites `.planning/research/UNIFIED-API-RESEARCH.md` explicitly.
- `.product/CLAUDE.md` quick reference (already updated to `run_experiment(ExperimentConfig) -> ExperimentResult`, `run_study(StudyConfig) -> StudyResult`)
- `.product/designs/library-api.md` (status "v2.0", last updated 2026-02-25 per CLAUDE.md index)

**Note**: `DECISION-AUDIT.md` CI-1 and CI-4 also flag stale `llem study` / `run_experiment` / `run_study` ghost references in 15+ documents. The *decision* is propagated; the *terminology cleanup* across all downstream documents is a separate incomplete item (see §2 below).

---

### 1.2 zeus-ml Package Rename to zeus

**Source**: `STACK.md §1`, `SUMMARY.md §1.3`

**Status**: FULLY PROPAGATED

Addressed in:
- `.product/designs/packaging.md` (2026-02-25): `zeus = ["zeus>=0.13.1"]` with explicit comment `# note: PyPI package is 'zeus' not 'zeus-ml'`
- `.product/designs/energy-backends.md` (2026-02-25): "Package note: The Zeus PyPI package is `zeus` (not `zeus-ml`). The old `zeus-ml` package stopped at v0.11.0..."
- `.product/NEEDS_ADDRESSING.md` decision #49: "Updated all to current: zeus>=0.13.1..."

---

### 1.3 FLOPs Demoted from Primary Metric to Reference Metadata

**Source**: `PITFALLS.md §CP-3`, `SUMMARY.md §1.4`, `DECISION-AUDIT.md §2.14`

**Status**: FULLY PROPAGATED

Addressed in:
- `.product/decisions/flops-estimation.md` (2026-02-25): top-level research annotation explicitly cites `PITFALLS.md CP-3` and `SUMMARY.md §1.4`; states `flops_per_output_token` is "reference metadata for cross-model normalisation (not a primary comparison metric — FLOPs do not vary between deployment configurations)"
- `.product/NEEDS_ADDRESSING.md` decision #36: "FLOPs demotion: Demoted from primary metric to reference metadata | decisions/flops-estimation.md (Accepted)"

---

### 1.4 Thermal Floor Increased: 30s → 60s Configurable

**Source**: `PITFALLS.md §CP-2`, `SUMMARY.md §3.3`

**Status**: FULLY PROPAGATED

Addressed in:
- `.product/decisions/warmup-strategy.md` (2026-02-25): W3 option table updated with `> **Superseded (2026-02-25):**` annotation citing `PITFALLS.md CP-2`, MLPerf Power (arXiv:2410.12032), and A100/H100 thermal ramp research. New decision: 60s default, configurable down to 30s
- `.product/NEEDS_ADDRESSING.md` decision #37

---

### 1.5 Warmup: Full-Length Runs Replace 2-Token Stubs

**Source**: `PITFALLS.md §MP-3`, `SUMMARY.md §3.4`

**Status**: FULLY PROPAGATED

Addressed in:
- `.product/decisions/warmup-strategy.md` (2026-02-25): W2 table has `> **Superseded (2026-02-25):**` annotation with full rationale from `PITFALLS.md MP-3`. Default is now 3 full-length warmup runs
- `.product/NEEDS_ADDRESSING.md` decisions #37, #38

---

### 1.6 NVML Accuracy Corrected: ±5 Watts, Not ±5 Percent

**Source**: `PITFALLS.md §CP-1`, `SUMMARY.md §3.2`

**Status**: FULLY PROPAGATED

Addressed in:
- `.product/designs/energy-backends.md` (2026-02-25): full accuracy table replaced with `> **Revised (2026-02-25):**` annotation, conditional formula `accuracy_pct = 5W / mean_power_W * 100`, GPU-state table. Cites NVML API Reference and Burtscher et al. arXiv:2312.02741
- `.product/decisions/reproducibility.md` (2026-02-25): `reproducibility_notes` text corrected from "±5–15%" to "NVML accuracy is ±5W; percentage depends on power draw"

---

### 1.7 CPU-GPU Synchronisation as Hard Measurement Requirement

**Source**: `PITFALLS.md §mP-1`, `SUMMARY.md §Should-Add`

**Status**: FULLY PROPAGATED

Addressed in:
- `.product/designs/energy-backends.md` (2026-02-25): "CPU-GPU Synchronisation (hard requirement)" section with code example and source citation
- `.product/NEEDS_ADDRESSING.md` decision #40: "`torch.cuda.synchronize()` before every measurement stop"

---

### 1.8 GPU Persistence Mode: Pre-flight Warning

**Source**: `PITFALLS.md §UA-1`, `SUMMARY.md §Should-Add`

**Status**: FULLY PROPAGATED

Addressed in:
- `.product/designs/energy-backends.md` (2026-02-25): "GPU Persistence Mode (pre-flight warning)" section; recorded in `EnvironmentSnapshot.gpu_persistence_mode: bool`
- `.product/NEEDS_ADDRESSING.md` decision #41

---

### 1.9 Minimum Measurement Duration: Warn if <10 Seconds

**Source**: `PITFALLS.md §MP-5`, `SUMMARY.md §Should-Add`

**Status**: FULLY PROPAGATED

Addressed in:
- `.product/designs/energy-backends.md` (2026-02-25): "Minimum Measurement Duration (quality warning)" section; `measurement_warnings: list[str]` added to `ExperimentResult`
- `.product/NEEDS_ADDRESSING.md` decisions #42, #43

---

### 1.10 Access Control: Delegate to huggingface_hub, Drop .env

**Source**: `DECISION-AUDIT.md §2.26`, `SUMMARY.md §3.7`

**Status**: FULLY PROPAGATED

Addressed in:
- `.product/decisions/access-control.md` (2026-02-25): entire decision revised with `> **Superseded (2026-02-25):** ... Rejected: .env file pattern` annotation. New decision delegates to `huggingface_hub` auth chain. Cites `DECISION-AUDIT.md §2.26`
- `.product/NEEDS_ADDRESSING.md` decision #44

---

### 1.11 Output Contract: Always Subdirectory (Drop Flat-File for Single Experiments)

**Source**: `DECISION-AUDIT.md §2.16`, `SUMMARY.md §3.8`, `FEATURES.md §Gap 2`

**Status**: FULLY PROPAGATED

Addressed in:
- `.product/decisions/output-storage.md` (2026-02-25): J1 option table has `> **Superseded (2026-02-25):**` annotation; now "always subdirectory" for single and multi. Rationale includes time-series Parquet sidecar files as v2.0 scope
- `.product/NEEDS_ADDRESSING.md` decision #46

---

### 1.12 Power Time-Series Capture + Parquet Sidecar Files

**Source**: `FEATURES.md §Gap 2`, `SUMMARY.md §4 Table Stakes`

**Status**: FULLY PROPAGATED

Addressed in:
- `.product/decisions/versioning-roadmap.md` (2026-02-25): "Power time-series capture (Zeus PowerMonitor or direct NVML polling)" listed in v2.0 scope
- `.product/decisions/output-storage.md` (2026-02-25): sidecar Parquet for time-series data in decision summary
- `.product/NEEDS_ADDRESSING.md` decisions #47, #55, #56, #57: full time-series design (1 Hz sampling, Parquet schema, metric suite)

---

### 1.13 Prefill/Decode Phase-Split Energy Pulled into v2.0

**Source**: `FEATURES.md §Gap 1`, `SUMMARY.md §4 Table Stakes`

**Status**: FULLY PROPAGATED

Addressed in:
- `.product/decisions/versioning-roadmap.md` (2026-02-25): "Prefill/decode phase-split energy attribution" listed in v2.0 scope under "Parameter completeness (previously v2.3)"
- `.product/NEEDS_ADDRESSING.md` decision #61: v2.0 scope confirmed

---

### 1.14 Environment Metadata Capture Pulled into v2.0

**Source**: `FEATURES.md §Gap 4`, `SUMMARY.md §4 Table Stakes`

**Status**: FULLY PROPAGATED

Addressed in:
- `.product/decisions/versioning-roadmap.md` (2026-02-25): "environment metadata capture (EnvironmentSnapshot in every ExperimentResult)" in v2.0 scope
- `.product/decisions/reproducibility.md`: `EnvironmentSnapshot` schema decision exists (multi-GPU fields, CUDA version detection)
- `.product/decisions/multi-gpu.md`: `EnvironmentSnapshot` fields documented

---

### 1.15 v2.0–v2.4 Micro-Versioning Collapsed

**Source**: `FEATURES.md §versioning-roadmap`, `SUMMARY.md §Roadmap`

**Status**: FULLY PROPAGATED

Addressed in:
- `.product/decisions/versioning-roadmap.md` (2026-02-25): `> **Superseded (2026-02-25):**` annotation; micro-versions collapsed; v2.0 = complete tool. Former v2.2 (Docker multi-backend) and study resume are v2.0 milestones. Decision #61, #62 in NEEDS_ADDRESSING.md

---

### 1.16 Version Pins Updated (zeus, CodeCarbon, Transformers, vLLM, TRT-LLM)

**Source**: `STACK.md §1–2`, `SUMMARY.md §5`

**Status**: FULLY PROPAGATED

Addressed in:
- `.product/designs/packaging.md` (2026-02-25): `zeus>=0.13.1`, `codecarbon>=3.2.2`, `transformers>=5.0`, `vllm>=0.15`, `tensorrt-llm>=1.0`, `nvidia-ml-py>=13.590.48`, `scipy>=1.12`, `pytest-mock>=3.12`, `pytest-xdist>=3.5`
- `.product/NEEDS_ADDRESSING.md` decision #49

---

### 1.17 Build Tool: Poetry → uv

**Source**: `STACK.md §7`, `SUMMARY.md §5`

**Status**: FULLY PROPAGATED

Addressed in:
- `.product/designs/packaging.md` (2026-02-25): comment "Build tool: uv (replaces Poetry at v2.0)"
- `.product/NEEDS_ADDRESSING.md` decision #48

---

### 1.18 Three-Layer Config Naming Simplified to Two Sources + Auto-Capture

**Source**: `ARCHITECTURE.md §C–D`, `DECISION-AUDIT.md §2.3`, `SUMMARY.md §3.1`

**Status**: FULLY PROPAGATED

Addressed in:
- `.product/decisions/architecture.md` (2026-02-25): sub-decision C now reads "Two config sources + auto-capture"; sub-decision D annotation notes simplification from "three questions" framework; research annotation cites `ARCHITECTURE.md` and explains that "Layer 3" was output metadata, not a config layer

---

### 1.19 SGLang Accelerated to v2.2 Candidate

**Source**: `STACK.md §2`, `SUMMARY.md §Confirmed Decisions`

**Status**: FULLY PROPAGATED

Addressed in:
- `.product/decisions/additional-backends.md` (2026-02-25): SGLang updated with `> **Updated 2026-02-25:**` annotation citing STACK.md section 2; status is "v2.2 candidate" with rationale

---

### 1.20 scipy Added; pytest-mock and pytest-xdist Added

**Source**: `STACK.md §5 and §8`, `SUMMARY.md §5`

**Status**: FULLY PROPAGATED

Addressed in:
- `.product/designs/packaging.md` (2026-02-25): `scipy>=1.12`, `pytest-mock>=3.12`, `pytest-xdist>=3.5` all present

---

### 1.21 NVLink Interconnect Power Gap Documented

**Source**: `PITFALLS.md §MP-2`

**Status**: FULLY PROPAGATED

Addressed in:
- `.product/decisions/multi-gpu.md` (2026-02-25): research annotation added citing `PITFALLS.md MP-2`; text states "summing per-GPU NVML energy understates the true total energy by approximately 3–10% depending on NVLink traffic"

---

### 1.22 Docker Energy Overhead Documented

**Source**: `PITFALLS.md §MP-1`

**Status**: FULLY PROPAGATED (documented as a limitation)

Addressed in:
- `.product/decisions/docker-execution.md`: Docker as ephemeral-per-experiment is the decided model; NVML pass-through via container toolkit is noted. The ~1–3% overhead is within NVML uncertainty. Not a blocker.

---

### 1.23 Library-First, Composition Config, Option C Architecture, Subprocess Isolation, Unified CLI, Extras Model, Fixed-Count Warmup, Error Handling — All Confirmed

**Source**: `SUMMARY.md §2`, `ARCHITECTURE.md`, `DECISION-AUDIT.md §2.1–2.7, 2.12–2.13, 2.18`

**Status**: FULLY PROPAGATED (confirmed, no change needed)

All of these are already in `.product/decisions/` with correct content. The research confirmed them and no updates were needed.

---

### 1.24 Competitive Landscape + Web Platform Patterns

**Source**: `competitive-landscape.md`, `web-platform-patterns.md`, `industry-patterns.md`, `energy-measurement-tools.md`

**Status**: FULLY PROPAGATED (in `.product/research/`)

These files are primarily synthesised from `.product/research/` files 01–16 (noted in their README: "Synthesised from 9 raw research documents in `.planning/phases/04.5-strategic-reset/research/`"). Content exists in:
- `.product/decisions/product-vision.md`: competitive positioning (deployment vs model)
- `.product/decisions/web-platform.md`: static leaderboard → FastAPI+React → live; outbound worker model (ClearML pattern)
- `.product/decisions/versioning-roadmap.md`: v4.0 web platform

Overlap with `.product/research/` files: `04-deployment-patterns.md`, `07-ml-energy-ecosystem.md`, `09-broader-landscape.md` (see §5 below).

---

## 2. Partially Propagated

Findings where the decision was accepted but the propagation through the document set is incomplete or an internal inconsistency was introduced.

---

### 2.1 `llem study` Terminology Cleanup — Decision Made, Propagation Incomplete

**Source**: `DECISION-AUDIT.md CI-2`, `SUMMARY.md §1.2`

**Status**: PARTIALLY PROPAGATED

The decision to unify to `llem run` is fully accepted (experiment-study-architecture.md, cli-ux.md). The terminology was propagated to `CLAUDE.md` quick reference (now shows 2 commands + 1 flag), and `NEEDS_ADDRESSING.md` item #34 documents the required cleanup list.

**What remains incomplete**: The actual text of the listed files still contains `llem study`. A grep confirms residual occurrences in:

- `.product/decisions/study-execution-model.md` (4 occurrences — in historical Decision D and SSOT table, some in non-superseded context)
- `.product/decisions/cli-ux.md` (5 occurrences — mostly in the historical record / supersession table, which is correct; one questionable: line 58 `llem study (separate command)` in the table body still reads as present tense in the "rejected" column, which is technically correct but could confuse)
- `.product/decisions/experiment-study-architecture.md` (1 — historical reference in Option A description, correct)

**What's missing**: Item #34 in `NEEDS_ADDRESSING.md` lists 15 files but these docs have not yet been updated. The NEEDS_ADDRESSING.md correctly flags this but none of the downstream documents have been touched. In particular, the high-risk files identified by DECISION-AUDIT.md CI-2 that remain unupdated:
- `.product/decisions/experiment-isolation.md` — still contains `llem study / StudyRunner` terminology (lines confirmed by grep: 9 occurrences of `run_study` or `llem study`)
- `.product/decisions/backward-compatibility.md` — still lists `run_experiment`, `run_study` in stable exports table
- `.product/decisions/live-observability.md` — section header still `llem study (multi-experiment study)`
- `.product/decisions/documentation-strategy.md` — `studies.md — llem study: YAML format`
- `.product/decisions/progressive-disclosure.md` — `llem run study.yaml OR llem study study.yaml`
- `.product/decisions/installation.md` — Step 6 example

**Where it should be**: All files listed in NEEDS_ADDRESSING.md item #34.

**Blocking**: Yes — implementers reading these files will see conflicting CLI commands.

---

### 2.2 Q5 Output Contract Inconsistency: experiment-study-architecture.md vs output-storage.md

**Source**: Introduced during the 2026-02-25 session — not in any `.planning/research/` file directly, but surfaced by the output contract change from `DECISION-AUDIT.md §2.16`

**Status**: PARTIALLY PROPAGATED — internal contradiction introduced

`output-storage.md` J1 was revised on 2026-02-25 to "always subdirectory." However, `experiment-study-architecture.md` Q5 (also 2026-02-25) still states:

> `| **Q5. Output** | Single → flat JSON. Multi → {name}_{timestamp}/ subdirectory. |`

And in the body:
> `Single experiment → flat JSON file: results/{model}_{backend}_{timestamp}.json`

This contradicts the revised `output-storage.md` decision. One of these two documents is wrong.

**Where it should be**: The Q5 row in `experiment-study-architecture.md` and its "Design Details" section should be updated to match `output-storage.md` (always subdirectory), or a cross-reference/supersession annotation should be added.

**Blocking**: Yes — contradictory output contract documentation will cause implementation error.

---

### 2.3 Bootstrap CI: BCa Recommended but Not Formally Decided

**Source**: `PITFALLS.md §CP-5`, `STACK.md §8`, `SUMMARY.md §3.6`

**Status**: PARTIALLY PROPAGATED

The research recommended BCa bootstrap (not percentile), 2,000 resamples, and separating energy CI from latency CI. This is partially addressed:

- `.product/decisions/future-versions.md` (2026-02-25): "Bootstrap resampling (1000 resamples, **BCa method where feasible**)." But 1,000 resamples (not 2,000) and "where feasible" is vague.
- `.product/designs/packaging.md`: `scipy>=1.12` is present (which supports BCa)
- `.product/decisions/result-schema-migration.md`: bootstrap CI fields remain in `v2.1` bucket (not v2.0)

**What's missing**:
1. The distinction between energy CI (requires multi-cycle, bootstrap over per-cycle totals) vs latency CI (per-request) is not documented anywhere in `.product/`
2. The 2,000 resamples recommendation (vs current 1,000) is not addressed
3. BCa is noted as "where feasible" rather than the hard default

**Where it should be**: `decisions/future-versions.md` (BCa method and resamples clarification); possibly `designs/result-schema.md` (the CI field definitions).

**Blocking**: No for v2.0 (CIs deferred); yes for v2.0 result schema fields if energy CI fields are included.

---

### 2.4 CodeCarbon Pydantic v2 Compatibility — Risk Noted but Not Resolved

**Source**: `STACK.md §3`, `SUMMARY.md §Gaps`

**Status**: PARTIALLY PROPAGATED

`STACK.md §3` identifies that CodeCarbon historically had internal Pydantic v1 dependencies and that CodeCarbon 3.x compatibility with our Pydantic v2 config models must be verified. The packaging design now pins `codecarbon>=3.2.2` but includes no compatibility verification note.

**What's missing**: No note in `designs/packaging.md` or `decisions/installation.md` flagging this as a "must verify before shipping `[codecarbon]` extra." No open item in `NEEDS_ADDRESSING.md`.

**Where it should be**: `designs/packaging.md` as a comment or note; or `NEEDS_ADDRESSING.md` as a HIGH item.

**Blocking**: Potentially yes — if CodeCarbon 3.x has internal Pydantic v1 compat layers, it may conflict with our strict Pydantic v2 models at install or runtime.

---

### 2.5 ECC Memory Status, GPU Clock, GPU Power Limit, CPU Governor — Mentioned but Not Decided

**Source**: `PITFALLS.md §UA-2, UA-3, UA-4, mP-3`

**Status**: PARTIALLY PROPAGATED

These four EnvironmentSnapshot additions are mentioned in `.product/decisions/reproducibility.md` (2026-02-25) as a research annotation:

> "- ECC memory status — UA-2 / GPU clock frequency at measurement time — UA-3 / GPU power limit vs default — mP-3"

But they are framed as "candidates" with no decision made on whether to include them.

**What's missing**: A decision — include or defer? `designs/reproducibility.md` should specify which fields are in `EnvironmentSnapshot` at v2.0.

**Where it should be**: `designs/reproducibility.md` EnvironmentSnapshot schema.

**Blocking**: No for v2.0 correctness, but yes for reproducibility completeness (missing `gpu_power_limit_w` means throttled experiments are undetectable).

---

### 2.6 MIG Energy Measurement Warning — Not in Any .product/ Decision

**Source**: `PITFALLS.md §mP-2`

**Status**: PARTIALLY PROPAGATED

The PITFALLS.md notes NVML energy counters report at physical GPU level, not MIG instance level. This is listed in PITFALLS.md as "mentioned briefly in old PITFALLS.md but not addressed in any decision document." The new `.product/decisions/` set does not address it either.

**What's missing**: A pre-flight check decision. `designs/energy-backends.md` should mention MIG mode detection and the associated warning.

**Where it should be**: `designs/energy-backends.md` "Measurement Correctness Requirements" section; or `decisions/experiment-isolation.md`.

**Blocking**: No — affects accuracy documentation only.

---

### 2.7 `study-design-hash` / Top-Up Study Workflow Unspecified

**Source**: `DECISION-AUDIT.md CI-5`

**Status**: PARTIALLY PROPAGATED

The hash excludes `n_cycles` (enabling "top-up" studies with the same hash). This is decided and correct. But the actual user workflow for aggregating results from two study runs (run at n_cycles=3, then "top up" to n_cycles=5 via a second run) is unspecified. DECISION-AUDIT.md P3.3 flagged this.

**What's missing**: A section in `designs/study-resume.md` or `decisions/output-storage.md` explaining the top-up workflow.

**Where it should be**: `designs/study-resume.md`.

**Blocking**: No for implementation; yes for user documentation clarity.

---

### 2.8 cycle_order CLI Effective Default — Unresolved Inconsistency

**Source**: `DECISION-AUDIT.md CI-4`

**Status**: PARTIALLY PROPAGATED

`study-execution-model.md` SSOT table says `cycle_order` CLI effective default = `"interleaved"`. But Pydantic default = `"sequential"`. The CI-4 inconsistency (where is this default applied, CLI or Pydantic?) has not been resolved with a supersession annotation or clarifying text.

**What's missing**: A note clarifying that `cycle_order: "interleaved"` is applied at the CLI layer (not in the Pydantic model), and cross-referencing to `cli-ux.md`.

**Where it should be**: `decisions/study-execution-model.md` SSOT field table.

**Blocking**: Low — but could cause a bug if the implementer reads only the Pydantic default.

---

### 2.9 Pareto Frontier Extraction from StudyResult — Scoped but Not Designed

**Source**: `FEATURES.md §Gap 3`, `SUMMARY.md §Should-Add`

**Status**: PARTIALLY PROPAGATED

The research recommended adding `pareto_optimal: list[ExperimentResult]` (or equivalent method) to `StudyResult`. This appears in:

- `.product/decisions/future-versions.md` (2026-02-25): "Enables Pareto frontier visualisation (accuracy vs efficiency)" — but this is framed as v3.0 (lm-eval integration), not v2.0
- `.product/decisions/versioning-roadmap.md` v2.0 scope: no mention of Pareto extraction

**What's missing**: The FEATURES.md research argues Pareto extraction belongs in v2.0 StudyResult (not visualisation, just data structure). This was not pulled into v2.0 scope and no explicit decision to defer it was made.

**Where it should be**: `decisions/versioning-roadmap.md` or `decisions/experiment-study-architecture.md` — either include in v2.0 or explicitly defer with rationale.

**Blocking**: No for v2.0 operation; yes for competitive positioning (ML.ENERGY v3 uses this as a headline feature).

---

### 2.10 --dry-run Grid Preview — Scoped but Not Designed

**Source**: `FEATURES.md §Gap 6`, `SUMMARY.md §4 Table Stakes`

**Status**: PARTIALLY PROPAGATED

The feature is in v2.0 scope:
- `decisions/versioning-roadmap.md` (2026-02-25): "`--dry-run` grid preview for studies with VRAM estimation"
- `decisions/cli-ux.md`: `--dry-run` mentioned for config validation

**What's missing**: No design document exists specifying what `--dry-run` shows for studies (grid count, configurations, estimated runtime, VRAM check). The FEATURES.md recommends: resolve sweep → show grid → estimate runtime → validate all configs → exit. FEATURES.md item #4 in the "Create new decision" list. `SUMMARY.md §Priority 2` also lists "dry-run-design.md" as a needed file.

**Where it should be**: A new `decisions/dry-run-design.md` or extended section in `decisions/cli-ux.md`.

**Blocking**: Yes for implementation — implementers need a spec for what `--dry-run` outputs.

---

### 2.11 VRAM Pre-Flight Estimation — Scoped but Not Designed

**Source**: `FEATURES.md §Gap 7`, `SUMMARY.md §Should-Add`

**Status**: PARTIALLY PROPAGATED

In v2.0 scope per versioning-roadmap.md ("VRAM estimation" mentioned under `--dry-run`). But no decision or design specifies the estimation formula, the function signature, or which backends it applies to. `DECISION-AUDIT.md P2.4` also calls for tightening the sweep grammar edge case for dotted notation — related (VRAM estimation is part of dry-run pre-flight).

**What's missing**: The `estimate_vram()` function design from `10-sweep-validation-patterns.md` (product research) has not been promoted into a design spec.

**Where it should be**: `designs/experiment-isolation.md` or a new `designs/dry-run.md`.

**Blocking**: Yes for dry-run implementation.

---

### 2.12 Sweep Grammar Edge Case — `pytorch.attn.implementation` with `extra="forbid"`

**Source**: `DECISION-AUDIT.md §2.9 P2.4`

**Status**: PARTIALLY PROPAGATED

The audit flagged that a nested dotted key (`pytorch.attn.implementation`) that doesn't exist in `PyTorchConfig` with `extra="forbid"` should produce a `ValidationError` at sweep resolution, not silently pass. This is not addressed anywhere in `.product/decisions/config-architecture.md`.

**What's missing**: An edge case entry in the sweep grammar specification.

**Where it should be**: `decisions/config-architecture.md` sweep grammar edge-case table.

**Blocking**: Medium — silently wrong sweep grammar is a scientific integrity issue.

---

## 3. Not Propagated

Findings that appear only in `.planning/research/` with no trace in `.product/`.

---

### 3.1 TRT-LLM Engine Cache: No Version-Based Invalidation

**Source**: `DECISION-AUDIT.md §2.17 P2.3`

**Status**: NOT PROPAGATED

`decisions/docker-execution.md` defines `trt_compile_key()` but excludes `tensorrt_llm.__version__` from the cache key. When TRT-LLM updates (format change between versions), cached engines become invalid silently. The DECISION-AUDIT.md P2.3 recommends adding the TRT-LLM version to the cache key.

**What's missing**: Either add `tensorrt_llm.__version__` to `trt_compile_key()`, or document the known limitation with an explicit cache invalidation procedure.

**Where it should be**: `decisions/docker-execution.md` TRT compilation section; `designs/docker-execution.md`.

**Blocking**: Yes — stale TRT engine cache produces incorrect results across version upgrades. No trace in `.product/`.

---

### 3.2 Subprocess Sync Checkpoints Missing

**Source**: `ARCHITECTURE.md §Challenge 3`, `DECISION-AUDIT.md §I`

**Status**: NOT PROPAGATED

optimum-benchmark uses two sync checkpoints (verify child alive after spawn; verify GPU accessible after isolation setup) to detect child failures early rather than waiting for the full timeout. The llem subprocess isolation design (`decisions/experiment-isolation.md`, `designs/experiment-isolation.md`) has no sync checkpoint mechanism — a child that fails during import/setup causes the parent to wait for the full timeout.

**What's missing**: A note in `designs/experiment-isolation.md` or `decisions/experiment-isolation.md` about sync checkpoints, or an explicit decision to omit them.

**Where it should be**: `designs/experiment-isolation.md`.

**Blocking**: No for correctness; yes for robustness (silent timeout on failed import is a bad UX).

---

### 3.3 Device Isolation Monitoring (Foreign GPU Processes)

**Source**: `ARCHITECTURE.md §What is missing`

**Status**: NOT PROPAGATED

optimum-benchmark's `device_isolation: true` detects foreign processes using the GPU during measurement, which would contaminate energy readings. This feature is absent from `.product/` decisions and designs with no note about deferral or rejection.

**What's missing**: A decision — include pre-flight GPU process check, or explicitly note that other GPU workloads will contaminate measurements.

**Where it should be**: `designs/energy-backends.md` or `decisions/experiment-isolation.md`.

**Blocking**: No for correctness; yes for measurement integrity (undocumented contamination risk).

---

### 3.4 LoRA TRT-LLM Error Message — Exact Text Not Specified

**Source**: `DECISION-AUDIT.md §2.25 P2.5`

**Status**: NOT PROPAGATED

The LoRA adapter decision (`decisions/lora-adapter-support.md`) correctly states TRT-LLM raises a validation error if `lora:` is specified. But the audit recommended specifying the exact error message text (what constraint, how to pre-merge weights, where to point merged weights). No such specification exists.

**What's missing**: Error message design in `decisions/lora-adapter-support.md` or `designs/experiment-config.md`.

**Where it should be**: `decisions/lora-adapter-support.md`.

**Blocking**: Low — UX issue, not architecture.

---

### 3.5 `torch.backends.cudnn.deterministic` as Reproducibility Control

**Source**: `DECISION-AUDIT.md §P3.4`

**Status**: NOT PROPAGATED

The audit recommended documenting `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False` as standard PyTorch determinism controls. `decisions/reproducibility.md` mentions `random_seed: 42` for sampling but does not mention cuDNN determinism controls.

**What's missing**: A note in `decisions/reproducibility.md` or `designs/reproducibility.md`.

**Where it should be**: `decisions/reproducibility.md`.

**Blocking**: No — but missing this means non-deterministic GPU operations are not documented as a known variance source.

---

### 3.6 Static `reproducibility_notes` String Not Made Structured

**Source**: `DECISION-AUDIT.md §P2.6`

**Status**: NOT PROPAGATED

`decisions/reproducibility.md` still uses a static string for `reproducibility_notes`. The audit recommended a structured `measurement_variance_notes: dict` per metric (since variance depends on GPU model and workload). The 2026-02-25 annotation in `reproducibility.md` corrected the NVML accuracy claim but did not change the field from string to structured.

**What's missing**: A decision on field type — keep static string (simpler) or make structured.

**Where it should be**: `designs/result-schema.md`.

**Blocking**: Low — design detail.

---

### 3.7 lm-eval Packaging Restructure Simplifies v3.0 Integration

**Source**: `FEATURES.md §versioning-roadmap "QUESTIONABLE"`

**Status**: NOT PROPAGATED

The research noted that lm-eval's Dec 2025 packaging restructure (base package no longer includes transformers/torch; backends are separate extras) makes the v3.0 lm-eval integration simpler than originally estimated. No annotation or note in `decisions/versioning-roadmap.md` or `decisions/future-versions.md`.

**What's missing**: An informational note in `decisions/future-versions.md` about the changed packaging.

**Where it should be**: `decisions/future-versions.md` v3.0 section.

**Blocking**: No — informational only.

---

### 3.8 Progress Callback API for Library Users

**Source**: `ARCHITECTURE.md §What is missing`

**Status**: NOT PROPAGATED

The architecture audit noted that the design includes Rich progress for the CLI but has no callback mechanism for library users wanting programmatic progress events. optimum-benchmark has a similar gap. This was noted as a "should add" but appears nowhere in `.product/`.

**What's missing**: A decision — include progress callbacks in v2.0 or defer explicitly.

**Where it should be**: `designs/library-api.md` or `decisions/architecture.md`.

**Blocking**: No for v2.0 functionality; yes if library users need progress events for long studies.

---

### 3.9 `pip freeze` Fragility for Non-Pip Packages

**Source**: `DECISION-AUDIT.md §2.21 P2.6`

**Status**: NOT PROPAGATED

The audit flagged that `pip freeze` misses conda-installed packages and system libraries (CUDA, cuDNN). `reproducibility.md` mentions `pip list --format=freeze` but doesn't specify the multi-source detection approach. `designs/reproducibility.md` has `torch.version.cuda` detection for CUDA (noted in NEEDS_ADDRESSING.md item #10, fixed). But the general fragility of pip-only capture for system libraries is not documented.

**What's missing**: A note in `designs/reproducibility.md` that conda-installed packages and system libraries (CUDA driver, cuDNN) are captured via direct API calls (`torch.version.cuda`, `nvmlSystemGetDriverVersion()`), not via pip freeze.

**Where it should be**: `designs/reproducibility.md`.

**Blocking**: Low — affects reproducibility completeness documentation.

---

## 4. New Issues Surfaced

Contradictions between `.planning/research/` findings and current `.product/` state that were not in the original DECISION-AUDIT.md scope.

---

### 4.1 Q5 Internal Contradiction: experiment-study-architecture.md vs output-storage.md

**Issue**: `experiment-study-architecture.md` Q5 (2026-02-25) states "Single → flat JSON." `output-storage.md` J1 (also 2026-02-25) was revised to "always subdirectory." These contradict each other. Both were updated in the same session.

**Impact**: Blocking. Implementation built from `experiment-study-architecture.md` will produce a flat JSON file for single experiments. Implementation built from `output-storage.md` will produce a subdirectory. The Parquet sidecar requirement makes `output-storage.md` logically correct; `experiment-study-architecture.md` Q5 needs a supersession annotation.

**Fix**: Add `> **Superseded (2026-02-25):** Q5 output contract revised in output-storage.md — always subdirectory (not flat JSON for single experiments). Single-experiment runs produce a timestamped subdirectory with result.json + timeseries.parquet.` to Q5 in `experiment-study-architecture.md`.

---

### 4.2 NEEDS_ADDRESSING.md Decision #50 — "All Result Fields at v2.0" Contradicts result-schema-migration.md

**Issue**: NEEDS_ADDRESSING.md decision #50 (2026-02-25) states: "Collapse v2.0/v2.1 field split — ship complete: CIs, baseline, per-device, schema_version". But `decisions/result-schema-migration.md` still shows bootstrap CI fields in `v2.1` bucket. If CIs are v2.0, the migration table is stale.

**Impact**: Medium. Implementation will be confused about whether to include CI fields in the v2.0 result schema.

**Fix**: Update `decisions/result-schema-migration.md` to move bootstrap CI fields from v2.1 to v2.0, or annotate the decision with a supersession note.

---

### 4.3 backward-compatibility.md Stable Exports Still List `run_experiment`, `run_study` — Now Correct but Needs Verification

**Issue**: `DECISION-AUDIT.md CI-1` identified `backward-compatibility.md` as listing `run_experiment` and `run_study` as stale pre-unification names. After the revert to split API, these are now the *correct* function names — but the file may still have context that references the old union `run()` in a way that needs cleanup.

**Impact**: Low — verify that `backward-compatibility.md` reflects the current split API correctly, not the superseded union `run()`.

---

## 5. Overlap with .product/research/

`.planning/research/` files that duplicate content already in `.product/research/` (files 01–16).

| `.planning/research/` File | Content | Covered by `.product/research/` |
|---|---|---|
| `competitive-landscape.md` | Tool comparison matrix; ML.ENERGY vs llem positioning | `09-broader-landscape.md` (970 lines, 38+ tools) |
| `energy-measurement-tools.md` | Zeus vs CodeCarbon accuracy; NVML mechanics; EnergyBackend Protocol | `03-codecarbon-zeus-energy.md`, `05-zeus-deep-dive.md`, `06-llenergymeasure-vs-zeus.md`, `08-energy-plugin-architecture.md` |
| `industry-patterns.md` | Library→CLI→web; pip extras; local-first results; outbound worker | `01-lm-eval-harness.md`, `02-mlflow-architecture.md`, `04-deployment-patterns.md` |
| `web-platform-patterns.md` | ML.ENERGY static JSON leaderboard; FastAPI+React; ClearML pattern | `04-deployment-patterns.md §3`, `07-ml-energy-ecosystem.md §4` |
| `README.md` | Index of 4 synthesis files | N/A — meta file only |

The `.planning/research/README.md` itself acknowledges this: "Synthesised from 9 raw research documents in `.planning/phases/04.5-strategic-reset/research/`." These four files are a summary layer over the earlier research, not new primary research. Their findings were incorporated into the `.product/` decisions during the strategic reset phase.

The five files that contain **genuinely new or extended research** not in `.product/research/` 01–16:
- `SUMMARY.md` — meta-synthesis with specific blocking items
- `FEATURES.md` — fresh 2026-02-25 peer feature audit (8 tools)
- `PITFALLS.md` — measurement methodology audit with cited papers (arXiv:2312.02741, MLPerf Power)
- `DECISION-AUDIT.md` — systematic audit of every decision against peer evidence (900+ lines)
- `ARCHITECTURE.md` — fresh architecture audit against optimum-benchmark, lm-eval, Zeus
- `STACK.md` — version audit against current PyPI as of 2026-02-25
- `UNIFIED-API-RESEARCH.md` — 10-tool survey on API return type patterns

---

## Summary Table

| Finding | Source | Status | Blocking? | `.product/` Target |
|---|---|---|---|---|
| Union return type reverted | UNIFIED-API-RESEARCH.md | Fully propagated | — | experiment-study-architecture.md Q3 |
| zeus-ml → zeus rename | STACK.md | Fully propagated | — | packaging.md |
| FLOPs demoted to metadata | PITFALLS.md CP-3 | Fully propagated | — | flops-estimation.md |
| Thermal floor 30s → 60s | PITFALLS.md CP-2 | Fully propagated | — | warmup-strategy.md |
| Full-length warmup default | PITFALLS.md MP-3 | Fully propagated | — | warmup-strategy.md |
| NVML accuracy ±5W not ±5% | PITFALLS.md CP-1 | Fully propagated | — | energy-backends.md |
| CPU-GPU sync hard requirement | PITFALLS.md mP-1 | Fully propagated | — | energy-backends.md |
| GPU persistence mode warning | PITFALLS.md UA-1 | Fully propagated | — | energy-backends.md |
| Min measurement 10s warning | PITFALLS.md MP-5 | Fully propagated | — | energy-backends.md |
| Auth → huggingface_hub | DECISION-AUDIT.md §2.26 | Fully propagated | — | access-control.md |
| Always subdirectory output | DECISION-AUDIT.md §2.16 | Fully propagated | — | output-storage.md |
| Power time-series / Parquet | FEATURES.md Gap 2 | Fully propagated | — | output-storage.md, versioning-roadmap.md |
| Phase-split energy → v2.0 | FEATURES.md Gap 1 | Fully propagated | — | versioning-roadmap.md |
| Environment metadata → v2.0 | FEATURES.md Gap 4 | Fully propagated | — | versioning-roadmap.md |
| Micro-versions collapsed | FEATURES.md | Fully propagated | — | versioning-roadmap.md |
| Version pins updated | STACK.md | Fully propagated | — | packaging.md |
| Poetry → uv | STACK.md §7 | Fully propagated | — | packaging.md |
| Three-layer → two sources | ARCHITECTURE.md §C-D | Fully propagated | — | architecture.md |
| SGLang → v2.2 candidate | STACK.md §2 | Fully propagated | — | additional-backends.md |
| scipy, pytest-mock, pytest-xdist | STACK.md §5,8 | Fully propagated | — | packaging.md |
| NVLink power gap documented | PITFALLS.md MP-2 | Fully propagated | — | multi-gpu.md |
| `llem study` → `llem run` rename | DECISION-AUDIT.md CI-2 | **Partially propagated** | YES | 15+ docs still unupdated |
| Q5 output contract inconsistency | New issue | **Partially propagated** | YES | experiment-study-architecture.md Q5 |
| Bootstrap BCa vs percentile | PITFALLS.md CP-5 | **Partially propagated** | No (deferred) | future-versions.md |
| CodeCarbon Pydantic compat | STACK.md §3 | **Partially propagated** | POTENTIALLY | packaging.md |
| ECC, clock, power limit fields | PITFALLS.md UA-2/3, mP-3 | **Partially propagated** | No | designs/reproducibility.md |
| MIG energy warning | PITFALLS.md mP-2 | **Partially propagated** | No | designs/energy-backends.md |
| Top-up study workflow | DECISION-AUDIT.md CI-5 | **Partially propagated** | No | designs/study-resume.md |
| cycle_order CLI default | DECISION-AUDIT.md CI-4 | **Partially propagated** | Low | study-execution-model.md |
| Pareto frontier extraction | FEATURES.md Gap 3 | **Partially propagated** | No | versioning-roadmap.md |
| --dry-run grid preview design | FEATURES.md Gap 6 | **Partially propagated** | YES | No design doc exists |
| VRAM pre-flight estimation | FEATURES.md Gap 7 | **Partially propagated** | YES (for dry-run) | No design doc exists |
| Sweep grammar edge case | DECISION-AUDIT.md §2.9 | **Partially propagated** | Medium | config-architecture.md |
| TRT cache version invalidation | DECISION-AUDIT.md §2.17 | **Not propagated** | YES | docker-execution.md |
| Subprocess sync checkpoints | ARCHITECTURE.md §I | **Not propagated** | No | designs/experiment-isolation.md |
| Device isolation monitoring | ARCHITECTURE.md | **Not propagated** | No | designs/energy-backends.md |
| LoRA TRT error message text | DECISION-AUDIT.md §2.25 | **Not propagated** | No | lora-adapter-support.md |
| cudnn.deterministic control | DECISION-AUDIT.md §P3.4 | **Not propagated** | No | decisions/reproducibility.md |
| Static reproducibility_notes | DECISION-AUDIT.md §P2.6 | **Not propagated** | No | designs/result-schema.md |
| lm-eval packaging simplification | FEATURES.md | **Not propagated** | No | decisions/future-versions.md |
| Progress callback for library users | ARCHITECTURE.md | **Not propagated** | No | designs/library-api.md |
| pip freeze fragility for system libs | DECISION-AUDIT.md §2.21 | **Not propagated** | No | designs/reproducibility.md |
| result-schema-migration v2.1 CI fields | New issue (#50 vs migration table) | **New contradiction** | Medium | result-schema-migration.md |

---

## Priority Ordering for Resolution

**Must resolve before implementation begins (blocking):**

1. **Q5 output contract** — add supersession annotation to `experiment-study-architecture.md` Q5 pointing to `output-storage.md`. One-line fix.
2. **TRT-LLM engine cache version invalidation** — add `tensorrt_llm.__version__` to `trt_compile_key()` or document explicit invalidation procedure in `docker-execution.md`.
3. **`--dry-run` grid preview design** — write a spec (new decision section in `cli-ux.md` or new `dry-run-design.md`) covering what is shown for studies before any CLI implementation work.
4. **VRAM pre-flight estimation** — promote the formula from `10-sweep-validation-patterns.md` into a design stub for `--dry-run`.
5. **`llem study` terminology cleanup** — the 15 files listed in NEEDS_ADDRESSING.md item #34 need updating before implementation, or implementers will build against the wrong CLI surface.

**Resolve before v2.0 ships (high):**

6. **CodeCarbon Pydantic v2 compatibility** — add verification note to `packaging.md`; add NEEDS_ADDRESSING.md item.
7. **result-schema-migration.md** — reconcile with NEEDS_ADDRESSING.md decision #50 (CIs in v2.0 vs v2.1 migration table).
8. **ECC/clock/power limit fields** — make a binary include/defer decision on each UA-2/UA-3/mP-3 item in `designs/reproducibility.md`.
9. **Sweep grammar edge case** — add `pytorch.attn.implementation` ValidationError case to `config-architecture.md` edge-case table.

**Informational / can defer:**

10. Subprocess sync checkpoints — note in `designs/experiment-isolation.md`
11. Device isolation monitoring — note in `designs/energy-backends.md`
12. BCa resamples (1,000 → 2,000) — update `future-versions.md`
13. Bootstrap energy CI vs latency CI distinction — add to `future-versions.md`
14. `cudnn.deterministic` — add to `decisions/reproducibility.md`
15. LoRA TRT error message — add to `lora-adapter-support.md`
16. Top-up study workflow — add to `designs/study-resume.md`
17. cycle_order CLI default — clarify in `study-execution-model.md`
18. Pareto frontier — make explicit defer decision in `versioning-roadmap.md`
19. Progress callback — decide in `designs/library-api.md`
20. lm-eval packaging simplification — note in `future-versions.md`
