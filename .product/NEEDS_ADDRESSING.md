# Needs Addressing

Items that need human review or resolution before Phase 5 implementation can proceed safely.
Grouped by severity: **[BLOCKING]** must resolve before coding; **[HIGH]** resolve soon;
**[MEDIUM]** best to resolve in Phase 5; **[LOW]** deferred is fine.

---

## [HIGH] Harmonisation Items (added 2026-02-25)

### 33. Align decisions/ with MADR 4.0 Lite

Adopt a lightweight version of [MADR 4.0](https://adr.github.io/madr/) for all `decisions/` files.
Current files mix decision content (options, rationale, consequences) with design content (code
stubs, Pydantic models, YAML schemas). During harmonisation:

- Strip implementation detail from `decisions/` → move to `designs/`
- Ensure each decision file can stand alone without code stubs
- Adopt MADR 4.0 section headings (Context, Decision Drivers, Considered Options, Decision Outcome, Consequences)
- Keep it "Lite" — skip the metadata fields we don't need (Consulted, Informed, Confirmation)
- Test: "does this file answer WHAT and WHY without HOW?" → if HOW is present, extract to `designs/`

### 34. Propagate `llem study` → `llem run` rename across all docs

Decision Audit found `llem study` / `run_study()` ghost references in 15+ active documents
after the CLI was unified to `llem run`. Must update: `experiment-isolation.md`,
`study-execution-model.md`, `live-observability.md`, `output-storage.md`,
`documentation-strategy.md`, `versioning-roadmap.md`, `installation.md`, and others.

### 35. Move design detail out of experiment-study-architecture.md

The `Design Details (Resolved)` section (load_yaml code stub, runner delegation pattern,
result unwrapping code) belongs in `designs/`. The decision doc should reference `designs/`
for implementation detail, per MADR separation principle.

---

## ~~[BLOCKING] Sweep Grammar — RESOLVED 2026-02-19~~

### ~~1. Sweep Grammar for Backend-Specific Params~~ — DECIDED

**Decision: Option A — dotted notation.** `sweep: {pytorch.batch_size: [1, 8, 32]}` adds a
dimension to that backend's grid only. Backends have independent Cartesian products.

| # | Option | Resolution |
|---|--------|-----------|
| A | Dotted notation: `pytorch.batch_size: [1, 8, 32]` | **CHOSEN** — most ergonomic; 4× less YAML than Option B for backend-specific sweeps |
| B | Backend-lock: `sweep:` valid only for universal params | Rejected — forces verbose `experiments:` list for common backend-specific sweep case |
| C | Ignore-missing: skip silently | Rejected — silent failure is never correct for a measurement tool |

Full resolved semantics (edge cases, algorithm, validation rules):
→ `decisions/config-architecture.md` § "Sweep Grammar Decision"
→ `designs/study-yaml.md` § "Mode A — Grid Sweep"

**Unblocks**: `grid.py`, `StudyConfig` validation logic.

---

### ~~2. `config_hash` — Layer 3 Fields~~ — RESOLVED 2026-02-19

**Decision:** Exclude Layer 3 fields. Field renamed `measurement_config_hash`.

Layer 3 fields (`datacenter_pue`, `grid_carbon_intensity_gco2_kwh`) have moved to user config
(Layer 1) and are no longer in ExperimentConfig (Layer 2). They are stored with results as
scientific context — not part of the measurement identity. Per-experiment overrides also excluded.

→ Updated in `designs/result-schema.md` § `measurement_config_hash`
→ Updated in `designs/library-api.md`

---

## [HIGH] Inconsistencies (Cross-Doc)

### 3. `StudyResult.schema_version`: `"1.0"` vs `"2.0"`

~~`schema_version: str = "1.0"` for `StudyResult`~~ — **FIXED 2026-02-19**

Updated `designs/result-schema.md` — `StudyResult.schema_version` default is now `"2.0"`.

---

### ~~4. `n_cycles` Default~~ — RESOLVED 2026-02-19

**Decision:** CLI effective default = **3** (from standard profile). Library API default = **1** (Pydantic).

`llem study file.yaml` with no `execution:` block → 3 cycles. Statistically sound by default.
`run_study(config)` library API → 1 cycle (Pydantic model default). Library callers control this explicitly.
Precedence documented in `decisions/cli-ux.md` § "Execution Profiles".

---

### ~~5. `llem status` / `campaign` references~~ — FIXED 2026-02-19

`decisions/cli-ux.md` fully rewritten with `llem config` and "Study pre-flight:".
`designs/cli-commands.md` confirmed clean (no `campaign` references).

---

### ~~6. `campaign.py` reference in docker-execution.md~~ — FIXED 2026-02-19

New `decisions/docker-execution.md` does not reference `campaign.py`.
Note: `src/llenergymeasure/orchestration/campaign.py` confirmed dead code in Phase 4 audit
(1,524 lines scheduled for removal in Phase 5). Target architecture uses `study/runner.py`.

---

### ~~7. `StudyManifest` vs `StudyResult` Disambiguation~~ — FIXED 2026-02-19

Disambiguation note added to both `designs/study-resume.md` § "StudyManifest vs StudyResult"
and `designs/result-schema.md` § "New: Study Result Schema". Relationship is now explicit.

---

## ~~[HIGH] Side-Effect Contradiction~~ — RESOLVED 2026-02-19

### ~~8. "Library is Side-Effect Free" vs `run_study()` Writing Manifest~~

**Decision:** `run_study(config, *, output_dir=<from user config>)` — default writes manifest;
`output_dir=None` suppresses all disk writes. Manifest co-located with results (no separate dotdir).

API details:
- `output_dir` not passed → reads `output.results_dir` from user config; `./results/` if absent
- `output_dir=Path(...)` → writes to explicit directory
- `output_dir=None` → side-effect-free (no manifest, no disk writes)
- Manifest lives at `{output_dir}/{study_name}/study_manifest.json` alongside experiment results
- No `.checkpoint/` dotdir — manifest is study data, not a hidden system cache (MLflow pattern)

Reasoning: `run_study()` is the one principled exception to side-effect-free. Long-running studies
(hours to days) need recovery. Library users have the same need as CLI users. Optuna, Ray Tune,
MLflow all write state during long runs without apology. Full rationale in `designs/library-api.md`.

→ Updated in `designs/library-api.md` (core principle + `run_study` signature)

---

## [MEDIUM] Missing Detail / Ambiguities

### ~~9. Synthetic Dataset `n` semantics~~ — NOTED in dataset.md

`SyntheticDatasetConfig.n` generates the pool; experiment-level `n` selects from it.
Validate `n ≤ dataset.n` at config parse time. Deferred to Phase 5 implementation.

---

### ~~10. CUDA Version Detection~~ — FIXED 2026-02-19

Multi-source detection implemented in `designs/reproducibility.md`:
Priority: `torch.version.cuda` → `/usr/local/cuda/version.txt` → `nvcc --version` → `None`.

---

### ~~11. Multi-GPU `EnvironmentSnapshot`~~ — FIXED 2026-02-19

`designs/reproducibility.md` updated: `gpu_names: list[str]`, `gpu_vram_gb: list[float]`,
`gpu_count: int`. Single-GPU case: `gpu_count=1`, single-item lists.

---

### ~~12. `runner: <profile-name>` in study.yaml~~ — DEFERRED

`runner: auto` + per-machine user config achieves the goal. Named runner profiles add complexity
for marginal benefit. Removed `runner:` field from `study.yaml` entirely (2026-02-19). Runner
selection is purely Layer 1 / CLI / env var — not in study file.

---

### ~~13. TRT Engine Cache Key Fields~~ — FIXED 2026-02-19

`trt_compile_key()` function defined in `decisions/docker-execution.md` § "TensorRT Engine
Compilation". Key: `model × precision × tp_size × pp_size × max_batch_size × max_seq_len ×
quantization`. `builder_opt_level` excluded (same engine, different build optimisation).

---

### ~~14. Docker Entrypoint~~ — FIXED 2026-02-19

Container entrypoint uses library API directly (not CLI re-entry). Thin Python script calls
`ExperimentOrchestrator(config).run()`. Updated in `decisions/docker-execution.md`.

---

### ~~15. CI Path-Filter Trigger~~ — FIXED 2026-02-19

`dorny/paths-filter@v3` pattern documented in `designs/testing.md` § "CI Path-Filter".
Phase 5 CI setup task: replace any `contains(github.event.pull_request.changed_files, ...)`
references with the `dorny/paths-filter` action.

---

### 16. AIEnergyScore Dataset — File Not Yet Created

**Where**: `designs/dataset.md`

The `aienergyscore.jsonl` file must be downloaded from HuggingFace Hub (pinned commit),
processed, and committed to the package. This is a Phase 5 task but requires planning:
1. Identify the correct commit hash to pin
2. Define the 1000-prompt sampling strategy (equal from WikiText, OSCAR, UltraChat)
3. Commit to `src/llenergymeasure/datasets/builtin/aienergyscore.jsonl` with provenance header

---

### ~~17. Carbon Table Coverage~~ — OBSOLETE 2026-02-25

> **Superseded (2026-02-25):** Static carbon intensity table dropped from base package entirely. Carbon intensity lookup delegated to CodeCarbon (`[codecarbon]` optional extra). Base package retains only user-specified `grid_carbon_intensity_gco2_kwh` override with simple arithmetic. See `decisions/carbon-intensity.md`.

---

### 18. `cold_start` Semantics with Docker

**Where**: `designs/study-yaml.md` (session 4 update), `decisions/docker-execution.md`

`cold_start: true` in study.yaml is confirmed. But the Docker cold start semantics are explicitly
marked "TBD" — it's unclear whether this means:
- Process-level model unload within a running container
- Container restart (full cold start)

**Recommendation**: For v2.0 (local only), `cold_start: true` = `del model; torch.cuda.empty_cache()`.
For v2.2 (Docker), `cold_start: true` = fresh `docker run` per experiment (ephemeral lifecycle
already achieves this). Clarify in docker-execution.md when v2.2 is designed.

---

### 19. `vLLM` and `TensorRT-LLM` Parameter Completeness

**Where**: `designs/experiment-config.md` (VLLMConfig, TensorRTConfig)

Both configs have `TODO: Complete parameter list — see PARAM-04 (v2.3)`. The current parameter
lists are incomplete stubs. This is intentional (deferred to v2.3) but should be noted:
Phase 5 implementation will have incomplete backend config coverage. v2.0 ships with whatever
is listed; users can use `extra: {param: value}` for unlisted params as an escape hatch.

---

### ~~20. FLOPs Estimation — No Decision Doc Exists~~ — DOCUMENTED 2026-02-19

**Where**: not yet documented anywhere in decisions/ or designs/

FLOPs is listed as a primary output metric in the product vision, but there is no decision
or design doc covering how it is estimated. Key open questions:

- Algorithm: palm-style (`6 * N * D`)? Roofline-based? `torch.flop_count_analysis`?
- What "N" means for QLoRA/LoRA adapters (trainable params only, or full model?)
- How to handle attention FLOPs with varying sequence lengths across prompts in a batch
- Multi-GPU: total FLOPs across all devices, or per-device?
- FLOPs vs MFU (Model FLOPs Utilisation) — do we report both?
- Verification against theoretical peak GPU FLOPs

**Blocking**: Not Phase 5 blocking, but `core/metrics.py` cannot be fully implemented without this.
Research agent running → see `research/14-flops-warmup-lora-multiGPU.md` when complete.
Decision stub needed: `decisions/flops-estimation.md`

---

### ~~21. Warmup Strategy — No Decision Doc Exists~~ — DOCUMENTED 2026-02-19

**Where**: `designs/experiment-config.md` § warmup fields, but no decisions/ doc

The warmup algorithm is referenced but not specified. Key open questions:

- CV (coefficient of variation) threshold: 0.05? 0.10? Configurable?
- Window size for rolling CV calculation: last N latencies, or all-to-date?
- Minimum warmup prompts regardless of CV convergence
- Maximum warmup cap (prevent infinite warmup on noisy hardware)
- What to do on CV convergence failure — error, warn and proceed, or discard run?
- Whether per-backend overrides are needed (vLLM JIT warmup vs PyTorch eager)

**Blocking**: Not Phase 5 blocking (can use a sensible default), but statistical
correctness depends on getting this right. Decision stub needed: `decisions/warmup-strategy.md`

---

### ~~22. LoRA / Adapter Support — No Decision Doc Exists~~ — DECIDED 2026-02-19 (defer to v2.1)

**Where**: not referenced anywhere in decisions/ or designs/

The current codebase has no LoRA support. The question is whether v2.0 should support it:

- LoRA/QLoRA adapters are how most researchers actually run customised models in production
- The core question: is a LoRA adapter a different "model" or a different "config" parameter?
  (This affects ExperimentConfig design — `model: base + adapter_path:` vs `model: adapter_hf_id`)
- FLOPs accounting: which parameters count? (Affects FLOPs metric — see item 20 above)
- Backend support: PyTorch (PEFT library) vs vLLM (native adapter support) vs TRT (none)

**Recommendation**: Defer LoRA to v2.1. v2.0 should only measure base models to keep scope
tight. Add a clear "not supported in v2.0" note in experiment-config.md.
Decision stub needed: `decisions/adapter-support.md`

---

### ~~23. Tensor Parallelism / Multi-GPU — No Decision Doc Exists~~ — DOCUMENTED 2026-02-19

**Where**: `EnvironmentSnapshot` notes `gpu_count`, but no decisions/ or designs/ doc

Multi-GPU inference is standard for models >7B at research scale. Key open questions:

- `tp_size` (tensor parallelism) is in TensorRTConfig — but what about PyTorch/vLLM?
- vLLM supports `tensor_parallel_size` natively; PyTorch requires `device_map="auto"` or Accelerate
- FLOPs accounting across devices: sum across GPUs or just primary device?
- `EnvironmentSnapshot` captures `gpu_count` — does `gpu_names` list all devices or just the primary?
- Is multi-GPU in scope for v2.0 or explicitly deferred?

**Recommendation**: Support multi-GPU passively (detect and record it) in v2.0, but don't
add new TP/PP sweep params beyond what the backends already support. Explicit multi-GPU
sweep support → v2.3 (parameter taxonomy release).
Decision stub needed: `decisions/multi-gpu.md`

---

### ~~25. Backend Install Hint Discrepancy: Planning Docs vs Code~~ — FIXED 2026-02-19

**Where**: `designs/observability.md` and `decisions/cli-ux.md` vs `config/backend_detection.py`

`designs/observability.md` and `decisions/cli-ux.md` both show:
```
vllm       not installed → pip install llenergymeasure[vllm]
tensorrt   not installed → pip install llenergymeasure[tensorrt]
```
But `backend_detection.py::get_backend_install_hint()` currently returns:
```
"Docker recommended — see docs/deployment.md"
```
for both `vllm` and `tensorrt`.

This is an actual code ↔ docs discrepancy. The code is right for the v2.0 deployment model
(Docker-first for vLLM/TRT). The docs need updating.

**Fix needed**: Update the `llem config` output examples in `designs/observability.md` and
`decisions/cli-ux.md` to show Docker hints for vLLM/TRT, not pip install hints.
HIGH priority — affects what users see and directly contradicts the deployment model.

Also: document in `decisions/cli-ux.md` that `KNOWN_BACKENDS` ordering is deliberate
(`pytorch → vllm → tensorrt` = PyTorch is the default in zero-config mode).

---

### 24. Preservation Audit Files — Stale CLI Names

**Where**: `preservation_audit/` directory (multiple files)

Many `preservation_audit/` files contain stale CLI names (`llem status`, `llem experiment`,
`campaign.yaml`) — these files document v1.x codebase state and should preserve the original
names as historical record. However, any "Phase 5 recommendation" or "planning gap" section
within those files should use current names.

**Fix needed**: Audit the "Phase 5 Recommendation" sections in preservation_audit/ files
and update any forward-looking CLI references. Historical description sections: leave as-is.
This is LOW priority — preservation audit is reference material, not a specification.

---

## ~~[LOW] Stale References / Naming — FIXED 2026-02-19~~

### ~~20. `designs/study-yaml.md` Still Uses "Campaign" in Several Places~~ — FIXED

All `campaign.yaml` → `study.yaml`, `llem campaign` → `llem study`, "campaigns" → "studies"
updated throughout. Historical naming notes (top of file, deprecation examples) intentionally
preserved.

### ~~21. `decisions/cli-ux.md` References `llem campaign` in Multiple Places~~ — NO ACTION NEEDED

The remaining `llem campaign` reference in cli-ux.md is in the "What Was Cut and Why" table,
correctly documenting that `llem campaign` was **renamed** to `llem study`. Historical record.

### ~~22. `decisions/installation.md` Step 6 Still Says `llem campaign`~~ — FIXED

Progressive Disclosure Install Flow updated: `llem run`, `llem study`, `llem config`.
Docker setup section updated to reflect v2.2 timing and Docker Hub (not GitHub Container Registry).

---

---

## ~~[BLOCKING] Command Names — RESOLVED 2026-02-19~~

All command naming decisions resolved in session 6. See `decisions/cli-ux.md` (fully updated).

| # | Decision | Resolution |
|---|----------|-----------|
| 23 | `llem experiment` → `llem run` | **`llem run`** — confirmed. Dominant CLI idiom (lm-eval, mlflow, dbt, cargo, poetry all use `run`). |
| 24 | `llem study` rename? | **Keep `llem study`** — `sweep` rejected (W&B optimization connotation). Redundancy solved by descriptive filenames (`llem study batch-size-effects.yaml`). |
| 25 | Status/check command | **`llem config`** — replaces both `llem status` (passive display) and `llem init` (setup). Passive: shows env + suggests config. `--init` planned v2.2+. |
| 26 | `llem init` add/keep rejection | **Folded into `llem config`**. Re-runnable config display replaces one-time init. |
| 27 | Library API: `run_study()` vs `run_sweep()` | **`run_study()`** — config type is `StudyConfig`, semantically accurate for measurement domain. `run_experiment()` + `run_study()` confirmed public API. |

**Additional decision (session 6):** User config expanded with `output:`, `measurement:`, `ui:`, `advanced:` sections. Layer 3 fields (carbon_intensity, PUE) moved from ExperimentConfig to user config defaults. See `designs/user-config.md` and `decisions/architecture.md` (new Separation of Concerns section).

---

## Summary: Decisions Needed Before Phase 5 Coding Starts

| # | Issue | Blocking Phase 5? |
|---|---|---|
| 23–27 | Command names (run/sweep/check/init + library API) | YES — touches CLI skeleton, all command files, library __init__ |
| 1 | Sweep grammar (backend-specific params) | YES — grid.py |
| 2 | config_hash Layer 3 exclusion | YES — result-schema |
| 4 | n_cycles default (1 vs 3) | YES — StudyConfig |
| 8 | Side-effect contradiction (run_study manifest) | YES — library-api |
| 3 | StudyResult schema_version "1.0" vs "2.0" | Quick fix |

---

---

## [DEFERRED] Structural Migration + Rewrite All .planning Root Docs

**Added**: 2026-02-19

### What Phase 4.5 Actually Produced

Phase 4.5 is NOT just "Phase 5 pre-work". It produced a **complete product vision,
architecture refactor, and roadmap** for the entire project. The `decisions/` and `designs/`
directories here are the canonical planning artifacts for all future work — not just phase 5.

This means `PROJECT.md`, `ROADMAP.md`, and `STATE.md` in `.planning/` are not just out of
date; they reflect a **different product model** (wrong command names, wrong CLI count,
`campaign` terminology, 9-command CLI, pre-decision API shapes, old phase scope).

**Do not treat these files as authoritative. Source of truth: `decisions/` and `designs/` here.**

### COMPLETE — Migration done 2026-02-19

All artifacts have been migrated to `.planning/product/redesign-planning/` (this directory).
The old `.planning/phases/04.5-strategic-reset/` directory is now empty.

Next steps are tracked in `TODO.md` in this directory:
- Step 0: User manual inspection + feedback
- Step 3: Rewrite PROJECT.md, ROADMAP.md, STATE.md from scratch
- Step 5: Replan Phase 5+ with GSD against rewritten docs

---

---

## [HIGH] Issues Found During DRY Refactor (2026-02-19)

### ~~26. Default Dataset Inconsistency: `alpaca` vs `aienergyscore`~~ — FIXED 2026-02-20

`designs/cli-commands.md` and `decisions/installation.md` updated to `aienergyscore`.
Pydantic model default in `designs/experiment-config.md` was already correct.
`decisions/cli-ux.md` line 19 already said AIEnergyScore — no change needed.

---

### ~~27. `designs/experiment-config.md` Still References Layer 3 Fields as ExperimentConfig Fields~~ — FIXED 2026-02-20

**Decision**: Layer 3 fields (`datacenter_location`, `grid_carbon_intensity_gco2_kwh`,
`datacenter_pue`) removed from `ExperimentConfig`. `experiment.yaml` is infrastructure-agnostic.

Rationale: no peer tool (MLflow, CodeCarbon, W&B, Nextflow) puts infrastructure context in the
experiment/run definition. Baking PUE or carbon intensity into experiment.yaml couples the
portable scientific definition to a specific physical context — breaks portability.

Override mechanism: env vars (`LLEM_CARBON_INTENSITY`, `LLEM_DATACENTER_PUE`,
`LLEM_DATACENTER_LOCATION`) for one-off overrides; user config for persistent machine-local defaults.

→ `designs/experiment-config.md`: fields removed from schema; YAML example updated; `config_hash` simplified (exclude set removed — no longer needed).
→ `decisions/architecture.md`: Layer 3 section and Override Precedence section updated; `datacenter_location` added to field placement table.

---

### ~~28. `compute_config_hash` in `designs/experiment-config.md` Has Unresolved TODO~~ — FIXED 2026-02-20

TODO comment removed. `compute_config_hash` simplified: exclude set removed (Layer 3 fields no
longer in ExperimentConfig), docstring updated to explain why no exclusion is needed.

---

### 29. `designs/observability.md` Has Conflicting `llem config` Output (TODO Not Resolved)

**Where**: `designs/observability.md` lines 230–232

The TODO comment in the `llem config` output section asks whether to show pip install hints
or Docker recommendations for vLLM/TRT. This was supposed to be resolved by NEEDS_ADDRESSING.md
item 25 (FIXED 2026-02-19), but the TODO comment remains in the file.

The resolved answer from item 25: show **Docker recommended** hints for vLLM/TRT, not pip
install. The "first use" example added during this session (2026-02-19 DRY refactor) still
shows pip install hints, which is wrong.

**Fix**: Update `designs/observability.md` `llem config` output examples to show Docker hints
for vLLM/TRT (not pip install). Remove the TODO comment.

---

### ~~30. Execution Model — Three Interrelated Open Decisions~~ — DECIDED 2026-02-20

→ `research/16-execution-profiles-patterns.md` (peer tool survey)
→ `decisions/study-execution-model.md` (full decisions + SSOT field inventory)

**Background**: What started as an ambiguity in `decisions/cli-ux.md` § "Execution Profiles"
surfaced three deeper interrelated design questions that must be decided together:

**Decision A — Measurement protocol placement**: Where do `n_cycles`, `cycle_order`,
`config_gap_seconds`, `cycle_gap_seconds` live?
- Option 1: `execution:` block in study.yaml (portable, self-documenting, current design)
- Option 2: User config only (machine-local, not portable)
- Option 3: CLI flags only (no YAML home)

**Decision B — Hashing semantics**: Does measurement protocol go in the study hash?
- Option 1: Yes — hash includes n_cycles (same study, different cycles = different hash)
- Option 2: No — hash covers experimental design only (parameter space); n_cycles is metadata

**Decision C — experiment.yaml ↔ study.yaml relationship**: Are they the same format?
- Option 1: `llem run` is syntactic sugar for `llem study` with one experiment and n_cycles=1
- Option 2: experiment.yaml and study.yaml are genuinely different formats; study embeds/references experiment configs

**Key insight from discussion (2026-02-20)**: study.yaml is semantically about "what parameter
space did I explore?" — not "how many times did I run it for rigour?". n_cycles and gap_seconds
are both measurement protocol (meta-concern), not experimental design. If so, the research
recommendation to split them (n_cycles → study YAML, gaps → user config) was wrong — they
belong together regardless of where that is.

**Unblocks**: `decisions/study-execution-model.md`, `designs/study-yaml.md`,
`designs/user-config.md`, `decisions/cli-ux.md` (Execution Profiles section)

---

### 31. `designs/experiment-config.md` YAML Examples Don't Show `lora:` or `warmup:` Blocks

**Where**: `designs/experiment-config.md` § "YAML Examples"

The YAML examples section shows single-backend and infrastructure context examples, but
doesn't show examples with `lora:` (newly added v2.0 feature) or `warmup:` configuration.
These should be added so users can see the complete syntax.

**Fix**: Add LoRA and warmup YAML examples in Phase 5 (low priority — designs doc, not blocking).

---

### ~~32. Experiment / Study Architecture — Full ADR Pending Decision~~ — DECIDED 2026-02-25

**Decision:** Option C accepted. See `decisions/experiment-study-architecture.md`.

- `ExperimentConfig` = pure data type (one measurement point, zero study knowledge)
- `StudyConfig` = thin resolved container (`list[ExperimentConfig]` + `ExecutionConfig`)
- Sweep resolution at YAML parse time, single runner `_run(StudyConfig)`
- Unified `llem run` command (no separate `llem study`)
- Unified `llem.run()` library API (no separate `run_experiment()` / `run_study()`)
- CLI surface: `llem run` + `llem config` + `llem --version` (2 commands + 1 flag)

→ `decisions/experiment-study-architecture.md` updated to Accepted
→ `decisions/cli-ux.md` updated (3 commands → 2 + flag)
→ `decisions/study-execution-model.md` Decision D superseded

---

## Decisions Made — Session 2026-02-25 (Harmonisation Completion)

### Research propagation decisions

| # | Decision | Resolution | Propagated to |
|---|----------|-----------|---------------|
| 36 | FLOPs demotion | Demoted from primary metric to reference metadata | `decisions/flops-estimation.md` (Accepted) |
| 37 | Thermal floor | 30s → 60s configurable (MLPerf-aligned); publication profile enforces 60s | `decisions/warmup-strategy.md` (Accepted) |
| 38 | Warmup strategy | 2-token → full-length default (3 runs); warms KV cache + decode path + thermal | `decisions/warmup-strategy.md` (Accepted) |
| 39 | Docker inter-experiment gaps | Auto-skip in Docker mode (container startup = natural thermal reset) | `decisions/warmup-strategy.md` |
| 40 | CPU-GPU sync | Hard requirement: `torch.cuda.synchronize()` before every measurement stop | `designs/energy-backends.md` |
| 41 | GPU persistence mode | Pre-flight warning (not error); recorded in EnvironmentSnapshot | `designs/energy-backends.md` |
| 42 | Min measurement duration | Warn if <10s; flag in measurement_warnings. Don't loop/enforce | `designs/energy-backends.md` |
| 43 | Measurement quality | `measurement_warnings: list[str]` in ExperimentResult. Result files primary, CLI summary secondary | `designs/energy-backends.md` |
| 44 | Access control | Delegate to `huggingface_hub` auth chain; drop `.env` pattern | `decisions/access-control.md` (revised) |
| 45 | Project-level config | No `.llem.yaml`. Two surfaces only: user config + experiment YAML. 0/6 peers use project config | No doc change needed (already correct) |
| 46 | Output contract | Always subdirectory (unified for single + study). Supports sidecar Parquet | `decisions/output-storage.md` (revised) |
| 47 | Time-series storage | Sidecar Parquet files. Full design deferred to dedicated discussion | `decisions/output-storage.md` (noted) |
| 48 | Build tool | Poetry → uv | `designs/packaging.md` |
| 49 | Version pins | Updated all to current: zeus>=0.13.1, codecarbon>=3.2.2, transformers>=5.0, vllm>=0.15, etc | `designs/packaging.md` |
| 50 | All result fields at v2.0 | Collapse v2.0/v2.1 field split — ship complete: CIs, baseline, per-device, schema_version | `designs/result-schema.md` (pending update) |

### Config & CLI decisions

| # | Decision | Resolution |
|---|----------|-----------|
| 51 | Backend default | Default to pytorch when multiple installed. Auto-detect when single installed |
| 52 | Escape hatch | Keep but rename `extra:` → `passthrough_kwargs:` to avoid Pydantic `extra="forbid"` confusion |
| 53 | Warmup/baseline scope | Keep in ExperimentConfig (per-experiment). User config can provide defaults but YAML always wins |
| 54 | ~~Execution profiles~~ | **Revised (2026-02-25):** Drop `--profile` entirely. 0/5 peers use named rigour profiles — just use flags/YAML fields directly. `n_cycles`, `cycle_order`, `gap_seconds`, `thermal_floor` set in study YAML `execution:` block; CLI flags override; effective values recorded in results. Execution environment profiles deferred to post-v2.0 |

### Time-series design decisions

| # | Decision | Resolution |
|---|----------|-----------|
| 55 | TS metrics | Full suite: gpu_power_w, gpu_temp_c, gpu_utilization_pct, cumulative_energy_j, gpu_memory_used_mb, tokens_per_sec (at each sample point) |
| 56 | Sampling rate | 1 Hz default (configurable). Sufficient for energy integration, low overhead |
| 57 | Storage format | Sidecar Parquet file per experiment (decided #47). Schema: `timestamp_s, gpu_power_w, gpu_temp_c, gpu_utilization_pct, cumulative_energy_j, gpu_memory_used_mb, tokens_per_sec` |

### Testing decisions

| # | Decision | Resolution |
|---|----------|-----------|
| 58 | Testing scope | Comprehensive integration tests for ALL 3 backends (not just PyTorch). Real GPU tests with `@pytest.mark.gpu`. CI needs GPU runners for each backend |

### Documentation decisions

| # | Decision | Resolution |
|---|----------|-----------|
| 59 | `--profile` flag | Dropped entirely. 0/5 peers use named rigour profiles. Use direct flags/YAML fields. Execution environment profiles deferred post-v2.0 |
| 60 | Documentation scope | MkDocs standard: README + quickstart + config reference + auto-generated API docs + 2-3 how-to guides (single experiment, study sweep, interpreting results). Matches lm-eval scope |

### Still to discuss (this session)

- [x] Testing strategy — scope decided (#58), CI infra TBD
- [x] Documentation — MkDocs standard (#60)
- [x] v2.0 scope reality check — confirmed: all items are v2.0. Incremental milestones within v2.0 (#61)

### Scope & versioning decisions

| # | Decision | Resolution |
|---|----------|-----------|
| 61 | v2.0 scope | All features confirmed v2.0. Incremental milestones within v2.0 to ship usable product between checkpoints. No separate v2.2 — Docker multi-backend and study resume are v2.0 milestones, not separate versions |
| 62 | v2.2 elimination | All former "v2.2" items (Docker multi-backend, study resume, Docker images, `llem config --init`) are now v2.0 milestones. Only v3.0 (lm-eval) and v4.0 (web) remain as separate future versions |

---

## See Also

**[preservation_audit/INDEX.md](preservation_audit/INDEX.md)** — Codebase preservation audit (2026-02-19).
47 features (23 original + 24 newly discovered) that exist in the current code but are silently absent
from the plans. Individual files in `preservation_audit/` — one per feature, with exact code
references, planning gap analysis, and Phase 5 recommendations.

