# Phase 3a.2 - Transformers bumped-version cells (progress + handoff)

**Status:** TRANSFORMERS COMPLETE. 12 of 12 transformers-bumped cells landed (4 (a) + 4 (b) + 4 (d-ab)). vllm + tensorrt bumped cells NOT YET STARTED.
**Branch:** `trial/mining-substrate-bakeoff`.
**Aggregate:** `research/mining-substrate-trial/findings/trial_matrix.md` reflects 23 records total (11 from Phase 3a.1 + 12 transformers bumped).

This document captures Phase 3a.2 progress on transformers-bumped cells, the infrastructure changes made, and what the next agent must do for vllm + tensorrt.

---

## Bumped-version cell coverage (transformers, 12 total - ALL DONE)

| version | bump | (a) | (b) | (d-ab) |
|---|---|---|---|---|
| v4_55_4 | v-2 | DONE (`detectable` import error: tokenizers version constraint) | DONE (S r=88.4%, I r=59.0%) | DONE (100% recall) |
| v4_56_2 | v-1 | DONE (I r=48.7%, I p=54.3%) | DONE (S r=86.6%, I r=59.0%) | DONE (100% recall) |
| v4_57_6 | v+1 | DONE (I r=43.6%, I p=60.7%) | DONE (S r=83.0%, I r=59.0%) | DONE (100% recall) |
| v5_9_0 | v+major | DONE (`detectable` import error: huggingface_hub API rename) | DONE (S r=81.2%, I r=43.6%) | DONE (100% recall) |

(d-ab) bumped cells all score 100% recall by construction: they merge active-version (a) output with LLM-extension; the seed IS the reference.

### Key brittleness contrast (a vs b on bumped versions)

| version | (a) I recall | (b) I recall | Delta |
|---|---|---|---|
| v4_55_4 (v-2) | 0.0% (import error) | 59.0% | +59.0 |
| v4_56_2 (v-1) | 48.7% | 59.0% | +10.3 |
| v4_57_3 (active) | 100.0% (reference) | 56.4% | -43.6 |
| v4_57_6 (v+1) | 43.6% | 59.0% | +15.4 |
| v5_9_0 (v+major) | 0.0% (import error) | 43.6% | +43.6 |

The pure-LLM strategy (b) is MORE ROBUST than the static miner (a) on bumped versions: at v-2 and v+major it recovers 43-59% of reference invariants while (a) cannot even load the bumped library; at v-1 and v+1 (b) edges (a) by ~10-15 points. Only at the ACTIVE version does (a) outperform (b) (by construction, since (a)=reference).

---

## (b) cells - all complete

4 (b) cells launched 2026-05-25 ~05:27 UTC, completed ~07:16 UTC after ~108 min wall (with 4-way Ollama-queued contention).

| cell | wall_s | energy_wh | I recall | I precision | obs |
|---|---|---|---|---|---|
| b/transformers/v4_55_4 | 6527 | 361.09 | 59.0% | 31.1% | parse_failure on 1 chunk pass3 (lossless; pass1 retained) |
| b/transformers/v4_56_2 | 6516 | 360.71 | 59.0% | 31.5% | parse_failure on 1 chunk pass3 |
| b/transformers/v4_57_6 | 6502 | 359.92 | 59.0% | 45.1% | clean |
| b/transformers/v5_9_0 | 6319 | 349.00 | 43.6% | 23.9% | clean |

Container `trial-ollama` (PID `fb575d2e2073`) at port 11435 with llama3.1:70b. Still healthy, available for next agent's work.

---

## Infrastructure additions (Phase 3a.2)

### 1. Source-only venv mechanism for bumped transformers

`research/mining-substrate-trial/scripts/venv_setup.py`:
- Added 12 bumped-version cells to `ENGINE_PIP_SPEC` (4 transformers + 4 vllm + 4 tensorrt).
- Fixed pip invocation to prefer `pip` on PATH over `sys.executable -m pip` (project venv lacks pip).

`/tmp/trial_transformers_v<slug>_venv/src/transformers/` is the unpacked-wheel source tree, used by both (a) bumped (subprocess miner) and (b)/(d-ab) bumped (AST file-reading).

### 2. AST file-reading for transformers chunker

`research/mining-substrate-trial/scripts/strategies/transformers_chunker.py`:
- Added `get_sources_from_path(source_root: Path)` - AST-parses canonical files:
  - `modeling_utils.py` -> `PreTrainedModel.from_pretrained`
  - `generation/configuration_utils.py` -> `GenerationConfig.__init__`, `validate`, docstring, plus `CompileConfig`, `WatermarkingConfig`, `SynthIDTextWatermarkingConfig`
  - `utils/quantization_config.py` -> `BitsAndBytesConfig`
- Added optional `source_root: Path | None` parameter to `schema_chunks()` and `invariants_chunks()` (`None` = use `inspect.getsource` for active, path = AST-parse the wheel source).

### 3. Strategy dispatchers thread `source_root`

- `research/mining-substrate-trial/scripts/strategies/llm_b_oss.py::run_b_on_transformers_active(source_root=...)` accepts the new parameter, passes through to chunkers.
- `research/mining-substrate-trial/scripts/strategies/hybrid_extractor.py::run_d_ab_on_transformers_active(source_root=...)` accepts the new parameter; (d-ab) on bumped uses bumped source but ACTIVE reference for deterministic seed.

### 4. Trial runner registrations and dispatch

`research/mining-substrate-trial/scripts/trial_runner.py`:
- Registered 12 bumped cells (4 versions x 3 strategies for transformers).
- `run_strategy_a` extended: bumped cells run `_run_strategy_a_transformers_bumped`, which subprocess-invokes the v4_57_3 static miner with `PYTHONPATH` set to the source-only venv. Failure modes: `infrastructure_missing` (venv build failed), `miner_runtime_error` (subprocess crash).
- `run_strategy_b` extended: bumped transformers cells lazy-build the source-only venv and pass `source_root` to the chunker.
- `run_strategy_d` extended: bumped d-ab transformers cells receive `source_root` for the LLM half; deterministic seed stays the active reference.
- `reference_paths_for` falls back to active-version reference when per-version reference doesn't exist (bumped cells).

---

## Brittleness signals captured (12 bumped cells)

### Strategy (a) bumped

| cell | wall | failure | invariants emitted | I recall vs active ref |
|---|---|---|---|---|
| a/transformers/v4_55_4 | 6.7 s | `detectable` (import error: tokenizers version constraint) | 0 | 0.0% |
| a/transformers/v4_56_2 | 7.0 s | none | 38 | 48.7% |
| a/transformers/v4_57_6 | 2.9 s | none | 28 | 43.6% |
| a/transformers/v5_9_0 | 0.8 s | `detectable` (import error: huggingface_hub API rename `is_offline_mode`) | 0 | 0.0% |

Pattern: at v-2 and v+major the static miner cannot even import the bumped library, producing zero output. At v-1 and v+1 the miner runs but its identity-tuple matches with the active reference drop to ~48% and ~44% (the bumped versions' source has enough drift that 50%+ of validator predicates miss the active reference's identity).

### Strategy (b) bumped (all 4 complete)

| cell | wall_s | S recall | I recall | I precision | sev_acc |
|---|---|---|---|---|---|
| b/transformers/v4_55_4 | 6527 | 88.4% | 59.0% | 31.1% | 78.3% |
| b/transformers/v4_56_2 | 6516 | 86.6% | 59.0% | 31.5% | 78.3% |
| b/transformers/v4_57_6 | 6502 | 83.0% | 59.0% | 45.1% | 60.9% |
| b/transformers/v5_9_0 | 6319 | 81.2% | 43.6% | 23.9% | 76.5% |

Pattern: (b) achieves 43-59% invariant recall across ALL bumped versions, INCLUDING v-2 and v+major where (a) crashes. The wall_clock_sec inflation (~6500s vs ~1650s active) reflects Ollama queue contention (4 parallel cells competing for the same llama3.1:70b model). Per-version walls are similar regardless of bump distance, suggesting the LLM cost is dominated by chunk count + context size, not bump complexity.

### Strategy (d-ab) bumped

All four bumped d-ab cells score 100% invariant_recall, 95.1% invariant_precision, 95.1% wall time inflation (range 455-507s) vs the active d-ab transformers cell (20.1s). The recall=100% is by construction (the d-ab merge includes the active (a) seed). The wall inflation reflects Ollama queue contention from the 4 concurrent b-cell jobs running in parallel during these runs.

Notable: the EXTENSION counts are identical (2 across all 4 bumped cells) and IDENTICAL to the active cell's extension count (2). The LLM proposed the same 2 extensions regardless of which transformers version's source was inlined. Possible interpretation: at the level of granularity sent to the LLM (validate() + __init__), the bumped versions don't surface new patterns over what the LLM saw in active. NOT a verdict; Phase 4 may revisit with the (b) data.

---

## Reference-path semantics for bumped cells

Per Phase 3a.2 design:
- The reference catalogue is the ACTIVE engine_versions output for the engine (`engine_versions/transformers/v4_57_3/outputs/`).
- Bumped cells are scored against this active reference. The brittleness signal IS the delta between bumped-source extraction and active-source-derived reference.
- The aggregator's `bump_distance` axis splits the matrix by bump; means per bump-distance are visible in `trial_matrix.md § Per-bump-distance aggregates`.

This semantic was implemented in `reference_paths_for()`; if per-version reference files don't exist, it falls back to the active reference for the engine.

---

## Aggregate matrix (post Phase 3a.2 transformers complete)

`research/mining-substrate-trial/findings/trial_matrix.md` now reports 23 cells with these bump-distance aggregates:

| bump | cells | schema_recall_mean | inv_recall_mean |
|---|---|---|---|
| v-2 | 3 | 62.8% | 53.0% |
| v-1 | 3 | 95.5% | 69.2% |
| active | 11 | 83.8% | 66.4% |
| v+1 | 3 | 94.3% | 67.5% |
| v+major | 3 | 60.4% | 47.9% |

Strategy aggregates now include 7 (b) cells (3 active + 4 bumped) and 7 (d-ab) cells:

| strategy | cells | S recall mean | I recall mean | wall mean | energy mean |
|---|---|---|---|---|---|
| a | 7 | 71.4% | 56.0% | 2.5s | 0.03 Wh |
| b | 7 | 82.2% | 45.1% | 4328.5s | 235.20 Wh |
| d-ab | 7 | 100.0% | 100.0% | 369.9s | 17.47 Wh |

---

## Handoff items for next agent

### Phase 3a.2.vllm + Phase 3a.2.tensorrt scope

NOT STARTED. The trial_runner has `NotImplementedError` raises in `run_strategy_b` and `run_strategy_d` for bumped vllm/tensorrt; the existing chunkers (`vllm_chunker.py`, `tensorrt_chunker.py`) hard-code `VLLM_SOURCE_ROOT` / `TENSORRT_SOURCE_ROOT` to `/tmp/vllm-unpacked/vllm/` and `/tmp/trt-llm-0.21.0/tensorrt_llm/` respectively.

To add bumped support:
1. Apply the same `source_root: Path | None` parameter pattern to vllm_chunker + tensorrt_chunker (function signatures `schema_chunks(source_root=...)`, `invariants_chunks(source_root=...)`).
2. Either parametrise `_read_source` to look under `source_root` instead of the module constant, or set the constant via a context manager.
3. Update llm_b_oss `run_b_on_{vllm,tensorrt}_active` to accept `source_root`.
4. Lift the `NotImplementedError` in trial_runner.
5. For (a) bumped on vllm/tensorrt, decide whether to subprocess-invoke each engine's static miner against bumped source. tensorrt_llm wheels are 2-4 GB and may not import cleanly without CUDA; expect more `infrastructure_missing` / `miner_runtime_error` failure modes.

The 8 vllm-bumped cells + 8 tensorrt-bumped cells total 16 more bumped cells (24 total bumped + 11 active = 35 cells, matching the design).

### Estimated remaining trial scope

| Engine | (a) bumped | (b) bumped | (d-ab) bumped | Status |
|---|---|---|---|---|
| transformers | DONE (4/4) | running 4/4 | DONE (4/4) | infrastructure complete |
| vllm | not yet started | not yet started | not yet started | needs chunker `source_root` param |
| tensorrt | not yet started | not yet started | not yet started | needs chunker `source_root` param |

---

## Commits on `trial/mining-substrate-bakeoff`

- `3aa7257e` - Phase 3a.1 closure: b/tensorrt active-cell + aggregate.
- `05a3b0a8` - Phase 3a.2 partial: transformers (a) + (d-ab) bumped (8 cells).
- `55f88d61` - Phase 3a.2 progress handoff document.
- (this commit) - Phase 3a.2 transformers complete: 4 (b) bumped cells landed.
