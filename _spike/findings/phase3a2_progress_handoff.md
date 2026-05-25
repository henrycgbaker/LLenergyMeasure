# Phase 3a.2 - Transformers bumped-version cells (progress + handoff)

**Status:** PARTIAL. 8 of 12 transformers-bumped cells landed; 4 (b) cells still running in background.
**Branch:** `trial/mining-substrate-bakeoff`.
**Aggregate:** `_spike/findings/trial_matrix.md` reflects all completed records (19 cells total: 11 from Phase 3a.1 + 8 new bumped).

This document captures Phase 3a.2 progress on transformers-bumped cells, the infrastructure changes made, and what the next agent must do.

---

## Bumped-version cell coverage (transformers, 12 total)

| version | bump | (a) | (b) | (d-ab) |
|---|---|---|---|---|
| v4_55_4 | v-2 | DONE (`detectable` import error) | running in background | DONE (100% recall) |
| v4_56_2 | v-1 | DONE (I r=48.7%, I p=54.3%) | running in background | DONE (100% recall) |
| v4_57_6 | v+1 | DONE (I r=43.6%, I p=60.7%) | running in background | DONE (100% recall) |
| v5_9_0 | v+major | DONE (`detectable` import error) | running in background | DONE (100% recall) |

(d-ab) bumped cells all score 100% recall by construction: they merge active-version (a) output with LLM-extension; the seed IS the reference.

---

## (b) cells still running

4 (b) cells launched 2026-05-25 ~05:27 UTC. PIDs and log paths:

| cell | PID | log |
|---|---|---|
| b/transformers/v4_55_4 | 1550698 (parent), 1550713 (worker) | `/tmp/cell_b_transformers_v4_55_4.log` |
| b/transformers/v4_56_2 | 1550699 (parent), 1550712 (worker) | `/tmp/cell_b_transformers_v4_56_2.log` |
| b/transformers/v4_57_6 | 1550700 (parent), 1550715 (worker) | `/tmp/cell_b_transformers_v4_57_6.log` |
| b/transformers/v5_9_0 | 1550701 (parent), 1550714 (worker) | `/tmp/cell_b_transformers_v5_9_0.log` |

Current progress (chunks completed; total ~47 per cell):
- v4_55_4: 31/47
- v4_56_2: 32/47
- v4_57_6: 31/47
- v5_9_0: 31/47

Estimated remaining wall time at observed pace (4-way Ollama-queued contention): ~45 min per cell, 4 cells in parallel = ~45 min total until all complete.

Container `trial-ollama` (PID `fb575d2e2073`) at port 11435 with llama3.1:70b. Confirmed healthy.

---

## Infrastructure additions (Phase 3a.2)

### 1. Source-only venv mechanism for bumped transformers

`_spike/scripts/venv_setup.py`:
- Added 12 bumped-version cells to `ENGINE_PIP_SPEC` (4 transformers + 4 vllm + 4 tensorrt).
- Fixed pip invocation to prefer `pip` on PATH over `sys.executable -m pip` (project venv lacks pip).

`/tmp/trial_transformers_v<slug>_venv/src/transformers/` is the unpacked-wheel source tree, used by both (a) bumped (subprocess miner) and (b)/(d-ab) bumped (AST file-reading).

### 2. AST file-reading for transformers chunker

`_spike/scripts/strategies/transformers_chunker.py`:
- Added `get_sources_from_path(source_root: Path)` - AST-parses canonical files:
  - `modeling_utils.py` -> `PreTrainedModel.from_pretrained`
  - `generation/configuration_utils.py` -> `GenerationConfig.__init__`, `validate`, docstring, plus `CompileConfig`, `WatermarkingConfig`, `SynthIDTextWatermarkingConfig`
  - `utils/quantization_config.py` -> `BitsAndBytesConfig`
- Added optional `source_root: Path | None` parameter to `schema_chunks()` and `invariants_chunks()` (`None` = use `inspect.getsource` for active, path = AST-parse the wheel source).

### 3. Strategy dispatchers thread `source_root`

- `_spike/scripts/strategies/llm_b_oss.py::run_b_on_transformers_active(source_root=...)` accepts the new parameter, passes through to chunkers.
- `_spike/scripts/strategies/hybrid_extractor.py::run_d_ab_on_transformers_active(source_root=...)` accepts the new parameter; (d-ab) on bumped uses bumped source but ACTIVE reference for deterministic seed.

### 4. Trial runner registrations and dispatch

`_spike/scripts/trial_runner.py`:
- Registered 12 bumped cells (4 versions x 3 strategies for transformers).
- `run_strategy_a` extended: bumped cells run `_run_strategy_a_transformers_bumped`, which subprocess-invokes the v4_57_3 static miner with `PYTHONPATH` set to the source-only venv. Failure modes: `infrastructure_missing` (venv build failed), `miner_runtime_error` (subprocess crash).
- `run_strategy_b` extended: bumped transformers cells lazy-build the source-only venv and pass `source_root` to the chunker.
- `run_strategy_d` extended: bumped d-ab transformers cells receive `source_root` for the LLM half; deterministic seed stays the active reference.
- `reference_paths_for` falls back to active-version reference when per-version reference doesn't exist (bumped cells).

---

## Brittleness signals captured so far (8 bumped cells)

### Strategy (a) bumped

| cell | wall | failure | invariants emitted | I recall vs active ref |
|---|---|---|---|---|
| a/transformers/v4_55_4 | 6.7 s | `detectable` (import error: tokenizers version constraint) | 0 | 0.0% |
| a/transformers/v4_56_2 | 7.0 s | none | 38 | 48.7% |
| a/transformers/v4_57_6 | 2.9 s | none | 28 | 43.6% |
| a/transformers/v5_9_0 | 0.8 s | `detectable` (import error: huggingface_hub API rename `is_offline_mode`) | 0 | 0.0% |

Pattern: at v-2 and v+major the static miner cannot even import the bumped library, producing zero output. At v-1 and v+1 the miner runs but its identity-tuple matches with the active reference drop to ~48% and ~44% (the bumped versions' source has enough drift that 50%+ of validator predicates miss the active reference's identity).

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

## Aggregate matrix (post Phase 3a.2 partial)

`_spike/findings/trial_matrix.md` now reports 19 cells with these bump-distance aggregates:

| bump | cells | schema_recall_mean | inv_recall_mean |
|---|---|---|---|
| v-2 | 2 | 50.0% | 50.0% |
| v-1 | 2 | 100.0% | 74.4% |
| active | 11 | 83.8% | 66.4% |
| v+1 | 2 | 100.0% | 71.8% |
| v+major | 2 | 50.0% | 50.0% |

(v-2/v+major numbers are skewed by d-ab=100% pairing with a=0% per cell. With more cells from (b) bumped, these aggregates will shift.)

---

## Handoff items for next agent

### Wait + capture the 4 (b) bumped cells

Poll for these JSONs to appear in `_spike/findings/trial_scores/`:
- `b__transformers__v4_55_4.json`
- `b__transformers__v4_56_2.json`
- `b__transformers__v4_57_6.json`
- `b__transformers__v5_9_0.json`

Estimated ETA ~45 min from 2026-05-25 06:15 UTC. After completion:
1. Refresh aggregate: `uv run python -m _spike.scripts.trial_aggregate`.
2. Stage + commit the 4 score JSONs + their `trial_runs/b/transformers/<slug>/` dirs.
3. Push.

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

The 4 remaining (b) bumped cells will land on `_spike/findings/trial_scores/` and `trial_runs/b/transformers/<slug>/` when they finish.
