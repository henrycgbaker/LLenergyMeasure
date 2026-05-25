# Phase 3a.2 - vllm bumped-version cells (progress + handoff)

**Status:** VLLM COMPLETE. 12 of 12 vllm-bumped cells landed (4 (a) + 4 (b) + 4 (d-ab)). tensorrt bumped cells NOT YET STARTED.
**Branch:** `trial/mining-substrate-bakeoff`.
**Aggregate:** `research/mining-substrate-trial/findings/trial_matrix.md` reflects 35 records total (11 active + 12 transformers-bumped + 12 vllm-bumped).

Phase 2.6 (namespace canonicalisation + chunker parametrisation + bumped-cell wiring) closed before this phase ran.

---

## Phase 2.6 rubric correction (b/tensorrt + d-ab/tensorrt active rescore)

Background: the b/tensorrt active cell reported invariant_recall=0.0% on the original Phase 3a.1 scoring because the LLM emitted identity tuples under `tensorrt_llm.<field>` while the reference catalogue uses `tensorrt.<field>`. Same engine, same field; disjoint identities.

Fix: `research/mining-substrate-trial/scripts/trial_scoring.canonicalise_namespace(ns, engine)` collapses `tensorrt_llm.X` -> `tensorrt.X` at identity-extraction time. Applied symmetrically on reference and cell. Pass-through for transformers + vllm (already consistent).

Effect (active cells rescored against existing trial_runs output; NO LLM re-extraction):

| cell | before | after (post-canonicalisation) |
|---|---|---|
| b/tensorrt/v0_21_0 | I_r=0.0% I_p=0.0% (intersection 0/31) failure=`silent` | I_r=25.8% I_p=20.5% (intersection 8/31) failure=`none` |
| d-ab/tensorrt/v0_21_0 | I_r=100.0% I_p=79.5% (by construction) | I_r=100.0% I_p=79.5% (unchanged - reference IS the seed) |

The b/tensorrt invariant_recall lifted from 0.0% to 25.8% - the remaining gap is REAL (cell emits 39 invariants, reference has 31; 8 intersect after canonicalisation). The LLM finds different predicates (e.g. `max_records gt` vs reference `max_records lt`) and different fields (`auto_parallel`, `calib_*` not in reference). That is the honest cell performance after the rubric fix.

`b/tensorrt` failure_mode lifted from `silent` to `none` - the cell is no longer classified as silent-fail because intersection is non-zero.

---

## Bumped-version cell coverage (vllm, 12 total - ALL DONE)

| version | bump | (a) | (b) | (d-ab) |
|---|---|---|---|---|
| v0_6_0 | v-2 | DONE (`detectable`: msgspec import error) | DONE (S r=75.6%, I r=34.6%) | DONE (extension=0; 100% recall by construction) |
| v0_6_6_post1 | v-1 | DONE (`detectable`: msgspec import error) | DONE (S r=57.8%, I r=34.6%) | DONE (extension=0; 100% recall) |
| v0_9_2 | v+1 | DONE (`detectable`: msgspec import error) | DONE (S r=87.4%, I r=30.8%) | DONE (extension=2; 100% recall) |
| v0_19_1 | v+major | DONE (`detectable`: msgspec import error) | DONE (S r=0.0%, I r=0.0%) `silent` from chunker fail | DONE (extension=0; 100% recall) |

(d-ab) bumped cells score 100% recall by construction: they merge the active (a) seed with LLM-extension; the seed IS the reference. Bumped d-ab differs in extension count - the LLM looked at bumped source for new patterns.

### Key brittleness contrast (a vs b on bumped vllm versions)

| version | (a) I recall | (b) I recall | Delta |
|---|---|---|---|
| v0_6_0 (v-2) | 0.0% (msgspec import) | 34.6% | +34.6 |
| v0_6_6_post1 (v-1) | 0.0% (msgspec import) | 34.6% | +34.6 |
| v0_7_3 (active) | 100.0% (reference) | 38.5% | -61.5 |
| v0_9_2 (v+1) | 0.0% (msgspec import) | 30.8% | +30.8 |
| v0_19_1 (v+major) | 0.0% (msgspec import) | 0.0% (chunker silent fail) | 0 (BOTH fail) |

Pattern: the static miner (a) cannot import bumped vllm at all - vllm has a hard transitive-dep on `msgspec` that the source-only venv doesn't install. (a) fails identically on all 4 bumps (deterministic, observable, `detectable`). (b) survives v-2 through v+1 (~31-38% recall), then collapses at v+major because vllm 0.19.1 restructured `config.py` into a `config/` subdirectory - the chunker's `_read_source("config.py")` returns empty, producing a single `source_extraction_failed` chunk.

The v+major collapse is a CHUNKER brittleness (file layout assumption) more than an LLM brittleness. Same engine, same library, drastically restructured code surface.

---

## (b) cells - all complete

4 (b) cells launched 2026-05-25 ~09:32 UTC, completed ~10:42 UTC after ~70 min wall (with 4-way Ollama-queued contention).

| cell | wall_s | energy_wh | S_r | I recall | I precision | sev_acc | failure_modes | obs |
|---|---|---|---|---|---|---|---|---|
| b/vllm/v0_6_0 (v-2) | 3969 | 209.0 | 75.6% | 34.6% | 14.1% | 100% | none | multipass; 9 inv chunks; pass2_dropped=1, pass3_added=22 |
| b/vllm/v0_6_6_post1 (v-1) | 4167 | 220.5 | 57.8% | 34.6% | 15.5% | 100% | none | multipass; 10 inv chunks; pass3_added=12 |
| b/vllm/v0_9_2 (v+1) | 4006 | 211.0 | 87.4% | 30.8% | 13.3% | 100% | none | multipass; 10 inv chunks; pass2_dropped=1, pass3_added=9 |
| b/vllm/v0_19_1 (v+major) | 875 | 41.1 | 0.0% | 0.0% | 0.0% | 0.0% | **silent** | 1 chunk (chunker source_extraction_failed); only 4 spurious invariants emitted |

Notable: invariant chunk count varies by version (9, 10, 10, 1) - the chunker emits more chunks when more validator methods exist in the source. v-2 has 9 (one missing vs v-1) because vllm 0.6.0 lacks `_verify_bnb_config` on ModelConfig. v+major has 1 (extraction-failed sentinel) because `config.py` no longer exists as a single file.

Schema_recall pattern is non-monotonic: v-2=75.6%, v-1=57.8%, active=97.0%, v+1=87.4%, v+major=0.0%. v-1 dips because vllm 0.6.6.post1 has some EngineArgs fields not present at active (or vice versa); the chunker correctly extracts source but identity tuples diverge. v+1 = 87.4% is robust.

### (d-ab) cells - all complete

4 d-ab cells launched 2026-05-25 ~10:47 UTC, completed ~10:54 UTC after ~7 min wall (4-way parallel; each cell is single LLM call so no multipass).

| cell | wall_s | energy_wh | extension | spurious | obs |
|---|---|---|---|---|---|
| d-ab/vllm/v0_6_0 (v-2) | 344 | 20.2 | 0 | 0 | clean |
| d-ab/vllm/v0_6_6_post1 (v-1) | 417 | 24.5 | 0 | 0 | clean |
| d-ab/vllm/v0_9_2 (v+1) | 144 | 8.5 | 2 | 1 | clean |
| d-ab/vllm/v0_19_1 (v+major) | 101 | 5.9 | 0 | 0 | clean (compact chunker output -> small prompt -> fast) |

The d-ab pattern is robust against the chunker's v+major collapse: the deterministic seed (active reference) is included in the prompt regardless of source extraction quality, so even when the chunker only produces 1 "source not found" chunk, the d-ab cell still emits the 26-invariant reference set.

---

## Infrastructure additions (Phase 2.6)

### 1. Namespace canonicalisation in `trial_scoring.py`

- Added `canonicalise_namespace(ns, engine=None)` helper.
- `invariant_identity()` applies canonicalisation at the namespace extraction step.
- Pass-through for transformers + vllm; collapses `tensorrt_llm.X` -> `tensorrt.X` for tensorrt.
- 2 new tests in `test_trial_scoring.py` (`test_canonicalise_namespace_collapses_tensorrt_llm_prefix`, `test_invariant_identity_canonicalises_tensorrt_llm_namespace`). All 18 tests pass.
- Inline copy in `research/mining-substrate-trial/scripts/strategies/llm_b_oss._invariant_identity` also handles canonicalisation (it drives multipass dedup; same convention drift would have allowed duplicate emissions otherwise).

### 2. Chunker parametrisation (vllm + tensorrt)

`research/mining-substrate-trial/scripts/strategies/vllm_chunker.py`:
- `_read_source(rel_path, source_root=None)` - falls back to `VLLM_SOURCE_ROOT` when source_root is None.
- `get_active_sources(source_root=None)` - threads to `_read_source`.
- `schema_chunks(source_root=None)` + `invariants_chunks(source_root=None)` - threads to `get_active_sources`.
- All call sites updated.

`research/mining-substrate-trial/scripts/strategies/tensorrt_chunker.py`: same shape.

### 3. Strategy dispatchers thread `source_root` for vllm + tensorrt

- `llm_b_oss.run_b_on_{vllm,tensorrt}_active(source_root=None)`.
- `hybrid_extractor.run_d_ab_on_{vllm,tensorrt}_active(source_root=None)`.
- `_run_b_generic` and `_run_d_ab_generic` carry the parameter.

### 4. Trial-runner bumped-cell registry + dispatch

`research/mining-substrate-trial/scripts/trial_runner.py`:
- Registered 8 bumped cells (4 vllm + 4 tensorrt) in `CELL_REGISTRY` (40 cells total now).
- `run_strategy_b` lifts `NotImplementedError` for bumped vllm + tensorrt; lazy-builds source-only venv via `venv_setup.ensure_source_only_venv`.
- `run_strategy_d` same.
- Added `_run_strategy_a_engine_bumped()` to subprocess-invoke the engine's active static miner against bumped source. Surfaces import errors as `detectable` failure modes (deliberately - the miner brittleness IS the trial signal).

---

## Brittleness observations on vllm

### (a) brittleness: dependency-import collapse

All 4 vllm (a) bumped cells fail identically: `ModuleNotFoundError: No module named 'msgspec'`. The source-only venv pattern (no transitive deps) doesn't satisfy vllm's import-time dependency on msgspec. The (a) miner's first action is `import vllm` which transitively imports `sampling_params -> msgspec`.

This is honest - the (a) strategy IS fragile in exactly this way. The brittleness signal is `detectable` (clean failure mode, observable from stderr, repairable by re-installing transitive deps). It's NOT silent.

**This pattern would not occur if the trial bake-off used canonical-container per-version setups** (CUDA-bearing image with full deps installed). The current source-only venv design surfaces this brittleness for free.

### (b) brittleness: chunker file-layout assumption

3 of 4 vllm (b) bumped cells survive (recall ~31-38%). The v+major cell collapses because `vllm/config.py` was refactored into `vllm/config/` (subdirectory) at version 0.19. The chunker's hardcoded `_read_source("config.py")` returns empty.

This is the substantively interesting brittleness signal: **the chunker assumes a STATIC FILE LAYOUT** at active version. When the engine refactors that layout (single file -> subdirectory), the chunker silently fails on those reads, producing degenerate chunks. The LLM has nothing to work with.

A more robust chunker design (Phase 3b candidate?) would discover modules by glob + AST landmark detection rather than hardcoded file paths.

### (b) recall ceiling on vllm: ~30-38%, lower than transformers

vllm (b) cells consistently score 30-38% invariant recall (excluding v+major collapse). Compare transformers (b) cells: 43-59% recall. Possible interpretations:
- vllm's reference has 26 invariants (compared to transformers' 39) - tighter set, narrower hit window.
- vllm uses `msgspec.Struct` + `_verify_args` pattern which the LLM may handle less precisely than transformers' `validate()` pattern.
- 4-way Ollama queue contention may have degraded output quality (longer wait between chunks) - though severity_acc=100% is suspicious of cleaner outputs.

Phase 4 may want to revisit if vllm's recall scales with chunker tuning.

### (d-ab) extension count: 0 / 0 / 2 / 0 across bumps

- v-2, v-1, v+major: extension=0. The LLM, shown the active reference + bumped source, did not propose new invariants. Possibly the active reference already covers what's in those bumped sources (close enough); or the LLM is conservative when the source diverges and won't propose extensions it can't verify.
- v+1 (0.9.2): extension=2, flagged_spurious=1. The LLM proposed 2 invariants in v0.9.2's source not in active, AND flagged 1 active reference invariant as no longer valid at v0.9.2. Mild novel-pattern detection.
- v+major: extension=0 even with a substantively different source. The chunker collapse means the LLM saw "extraction failed" as the source, not the bumped code itself. d-ab merged the active reference unchanged.

---

## Reference-path semantics for bumped cells (unchanged from transformers)

- Reference for vllm bumped cells = ACTIVE vllm reference (`engine_versions/vllm/v0_7_3/outputs/`).
- Bumped cells are scored against active reference. Brittleness IS the delta between bumped-source extraction and active-source reference.
- `reference_paths_for()` falls back to active when per-version reference doesn't exist (unchanged from Phase 3a.2 transformers).

---

## Aggregate matrix (post Phase 3a.2 vllm complete)

`research/mining-substrate-trial/findings/trial_matrix.md` now reports 35 cells with these strategy aggregates:

| strategy | cells | S recall mean | I recall mean | wall mean | energy mean |
|---|---|---|---|---|---|
| a | 11 | 45.5% | 35.7% | 2.6s | 0.03 Wh |
| b | 11 | 72.4% | 40.1% | 3938.0s | 211.63 Wh |
| d-ab | 11 | 100.0% | 100.0% | 326.9s | 16.49 Wh |

Per-engine:

| engine | cells | S recall mean | I recall mean |
|---|---|---|---|
| tensorrt | 3 (just active; bumped not yet) | 85.4% | 75.3% |
| transformers | 17 (5 strategies x 5 versions, partial) | 76.9% | 59.1% |
| vllm | 15 (active + bumped) | 61.2% | 49.2% |

Per-bump-distance:

| bump | cells | S recall mean | I recall mean |
|---|---|---|---|
| v-2 | 6 | 60.7% | 48.9% |
| v-1 | 6 | 74.1% | 57.1% |
| active | 11 | 83.8% | 68.8% |
| v+1 | 6 | 78.4% | 55.6% |
| v+major | 6 | 46.9% | 40.6% |

---

## Handoff items for next agent

### Phase 3a.2.tensorrt scope

NOT STARTED. Infrastructure ready:
- `tensorrt_chunker.{schema_chunks, invariants_chunks}` accept `source_root: Path | None`.
- `llm_b_oss.run_b_on_tensorrt_active(source_root=None)`.
- `hybrid_extractor.run_d_ab_on_tensorrt_active(source_root=None)`.
- `trial_runner` registers 4 tensorrt bumped cells (`v0_19_0`, `v0_20_0`, `v1_0_0`, `v1_2_1`).
- `venv_setup.ENGINE_PIP_SPEC` covers all 4 bumped tensorrt versions.

To run:
```bash
# 1. Build all 4 source-only venvs (each ~2-4 GB wheel download)
for v in v0_19_0 v0_20_0 v1_0_0 v1_2_1; do
  uv run python -m _spike.scripts.venv_setup --engine tensorrt --version-slug "$v"
done

# 2. (a) cells (cheap; ~10s each)
for v in v0_19_0 v0_20_0 v1_0_0 v1_2_1; do
  uv run python -m _spike.scripts.trial_runner --strategy a --engine tensorrt --version-slug "$v"
done

# 3. (b) cells - 4-way parallel via nohup
for v in v0_19_0 v0_20_0 v1_0_0 v1_2_1; do
  nohup uv run python -m _spike.scripts.trial_runner --strategy b --engine tensorrt --version-slug "$v" \
    > /tmp/cell_logs/b_tensorrt_${v}.log 2>&1 &
done

# 4. (d-ab) cells - 4-way parallel via nohup (after (a) done)
for v in v0_19_0 v0_20_0 v1_0_0 v1_2_1; do
  nohup uv run python -m _spike.scripts.trial_runner --strategy d-ab --engine tensorrt --version-slug "$v" \
    > /tmp/cell_logs/d-ab_tensorrt_${v}.log 2>&1 &
done
```

### Expected brittleness signals on tensorrt

- (a) bumped: likely `detectable` import errors. tensorrt_llm wheels are 2-4 GB and have heavy CUDA dependencies. Source-only venv likely won't import.
- (b) bumped: the chunker reads `llmapi/llm_args.py` + `llmapi/build_cache.py`. If tensorrt restructured these (likely between v0.19 and v1.0), the chunker will silent-fail like vllm/v0_19_1 did. Phase 2.6 namespace canonicalisation means the cells won't all score 0.0% even if recall is low.
- (d-ab) bumped: 100% recall by construction; extension counts will vary.

### Total trial scope after tensorrt bumped completes

35 + 12 = 47 cells. Plus the Phase 3b hybrid catalogue cells (not started). Phase 4 synthesis is the next stage.

---

## Commits on `trial/mining-substrate-bakeoff`

Pre-Phase-2.6:
- `3aa7257e` - Phase 3a.1 closure
- `05a3b0a8` - Phase 3a.2 transformers (a) + (d-ab)
- `55f88d61` - Phase 3a.2 progress handoff (transformers)
- `cf172fb2` - Phase 3a.2 transformers (b) bumped
- `a8b957d3` - Phase 3a.1 namespace finding decisions log

This commit: Phase 2.6 + Phase 3a.2 vllm closure.
