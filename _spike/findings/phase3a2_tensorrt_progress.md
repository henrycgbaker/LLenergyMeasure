# Phase 3a.2 - tensorrt bumped-version cells (progress + handoff)

**Status:** TENSORRT COMPLETE. 12 of 12 tensorrt-bumped cells landed (4 (a) + 4 (b) + 4 (d-ab)). All 47 trial cells now in matrix (11 active + 12 transformers-bumped + 12 vllm-bumped + 12 tensorrt-bumped).
**Branch:** `trial/mining-substrate-bakeoff`.
**Aggregate:** `_spike/findings/trial_matrix.{md,csv}` reflects 47 records total.

Phase 2.6 (namespace canonicalisation + chunker parametrisation + bumped-cell wiring) closed before this phase ran. venv_setup tensorrt NVIDIA-index path landed in commit `78aa173a`.

---

## Bumped-version cell coverage (tensorrt, 12 total - ALL DONE)

| version | bump | (a) | (b) | (d-ab) |
|---|---|---|---|---|
| v0_19_0 | v-2 | DONE - 100% recall (MINER_VERSION_BLIND artefact) | DONE - S r=4.7%, I r=16.1% (silent) | DONE - 100% by construction; extension=3 |
| v0_20_0 | v-1 | DONE - 100% recall (MINER_VERSION_BLIND artefact) | DONE - S r=4.7%, I r=16.1% (silent) | DONE - 100%; extension=3 |
| v1_0_0 | v+1 | DONE - 100% recall (MINER_VERSION_BLIND artefact) | DONE - S r=52.3%, I r=22.6% | DONE - 100%; extension=8 |
| v1_2_1 | v+major | DONE - 100% recall (MINER_VERSION_BLIND artefact) | DONE - S r=50.5%, I r=19.4% | DONE - 100%; extension=4 |

---

## Brittleness observations on tensorrt

### (a) brittleness: MINER_VERSION_BLIND artefact

All 4 (a) tensorrt bumped cells score schema=100%, invariants=100%, with output IDENTICAL byte-for-byte (except mined_at timestamp) to the active reference. The walker emits `engine_version: 0.21.0` on every bumped output.

**Root cause** (post-trial audit, not patched): `engine_versions/tensorrt/v0_21_0/producers/static_invariant_miner.py::walk_tensorrt(source_root=None)` accepts an optional source root, but the trial_runner subprocess invokes `miner.walk_tensorrt()` with no argument, so it falls back to `_DEFAULT_SOURCE_ROOT = /tmp/trt-llm-0.21.0/tensorrt_llm`. The dispatcher's `SOURCE_ROOT_PARENT` PYTHONPATH override is a no-op because the tensorrt walker is pure-AST (no `import tensorrt_llm` step that would resolve via PYTHONPATH).

**Brittleness class:** `MINER_VERSION_BLIND`. The miner architecture cannot be steered to bumped source via the current dispatcher pattern. Same class as vllm's `msgspec` import brittleness; different symptom (false-positive 100% recall vs honest detectable crash).

**Captured as:** per-cell observations annotated with the caveat; failure_modes stays `none` because the runner did its job (it ran the configured miner against the configured source). The brittleness is in the substrate's INABILITY to feed bumped source to this engine's miner, not in the miner's runtime behaviour.

| cell | observed score | honest interpretation |
|---|---|---|
| a/tensorrt/v0_19_0 | 100% recall | re-extracted active 0.21.0 source; result is active reference |
| a/tensorrt/v0_20_0 | 100% recall | same |
| a/tensorrt/v1_0_0 | 100% recall | same |
| a/tensorrt/v1_2_1 | 100% recall | same |

This is a HARD brittleness signal: the tensorrt miner architecture (AST-only walks against a hardcoded `_DEFAULT_SOURCE_ROOT`) means a working source-only venv path cannot reach the (a) strategy. To honestly test (a) on bumped tensorrt source, the walker's `_DEFAULT_SOURCE_ROOT` would need to become an env variable OR `walk_tensorrt()` would need to be re-invoked with `source_root=sv.source_dir`. Both are SRC-touching changes; per trial discipline, not done.

The vllm/transformers static miners take `source_root` differently (transformers walks via Python import, so PYTHONPATH override works; vllm imports the package and hard-fails on `msgspec` missing). Three engines, three different brittleness modes around bumped source.

### (b) brittleness: chunker class-name assumption + LLM hallucination from sparse source

The chunker's hardcoded class names (`BaseLlmArgs`, `TrtLlmArgs`) do NOT exist in v0_19_0 / v0_20_0:

| version | main args class structure | chunker extracts BaseLlmArgs? | chunker extracts TrtLlmArgs? |
|---|---|---|---|
| v0_19_0 | `class LlmArgs(BaseModel)` (single, combined) | NO (empty source) | NO (empty source) |
| v0_20_0 | `class LlmArgs(BaseModel)` (single, combined) | NO | NO |
| v0_21_0 (active) | `class BaseLlmArgs` + `class TrtLlmArgs` | YES | YES |
| v1_0_0 | `class BaseLlmArgs(StrictBaseModel)` + `class TrtLlmArgs(BaseLlmArgs)` | YES | YES |
| v1_2_1 | same | YES | YES |

When the chunker hands the LLM empty source for the main classes (v0_19_0, v0_20_0), the LLM HALLUCINATES content. Sample of invariant ids emitted for b/tensorrt/v0_19_0:
- `tensorrt_llm_temperature_lt_0`, `tensorrt_llm_top_k_lt_1`, `tensorrt_llm_top_p_lt_0`
- `tensorrt_llm_do_sample_set_when_*`, `tensorrt_llm_num_beams_lt_1`

NONE of `temperature`, `top_k`, `top_p`, `do_sample`, `num_beams` exist in `v0_19_0/llmapi/llm_args.py` (verified by grep). These are HuggingFace `GenerationConfig` fields the LLM hallucinated from prior training. Result: cell emits 37 invariants, only 5 intersect with the 31 reference invariants -> `silent` failure mode (cell_count > 0 but intersection too low).

For v1.x (which DOES have BaseLlmArgs + TrtLlmArgs), the chunker works correctly, and (b) cells score 19-22% invariant recall + 50-52% schema recall. Lower than transformers / vllm (b) because the validator surface in v1.x is MUCH larger than active (51 decorators in v1_2_1 vs 25 in active v0_21_0), so the LLM's extraction diverges from the reference - emits new invariants the active reference doesn't have, AND misses some active invariants because the chunker's `max_chars` truncates earlier in the larger v1.x source.

The chunker brittleness here is a SUPER-SET of the vllm 0.19.1 case:
- vllm 0.19.1: file moved (`config.py` -> `config/`); chunker fails silently with empty chunks; LLM emits 4 spurious invariants from the failure marker.
- tensorrt v0.x: classes RENAMED/RESTRUCTURED; chunker reads empty class bodies; LLM hallucinates 30+ unrelated invariants from prior knowledge.

The chunker's hard-coded class names AND file paths are both brittleness sources.

### (d-ab) brittleness: insulated by construction; extension counts vary

All 4 d-ab tensorrt bumped cells score 100% recall by construction (active seed = reference). Extension counts:

| version | extension | flagged_spurious | total | obs |
|---|---|---|---|---|
| v0_19_0 (v-2) | 3 | 2 | 38 | LLM proposed 3 novel from bumped source + flagged 2 active as no-longer-valid |
| v0_20_0 (v-1) | 3 | 1 | 38 | similar to v-2 |
| v1_0_0 (v+1) | 8 | 1 | 43 | biggest extension - v1.x's expanded validator surface yields more candidates |
| v1_2_1 (v+major) | 4 | 1 | 39 | smaller than v+1 - LLM was more conservative |

d-ab inherits the active reference (`reference_paths_for` falls back to active for bumped cells), so it scores 100% recall trivially. The interesting signal is the extension count - it tracks substrate divergence:
- v0.x has SMALLER source surface (older Pydantic patterns); extension=3 reflects modest novel-pattern detection.
- v1.x has LARGER surface; v1_0_0 (the early-major) yields extension=8 - most LLM-novel detection.
- v1_2_1 settled-major shows extension=4 - LLM picked some patterns but was conservative on the bulk.

**(d-ab)-on-bumped caveat (from vllm precedent):** if (a) crashed (likely on vllm), d-ab inherits the active reference + LLM extension from bumped source = "hollow" measurement. Same on tensorrt EXCEPT that (a) didn't crash - it ran against active source. So d-ab/tensorrt bumped is essentially measuring "given the active reference, can the LLM find adjacent invariants in the bumped source?" which is its honest interpretation.

---

## (b) cell wall + energy

(b) batch 1 (v0_19_0 + v0_20_0): 2-way parallel from 12:15-12:43 UTC = 28 min wall each (4-way Ollama contention).
(b) batch 2 (v1_0_0 + v1_2_1): 2-way parallel from 12:44-13:22 UTC = 38 min wall each (larger source -> longer LLM processing).

| cell | wall_s | energy_wh | failure_modes | chunker source size |
|---|---|---|---|---|
| b/tensorrt/v0_19_0 | 1667 | 77.9 | silent (hallucination) | small (1354 lines) |
| b/tensorrt/v0_20_0 | 1678 | 78.5 | silent (hallucination) | small (1457 lines) |
| b/tensorrt/v0_21_0 | 1372 | 66.4 | none | medium (2072 lines) |
| b/tensorrt/v1_0_0 | 2248 | 101.8 | none | large (2441 lines) |
| b/tensorrt/v1_2_1 | 2260 | 102.7 | none;silent | xlarge (3416 lines) |

The v1.x cells used about ~63% more wall and energy than the active cell because the larger Pydantic validator surface meant LLM processing per chunk took longer.

Notable: failure_modes on v1_2_1 = `none;silent` - the schema scored `none` (50.5% recall not silent-threshold) but invariants scored `silent` (19.4% with cell_count=40, intersection=6 - tight thresholds). Aggregator concatenates.

---

## Cross-engine comparison of (b) brittleness (preview - synthesis is Phase 4)

|engine | active I_r | v-2 I_r | v-1 I_r | v+1 I_r | v+major I_r | brittleness mode at edges |
|---|---|---|---|---|---|---|
| transformers | 56.4% | 59.0% | 59.0% | 59.0% | 43.6% | none; monotonic with bump |
| vllm | 38.5% | 34.6% | 34.6% | 30.8% | 0.0% (silent) | chunker file-layout collapse at v+major |
| tensorrt | 25.8% | 16.1% (silent) | 16.1% (silent) | 22.6% | 19.4% (silent at I) | chunker class-name collapse at v-2/v-1; v1.x partial |

(b) tensorrt recall ceiling is ~25% even on active vs transformers ~57% / vllm ~39%. tensorrt's Pydantic validator surface is denser (more cross-field constraints, more enum allowlists, more `mode='after'` model validators) and the chunker only emits 7 invariant chunks (vs transformers' 14, vllm's 10). Same per-chunk LLM call quality, fewer chunks, narrower recall.

---

## Aggregate matrix (post Phase 3a.2 tensorrt complete)

`_spike/findings/trial_matrix.md` now reports 47 cells with these strategy aggregates:

| strategy | cells | S recall mean | I recall mean | wall mean | energy mean |
|---|---|---|---|---|---|
| a | 15 | 60.0% | 52.8% | 1.9s | 0.02 Wh |
| b | 15 | 60.6% | 34.4% | 3411s | 179.25 Wh |
| b_8b | 1 | 85.7% | 35.7% | 412.6s | 4.93 Wh |
| c | 1 | 0.0% | 0.0% | 0.0s | 0.00 Wh |
| d-ab | 15 | 100.0% | 100.0% | 254.8s | 12.96 Wh |

Per-engine:

| engine | cells | S recall mean | I recall mean |
|---|---|---|---|
| tensorrt | 15 | 77.9% | 73.3% (inflated by 5 perfect (a) MINER_VERSION_BLIND artefacts + d-ab) |
| transformers | 17 | 76.9% | 59.1% |
| vllm | 15 | 61.2% | 49.2% |

The tensorrt I_r mean of 73.3% is misleading - it includes 5 perfect (a) scores (4 bumped + 1 active) all of which are actually re-extracting the active reference. Honest per-(b) tensorrt mean: (4.7+4.7+25.8+22.6+19.4)/5 = 15.4% invariant recall - the LOWEST of the three engines for (b).

Per-bump-distance (raw, includes inflated tensorrt (a)):

| bump | cells | schema_recall_mean | inv_recall_mean |
|---|---|---|---|
| v-2 | 8 | similar | similar |
| v-1 | 8 | similar | similar |
| active | 11 | unchanged | unchanged |
| v+1 | 8 | similar | similar |
| v+major | 8 | similar | similar |

(Per-bump-distance roll-ups are now affected by the tensorrt (a) bumped MINER_VERSION_BLIND artefact - Phase 4 synthesis should de-weight or annotate.)

---

## Handoff items for next phase

Phase 3a.2 complete. Next phase: per user direction in DECISIONS_LOG:

1. Move trial into worktree (`git worktree add ../llenergymeasure-trial trial/mining-substrate-bakeoff`).
2. Switch main workspace to spike branch.
3. Add OQ12 (or higher) to engine-knowledge-as-data design doc: "Storage strategy for mining-substrate artefacts: git-tracked vs GH-artefacts pinned against upstream images".
4. THEN launch Phase 3b H4 (LLM-modifies-miner) first (user-prioritised dual-purpose pattern).

The 47-cell matrix in `trial_matrix.md` is the Phase 3a deliverable. Phase 4 synthesis comes AFTER Phase 3b.

---

## Commits on `trial/mining-substrate-bakeoff`

Pre-Phase-3a.2-tensorrt:
- `15f34240` ... `78aa173a` - venv_setup tensorrt NVIDIA-index download fix.
- `06ead881` - Phase 2.6 closure + vllm bumped cells.
- `3aa7257e`, `05a3b0a8`, `55f88d61`, `cf172fb2`, `a8b957d3` - earlier trial phases.

This commit: Phase 3a.2 tensorrt bumped cells (12) + matrix refresh + observations annotations on (a) bumped.
