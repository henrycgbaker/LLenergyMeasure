# Phase 3a (active + bumped) complete - cross-engine brittleness summary

**Status:** Phase 3a CLOSED. 47 cells in `_spike/findings/trial_matrix.{md,csv}`.
**Branch:** `trial/mining-substrate-bakeoff`.
**Authored:** 2026-05-25.

This document summarises Phase 3a (active + bumped) findings without synthesising Phase 4 verdicts. Phase 4 synthesis is the next stage and SHOULD draw conclusions; this doc only catalogues what the 47 cells observed.

---

## Coverage

| | active | v-2 | v-1 | v+1 | v+major | total |
|---|---|---|---|---|---|---|
| transformers | a, b, b_8b, c, d-ab, d-ac (6) | a, b, d-ab (3) | a, b, d-ab (3) | a, b, d-ab (3) | a, b, d-ab (3) | 18 |
| vllm | a, b, d-ab (3) | a, b, d-ab (3) | a, b, d-ab (3) | a, b, d-ab (3) | a, b, d-ab (3) | 15 |
| tensorrt | a, b, d-ab (3) | a, b, d-ab (3) | a, b, d-ab (3) | a, b, d-ab (3) | a, b, d-ab (3) | 15 |
| **total** | 12 | 9 | 9 | 9 | 9 | **48 unique cells** |

48 unique (engine, version, strategy) tuples; aggregate matrix shows 47 because one cell (`c/transformers/v4_57_3`) was emitted as a SKIP record with placeholder zeros (key_absent failure_mode) and one stays at the same identity in the count - 47 score JSONs total.

---

## (a) brittleness across engines

The (a) strategy = pure static-mining via engine-specific walker, against bumped source. Each engine surfaces (a) brittleness DIFFERENTLY:

| engine | (a) bumped failure mode | symptom | classified as |
|---|---|---|---|
| transformers | MINER_LANDMARK_MISSING / ImportError | walker imports class to inspect; bumped version lacks it; raises | `detectable` |
| vllm | ModuleNotFoundError: msgspec | walker imports vllm package; package has hard transitive dep on msgspec; source-only venv doesn't install it | `detectable` |
| tensorrt | MINER_VERSION_BLIND (artefact) | walker is pure-AST; reads hardcoded `_DEFAULT_SOURCE_ROOT`; PYTHONPATH override has no effect; emits active reference unchanged | `none` (substrate wiring artefact, not honest measurement) |

The transformers + vllm (a) bumped failures are HONEST `detectable` brittleness - the substrate runs the bumped miner, the miner crashes cleanly on bumped source, the trial captures the failure mode. The tensorrt (a) bumped result is NOT honest - the substrate's PYTHONPATH plumbing assumes import-time landmark discovery, which tensorrt's AST-only walker bypasses. Trial discipline ("don't fix the miner") preserves the artefact; observations annotated on each score JSON document the caveat.

| (a) cell | recall reported | reality |
|---|---|---|
| transformers/v4_55_4 (v-2) | 0.0% | landmark missing - PreTrainedConfig signature shift; detectable crash |
| transformers/v4_56_2 (v-1) | 48.7% | walker ran; partial extraction (signature drift but landmarks found) |
| transformers/v4_57_3 (active) | 100% | reference cell |
| transformers/v4_57_6 (v+1) | 43.6% | walker ran; partial |
| transformers/v5_9_0 (v+major) | 0.0% | landmark missing; detectable crash |
| vllm all 4 bumped | 0.0% | msgspec import error; detectable crash |
| tensorrt all 4 bumped | 100.0% | MINER_VERSION_BLIND artefact (not honest) |

**Across-engine (a) brittleness observation:**
- transformers (a) shows the cleanest brittleness signal: v-1 and v+1 still work (43-49% recall), v-2 and v+major both crash with `detectable`.
- vllm (a) crashes uniformly across all 4 bumps with `detectable`.
- tensorrt (a) is invisible to bumped-source pressure under the current substrate; would need walker refactor to surface.

Three engines, three distinct (a) brittleness profiles. NONE of these are silent failures (failure_modes are honest); the brittleness is at the substrate-engine seam.

---

## (b) brittleness across engines

The (b) strategy = pure LLM extraction from chunked source. Each engine surfaces (b) brittleness DIFFERENTLY:

| engine | (b) recall trajectory | brittleness at edges |
|---|---|---|
| transformers | active 56%, v-2 59%, v-1 59%, v+1 59%, v+major 44% | none; monotonic with bump distance |
| vllm | active 39%, v-2 35%, v-1 35%, v+1 31%, v+major 0% (silent) | chunker file-layout collapse: `config.py` -> `config/` subdir at vllm 0.19 |
| tensorrt | active 26%, v-2 16% (silent), v-1 16% (silent), v+1 23%, v+major 19% | chunker class-name collapse: `LlmArgs` -> `BaseLlmArgs+TrtLlmArgs` between v0.20 and v0.21 |

(b) recall ceiling per engine:
- transformers ~57%. Stable across bumps.
- vllm ~38% (active). Lower than transformers; chunker has fewer effective chunks.
- tensorrt ~26% (active). Lowest; chunker emits 7 invariant chunks vs transformers 14; denser Pydantic surface.

**Cross-engine (b) recall variation per request:**
- **transformers ~50%** (mean of all 5 versions = 55.4%)
- **vllm ~26%** (mean = 27.7%, including the v+major silent-fail at 0%)
- **tensorrt ~16%** (mean = 19.0%, including 2 silent-fail bumped cells)

The brittleness manifests differently:
- vllm: file moved -> chunker reads empty -> LLM emits 4 spurious invariants from "extraction_failed" marker -> silent fail with low cell_count.
- tensorrt: CLASS RENAMED -> chunker reads empty for that class -> LLM HALLUCINATES 30+ invariants from generic GenerationConfig prior knowledge -> silent fail with high cell_count + low intersection (16-19% recall feels like "kind of working" but is mostly hallucinated content).

The hallucination mode on tensorrt v0_19_0 / v0_20_0 is the more INSIDIOUS brittleness signal: the metrics look "okay" (16% recall, 100% sev_acc) but the underlying content is mostly invented. Phase 4 synthesis should flag this as a distinct failure mode that the rubric currently classifies as `silent` but could be subdivided into `silent-empty-extraction` (vllm v+major) vs `silent-hallucinated-from-empty-chunks` (tensorrt v-2/v-1).

---

## (d-ab) brittleness across engines

The (d-ab) strategy = active static mining seed + LLM extends on bumped source. All 15 d-ab cells score 100% recall by construction (the seed IS the reference). The interesting signal is extension counts:

| engine | active ext | v-2 ext | v-1 ext | v+1 ext | v+major ext |
|---|---|---|---|---|---|
| transformers | (n/a in d-ab) | 0 | 0 | 0 | 0 |
| vllm | (n/a in d-ab) | 0 | 0 | 2 | 0 |
| tensorrt | (n/a in d-ab) | 3 | 3 | 8 | 4 |

Pattern:
- transformers d-ab is CONSERVATIVE - extension=0 across all bumps. The LLM looks at bumped transformers source vs active reference and proposes NO new invariants.
- vllm d-ab is similarly conservative except v+1 (0.9.2) where 2 novel invariants emerged.
- tensorrt d-ab is GENEROUS - 3-8 extensions per bump. Reflects how much the tensorrt validator surface evolved across the bump-window. v+1 (1.0.0) yields the biggest extension (8) - early-major restructuring offers the most novel patterns.

**(d-ab)-on-bumped caveat (documented from vllm precedent):** if (a) crashed (likely on vllm + transformers v-2/v+major), d-ab inherits active reference + LLM extension from bumped source = "hollow" measurement of bumped specifics. The 100% recall is by construction (seed=reference), and the LLM extension comes from bumped source via chunker. For vllm this means: extensions came from bumped chunker output; when chunker collapsed (v+major), extension=0. For tensorrt: extensions came from bumped chunker, which DID work on v1.x (BaseLlmArgs class exists) and PARTIALLY worked on v0.x (hallucinated chunks).

This is why d-ab tensorrt extension counts are higher than vllm: when the chunker provides "wrong" source (hallucinated chunks on v0.x), the LLM gets MORE novel material than when the chunker provides empty source (vllm v+major).

So d-ab extensions PARTIALLY measure chunker brittleness, not honest LLM-on-bumped behaviour. Phase 4 synthesis should disentangle.

---

## Headline cross-engine observations

1. **(a) brittleness has three distinct modes** across our 3 engines (landmark missing / dep import / version-blind walker). All three are valuable trial data; none is "the (a) brittleness".

2. **(b) recall variation: ~50% > ~26% > ~16%** (transformers > vllm > tensorrt). Tighter validator surface = lower (b) recall. Schema_recall follows similar order.

3. **(b) brittleness at v+major manifests as silent failure** on vllm + tensorrt; on transformers, it just degrades smoothly (59% -> 44%). vllm: chunker file-layout assumption. tensorrt: chunker class-name assumption + LLM hallucination. transformers: no brittleness at v+major.

4. **(d-ab) recall is 100% by construction; extension counts vary** by engine architecture stability. tensorrt's extension counts (3-8) are highest because the validator surface grew most.

5. **Wall + energy:** (a) ~2s + 0Wh < (d-ab) ~250s + 13Wh < (b) ~3400s + 179Wh. (b) is ~1700x more expensive in energy than (a), ~14x more expensive than (d-ab). Mineable for cost-quality Pareto in Phase 4.

6. **Reference inflation in tensorrt:** the I_r mean of 73.3% across all tensorrt cells is misleading - the 4 (a) bumped cells contribute 5 perfect-100% measurements that aren't honest. Removing those, the (b)-tensorrt mean is ~15% I_r, lowest of three engines. Phase 4 must de-weight these.

7. **Sev_acc is 100% for (b) tensorrt cells but 75-78% for transformers (b) and 100% for vllm (b)**. The LLM more reliably emits the right severity tag when fewer invariant types exist; tensorrt's narrower surface helps here.

8. **(b) is the only strategy with `silent` failures**. (a) failures are `detectable` (clean stderr trace). (d-ab) failures are not possible (100% by construction). Only (b) produces `silent` failures - and they're concentrated at engine-version edges (vllm v+major, tensorrt v0.x/v1.x).

9. **The 8B llama model (`b_8b`) is ~16x cheaper than 70B (b)** on transformers active (412s + 4.9Wh vs 1649s + 81Wh) but drops invariant recall from 56% to 36% (~20 pp). One-data-point indicator for Phase 4 cost-quality slider.

---

## Cells that need Phase 4 special handling

| cell | issue | mitigation |
|---|---|---|
| a/tensorrt/{v0_19_0, v0_20_0, v1_0_0, v1_2_1} | MINER_VERSION_BLIND artefact; 100% recall not honest | de-weight from aggregates or relabel honestly |
| b/tensorrt/{v0_19_0, v0_20_0} | hallucinated content silently classified as `silent` | distinguish from chunker-empty silent-fails |
| a/transformers/{v4_55_4, v5_9_0} + a/vllm/all-bumped | detectable crashes | these ARE honest; treat as baseline brittleness |
| b/vllm/v0_19_1 | chunker file-layout collapse | honest chunker brittleness |

---

## Aggregate snapshot

```
strategy   cells   schema_r_mean   inv_r_mean   wall_mean_s   energy_mean_wh
a          15      60.0%           52.8%        1.9           0.02
b          15      60.6%           34.4%        3411.4        179.25
b_8b       1       85.7%           35.7%        412.6         4.93
c          1       0.0%            0.0%         0.0           0.00
d-ab       15      100.0%          100.0%       254.8         12.96
```

```
engine        cells   schema_r_mean   inv_r_mean   wall_mean_s
tensorrt      15      77.9%           73.3% [artefact-inflated]   643.9
transformers  17      76.9%           59.1%        1758.3
vllm          15      61.2%           49.2%        1058.9
```

---

## Phase 3b readiness signal

Phase 3a infrastructure is exercised across 3 engines x 5 versions x 3 strategies = 45 cells (plus 2 active-only b_8b + c experiments = 47). Substrate components proven:

- `trial_runner._run_strategy_a_engine_bumped()` - works for transformers + vllm; fails silently for tensorrt (MINER_VERSION_BLIND). Phase 3b H4 (LLM-modifies-miner) is the natural fix path - subagent could propose `walk_tensorrt(source_root=path)` invocation patch.
- `tensorrt_chunker / vllm_chunker / transformers_chunker` with `source_root` param - works on transformers + tensorrt v1.x + vllm v-2..v+1; collapses (empty source) on vllm v+major + tensorrt v0.x. Phase 3b candidate: glob+AST chunker (Tier 2 / 3).
- `trial_scoring.canonicalise_namespace` - works correctly for all 3 engines; b/tensorrt active rescore from 0.0% -> 25.8% is the smoking gun.
- `hybrid_extractor.run_d_ab_on_<engine>_active(source_root=path)` - works on all 3 engines; produces meaningful extension counts.

Phase 3b H4 priority (user-prioritised): patches the (a) miner gaps - especially the tensorrt MINER_VERSION_BLIND - via LLM-proposed patches against the active walker. Cross-pollinates with spike branch's mining-substrate refactor.

The 47-cell trial dataset is sufficient to launch Phase 3b. Phase 4 (synthesis) is queued AFTER Phase 3b cells land. Trial matrix discipline preserved throughout.
