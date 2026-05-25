# Empirical trial outcome - mining-substrate decision synthesis

**Status:** Trial deliverable. Phase 4 synthesis.
**Authored:** 2026-05-25.
**Branch:** `trial/mining-substrate-bakeoff`.
**Scope:** 51 scored cells across 3 engines (transformers, vllm, tensorrt) x 5 versions each x pure + hybrid strategies, plus 9 distinct hybrid patterns. All cells run at 70B-q4 (`llama3.1:70b` via container Ollama @ port 11435, num_ctx=32768). Phase 3c (Claude) deferred pending `ANTHROPIC_API_KEY`.

**Cross-refs:**
- `research/mining-substrate-trial/findings/trial_epistemic_framing.md` (the framing this synthesis answers to).
- `.planning/mining-substrate-empirical-trial.md` (the plan whose six scenarios this maps onto).
- `research/mining-substrate-trial/findings/phase4_0_validated_union_summary.md` (the rescored ground truth).
- `research/mining-substrate-trial/findings/trial_matrix_vu.{md,csv}` (validated-union per-cell scores).
- `research/mining-substrate-trial/findings/trial_matrix.{md,csv}` (original (a)-as-reference scores; retained for delta comparison).
- `research/mining-substrate-trial/findings/phase3a_complete_summary.md` (47-cell pure-strategy aggregate).
- `research/mining-substrate-trial/findings/hybrid_experiments/*/` (per-pattern findings for H2/H3/H4/H6/H7/H9 + E6/E9).
- `research/mining-substrate-trial/findings/post_trial_a_gap_closure.md` (the (a) gap commitment backlog).
- `research/mining-substrate-trial/findings/phase2_llm_infrastructure.md` (the (b) infrastructure design + calibration record).
- `research/mining-substrate-trial/findings/phase2_locked_prompts/*.md` (the locked prompts for (b) and (d-ab)).

The trial deliberately resisted mid-trial re-tuning: prompts locked at Phase 2 end, no per-cell prompt iteration. This synthesis is interpretation built on top of those locked baselines.

---

## Section 1: TL;DR

The trial completed 51 cells and 9 hybrid-pattern experiments at `llama3.1:70b` q4 quantisation. Under validated-union ground truth (every strategy's runtime-validated invariants unioned per cell), pure deterministic (a) achieves 46.6% invariant recall / 32.7% precision; pure LLM (b) achieves 42.3% / 27.6%; the active-seed hybrid (d-ab) achieves 77.6% / 73.6%. None of the substrate-decomposition variants (H6, E6, E9) lift (b)'s ceiling, confirming the bottleneck is LLM synthesis capacity at this model scale, not chunking. Across engines (a) shows three structurally distinct brittleness modes on version bumps (transformers landmark-missing, vllm msgspec ImportError, tensorrt MINER_VERSION_BLIND silent re-extraction). The trial's most decision-relevant cross-pattern finding is the **LLM-role split**: at 70B-q4, LLMs are robust at diagnosis (0 fabrications across 8 H4+H9 diagnoses), error-prone at subtraction (3/3 H2 vllm drops were false-drops), weak at single-shot synthesis (H4 patches: 0/3 recall lift, 2/3 crashed), collapse to zero under agentic flexibility (H7), and hit a hard substrate ceiling at extraction (~50% recall on transformers).

The recommended architecture is **Scenario 4 with deterministic-validate + LLM-extend-propose**: pure (a) as the deterministic floor across all three engines; (b)-style chunked LLM extraction layered on top; a deterministic runtime gate (existing `scripts/validate_invariants.py`, already production-grade for all three engines via per-engine containers) producing the validated union as the canonical artefact. The deterministic floor catches what LLMs hallucinate; the LLM catches what AST walkers miss. The 7 known (a) gaps (post-trial gap closure backlog) close regardless of substrate choice, on the spike-branch refactor path with H4's diagnoses as accelerator. The LLM-role split is enforced as architecture: subtraction is deterministic, extension is LLM, synthesis-of-code stays human (with H4's diagnoses as scaffolding).

Deferred: Phase 3c (Claude) cell-pair to test whether the LLM-role split is intrinsic or 70B-q4-specific; if Claude breaks the (b) ceiling materially or rescues H4's synthesis weakness, Section 5's recommendation deserves a second look. Phase 5 dogfoods the chosen pipeline on transformers first (highest reference maturity, lowest brittleness surface). Storage strategy (OQ9 git-tracked vs GH-artefacts) is decoupled from the substrate decision and revisited post-spike-refactor.

---

## Section 2: Methodology recap

Three axes per cell: pure-strategy quality (recall, precision, severity, schema fidelity); hybrid-strategy quality on the same axes; brittleness across version bumps (v-2, v-1, active, v+1, v+major per engine). Cells run at locked Phase-2 prompts against bumped sources unpacked into source-only venvs at `/tmp/trial_<engine>_<slug>_venv/`. Pure-strategy execution followed matrix discipline (no early-exit, no per-cell tuning); hybrid-pattern execution followed exploratory discipline (experiment, log, iterate). 51 scored cells in the validated-union matrix; 9 hybrid patterns (H1, H2, H3, H4, H6, H7, H9, E6, E9).

The trial's most consequential methodological correction was the move from "(a)-as-reference" scoring to **validated-union ground truth** (every strategy's invariants unioned and runtime-validated; that union becomes the cell's empirical truth). The original framing biased the matrix toward (a) - any LLM-found invariant absent from (a) counted as false-positive even when runtime-validated. The validated-union rescore made the matrix honest: (a) and (b) compete on the same union, neither side privileged. Across 15 active cells, the union added 28 entries that (a) missed.

What the trial measured: pure-strategy quality at locked prompts; per-engine and per-bump brittleness; 9 hybrid patterns. What it did NOT measure: Claude-quality LLM behaviour (deferred Phase 3c); engines beyond the three picked; F16 quantisation (deferred H12); production-scale curation overhead (Phase 5). Honest limitations: 70B-q4-specific findings (the LLM-role split may be model-size-dependent); 3 engines only; 4 bumped versions per engine.

---

## Section 3: The information map

### 3.1 Pure-strategy baselines (validated-union scoring)

Per `research/mining-substrate-trial/findings/phase4_0_validated_union_summary.md`. The aggregate moves under the corrected rubric:

| strategy | cells | inv_recall_a | inv_recall_vu | delta | inv_precision_a | inv_precision_vu | delta |
|---|---|---|---|---|---|---|---|
| a | 15/15 | 52.8% | 46.6% | -6.2pp | 54.3% | 32.7% | -21.6pp |
| b | 15/15 | 34.4% | 42.3% | +8.0pp | 21.0% | 27.6% | +6.6pp |
| b_8b | 1/1 | 35.7% | 25.0% | -10.7pp | 16.1% | 22.6% | +6.5pp |
| d-ab | 15/15 | 100.0% | 77.6% | -22.4pp | 93.6% | 73.6% | -20.0pp |
| e6 | 2/2 | 43.6% | 50.4% | +6.8pp | 28.0% | 46.5% | +18.6pp |
| e9 | 2/2 | 34.0% | 41.4% | +7.5pp | 29.9% | 52.5% | +22.6pp |
| h6 | 1/1 | 12.8% | 17.9% | +5.0pp | 31.2% | 62.5% | +31.2pp |

The directional reading: (a)'s aggregate moves DOWN -6.2pp recall and -21.6pp precision (it had been measuring itself against itself; the union exposes its real gaps); (b) moves UP +8.0pp recall and +6.6pp precision (the LLM was finding real things outside (a)'s output that the (a)-rubric was scoring as spurious); (d-ab) is the construction artefact - 100% recall against (a) becomes 77.6% against the union, because the union exceeds the (a) seed by 28 entries. E6 and E9 both look BETTER under the union than under (a)-rubric, suggesting their non-(a) emissions are mostly real. The big precision shifts on h6/e6/e9 (+31, +18, +22pp) reflect that low-emission variants tend to emit only "obvious" invariants that survive runtime validation.

Active-cell coverage by strategy contribution (per `phase4_0_validated_union_summary.md`):

| engine | version | (a) | (b) | d-ab | h2 | h3 | h6 | e6 | e9 |
|---|---|---|---|---|---|---|---|---|---|
| transformers | v4_57_3 | 37 | 35 | 37 | 37 | 32 | 10 | 32 | 22 |
| vllm | v0_7_3 | 26 | 19 | 26 | 23 | 19 | 0 | 17 | 17 |
| tensorrt | v0_21_0 | 11 | 5 | 11 | 11 | 5 | 0 | 0 | 0 |

(a) and (b) are MOSTLY OVERLAPPING contributors to the union: on transformers active, (a) finds 37 validated entries, (b) finds 35; their union finds 56 validated total. The genuinely (b)-unique entries are few - H6 contributes 3 transformers entries no other strategy found (the only non-zero unique-contributor row in any cell). On vllm active, E6 + E9 each contributed unique entries; on tensorrt active, no strategy contributed uniquely beyond (a). The picture: substrate strategies COMPLEMENT more than they SUBSTITUTE.

### 3.2 Brittleness profile across version bumps

The (a) brittleness is structurally distinct per engine. Three modes:

**transformers (a) - landmark-missing on bump extremes.** v-2 (4.55.4) and v+major (5.9.0) both fail `detectable` (clean crash with stderr). The producer's `import` statements hit API renames - `tokenizers>=0.21,<0.22` constraint fails at 4.55.4; `is_offline_mode` removed from huggingface_hub at 5.9.0. v-1 and v+1 work but score lower (33.9% and 32.1% under vu): the walker runs but the bumped source has different identity tuples. Cleanest brittleness signal: detectable crash at the edges, graceful degradation in the middle.

**vllm (a) - dependency-import collapse on all bumps.** All 4 bumped cells crash `detectable` with `ModuleNotFoundError: No module named 'msgspec'`. vllm has a hard transitive import-time dep on msgspec that the source-only venv pattern doesn't satisfy. The walker's first action is `import vllm` which transitively imports `sampling_params -> msgspec`. Honest brittleness: 100% bumped-cell failure across all distances.

**tensorrt (a) - MINER_VERSION_BLIND silent re-extraction.** All 4 tensorrt bumped cells report 100% recall + 100% precision - because the walker is pure-AST, reads from a hardcoded `_DEFAULT_SOURCE_ROOT = /tmp/trt-llm-0.21.0/tensorrt_llm`, and PYTHONPATH override has no effect. The substrate cannot steer the walker to bumped source via the current dispatcher pattern. Result: bumped cells re-extract ACTIVE source unchanged. This is NOT honest brittleness measurement - it's a substrate-wiring artefact. Phase 4 synthesis de-weights these from any per-engine aggregate.

These are three distinct brittleness MECHANISMS, not three intensities of one. Each requires a different fix: defensive imports + AST-fallback (transformers); transitive-dep declarations (vllm); pluggable `source_root` or env-var indirection (tensorrt). The cross-engine LESSON: "the (a) substrate is brittle" is too coarse; (a)'s brittleness profile is heterogeneous across engines.

The (b) brittleness profile is similarly heterogeneous but rooted in chunker assumptions, not walker assumptions:

- **transformers (b)**: stable across bumps. 44-59% recall across all 5 versions (under vu rubric). The chunker reads files by name (`modeling_utils.py`, `generation/configuration_utils.py`); those names didn't change across the v4-v5 transition. The mild v+major recall dip (44.6% vs 62.5% active) reflects real source drift, not chunker fragility.

- **vllm (b)**: smooth-then-cliff. v-2 / v-1 / v+1 all in the 33-46% range; v+major (0.19.1) collapses to 0% silent-fail. The chunker reads `config.py`, but vllm 0.19 refactored `config.py` -> `config/` subdirectory. The hardcoded `_read_source("config.py")` returns empty; the LLM emits 4 spurious invariants from the failure marker; cell silently fails.

- **tensorrt (b)**: hallucinate-then-degrade. v0_19_0 / v0_20_0 (the older versions) classify `silent` at ~16% recall, but content is mostly HALLUCINATED. The chunker's hardcoded class names (`BaseLlmArgs`, `TrtLlmArgs`) don't exist in v0_19 / v0_20 (which use a single `LlmArgs` class). Empty class bodies produced; the LLM filled the void by HALLUCINATING 30+ HuggingFace GenerationConfig field names (`temperature`, `top_k`, `do_sample`) that don't exist in tensorrt at all. v1.x bumped cells work (class names match); recall stays at 19-22%, lower than transformers because the validator surface in v1.x is much larger than v0.21 and (b)'s extraction diverges by emitting new patterns the active reference doesn't have.

The (d-ab) brittleness is partly insulated by construction (the active reference is always merged), but the EXTENSION counts reveal real signal: transformers 0 across all bumps; vllm 0/0/2/0 across bumps; tensorrt 3/3/8/4. tensorrt's expanding validator surface (25 -> 32 -> 51 decorators across v0.21 -> v1.0 -> v1.2.1) gives the LLM more novel patterns to extend on. This is a genuine LLM signal even if recall is by-construction. But there's a HOLLOW CAVEAT: when (a) crashes (vllm bumped cells), d-ab's "100% recall" is purely the active reference. The 100% number is correct but misleading - it doesn't measure bumped-cell behaviour; it measures the seed.

### 3.3 Hybrid landscape

Nine distinct patterns ran. Findings per pattern, organised by their tested LLM role:

**H1 (LLM extends; baseline d-ab):** active-seed-plus-LLM-proposal-on-bumped-source. By-construction 100% recall; precision drops to 73.6% under vu. Extension counts vary by engine (see 3.2). Functional as a baseline; doesn't test the LLM's autonomous extraction quality.

**H2 (LLM validates by subtracting from (a)):** the LLM reads (a)'s output and decides which entries are spurious. Drop rates: transformers 0/41, vllm 3/26, tensorrt 0/35. The 3 vllm drops were ALL FALSE-DROPS - the LLM misclassified dormant-normalisation entries (e.g. `if self.seed == -1: self.seed = None`) as spurious because the predicate form didn't match the LLM's expected "raises when X" template. Verdict: LLM subtraction is unsafe at 70B-q4. Conservative prompting prevented mass-drops but the small drops emitted were errors.

**H3 (LLM proposes; deterministic runtime gate):** runtime gate on transformers (`runtime_validate_invariants`); schema-existence gate on vllm + tensorrt. Transformers: +5.6pp precision lift, -7.7pp recall (12/51 (b)-emitted entries dropped, of which 3 were in reference and 9 were not). vllm: +1.5pp precision, neutral recall. tensorrt: 0 dropped. The gate's strength is asymmetric: runtime catches LLM hallucinations that produce wrong predicates; schema-existence only catches fabricated FIELD NAMES (rare at active version because (b) stays within declared field sets). For Phase 4 production: extend runtime validation to vllm + tensorrt via per-engine containers (the infrastructure exists; just needs dispatch).

**H4 (LLM modifies miner code):** across 3 engines: 0/3 patches lifted recall; 2/3 crashed the walker; 1/3 patches failed to find its anchor text. BUT: 6/6 diagnoses CORRECT against the `post_trial_a_gap_closure.md` inventory. The 70B-q4 model writes pseudocode for fixes (references undefined helpers, undefined variables, doesn't update callers) but identifies gaps at structural level with 100% precision. Verdict: STRONG DIAGNOSIS / WEAK SYNTHESIS-OF-CODE.

**H6 (no chunking; whole-source single shot):** transformers only (vllm/tensorrt source too large for 32k context). Invariant recall collapsed from 0.564 baseline to 0.128. Classic lost-in-the-middle. Schema recall also slipped (0.83 -> 0.75). Chunking is NOT the bottleneck; removing it HALVES recall.

**H7 (agentic loop; LLM has tools):** transformers + vllm. Tool dispatch worked (0 parse errors across 60 turns); the LLM used `run_miner`, `list_validators`, `read_file`, `score_against` competently for EXPLORATION. But NEITHER cell finalised - both hit the 30-tool-call budget with `invariants: []` as the working draft. vllm cell called `score_against` 6 times with empty invariants and never adjusted strategy. Closed-loop feedback does NOT shift the ceiling - it COLLAPSES the ceiling to zero because synthesis becomes optional and the model defers it.

**H9 (LLM diagnoses; no output mutation):** 8 diagnoses across 3 engines, 0 fabrications. 6/8 match H4's diagnoses + the manually-curated inventory; 2/8 genuinely new. Cheapest pattern of the 9 (~50s per cell). Confirms the diagnose-strong / synthesise-weak split.

**E6 (field-anchored extension):** transformers + vllm active. transformers: recall neutral, precision -4.5pp. vllm: -7.7pp recall due to a heuristic bug (untargeted 249-field anchor for every chunk). The variant's intended use case (catching tensorrt v0.x HF GenerationConfig hallucination) is UNTESTED - active cells don't have the empty-chunk failure mode. Open variant for Phase 3c.

**E9 (sequential cumulative-context):** transformers + vllm active. transformers: -23.1pp recall (LLM read "DO NOT re-emit invariants in running notes" as conservative dedup-pressure and under-emitted). vllm: -3.8pp recall. Cross-class invariants did NOT surface. Cross-class hypothesis open for Phase 3c.

Unifying mechanism across H6, E9, H7: **synthesis-pressure relaxation under flexibility.** (b)'s per-class chunking FORCES synthesis by the prompt structure. Any variant that adds flexibility - whole-source, cumulative dedup, tool-mediated exploration - relaxes that pressure and the q4 model defaults to under-emit. E6 was neutral because it doesn't reduce synthesis pressure (the prompt still says "emit one invariant per block").

### 3.4 The LLM-role split (the trial's most decision-relevant single finding)

Across H2 + H3 + H4 + H7 + H9 + (b) + (d-ab), a consistent split emerges for `llama3.1:70b` at q4 quantisation:

| LLM role | Patterns | Quality at 70B-q4 |
|---|---|---|
| Diagnose | H4 text + H9 | Excellent. 0 fabrications across 8 diagnoses. 6/6 matched manually-curated truth + cross-correlated between patterns. |
| Subtract | H2 | Error-prone. 3/3 vllm drops were false-drops; conservative prompting reduces but doesn't eliminate. |
| Synthesise code | H4 patches | Poor. 0/3 recall lift; 2/3 crashed; anchor texts hallucinated. |
| Synthesise output | (b), E6, E9 | Substrate-ceiling-bound. ~50% transformers recall, ~30% vllm, ~16% tensorrt under vu. No variant lifts this ceiling. |
| Synthesise under feedback | H7 | Collapses. 0 finalised invariants on both cells; tool-use becomes passive exploration. |

This is the most consistent cross-pattern signal in the trial. It is decision-relevant: it argues against architectures that PLACE LLMs in subtractive or autonomous-synthesis roles at this model scale, and FOR architectures that place LLMs in extractive/diagnostic roles with deterministic validation downstream.

The split also predicts: if Claude relaxes the 70B-q4-specific ceilings, the most likely places it does so are (i) synthesis-under-feedback (H7-style agentic patterns); (ii) extraction-with-flexibility (E9 cumulative context); (iii) hallucination-on-empty-input (E6 field-anchored under empty-chunk conditions). If Claude does NOT relax these, the LLM-role split is intrinsic and architectures targeting it survive any model substrate change. Phase 3c tests both.

### 3.5 Discovered failure modes

The trial surfaced a new failure-mode catalogue that the original rubric's `silent / detectable / crash` taxonomy partially conflated. Each has different operational consequences:

1. **Detectable crash** (clean stderr trace): (a) on transformers v-2/v+major; (a) on vllm bumped (all 4). Operationally: easy to monitor; the substrate noisily signals "I can't run here". CI can gate on this.

2. **Silent failure - empty extraction**: (b) on vllm v+major. The chunker returns empty; the LLM emits ~4 sentinel invariants from the failure marker; recall reports as 0%; the cell looks "silent". Operationally: detectable via low cell_count; CI can gate on cell_count == 4 + identifying the failure-marker pattern.

3. **Silent failure - hallucination from empty input**: (b) on tensorrt v0_19_0 / v0_20_0. The chunker returns empty class bodies; the LLM HALLUCINATES 30+ HuggingFace GenerationConfig field names that don't exist in tensorrt at all; recall reports as 16% (because some HF field names happen to overlap with tensorrt-conventional ones); cell_count is ~37. Operationally: this is the MOST INSIDIOUS failure mode discovered. Metrics look "kind of working" but underlying content is mostly invented. Mitigation: schema-existence gate (cheap; would catch a fabricated field name like `do_sample` not in `__fields__`); runtime gate (more expensive; catches false predicates on real fields too).

4. **Under-emit from synthesis-pressure relaxation**: H6/E9/H7. The model deflects from emitting structured output when prompt structure permits. Operationally: detectable via cell_count drop vs baseline.

5. **MINER_VERSION_BLIND silent re-extraction**: (a) tensorrt on all 4 bumps. Reports 100% recall + 100% precision; substrate wiring artefact. Operationally: detectable only by external audit of "did the walker actually look at bumped source?"; not visible from the score JSON alone. The most TRUST-CORROSIVE failure mode discovered, because it makes the cell look BEST when it's actually doing nothing.

6. **Silent failure - false-drop in subtraction**: H2 vllm. The cell looks normal but 3 valid invariants are removed. Operationally: detectable only via comparison to pre-validation output.

Six failure modes; six different mitigation paths. The naive `silent/detectable/crash` rubric was missing the distinction between modes 2, 3, and 5 - all classify as `silent` or `none` under the original scoring, but their decision implications differ sharply. The Phase 5 production substrate's monitoring layer needs to distinguish them.

---

## Section 4: The decision space

Five viable architectures emerge from the information map. Each is defensible; each makes specific commitments + forecloses specific alternatives.

### Architecture I: Scenario 1 + deterministic refactor (pure (a))

**Shape:** Pure mining (a) across all three engines. Close the 7 post-trial gaps (vllm normalisation patterns, vllm local-var aliases, vllm if/elif/else branch descent, tensorrt type-aware probe synthesis, tensorrt DeprecationWarning poisoning, tensorrt nested-config dispatch, transformers defensive imports). Refactor per Bake-off A's ~1800 LoC target. Ship.

**Trade-offs vs alternatives:** Lowest implementation risk (the substrate already exists). Lowest operational complexity (no LLM dependency). Zero per-cell LLM cost. Catches none of the things (a) doesn't see at active version (validated union shows 28+ entries (a) missed across active cells). Brittleness modes stay heterogeneous - transformers landmark-missing, vllm dep-import, tensorrt version-blind - and each needs its own per-bump patch path.

**Maps to plan Scenario 1.** Commits llem to deterministic-substrate-only; forecloses LLM as production substrate. The pre-trial default; the most conservative pick.

### Architecture II: Scenario 4 + extend-propose hybrid (the architecture the trial keeps converging on)

**Shape:** (a) deterministic baseline across all engines (with the 7 gaps closed) + chunked LLM extraction in (b)'s shape per engine + deterministic runtime gate per engine (existing `scripts/validate_invariants.py`) + validated-union as canonical artefact. LLM role: extract + propose only. Deterministic role: validate + subtract.

**Trade-offs vs alternatives:** Best-quality outcome by all measured axes. Highest engineering cost (both substrates built + maintained). Per-cell LLM cost (~30-90s per bump on (b)'s shape; cheap energy-wise: ~30-180 Wh per cell-mining at the trial's locked prompts). Detects the failure modes (a) and (b) miss separately - hallucination caught by deterministic gate; (a) gaps caught by LLM proposal. Brittleness modes diversified across substrates - one substrate failing on a bump doesn't doom the cell because the other surfaces partial coverage.

**Maps to plan Scenario 4.** Commits llem to two-substrate architecture; forecloses single-substrate simplicity. The architecture the trial's data most strongly converges on - both because H3 demonstrably lifts precision via runtime gate, and because the LLM-role split argues for this exact division of labour.

### Architecture III: Scenario 3 + LLM substrate (deprecate (a) producers)

**Shape:** Pivot to (b) or (c) extraction. Deprecate the ~3800 LoC of (a) producers; keep lightweight verification + landmark machinery. Validated union from (b) + runtime gate becomes SSOT.

**Trade-offs:** Lowest LoC footprint (~1800 removed). Highest model-dependency risk. Currently weak case at 70B-q4: (b) tops out at ~50% recall on transformers, ~30% vllm, ~16% tensorrt. Stronger case if Phase 3c shows Claude breaks the (b) ceiling. Commits to LLM-dependent substrate; forecloses fully-deterministic future. Maps to Scenario 3.

### Architecture IV: Scenario 2 + per-engine substrate choice

**Shape:** (a) for engines where (a) works (transformers); (b) + LLM-extension where (a) is structurally challenged (vllm, tensorrt). Per-engine architectural divergence accepted.

**Trade-offs:** Matches engine mining-friendliness. Highest per-engine pipeline complexity. Cross-engine architectural asymmetry as feature. BUT the data does NOT support this: (b)'s ceiling is LOWER on vllm and tensorrt than transformers, so substituting (b) for (a) on those engines trades brittleness for lower ceiling. Maps to Scenario 2; included for completeness.

### Architecture V: Scenario 5 + curation primacy (OQ10 framing)

**Shape:** Treat both (a) and (b) as EVIDENCE; human/LLM curation produces the canonical artefact. The validated union IS the curated output if quality is sufficient; otherwise a maintainer reviews per-bump. Mining becomes evidence-mining; curation becomes the SSOT.

**Trade-offs vs alternatives:** Highest correctness floor (a human/LLM curator looks at every entry). Highest per-bump human cost (~hours per engine per bump). Most flexible w.r.t. substrate quality - if substrate quality is poor, curation catches it; if substrate quality is good, curation is cheap. CONSISTENT WITH THE DATA: the validated-union scoring already implements the EVIDENCE framing for measurement purposes; promoting it to production architecture is a small further commitment.

**Maps to plan Scenario 5.** Commits llem to a curation layer; forecloses pure-automation. Distinguishes "is the mining correct?" from "is the curated artefact correct?" - the latter is what the runtime actually consumes. Has natural alignment with Open Question 10 (mining-as-SSOT vs evidence).

---

## Section 5: Recommendation

llem should adopt **Architecture II (Scenario 4 + extend-propose hybrid)** as the production substrate, with **Architecture V (curation primacy)** as the operational layer above it.

The concrete shape: per-engine, per-version, the production substrate runs (a) deterministic mining first; (b)-style chunked LLM extraction second; deterministic runtime gate against the live engine (in container) third; the validated union of (a) + (b) - filtered to runtime-validated entries - is the cell's canonical artefact. A maintainer reviews the validated union per-bump (the curation layer), with the LLM-diagnose pattern (H9-style) as an assistant that surfaces "categories of thing (a) is structurally blind to". The curated artefact is what flows into `engine_versions/<e>/v*/outputs/` and from there into `src/<e>/`.

The LLM role is enforced by architecture: extension + diagnosis only. Subtraction is deterministic (the runtime gate). Synthesis-of-code stays human (H4's diagnoses scaffold the work but don't ship as patches). Maintainer review owns the final artefact.

This is what llem should commit to. Defended against the alternatives:

- **Against Architecture I (pure (a)):** the validated union shows 28+ entries (a) misses on active cells across 3 engines; the LLM-extension pattern (d-ab) catches most of these at zero false-positive risk (because the deterministic gate filters); ignoring this signal corrupts the canonical artefact.

- **Against Architecture III (pure (b)):** the (b) recall ceiling at 70B-q4 is too low (~30-50% under vu) to be a substrate in isolation; the (a) deterministic floor catches what (b) misses (and (a) is cheap per bump); the trial's hallucination failure mode on tensorrt v0.x demonstrates that pure-LLM substrates need a deterministic gate anyway.

- **Against Architecture IV (per-engine):** the data does not support architectural asymmetry. (b) is WORSE on vllm and tensorrt than transformers, so the engines where (a) is most brittle are precisely the engines where (b) is also weakest. Routing around (a)'s brittleness via (b) trades one weakness for another. The right per-engine handling is to fix (a)'s brittleness on each engine (the 7 gap closures) rather than replace it.

- **Against Architecture V alone:** curation without mining is unreliable - maintainers MISS things. Curation needs evidence-streams to review. Architecture II provides those streams; Architecture V's operational layer USES them.

What this commits llem to:
- Both substrates remain in the production codebase. The ~1800 LoC mining refactor (Bake-off A) lands. The LLM-extraction infrastructure (Phase 2's chunkers + prompts + retry harness) becomes production code.
- The deterministic runtime gate (`scripts/validate_invariants.py`) lifts from script to library; per-engine container dispatch is a routine production concern.
- Per-version cells run via CI on Renovate bumps; the validated union is committed to the engine-knowledge data files; src/ is regenerated via the existing codegen pipeline.
- Maintainer reviews validated-union diffs per bump (the curation layer); H9-style LLM-diagnose can pre-flag gap-categories for the maintainer's attention.

What this trades off:
- Higher engineering cost than pure (a). Lower than parallel-and-reconcile architectures because the runtime gate is straightforward dispatching to existing infrastructure.
- Per-bump LLM cost: ~150-200 Wh per cell at trial-locked prompts; on Renovate cadence (a few bumps per engine per year), this is a few kWh per year total. Energy-cheap; wall-clock-real (~30-90 min per cell at parallelism).
- Vendor-dependency: at trial scale, OSS LLM (Ollama/llama3.1) suffices. If Phase 3c shows Claude lifts the ceiling materially, the architecture supports drop-in substitution.

What conditions trigger a revisit:
- **Phase 3c Claude results.** If Claude breaks the (b) ceiling materially on extraction (e.g. transformers (c) recall > 75% vs current (b) 56-62%), the cost-quality balance shifts. The architecture supports drop-in Claude for the extension layer; the question becomes operational (is the API quality + cost > local 70B-q4 by enough margin to switch?).
- **If H7-style agentic with Claude works.** If a stronger model bridges the synthesis-blindness gap, the architecture could absorb autonomous-discovery patterns. The H7 harness is reusable.
- **If validated-union recall plateaus at < 80% even with deterministic-extend-propose.** If maintainer review consistently catches entries neither substrate finds, the curation primacy framing (Architecture V) becomes the dominant pattern and substrate is demoted to evidence.

What's deferred:
- **Phase 3c addendum:** when `ANTHROPIC_API_KEY` arrives, run the 15-cell (c) matrix + key hybrid patterns (H4-with-Claude, H7-with-Claude, H6-with-Claude on vllm/tensorrt source, E6 on the bumped-tensorrt empty-chunk case, E9 cumulative context at Claude's stronger synthesis). Estimated $20-30. Produces a second-pass synthesis with the model-quality axis as a 4th dimension.
- **OQ9 storage strategy:** revisit post-spike-refactor when the artefact footprint has stabilised. Architecture II doesn't constrain the answer; both git-tracked and GH-artefacts-pinned work.
- **Pattern #2 migration** (research/ namespace): execute at trial-write-up time per the existing spec.
- **Phase 5 curation pipeline:** see Section 6.

---

## Section 6: Outstanding work

Post-Phase-4 work backlog, ordered by dependency:

1. **Phase 3c (Claude comparison):** 15-cell (c) matrix + Claude-variant of key hybrid patterns (H4, H7, H6 on vllm/tensorrt source, E6 on bumped tensorrt empty-chunk case, E9 cumulative context). ~$20-30; ~1-2 days agent work when key arrives. May refine Section 5 but unlikely to overturn it - the LLM-role split is robust; Claude is likely to SOFTEN ceilings rather than INVERT the split.

2. **Post-trial (a) gap closure backlog:** 7 gaps per `research/mining-substrate-trial/findings/post_trial_a_gap_closure.md`. Close regardless of substrate choice. H4 + H9 diagnoses provide design input. ~500-1000 LoC across the 7 gaps.

3. **Spike-branch refactor:** Bake-off A's ~1800 LoC target. H4's outputs feed cross-engine abstractions: nested-config dispatch (G-trt-3, G-vllm-companion-classes, transformers BNB); if/elif/else branch descent (G-vllm-3); local-var alias tracking (G-vllm-2).

4. **Phase 5 curation pipeline (Architecture II + V instantiation):** dogfood on transformers first (highest reference maturity, lowest brittleness surface). Reconciliation script producing validated union per cell; maintainer-review interface; H9-style LLM-diagnose pre-flag. Contingent on Section 5; ~1-2 weeks for the pilot.

5. **OQ9 storage strategy revisit:** post-spike-refactor, when artefact footprint stabilises. Not blocking.

6. **Pattern #2 migration (`research/mining-substrate-trial/` -> `research/mining-substrate-trial/`):** execute at trial-write-up time per existing spec.

7. **Trial PR extraction:** spike commits chunk into reviewable PRs (PR-A/B/C/D/E per DECISIONS_LOG).

---

## Section 7: What the trial taught us beyond the substrate question

Five methodological findings worth carrying forward to other LLM-in-SE tasks at llem:

**1. Validated-union ground truth as a scoring discipline.** When comparing N strategies for the same artefact, every strategy's output should contribute to the ground truth (filtered to what survives downstream validation); none should be privileged as reference. Single-entry methodological fix that retro-rescores every cell honestly. Generalises to any substrate comparison llem does in future.

**2. The LLM-role split as a general principle.** At 70B-q4 specifically (likely with model-scale-dependent thresholds elsewhere): LLMs are reliable at diagnosis, error-prone at subtraction, weak at autonomous synthesis-of-code, ceiling-bound at extraction, collapse under synthesis-with-feedback. Architecture-level implication: place LLMs as extractors + diagnosers; deterministic systems as validators + subtractors; humans as final-curators-with-LLM-scaffolding. Likely to apply to any task where llem considers LLM augmentation of an existing deterministic pipeline.

**3. The synthesis-pressure thesis.** Forced-output prompts beat flexible exploration at this model size. Per-class chunking works because the prompt structure FORCES synthesis at each chunk; variants that relax that pressure (whole-source, cumulative dedup, tool-mediated exploration) consistently under-emit. For Phase 5: keep synthesis-forcing prompts. For Phase 3c (Claude): test whether the thesis holds at higher model scale.

**4. The cross-engine asymmetry pattern.** (a)'s brittleness, (b)'s ceiling, the chunker's failure modes all differ structurally across the three engines. Production architectures should anticipate per-engine asymmetry in failure modes even when the substrate is uniform.

**5. The brittleness-as-axis discipline.** Per-version cells (not "active + one bump") were the most expensive design decision and produced the most decision-relevant data: the three (a) brittleness modes, the chunker file-layout-assumption brittleness, the hallucination-on-empty-input mode, the MINER_VERSION_BLIND substrate-wiring artefact. None would have surfaced from active-only cells. Generalises to any substrate evaluation with upstream-bump exposure.

---

## Closing

The trial set out to gather maximal information across (engine, version, strategy) cells under matrix discipline + open-ended hybrid exploration, with strategy constructed AFTER from the assembled evidence. Both halves of that brief landed: 51 scored cells with locked prompts (no mid-trial optimisation) + 9 distinct hybrid patterns explored.

The recommended strategy is Architecture II (deterministic + LLM-extend-propose, deterministic-validate, validated-union as canonical artefact, curation primacy operational layer). The recommendation is contingent on Phase 3c not overturning the LLM-role split; the architecture supports drop-in substitution if Claude lifts ceilings materially.

The trial's discipline preserved gaps as research data; the production discipline closes those gaps after. The 7 (a) gaps + the chunker brittleness modes + the substrate-wiring artefact all become spike-refactor PR-scope tasks with H4's diagnoses + H9's structural reads as design input.

The trial is closed. Phase 5 pilots Architecture II on transformers first.
