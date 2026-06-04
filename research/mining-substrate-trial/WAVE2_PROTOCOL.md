# Mining-substrate trial - Wave 2 protocol (cost-frontier rewrite)

**Status:** Locked pre-registration. Authored 2026-06-04. Supersedes the recall-maximisation framing of the first draft (deleted same day after user direction).
**Scope:** Internal research; goal is to identify the cheapest substrate that gives reliable coverage when run as a CI step on every upstream-engine bump.
**Hardware budget:** 4x A100-40GB (160 GB pooled VRAM). No remote-API LLMs in Wave 2 (Claude/GPT deferred to a possible Wave 3).
**Cross-refs:** `findings/trial_epistemic_framing.md` (Wave 1 protocol), `RESEARCH_WRITEUP.md` (Wave 1 results), `DECISIONS_LOG.md` (live narrative continues).

---

## 1. The actual decision Wave 2 is making

llem will eventually run the chosen substrate as a CI step. Every time a `renovate` PR bumps `transformers` / `vllm` / `tensorrt-llm` (or the future SGLang / LMDeploy), the substrate runs against the new engine source and emits the updated invariants + schema. The CI step has a real budget: ideally sub-minute, plausibly sub-10-minute, hard cap somewhere in the low tens of minutes.

This reframes the substrate question. Wave 1 measured recall + precision in isolation. Wave 2 measures recall + precision **per unit of compute cost** and identifies where on the cost-recall Pareto frontier the production substrate should sit.

The four sub-questions Wave 2 answers:

1. **Is pure deterministic the way?** If yes, which deterministic substrate? The existing AST walkers, tree-sitter, Pydantic-native, runtime-trace, or hypothesis-driven fuzzing? Cost target: seconds per bump, zero LLM dollars.
2. **Is pure LLM the way?** If yes, what's the cheapest LLM that gives acceptable coverage? Cost target: minutes per bump, single-LLM-call energy budget.
3. **Is a cheap hybrid the way?** If yes, where exactly is the LLM most leverage per dollar? (Det floor + small-LLM extension only? Det extract + LLM gate on failure only? Det everything + LLM diagnose-only as periodic audit?) Cost target: somewhere between the two above.
4. **What's the ceiling if cost is no object?** Benchmark with the strongest local LLM (Llama-3.3-70B fp16, DeepSeek-Coder-V2-236B) so we know the upper bound when reasoning about cost-quality trades.

---

## 2. Cost tiers

All Wave 2 strategies are pre-classified by per-cell cost envelope. The classification drives which sub-wave a strategy belongs to.

| Tier | Per-cell wall-clock | Per-cell energy | Per-cell $$ (rough) | Eligible for CI? | Purpose |
|---|---|---|---|---|---|
| A (CI-affordable) | < 60 s | < 1 Wh | $0 | Yes, every bump | Production candidate |
| B (CI-tolerable) | 1-15 min | 1-20 Wh | < $0.10 | Yes, with care | Production fallback |
| C (benchmark only) | 15 min - 4 h | 20-500 Wh | $0.10 - $5 | No | Ceiling-finding only; informs cost-quality trade reasoning |

Cost estimates are wall-clock x measured power x electricity price ($0.15/kWh assumed) and exclude amortised model-pulling. The classification is enforced at protocol-lock time; strategies that empirically exceed their tier during execution get re-tiered in `findings/wave2_deviations.md` but the original assignment is preserved.

---

## 2bis. Systematic cell-selection principle

Wave 1 ran a full strategy x engine x version matrix (51 cells); Wave 2 deliberately does NOT replicate that breadth. Instead, every Wave 2 cell is selected to maximally distinguish between competing hypotheses on a specific axis. A cell is "informative" if its outcome changes the production recommendation; cells that only confirm what other cells already establish are skipped.

This produces a tighter ~40-50 cell matrix biased toward axes Wave 1 left open, with each Tier C "benchmark only" pattern represented by a single canonical cell (typically transformers active, the highest-baseline-reference engine) rather than a full sweep. Combinatorial explosion is the explicit failure mode being avoided.

Design axes that drive cell selection:

| Axis | Levels | Wave 1 coverage | Wave 2 coverage |
|---|---|---|---|
| Substrate (det vs LLM vs hybrid) | 3 | full | full |
| Det approach | walker / tree-sitter / pydantic-native / runtime-trace / fuzz | walker only | walker + 4 alts |
| LLM scale | 8B / 32B / 70B / 200B+ | 8B + 70B at q4 | 7B-14B small sweep + 32B-236B big benchmark |
| LLM call shape | 1-shot / vote / multi-step | 1-shot only (b); H4 multi-step patch | 1-shot + k=3 vote + critic + reflective + ToT (one cell each) |
| Substrate feed | source / stubs / docs / AST / RAG | source only | source + stubs + AST (one cell each); skip docs + RAG (heavy infra, low decision value) |
| Temperature | 0 / non-zero | 0 only | 0 + 0.3 in small-LLM sweep; 3-temp ablation on one cell |

The Tier C "informative-benchmark" cells (section 3.4) are each chosen to answer one such design-axis question with the cheapest experiment that yields a clear yes/no signal.

---

## 3. The Wave 2 strategy space, re-tiered

### Tier A: pure-deterministic substrates (the priority)

This is where Wave 2 has the most evidence to contribute. Wave 1 left (a) at ~46% validated-union recall, and the question "can we lift the deterministic floor cheaply?" is largely open.

| ID | Substrate | Cost | Coverage scope | Hypothesis |
|---|---|---|---|---|
| W2-a (Wave 1 baseline) | Handwritten per-engine AST walkers | A | 3 engines x 5 versions | Existing baseline; held in for comparison |
| W2-a-treesitter | tree-sitter Python grammar; universal query patterns | A | 3 engines x active + 2 bumps | Universal parser cuts per-engine LoC; recall floor depends on query expressiveness |
| W2-a-pydantic-native | engine-imported; `Model.__fields__` + `model_validator` reflection | A | 3 engines x active | Engine's own framework exposes the validator surface; ~50 LoC per engine |
| W2-a-runtime-trace | monkey-patch validator entry points at import; capture raising arg patterns | A | transformers + vllm active (pilot) | Dynamic invariant discovery survives upstream refactor |
| W2-a-fuzz | hypothesis-strategies on engine config; cluster raises into invariants | B | transformers active (pilot only) | Pure black-box discovery; high promise but possibly expensive depending on convergence |

### Tier B: cost-frontier-driven LLM sweep (Wave 2f)

The most consequential addition under the reframe. Wave 1 only tested 70B-q4 + 8B-q4. Wave 2f asks: what's the smallest LLM that gives CI-acceptable coverage?

| ID | Model | Footprint | Tier | Hypothesis |
|---|---|---|---|---|
| W2f-qwen-coder-7b | Qwen2.5-Coder-7B-Instruct fp16 | ~15 GB (1 A100) | A or B | Code-specialised; cheapest viable candidate |
| W2f-deepseek-coder-v2-lite | DeepSeek-Coder-V2-Lite-Instruct 16B (q4 or fp16) | ~10-32 GB (1 A100) | B | MoE code specialist at small scale |
| W2f-phi4-14b | Phi-4-14B-Instruct fp16 | ~28 GB (1 A100) | B | Reasoning-strong small model |
| W2f-llama31-8b | Llama-3.1-8B-Instruct fp16 (Wave 1 baseline upscaled from q4) | ~16 GB (1 A100) | B | Unquantised version of Wave 1's 8B probe |

**Execution shape:** each small model x 3 engines x active = 12 cells per sub-pattern. Run on the standard (b) pipeline AND on the cheapest hybrid (d-ab) to map both pure-LLM and hybrid cost curves. Total 2f: ~96 cells, but each cell is fast (~2-10 min) so total wall ~1-2 days.

### Tier C: big-LLM ceiling benchmark (reduced from original 2a)

Kept as a benchmark only. Establishes the upper bound; not a production candidate.

| ID | Model | Footprint | Hypothesis |
|---|---|---|---|
| W2C-llama33-70b-fp16 | meta-llama/Llama-3.3-70B-Instruct (fp16) | ~140 GB across 4 GPUs | Unquantised baseline; q4 vs fp16 confound elimination |
| W2C-qwen-coder-32b-fp16 | Qwen/Qwen2.5-Coder-32B-Instruct (fp16) | ~64 GB | Code-specialised at medium scale |
| W2C-deepseek-coder-v2-q4 | deepseek-ai/DeepSeek-Coder-V2-Instruct (q4_K_M) | ~120 GB | MoE code specialist at large scale |
| W2C-mixtral-8x22b-q4 | Mixtral-8x22B-Instruct (q4_K_M) | ~80 GB | Reasoning-strong dense-MoE baseline |

**Execution shape (reduced):** each model x 2 engines (transformers + vllm) x active only = 8 cells per model = 32 cells total. NOT a full 3x5 sweep. The point is to find the ceiling, not to map per-engine asymmetry at large scale.

### Tier B+: cheap-hybrid candidates (reduced from original 2c)

Two patterns survive the cost cut:

| ID | Pattern | Tier | Why this one survives |
|---|---|---|---|
| W2-h15 | Closed-loop H3+H2: det extracts -> runtime gate -> failure list back to LLM -> LLM re-emits accounting for failures | B | LLM invoked only when det partially fails; per-cell LLM cost is sub-(b) |
| W2-h11-small | Self-consistency on small-LLM (b) with k=3 votes at t=0.5 | B | If small-LLM stability is the bottleneck, k=3 voting is cheap variance reduction |

### Tier C single-cell informative benchmarks (the 2-cell re-add)

Two cells from the previously-dropped 2b list survive as informative single-cell benchmarks because their substrate-prep cost is cheap and the result directly answers a design-axis question. Each runs on transformers active only with the same locked (b) prompts.

| ID | What it tests | Single cell justification |
|---|---|---|
| W2-b-stub-bench | Does the pyright type-stub surface alone (no function bodies) carry the validator signal? | One cell answers yes/no. If recall on transformers active is < 30%, stub substrate is dead. If recall >= (b) baseline, stub-as-substrate becomes a real Wave 3 candidate. |
| W2-b-tree-bench | Does feeding tree-sitter-parsed AST nodes (vs raw text) shift the LLM extraction quality? | One cell. If recall is within +-5pp of (b) baseline, the substrate shape is not the bottleneck (confirms Wave 1 H6 finding). If clearly different, opens a new design axis. |

### Patterns dropped entirely (heavy AND not CI-realistic)

Per user direction 2026-06-04: any pattern that is both LLM-heavy AND unrealistic for CI on every upstream-engine bump is dropped, not deferred. The following are out of Wave 2 entirely:

- **W2-b-doc** - Sphinx XML build infrastructure per engine; heavy substrate-prep, low pre-trial signal.
- **W2-b-rag** - vector DB indexing + retrieval per cell; heavy substrate-prep, unproven leverage.
- **W2-h10 critic-loop** - 3x LLM calls minimum; not CI-realistic; multi-agent debate is exploratory not operational.
- **W2-h12 model-ensemble** - N-model cost; Wave 1 already showed strategies complement-not-substitute (unique-contribution count near zero except H6/E6/E9 single-digit entries), so ensemble unlikely to help.
- **W2-h13 tree-of-thought** - planner + executor multi-call; not CI-realistic.
- **W2-h14 reflective self-revision** - 2x LLM calls; Wave 1's H6/E9 evidence already suggests synthesis-blindness at q4 is intrinsic, not correctable by re-reading.
- **W2-h16 standalone temp sweep** - folded into 2f instead (each small model runs at t=0 + t=0.3, costs only 2x per model rather than 4x).
- **W2-h17 iterative walker-mutate-rerun** - Wave 1 H4 single-shot already showed 0/3 patches lift recall and 2/3 crashed; iteration likely amplifies the failure mode rather than fixing it.

### Tier A external validity: SGLang + LMDeploy

| Engine | Walker LoC budget | Strategies run on this engine |
|---|---|---|
| SGLang | 400 LoC | W2-a, W2-a-treesitter, W2-a-pydantic-native; if any W2f small-LLM gets Tier-A cost on existing engines, run that here too |
| LMDeploy | 400 LoC | Same |

**Execution shape:** vendor at 5 versions each; pure-strategy matrix (a + treesitter + pydantic-native) = 30 cells. Skip the heavy LLM strategies on these engines until Wave 2.0 evidence picks a winner.

---

## 4. The four-question matrix

How each strategy bucket answers each Wave 2 sub-question:

| Question | Strategies that answer it | Evidence shape |
|---|---|---|
| 1. Is pure deterministic the way? | W2-a (baseline), W2-a-treesitter, W2-a-pydantic-native, W2-a-runtime-trace, W2-a-fuzz | Compare validated-union recall + per-bump cost; identify the deterministic Pareto-optimal substrate |
| 2. Is pure LLM the way? Cheapest viable LLM? | W2f sweep across 4 small LLMs on (b) | Identify the smallest LLM where recall is within X pp of the Tier C ceiling |
| 3. Is cheap hybrid the way? Where's the LLM leverage? | W2-h15 (closed-loop), W2-h11-small (self-consistency on small LLM), Wave 1's d-ab re-run with small LLMs | Compare hybrid recall vs (pure-det + pure-LLM) on each engine; cost-per-recall-pp analysis |
| 4. Ceiling at unlimited cost | W2C big-LLM benchmark cells | Single number per engine: what's the maximum recall any (b)-shaped strategy can hit on 4xA100? |

The synthesis answers the production question by intersecting the four answers:

- If pure-det Pareto-optimal achieves >= 80% of W2C ceiling at A-tier cost: ship pure-det.
- Else if cheap-hybrid achieves >= 90% of W2C ceiling at B-tier cost: ship cheap-hybrid.
- Else if cheapest viable LLM achieves >= 80% of W2C ceiling at B-tier cost: ship pure-small-LLM.
- Else: ship the most cost-effective B-tier strategy and document the gap; defer the upper-cost work to maintainer-review (Wave 1 Architecture V).

---

## 5. Discipline rules (carried from Wave 1, plus cost discipline)

Wave 1's five rules apply. Adding three Wave-2-specific:

**F. Every cell records wall-clock + energy.** No exceptions. The cost-Pareto-frontier analysis needs both. Use the existing `llenergymeasure.energy.select_energy_sampler` infrastructure to record GPU energy; record wall via `time.perf_counter`. Drop the per-cell record into `findings/wave2_costs.jsonl`.

**G. Tier-A strategies get tier-A scoring discipline.** No multi-pass refinement, no retries beyond what production CI would tolerate. If a strategy needs 3 retries to converge in the trial, it's not Tier A even if the median run is < 60 s. Record the retry rate.

**H. Tier C cells run last.** Cells are run sub-wave at a time; A first, then B, then C. Early A/B evidence can shrink C if a clear winner emerges (e.g. if Tier A pure-det beats W2-h15 in cost-adjusted terms, big-LLM benchmark on hybrid patterns is not needed).

---

## 6. Execution plan

### Wave 2.0 - cost floor and ceiling (target: 5-7 calendar days, ~28 cells)

| Step | Cells | Deliverable |
|---|---|---|
| 1. Land scaffolding + protocol | 0 | This doc + `scripts/strategies/wave2/` stubs |
| 2. Pin all model digests | 0 | `findings/wave2_model_digests.toml` filled |
| 3. Run W2-a-treesitter on 3 engines x active | 3 | Tier-A det candidate; aggregate to `wave2a_det_alternatives.md` |
| 4. Run W2-a-pydantic-native on 3 engines x active | 3 | Tier-A det candidate; aggregate to same |
| 5. Run W2-a-runtime-trace pilot (transformers + vllm active) | 2 | Tier-A; decide expansion |
| 6. Run W2-a-fuzz pilot (transformers active) | 1 | Tier-B; decide expansion |
| 7. Run W2f small-LLM sweep at t=0: 4 models x transformers active | 4 | Identify cheapest viable LLM |
| 7b. Best 1 of 4 small models at t=0.3: same 3 cells | 3 | Temperature variance probe (folds in old h16 axis) |
| 7c. Best 1 of 4 small models on vllm + tensorrt active | 2 | Cross-engine viability of small-LLM winner |
| 8. Run W2C big-LLM ceiling: 3 best models x transformers active | 3 | Ceiling-finding |
| 8b. Best big model on vllm + tensorrt active | 2 | Cross-engine ceiling |
| 9. Tier C single-cell informative benchmarks: W2-b-stub-bench + W2-b-tree-bench on transformers active | 2 | Substrate-feed-shape yes/no signal |
| 10. Cost/recall Pareto plot | 0 | `findings/wave2_cost_frontier.md` |
| Total Wave 2.0 | ~25 | |

### Wave 2.1 - cheap hybrid + external validity (target: 3-5 calendar days, ~15 cells)

| Step | Cells | Deliverable |
|---|---|---|
| 11. Run W2-h15 closed-loop on transformers + vllm active (small LLM picked from 2f) | 2 | Cheap-hybrid candidate; aggregate to `wave2h_cheap_hybrid.md` |
| 12. Run W2-h11-small on transformers active | 1 | One-cell variance-vote benchmark |
| 13. Vendor SGLang + LMDeploy (3 versions each: v-1 + active + v+1, NOT full 5) | 0 | engine_versions/ trees + per-engine chunkers |
| 14. Run W2-a + Pareto-winning Tier A strategy on SGLang x 3 versions | 6 | External validity |
| 15. Run W2-a + Pareto-winning Tier A strategy on LMDeploy x 3 versions | 6 | External validity |
| 16. Wave 2 synthesis | 0 | Update `RESEARCH_WRITEUP.md` with Wave 2 sections; produce `findings/wave2_synthesis.md` with the production recommendation |
| Total Wave 2.1 | ~15 | |

Wall-clock estimate: ~8-10 calendar days end-to-end (Wave 2.0 + 2.1 ~= 40 cells total). Tractable on the 4xA100 budget.

---

## 7. Success criteria

Wave 2's headline question: **what's the cheapest substrate strategy that gives reliable coverage in CI?**

Decision rules:

- **If a Tier A strategy achieves >= 65% validated-union recall AND >= 65% precision on at least 3 of 3 engines:** that's the production substrate. Cost frontier is settled; reasoning about LLM augmentation becomes optional.
- **If no Tier A strategy clears that bar but a Tier B strategy does:** ship the Tier B strategy; document the CI cost (likely a few minutes per bump).
- **If neither Tier A nor Tier B clears the bar but Tier C does:** the production substrate is "Wave 1's Architecture II + V" essentially unchanged, and Wave 2's contribution is the comprehensive negative result on Tier A/B alternatives. This is still decision-relevant.
- **Always record cost per recall pp.** Even if a Tier A strategy "wins" by absolute cost, the slope of recall improvement per dollar might justify a Tier B hybrid for a specific use case.

---

## 8. Known threats to Wave 2 validity

**T1. Model-pinning drift on small models.** Small models bump weights frequently. Mitigation: digest-pinning per discipline F.

**T2. Hardware contention.** 4x A100 shared. Mitigation: sequential cell execution; concurrency only for small (1xA100-fitting) models in the W2f sweep.

**T3. Cost-tier classification drift.** A strategy classified Tier A pre-execution might run > 60 s on adversarial inputs. Mitigation: log to `wave2_deviations.md` but preserve original tier for synthesis.

**T4. tree-sitter Python grammar lag.** Recent Python syntax may not parse. Mitigation: pin tree-sitter-python version; AST fallback.

**T5. Engine container pinning for runtime-trace.** W2-a-runtime-trace imports the engine; depends on container determinism. Mitigation: container digests in `wave2_model_digests.toml`.

**T6. Fuzz convergence variance.** W2-a-fuzz's wall-clock depends on hypothesis's search budget. Mitigation: fixed budget per cell (e.g. 60s wall or 1000 examples, whichever first); record budget exhaustion as a failure mode.

---

## 9. Out-of-scope (Wave 3 candidates, deferred)

- Claude / GPT API runs.
- Statistical inference (bootstrap CIs, seed variance, multi-run agreement).
- Layer B differential testing (behavioural validation as fourth gate).
- llama.cpp / consumer-GPU engines.
- The expensive hybrid patterns (W2-h10, W2-h12 full, W2-h13, W2-h14, W2-h16, W2-h17).
- Pure-LLM substrate alternatives (b-stub, b-doc, b-rag, b-tree).
- Property-based test generation, SMT/Z3 targets, mutation-based discovery beyond fuzz.

---

## 10. Wave 2 cross-references (forward)

- `findings/wave2_model_digests.toml` - pinned model + container SHAs.
- `findings/wave2a_det_alternatives.md` - W2-a-treesitter / W2-a-pydantic-native / W2-a-runtime-trace / W2-a-fuzz aggregate.
- `findings/wave2f_small_llm_sweep.md` - 2f small-LLM aggregate.
- `findings/wave2C_ceiling.md` - 2C big-LLM benchmark.
- `findings/wave2h_cheap_hybrid.md` - 2-h15 + 2-h11-small aggregate.
- `findings/wave2_cost_frontier.md` - cost/recall Pareto plot + analysis.
- `findings/wave2_synthesis.md` - the production recommendation.
- Updated `RESEARCH_WRITEUP.md` with Wave 2 sections appended.
- Updated `findings/trial_matrix_vu.md` with new strategy columns.

---

*Protocol locked 2026-06-04 (cost-frontier rewrite). Earlier draft (recall-maximisation framing) was superseded same day; not preserved. Changes to this document after 2026-06-04 are deviations and must be logged in `findings/wave2_deviations.md` with rationale.*
