# Empirical trial roadmap — mining strategies across engines and version bumps

**Status:** Proposed (no execution yet).
**Opened:** 2026-05-25 (post end-of-session-4 bake-off; user direction to scale up).
**Cross-ref:** `research/mining-substrate-trial/findings/mining_strategy_bakeoff.md` (the framing this roadmap operationalises), `research/mining-substrate-trial/DECISIONS_LOG.md` (chronological log).

This document is the execution plan for the full empirical trial the user proposed:

> Full empirical trial across all 3 engines, across various version bumps major and minor:
> a) pure mining & discovery (status quo)
> b) pure OSS LLM (running on GPUs, with proper setup)
> c) pure Claude API
> d) mixture of a & b / a & c

> Final result written to `engine_versions/` where we'd have human curation (curated.yaml, overlay.yaml) → injected into `src/` once maintainer-reviewed.

The trial output is a quantitative answer to: **which mining substrate(s) should llem invest in long-term?**

---

## The matrix

3 engines × 2-3 version bumps each × 4 strategies = **24-36 cells**, each producing:
- A candidate `schema.discovered.json`
- A candidate `invariants.proposed.yaml`
- Scored against an authoritative reference (see scoring below)
- With wall-clock and energy cost recorded

### Engines

| Engine | Active version | Mining maturity today |
|---|---|---|
| transformers | 4.57.3 | Mature: 41 invariants, full deterministic stack (schema + static + dynamic) |
| vllm | 0.7.3 | Skeletal invariants (10 proposed, 0 validated); no dynamic miner |
| tensorrt | 0.21.0 | Almost-empty invariants (3 proposed, 1 validated); no dynamic miner |

### Version bumps

Per engine, pick **one minor** and **one major** bump beyond the active version. Minor tests adaptation to small upstream changes; major tests adaptation to architectural shifts.

| Engine | Minor bump candidate | Major bump candidate |
|---|---|---|
| transformers | 4.58.x or 4.59.x (whatever latest 4.x is) | 5.9.0 (already tested for v4 producer transfer in bake-off D) |
| vllm | 0.8.x or 0.9.x | 0.16.x / 0.18.x / 0.19.x (per existing vendored archive intent) |
| tensorrt | 0.22.x or later | 1.0.x / 1.2.x (per existing vendored archive intent) |

Concrete versions to lock in: probe PyPI / GitHub for the latest stable in each band before locking. **Recommended:** lock at design time, don't drift mid-trial.

### Strategies

| ID | Name | What it does |
|---|---|---|
| **a** | Pure mining | Run the existing handwritten producers (schema_introspector + static_invariant_miner + dynamic_invariant_miner). transformers has full stack; vllm + tensorrt need lift to parity. |
| **b** | Pure OSS LLM | Run a local LLM (llama3.1:70b via Ollama, or alternative if needed) on raw engine source. With proper LLM-side setup: chunking, structured output, retry, possibly multi-pass refinement. |
| **c** | Pure Claude API | Same prompt as (b) but Claude as the model. Higher quality reference; vendor dependency. |
| **d-ab** | Hybrid: mining + OSS LLM | Run (a) first; LLM reads the output + source; identifies gaps; extends invariants; flags risks. |
| **d-ac** | Hybrid: mining + Claude | Same as d-ab with Claude as the verifier/extender. |

The user's framing was 4 strategies (a, b, c, d) but d splits naturally into d-ab and d-ac. **Decision: include both d-ab and d-ac unless cost is a blocker; they answer different questions.**

---

## Per-engine readiness assessment

Trial fairness requires that each engine has equally mature deterministic mining (strategy a) before the LLM strategies are scored — otherwise the LLM looks artificially good on engines where the deterministic baseline is undermatured.

| Engine | What's missing | Estimated work to parity |
|---|---|---|
| transformers | Nothing critical | 0 (baseline) |
| vllm | Dynamic miner; invariant coverage 10 → ~30-40 | 2-3 days (use static miner deeply; possibly add dynamic miner against vllm.EngineArgs) |
| tensorrt | Dynamic miner; invariant coverage 3 → ~15-25 | 2-3 days (TrtLlmArgs static walk + some dynamic probing) |

**Caveat:** the "missing" diagnosis is partly an indictment of `static_invariant_miner` reach for those engines. Bake-off A flagged the refactor potential of the miners; if we refactor first, "bringing vllm + tensorrt to parity" may overlap with the refactor.

### Decision point before Phase 2

After Phase 1 (mining parity) but before Phase 2 (LLM extraction), there's a meaningful choice:

- **Refactor first**: collapse the ~3800-LoC deterministic stack to ~1800 LoC per Bake-off A's analysis, then extend vllm + tensorrt static miner reach on the refactored shape. Larger upfront investment, cleaner per-engine work.
- **Extend first, refactor later**: extend vllm + tensorrt mining on the current `_base.py` shape; refactor (or pivot to LLM) post-trial. Lower investment per engine but stalls the refactor.

The roadmap below assumes **extend first, refactor later** — gets to the trial faster, keeps the refactor option open based on trial outcome.

---

## Per-strategy infrastructure needs

### Strategy (a) — pure mining

Already exists for transformers. For vllm + tensorrt:
- Audit static miner coverage gaps per engine (compare against known invariant lists in upstream tests, e.g. `vllm/tests/test_engine_args.py`)
- Extend `static_invariant_miner.py` to walk relevant validators (EngineArgs `__post_init__`, etc.)
- Optionally add a dynamic miner for runtime-only invariants (cluster probing as in transformers)

Estimated: 5-6 days total across both engines.

### Strategy (b) — pure OSS LLM

Bake-off B showed a single-prompt single-pass approach hits ~50% recall. Need to improve via:

1. **Chunking strategy**: split engine source by class/method, prompt-per-chunk, then merge. Avoids single-prompt truncation/timeout.
2. **Structured-output constraint**: use Ollama's `format: "json"` mode AND validate output against an explicit JSON Schema. Retry on parse failure with corrective re-prompt.
3. **Few-shot prompting**: prepend 2-3 ground-truth examples from `engine_versions/transformers/v4_57_3/outputs/`.
4. **Multi-pass refinement**: extract → verify against source → extend. "Reflexion"-style.
5. **Possibly LangChain or instructor**: for orchestration. Decision: try without first (lower complexity); add if needed.
6. **Model variant probe**: test 8B (faster, lower quality) and 70B (slower, higher quality) on a known cell. If 8B is 80% as good for 10x speed, that changes the trial economics.

Estimated infrastructure: 3-4 days.

### Strategy (c) — pure Claude API

Setup is much simpler than (b) (no local-inference engineering). Needs:
1. `ANTHROPIC_API_KEY` from user.
2. `uv pip install anthropic`.
3. Same prompt/output infrastructure as (b) (reuse).
4. Cost cap (set $/run budget; the trial is bounded).

Estimated: 1 day on top of (b)'s infrastructure.

### Strategy (d-ab / d-ac) — hybrid

Define the integration pattern explicitly:

```
deterministic_output = run_a(engine, version)   # produces schema + invariants
llm_review = llm_prompt(deterministic_output, source)
  → "what's missing? what's wrong? what's nuanced?"
  → returns: extensions, corrections, risk-flags
hybrid_output = merge(deterministic_output, llm_review)
```

Sub-questions:
- Does LLM see the full deterministic output, or just the invariant list (token-bound)?
- Reconciliation rules: if LLM contradicts deterministic, who wins? (Initial answer: surface as conflict for human review; don't auto-resolve.)
- Output shape: extend `invariants.proposed.yaml` with `added_by: llm_verifier` and `flagged_for_review: true` on conflicts.

Estimated: 2-3 days on top of (b) + (c).

---

## Scoring rubric

Each cell produces (schema, invariants) candidates. Scored against a **reference** for that (engine, version) pair.

### Reference construction

For each (engine, version) pair, the reference is:
- **Schema**: union of all strategy outputs + manual review to remove spurious + add overlooked items. Human-validated.
- **Invariants**: same — union → manual review → validated set.

This is bootstrapping (no oracle a priori for new versions). For active versions where mature mining exists (transformers v4_57_3), the existing `engine_versions/<e>/v*/outputs/` is the starting reference.

### Metrics per cell

| Metric | Schema | Invariants |
|---|---|---|
| Recall (% of reference present) | ✓ | ✓ |
| Precision (% of output that's in reference; complement = spurious rate) | ✓ | ✓ |
| Type accuracy (% of overlapping fields with matching type spec) | ✓ | N/A |
| Severity accuracy (% with matching error/warn/dormant) | N/A | ✓ |
| Wall-clock per run | ✓ | ✓ |
| Energy per run (Wh) | ✓ | ✓ |
| Marginal cost per version bump (additional engineering needed) | ✓ | ✓ |

### Aggregated decision criteria

Per-strategy summary:
- **Average recall × precision** across all (engine, version) cells.
- **Robustness to version bumps**: variance in recall between minor and major bumps.
- **Per-bump engineering cost**: incremental LoC needed per bump (strategy a) vs incremental LLM cost (strategies b/c/d).
- **Silent vs noisy failure modes**: % of failures detectable by the strategy's own machinery.

---

## Curation pipeline design

The user's framing:

> Final result written to `engine_versions/` where we'd have human curation (curated.yaml, overlay.yaml) → injected into `src/` once maintainer reviewed.

This is the consumption side. The trial outputs candidate `schema.discovered.json` + `invariants.proposed.yaml` per cell. The curation layer is:

```
Strategy output(s) per (engine, version)
       |
       v
Reconciliation step: merge multi-strategy outputs into a single proposed.yaml
       |
       v
Human review: examine proposed entries; promote to validated.yaml or reject
       |
       v
engine_versions/<e>/v<safe>/outputs/{schema.discovered.json, curated.yaml, invariants.validated.yaml, overlay.yaml}
       |
       v
make sync / regen scripts: copy to src/llenergymeasure/engines/<e>/
       |
       v
Codegen produces engines/<e>/config.py
```

What's NEW vs today's pipeline:
- **Reconciliation step**: merges (a) deterministic + (b/c) LLM + (d) hybrid outputs into a single candidate. Probably a sub-script in `scripts/engine_producers/`.
- **Conflict surfacing**: when strategies disagree, the merged output annotates `x-conflict: [strategies]` and the human reviewer decides.
- **`overlay.yaml` survives** as the human-curation channel (already does this); the only change is it now sits ON TOP of multi-strategy candidate, not just `schema.discovered.json`.

The codegen + sync side stays as-is.

---

## Phasing

### Phase 1: Foundation (week 1)

Goal: get to a fair starting line for the trial.

- **Day 1**: bring vllm static invariant miner to ~20-30 invariants (audit + extend). Reuse existing detector classes. Output: updated vllm v0_7_3 invariants.proposed.yaml.
- **Day 2**: same for tensorrt (target 15-25 invariants).
- **Day 3**: probe + decide on minor/major versions to test. Set up isolated venvs per (engine, version). Document in `research/mining-substrate-trial/findings/empirical_trial_setup.md`.
- **Day 4**: build the reference set for each (engine, version) cell. For active versions: existing outputs are the seed reference. For bumps: run strategy (a) and manually verify the output before locking the reference.
- **Day 5**: tooling — write `research/mining-substrate-trial/scripts/trial_runner.py` that orchestrates per-cell execution and scoring. Stub all strategies; real implementations come in Phases 2-4.

### Phase 2: LLM extraction infrastructure (week 1-2)

Goal: build the (b) / (c) infrastructure once; reuse it.

- **Day 6**: chunking strategy + per-chunk prompt design. Test on transformers v4_57_3 schema.
- **Day 7**: structured-output validation + retry + multi-pass refinement loop. Test on same.
- **Day 8**: invariants extraction prompt (which is harder than schema). Test on same. Target: lift Bake-off B's 50% recall to 75%+.
- **Day 9**: Claude API setup (assumes user provides key). Plug into the same infrastructure.
- **Day 10**: hybrid strategy (d) design + implementation. Reuse single-strategy plumbing.

### Phase 3: Run the matrix (week 2-3)

Goal: produce data for the 24-36 cells.

- **Day 11-15**: per-cell execution. Order by ease: start with transformers (mature reference) then vllm then tensorrt. Within engine: active version first (baseline confidence) then bumps.
- Each cell run produces: candidate outputs + scoring report + cost log.
- Cells can run in parallel where independent (different engines / strategies don't compete for ollama or API rate).

### Phase 4: Synthesis (week 3)

Goal: turn the data into a decision.

- **Day 16-17**: aggregate scores across cells; build the decision matrix; identify which strategy (or combination) is recommended.
- **Day 18**: write the decision doc as `research/mining-substrate-trial/findings/empirical_trial_outcome.md`. Update the design doc Open Question 11 status.

### Phase 5: Curation pipeline (week 3-4)

Goal: operationalise the winning strategy through the curation → src injection flow.

- **Day 19-20**: build the reconciliation script; write the maintainer-review interface (CLI or markdown-diff per cell).
- **Day 21-22**: dogfood on transformers; full cycle from extraction to src codegen.

**Total: ~4-5 weeks of focused work.**

---

## Risks + open questions

1. **Reference-set bootstrapping**: for engines + versions we've never mined, there's no oracle. Mitigation: human-validated reference from union-of-strategies (Day 4). Risk: human time is the rate limit.
2. **LLM cost ceiling for (c)**: 36 cells × multiple calls per cell × API tokens = bounded but real. Estimate at design time; set per-cell budget.
3. **Ollama context limits**: 128k context for llama3.1:70b is high but vllm engine source is HUGE. May force more aggressive chunking. Worth probing early in Phase 2.
4. **Dynamic miner extension to vllm/tensorrt** (Phase 1 sub-decision): may be too costly for the trial. Mitigation: include "no dynamic miner" as a deliberate strategy variant rather than padding Phase 1.
5. **Major version bumps may not have stable releases yet** for all engines. Mitigation: probe early; downgrade to "latest pre-release" if needed; document.
6. **The trial result might be inconclusive**: e.g. all four strategies hit similar quality and the choice becomes about non-quality factors (cost, vendor risk, maintenance ergonomics). Mitigation: anticipate this; design synthesis to be specific even in the inconclusive case.
7. **What if (b) plateaus far below (c)?** Decision then becomes vendor-dependency cost vs quality lift. Worth doing the cost estimation in Phase 4 even when quality is decisive.

---

## What I'm NOT proposing

- **Productionising any of this**. The trial outcome is data for the decision; productionisation is downstream.
- **Replacing the existing codegen / regen scripts**. Those stay unchanged. The trial only affects what feeds into the mining outputs.
- **A revised CI architecture**. CI changes follow the decision, not the trial.

---

## Next concrete action

If the user approves this roadmap, the immediate next step is Phase 1, Day 1: extend vllm static invariant miner. The work is well-scoped (read `vllm/engine/arg_utils.py::EngineArgs.__post_init__` and similar; extend the AST walker to emit invariants for the validation logic). Output is a measurable lift in vllm's `invariants.proposed.yaml` from 10 → ~25-30 entries.

Alternative starting points if the roadmap is too ambitious:
- **Scope-reduce to 1 engine, 2 versions, 4 strategies = 8 cells**. Run only transformers; defer vllm/tensorrt expansion. Cheaper, still answers the substrate question.
- **Scope-reduce to schema-only** (no invariants). Halves the work per cell; loses the most-decision-relevant dimension though (invariants is where the LoC gap matters).
- **Scope-reduce to 1 strategy at a time** (sequence rather than parallelise). Slower wall-clock but cleaner learnings per phase.
