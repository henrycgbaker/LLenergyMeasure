# Trial epistemic framing — maximal information gathering, not winner picking

**Status:** Authoritative framing for the mining-substrate empirical trial.
**Authored:** 2026-05-25.
**Cross-ref:** `.planning/mining-substrate-empirical-trial.md` (the execution plan this framing governs), `research/mining-substrate-trial/findings/mining_strategy_bakeoff.md` (preliminary single-engine bake-off this framing supersedes).

This doc settles a question that's easy to get wrong: **what is the trial actually for?**

---

## The framing in one sentence

The trial's purpose is to **iteratively work through every (engine, version, strategy) cell to gather as much information as possible across all 4 strategy options (a / b / c / d). Strategy gets constructed AFTER, from the assembled evidence — not chosen mid-trial.**

---

## What this changes from naive "pick a winner" thinking

| Naive framing ("pick a winner") | Correct framing ("maximal information") |
|---|---|
| If strategy X looks bad on early cells, deprioritise its remaining cells | Run every cell anyway; "looks bad on transformers" might be "great on tensorrt" or might reveal a recoverable failure mode |
| Optimise per-cell prompt to maximise that cell's score | Use the SAME prompt across cells (within a strategy); per-cell tuning conflates strategy-quality with prompt-engineering effort |
| Drop the worst-performing strategy mid-trial to save cost | Complete the matrix; the worst performer's failure modes are decision-relevant evidence |
| Synthesis = "which strategy won?" | Synthesis = "what does the data tell us about each strategy's strengths, weaknesses, failure modes, version-robustness profile, cost profile, and how they combine?" |
| Single recommendation: "use substrate X" | Multi-dimensional strategy: "use substrate X for engines like Y, substrate Z for engines like W, hybrid Q for the version-bump-stability problem, …" |
| Early-exit when "obvious" winner emerges | Resist the urge — early signal can mislead; complete-the-matrix discipline is what makes the synthesis grounded |

The bake-off cycle that preceded this trial (Bake-offs A, B, D) was deliberately one-engine, one-version single-shot — designed to be cheap and exploratory. The TRIAL is the systematic version: cheap exploration is done; rigour comes from completion.

---

## Why maximal information matters here

The substrate decision isn't "pick the best for the average case." It's strategic and likely multi-faceted:

1. **The right substrate may differ per engine.** transformers (mature Python class hierarchy with Sphinx docstrings, kwargs-popping patterns) is a different extraction problem than vllm (msgspec-typed dataclasses with sprawling EngineArgs) or tensorrt (Pydantic with NVIDIA-specific runtime gates). A strategy that wins on one may not generalise.

2. **The right substrate may differ per knowledge category.** Schema discovery is mechanical (walk fields → emit types); invariant mining is semantic (analyse validator logic → emit predicates). LLMs may shine on one and not the other. Deterministic mining has different LoC costs across the two.

3. **The right substrate may differ per maintenance horizon.** A substrate that's high-quality but high per-bump cost (e.g. requires per-version prompt iteration) may lose to a slightly-lower-quality substrate that's nearly-free per bump. The version-bump cells are how we measure this.

4. **Hybrid strategies' value depends on what each pure strategy misses.** The case for (d-ab / d-ac) hybrid is "LLM catches what deterministic misses." We can only quantify that gap by completing both pure-strategy cells AND comparing. Skipping pure-strategy cells of a "looks-bad" substrate forecloses the hybrid analysis.

5. **Negative findings are positive information.** "Strategy (b) catastrophically fails on tensorrt invariants because Ollama can't handle the 80k-token engine source even chunked" is a real, decision-relevant fact. It's not a reason to skip those cells — it's WHY we run them.

---

## What "maximal" actually means dimensionally

The trial gathers information along these axes for each cell:

1. **Coverage**: recall (% of reference items the strategy surfaced).
2. **Precision**: spurious rate (% of strategy output not in reference).
3. **Type/severity fidelity**: on overlapping items, how accurately does the strategy classify them?
4. **Wall-clock per cell**: end-to-end time.
5. **Token + energy cost per cell**: for the LLM strategies, real cost; for deterministic, electricity proxy.
6. **Failure-mode profile**: silent miss vs noisy break vs hallucinated entry. Different failure types have different operational consequences.
7. **Sensitivity to setup**: for LLM strategies, how much does prompt iteration / chunking / model variant affect the result?
8. **Engineering cost per future bump**: incremental LoC for deterministic; incremental prompt / verification work for LLM.
9. **Adjacent observations**: things noticed in passing that aren't in the scoring rubric but might matter (e.g. "LLM consistently hallucinates `*_v2` field names for version-bumped engines — interesting failure mode worth studying").

For each cell we want all 9 dimensions captured. The synthesis then has a 9-dimensional × 5-strategy × 9-cell-pair landscape to construct the strategy from.

---

## What this means for execution

### Complete the matrix

All ~45 cells run. No early-exit. No "we know enough now."

If cost / time forces a triage decision, the right triage is to drop a (engine, version) pair (i.e. a column of 5 cells), NOT a strategy (i.e. a row of 9 cells). Strategy-completeness is the trial's central asset; dimension-completeness within a kept cell is the second.

### Hold per-strategy prompts constant across cells

Within a strategy, the prompt / chunking / orchestration setup is FIXED after Phase 2 calibration. Don't tune per-cell — that conflates strategy-quality with prompt-engineering effort. If a strategy underperforms on one cell, that's a property of the strategy at that cell, not a signal to re-tune.

The exception: Phase 2's calibration phase IS prompt iteration, but on transformers only (the cell with strongest ground truth). Once Phase 2 closes, prompts lock.

### Capture failure modes explicitly

For every cell, the report includes a `failure_modes` field — a structured note on HOW the strategy failed where it failed. "Silent miss of compile_config field" is more useful than "12 fields missed." The synthesis can cluster failure modes and reason about whether they're fixable, fundamental, or strategy-specific.

### Adjacent observations get a place to land

Each cell run produces a structured score + a free-text "observations" section. Things like "Ollama at 32k context returns truncated JSON intermittently — added retry-on-truncation handling, may not be needed for Claude" go here. Surprise findings accumulate into the synthesis as side-evidence.

### Synthesis is multi-axis, not a leaderboard

The Phase 4 outcome doc (`research/mining-substrate-trial/findings/empirical_trial_outcome.md`) is structured as:

1. **The information map** — what we now know about each strategy's strength/weakness profile across the 9 dimensions × 9 cells.
2. **The decision space** — what realistic strategies one COULD construct from this map (3-5 viable architectures, each defended against the alternatives).
3. **The recommended strategy** — which one llem should adopt, why, and what trade-offs we're accepting.

Step 1 is the information artefact. Step 2 reasons about combinations. Step 3 is the actual decision. Each step is grounded in the prior.

---

## Pure vs hybrid: two different execution disciplines

The trial has TWO modes of work running concurrently, with DIFFERENT discipline rules:

### Pure strategies (a, b, c) — matrix discipline

The pure strategies establish baselines. For these:

- **Complete the matrix.** Every (engine, version) cell × strategy gets run.
- **Hold prompts/setup constant across cells.** Once Phase 2 closes the LLM-side setup, don't tune per-cell.
- **No early-exit.** Even if patterns look obvious.
- **Capture failure modes + observations** alongside scores.

Pure-strategy cells produce the clean comparison data. Their value depends on completeness and consistency.

### Hybrid space (d) — exploratory discipline

The hybrid space is **the heart of the PoC**. Not a single fixed pattern to score, but a research subject in its own right: how / whether / when does combining deterministic mining with agentic LLM calls produce better outcomes than either alone?

Variants worth exploring (illustrative, not exhaustive):

- **Deterministic → LLM validates**: LLM reads (a)'s output + source, flags suspect entries (likely wrong / outdated / spurious).
- **Deterministic → LLM extends**: LLM reads (a)'s output + source, proposes additions (what (a) missed).
- **Deterministic → LLM diagnoses**: LLM reads (a)'s output + the source that produced it, explains WHY (a) missed what it missed (e.g. "the AST walker didn't recurse into this @cached_property").
- **LLM proposes → deterministic verifies**: LLM reads source, proposes entries; runtime verification (probing) confirms each.
- **Multi-pass LLM agent**: extract → verify against source → extend → reconcile, with intermediate "is this real?" checks.
- **LLM-as-orchestrator**: LLM decides which deterministic miner to run on which validator method, refines targets, re-runs with adjusted scope.
- **LLM-as-curator**: LLM reads everything (deterministic output + source + adjacent docs) and produces the final curated artefact human reviewers would sign off on.

For hybrid work:

- **Experiment freely.** Spawn subagents (sonnet for mechanical variants, opus/general-purpose for novel patterns). Try a pattern; log what happened; iterate.
- **Log everything.** Each hybrid experiment goes into `research/mining-substrate-trial/findings/hybrid_experiments/` with: pattern description, prompt(s), output, score, observations, next-iteration ideas.
- **No "right answer" pre-committed.** The point is to discover what works, what doesn't, and what the failure modes look like. Negative findings ("LLM-as-orchestrator hallucinated method names that didn't exist, broke pipeline") are valuable.
- **Iterate on what's interesting.** If a hybrid pattern shows promise on transformers, try it on vllm. If it fails interestingly, document the failure and try a variant.
- **Keep going.** The session's value scales with the diversity of explored patterns × engines × versions.

The synthesis (Phase 4) gets to look at BOTH: the clean baseline matrix from pure strategies AND the explored hybrid landscape. Strategy gets constructed from BOTH evidence streams.

## What this means for the agent executing the trial

Operationally:

- **For pure strategies**: complete the matrix, hold prompts constant, no early-exit. Per the discipline above.
- **For hybrid work**: experiment, spawn subagents, log everything, iterate based on what's interesting. Per the discipline above.
- **Capture observations beyond the rubric.** Each cell run AND each hybrid experiment includes free-text findings that don't fit a metric.
- **Triage by column, not row.** If you have to skip pure-strategy cells, drop a (engine, version) pair, not a strategy.
- **Don't write the synthesis until Phase 4.** Per-cell reports + hybrid-experiment notes are facts; synthesis is interpretation. Don't conflate.
- **Resist "decisive early signal".** A strategy that wins on the first 3 cells may be hitting easy cases. Complete the matrix before drawing conclusions.
- **The hybrid exploration is open-ended.** There's no fixed "done" for hybrid; "done enough" is when the diversity of explored patterns produces synthesis-ready signal about which hybrid shapes (if any) outperform pure strategies. Probably ~5-10 distinct hybrid patterns × 2-3 (engine, version) pairs each.

The execution discipline IS the rigour. The trial's value is destroyed by mid-trial optimisation toward what looks promising on pure strategies. The trial's value is AMPLIFIED by diverse exploration in the hybrid space.

---

## Why we're writing this down

The temptation to early-exit, prompt-tune mid-trial, or pick a winner from partial data is real and constant. A fresh-context agent picking up the plan might naturally optimise toward "be efficient" or "show clear progress" — both of which corrode the trial's information value if applied mid-execution.

This doc is the explicit counterweight: efficiency / progress is COMPLETING the matrix, not concluding early. Document it loudly because the wrong execution discipline silently invalidates the entire trial.

---

## Reference for the agent

The handoff prompt that initiates the trial should reference this doc. The plan (`mining-substrate-empirical-trial.md`) cross-references it. The Phase 4 synthesis structure follows from it.

When in doubt: **run the cell, capture the data, hold the prompts, defer the interpretation.**
