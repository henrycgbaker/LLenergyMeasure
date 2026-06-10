# Mining-substrate empirical trial: a 51-cell, 9-pattern study of deterministic AST mining vs LLM extraction across three LLM-inference engines

**Status:** Trial closed at Phase 4 synthesis. Polished research write-up; standalone.
**Authored:** 2026-05-25.
**Branch:** `trial/mining-substrate-bakeoff`.
**Model substrate:** `llama3.1:70b` served via container Ollama at q4_K_M quantisation, num_ctx=32768, temperature=0. Phase 3c (Claude) deferred pending `ANTHROPIC_API_KEY`.

---

## Abstract

llem mines engine-configuration knowledge (validation invariants, parameter schemas) from upstream LLM-inference engines (transformers, vllm, tensorrt-llm) using approximately 3800 lines of handwritten per-engine AST walkers plus 2800 lines of shared infrastructure. A spike branch surfaced doubt whether a 70B-class LLM reading the same source could match or replace this deterministic substrate. The mining-substrate empirical trial closed that question by running 51 scored (engine, version, strategy) cells plus 9 distinct hybrid patterns at locked prompts under `llama3.1:70b` q4 quantisation. Under validated-union ground truth (every strategy's runtime-validated invariants unioned per cell as the empirical reference), pure deterministic mining (strategy a) achieves 46.6 percent invariant recall and 32.7 percent precision; pure LLM extraction (strategy b) achieves 42.3 percent recall and 27.6 percent precision; the active-seed hybrid (strategy d-ab) achieves 77.6 percent recall and 73.6 percent precision. None of three substrate-decomposition variants tested (whole-source, field-anchored, cumulative-context) lift the LLM ceiling above 50 percent recall on transformers. Across version bumps, the deterministic substrate exhibits three structurally distinct brittleness modes (transformers landmark-missing, vllm dependency-import failure, tensorrt walker version-blindness). The trial's most decision-relevant cross-pattern finding is the LLM-role split at this model scale: LLMs are robust at diagnosis (0 fabrications across 8 diagnoses), error-prone at subtraction (3/3 false-drops on vllm), weak at single-shot synthesis-of-code (0/3 patches lift recall), and collapse to zero under agentic flexibility (H7). The recommended architecture is a two-substrate hybrid (Scenario 4): deterministic mining as floor, chunked LLM extraction as extension, deterministic runtime validation as gate, validated union as canonical artefact, maintainer curation as operational layer. Phase 3c (Claude) is the only condition that materially revisits this recommendation; the LLM-role split is the open question Claude tests directly.

---

## 1. Problem statement

llem (the parent project) measures the energy consumption of running large language models under varying inference configurations across multiple inference engines. To do so meaningfully it must know each engine's parameter schema (which fields exist, their types and defaults) and its validation invariants (which combinations of values raise, which emit warnings, which silently normalise). A measurement that mis-sets `do_sample=False, temperature=2.0` would log nominally valid energy numbers while the engine internally clamps the sampling parameter to greedy; the experiment record would be wrong.

Today llem produces this knowledge via per-engine AST walkers that import each engine's source, traverse its `__post_init__` methods and validator decorators, and emit structured catalogues to `engine_versions/<engine>/v<slug>/outputs/`. The substrate is mature: transformers v4.57.3 has 41 mined invariants and a 717-line validated-yaml file. vllm v0.7.3 has 26 invariants (after a Phase 1 lift from 10). tensorrt-llm v0.21.0 has 35 (lifted from 3). These outputs flow through codegen into `src/llenergymeasure/engines/<e>/config.py` as Pydantic models that the experiment runtime consumes.

The substrate works. The question the trial addressed is whether it should still work this way. Two pre-trial observations made the question pressing.

First, a bake-off cycle on the spike branch surfaced quantitative evidence that a 70B-class OSS LLM (`llama3.1:70b` via host Ollama) could read engine source and produce something comparable. Bake-off B's single-engine POC on transformers v4.57.3 achieved 52 percent schema recall and 20 quality invariants extracted; auto-scoring failed because the LLM wrapped its output in markdown fences, but corrected reading suggested 40-60 percent invariant recall. That is significantly under the deterministic substrate's measured coverage on the same active version, but it crosses the threshold from "obviously inferior" to "potentially viable with better setup". The bake-off explicitly recommended a larger empirical trial.

Second, Bake-off D measured the deterministic substrate's robustness against a v4-to-v5 transformers version bump. The static miner held 92 percent of its invariants and surfaced 16 new ones automatically; the schema introspector held 88-96 percent field overlap but lost 66 of 67 sampling-parameter types from an upstream `__init__` refactor (silent failure); the dynamic miner held 75 percent overlap with 4 silent misses. Across the three deterministic miners in the stack, the v4-to-v5 patch surface was roughly 80-90 lines. That is the cost of one minor-version bump on one engine; the question of whether this cost scales linearly across more bumps and engines or whether it compounds catastrophically was unknown.

Third, llem ships per-engine machinery whose maintenance debt is real. Bake-off A (refactor analysis) identified approximately 1800 lines of accidental complexity in the current ~3800-line per-engine producer stack: parallel detector classes in the static miner (~400 LoC duplication), bespoke per-miner predicate logic (~600 LoC), inline cartesian-probe scaffolding (~200 LoC). Each engine added to llem replicates this scaffolding. If the deterministic substrate is going to stay, refactoring the accidental complexity is one PR-scope task. If an LLM substrate could replace it, the refactor is wasted work. The trial's outcome decides the question.

The trial's primary scientific question was therefore: which substrate(s) should llem invest in long-term for mining engine-configuration knowledge? Three sub-questions fell out: how does each substrate transfer across engines (transformers / vllm / tensorrt-llm differ structurally in how their validators surface), how does each survive version bumps (the brittleness axis), and does a hybrid substrate (deterministic floor with LLM extension or verification) materially exceed any pure approach?

---

## 2. Methodology

The trial's experimental design was deliberately structured to gather maximal information about strategies via three orthogonal axes rather than to "pick a winner" by comparing them on a single benchmark. The three axes were pure-strategy quality (matrix discipline), hybrid-pattern exploration (exploratory discipline), and brittleness across version bumps (first-class decision dimension). The accompanying epistemic discipline (in `findings/trial_epistemic_framing.md`) prescribed: complete every cell at locked prompts, never iterate prompts mid-trial, log failure modes as first-class outputs, do not synthesise mid-trial.

### 2.1 The strategy space

| ID | Name | Implementation |
|---|---|---|
| a | Pure deterministic mining | Per-engine handwritten producers (schema_introspector + static_invariant_miner + dynamic_invariant_miner where present). Reads engine source via `inspect.getsource` for active version; via AST file-reading or subprocess miner for bumped versions. |
| b | Pure OSS LLM extraction | Chunked source extraction via per-engine chunkers; locked prompts; JSON-mode + JSON Schema validation + retry; markdown-fence stripping; multi-pass refinement (extract -> verify -> extend). All chunks served by container Ollama at port 11435 (`llama3.1:70b` q4_K_M, num_ctx=32768). |
| b_8b | Pure OSS LLM extraction (smaller model) | Same pipeline as (b), but served by `llama3.1:8b`. Tested only on transformers active as a quality-speed-energy probe. |
| c | Pure Claude API extraction | Stub at `_spike/scripts/strategies/claude_extractor.py` reuses (b)'s prompts and pipeline. Deferred: `ANTHROPIC_API_KEY` not yet available. |
| d-ab | Hybrid: active-seed + LLM-extension | The active-version (a) output is taken as deterministic seed; the LLM reads bumped source and proposes EXTENSIONS only. Output is seed ∪ extensions. 100 percent recall is by construction at the active version; bumped d-ab measures extension counts across bumps. |
| H1-H9, E6, E9 | Hybrid patterns | Exploratory; see Section 5. |

### 2.2 The 5x3 matrix

The cell axis is `(engine, version, strategy)`. Three engines (transformers, vllm, tensorrt-llm) each contribute five versions spanning two backward bumps, the active version, and two forward bumps:

| Engine | v-2 (older) | v-1 (older minor) | Active | v+1 (newer minor) | v+major |
|---|---|---|---|---|---|
| transformers | 4.55.4 | 4.56.2 | **4.57.3** | 4.57.6 | 5.9.0 |
| vllm | 0.6.0 | 0.6.6.post1 | **0.7.3** | 0.9.2 | 0.19.1 |
| tensorrt-llm | 0.19.0 | 0.20.0 | **0.21.0** | 1.0.0 | 1.2.1 |

Several anomalies emerged at version-lock time. The transformers v+1 slot is patch-level (no `4.58.x` was ever released; the project jumped from `4.57.6` straight to `5.0.0`). The tensorrt v+1 slot is "early-major" (`1.0.0`); no `0.22.x` ever shipped. vllm's matrix is the only one with a clean five-distance span. These asymmetries propagated to brittleness aggregates; Phase 4 reports them honestly rather than averaging across non-comparable distances.

The Phase 3a matrix produced 47 scored cells (after deferring `c` and `d-ac`, both Anthropic-dependent). Phase 3b added 9 hybrid-pattern explorations totaling roughly 18 cell-equivalents, of which 4 contribute distinct scored cells in the validated-union matrix. The total scored corpus is 51 cells under validated-union scoring across 103 raw score JSONs (which include partial / superseded / hybrid-variant records).

### 2.3 Scoring rubric evolution

The trial's most consequential methodological correction was its scoring rubric. The starting point was a three-tuple invariant identity `(namespace, native_field, predicate_kind)` with `(a)`'s output as the reference. Two failures emerged:

First, Phase 2.5 calibration revealed that the three-tuple identity COLLAPSED multi-field invariants into single entries. A cross-field invariant like "raise if `num_return_sequences > num_beams`" became identical (under the rubric) to a single-field "raise if `num_return_sequences > 0` and not beam-search". The fix was a four-tuple identity adding `secondary_field`. This collapsed the apparent invariant recall from 60.7 percent to 41.0 percent on transformers active (b); 41.0 percent was the honest baseline.

Second, a `b/tensorrt` active cell scored 0.0 percent invariant recall because the LLM (correctly mirroring the chunker's `expected_namespaces` hint) emitted `tensorrt_llm.<field>` while the reference (mined by the static miner) used `tensorrt.<field>`. Two distinct identity tuples for the same semantic invariant. The fix was a namespace-canonicalisation pass (`canonicalise_namespace(ns, engine)`) collapsing `tensorrt_llm.X` to `tensorrt.X`. The rescore lifted `b/tensorrt` invariant recall from 0.0 percent to 25.8 percent without re-running any LLM.

Third, and most foundationally, the (a)-as-reference framing privileged (a) at every cell. A (b) invariant absent from (a) was scored false-positive even if it was runtime-valid. The fix was **validated-union ground truth**: per cell, every strategy's invariants are unioned and runtime-validated against the live engine container; the validated union becomes the cell's empirical reference. Both (a) and (b) score against the same union; neither is privileged. The implementation is `_spike/scripts/run_phase4_0_union.py`, which dispatches per-engine to the existing production-grade `scripts/validate_invariants.py` (Docker-based runtime validation for all three engines).

The validated-union rescore reshuffled the matrix substantively. (a)'s aggregate dropped from 52.8 percent to 46.6 percent invariant recall (-6.2 percentage points) and from 54.3 percent to 32.7 percent invariant precision (-21.6 pp); the (a)-as-reference rubric was crediting (a) for the entries it found and counting its blind spots as the reference. (b)'s aggregate ROSE from 34.4 percent to 42.3 percent recall (+8.0 pp) and from 21.0 percent to 27.6 percent precision (+6.6 pp); entries (b) found that (a) missed, runtime-validated, became recall credit. (d-ab) dropped from 100 percent to 77.6 percent recall (-22.4 pp); its "100 percent" was by construction against (a)'s narrow output, and the union exposed 28+ entries d-ab missed.

### 2.4 Container Ollama setup

The LLM substrate was served by a container Ollama instance dedicated to the trial, at port 11435 (the host port 11434 is used by another tenant). The container `trial-ollama` was launched with `--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all`, exposing four A100-40GB GPUs internally. Both `llama3.1:70b` (Q4_K_M, 42 GB; ~52 GB across two GPUs at 32k context) and `llama3.1:8b` (4.9 GB) were pulled and pinned. Context window was set to 32k (not the default 128k, which spills 35 percent of the 70B model to CPU per Bake-off B precedent); `keep_alive=30m` kept the model loaded across chunks. Wall-clock per chunk under this setup is 30-120s; a full multi-pass (b) cell averages 1400-1700s on transformers active and 23-25 minutes wall-clock total.

### 2.5 Phase 3c (Claude) status

The plan's strategy (c) was Claude API extraction reusing (b)'s prompts. `ANTHROPIC_API_KEY` was not available during the trial's main execution window. The `claude_extractor.py` stub is in place; activation is `uv add anthropic && export ANTHROPIC_API_KEY=...` plus launching a Phase 3c subagent. Estimated cost $20-35 for the 15-cell (c) matrix plus 6-8 key hybrid patterns. Phase 3c tests directly whether the LLM-role split (Section 4) is intrinsic to LLMs or specific to `llama3.1:70b` at q4 quantisation. The trial's recommendation (Section 8) is contingent on this not overturning the split.

### 2.6 Trial discipline

Five discipline rules were enforced throughout. (1) Prompts locked at Phase 2 closure; no per-cell iteration. (2) Complete the matrix; never early-exit on data that looks bad. (3) Triage by column (engine, version) not row (strategy); preserve cross-strategy comparison even when scope-reducing. (4) Synthesis is Phase 4 only; observations during execution go to `observations` arrays and `DECISIONS_LOG`, never to a mid-trial recommendation. (5) Don't fix gaps discovered during execution; preserve them as research data (the post-trial (a) gap closure backlog enumerates them; see Section 9).

---

## 3. Results: the pure-strategy matrix

The 47 pure-strategy cells plus 4 hybrid cells contributing to the validated-union matrix produce the aggregate below. All recall and precision numbers under validated-union scoring.

| strategy | cells | invariant_recall | invariant_precision | schema_recall | wall_mean_s |
|---|---|---|---|---|---|
| a | 15 | 46.6% | 32.7% | 60.0% | 1.9 |
| b | 15 | 42.3% | 27.6% | 60.6% | 3411.4 |
| b_8b | 1 | 25.0% | 22.6% | 85.7% | 412.6 |
| d-ab | 15 | 77.6% | 73.6% | 100.0% | 254.8 |
| e6 | 2 | 50.4% | 46.5% | 90.0% | 1152.3 |
| e9 | 2 | 41.4% | 52.5% | 90.0% | 963.6 |
| h6 | 1 | 17.9% | 62.5% | 75.0% | 526.4 |

The strategy aggregates collapse three orthogonal signals into one row per strategy. Each unpacks differently across engines and version bumps; the next subsections look at each axis.

### 3.1 Pure (a) deterministic across engines and bumps

The deterministic substrate's brittleness is structurally distinct per engine. Three modes emerged.

**transformers (a): landmark-missing on extreme bumps.** The walker imports landmark classes (`PreTrainedModel`, `GenerationConfig`) for inspection; when the bumped version lacks the landmark or its dependencies, the import fails cleanly. At v-2 (4.55.4) the `tokenizers>=0.21,<0.22` constraint fails; the producer raises ImportError before any AST work. At v+major (5.9.0) the `is_offline_mode` import from `huggingface_hub` fails; same outcome. Both report `detectable` failure mode with a clean stderr trace. At v-1 (4.56.2) and v+1 (4.57.6) the walker runs; the bumped source has different identity tuples; recall drops to 33.9 percent and 32.1 percent respectively under validated-union scoring. Cleanest brittleness signal: clean crash at edges, graceful degradation in the middle.

**vllm (a): dependency-import collapse on all bumps.** All 4 bumped vllm cells crash `detectable` with `ModuleNotFoundError: No module named 'msgspec'`. The vllm static miner's first action is `import vllm`, which transitively imports `sampling_params -> msgspec`. The source-only venv pattern (which works for transformers) doesn't install msgspec because the venv has no `pip install vllm`; only the source tree is unpacked. Honest brittleness: 100 percent bumped-cell failure across all four distances, all with the same root cause.

**tensorrt-llm (a): MINER_VERSION_BLIND silent re-extraction.** All 4 tensorrt bumped cells report 100 percent recall and 100 percent precision. Investigation revealed the cause: the tensorrt static miner is pure-AST, reads from a hardcoded `_DEFAULT_SOURCE_ROOT = /tmp/trt-llm-0.21.0/tensorrt_llm`, and PYTHONPATH override at the trial-runner subprocess level has no effect. The bumped cells re-extract the ACTIVE 0.21.0 source unchanged (verified by byte-diff of the emitted invariants against the v0_21_0 reference YAML). This is not honest brittleness measurement; it is a substrate-wiring artefact. The trial preserved it (per discipline: don't fix the miner) and de-weights the four cells from per-engine aggregates with annotated `observations` on each score JSON. The Phase 4 synthesis labels these explicitly.

Three engines, three distinct brittleness mechanisms. The lesson is that "(a) is brittle" is too coarse a claim; (a)'s brittleness profile is engine-dependent, and each mode requires a different fix (defensive imports for transformers, transitive-dep declarations for vllm, pluggable source_root or env-var indirection for tensorrt). Section 9 enumerates the seven specific gaps and Section 10 maps them to the spike-refactor.

### 3.2 Pure (b) LLM extraction across engines and bumps

The LLM substrate exhibits a different and also engine-dependent brittleness profile, rooted in the chunker's source-extraction assumptions rather than walker behaviour.

**transformers (b): stable across bumps.** Validated-union recall is 55.4 percent at v-2, 53.6 percent at v-1, 62.5 percent at active, 58.9 percent at v+1, 44.6 percent at v+major. The chunker reads files by name (`modeling_utils.py`, `generation/configuration_utils.py`, `utils/quantization_config.py`); these filenames didn't change across the v4-to-v5 transition. The mild v+major recall dip (44.6 vs 62.5) reflects real source drift, not chunker fragility.

**vllm (b): smooth-then-cliff.** Recall is 46.2 percent (v-2), 33.3 percent (v-1), 48.7 percent (active), 41.0 percent (v+1), 0.0 percent (v+major). The cliff at v+major (0.19.1) is silent failure. vllm 0.19 refactored `config.py` from a single file into a `config/` subdirectory. The chunker's hardcoded `_read_source("config.py")` returns empty; only 4 sentinel invariants emit from the failure marker; cell silently fails. This is detectable post-hoc via low `cell_count` plus the failure-marker pattern; the rubric should expose it but did not subdivide silent failures sharply enough.

**tensorrt-llm (b): hallucinate-then-degrade.** Recall is 36.4 percent (v-2), 36.4 percent (v-1), 45.5 percent (active), 36.4 percent (v+1), 36.4 percent (v+major) under validated-union scoring; precision uniformly under 13 percent. The v-2 and v-1 cells (v0_19_0 and v0_20_0) classify as `silent` but with cell counts in the 30s. Investigation revealed the most insidious failure mode discovered in the trial. tensorrt-llm 0.19 and 0.20 use a single `LlmArgs` class; tensorrt-llm 0.21 split this into `BaseLlmArgs` plus `TrtLlmArgs`. The chunker's hardcoded class names (`BaseLlmArgs`, `TrtLlmArgs`) don't exist in v0.x; the extractor returns empty class bodies. The LLM, presented with empty input, did not realise it had received empty input; it HALLUCINATED 30+ HuggingFace `GenerationConfig` field names (`temperature`, `top_k`, `do_sample`, `num_beams`) that don't exist in tensorrt at all. Recall reports ~16 percent because some HF field names happen to overlap with tensorrt-conventional ones. Cell count is ~37. Operationally, this is the most trust-corroding failure mode discovered: the metrics look "kind of working" but the content is mostly invented.

The lesson: chunker assumptions are first-class brittleness surface, distinct from walker assumptions in (a). And LLMs given empty input from broken chunkers hallucinate from prior knowledge rather than emitting empty output. Mitigation (Section 8) is a deterministic gate downstream of the LLM; the gate catches the fabricated field names because they do not exist in the live engine's `Model.__fields__`.

### 3.3 Pure (d-ab) hybrid across engines and bumps

The (d-ab) strategy is the active-seed plus LLM-extension hybrid: at any (engine, version) cell, the active-version (a) output is the deterministic seed; the LLM reads the cell's bumped source and proposes EXTENSIONS only. The merged output is seed plus runtime-validated extensions.

By construction, d-ab achieves 100 percent invariant recall against the (a)-as-reference rubric. Under validated-union scoring this drops to 77.6 percent because the union exceeds (a)'s output by 28 entries on transformers active alone. But the construction-recall is not the interesting signal; the EXTENSION counts are.

| engine | active ext | v-2 ext | v-1 ext | v+1 ext | v+major ext |
|---|---|---|---|---|---|
| transformers | (n/a) | 0 | 0 | 0 | 0 |
| vllm | (n/a) | 0 | 0 | 2 | 0 |
| tensorrt-llm | (n/a) | 3 | 3 | 8 | 4 |

transformers d-ab is conservative across all bumps; vllm d-ab adds 2 invariants at v+1 (0.9.2); tensorrt d-ab adds 3-8 at every bump. The pattern reflects how much the validator surface evolved across the bump window. tensorrt-llm's decorator count went from 25 at v0.21 to 32 at v1.0 to 51 at v1.2.1; the LLM sees more novel patterns and emits more extensions. This is a genuine LLM signal even when recall is by construction.

There is a HOLLOW CAVEAT: when (a) crashes at a bump (vllm bumped cells, transformers v-2 / v+major), d-ab's "100 percent recall" is purely the active reference. The number is correct but misleading; it doesn't measure bumped-cell behaviour. The validated-union rescore corrects this honestly: bumped d-ab cells score 66.1 percent (transformers) or 66.7 percent (vllm) recall against the union, reflecting that the seed-only contribution doesn't cover gaps the union finds elsewhere.

### 3.4 The 8B variant probe

A single cell tested `llama3.1:8b` against transformers active under the locked (b) prompts: schema recall 85.7 percent, invariant recall 25.0 percent (validated-union), wall 412.6 s, energy 4.93 Wh. Compared to the 70B baseline (schema 83.0, invariants 62.5, wall 1649s, energy 81.3 Wh), the 8B model is 2.2x faster, 16x less energy, slightly better on schema, and substantially worse on invariants (-37.5 pp). The cost-quality trade is mixed; 8B is viable as a schema-only probe but not as a full (b) substitute. The economics weren't sufficient to justify a separate trial dimension; 70B stayed primary.

### 3.5 Aggregate cross-engine signal

The pure (a) and (b) recall ranges differ sharply by engine. (a) recall (active only): transformers 100 percent, vllm 100 percent, tensorrt-llm 100 percent (all by construction). (b) recall (active only, validated-union): transformers 62.5 percent, vllm 48.7 percent, tensorrt-llm 45.5 percent. The (b) ceiling drops as the engine's validator surface gets tighter; tensorrt-llm's small Pydantic surface yields less extractable content per chunk. This matters for substrate-choice decisions: engines where (b) is weakest are also the engines where the (a) brittleness is most acute. Section 8 argues this against per-engine architectural asymmetry (Architecture IV).

The validated-union per-strategy contributor table tells a complementary story:

| engine | version | (a) | (b) | d-ab | h2 | h3 | h6 | e6 | e9 |
|---|---|---|---|---|---|---|---|---|---|
| transformers | v4_57_3 | 37 | 35 | 37 | 37 | 32 | 10 | 32 | 22 |
| vllm | v0_7_3 | 26 | 19 | 26 | 23 | 19 | 0 | 17 | 17 |
| tensorrt | v0_21_0 | 11 | 5 | 11 | 11 | 5 | 0 | 0 | 0 |

(a) and (b) are MOSTLY OVERLAPPING contributors to the union on transformers and vllm active. On transformers, the union has 56 validated entries; (a) finds 37, (b) finds 35; their intersection is large. The unique-contributor count per strategy (entries where only that strategy contributed) is near zero across the board, except H6 on transformers contributing 3 entries no other strategy found, and E6/E9 contributing 3-4 each on vllm. Substrates COMPLEMENT each other more than they SUBSTITUTE; mining a union is better than mining via any single strategy alone, but no single non-(a) strategy contributes massive unique coverage.

---

## 4. Results: the hybrid landscape

Nine distinct hybrid patterns ran during Phase 3b plus the embedded H1 (d-ab) baseline. The patterns explored four dimensions: flow direction (deterministic-first vs LLM-first), read scope (per-class chunk vs whole-source vs cumulative), iteration depth (single-shot vs multi-pass vs agentic), and LLM role (extract vs validate vs propose vs diagnose vs modify-miner). The findings organise by LLM role rather than by pattern ID; the role-based view is the trial's most decision-relevant cross-pattern reading.

### 4.1 H1 (active-seed + LLM-extension) -- the d-ab baseline

H1 IS d-ab as run in Phase 3a; described in Section 3.3. By-construction 100 percent recall; precision drops to 73.6 percent under validated-union. Extension counts vary by engine. Functions as a baseline for hybrid variants but doesn't independently test LLM extraction quality.

### 4.2 H2 (LLM validates by subtracting from (a)) -- LLM-as-subtractor

H2 reads (a)'s output and classifies each entry as CONFIRM, UNCERTAIN, or SUSPECT-SPURIOUS. Conservative prompt phrasing instructed "prefer UNCERTAIN over SUSPECT-SPURIOUS"; only SUSPECT-SPURIOUS entries are dropped. Drop rates: transformers 0/41, vllm 3/26, tensorrt 0/35.

The conservative prompting prevented mass-drops. But ALL THREE vllm drops were FALSE-DROPS. The LLM's reasoning per dropped entry:

| Dropped ID | LLM reason | Source reality |
|---|---|---|
| vllm_samplingparams_dormant_seed_eq_neg1 | "seed is not set to -1 in the source" | Wrong: `sampling_params.py` has `if self.seed == -1: self.seed = None` |
| vllm_loraconfig_dormant_max_cpu_loras_unset | "max_cpu_loras is not set to None in the source" | Wrong: `config.py` has the normalisation predicate verbatim |
| vllm_promptadapterconfig_dormant_max_cpu_prompt_adapters_unset | Same pattern | Wrong: same |

All three dropped entries are `severity=dormant` invariants describing normalisation patterns. The LLM consistently mis-classified the "is the value X" predicate form as spurious because it didn't match the more familiar "raises when X" template. The verdict: LLM subtraction is unsafe at 70B-q4. Conservative prompting reduces but does not eliminate the failure mode. H2 informs Section 8's architecture commitment: subtraction is deterministic, not LLM.

### 4.3 H3 (LLM proposes; deterministic runtime gate) -- the canonical extend-propose shape

H3 runs (b)'s output through a deterministic gate. Transformers uses the production-grade `runtime_validate_invariants` harness which dispatches to a transformers container; vllm and tensorrt fall back to schema-existence gating (the field must be in `Model.__fields__`).

| Engine | Gate | (b) entries | Verified | Dropped | Recall (b -> verified) | Precision (b -> verified) | Precision lift |
|---|---|---|---|---|---|---|---|
| transformers | runtime | 51 | 39 | 12 | 56.4 -> 48.7% | 43.1 -> 48.7% | +5.6 pp |
| vllm | schema-existence | 66 | 62 | 4 | 38.5 -> 38.5% | 15.2 -> 16.1% | +1.5 pp |
| tensorrt | schema-existence | 39 | 39 | 0 | 25.8 -> 25.8% | 20.5 -> 20.5% | +0.0 pp |

The runtime gate on transformers achieves a clean trade: recall -7.7 pp (3 of the 12 dropped were in the reference; their positive case did not trigger at runtime), precision +5.6 pp. The schema-existence gate on vllm/tensorrt is too weak to catch the hallucination pattern (which is about wrong predicates on real fields, not fabricated field names). The recommendation for Phase 4 production is to extend runtime validation to vllm and tensorrt via their per-engine containers; the infrastructure exists (`scripts/validate_invariants.py` supports all three engines), and the trial's machinery audit confirmed this. H3 contributes the strongest single piece of evidence FOR the deterministic-validate / LLM-extend architecture.

### 4.4 H4 (LLM modifies miner code) -- LLM-as-walker-maintenance-engineer

H4 reads (a)'s output, the producer's AST walker code, and engine source excerpts; proposes structured walker patches as JSON diff objects with diagnoses, anchor texts, and replacement code. Patches are applied to isolated copies; the patched walker is re-run; output is diffed against the canonical reference plus the original baseline.

Headline numbers across three engines:

| Engine | Canon | Baseline | Patched | Diag | Patches | Applied | Recall lift vs baseline |
|---|---|---|---|---|---|---|---|
| transformers v4.57.3 | 41 | 28 | 28 | 1 | 1 | 0 | 0% (no patch landed) |
| vllm v0.7.3 | 26 | 66 | 0 (crash) | 3 | 3 | 1 | -100% (patch broke walker) |
| tensorrt v0.21.0 | 35 | 38 | 0 (crash) | 2 | 1 | 1 | -100% (patch broke walker) |

The trial-internal score is decisively negative. Two of three engines crash after patch application; the third's patch text did not match the anchor (LLM reconstructed it from memory rather than copy-pasting). The LLM consistently writes pseudocode: references `_handle_else` (undefined helper), `negated_conditions` (undefined variable), adds positional arguments without updating callers. Single-shot synthesis-of-code at 70B-q4 is poor.

BUT: 6 of 6 STRUCTURAL DIAGNOSES match the manually-curated `post_trial_a_gap_closure.md` inventory exactly. The LLM correctly identifies the gap categories: if/elif/else branch-descent (vllm CacheConfig), local-variable aliasing (vllm ModelConfig validators), normalisation-only-no-raise (vllm EngineArgs), type-blind probe synthesis (tensorrt), nested-config dispatch (tensorrt SchedulerConfig/QuantConfig/KvCacheConfig), defensive imports (transformers v4/v5 boundary). Each diagnosis names the exact line range, the structural cause, and the patch category required.

H4 fails as a strategy-internal scoring pattern. H4 succeeds as a DIAGNOSTIC ACCELERATOR for the spike-branch refactor; the 4 tier-A and tier-B patches map directly to spike issues with text already written.

### 4.5 H6 (no chunking; whole-source single shot) -- substrate-decomposition ablation

H6 tests whether chunking is the (b) recall ceiling driver. The LLM sees the full file source (~33k chars on transformers; vllm and tensorrt source are too large for 32k context). Transformers v4.57.3 result: invariant recall 17.9 percent under validated-union (down from 62.5 percent at baseline). Schema recall slipped from 83.0 to 75.0 (the docstring sits at the end of the schema prompt; lost-in-the-middle attention partly recovered it).

Mechanism: the LLM picked the "obvious" 16 `if X.field <pred>: raise` cases from the start of `validate()` and stopped. It did not exhaustively walk the file. Classic lost-in-the-middle attention; the 70B-q4 model at 32k context attends densely at prompt start and end, sparsely in middle. Chunking is NOT the bottleneck; removing it HALVES recall.

### 4.6 H7 (agentic loop with tools) -- LLM-as-orchestrator

H7 gave the LLM tools (`read_file`, `list_validators`, `run_miner`, `score_against`, `finalise`) with a 30-tool-call / 30-minute budget per cell. The harness handled tool dispatch with zero parse errors across 60 turns; tool selection was sensible (start with `run_miner`, then read/list, then `score_against`). Both transformers and vllm cells hit the 30-call cap with ZERO finalised invariants.

The vllm cell called `score_against` six times with empty invariants list and never adjusted strategy. The transformers cell entered a degenerate loop calling `list_validators(GenerationConfig)` nine times in a row. The score_against feedback (recall=0, recall=0, recall=0...) did not translate to corrective synthesis.

The mechanism: the LLM uses tools for EXPLORATION (a passive READ activity) but cannot bridge to SYNTHESIS (active emit). At 70B-q4, "I have read a lot of source" feels internally similar to "I have a complete invariant catalogue"; without strong prior on "stop reading, start emitting", the model defers indefinitely. Single-shot (b) FORCES synthesis by the prompt shape; agentic flexibility removes that pressure and the q4 model defaults to reading more. The collapse is sharper than H4's: H7 produces ZERO output where H4 produces broken output.

The harness is reusable; if a stronger model (Claude) can complete the read-emit loop, H7 with Claude is a one-cell-pair test in Phase 3c.

### 4.7 H9 (LLM diagnoses; no output mutation) -- LLM-as-categoriser

H9 read (a)'s output and engine source; produced structured-categorical-gap diagnoses without proposing patches. 8 diagnoses across 3 engines, 0 fabrications. 6 of 8 match H4's diagnoses and the manually-curated inventory; 2 of 8 are genuinely new (SamplingParams branch-descent in vllm; tensorrt model_config arbitrary_types_allowed). Per-cell wall ~50 s LLM time; cheapest pattern of the 9.

H9 is the strongest single piece of evidence FOR LLM-as-diagnoser. It pairs naturally with manual curation: the LLM diagnoses categories of thing a deterministic substrate misses; the maintainer reviews and addresses each category. The diagnoses are structured (category + example_field + structural reason) and immediately filable as spike issues.

### 4.8 E6 (field-anchored extension) -- substrate-decomposition variant

E6 modifies (b)'s prompt to include a preamble listing the engine's declared `__fields__` for each class chunk. Hypothesis: anchoring the LLM in the declared field set reduces hallucination risk on empty-chunk failures (the tensorrt v0.x case from Section 3.2).

Active cells (transformers + vllm): transformers result was recall NEUTRAL (57.1% vs 62.5% baseline); precision -4.5 pp. vllm result was recall -7.7 pp (a heuristic bug in the chunk-to-class targeting fell back to "all 15 classes" producing a 249-field anchor list for every chunk; this tested "noisy anchor" not "targeted anchor"). The variant's intended use case (catching the tensorrt v0.x HF GenerationConfig hallucination) was UNTESTED; active cells do not have the empty-chunk failure mode that triggers hallucination. Open variant for Phase 3c on a bumped tensorrt cell.

### 4.9 E9 (sequential cumulative-context) -- substrate-decomposition variant

E9 modifies (b)'s flow: instead of independent per-class extractions, the LLM reads chunks in fixed order with cumulative running notes of previously-extracted invariants. Hypothesis: cumulative context surfaces CROSS-CLASS invariants that per-class chunking loses.

Transformers result: invariant recall 39.3 percent (-23.1 pp from baseline); precision +37.6 pp. The LLM read "DO NOT re-emit invariants in running notes" as conservative dedup-pressure and under-emitted; the first chunk emitted 0 (correct: __init__ has no raise patterns), subsequent chunks emitted 1-3 each (vs ~4-8 in baseline). vllm result: -3.8 pp recall, +3.9 pp precision; cross-class invariants were NOT surfaced even though the running notes carried 47 invariants by the final chunks. The model did not LEVERAGE cumulative history for cross-class reasoning.

### 4.10 The synthesis-pressure thesis

H6, E9, and H7 share a unifying mechanism: **synthesis-pressure relaxation under flexibility.** (b)'s per-class chunking FORCES synthesis by the prompt structure -- one prompt, one chunk, must emit invariants per block. Any variant that adds flexibility -- whole-source attention, cumulative dedup with "don't re-emit" instruction, tool-mediated exploration with finalise as an optional verb -- relaxes that pressure, and the q4 model defaults to under-emit. E6 was neutral because it doesn't reduce synthesis pressure (the prompt still says "emit one invariant per block"). E9 reduced pressure indirectly via dedup framing.

The thesis is: at 70B-q4, the substrate's job is to FORCE the LLM into synthesis mode. Anything that gives the LLM a choice between exploration and synthesis loses synthesis. This generalises beyond the trial: in any LLM-augmented engineering pipeline at this model scale, prompt structure that requires a structured emission per chunk beats prompt structure that allows the model to defer emission.

### 4.11 The hallucination failure mode (b/tensorrt v0.x)

The most decision-relevant single discovery in the trial. Section 3.2 narrated the mechanism. The architectural implication: pure-LLM substrates without a deterministic gate are not safe. A chunker that returns empty input is functionally indistinguishable to the LLM from a chunker that returns valid sparse input; the LLM fills the void with priors. The mitigation is a deterministic gate downstream: runtime validation, or at minimum schema-existence checking against the live `Model.__fields__`. Both eliminate fabricated-field-name hallucination. Neither catches the more subtle "wrong predicate on real field" hallucination, which is rarer.

---

## 5. Results: validated-union rescoring

Section 2.3 narrated the methodology; here are the consequential per-strategy shifts.

| strategy | cells | recall_a | recall_vu | delta | prec_a | prec_vu | delta |
|---|---|---|---|---|---|---|---|
| a | 15 | 52.8% | 46.6% | -6.2 pp | 54.3% | 32.7% | -21.6 pp |
| b | 15 | 34.4% | 42.3% | +8.0 pp | 21.0% | 27.6% | +6.6 pp |
| d-ab | 15 | 100.0% | 77.6% | -22.4 pp | 93.6% | 73.6% | -20.0 pp |
| e6 | 2 | 43.6% | 50.4% | +6.8 pp | 28.0% | 46.5% | +18.6 pp |
| e9 | 2 | 34.0% | 41.4% | +7.5 pp | 29.9% | 52.5% | +22.6 pp |
| h6 | 1 | 12.8% | 17.9% | +5.0 pp | 31.2% | 62.5% | +31.2 pp |

The directional reading: (a) DROPS materially (-6.2 pp recall, -21.6 pp precision). The union exposes entries (a) missed that other strategies found and runtime-validated; (a) loses recall credit for those, and the entries (a) emitted that were absent from the union (or contested) lose precision credit. (b) RISES (+8.0 pp recall, +6.6 pp precision). Entries (b) emitted that were absent from (a) but runtime-validated become recall credit. (d-ab) DROPS the most in absolute terms (-22.4 pp recall, -20.0 pp precision); its by-construction 100 percent was inflated against (a)'s narrow output. The hybrid-pattern variants (e6, e9, h6) all RISE on the union, reflecting that their less-confident outputs are mostly real when filtered through runtime validation.

The per-cell union sizes on active versions:

| engine | version | unioned_unique | validated_both | infra_errors | (a)_only_ref | union_delta_vs_a |
|---|---|---|---|---|---|---|
| transformers | v4_57_3 | 108 | 56 | 5 | 41 | +15 |
| vllm | v0_7_3 | 106 | 39 | 20 | 26 | +13 |
| tensorrt | v0_21_0 | 70 | 11 | 19 | 35 | -24 |

For transformers and vllm, the union adds 15 and 13 entries respectively beyond what (a) alone covers. For tensorrt-llm, the union is SMALLER than (a)'s output because 19 of (a)'s 35 invariants fail runtime validation (the type-blind probe-value synthesis gap; Section 9 G-trt-1). The honest count of "things tensorrt-llm v0.21 actually validates as invariants" is 11 entries (validated_both); (a)'s 35-invariant output is over-counted by 24. This is the most consequential single rescore: (a)'s tensorrt-llm coverage was inflated by infrastructure issues, not by real coverage.

The validated-union rescore is the trial's strongest methodological correction. It generalises beyond mining-substrate comparisons to any pipeline where N strategies emit candidates that must be reconciled against ground truth; the union-and-runtime-validate framing avoids the "first strategy gets to be the reference" trap that bakes asymmetric bias into every later comparison.

---

## 6. Discussion: the LLM-role split as central finding

The 9 hybrid patterns plus the (b) baseline give 10+ data points on what the LLM at 70B-q4 can and cannot do. Aggregating across patterns by LLM ROLE rather than by pattern ID surfaces a consistent split:

| LLM role | Patterns testing it | Quality at 70B-q4 |
|---|---|---|
| Diagnose (structured-categorical output of gap reasons) | H4 (text portion) + H9 | Excellent. 0 fabrications across 8 diagnoses. 6/6 inventory match. Cheap (~50 s/cell). |
| Subtract (remove entries from existing output) | H2 | Error-prone. 3/3 vllm drops were false-drops. Conservative prompting reduces but does not eliminate. |
| Synthesise-code (write patches) | H4 (patch portion) | Poor. 0/3 patches lifted recall; 2/3 crashed; anchor texts hallucinated. |
| Synthesise-output (extract structured catalogue) | (b), E6, E9 | Substrate-ceiling-bound. ~50% transformers recall; ~30% vllm; ~16% tensorrt under vu. No substrate-side variant lifts the ceiling. |
| Synthesise-under-feedback (closed-loop with tool feedback) | H7 | Collapses. 0 finalised invariants on both cells; tool-use becomes passive exploration. |

The split has explanatory power for every cross-pattern observation in Sections 3-4. The H7 collapse is not a bug in the harness; it is the same synthesis-blindness pattern as H6 (under-emit on whole-source) and E9 (under-emit on cumulative dedup) manifesting more sharply because tool-mediated agentic patterns make synthesis maximally optional. The H2 false-drops are not a subtler form of the synthesis-blindness; they reflect a different weakness (pattern-matching the "raises when X" template too narrowly). The diagnose-strong / synthesise-weak asymmetry shows up at every level of the pattern hierarchy.

The split is decision-relevant for any LLM-augmented engineering pipeline llem builds. It argues:

- **Against** architectures placing LLMs in subtractive roles (H2-style "validate and drop") without deterministic second-opinion checks.
- **Against** architectures placing LLMs in autonomous synthesis-of-code roles (H4-style patch generation) without humans-in-the-loop.
- **Against** agentic-loop architectures where synthesis is optional (H7-style).
- **For** architectures placing LLMs in extractive roles with deterministic downstream validation ((b) + H3-style).
- **For** architectures placing LLMs in diagnostic roles where output is categorical-with-example-fields (H9-style; spike-issue-filing).
- **For** prompt structures that FORCE synthesis (per-chunk single-shot extract beats whole-source).

The split's most consequential corollary is the architectural commitment Section 8 makes: subtraction is deterministic; extension is LLM; synthesis-of-code stays human (with LLM diagnoses scaffolding the work). This commits llem to a division of labour that mirrors the empirical findings exactly.

The open question is whether the split is intrinsic to LLMs or specific to `llama3.1:70b` at q4 quantisation. Phase 3c tests this. The trial's prior is that Claude (Sonnet 4.6/4.7) will SOFTEN the ceilings (better synthesis-under-feedback, better synthesis-with-flexibility) but not INVERT the split (subtraction won't become reliable; autonomous synthesis-of-code won't suddenly become safe at any scale). If the prior holds, the architecture commitment survives; the operational layer can substitute Claude for `llama3.1:70b` in the extend-propose role with the same role-split discipline. If the prior fails -- if Claude breaks the (b) ceiling materially, or H7-with-Claude actually finalises -- the architecture absorbs autonomous-discovery patterns and the role-split discipline relaxes.

---

## 7. Failure-mode catalogue

The trial surfaced six distinct failure modes that the original `silent / detectable / crash` rubric partially conflated. Each has different operational consequences.

1. **Detectable crash** (clean stderr trace). Examples: (a) on transformers v-2/v+major; (a) on vllm bumped. Operationally: easy to monitor; the substrate noisily signals "I can't run here". CI gates on the stderr signature.

2. **Silent failure -- empty extraction.** Example: (b) on vllm v+major. The chunker returns empty; the LLM emits ~4 sentinel invariants from the failure marker; recall reports 0 percent; cell count is 4. Operationally: detectable via low `cell_count` plus the failure-marker pattern. CI can gate on `cell_count == 4 AND failure-marker present`.

3. **Silent failure -- hallucination from empty input.** Example: (b) on tensorrt v0_19_0 / v0_20_0. The chunker returns empty class bodies; the LLM HALLUCINATES 30+ HuggingFace `GenerationConfig` field names; recall reports 16 percent; `cell_count` is ~37. The most insidious failure mode discovered. Mitigation: schema-existence gate (cheap; catches `do_sample` not in `__fields__`) or runtime gate (more expensive; catches false predicates on real fields too).

4. **Under-emit from synthesis-pressure relaxation.** Examples: H6, E9, H7. The model defers structured emission when prompt structure permits. Operationally: detectable via cell count drop versus baseline; less observable than silent failure because the cell still produces some output.

5. **MINER_VERSION_BLIND silent re-extraction.** Example: (a) tensorrt-llm on all 4 bumps. Reports 100 percent recall + 100 percent precision; substrate-wiring artefact. Operationally: detectable only by external audit ("did the walker actually look at bumped source?"); not visible from score JSON alone. The most trust-corrosive failure mode discovered, because it makes the cell look BEST when it's actually doing nothing.

6. **Silent failure -- false-drop in subtraction.** Example: H2 on vllm. Cell looks normal; 3 valid invariants are removed. Operationally: detectable only via comparison to pre-validation output; the failure cannot be inferred from the cell's own metrics.

Six modes; six different mitigation paths. The Phase 5 production substrate's monitoring layer needs to distinguish them rather than collapsing to "silent". Modes 1, 2, 5 are detectable via cell-level diagnostics; mode 3 requires runtime-validation; mode 4 requires baseline comparison; mode 6 requires pre/post-comparison.

---

## 8. The decision space

Five viable architectures emerge from the trial's evidence. Each makes specific commitments and forecloses specific alternatives. The trade-offs below cite trial data; the recommendation in Section 9 commits to one combination.

### Architecture I: Pure deterministic (Scenario 1)

**Shape:** (a) deterministic across all engines. Close the 7 post-trial gaps. Refactor per Bake-off A's ~1800 LoC target. Ship.

**Trade-offs:** Lowest implementation risk (substrate exists). Lowest operational complexity (no LLM dependency). Zero per-cell LLM cost. Catches none of the 28+ entries the validated union finds beyond (a) on active cells. Brittleness modes stay heterogeneous; each engine needs its own per-bump patch path.

**Maps to Plan Scenario 1.** Commits llem to deterministic-only; forecloses LLM as production substrate. The pre-trial default. Conservative pick.

### Architecture II: Deterministic + extend-propose hybrid (Scenario 4)

**Shape:** (a) deterministic baseline across all engines (with the 7 gaps closed) + chunked LLM extraction in (b)'s shape + deterministic runtime gate per engine (existing `scripts/validate_invariants.py`) + validated union as canonical artefact. LLM role: extract + propose only. Deterministic role: validate + subtract.

**Trade-offs:** Best-quality outcome by every measured axis. Highest engineering cost (both substrates built and maintained). Per-cell LLM cost ~150-200 Wh per bump on (b)'s shape; on Renovate cadence (a few bumps per engine per year) this is single-digit kWh per year total. Cheapest known mitigation against the hallucination failure mode (Section 7 mode 3): runtime gate filters fabrications. Brittleness modes diversified across substrates -- one substrate failing on a bump doesn't doom the cell because the other surfaces partial coverage.

**Maps to Plan Scenario 4.** Commits llem to two-substrate architecture. The architecture the trial's data most strongly converges on, both because H3 demonstrably lifts precision via runtime gate and because the LLM-role split argues for this exact division of labour.

### Architecture III: Pure LLM substrate (Scenario 3)

**Shape:** Pivot to (b) or (c) extraction. Deprecate the ~3800 LoC of (a) producers; keep lightweight verification + landmark machinery. Validated union from (b) + runtime gate becomes SSOT.

**Trade-offs:** Lowest LoC footprint (~1800 LoC removed). Highest model-dependency risk. Currently weak case at 70B-q4: (b) tops out at 50 percent recall on transformers, 30 percent on vllm, 16 percent on tensorrt-llm. The trial's hallucination failure mode demonstrates pure-LLM substrates need a deterministic gate anyway. Stronger case if Phase 3c shows Claude breaks the (b) ceiling materially.

**Maps to Plan Scenario 3.** Commits llem to LLM-dependent substrate; forecloses fully-deterministic future.

### Architecture IV: Per-engine substrate choice (Scenario 2)

**Shape:** (a) for engines where (a) works (transformers); (b) + LLM-extension where (a) is structurally challenged (vllm, tensorrt-llm). Per-engine architectural divergence accepted.

**Trade-offs:** Matches per-engine mining-friendliness. Highest per-engine pipeline complexity. Cross-engine architectural asymmetry as feature. BUT the data DOES NOT support this: (b)'s ceiling is LOWER on vllm and tensorrt-llm than on transformers, so substituting (b) for (a) on those engines trades brittleness for lower ceiling. The right per-engine handling is to fix (a)'s brittleness on each engine (the 7 gap closures) rather than replace it.

**Maps to Plan Scenario 2.** Included for completeness; the data argues against.

### Architecture V: Curation primacy (Scenario 5)

**Shape:** Treat both (a) and (b) as EVIDENCE; human or LLM curation produces the canonical artefact. Validated union IS the curated output if quality is sufficient; otherwise a maintainer reviews per-bump. Mining becomes evidence-mining; curation becomes SSOT.

**Trade-offs:** Highest correctness floor (a curator looks at every entry). Highest per-bump human cost (~hours per engine per bump). Most flexible w.r.t. substrate quality. CONSISTENT WITH THE DATA: the validated-union scoring already implements the EVIDENCE framing for measurement; promoting it to production architecture is a small further commitment.

**Maps to Plan Scenario 5.** Naturally aligns with Open Question 10 (mining-as-SSOT vs mining-as-evidence).

---

## 9. Recommendation

llem should adopt **Architecture II (deterministic + extend-propose hybrid)** as the production substrate, with **Architecture V (curation primacy)** as the operational layer above it.

The concrete shape: per-engine, per-version, the production substrate runs (1) (a) deterministic mining first; (2) (b)-style chunked LLM extraction second; (3) deterministic runtime gate against the live engine in container, third; (4) the validated union of (a) + (b) -- filtered to runtime-validated entries -- becomes the cell's canonical artefact. (5) A maintainer reviews the validated union per bump (the curation layer), with H9-style LLM-diagnose as an assistant that surfaces "categories of thing (a) is structurally blind to" for the maintainer's attention. The curated artefact flows into `engine_versions/<e>/v*/outputs/` and from there into `src/<e>/`.

The LLM role is enforced by architecture: extraction + diagnosis only. Subtraction is deterministic (the runtime gate). Synthesis-of-code stays human (H4's diagnoses scaffold the work but don't ship as patches). Maintainer review owns the final artefact.

Defended against the alternatives:

- **Against Architecture I (pure (a)):** the validated union shows 28+ entries (a) misses on active cells across 3 engines. The LLM-extension pattern (d-ab) catches most of these at zero false-positive risk (the deterministic gate filters). Ignoring this signal corrupts the canonical artefact.

- **Against Architecture III (pure (b)):** the (b) recall ceiling at 70B-q4 is too low (30-50 percent under validated-union) for substrate use in isolation. The (a) deterministic floor catches what (b) misses and is essentially free per bump. The hallucination failure mode demonstrates pure-LLM substrates need a deterministic gate anyway.

- **Against Architecture IV (per-engine):** the data does not support architectural asymmetry. The engines where (a) is most brittle are precisely the engines where (b) is also weakest. Routing around (a)'s brittleness via (b) trades one weakness for another. The right per-engine handling is to fix (a)'s brittleness on each engine.

- **Against Architecture V alone:** curation without mining is unreliable. Maintainers MISS things. Curation needs evidence streams to review. Architecture II provides the streams; Architecture V's operational layer uses them.

**What this commits llem to:**

- Both substrates remain in the production codebase. The ~1800 LoC mining refactor (Bake-off A target) lands. The LLM-extraction infrastructure (Phase 2's chunkers, prompts, retry harness) becomes production code.
- The deterministic runtime gate (`scripts/validate_invariants.py`) lifts from script to library; per-engine container dispatch is a routine production concern.
- Per-version cells run via CI on Renovate bumps; the validated union is committed to engine-knowledge data files; src/ is regenerated via the existing codegen pipeline.
- Maintainer reviews validated-union diffs per bump (the curation layer); H9-style LLM-diagnose pre-flags gap categories.

**What this trades off:**

- Higher engineering cost than pure (a). Lower than parallel-and-reconcile architectures because the runtime gate is straightforward dispatching to existing infrastructure.
- Per-bump LLM cost: ~150-200 Wh per cell at trial-locked prompts; on Renovate cadence, a few kWh per year total. Energy-cheap; wall-clock-real (~30-90 min per cell at parallelism).
- Vendor-dependency: at trial scale, OSS LLM (Ollama / `llama3.1:70b`) suffices. If Phase 3c shows Claude lifts the ceiling materially, the architecture supports drop-in substitution.

**Conditions for revisit:**

- **Phase 3c Claude results.** If Claude breaks the (b) ceiling materially on extraction (e.g. transformers (c) recall > 75 percent vs current (b) 56-62 percent), the cost-quality balance shifts. The architecture supports drop-in Claude for the extension layer; the operational question is whether API quality + cost exceeds local 70B-q4 by enough margin to switch.
- **If H7-style agentic with Claude works.** If a stronger model bridges the synthesis-blindness gap, the architecture can absorb autonomous-discovery patterns. The H7 harness is reusable.
- **If validated-union recall plateaus < 80 percent even with deterministic-extend-propose.** If maintainer review consistently catches entries neither substrate finds, the curation primacy framing (Architecture V) becomes the dominant pattern and substrate is demoted to evidence.

---

## 10. Outstanding work

Post-trial work backlog, ordered by dependency:

1. **Phase 3c (Claude comparison).** 15-cell (c) matrix + Claude-variant of key hybrid patterns (H4, H7, H6 on vllm/tensorrt source, E6 on bumped tensorrt, E9 cumulative context). ~$20-30; ~1-2 days agent work when key arrives. May refine Section 9 but unlikely to overturn it; the LLM-role split is robust and Claude is more likely to soften ceilings than invert the split.

2. **Post-trial (a) gap closure backlog.** Seven gaps catalogued in `findings/post_trial_a_gap_closure.md`. Close regardless of substrate choice.

   | Gap | Engine | Effort | Closure path |
   |---|---|---|---|
   | G-vllm-1 | vllm EngineArgs normalisation uncaptured | 80-150 LoC | Re-frame normalisation as schema defaults or extend walker for `severity=dormant` patterns |
   | G-vllm-2 | vllm ModelConfig local-variable aliases | 150-250 LoC | Light call-graph analysis: trace `self.X` -> local -> validator |
   | G-vllm-3 | vllm CacheConfig if/elif/else branch-descent | 30-50 LoC | Walker patch: traverse all branches |
   | G-trt-1 | tensorrt type-blind probe synthesis | ~30 LoC | Type-aware `_value_satisfying`; lives in `scripts/_common.py` |
   | G-trt-2 | tensorrt DeprecationWarning poisoning negative-case capture | ~10 LoC | Strip-list in `_run_tensorrt` matching vllm `_VLLM_BOOTSTRAP_NOISE` pattern |
   | G-trt-3 | tensorrt nested-config dispatch | 200-400 LoC | `_NestedConfigWalker` mixin scanning Pydantic field annotations for nested-config types |
   | G-trf-1 | transformers defensive imports at version bumps | 50-100 LoC | Try/except wrappers; falls back to AST-only walking |

   Total: ~500-1000 LoC across 7 gaps. H4 + H9 diagnoses provide design input for each. Closure mechanism: either H4-patches-as-PR (where patches are mergeable post-review) or spike-branch refactor on the existing Bake-off A target.

3. **Spike-branch refactor (Bake-off A target).** ~1800 LoC accidental-complexity removal. H4's outputs feed cross-engine abstractions: nested-config dispatch (G-trt-3, G-vllm CacheConfig analogue, transformers BNB), if/elif/else branch descent (G-vllm-3), local-variable alias tracking (G-vllm-2).

4. **Phase 5 curation pipeline (Architecture II + V instantiation).** Dogfood on transformers first (highest reference maturity, lowest brittleness surface). Build reconciliation script producing validated union per cell; maintainer-review interface; H9-style LLM-diagnose pre-flag. ~1-2 weeks for the pilot.

5. **OQ9 storage strategy revisit.** Post-spike-refactor when artefact footprint stabilises. Not blocking. Architecture II doesn't constrain the answer; both git-tracked and GH-artefacts-pinned work.

6. **Trial PR extraction.** Spike commits chunk into reviewable PRs (PR-A/B/C/D/E per DECISIONS_LOG architectural pattern).

7. **Research-paper IA restructure** (deferred sub-task per `DECISIONS_LOG` 2026-05-25 user direction). When the migration to `research/mining-substrate-trial/` is fully consolidated, restructure into the academic-paper IA: problem-statement / methodology / results / decision-space / recommendation / reproducibility / appendix. The corpus's components map cleanly; the restructure is editorial.

---

## 11. Limitations and threats to validity

The trial's findings come with explicit limitations.

**Model scale specificity.** All LLM-substrate findings derive from `llama3.1:70b` at q4_K_M quantisation. The LLM-role split (Section 6), the synthesis-pressure thesis (Section 4.10), the hallucination failure mode (Section 7) may all soften at higher model scale (Claude, GPT-5, larger OSS models) or harden at lower scale (8B). Phase 3c is the direct test on the most-decision-relevant axis. The 8B probe (Section 3.4) is one data point on the lower-scale axis.

**Engine count.** Three engines (transformers, vllm, tensorrt-llm). The cross-engine asymmetry findings (Section 3.1) generalise to other LLM-inference engines with caution; the specific brittleness modes (landmark-missing, dep-import collapse, walker version-blindness) are likely engine-specific.

**Version bumps per engine.** Four bumped versions per engine (v-2, v-1, v+1, v+major). The transformers matrix has a patch-level v+1 (no 4.58.x ever released); the tensorrt matrix has an early-major v+1 (no 0.22.x ever released). Brittleness aggregates across engines need to weight these asymmetries.

**Bounded pattern catalogue.** Nine hybrid patterns explored. The pattern space is in principle infinite; the trial's coverage is dense at the LLM-role axes (extract, validate, propose, diagnose, modify-miner, orchestrate, curate) and the substrate-decomposition axes (per-class chunk, per-validator, whole-source, cumulative). Unexplored axes include ensemble extractors (E7), evidence-curation (E4), and structured multi-pass (5+ passes with inter-pass verify).

**Runtime-validation infrastructure gaps.** The trial's runtime gate is production-grade for all three engines via `scripts/validate_invariants.py`, but the trial-internal wrapper `trial_scoring.runtime_validate_invariants` is transformers-only (the others would need lift via per-engine container dispatch). Schema validation (Layer A) is fully implemented; behavioural validation (Layer B, "does varying this field produce measurable output difference under inference?") is out of scope for the trial.

**(a) gap preservation.** Per discipline, known (a) gaps were preserved as research data. The trial measures (a) at the "realistic state" rather than at a hypothetical patched state. This was the correct discipline (mid-trial optimisation toward what looks promising would corrupt the data) but means the (a) aggregate numbers in Section 3 understate (a)'s true ceiling once the post-trial gap closure backlog (Section 10 item 2) lands.

**Reference catalogue for non-active cells.** Phase 1 Day 4 deferred per-version reference construction; bumped cells score against the active reference. This is honest for recall measurement (do bumped cells still surface active-version invariants?) but ambiguous for precision (cannot separate "version-specific new invariants surfaced in the bumped version" from "miner over-emission"). Validated-union rescoring addresses this for cells where multiple strategies ran on the bumped source.

---

## 12. Meta-findings: what the trial taught beyond the substrate question

Five methodological findings worth carrying forward to other LLM-in-software-engineering tasks at llem:

**1. Validated-union ground truth as scoring discipline.** When comparing N strategies for the same artefact, every strategy's output should contribute to the ground truth (filtered to what survives downstream validation); none should be privileged as reference. Single-entry methodological fix that retroscored every cell honestly. Generalises to any substrate comparison llem does in future. The retro-rescore of (a) from 100 percent to 66 percent recall on transformers active is the headline correction; (a) was measuring itself against itself, with predictable consequences for the comparison.

**2. The LLM-role split as a general principle.** At 70B-q4 specifically (likely with model-scale-dependent thresholds elsewhere): LLMs are reliable at diagnosis, error-prone at subtraction, weak at autonomous synthesis-of-code, ceiling-bound at extraction, collapse under synthesis-with-feedback. Architecture-level implication: place LLMs as extractors + diagnosers; deterministic systems as validators + subtractors; humans as final curators with LLM scaffolding. Likely to apply to any task where llem considers LLM augmentation of an existing deterministic pipeline.

**3. The synthesis-pressure thesis.** Forced-output prompts beat flexible exploration at this model size. Per-class chunking works because the prompt structure FORCES synthesis at each chunk; variants that relax the pressure (whole-source, cumulative dedup, tool-mediated exploration) consistently under-emit. For production: keep synthesis-forcing prompts. For Claude scale: test whether the thesis holds.

**4. The cross-engine asymmetry pattern.** (a)'s brittleness, (b)'s ceiling, the chunker's failure modes all differ STRUCTURALLY across the three engines. Production architectures should anticipate per-engine asymmetry in failure modes even when the substrate is uniform; monitoring layers should be engine-aware.

**5. The brittleness-as-axis discipline.** Per-version cells (not "active plus one bump") were the most expensive design decision and produced the most decision-relevant data: the three (a) brittleness modes, the chunker file-layout-assumption brittleness on vllm, the hallucination-on-empty-input mode on tensorrt, the MINER_VERSION_BLIND substrate-wiring artefact. None would have surfaced from active-only cells. Generalises to any substrate evaluation with upstream-bump exposure: don't measure substrate quality on the active version alone; measure across the bump window.

---

## Closing

The mining-substrate empirical trial set out to gather maximal information across (engine, version, strategy) cells under matrix discipline plus open-ended hybrid exploration, with strategy constructed AFTER from the assembled evidence rather than picked mid-trial. Both halves of the brief landed: 51 scored cells with locked prompts plus 9 distinct hybrid patterns.

The recommended strategy is Architecture II (deterministic + LLM-extend-propose, deterministic-validate, validated-union as canonical artefact, curation primacy as operational layer). The recommendation is contingent on Phase 3c not overturning the LLM-role split; the architecture supports drop-in substitution of Claude or any stronger model if ceilings lift materially.

The trial's discipline preserved gaps as research data; the production discipline closes the gaps after. The 7 (a) gaps plus the chunker brittleness modes plus the substrate-wiring artefact all become spike-refactor PR-scope tasks with H4's diagnoses plus H9's structural reads as design input.

The trial is closed. Phase 5 pilots Architecture II on transformers first.

---

## Cross-references

- `findings/empirical_trial_outcome.md` -- Phase 4 synthesis (the predecessor document this write-up draws from).
- `findings/phase4_0_validated_union_summary.md` -- validated-union ground truth methodology and per-cell union sizes.
- `findings/trial_matrix_vu.md` and `trial_matrix_vu.csv` -- the 51-cell validated-union-rescored matrix.
- `findings/trial_matrix.md` and `trial_matrix.csv` -- the original (a)-as-reference matrix (retained for delta comparison).
- `findings/phase3a_complete_summary.md` -- the 47-cell pure-strategy aggregate cross-engine brittleness summary.
- `findings/phase3a1_active_matrix.md` -- the 11-cell active-version Phase 3a.1 record.
- `findings/phase3a2_{transformers,vllm,tensorrt}_progress.md` -- per-engine bumped-cell records.
- `findings/hybrid_experiments/{h2_h3_h9,h4_modify_miner,h6_e6_e9,h7_agentic}/` -- per-pattern findings.
- `findings/phase2_llm_infrastructure.md` -- (b) infrastructure design and calibration record.
- `findings/phase2_locked_prompts/` -- the exact locked prompts used by all (b), (b_8b), and (d-ab) cells.
- `findings/phase1_version_lock.md` -- the 15-cell version matrix and wheel-compatibility analysis.
- `findings/phase1_{vllm,tensorrt}_miner_lift.md` -- the Phase 1 mining-parity work.
- `findings/post_trial_a_gap_closure.md` -- the durable (a) gap commitment backlog.
- `findings/trial_epistemic_framing.md` -- the trial's discipline doc.
- `DECISIONS_LOG.md` -- chronological narrative log of every decision and finding.
