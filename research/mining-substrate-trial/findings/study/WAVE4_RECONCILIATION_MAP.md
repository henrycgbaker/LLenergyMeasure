# Wave 4 reconciliation + reuse audit: mapping waves 1-4 onto the prior PoC

Status: audit doc, 2026-06-10. Scope: reconcile the recent waves 1-4
(`spike/engine-knowledge-as-data`) against the PRIOR Wave-2 PoC (the W-A..W-G
workflow taxonomy + the `strategies/` framework + the `findings/hybrid_experiments/`
h2-h9 batch). Re-diagnose the agentic ZERO (prior h7 AND recent wave-4b). Recommend
how to get a correct agentic result.

TL;DR

1. Waves 1-3 are a re-run of the prior **W-E / d-then-llm-extend** workflow and the
   prior **H3 (propose -> gate verifies)** experiment, but with three genuine
   advances: kwargs-emission (the cross-field validation lever), the REAL runtime
   gate as SSOT (not reference-scoring), and a 2026 model-tier sweep on the current
   cells. Wave 4a is the prior **W-C / strategy-(b) pure-LLM** re-run, same advances.
   Wave 4b/4c is the prior **H7 agentic loop** re-attempted on a new (LangGraph)
   harness.
2. `strategies/llm_extractor.py`'s **LLMBackend/OllamaBackend abstraction is the one
   cleanly reusable layer** (modulo the AnthropicBackend, which is superseded by the
   Agent-tool Opus path). The **chunkers are STALE** (version-pinned to old wheels)
   and already superseded by `wave2_llm_source.py`. The agentic harness, scorers, and
   `run_*` drivers are stale (reference-scoring + old paths + old cells).
3. **h7's ZERO is largely a HARNESS ARTEFACT, not a robust model finding.** The
   harness forces ALL-AT-ONCE synthesis: the only way to produce output is a single
   `finalise(invariants_yaml=<ENTIRE catalogue>)` call, and the only way to self-check
   is `score_against(emitted_yaml=<ENTIRE catalogue>)`. There is no incremental emit.
   The model never crosses the "stop reading, assemble the whole YAML in one shot"
   threshold under a 30-call budget. An **incremental-emit** harness changes the task
   shape and plausibly clears the 0.
4. The recent wave-4b 0 is a SEPARATE, even-more-clearly-artefactual failure: a
   tool-contract bug (7s wall, 0 emitted, no agent_error) - not synthesis-blindness.
5. RECOMMENDATION: **adapt the recent LangGraph harness** (already on the current
   cells + runtime gate + incremental `emit_invariant`), fix its tool contract, and
   run devstral:24b + qwen2.5-coder:32b vs the llama3.1:70b h7 baseline. Do NOT revive
   the prior `agentic_tool_harness.py` as-is (stale all-at-once + reference-scoring).

---

## 1. The map: waves 1-4 onto W-A..W-G + h2-h9

The prior taxonomy lives in `WAVE2_WORKFLOWS.md` (W-A..W-G) and `WAVE2_PRIMITIVES.md`
(axes); the prior experiments are the `findings/hybrid_experiments/` h2-h9 batch.

| Recent wave | Prior workflow (W-*) | Prior experiment (h*) | Prior assembly axis | What it re-establishes | What is GENUINELY NEW |
|---|---|---|---|---|---|
| **W1** det-then-llm-extend, match-only, gemma3:12b + Opus | **W-E** (universal/det floor + LLM extend) | overlaps **H3** (propose -> gate verifies) + d-ab | `det-then-llm-extends` | LLM net-new tail exists but does not auto-confirm; bottleneck is the validation path, not LLM recall | The gate is the REAL `validate_invariants.py` runtime gate run in-container (`study_gt_pilot`), not `trial_scoring` reference-scoring. Localises the bottleneck to the *single-field auto-synthesis* limit. Current cells (vllm 0.19.1, tensorrt 1.2.1). |
| **W2** Arm A: det-then-llm-extend + **kwargs-emission**, Opus + gemma | **W-E** | **H3** with a stronger gate | `det-then-llm-extends` + `llm-then-det-validates` | (prior had no kwargs lever) | **kwargs-emission**: the LLM also emits constructible `kwargs_positive/negative`, making the cross-field tail gate-confirmable. 8 verified-real cross-field confirms - first cross-field GT-growth. Cross-field error-locus attribution fix. This lever does NOT exist anywhere in the prior PoC. |
| **W3** tier sweep: size x code-tuning 2x2 (gemma-12b, qwen-7b/32b, llama-70b) + Opus ceiling | **W-E** (model-tier study within it) | extends **H7's** "stronger model might" hypothesis into actual OSS tiers | `det-then-llm-extends` | prior only had a single 70B-q4 datapoint and a "stronger model untested" note | The full **2026 model-tier sweep**: scale threshold for the cross-field tail sits 12B->32B; a code-tuned 32B beats a general 70B (more cross-field, higher precision, faster). Internals-guard validated. None of this exists in the prior PoC (single model). |
| **W4a** pure-LLM (no floor) + kwargs, chunked prompt | **W-C** (pure LLM) | strategy **(b)** | `llm-only` | the floor is load-bearing: pure-LLM recall is far below det-then-llm-extend | kwargs carried into pure-LLM; runtime-gate scoring; the 2026 roster (incl. MoE qwen3-coder:30b). ~12-18% GT recall confirms the prior "(b) ~50% reference-recall" does not translate to gate-confirmed recall once you score against the real gate. |
| **W4b** agentic extract-only (LangGraph): list/grep/read + `emit_invariant`; gate scores after | **W-C/agentic** | **H7** (agentic loop) | `agentic-tool-use` | (prior h7: agentic at 70B-q4 collapses to 0) | A NEW **LangGraph** harness with **incremental** `emit_invariant` (one invariant per tool call) - structurally different from h7's all-at-once `finalise`. Current cells + runtime gate. BUT this run FAILED on a tool-contract bug (see SS3), so it has not yet tested the incremental hypothesis. |
| **W4c** agentic + `gate_probe` (gate-as-tool): agent tests each probe vs the REAL gate, self-corrects | **W-D/closed-loop**-adjacent | **H7** with `score_against` upgraded to a per-probe runtime gate | `closed-loop-feedback` + `agentic-tool-use` | (prior h7: `score_against` feedback ignored; reference-scoring of the whole draft) | `gate_probe` calls the REAL container gate per single probe (not whole-catalogue reference-scoring). This is the closed-loop-feedback shape `WAVE2_PRIMITIVES.md` axis-3 marked "untested but scaffolded (h15)". Built; not yet run clean. |

Prior experiments NOT re-touched by waves 1-4 (and why they stay relevant):

- **H2** (LLM validates/subtracts (a)): the prior `subtract` role - error-prone at
  70B-q4 (3/3 vllm false-drops). Waves 1-4 do not use a subtract role; the recent
  internals-guard is a *deterministic* filter, not an LLM subtract. H2's "subtract is
  untrusted" finding still stands and is why the recent work keeps the LLM in
  propose-only.
- **H4** (LLM patches the miner): the prior `patch-code` role / W-D extract path -
  0/3 patches lifted recall. Waves 1-4 deliberately avoid W-D (no LLM-patches-producer
  wave). H4's negative result is the reason.
- **H9** (LLM diagnoses): the prior `diagnose` role / W-F - excellent (0 fabrications).
  Untouched by waves 1-4; remains the strongest cheap LLM role and a candidate for a
  future wave.

Net: waves 1-3 are **W-E re-instantiated with a runtime gate + a kwargs lever + a tier
sweep**; wave 4a is **W-C/(b) re-instantiated**; wave 4b/4c is **H7 re-attempted on a
new harness**. The genuinely-new contributions are (a) kwargs-emission as the
cross-field validation lever, (b) the runtime gate as SSOT replacing reference-scoring,
(c) the 2026 model-tier characterisation, (d) the current cells. The *workflow shapes*
are all prior art.

---

## 2. Reuse assessment of `strategies/`

Staleness boundary (what makes a module STALE for the current study):

- **Old cells/versions**: prior code targets transformers 4.57.3, vllm 0.7.3,
  tensorrt 0.21.0 (verified: `engine_versions/<engine>/<vslug>/outputs/` exists ONLY
  for those three slugs). Current study cells are vllm 0.19.1 + tensorrt 1.2.1, GT at
  `findings/study/ground_truth/<engine>/<vslug>/invariants/PILOT_GT.yaml`.
- **Old source paths**: chunkers read pre-unpacked wheels at hard-coded
  `/tmp/vllm-unpacked/`, `/tmp/trt-llm-0.21.0/`. Current source lives at
  `/tmp/trial_<engine>_<vslug>_venv/src/` (resolved by `wave2_llm_source.source_files_for`).
- **Reference-scoring not runtime gate**: prior harness scores via
  `trial_scoring.score_invariants(reference_invariants=...)` against
  `invariants.proposed.yaml`. Current study's SSOT is the runtime gate
  (`study_gt_pilot` -> `validate_invariants.py` in-container).
- **AnthropicBackend (`ANTHROPIC_API_KEY`)**: the current study's "Opus rung" is the
  Agent subagent tool (whole-source one-call), not the SDK key path.

| Module | Verdict | Reason | Adapt cost to current study |
|---|---|---|---|
| `llm_extractor.py` -> **`LLMBackend` / `OllamaBackend`** | **CURRENT / REUSABLE** | Pure transport: HTTP to Ollama, streaming, num_ctx/num_predict/temp, retry. No version/path/scoring coupling. This IS the model-call layer. | Near-zero. Already conceptually re-implemented inline as `wave2_llm_cells.ollama_generate` (:11435, temp 0, num_ctx 16384). Could import `OllamaBackend` directly instead; only diff is the recent code defaults to :11435-container and an inline generate. The `extract_with_retry` + fence-stripping + YAML-salvage helpers are reusable too. |
| `llm_extractor.py` -> **`AnthropicBackend`** | **STALE (superseded)** | `ANTHROPIC_API_KEY` SDK path; the current study uses the Agent subagent tool for the Opus ceiling. | Don't port. Keep only if a programmatic Opus rung is later wanted. |
| `llm_extractor.py` -> fence/parse/filter helpers | **REUSABLE** | `strip_code_fences`, `parse_yaml_block` (with prefix-salvage), `is_internal_plumbing_field`. Version-agnostic. | Near-zero; `wave2_llm_cells.parse_invariants` overlaps but the salvage path is strictly better. |
| `transformers_chunker.py` / `vllm_chunker.py` / `tensorrt_chunker.py` | **STALE** | Version-pinned (vllm 0.7.3 hard-coded wheel path; tensorrt 0.21.0; transformers 4.57 via `inspect.getsource`). Hand-curated per-class chunk lists for the OLD class shapes. | Already SUPERSEDED by `wave2_llm_source.py` (version-agnostic AST validator-body extraction + greedy packing, paths from `source_files_for`). No reason to adapt the old chunkers; the new chunker is the right primitive. Salvage value: the per-class companion-inlining *idea* (e.g. GuidedDecodingParams inlined into SamplingParams) if a future cell needs companion-following, but the new validator-body chunker covers the current cells. |
| `agentic_tool_harness.py` | **STALE (as-is); design-reusable** | All-at-once `finalise`/`score_against`; reference-scoring via `trial_scoring`; sandbox roots hard-coded to old venvs (`v4_57_6`, `v0_7_3`, `v0_21_0`); `run_miner`/`score_against` read `engine_versions/.../invariants.proposed.yaml`. | High to adapt in place; NOT worth it - the recent `wave4_agentic.py` LangGraph harness already targets the current cells + runtime gate with incremental emit. Reuse the *sandbox-resolution + parse-robustness ideas*, not the module. |
| `hybrid_extractor.py` (d-ab/d-ac scaffold) | **STALE** | transformers v4_57_3 only; writes to old outputs dir; reference-scoring contract. | The recent `wave1.py` (det-then-llm-extend) already re-implements this against the runtime gate. Don't port. |
| `prompts.py` | **STALE (few-shot), pattern-reusable** | Few-shot examples all from transformers v4_57_3; schema/invariants prompt locked for the old contract; no kwargs emission. | The current study's prompts (`pure_llm_kwargs_prompt.md`, the wave-2 kwargs prompt `7cd74960`) supersede it because they add kwargs-emission. Salvage the "no fences / no `_` fields / only-source-visible" rules (already carried). |
| `claude_extractor.py`, `llm_b_oss.py`, `h4_engine_configs.py`, `run_h2_h3_h9.py`, `run_h4_modify_miner.py`, `run_h6_e6_e9.py`, `run_h7_agentic.py`, `run_calibration.py`, `run_*` | **STALE** | All wired to old cells + reference-scoring + old paths. | Don't port. They are the *prior experiment drivers*; their findings are captured in `findings/hybrid_experiments/`. The recent `scripts/phase1/wave*.py` are the current equivalents. |

Direct answers to the two specific reuse questions:

- **Is `LLMBackend`/`OllamaBackend` reusable as the model-call layer?** YES - it is the
  cleanest reusable piece. It is pure transport with no version/path/scoring coupling.
  The recent code already does the equivalent inline (`wave2_llm_cells.ollama_generate`);
  importing `OllamaBackend` would be a strict consolidation (gains streaming + retry +
  the salvage parser). The only thing to drop is `AnthropicBackend`.
- **Are the chunkers reusable?** NO - they are version-pinned and hand-curated for the
  old class shapes, and already superseded by `wave2_llm_source.py`'s version-agnostic
  validator-body chunker. Keep the *companion-inlining idea* on the shelf; don't port
  the modules.

---

## 3. Re-diagnosis of h7's ZERO

### The harness mechanic (read directly from `agentic_tool_harness.py`)

The tool set is exactly five tools (lines 17-24, 461-468, 548-647):

- `read_file(path, line_start, line_end)` - read a source slice.
- `list_validators(engine, class_name)` - AST method-name lookup.
- `run_miner(engine, version_slug)` - replay the (a) artefact summary.
- `score_against(emitted_yaml, engine, version_slug)` - score the **WHOLE** draft.
- `finalise(invariants_yaml)` - submit the **ENTIRE** catalogue; loop ends.

Critically:

- **There is no incremental-emit tool.** The catalogue does not accumulate in harness
  state across calls. The only way to register output is to pass the *entire*
  `invariants_yaml` string in one `finalise` call (`tool_finalise`, line 461: it just
  sets `state["final_yaml"] = invariants_yaml`, overwriting). `score_against` likewise
  takes the *entire* `emitted_yaml` each call and recomputes from scratch.
- So the agent's task is: read source across a 30-call budget, then **assemble and emit
  the complete YAML catalogue in a single tool call.** That single emission is the hard
  part - it must hold the whole catalogue in the output buffer at once, formatted
  correctly, with no incremental scaffolding.
- The system prompt even says "Build your invariants list incrementally. Call
  score_against with a draft YAML" (line 639) - but "incrementally" here means
  *re-send the growing whole draft each time*, which is exactly the all-at-once burden
  re-paid every call. The h7 traces show the model calling `score_against` with
  `invariants: []` repeatedly: it never paid that assembly cost even once.

### The empirical signature (from the committed artefacts)

- Both `finalised_invariants.yaml` are **0 bytes**; both `score.json` say
  `"finalise never called"`, recall 0, precision 0.
- `h7_summary.md`: tool dispatch was 100% successful (0 parse errors); the model
  explored competently (systematic file slices, sensible tool selection) but **never
  drafted a single invariant** and **never called finalise**. It hit `max_calls` while
  still in exploration mode on both cells.

### Verdict: largely a HARNESS ARTEFACT, not a robust model finding

The prior write-up's own "synthesis-blindness" framing already half-concedes this:
"single-shot patterns FORCE synthesis by the prompt shape (one call -> emit); agentic
patterns make synthesis OPTIONAL, and the model defers it indefinitely." That is a
precise description of an **all-at-once harness limitation**: when emission is a single
giant terminal action that competes with "read one more file," a 70B-q4 model under a
turn budget will keep reading. The 0 is the harness's emission shape interacting with a
weak model's planning, not evidence the model *cannot* produce invariants - strategy
(b) at the SAME model scale produced 0.385-0.538 reference-recall because its prompt
shape *forced* the single emission.

Three caveats keep this from being "purely artefact":

1. The model also ignored `score_against=0` feedback six times - a genuine (if mild)
   weak-feedback-assimilation signal at 70B-q4.
2. Path/class-name hallucination burned real turns (transformers `v4_57_3` vs `v4_57_6`
   venv; `list_validators(ModelConfig)` on transformers) - a real small-model
   brittleness, though also a harness ergonomics issue (`[]` vs "class not found").
3. We only have one model (llama3.1:70b q4) on the old cells, so "robust" was never
   warranted on N=1 anyway.

So: **h7's 0 is a harness artefact (all-at-once emission under a turn budget) compounded
by 70B-q4 planning weakness, NOT a fundamental "the model can't synthesise invariants"
finding.** An incremental-emit harness + a 2026 agentic-tuned model could plausibly
succeed. This is exactly what wave-4b was meant to test - and didn't, because of a
separate bug.

### The recent wave-4b 0 is a DIFFERENT, even-clearer artefact

`wave4_agentic.py` *does* fix the harness shape: it adds an **incremental
`emit_invariant`** tool (one invariant per call, appended to a `catalogue` list, lines
86-115) and gates AFTER. That is the right structural fix for the h7 limitation. But the
committed result JSONs show the run failed for an unrelated reason:
`n_emitted: 0`, `agent_error: ""` (empty), `wall_sec: 7.4` (llama-70b) / `7.7`
(qwen2.5-coder:14b). Seven seconds and zero emissions with no recorded error is NOT
synthesis-blindness - it is the agent terminating almost immediately. Consistent with
the documented tool-contract bug: the model calls `grep_source(filename=..., context_lines=...)`
but the tool's signature is `grep_source(pattern)` only (line 71) - LangChain's
`create_react_agent` returns the tool-arg validation error to the model as an
observation; the model retries with the same malformed kwargs and the ReAct graph
dead-ends without ever reaching `emit_invariant`. (Note `langgraph` is not importable in
the bare interpreter here; these runs used a venv.) The empty `agent_error` + 7s wall +
`invariants: []` catalogue are the fingerprint of an early tool-contract dead-end, not a
model-capability result. **Wave-4b currently tests nothing about agentic synthesis.**

---

## 4. Recommendation: how to get a CORRECT agentic result

Adopt **the recent LangGraph harness, fixed** - do NOT revive `agentic_tool_harness.py`.
Rationale: the recent harness already (a) targets the current cells + runtime gate,
(b) has the structurally-correct **incremental `emit_invariant`** that the h7 limitation
demanded, (c) carries kwargs-emission. The prior harness is all-at-once + reference-
scoring + old paths - adapting it would mean rebuilding exactly what `wave4_agentic.py`
already is. So: fix the recent harness's tool contract, then run the model contrast.

### Tool set (the contract to lock)

1. `list_source_files()` - keep.
2. `read_source(path, start_line, end_line)` - keep.
3. `grep_source(pattern, ...)` - **FIX THE CONTRACT.** Either (a) widen the signature to
   absorb the kwargs models actually emit (`grep_source(pattern, file=None, context_lines=0)`),
   or (b) keep it single-arg but make the docstring schema explicit and have the tool
   tolerate/ignore extra kwargs rather than error. (a) is better: a `file` filter and
   `context_lines` are genuinely useful and models clearly expect them. This single fix
   is what unblocks wave-4b.
4. `emit_invariant(...)` - **keep; this is the load-bearing fix vs h7.** One invariant
   per call, appended to harness state. This is what converts the all-at-once burden
   into incremental synthesis. Make the schema forgiving (accept `match_fields_json`
   as object OR string; default unconstructible kwargs to null with a clear note).
5. `gate_probe(...)` (4c only) - keep, but treat as the EXPENSIVE arm: one container
   gate call per probe IS the per-bump CI cost, and that cost is itself the finding.
   Budget-cap it (e.g. <= 20 probes/cell) so a single cell cannot blow the CI budget.

Do NOT re-add an all-at-once `finalise`. Termination should be "model says DONE" or a
step budget; the catalogue is whatever `emit_invariant` accumulated. (Optionally add a
trivial `done()` tool so termination is an explicit tool call, avoiding the prose-vs-
tool-call ambiguity that the ReAct prompt already fights.)

### Models to test (CI-affordable, on the containerized ollama :11435)

Anchor on the contrast that makes the h7 re-diagnosis falsifiable:

- **llama3.1:70b** - REPLICATE the h7 baseline on the FIXED incremental harness + the
  current cells. If it now emits > 0 invariants, that directly demonstrates h7's 0 was
  the all-at-once artefact, not a model wall. This is the single most important run.
- **devstral:24b** - the agentic-tuned datapoint (built for tool-use/ReAct). The
  hypothesis is that an agentic-tuned mid model is the sweet spot for *driving the tool
  loop* even if a code-tuned model is better at raw extraction. Primary new candidate.
- **qwen2.5-coder:32b** - the wave-3 extraction winner (code-tuned, reached the cross-
  field tail in det-then-llm-extend). Tests whether the wave-3 extraction champion also
  wins in agentic shape, or whether agency needs agentic-tuning (devstral) instead.
- Optional secondary: **qwen3-coder:30b** (MoE 30B-A3B cost-frontier datapoint) and
  **command-r:35b** (tool-tuned, RAG-oriented) - run only if the three above leave the
  agentic-tuning-vs-code-tuning question open.

Keep it cheap: 2 cells (vllm 0.19.1, tensorrt 1.2.1) x {llama-70b, devstral-24b,
qwen-32b} x {4b extract-only}. That is 6 cells, all local GPU, sequential on one ollama
so wall-sec stays a clean cost signal - directly comparable to the wave-1/2/3 cost
tables. Run 4c (`gate_probe`) only on the single best 4b model on ONE cell first,
because per-probe container gating is the expensive arm; expand only if 4b shows the
agentic shape is competitive at all.

### Decision gate for the agentic wave

- If FIXED-harness llama-70b emits > 0 and devstral/qwen-32b reach gate-confirmed recall
  in the ballpark of wave-3's det-then-llm-extend at similar cost -> agentic is a viable
  shape; h7's 0 is confirmed as an artefact; carry agentic into the assembly x call-shape
  design space.
- If even the FIXED incremental harness yields ~0 across the 2026 roster -> THEN the
  "agentic synthesis is hard for local models" finding becomes robust (it currently is
  not), and single-shot/chunked (4a) + det-then-llm-extend stays the recommended shape.

This keeps the spend bounded (6 local-GPU cells) while making the h7 re-diagnosis
empirically falsifiable - which the current wave-4b run, broken by the tool-contract bug,
has not yet done.

---

## Cross-references

- Prior taxonomy: `WAVE2_WORKFLOWS.md` (W-A..W-G), `WAVE2_PRIMITIVES.md` (axes).
- Prior experiments: `findings/hybrid_experiments/{h2_h3_h9_batch_summary.md,
  h7_agentic/h7_summary.md, h4_modify_miner/h4_summary.md}`.
- Recent waves: `findings/study/PHASE1_WAVE{1,2,3}_FINDINGS.md`, `PHASE1_WAVE4_PREREG.md`.
- Recent scripts: `scripts/phase1/wave{1,4_pure,4_agentic}.py`, `scripts/wave2_llm_{source,cells}.py`.
- Prior framework: `scripts/strategies/{agentic_tool_harness.py, llm_extractor.py,
  hybrid_extractor.py, prompts.py, *_chunker.py, run_*.py}`.
