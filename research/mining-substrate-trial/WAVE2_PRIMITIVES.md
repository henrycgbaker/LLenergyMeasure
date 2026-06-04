# Wave 2 primitives inventory

**Status:** Locked 2026-06-05 (v1; will be refined as evidence comes in).

The decision landscape Wave 2 characterises is a cross-product of these axes. Each axis lists levels worth probing. An "experiment" is a single point in this multi-dimensional space, run on a (task, engine, version) triple, producing per-task recall + precision + cost + failure modes.

Comprehensive ≠ exhaustive. We characterise every axis but only probe interaction effects where there's reason to expect them.

## Axis 1: Substrate primitive (what reads the engine source)

| Level | What it reads | Cost shape |
|---|---|---|
| `hand-walker` | Engine source via handwritten AST walker (Wave 1 baseline; current main) | Free per-bump; expensive per-version-vendoring |
| `tree-sitter-universal` | Engine source via tree-sitter queries | Free per-bump; near-zero per-version |
| `framework-reflection` | Engine via `Model.__fields__` / `dataclass.fields` / `msgspec.json.schema` | Near-zero per-bump; needs full engine importable |
| `runtime-trace` | Engine via monkey-patched validators + synthetic config perturbations | Medium per-bump; needs full engine importable |
| `behavioural-fuzz` | Engine via hypothesis-style fuzzing | Medium per-bump; needs full engine runnable |
| `pyright-stubs` | Generated type stubs (no bodies) | Cheap per-bump; pyright runtime needed |
| `sphinx-xml` | Engine documentation XML | Cheap; only works if engine ships Sphinx |
| `rag-over-source` | Vector-DB index of source | Expensive setup; cheap per-bump |
| `llm-reads-raw` | LLM reads source directly (Wave 1 strategy `b`) | Expensive per-bump (LLM tokens) |

## Axis 2: LLM role (what the LLM is asked to do)

| Level | Wave 1 evidence | Notes |
|---|---|---|
| `extract` (propose) | Substrate-ceiling-bound at 70B-q4 (~50%) | Most common role |
| `diagnose` (categorical gaps) | Excellent at 70B-q4 (0 fabrications across 8 diagnoses) | Cheap; very promising |
| `patch-code` (writes producer/walker patches) | Poor at 70B-q4 (0/3 patches lifted recall) | Heavily model-scale dependent |
| `gate` (yes/no on candidates) | Untested in Wave 1 | Decision-relevant for safety |
| `decide` (which strategy applies / merge conflicts) | Untested | Useful in workflow assembly |
| `curate` (ongoing knowledge maintenance) | Untested | The "LLM-as-maintainer" pattern |
| `subtract` (drop entries) | Error-prone (3/3 vllm drops were false-drops) | Untrusted role |

## Axis 3: Workflow assembly (how primitives compose)

| Level | Shape | Wave 1 status |
|---|---|---|
| `det-only` | Substrate produces catalogue; gate validates | Baseline |
| `llm-only` | LLM produces; gate validates | Tested (strategy b) |
| `det-then-llm-extends` | Substrate baseline + LLM adds | Tested (d-ab); strong |
| `llm-then-det-validates` | LLM proposes; gate filters | Tested (H3); strongest precision lift |
| `closed-loop-feedback` | LLM emits; gate rejects subset; LLM re-emits | Untested but scaffolded (h15) |
| `det-then-llm-then-det` | Substrate baseline; LLM extends; gate validates | Wave 1 Architecture II |
| `llm-ensemble` | N LLMs propose; vote; gate validates | Untested |
| `llm-self-consistency` | Same LLM k runs at non-zero temp; vote | Untested but scaffolded (h11) |
| `det-ensemble` | N substrates propose; union; gate validates | Implicit in validated-union scoring |
| `det-then-llm-patches-det` | Substrate fails; LLM patches substrate; rerun | Tested (H4); failed at 70B-q4 |
| `agentic-tool-use` | LLM drives tools to explore + emit | Tested (H7); collapsed at 70B-q4 |

## Axis 4: Model scale (when LLM is involved)

| Level | Examples | Per-cell wall (rough) | Per-cell $ |
|---|---|---|---|
| `small` | Qwen-Coder-7B fp16, DeepSeek-Coder-V2-Lite | 5-15 min | $0.01-0.05 |
| `medium` | Qwen-Coder-32B fp16, Llama 3.1-8B fp16, Phi-4-14B | 15-30 min | $0.05-0.20 |
| `large` | Llama 3.3-70B fp16, Llama 3.1-70B q4, Mixtral 8x22B q4 | 30-90 min | $0.20-1.00 |
| `xlarge-OSS` | DeepSeek-Coder-V2-236B q4 | 60-120 min | $1-2 |
| `frontier-API` (deferred) | Claude Sonnet 4.6/4.7, Opus | 5-15 min API | $1-5 |

Wave 2 covers small + medium + large at 4xA100. xlarge-OSS for benchmark only. Frontier-API deferred to Wave 3.

## Axis 5: Call shape

| Level | What changes |
|---|---|
| `single-shot t=0` | Deterministic single LLM call |
| `single-shot t>0` | One call at non-zero temperature (variance probe) |
| `k-vote t>0` | k=3 or k=5 calls at t>0; aggregate by majority |
| `chunked-flat` | Source split into per-class chunks; one call per chunk |
| `chunked-cumulative` | Chunked + previous context carried |
| `multi-step-chained` | extract -> verify -> revise (no tools) |
| `tool-mediated-agentic` | LLM drives tools to read / probe / emit |

## Axis 6: Task

- `schema` - enumerate fields/types/defaults.
- `invariants` - extract validation predicates.
- `invalid-configs` (provisional) - corpus of known-bad config tuples.

Per-axis behaviour differs sharply. A primitive that wins on schema may lose on invariants (tree-sitter probe already showed this).

## Axis 7: Engine

- `transformers` - Python class hierarchy + Sphinx docs + per-quantizer fan-out
- `vllm` - dataclasses + msgspec + sprawling EngineArgs + env-var surface
- `tensorrt-llm` - Pydantic + NVIDIA-specific runtime gates + plugin config tree
- (later) `sglang`, `lmdeploy` - external validity, Wave 3

## Axis 8: Version situation

- `active-extract` - run substrate on the currently-pinned version (no bump involved)
- `bump-extract` - run substrate on a fresh upstream-bumped version (the CI scenario)
- `bump-update` - run substrate + propose update to producer / catalogue (the self-update test)

## What gets characterised per cell

Every experiment cell records:

- Per-task recall + precision against ground truth + against validated-union (both, for cross-Wave comparison).
- Wall-clock + GPU energy + estimated $.
- Failure mode tag (`silent / detectable / crash / hallucinated-from-empty / under-emit / over-emit / gate-rejected-most-output`).
- Self-update binary: did the workflow produce a usable updated artefact without human intervention?
- Hallucination rate: gate-rejected entries / total proposed (pure-LLM safety signal).
- Source-citation rate (LLM cells): proportion of entries with verifiable source-line citation.
- Observations free-text: anything noticed in passing worth carrying to synthesis.

## What gets characterised per axis (not per cell)

After running the cells:

- **Cost-recall frontier per task** with each substrate primitive's curve.
- **Marginal cost of LLM** in each assembly shape (how much recall per $ does adding the LLM step buy).
- **Substrate complementarity**: which substrate-pairs catch disjoint entries (i.e. union is much larger than max).
- **Bump-survivability**: which primitives degrade gracefully under bump-extract and bump-update.
- **Failure-mode interactions**: which assembly shapes mask vs surface which failure modes.
- **Self-update success rate per workflow shape** across (engine, bump-pair) combinations.

## Out of scope for Wave 2

- LangGraph dep (use minimal harness for multi-step assemblies)
- SGLang / LMDeploy vendoring + characterisation (Wave 3)
- Claude / GPT API runs (Wave 3)
- Statistical inference (bootstrap CIs, seed-variance) - record point estimates only
- Layer B behavioural validation (use Layer A runtime gate as SSOT)
- Property-based test generation, SMT / Z3 targets

## Cross-references

- `WAVE2_SCOPE.md` - the framing and the production target.
- `WAVE2_PROTOCOL.md` - the experimental protocol (to be rewritten against this primitives inventory).
- `DECISIONS_LOG.md` - chronological narrative.
- `findings/wave2_treesitter_probe.md` - first cell completed (tree-sitter substrate, both tasks).
