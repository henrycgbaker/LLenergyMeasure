# Phase 4: Codebase Audit — Context

## Phase Goal

Thoroughly audit the codebase to identify stubs, dead code, unimplemented features, over-engineering, and gaps between planning documents and actual implementation. The audit drives Phase 5 (Refactor & Simplify) with actionable findings.

## Key Decisions

### Audit Scope & Depth

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Scope breadth | **Everything deep** — src/, tests/, configs/, docs/, docker/ | Full audit before major refactor; no blind spots |
| Unwired features | **Flag as 'unwired'** — mark features that exist but aren't connected end-to-end | Phase 5 decides whether to wire up or remove; audit doesn't prejudge |
| Documentation | **Flag stale docs** — check if CLAUDE.md/README.md files match current code | Docs grew across 10+ phases; freshness unknown |
| Planning doc cross-reference | **Yes** — compare every feature promised in Phases 1-3 against implementation | Ensures nothing was lost in translation |

### Finding Classification

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Severity scheme | **3-tier: remove / simplify / keep** | Clean decision tree; each finding gets a clear action |
| Simplicity benchmark | **Research tool norms** — compare against lm-eval-harness, nanoGPT benchmarks, ML experiment frameworks | Research tools tend to be lean and direct; right comparison class |
| Coherence tracking | **Dedicated category** — track naming inconsistencies, mixed patterns, style drift | Coherence is a primary goal alongside simplification |
| Historical tracing | **Trace origins** — note which phase introduced each finding | Helps Phase 5 understand what can safely change and why it exists |

### Output Format & Actionability

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Report format | **Single audit report** — one comprehensive document with sections per module | Easy to review as a whole; user reviews before Phase 5 acts |
| Actionability | **Findings + Phase 5 skeleton** — grouped tasks ready for planner to refine | Bridges audit → execution without a gap |
| Reference granularity | **Module-level** — reference module/directory, not exact line numbers | Stable across code changes; enough to locate issues |
| Metrics | **Incidental** — capture metrics as they emerge but don't optimise for counting | Useful context, not the driving force |

### Verification Method

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Completeness check | **Both** — module checklist + planning doc cross-reference | Belt and suspenders; no blind spots |
| Test quality | **Full audit** — check for tests with no real assertions, tests for removed features, missing tests | Tests that always pass give false confidence |
| Execution path tracing | **Trace all paths** — map each path from CLI entry to result output | Dead branches, unreachable code, path-specific bugs surfaced |
| Review flow | **Present findings, user decides** — section-by-section review before Phase 5 | User maintains control over what gets changed |

### Simplification Philosophy

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Coherence target | **User experience** — primarily about what users interact with (CLI, config, output) | Internal architecture serves the user experience |
| CLI surface assessment | **Audit should evaluate** — compare against similar research tools and recommend | No preconceptions; let evidence drive recommendations |
| Detection system unification | **Yes, specific analysis** — evaluate whether docker_detection, backend_detection, env_setup overlap or conflict | Multiple detection systems added across phases may have redundancy |
| Breaking changes tolerance | **Break freely** — pre-v2.0.0, no backwards compatibility needed | Simplify aggressively, rename anything, restructure everything |
| SSOT introspection | **Keep** — right level of abstraction for research-grade correctness | Prevents config drift; worth the complexity |
| Auto-detection philosophy | **Auto-detect but show what happened** — user sets preferences in config, tool acts on their behalf while printing decisions | Transparency over magic; initial config captures preferences |
| Functionality reduction | **Primarily simplify internals** — can recommend removing functionality if benchmarking shows we're doing too much, but core focus is keeping functionality while simplifying what happens under the hood | Research tool comparison may surface features that similar tools don't bother with |
| **Industry standard authority** | **Defer to similar research tools** — when in doubt, do what lm-eval-harness / vLLM benchmarks / DeepSpeed benchmarks / nanoGPT / MLPerf do | User defers to industry norms; our tool should not be an outlier in complexity or approach |

### Research-Driven Audit Methodology

**CRITICAL: The audit must be heavily research-informed.** Before evaluating any aspect of the codebase, research how comparable tools handle the same concern. Use extensive web search through similar tools' GitHub repositories, documentation, and codebases.

**Reference tools to research (search their repos, docs, and code):**
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — industry standard for LLM evaluation; how do they handle config, CLI, backends, Docker?
- [vLLM benchmarks](https://github.com/vllm-project/vllm/tree/main/benchmarks) — how does vLLM's own benchmarking work? CLI surface, config format, result output
- [DeepSpeed benchmarks](https://github.com/microsoft/DeepSpeed) — how do they handle multi-backend, config, campaign-like workflows?
- [nanoGPT](https://github.com/karpathy/nanoGPT) — minimal research tool; what's the simplest viable approach?
- [MLPerf inference](https://github.com/mlcommons/inference) — formal benchmarking standard; how do they structure config/execution?
- [LLMPerf](https://github.com/ray-project/llmperf) — LLM performance benchmarking; CLI, config, output patterns
- [GenAI-Perf](https://github.com/triton-inference-server/perf_analyzer) — NVIDIA's inference benchmarking tool

**For each audit area, the research should answer:**
1. How do similar tools handle this? (CLI commands, config format, execution model, Docker usage, result output)
2. What's the typical complexity level for this feature in research tools?
3. Are we doing something no similar tool does? If so, is it justified or over-engineering?
4. What patterns are industry standard that we should adopt?

**The audit should produce concrete "industry does X, we do Y" comparisons** — not vague "this seems complex". Evidence-based recommendations grounded in what actually ships in comparable tools.

### Core Workflow Definition

The audit should evaluate the codebase against this intended core workflow:

```
1. Generate configs (programmatic grid expansion → valid YAML files)
2. Run experiments via dedicated backends (each in Docker containers)
3. Record raw results
```

**Key workflow decisions:**
- **Campaign-centric**: The campaign (series of experiments) is the core entry point, not single experiments
- **Config generation should be separate from execution**: Generate config files first, then feed to campaign runner (currently grid expansion is embedded in campaign YAML)
- **Experiment vs Campaign**: Audit should evaluate whether these should merge into one command. If merged, keep the `experiment` name with campaign functionality
- **Results analysis is secondary**: Downstream analysis (aggregation, comparison) is nice-to-have if not confused/bloated — raw recording is the core
- **All backends Docker?**: Audit should evaluate whether unifying all backends (including PyTorch) to Docker-only would simplify the execution model (currently PyTorch runs locally, vLLM/TensorRT in Docker)
- **Backend quality over quantity**: Audit each backend's integration completeness — not just importable but actually functional end-to-end

### What the Audit Should Specifically Investigate

1. **Execution path complexity**: How many distinct paths exist from CLI entry to result output? Which are dead? Which overlap?
2. **Detection system unification**: Can docker_detection + backend_detection + env_setup + container strategy selection be unified or reduced?
3. **CLI command surface**: Compare against lm-eval-harness, nanoGPT, etc. — are we offering commands that research tools don't need?
4. **Config generation separation**: Is grid expansion cleanly separable from campaign execution?
5. **Backend completeness**: Does each backend (PyTorch, vLLM, TensorRT) actually work end-to-end, or are there silent failures?
6. **Docker-only model**: Would moving all backends to Docker simplify the local/Docker execution split?
7. **Archaeological layers**: Patterns that made sense in Phase X but are now redundant due to later decisions

### Deferred Ideas (Out of Phase 4 Scope)

- Renaming `lem` → `llem` (Phase 5 scope)
- Actual removal/refactoring of code (Phase 5 scope)
- Parameter expansion (Phase 6 scope)
- New features or capabilities (future phases)

Phase 4 produces the audit report. Phase 5 acts on it.

## Constraints

- Audit is read-only — no code changes in Phase 4
- Report is presented to user section-by-section for review before Phase 5
- **Research-first methodology** — every audit area must include web search of similar tools' codebases before making recommendations. "Industry does X, we do Y" is the required format for simplification recommendations. Do not recommend changes based on opinion alone.
- Every module/directory must be explicitly checked off (no blind spots)
- Every feature from Phases 1-3 ROADMAP must be cross-referenced against implementation
- When the audit finds something questionable, the default question is "do similar research tools do this?" — if no, recommend removal/simplification unless there's a clear justification specific to energy measurement
