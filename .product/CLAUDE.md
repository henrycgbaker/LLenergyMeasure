# Claude Code Instructions — Product Design

**This directory is the upstream SSOT** for all LLenergyMeasure product decisions.
`.planning/` phases and GSD execution are downstream — they implement what is decided here.

---

## Directory Structure (ADR Pattern)

This directory follows the **Architecture Decision Record (ADR)** pattern (Nygard 2011),
extended with a research layer and separate design specs.

| Directory | Analogy | Question answered | What goes here |
|-----------|---------|------------------|----------------|
| `research/` | Evidence file | "What do peers/papers do? What's the data?" | Full peer codebase code, benchmark numbers, API docs, quotes from papers. **Verbose is fine.** |
| `decisions/` | ADR | "What did we decide, why, and what did we reject?" | Decision statement, rationale, rejected alternatives + why, constraints, trade-offs. **Full discussion.** |
| `designs/` | Technical Design Doc | "How do we implement this decision?" | Pydantic models, YAML schemas, code stubs, call graphs, diagrams. |

**The test for where something belongs**:
1. Evidence from an external source → `research/`
2. What we chose, why we chose it, what we rejected → `decisions/`
3. How to implement it in code or config → `designs/`

---

## DRY Rule

**Never duplicate content across directories. Cite instead.**

- If evidence lives in `research/`, `decisions/` cites it — does not copy it
- If rationale lives in `decisions/`, `designs/` references it — does not re-explain it
- If the same peer example appears in two docs, it belongs in `research/` only

---

## What Each Directory Contains (and Doesn't)

### `research/`

**Contains**: Raw evidence. Peer codebase code verbatim. Benchmark numbers. Literature quotes. API documentation. Confidence ratings. Comparison tables of peer tools.

**Does not contain**: Our decisions. Our design choices. Anything about how we will implement things.

### `decisions/`

Full ADR (Architecture Decision Record) style. Each file covers one decision area and must
capture the **debate** (what was considered and why rejected) alongside the outcome.

**Required sections (all must be present; use `N/A` where content doesn't exist):**

```
## Context
What situation prompted these decisions? What constraints/forces shaped the decision space?

## Considered Options
For each decision: list ALL alternatives with honest pros/cons for each option.
This is the debate section — not just the winning option.

| Option | Pros | Cons |
|--------|------|------|
| **[Chosen — bold]** | ... | ... |
| Option B | ... | ... |

## Decision
[Active voice: "We will..." or "We chose..."]
Rationale: [Why this option over the others.]

## Consequences
Positive: ...
Negative / Trade-offs: ...
Neutral / Follow-up decisions triggered: ...

## Related
[links to designs/, research/, other decisions/]
```

**Header block (top of every file):**
```
**Status:** Accepted | Proposed | Deprecated | Superseded by [link]
**Date decided:** YYYY-MM-DD
**Last updated:** YYYY-MM-DD
**Research:** [../research/NN-file.md](link) | N/A
```

**Files with multiple sub-decisions** (e.g. K1/K2/K3) repeat Considered Options + Decision
per sub-decision. Add a brief overall Context at the top of the file.

**Status vocab:** `Accepted` | `Proposed` | `Deprecated` | `Superseded by [link]`
(Retire the old `Confirmed`/`DECIDED`/`DRAFT` vocabulary — use the ADR standard terms.)

**Contains**: Context. Full debate (all options with pros/cons). Decision rationale. Consequences.
Rejected alternatives with reasoning. Brief peer tool references ("lm-eval uses X").

**Does not contain**: Implementation code (Python functions, Pydantic models). Verbatim peer
codebase code. YAML field definitions. Anything about *how* to implement the decision.

### `designs/`

Technical Design Doc style. Each file covers one implementation area.

**Contains**: Pydantic model definitions. YAML field schemas. Python code stubs. Call graphs. Diagrams. Algorithm pseudocode. File/module structure.

**Does not contain**: The "why" behind decisions (cite `decisions/` instead). Raw peer analysis (cite `research/` instead). Repeated rationale.

---

## Reference Requirements

Every decision and design proposal must be grounded in evidence:
- At minimum: one peer codebase, paper, or authoritative source
- Preferred: cite the `research/` file that contains the full analysis
- Do not assert "X is the best approach" without a citation

**Research-first on uncertainty**: Whenever anything is unclear — a design choice, a technical approach, peer tool behaviour — do fresh research first. Add findings to `research/` (its SSOT). Then weave citations through `decisions/` and `designs/`. Do not guess from general knowledge without checking real peer implementations.

---

## Working Principles

1. **Always confirm decisions with the user** before writing any design or code.
   The facilitated decision model: present options + evidence → user decides → write the doc.

2. **Internal consistency first.** Surface contradictions to the user; never resolve them silently.

3. **No backwards compatibility with v1.x plans.** Old ROADMAP.md phases and CLI commands
   are superseded. All implementation works from this directory.

4. **Never stale**: update docs when decisions change. Record date of change and reason.

5. **Preserve the record**: This directory is a **cumulative decision log**, not a wiki.
   Never delete or overwrite rejected/superseded content — annotate it in-place:
   - Superseded design: add `> **Superseded YYYY-MM-DD:** [brief reason + link to new decision]`
     immediately before the outdated section.
   - Rejected alternative: add `**Rejected (YYYY-MM-DD):** [reason]` inline below the option.
   - The reasoning for rejections is as valuable as the accepted decision — it prevents
     re-litigating the same ideas in future sessions.

---

## What NOT to use (Stale)

- `.planning/PROJECT.md` — wrong command names, old CLI shape, campaign terminology
- `.planning/ROADMAP.md` — old phase scope, stale feature lists
- `.planning/phases/` — historical record only, not for implementation guidance

---

## Key Confirmed Decisions (Quick Reference)

- **CLI**: 2 commands + 1 flag — `llem run`, `llem config`, `llem --version`
- **Library**: `run_experiment(ExperimentConfig) -> ExperimentResult`, `run_study(StudyConfig) -> StudyResult` — stable from v2.0.0 via `__init__.py`
- **Config**: Composition — `ExperimentConfig` with optional backend sections (`pytorch:`, `vllm:`, `tensorrt:`)
- **Sweep**: Dotted notation — `pytorch.batch_size: [1, 8]`
- **Results**: `ExperimentResult`; `measurement_config_hash`; `StudyResult`
- **Runner**: Docker-first when available; `local` fallback with nudge; per-backend config via `runners:` YAML; env var override `LLEM_RUNNER_{BACKEND}=docker:image`
- **Errors**: `LLEMError` hierarchy; exit codes 0/1/2/130
- **API stability**: `__init__.py` exports only; one minor version deprecation window

See `decisions/` files for full rationale.
