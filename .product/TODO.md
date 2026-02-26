# TODO: Finalising the Product Redesign

This is the work plan for getting from "decisions and designs exist" to
"Phase 5 can be planned and implemented". Complete in order.

---

## 0. Manual Inspection + Feedback Pass

**Owner: User**

Read through the entire `decisions/` and `designs/` directories manually.
Leave inline feedback, questions, and TODOs directly in the files.

Key areas to inspect:
- `decisions/cli-ux.md` — command design, flags, zero-config UX
- `designs/study-yaml.md` — sweep grammar, execution block, full schema
- `designs/experiment-config.md` — field list, backend sections, removals
- `designs/result-schema.md` — ExperimentResult shape, new fields
- `designs/library-api.md` — public API surface, output_dir sentinel pattern
- `decisions/architecture.md` — library structure, separation of concerns
- `decisions/future-versions.md` — version boundary review (v2.0 scope may expand)

**Specifically flag:**
- Anything that feels wrong or underspecified
- Scope questions (should this be v2.0 or later?)
- Consistency issues between docs
- Missing areas not yet documented

---

## 1. Scope Out Missing Decision & Design Areas

**Owner: Claude + User (facilitated)**

After the manual inspection, identify decision areas not yet covered.
Current candidates (not yet confirmed as gaps — verify during inspection):

- [ ] Streaming inference handling (TTFT instrumentation, how to surface per-phase metrics)
- [ ] LoRA adapter support — still in scope for v2.0? How does it interact with new ExperimentConfig?
- [ ] Tensor parallelism / multi-GPU — how does this fit in the new design?
- [ ] FLOPs estimation — methodology, fallback chain, how it surfaces in ExperimentResult
- [ ] Warmup strategy — CV-based convergence, how configured in ExperimentConfig
- [ ] Interactive backend selection (zero-config mode — how does `llem run --model X` prompt for backend?)
- [ ] Dataset handling for HPC / air-gapped environments (pinned dataset can't be downloaded)
- [ ] Pre-flight check design — what exactly is checked, in what order, error format
- [ ] `llem config` output design — what exactly does it print?
- [ ] Experiment ID / provenance — how are experiments identified across runs?

---

## 2. Internal Consistency Review

**Owner: Claude + User (every decision validated)**

Systematically review all decisions and designs for contradictions and gaps.
The product has developed across 6+ sessions — inconsistencies are expected.

Process:
1. Claude reads all docs and surfaces suspected inconsistencies
2. For each: present the conflict clearly, propose resolution
3. User confirms resolution
4. Claude updates the relevant docs

Known inconsistency candidates:
- [x] Command name in some older docs may still say `llem experiment` (not `llem run`) — fixed 2026-02-19
- [x] `llem status` may still appear in some docs (should be `llem config`) — fixed 2026-02-19
- [ ] Some docs may reference the old 9-command CLI
- [ ] `output_dir` sentinel pattern — verify consistent across library-api.md and result-schema.md
- [ ] Group 7 decisions (release-process.md, documentation-strategy.md) — PENDING CONFIRMATION
- [ ] Draft decisions (result-schema-migration.md, reproducibility.md, access-control.md,
  additional-backends.md, hpc-slurm.md, study-resume.md, local-result-navigation.md) — need user review
- [ ] Future version boundary question: which v2.1–v2.3 features belong in v2.0?

---

## 3. Update `.planning/` Root Documents

**Owner: Claude (writing), User (confirmation)**

Once the redesign is internally consistent and comprehensive:

Rewrite from scratch (do NOT edit the current versions — they are too stale):
- [ ] `.planning/PROJECT.md` — what it is, who it's for, requirements, CLI commands,
  package structure, key decisions, versioning
- [ ] `.planning/ROADMAP.md` — milestones (v2.0–v4.0), phase descriptions, success criteria.
  Phase 5+ entirely replanned. Phase numbering may change.
- [ ] `.planning/STATE.md` — current position, next action

**This is MAJOR disruption to existing plans.** The old phase structure (phases 5–10
as currently described) will be entirely replaced. Existing phase descriptions are
based on the old 15-command CLI and campaign-based architecture.

---

## 4. Update Root CLAUDE.md

**Owner: Claude**

Once `PROJECT.md` and `ROADMAP.md` are rewritten:

- Update root `CLAUDE.md` to reference the new decisions and designs
- Add modular references to key decision files (e.g. "CLI design: see decisions/cli-ux.md")
- Remove stale architectural guidance that contradicts the redesign
- Add pointer to `.planning/product/redesign-planning/` as the SSOT

---

## 5. Replan Phase 5 Onwards with GSD

**Owner: Claude + GSD framework**

Once all above is done:

```
/gsd:plan-phase 5
```

Phase 5 will be planned against:
- `decisions/` and `designs/` in this directory
- Rewritten `ROADMAP.md` and `PROJECT.md`
- The 4 P0 bugs from the codebase audit (`.planning/phases/04-codebase-audit/AUDIT-REPORT.md`)
- The preservation audit (`preservation_audit/` in this directory) — 47 features

The GSD planner will break Phase 5 into waves of parallel work. Likely scope:
- Library restructure (`__init__.py`, public API, module layout)
- CLI 15 → 3 commands (new Typer app, remove dead commands)
- ExperimentConfig cleanup (field removals and renames)
- P0 bug fixes
- Dead code removal (1,524 lines)
- Result schema updates (ExperimentResult rename, new fields)
- Tests at v2.0 library boundary

---

## Progress Tracker

| Step | Status | Owner |
|------|--------|-------|
| 0. Manual inspection | Not started | User |
| 1. Missing decision areas | Not started | Claude + User |
| 2. Consistency review | Not started | Claude + User |
| 3. Rewrite root planning docs | Not started | Claude |
| 4. Update root CLAUDE.md | Not started | Claude |
| 5. Replan Phase 5+ with GSD | Not started | Claude + GSD |
