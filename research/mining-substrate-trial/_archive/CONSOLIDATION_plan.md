# Consolidation / cleanup plan: mining-substrate-trial study

Audit + tightening plan for the sprawling writeups + scripts + results on
`spike/engine-knowledge-as-data`. Authored 2026-06-10. Nothing is deleted by this
doc; it is an executable proposal. The draft canonical findings doc is
`CANONICAL_FINDINGS.md` (companion).

## TL;DR

There are TWO layers tangled together: (1) the **Round-0/0b deterministic-baseline
+ bump-robustness study** (the older "5-version-window" program: schema/invariant
cost frontier + the 15-cell major-vs-minor gradient), and (2) the **Phase-1 LLM
pattern waves 1-4** (kwargs lever -> tier 2x2 -> construction-grounding). Each layer
has its own per-wave docs PLUS a synthesis PLUS a results tree. The redundancy is:
prereg-vs-findings doc pairs, three competing top-level "results/synthesis" docs,
two synthesis docs for the LLM waves, and ~3MB of reproducible LLM intermediates
(`_raw.txt`, `_corpus.yaml`) committed alongside the load-bearing `.json`/GT. The
README still describes the long-dead PoC trial, not the current study.

Recommended end state: ONE canonical findings doc per layer, per-wave details kept
but clearly marked "detail", intermediates gitignored, the PoC corpus archived, and
the README rewritten to point at the canonical docs.

---

## 1. Inventory + redundancy map

### 1A. LLM-wave writeups (`findings/study/`) - the core of this audit

| file | role | verdict |
|---|---|---|
| `CANONICAL_FINDINGS.md` (NEW, this PR) | one-page waves-1-4 story | **KEEP - new canonical** |
| `STUDY_SYNTHESIS.md` (untracked) | prior one-page waves-1-4 synthesis | **SUPERSEDED by CANONICAL_FINDINGS** - 95% overlap; it was the draft that CANONICAL_FINDINGS replaces. CUT (never committed). |
| `PHASE1_WAVE1_FINDINGS.md` | wave-1 detail (validation bottleneck) | **KEEP as detail** - load-bearing per-wave record |
| `PHASE1_WAVE2_FINDINGS.md` | wave-2 detail (kwargs lever, 8 confirms) | **KEEP as detail** |
| `PHASE1_WAVE3_FINDINGS.md` | wave-3 detail (size x tuning 2x2) | **KEEP as detail** |
| `PHASE1_WAVE4_FINDINGS.md` | wave-4 detail (construction-grounding) | **KEEP as detail** |
| `PHASE1_WAVE1_PREREG.md` | wave-1 pre-registration | **MERGE into findings, then ARCHIVE** - prereg content (locked design point, prompt hash, deviation log) is valuable provenance but each prereg is ~80-150 lines that duplicate the findings' setup. Fold the locked-design + deviations into a short "Pre-registration" section at the top of each `_FINDINGS.md`, archive the standalone prereg. (Or keep the 4 prereg files but move to `_archive/prereg/` - cheaper, preserves them verbatim.) |
| `PHASE1_WAVE2_PREREG.md` | wave-2 prereg | same |
| `PHASE1_WAVE3_PREREG.md` | wave-3 prereg | same |
| `PHASE1_WAVE4_PREREG.md` | wave-4 prereg | same |
| `WAVE4_RECONCILIATION_MAP.md` | maps waves 1-4 onto the prior PoC W-A..W-G / h2-h9; re-diagnoses agentic=0 | **KEEP** - genuinely load-bearing (it is why agentic was dropped and why the chunkers/strategies are stale); referenced by CANONICAL_FINDINGS. Could trim the per-module reuse table once the stale `strategies/` is archived (item 1E). |
| `LLM_PATTERNS_NEXT_SESSION.md` | next-session bootstrap for the LLM-pattern phase | **STALE / ARCHIVE** - written before waves 3-4 ran (HEAD `454a25f4`, pre-wave-3); its "core deliverable: size x shape matrix" and "first steps" are now done/superseded by the actual waves. Keep only if a future session resumes; archive, do not keep at top level. |
| `REVIEW_northstar_strategy.md` (untracked) | adversarial review: "Phase 1 measured the wrong quantity (single-version recall, not cross-bump currency)" | **KEEP - commit it.** This is the single most important strategic caveat in the corpus and is currently UNTRACKED (at risk of loss). It belongs in the record and should be cross-linked from CANONICAL_FINDINGS' caveats. Do not cut. |

### 1B. Round-0/0b baseline + bump-robustness writeups - the OTHER layer

| file | role | verdict |
|---|---|---|
| `FULL_MATRIX.md` | authoritative 15-cell gradient (major 53% vs minor 76-100%); retracts the rebound metric; GT integrity 908/913 REAL | **KEEP - canonical for the bump-robustness layer** |
| `STUDY_RESULTS.md` (top-level) | milestone-record synthesis of the 5-version study (cost frontier + bump robustness) | **KEEP but DEMOTE/RECONCILE** - its Section 4 is explicitly "SUPERSEDED by FULL_MATRIX". Trim the superseded section to a pointer; keep the cost-frontier finding (25%->47% det ceiling) which FULL_MATRIX does not restate. This is the closest thing to a canonical doc for layer (1). |
| `FANOUT_FINDINGS.md` | chronological detail for the baseline layer (schema fan-out + first invariant cells); has a big re-base note | **KEEP as detail / ARCHIVE-candidate** - superseded by FULL_MATRIX for the gradient and by STUDY_RESULTS for the cost frontier; retains chronological provenance + the schema-gate-generalises result. Demote to detail (move under an `_archive/` or keep but stop citing as headline). |
| `ROUND0B_BASELINE.md` | deterministic-baseline primitives + surfacing-recall baseline | **KEEP as detail** - documents the det-floor primitives (the `improved-det-v2` lineage the LLM waves build on); referenced provenance. |

### 1C. Top-level docs (`research/mining-substrate-trial/*.md`)

| file | role | verdict |
|---|---|---|
| `README.md` | dir entry point | **REWRITE - stale.** Describes the dead PoC trial (Phases 1-4, `empirical_trial_outcome.md`, `ANTHROPIC_API_KEY` Phase 3c) and links files that are the old bake-off. Should describe the CURRENT two-layer study and point at `STUDY_DESIGN.md` / `STUDY_RESULTS.md` / `findings/study/CANONICAL_FINDINGS.md`. |
| `STUDY_DESIGN.md` | objective + method + locked params + execution log (Section 15) | **KEEP - canonical design doc.** The pre-registered program spec; load-bearing. |
| `STUDY_RESULTS.md` | see 1B | KEEP (demote Section 4) |
| `RESEARCH_WRITEUP.md` (522 lines) | the predecessor strategy bake-off writeup (LLM-proposes/det-disposes recommendation) | **ARCHIVE** - prior-PoC artefact; its central recommendation is now operationalised by the study. Keep for lineage but move out of the hot path. |
| `DECISIONS_LOG.md` (4185 lines) | running chronological PoC narrative | **ARCHIVE** - the PoC decision log; huge, chronological, superseded by the structured docs. Keep for forensic provenance under `_archive/`. |
| `WAVE2_*.md` (9 files: SCOPE/WORKFLOWS/PRIMITIVES/PROTOCOL/INFRA_SETUP/EXPERIMENT_QUEUE/RESEARCH_OUTCOMES/NEXT_SESSION + EXPERIMENT_QUEUE) | the prior Wave-2 PoC planning/taxonomy corpus | **ARCHIVE as a set.** The W-A..W-G taxonomy + axes (WORKFLOWS/PRIMITIVES) are still cited by WAVE4_RECONCILIATION_MAP, so keep them readable, but they are PoC-era planning docs locked 2026-06-05; move the whole `WAVE2_*` set to `_archive/wave2_poc/`. The reconciliation map's cross-refs are the only live consumers - update those links. |

### 1D. Scripts (`scripts/`)

| group | files | verdict |
|---|---|---|
| **current LLM-wave runners** | `phase1/wave1.py`, `phase1/wave3_tier_sweep.sh`, `phase1/wave3_dump_confirmed.py`, `phase1/wave3_reparse_lenient.py`, `phase1/wave4_pure.py`, `phase1/wave4_pure_sweep.sh`, `phase1/wave4_construct.py`, `phase1/wave4_construct_sweep.sh`, `phase1/wave4_agentic.py`, `phase1/wave4_selfconsistency.py`, `phase1/wave4_multistage.py` | **KEEP** - the live runners. Note `wave4_selfconsistency.py` + `wave4_multistage.py` exist but no committed results yet (self-consistency is the pending strategy) - keep, they are staged. |
| **current shared helpers** | `wave2_llm_cells.py`, `wave2_llm_source.py`, `wave2_runner.py`, `wave2_run_all.py`, `wave2_llm_source.py`, `study_gt_pilot.py`, `round0b/{gate,recall,bump_delta}.py`, `gt_adapter.py`, `gt_scoring.py`, `compute_substrate_analysis.py`, `run_substrate_matrix.py` | **KEEP** - shared gate/score/source primitives the waves import. (`wave2_*` here = the CURRENT study's shared layer, distinct from the `WAVE2_*.md` PoC docs - the naming collision is confusing; consider renaming to `study_llm_*` later, NOT in this cleanup.) |
| **PoC strategies framework** | `strategies/` (everything: `agentic_tool_harness.py`, `llm_extractor.py`, `hybrid_extractor.py`, `prompts.py`, `*_chunker.py`, `claude_extractor.py`, `llm_b_oss.py`, `h4_engine_configs.py`, `run_*.py`, `strategies/wave2/*`, `test_*`) | **ARCHIVE** - per WAVE4_RECONCILIATION_MAP SS2, ALL of `strategies/` is STALE (version-pinned cells, reference-scoring not runtime-gate, old `/tmp` paths) EXCEPT the `LLMBackend`/`OllamaBackend` transport in `llm_extractor.py` and its fence/parse helpers. The current waves already re-implement that inline (`wave2_llm_cells.ollama_generate`). RECOMMENDATION: move all of `strategies/` to `_archive/strategies_poc/`; do NOT port. If a future consolidation wants the OllamaBackend, lift just that class. |
| **PoC trial drivers** | `trial_runner.py`, `trial_aggregate.py`, `trial_scoring.py`, `test_trial_scoring.py`, `run_phase4_0_union.py`, `study_gt_pilot.py`(shared), `rescore_wave1_vs_gt.py`, `venv_setup.py` | **MIXED** - `trial_scoring.py` is still the runtime-gate dispatch the current study uses (`runtime_validate_invariants_dispatch`) - KEEP. `trial_runner.py`/`trial_aggregate.py`/`run_phase4_0_union.py` are PoC drivers - ARCHIVE-candidate. `rescore_wave1_vs_gt.py` is a one-off - ARCHIVE. Audit imports before moving (item 4). |
| **stale build artefact** | `scripts/phase1/__pycache__/wave4_multistage.cpython-312.pyc` | **GITIGNORE + git rm if tracked** (it is untracked here, but add the pattern). |

### 1E. Results dirs (`findings/study/phase1_wave{1,2,3,4}/results/`)

The retention question. Per file-type, within each cell `<wave>_<model>_<engine>_<vslug>`:

| artefact | what it is | retain? |
|---|---|---|
| `*.json` | gate-scored result (verdict counts, recall, precision, GT-growth keys, wall) | **KEEP - load-bearing.** This is the actual result; small (~1KB each); 33 files. |
| `*_CONFIRMED.yaml` | the re-gated confirmed-invariant detail (the verified-real list) | **KEEP - load-bearing** (8 files; the auditable confirms). |
| `*_corpus.yaml` | the parsed/deduped LLM proposals + kwargs (intermediate) | **DROP from git, gitignore.** 25 files, ~1.3MB. Reproducible from `_raw.txt` + the parser; an intermediate. Keep on disk, stop committing. (Borderline: corpus is the parsed candidate set; if you want ONE auditable corpus per headline cell, keep just the qwen2.5-coder:32b construct corpus and gitignore the rest. Otherwise drop all.) |
| `*_raw.txt` | raw LLM stdout before parsing | **DROP from git, gitignore.** 24 files, ~1.7MB. Pure reproducible intermediate; never cited. |
| `phase1_wave{1,2}/llm_proposed/*.yaml` | the gated Opus proposals (waves 1-2) | **KEEP** - these are the canonical Opus outputs (small, 4 files), cited by the wave-1/2 findings as the committed canonical runs. |
| `phase1_wave2/results/wave2_armA_confirmed_*.yaml` | confirmed lists | **KEEP** (load-bearing, like CONFIRMED.yaml). |

Net: committing ~3MB of `_raw.txt` + `_corpus.yaml` (49 files) is the single biggest
bloat. They are reproducible intermediates. Drop + gitignore.

### 1F. Untracked-and-at-risk (commit decisions)

- `findings/study/CANONICAL_FINDINGS.md` - NEW, commit.
- `findings/study/CONSOLIDATION_plan.md` - this doc; commit or keep local per your call.
- `findings/study/REVIEW_northstar_strategy.md` - **COMMIT** (load-bearing caveat, see 1A).
- `findings/study/STUDY_SYNTHESIS.md` - **do NOT commit; CUT** (superseded by CANONICAL_FINDINGS).
- `phase1_wave4/results/w4c_qwen2_5_32b_vllm_v0_19_1.json` + `_corpus.yaml` - the
  in-flight 70B-vs-32B head-to-head outputs; the `.json` is a result (keep once the
  run finishes + is folded into wave-4 findings), the `_corpus.yaml` follows the
  gitignore rule (drop).

---

## 2. Proposed tightened structure (the canonical set)

```
research/mining-substrate-trial/
  README.md                      # REWRITTEN: two-layer study + pointers
  STUDY_DESIGN.md                # KEEP: the pre-registered program spec
  STUDY_RESULTS.md               # KEEP: layer-1 canonical (cost frontier; demote SS4 -> pointer)
  findings/study/
    CANONICAL_FINDINGS.md        # NEW: layer-2 (LLM waves 1-4) one-page canonical
    FULL_MATRIX.md               # KEEP: layer-1 bump-robustness canonical
    REVIEW_northstar_strategy.md # COMMIT: the strategic caveat
    WAVE4_RECONCILIATION_MAP.md  # KEEP: PoC reconciliation + agentic re-diagnosis
    ROUND0B_BASELINE.md          # KEEP (detail): det-floor primitives
    PHASE1_WAVE{1,2,3,4}_FINDINGS.md  # KEEP (detail): per-wave records (+ folded prereg section)
    phase1_wave{1,2,3,4}/
      results/*.json             # KEEP: scored results
      results/*_CONFIRMED.yaml   # KEEP: verified confirms
      results/wave2_armA_confirmed_*.yaml
      llm_proposed/*.yaml        # KEEP (w1/w2 Opus canonical)
      *.md (prompts)             # KEEP: locked prompts
    _archive/
      prereg/PHASE1_WAVE{1,2,3,4}_PREREG.md
      LLM_PATTERNS_NEXT_SESSION.md
      FANOUT_FINDINGS.md         # (optional demote)
  _archive/
    RESEARCH_WRITEUP.md
    DECISIONS_LOG.md
    wave2_poc/WAVE2_*.md
    strategies_poc/  (the stale scripts/strategies/ tree, if moved)
  scripts/
    phase1/wave*.py + *.sh       # current runners
    wave2_llm_*.py, study_gt_pilot.py, round0b/, gt_*.py, trial_scoring.py  # shared
    (strategies/ + trial_runner/aggregate/run_phase4_0_union -> _archive or keep, per import audit)
```

Two canonical docs (one per layer) + per-wave detail + design/spec. The `_archive/`
keeps the PoC corpus readable for lineage without it being in the hot path. Results
retention: scored `.json` + `CONFIRMED.yaml` committed; `_raw.txt` + `_corpus.yaml`
gitignored.

### Merge / cut / keep summary

- **MERGE**: each `PHASE1_WAVE{n}_PREREG.md` -> a short "Pre-registration" section at
  the top of its `_FINDINGS.md` (or, cheaper, just move prereg files to
  `_archive/prereg/` verbatim). Demote `STUDY_RESULTS.md` Section 4 to a one-line
  pointer to `FULL_MATRIX.md`.
- **CUT** (never commit): `STUDY_SYNTHESIS.md` (superseded by CANONICAL_FINDINGS).
- **ARCHIVE**: `LLM_PATTERNS_NEXT_SESSION.md`, `RESEARCH_WRITEUP.md`,
  `DECISIONS_LOG.md`, all `WAVE2_*.md`, all `scripts/strategies/`, the PoC trial
  drivers (post import-audit), optionally `FANOUT_FINDINGS.md`.
- **KEEP**: `CANONICAL_FINDINGS.md`, `FULL_MATRIX.md`, `STUDY_RESULTS.md`,
  `STUDY_DESIGN.md`, `WAVE4_RECONCILIATION_MAP.md`, `ROUND0B_BASELINE.md`,
  `REVIEW_northstar_strategy.md`, the 4 `_FINDINGS.md`, the locked prompts, the
  `.json`/`CONFIRMED.yaml`/`llm_proposed` results, the current runners + shared
  helpers.
- **GITIGNORE**: `*_raw.txt`, `*_corpus.yaml` (or all but one headline corpus),
  `__pycache__/`, `*.pyc`.

---

## 3. The naming gotcha (flag, do not auto-fix)

Two unrelated `wave2`/`w4c` namespaces collide and will confuse future readers:

1. `WAVE2_*.md` (top-level docs) = the PoC Wave-2 planning corpus. `scripts/wave2_*.py`
   = the CURRENT study's shared LLM helpers. Different "wave 2". A later rename of the
   script helpers to `study_llm_*` would remove the trap; out of scope for this cleanup.
2. `w4c_*` result files = **construction-grounding** (the `wave4_construct.py`
   `shape: construct_grounded` runner writes `phase1_w4c_*`), NOT the prereg's "4c
   agentic gate-as-tool". The agentic shapes (4b/4c in PHASE1_WAVE4_PREREG) emitted 0
   and have NO committed result files. CANONICAL_FINDINGS uses "construction-grounding"
   in prose to avoid the collision; the `w4c_` filename prefix is a latent trap. Note
   in the wave-4 findings; rename to `w4cg_`/`construct_` only if you re-run.

---

## 4. Cleanup plan (commands to execute - review before running)

All paths relative to `research/mining-substrate-trial/`. Run from there. NOTHING
below is executed by this doc.

### Step 0 - finish the in-flight run first
The 70B-vs-32B construction-grounding head-to-head (`wave4_construct.py --model
llama3.1:70b`) is RUNNING. Let it finish, fold its number into
`PHASE1_WAVE4_FINDINGS.md` + the CANONICAL_FINDINGS placeholder, THEN clean up (so
its `_raw.txt`/`_corpus.yaml` are gitignored from the start, not committed-then-removed).

### Step 1 - gitignore the intermediates
Create `research/mining-substrate-trial/.gitignore` (or append to the repo root's,
scoped):
```
# reproducible LLM mining intermediates - regenerate from runners, do not commit
findings/study/phase1_wave*/results/*_raw.txt
findings/study/phase1_wave*/results/*_corpus.yaml
# build artefacts
**/__pycache__/
*.pyc
```
Per global convention, prefer `.git/info/exclude` for local-only tooling entries;
the data-intermediate ignores above are project-shared, so a committed `.gitignore`
here is correct.

### Step 2 - remove the already-tracked intermediates from git (keep on disk)
```
git rm --cached findings/study/phase1_wave*/results/*_raw.txt
git rm --cached findings/study/phase1_wave*/results/*_corpus.yaml
# (24 raw + 25 corpus = 49 files, ~3MB out of git history going forward)
```
(Optionally keep ONE auditable corpus: re-add
`findings/study/phase1_wave4/results/w4c_qwen2_5-coder_32b_vllm_v0_19_1_corpus.yaml`
+ its tensorrt twin with `git add -f` and exempt them in .gitignore.)

### Step 3 - create the archive tree and move PoC + stale docs
```
mkdir -p _archive/wave2_poc _archive/prereg _archive/strategies_poc
git mv DECISIONS_LOG.md RESEARCH_WRITEUP.md _archive/
git mv WAVE2_*.md _archive/wave2_poc/
git mv findings/study/PHASE1_WAVE1_PREREG.md findings/study/PHASE1_WAVE2_PREREG.md \
       findings/study/PHASE1_WAVE3_PREREG.md findings/study/PHASE1_WAVE4_PREREG.md \
       _archive/prereg/
git mv findings/study/LLM_PATTERNS_NEXT_SESSION.md _archive/
# optional:
# git mv findings/study/FANOUT_FINDINGS.md _archive/
```

### Step 4 - audit + archive the stale scripts (DO the import audit first)
```
# find live importers of strategies/ and the PoC trial drivers:
grep -rn "from strategies" scripts/ ; grep -rn "import strategies" scripts/
grep -rn "trial_runner\|trial_aggregate\|run_phase4_0_union\|rescore_wave1" scripts/
```
Only after confirming the current `phase1/wave*.py` + `wave2_llm_*.py` chain does NOT
import them:
```
git mv scripts/strategies _archive/strategies_poc/strategies
git mv scripts/trial_runner.py scripts/trial_aggregate.py \
       scripts/run_phase4_0_union.py scripts/rescore_wave1_vs_gt.py _archive/strategies_poc/
# KEEP scripts/trial_scoring.py (live runtime-gate dispatch) unless the audit says otherwise.
```
Update the cross-ref links in `WAVE4_RECONCILIATION_MAP.md` (its SS2 table + the
Cross-references block point at `scripts/strategies/...` and `WAVE2_*.md`) to the new
`_archive/` paths.

### Step 5 - delete the superseded draft + commit the keepers
```
rm findings/study/STUDY_SYNTHESIS.md     # superseded by CANONICAL_FINDINGS (untracked; just rm)
git add findings/study/CANONICAL_FINDINGS.md \
        findings/study/REVIEW_northstar_strategy.md \
        findings/study/CONSOLIDATION_plan.md
```

### Step 6 - rewrite the README
Replace the dead-PoC README body with: the two-layer framing, status (Phase 1 waves
1-4 complete on `spike/engine-knowledge-as-data`, not on main), and a Quick-links
block pointing at `STUDY_DESIGN.md`, `STUDY_RESULTS.md`, `FULL_MATRIX.md`,
`findings/study/CANONICAL_FINDINGS.md`, `findings/study/REVIEW_northstar_strategy.md`,
with `_archive/` noted as PoC lineage.

### Step 7 - demote STUDY_RESULTS Section 4 + fold prereg sections (optional polish)
Trim `STUDY_RESULTS.md` Section 4 to a pointer to `FULL_MATRIX.md` (the numbers there
are already flagged SUPERSEDED). If folding preregs rather than bulk-archiving them,
add a 5-line "Pre-registration (locked)" block atop each `_FINDINGS.md` before
archiving the standalone prereg.

### Verification after cleanup
```
git status --porcelain            # expect only intended moves/adds
du -sh findings/study/phase1_wave* # confirm intermediates gone from the tracked set
git ls-files findings/study | wc -l  # should drop ~49 from 228
grep -rl "scripts/strategies\|WAVE2_" findings/ *.md  # confirm no dangling hot-path links
```

---

## 5. Scope notes / what this plan deliberately does NOT do

- Does NOT rename `scripts/wave2_*.py` -> `study_llm_*` (would touch every runner's
  imports; separate refactor). Flagged in SS3.
- Does NOT re-run or re-score anything; the `.json` results stand.
- Does NOT touch the GT (`ground_truth/`) or the production `_pydantic_lift.py`.
- Does NOT delete the PoC corpus - archives it (lineage + the reconciliation map still
  cites the W-A..W-G taxonomy).
- Leaves the 70B-vs-32B placeholder in CANONICAL_FINDINGS until the in-flight run lands.
