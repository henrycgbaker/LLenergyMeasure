# Engine-config invariant mining: consolidated findings

The single, self-contained writeup of the whole mining-substrate study: the
experimental design, the predecessor PoC bake-off, the deterministic-baseline /
bump-robustness layer, the LLM-pattern waves (1-4), and the cross-bump degradation
work (wave 5). It folds together every load-bearing result in the corpus into one
narrative and states honestly where the evidence is thin.

This doc is the canonical entry point. Per-wave detail, the locked program spec,
and the predecessor-PoC artefacts are kept for provenance and cross-referenced
inline; the stale/superseded material is moved to `_archive/` (see
[Sources and archive](#sources-and-archive) at the end).

---

## 0. The question and the north star

llem needs to keep **engine-config knowledge** current - the schema (what knobs
exist) and the **invariants** (the rules the engine enforces at config
construction: `if <pred>: raise/warn`, a `Field(ge/gt/le/Literal)` constraint, a
cross-field relation, a presence/conditional guard) - across upstream version
bumps of the inference engines (vLLM, TensorRT-LLM, transformers).

The north star is a **cheap, effective, general CI workflow** that does this
**affordably on every bump**. Its principles, fixed up front:

- The **engine owns its SSOT.** We do not re-encode the rules; we observe them.
- A **runtime gate** validates mined knowledge **in the engine's own container**
  ("observe, don't re-encode"): construct a config that should violate a rule,
  observe whether the engine raises. The gate, not any miner, is the adjudicator.
- **Mine comprehensively, expose a subset.** Candidate generators (deterministic
  miner, OSS LLM, Opus) propose; the gate disposes. The allowlist is applied at
  EXPOSURE time, not mining time - we want to know what is out there.
- **Cost is understood ordinally**, not as a plotted dollar frontier:
  deterministic (~free) < small-OSS < mid-OSS < large-OSS < Opus. The research
  question is "does the cheap rung suffice?"

The product property that matters is **cross-bump currency**: does carried
knowledge survive a bump, and does the workflow **notice when it does not**,
without a human diffing source. Keep that sentence in mind; Section 6 shows the
study spent most of its effort adjacent to it before testing it directly.

---

## 1. Method and experimental design

The full pre-registered spec is `STUDY_DESIGN.md` (locked program, Sections 1-15);
this is the operative summary.

**Two co-equal headline deliverables** were designed from the start:
(A) a **recall-cost frontier** - the cheapest workflow that holds high catalogue
recall vs ground truth on a *fixed* version (snapshot completeness); and (B)
**bump-delta-recovery + GT-growth** - per bump-pair, the fraction of changed
entries the workflow tracks *without a human editing its code*, plus how many
runtime-confirmed entries it surfaces that GT lacked. The design explicitly warned
that the cheapest-at-recall design and the cheapest-bump-robust design **may not be
the same** - the production choice needs both frontiers. (B) is the north-star
property; this is worth flagging because the LLM waves over-invested in (A).

**The design space** (axes from `WAVE2_PRIMITIVES.md`):
- **Role:** extract / extend-residual / gate / diagnose / diff-review / curate
- **Assembly:** det-only / llm-only / det-then-llm-extend / llm-then-det-gate /
  closed-loop / ensemble-vote / self-consistency / det-then-llm-patches-det
- **Call-shape:** single / k-vote / chunked / chained / agentic
- **Model tier (cost gradient):** OSS-small (7-14B) / OSS-mid (~32B) /
  OSS-high (~70B) / Opus. Expensive rungs included EARLY for diversity, PRUNED
  later if they do not justify cost.

**Engines x versions** - a 5-version window per engine (15 cells), chosen so the
window spans both minor bumps and one major boundary:

| engine | in-window versions | bump character |
|---|---|---|
| transformers | 5.6.2, 5.7.0, 5.8.1, 5.9.0, 5.10.2 | minor-only (all v5) |
| vllm | 0.18.1, 0.19.1, 0.20.0, 0.21.0, 0.22.0 | post-`config/` subpackage minors |
| tensorrt-llm | 0.20.0, 0.21.0, 1.0.0, 1.1.0, 1.2.1 | spans 0.x -> 1.x **MAJOR** |

**Locked parameters:** all 15 cells runtime-gated; GT = LLM-adjudicated union
(N=2 passes: entry-point + class-hierarchy) validated by the runtime gate (no human
checkpoint; validity rests on runtime confirmation); OSS chunking at 16k-22k chars;
det-baseline refine cap ~3 rounds to plateau.

**Identity scoring (two layers, locked rev 15.2):**
- **Tolerant key (headline recall axis):** `(leaf_native_field, coarse_predicate_bucket)`.
- **Constraint key (strict):** `(leaf, coarse_bucket, canonical_predicate_value)` -
  keyed on what the invariant *asserts*, not where it is declared. Confirmation is
  **per constraint**, so an easy sibling can no longer confirm a hard one (this
  fixed a class of false-confirms on per-subclass constraints).

**The mandatory guard:** every cross-field confirm is **adversarially
source-verified** (refute-first: assume the entry is wrong until the cited source
line proves predicate + outcome + field + bound match exactly, and the positive
probe is checked to fire the *claimed* rule, not an incidental error). The
inflation class recurs, so this is non-negotiable.

**The cost model** is ordinal with an energy gradient: det ~ 0 (CPU-sec); OSS =
GPU-energy rising with model size; Opus = token cost. One-time dev is amortised
separately. The deliverable is the *ordering* and the answer to "does the cheap
rung suffice," not a dollar plot.

---

## 2. The predecessor PoC: the bake-off that set the architecture

Before the structured study, a strategy bake-off (`RESEARCH_WRITEUP.md`, archived)
compared pure-deterministic, pure-LLM, and hybrid extraction across the engines
under **validated-union scoring** (every strategy scored against the runtime-validated
union, so each loses credit for its own blind spots). This is where the
architecture and the LLM-role split were settled.

**Headline strategy numbers (validated-union):**

| strategy | cells | inv. recall | inv. precision | wall (s) |
|---|---|---|---|---|
| (a) pure deterministic | 15 | 46.6% | 32.7% | 1.9 |
| (b) pure OSS LLM (llama-70B, chunked) | 15 | 42.3% | 27.6% | 3411 |
| (d-ab) active-seed + LLM-extension | 15 | 77.6% | 73.6% | 255 |

**The LLM-role split (the central PoC finding)** - at 70B-q4, the LLM is good at
some roles and actively bad at others:

| LLM role | quality | evidence |
|---|---|---|
| **Diagnose** (structured gap reasons, no mutation) | **excellent** | H9: 0 fabrications across 8 diagnoses, 6/8 match the gap inventory, ~50s/cell |
| Subtract (remove entries) | error-prone | H2: 3/3 vllm drops were FALSE-drops (misclassified normalisation patterns) |
| Synthesise code (write miner patches) | poor | H4: 0/3 patches lifted recall (2/3 crashed, anchor-text hallucinated) - but 6/6 structural *diagnoses* were correct |
| Synthesise output (extract catalogue) | substrate-ceiling-bound | (b)/E6/E9: ~50% transformers, ~30% vllm, ~16% tensorrt; no decomposition variant lifts the ceiling |
| Closed-loop / agentic | collapses | H7: **0 finalised invariants** - tool-use defaults to passive reading, never bridges to synthesis |

**Two PoC results the later waves should have weighted more heavily:**
- **The synthesis-pressure thesis:** per-class chunking *forces* the LLM into
  synthesis mode; every variant that relaxes the pressure under-emits -
  whole-source single-shot (H6) **halves** recall (62.5% -> 17.9%), cumulative-dedup
  (E9) loses cross-class invariants, agentic (H7) defaults to read-only. Chunking is
  not the bottleneck; removing it hurts.
- **The hallucination-on-empty-input failure:** when the chunker handed the LLM
  empty class bodies (old tensorrt, class names absent), the LLM did not notice the
  empty input and **hallucinated 30+ non-existent HuggingFace field names**. Only
  the downstream runtime gate caught it (those fields do not exist in the live
  engine). This is the single strongest argument for the gate.

**The PoC recommendation:** deterministic-mine first, chunked-LLM-extend second,
**runtime-gate third**, validated union becomes the cell artefact, maintainer
curates per bump with LLM-diagnose as an assistant. **LLM role enforced by
architecture: extraction + diagnosis only; subtraction is deterministic (the gate);
code-synthesis stays human.** Everything after this inherits that shape.

---

## 3. Layer 1 - the deterministic baseline and the bump-robustness gradient

This layer built the 15-cell ground truth and asked how a *fixed* deterministic
substrate survives real upstream bumps. Canonical detail: `FULL_MATRIX.md`,
`STUDY_RESULTS.md`, `findings/wave2_bump_survivability.md`, `ROUND0B_BASELINE.md`.

**GT integrity (the foundation everything else is scored against).** All 15 cells
are built from 2 Opus passes (entry-point + class-hierarchy), runtime-gated, and
**adversarially source-reviewed**: **908 of 913 reviewed entries verified REAL
(99.5%)** - 0 fabrications, 1 false-confirm (a transformers 5.9.0 mech-source
watermarking probe that raised an incidental `AttributeError`, excluded from the
validated GT), 4 mis-stated/imprecise (each still pointing at a real source rule).
The Opus basis that powers the gradient carried **zero** non-real entries.

**The bump gradient (PERSIST, field-level, tolerant key).** PERSIST = fraction of
the earlier cell's mined knobs whose field+bucket still exists in the next cell:

| engine | consecutive bumps (persist %) |
|---|---|
| tensorrt | 0.20->0.21 89.3, **0.21->1.0 (MAJOR) 53.1**, 1.0->1.1 93.8, 1.1->1.2.1 91.5 |
| vllm | 0.18->0.19 78.4, 0.19->0.20 91.9, 0.20->0.21 86.2, 0.21->0.22 85.1 |
| transformers | 5.7->5.8 75.9, 5.8->5.9 100.0, 5.9->5.10 98.9 |

**Headline:** the lone MAJOR boundary (tensorrt 0.21->1.0) persists only **53%** of
mined config-knobs - roughly half churns - versus **76-100%** across the eleven
minor bumps. A major bump reorganises the config-validation surface about 2x as much
as a minor one. This is the core empirical case for a runtime gate that
re-validates carried knowledge against the live engine, especially on majors.
(A separate "survivor re-bound rate" metric was **retracted** - it conflated real
constraint changes with predicate-*encoding* variance; only encoding-agnostic
PERSIST is trustworthy.)

**The deterministic ceiling.** At constraint grain on the frozen reference set,
bare mechanical mining (`improved-det-v2`) confirms **25%**; adding the production
PluginConfig walk lifts it to **46.7%**. The deterministic floor is real but
partial.

**THE CLIFF - the load-bearing risk of the whole architecture.** Carrying a fixed
substrate across a real major refactor, recall vs each version's own GT:

| engine | v_old recall | v_new recall | direction |
|---|---|---|---|
| transformers (4.57.3->5.6.2) | 0.404 | 0.416 | flat |
| **vllm (0.7.3->0.19.1)** | **0.513** | **0.147** | **COLLAPSE -0.366** |
| tensorrt (0.21->1.2.1) | 0.270 | 0.400 | rise +0.130 |

vllm 0.19.1 moved a large fraction of invariants from imperative `raise` to
**declarative `Field(ge/gt/le/Literal)`** constraints. The substrate was built for
imperative checks; it cannot see declarative ones, so it **falls off a cliff exactly
where upstream refactored** - and **nothing in the deterministic output signals the
collapse.** It silently emits fewer invariants. tensorrt moved the *other* way
(C++-only validators became visible Python pydantic), so the same substrate recalls
*more*. Bump-survivability is driven by *how* the surface changed, not by churn
magnitude.

Two implications, both load-bearing:
1. **A static floor is not bump-robust on its own.** Robustness needs either a new
   declarative-constraint primitive, or an LLM tail that reads the diff and recovers
   the residual, or both. A deterministic **Primitive 8** (extract pydantic
   `Field(...)` + `Literal`/enum + glob the `config/*.py` subpackage) was built and
   **mechanically recovers ~44% of the vllm cliff with no LLM** (0.147 -> 0.309,
   leaf+bucket), and generalises (tensorrt cells also lift). This is the
   highest-ROI, most north-star-aligned engineering result in the study.
2. **The substrate does not know what it does not know.** A self-updating workflow
   cannot rely on the substrate to detect its own degradation. An **external signal**
   is required - the runtime gate's acceptance rate, or an LLM diff-reviewer. This is
   exactly what wave 5 (Section 5) builds.

---

## 4. Layer 2 - the LLM-pattern waves (1-4): how far the OSS rung gets on a frozen cell

Waves 1-4 dropped the floor and varied STRATEGY x MODEL-TIER on two frozen cells
(vllm 0.19.1, GT=80; tensorrt 1.2.1, GT=61), scoring gate-confirmed recall vs GT.
This is deliverable (A) - snapshot recall. Per-wave detail:
`PHASE1_WAVE{1,2,3,4}_FINDINGS.md`; PoC reconciliation: `WAVE4_RECONCILIATION_MAP.md`.

**Wave 1 - the bottleneck is VALIDATION, not LLM recall.** det-then-llm-extend
(match-only) yielded **0 gate-confirmed lift** over the floor on both cells x both
rungs (gemma3:12b and Opus). Not because the LLM found nothing - Opus surfaced ~50
real net-new cross-field relations - but because the single-field auto-synthesis
gate **could not probe them.** The validation PATH, not recall, was binding.

**Wave 2 - the kwargs-emission lever unlocks the cross-field tail.** Having the LLM
also emit constructible `kwargs_positive/negative` made the tail gate-confirmable:
Opus **0 -> 8 verified-real** cross-field confirms (the first cross-field constraints
folded into the GT). gemma3:12b **failed** the lever (17/25 proposals failed; 0
verified-real). The tail is real but reachable only at Opus cost on this shape. Also
surfaced + fixed a gate soundness gap (cross-field confirm attribution by error
locus).

**Wave 3 - scale is the threshold; code-tuning sharpens (the size x tuning 2x2).**
Verified-real cross-field confirms: small (gemma-12b, qwen-coder-7b) = 0; mid/large
(qwen-coder-32b = 5, llama-70b = 3) reach it; Opus = 8. **Scale is the threshold to
reach the tail at all (between 12B and 32B); code-tuning sharpens within the capable
regime** - a code-tuned 32B BEAT a general 70B on coverage (5 vs 3 cross-field),
precision, speed (~2680s vs ~4436s), and cleanliness (zero internal-noise vs the
70B's `_api_process_rank`). Validated the internals-guard.

**Wave 4 - the OSS strategy frontier.**
- **pure-LLM / prompt (4a):** tensorrt **100% infra-blocked** (0/61) - the LLM omits
  required ctor fields, so pydantic "field required" fires before any validator runs.
- **CONSTRUCTION-GROUNDING is THE OSS lever.** Inject each class's AST-extracted
  constructor signature (required/optional fields + types) so construction REACHES
  the real validators. Breaks the tensorrt infra wall (qwen-coder-32b **0 -> 20**
  verified-real) and lifts vllm precision. It is itself a det+LLM hybrid:
  deterministic AST does construction-context discovery, the LLM synthesises. It is
  **model-specific to the qwen2.5-coder line** - does NOT generalise to qwen3-coder
  (MoE) or deepseek (both regress on tensorrt).
- **The 70B-vs-32B construction-grounding head-to-head (vllm 0.19.1).** The decisive
  test of "does scale beat code-tuning":

  | model | tier | gate-confirmed | recall vs GT | infra_err | wall (s) |
  |---|---|---|---|---|---|
  | **qwen2.5-coder:32b** | code-tuned 32B | 30 | **21** | 37 | **1065** |
  | qwen2.5:32b | general 32B | 22 | 15 | 31 | 814 |
  | llama3.1:70b | general 70B | 20 | 15 | 26 | 3153 |
  | qwen2.5:72b | general 72B | 0 | **0** (collapsed) | 6 | 5738 |

  **Scale does NOT substitute for code-tuning.** The code-tuned 32B wins on recall
  (21 vs 15/15) at **3-5x less wall time** than the general 70B/72B. The general 72B
  effectively collapsed (16 gateable, 0 confirmed, slowest of all) - but **that row is
  partly degenerate and should be read with caution:** the 72B emitted only 19 raw
  proposals vs the coder-32B's 194 and the 70B's 112 (a 6-10x under-emission that
  looks more like a serving/format failure than a clean "scale is useless" result).
  The defensible leg of "code-tuning beats scale" is the **general 70B**, which
  produced real output and still lost on recall (15 vs 21) at 3x the wall, plus the
  wave-3 result (coder-32B 5 vs llama-70B 3 cross-field). Together these argue for a
  **local mid code-model** as the OSS workhorse; the 72B collapse corroborates
  weakly, not decisively.
- **AGENTIC (LangGraph) is a POOR strategy for OSS:** ollama tool-call flakiness + no
  incremental synthesis, even with devstral. The prior "agentic=0" (PoC H7) is an
  all-at-once-harness + read-only-collapse artefact, not a fresh model finding
  (re-diagnosed in the reconciliation map). Exploration is better done
  deterministically. This matches the PoC synthesis-pressure thesis exactly.
- **LANGCHAIN multi-stage chains - and an honest caveat (see Section 7).** A 2-stage
  chain (STAGE1 find rules, STAGE2 construct probes) was built to test "70B +
  decomposition beats 32B single-shot." It first scored **0 confirmed / 143
  infra_error** - a catastrophic result that turned out to be a **bug**: stage 2 was
  decoupled from the source, so the LLM constructed probes blind. After the fix
  (passing the source chunk into stage 2), the same 32B chain went **143 -> 47
  infra_error, 0 -> 28 confirmed, recall 0 -> 20** - **competitive with single-shot
  (recall 21) but did not exceed it.** A hybrid chain (consume the deterministic
  floor, extend it) reached hybrid_recall 49 (floor 44 + LLM lift 5), **trailing**
  the single-shot construction-grounded hybrid (55). Conclusion: chains are not
  fundamentally broken, but as built they are at best on par with single-shot and the
  hybrid chain underperformed. **We likely did not set langchain up properly; this
  warrants future inspection** - it is the most under-explored cell.
- **A second gate soundness fix:** reject type-coercion-artifact confirms (a pydantic
  parsing/literal error on the probed field, not the labelled semantic rule). Gated
  on `not expected_strict` so the GT is untouched. (Caveat: the guard is
  pydantic-only; it silently skips msgspec.)
- **Residual analysis:** the tensorrt "ceiling" is a **study-floor artefact** - 20 of
  the 25 missed tensorrt GT keys are one class (PluginConfig) whose Literal/enum
  constraints the study's validator-body floor cannot see, but the *production*
  `_pydantic_lift.py` already extracts. So production tensorrt recall is materially
  higher than the study number.

**The snapshot-recall answer (deliverable A):**

| cell | floor alone | + construct-grounded LLM net-new | HYBRID |
|---|---|---|---|
| vllm 0.19.1 | 44 (55%) | +11 | **55/80 (69%)** |
| tensorrt 1.2.1 | 35 (57%) | +1 | **36/61 (59%, understated)** |

On a frozen cell, a construction-grounded local 32B code-model adds a real
cross-field tail on vllm (+11) and almost nothing on tensorrt (+1, and that surface
is already covered by production det). The LLM's extraction value is concentrated in
exactly one cell's cross-field tail.

---

## 5. Layer 3 (wave 5) - cross-bump: testing the actual product property

Waves 1-4 measured snapshot recall. The strategic review (Section 6) is blunt that
this is the *wrong* axis for the north star. Wave 5 builds and runs the two
cross-bump experiments the design always called first-class. Runners:
`scripts/phase1/wave5_gate_acceptance.py`, `scripts/phase1/wave5_bump_diagnose.py`.

### 5a. The gate-acceptance degradation signal (the external alarm)

The cliff finding said the substrate cannot raise an alarm about its own decay; the
named-but-never-built external signal is the **runtime gate's acceptance rate.** This
runner carries an old version's confirmed-GT catalogue (each entry already has its
`kwargs` probe) forward and **re-gates it against the new version's container**. The
acceptance-rate DROP is the alarm.

| bump (vllm) | old catalogue | acceptance vs new | n_broke |
|---|---|---|---|
| 0.18.1 -> 0.19.1 | 101 | 95.0% | 5 |
| 0.19.1 -> 0.20.0 | 105 | 94.3% | 6 |
| 0.19.1 -> 0.21.0 | 105 | 94.3% | 6 |
| 0.18.1 -> 0.22.0 | 101 | 93.1% | 7 |
| 0.19.1 -> 0.22.0 | 105 | 92.4% | 8 |

**The signal works and trends downward as the bump span widens:** acceptance falls
toward 92.4% and the broken count rises toward 8 on the widest span. A reviewer
watching only the gate's accept/reject delta - with no manual source diff - would see
the catalogue going stale. This is the degradation alarm the deterministic substrate
structurally cannot raise about itself, now demonstrated. Three honest qualifiers:
- It is **not strictly monotonic in minor-step distance** (the 4-step 0.18.1->0.22.0
  at 93.1% is *less* degraded than the 3-step 0.19.1->0.22.0 at 92.4%, because the
  start points and catalogue sizes differ); read it as a trend, not a clean monotone.
- `n_broke` **conflates two failure modes:** `failed` (the rule genuinely no longer
  holds - 2 to 4 per bump) and `infra_error` (the carried probe will not even
  construct on the new version - 3 to 4 per bump). Both are legitimate "this entry no
  longer cleanly confirms" signals, but only `failed` means a constraint changed; the
  infra component is noisier (a constructor-signature drift, not a rule change).
- These are all *minor* vllm bumps, so the decay is gentle (a 5-8 invariant spread on
  a ~100 catalogue). The major-bump cliff - tensorrt 0.21->1.0, where layer 1 showed
  53% PERSIST - would drive a far sharper drop and remains the strongest un-run cell.

### 5b. The LLM bump-diagnose (the W-F diff-reviewer role)

The PoC's strongest LLM role was **diagnose** (H9: 0 fabrications). Wave 5 tests it
on a bump: per new-source chunk, give the LLM the old catalogue entries for that
chunk's classes + the new source, ask for a structured `broke` / `new` diagnosis,
score against the actual GT diff (old leaves vs new leaves) by tolerant leaf.

`vllm 0.19.1 -> 0.20.0`, qwen2.5-coder:32b (`broke`/`new` flagged vs the actual GT
diff, ~640s wall):

| axis | actually | LLM flagged | true positives | precision | recall |
|---|---|---|---|---|---|
| broke (stale entries) | 4 | 30 | 2 | **0.067** | 0.50 |
| new (new surface) | 17 | 76 | 5 | **0.066** | 0.29 |

**Honest reading - the diagnose alarm FIRES but is far too NOISY to trust alone.**
The LLM did raise an alarm and caught half the actually-broke entries (2/4) and ~29%
of the new surface (5/17) - but it buried those few true positives in ~28 false
"broke" and ~71 false "new" flags (precision ~7%). As a *standalone* detector it is
unusable: a reviewer handed 106 flags to find 7 real changes would not trust it.
This RE-TEMPERS the diagnose-role optimism: the PoC's "H9: 0 fabrications" was a
*categorical-gap* task on a frozen cell, NOT this broke/new-vs-GT-diff scoring across
a real bump - the tasks are different and the bump version is much harder. The
correct conclusion is the proposer/disposer one: **the LLM diagnose is a noisy
*proposer*, and the runtime gate is the *disposer*.** The clean cross-bump signal is
the **gate-acceptance rate (5a)**, not the raw LLM diagnose (5b); pairing them (LLM
proposes candidate changes, gate confirms which are real) is the defensible shape,
and the LLM-alone diagnose number says do NOT ship the diagnose role ungated.

**Methodology note (important, do not silently trust a zero):** the bump-diagnose's
*first* run reported `wall_sec 0.0`, `diag_broke 0`, `diag_new 0` - which looks like
"the LLM stayed silent" but was actually an artefact: the new-version source had been
reaped from `/tmp` (the trial venvs are transient), so `source_files_for` resolved to
a missing path, 0 chunks were produced, and the per-chunk loop never called the LLM.
Restoring the surviving v0.20.0 source (`/tmp/vllm-0.20.0/`) into the expected path
fixed it. This is a live instance of a known methodology gap: **the chunked source
inputs are not committed**, so a reaped `/tmp` silently zeroes a run. Commit the
chunked inputs (cheap fix) before relying on any cross-bump number.

---

## 6. The strategic correction (stated honestly)

An adversarial north-star review (`REVIEW_northstar_strategy.md`) made one
correction that the rest of this doc must not bury:

> **Waves 1-4 optimised single-version recall, which is not the product property.**
> The north star is cross-bump currency. The prior PoC already measured the actual
> property and got the most important result in the corpus - the silent -0.366 vllm
> cliff - and waves 1-4 then spent four waves, a 7B-70B+Opus tier sweep, a
> construction-grounding strategy, a LangGraph harness, and an enormous doc trail
> optimising recall on the two frozen cells the cliff finding had already shown were
> the wrong unit of analysis.

The review's load-bearing points, carried forward intact:
1. **No cross-bump test existed** through wave 4 - the one thing the design must do.
   Wave 5 (Section 5) is the direct answer; it should have come first.
2. **The study floor is sandbagged.** `improved-det-v2` predates the production
   pydantic-lift and misses PluginConfig, so every "LLM lift over floor" number is
   measured against a too-low baseline. The correct comparison - production-det floor
   vs production-det + LLM, *on a bump* - was never run.
3. **The runtime consumes only a few dozen error/dormant rules.** The cross-field
   tail waves 2-4 spent Opus and GPU-hours to reach has **no runtime consumer today**
   and is in no shipped corpus. "Comprehensive discovery" is the design, but it is
   only worth its cost if something downstream consumes it or it improves
   bump-currency; the tail demonstrably does neither yet.
4. **The most defensible LLM role is diff-review/diagnose on a bump, not
   extraction** - but with a sharp qualifier from wave 5b. A better deterministic
   miner largely subsumes extraction (the PoC and the Primitive-8 cliff-recovery both
   show this). The PoC's H9 diagnose scored 0 fabrications, which motivated this role
   - but the actual cross-bump diagnose (wave 5b) is **noisy: ~7% precision**, useful
   only as a *gated proposer*, not a standalone detector. So the LLM's north-star role
   is real but narrower than waves 1-4 assumed: propose candidate changes cheaply, let
   the gate confirm them. The clean cross-bump alarm is the gate-acceptance rate, not
   the LLM.
5. Other standing caveats: **N=2 cells, single-shot, directional**;
   construction-grounding is qwen-coder-line-specific (a hidden version-currency
   problem inside the proposed solution); per-bump **cost was never measured in
   absolute terms** (only ordinally), so "CI-affordable" is asserted, not shown;
   tier-sweep confounds (Opus ran whole-source, OSS ran chunked).

This correction is *why* wave 5 exists and why the synthesised answer below leads
with the deterministic floor + the gate + the diagnose role, not with the extraction
tier sweep.

---

## 7. The langchain caveat (called out prominently)

The langchain/langgraph cells are the **least-trustworthy** in the corpus, and the
study's negative conclusions about them should be read as **"likely a setup
problem," not "chains do not work":**

- The multi-stage chain's first result (0 confirmed / 143 infra) was a **plain bug**
  (stage 2 decoupled from the source). We only caught it because the number was
  absurd. A subtler mis-wiring could have produced a *plausible-but-low* number we
  would have banked as a real finding.
- Even after the fix, the chain only reached **parity** with single-shot, and the
  hybrid chain **underperformed** the single-shot hybrid. That is suspicious: a
  well-constructed multi-stage chain with proper state passing, inter-stage
  validation, and retries should *not* trail a single prompt. The likely explanation
  is our chain construction was naive (no validation between stages, no retry on
  parse failure, weak state threading), not that decomposition is worthless.
- The agentic (LangGraph ReAct) cells collapsed to read-only - consistent with the
  PoC's H7, but again confounded by OSS tool-call flakiness in our harness.

**Future inspection warranted:** a properly engineered langchain pipeline (typed
state, per-stage gate feedback, retries, possibly a stronger stage-1 model) is an
open and promising cell. Treat the current langchain numbers as a floor, not a
verdict.

---

## 8. The synthesised answer

**Production design = the production deterministic floor (pydantic-lift +
Primitive 8) + the runtime gate + an LLM in its two *defensible* roles:** (i) a
**construction-grounded local mid code-model (~32B, qwen2.5-coder line)** as the
optional extraction topping for the cross-field tail on engines that have one, and
(ii) the **LLM diff-reviewer/diagnose** on each bump as a *gated proposer* of
candidate changes (wave 5b shows it is too noisy - ~7% precision - to trust ungated;
the gate filters it). Plus the two gate soundness guards and the internals-guard at
exposure time. The clean degradation alarm is the gate-acceptance rate itself, not
the LLM diagnose.

The reasoning, layer by layer:
- The **deterministic floor does the load-bearing extraction** (vllm 55%, tensorrt
  57% study-floor, higher in production), it is free and reproducible, and a
  deterministic primitive recovers ~44% of the worst observed bump cliff.
- The **runtime gate is mandatory regardless** - it is the SSOT-respecting
  adjudicator and the thing that catches LLM hallucination and stale carried
  knowledge. The **gate-acceptance rate is the degradation alarm** (wave 5a,
  demonstrated).
- The **LLM is a topping, not the workhorse.** Construction-grounding adds a real but
  cell-specific tail (vllm +11, tensorrt +1); a code-tuned 32B is the OSS efficiency
  winner (beats general 70B at 3x less wall; the 72B row is degenerate). The LLM's
  narrow north-star value is the cheap diagnose/diff-review role *as a gated proposer*
  (wave 5b: ~7% precision ungated), not extraction and not a standalone alarm.
- **"Just call Opus per bump" is the wrong default:** it keeps the expensive half,
  pays to re-derive what det does free and reproducibly, sacrifices reproducibility,
  and still needs the gate. Opus is justified only as a small, expensive topping for
  the residual cross-field tail *if* a runtime consumer for that tail ever exists.

---

## 9. Cost (ordinal - the deliverable, not a dollar plot)

deterministic (~free, reproducible, CPU-sec) < local OSS code-32B (minutes/bump on a
local GPU; qwen2.5-coder:32b ~1065s vllm / ~416s tensorrt per cell) < large general
OSS (slower AND worse: llama-70B ~3153s, qwen-72B ~5738s and collapsed) < Opus (API
tokens/bump, non-reproducible). The cheap rungs own the bulk; the code-tuned 32B is
the efficiency winner among OSS. **Open gap:** the *absolute* per-bump wall +
GPU-minutes of the recommended stack on the CI runner profile was never measured, so
"CI-affordable" remains ordinally argued, not demonstrated (review point #5).

---

## 10. What is VALIDATED vs OPEN

**Validated:**
- The harness <-> gate integration; gate catches hallucination (PoC) and stale
  carried knowledge (wave 5a).
- Validation-path-is-the-bottleneck (wave 1); the kwargs lever (wave 2).
- The scale-threshold + code-tuning tier story (wave 3), reconfirmed on the *general
  70B* leg of the wave-4 head-to-head: **code-tuning beats scale** (the 72B leg is
  degenerate and only corroborates weakly).
- Construction-grounding as the OSS infra-wall lever (wave 4); the hybrid snapshot
  recall (vllm 69% / tensorrt >=59%).
- Two gate soundness fixes; agentic is wrong for OSS (consistent across PoC + wave 4).
- The major-vs-minor bump gradient (layer 1: 53% vs 76-100% PERSIST); the silent
  vllm cliff (-0.366) and its ~44% deterministic Primitive-8 recovery.
- **The cross-bump degradation signal fires** (wave 5a): gate acceptance trends down
  with bump span (95.0% -> 92.4% across the vllm minors).
- **The LLM bump-diagnose is a noisy proposer, NOT a clean detector** (wave 5b):
  ~7% precision, ~0.3-0.5 recall on vllm 0.19.1->0.20.0 - useful only gated.

**Open / future (ranked by north-star value):**
1. The **gate-acceptance signal on a MAJOR bump** (tensorrt 0.21->1.0) - the sharp
   cliff, the strongest un-run cell.
2. **Production-det floor (not the sandbagged study floor) re-measured across a bump**
   - the correct floor-vs-floor+LLM comparison.
3. The **LLM bump-diagnose on a MAJOR bump**, and *gated* (LLM proposes -> gate
   confirms) to recover usable precision from the noisy raw diagnose (wave 5b).
4. **Absolute per-bump cost** of the recommended stack on the CI profile.
5. The **auto-PR-for-review** end-to-end workflow (bump -> re-mine -> gate ->
   open PR with the catalogue delta; is it a rubber-stamp?).
6. A **properly engineered langchain pipeline** (Section 7) - the under-explored cell.
7. Generalising construction-grounding beyond the qwen-coder line; self-consistency
   (k-vote) upside.

---

## 11. Caveats (honest)

- **N=2 frozen cells for the recall waves, mostly single-shot, directional** - not a
  frontier point. Conclusions stated as "production design" exceed what 2 cells prove.
- **The study floor (`improved-det-v2`) understates the production miner**, so the
  tensorrt hybrid number is a lower bound and every "LLM lift over floor" is measured
  against a sandbagged baseline.
- **Cross-field confirms always require adversarial source-verification** (the
  inflation class recurs).
- **Construction-grounding is model-family-specific** (qwen2.5-coder), itself a
  version-currency risk.
- **The internals-guard and the type-coercion guard** are applied in analysis /
  pydantic-only respectively; the latter silently skips msgspec.
- **Chunked source inputs are not committed**, so a reaped `/tmp` can silently zero a
  cross-bump run (Section 5b) - commit the inputs before trusting cross-bump numbers.
- **The langchain cells are likely mis-configured** (Section 7) - a floor, not a
  verdict.
- **Per-bump cost is ordinal only**; "CI-affordable" is not yet shown in absolute
  terms.
- Some of the residual ~30-40% of GT that neither floor nor LLM reaches is arguably
  internals-guard territory (observability config classes) that should not be in GT.

---

## Sources and archive

**Live canonical detail (kept, cross-referenced above):**
- `STUDY_DESIGN.md` - the locked pre-registered program spec.
- `FULL_MATRIX.md` - the authoritative 15-cell bump-robustness matrix + GT integrity.
- `findings/wave2_bump_survivability.md` - the cliff + Primitive-8 recovery.
- `ROUND0B_BASELINE.md` - the deterministic-floor primitives.
- `PHASE1_WAVE{1,2,3,4}_FINDINGS.md` - per-wave detail.
- `WAVE4_RECONCILIATION_MAP.md` - maps the waves onto the PoC taxonomy; re-diagnoses
  agentic=0.
- `REVIEW_northstar_strategy.md` - the full adversarial strategic review (Section 6).
- `REVIEW_methodology.md` - the methodology review.
- Results: `phase1_wave{1,2,3,4}/results/*.json` + `*_CONFIRMED.yaml`;
  `phase1_wave{1,2}/llm_proposed/*.yaml`; cross-bump JSONs under
  `phase1_wave5/results/` (gate-acceptance + diagnose).

**Archived for provenance (under `research/mining-substrate-trial/_archive/`,
superseded by this doc - see that dir):**
- `RESEARCH_WRITEUP.md` - the predecessor PoC bake-off (Section 2 folds its signal).
- `DECISIONS_LOG.md` - the PoC chronological decision log.
- `wave2_poc/WAVE2_*.md` - the PoC Wave-2 planning/taxonomy corpus.
- `prereg/PHASE1_WAVE{1,2,3,4}_PREREG.md` - per-wave pre-registrations.
- `STUDY_RESULTS.md`, `FANOUT_FINDINGS.md` - layer-1 synthesis/detail folded into
  Section 3.
- `CANONICAL_FINDINGS.md`, `STUDY_SYNTHESIS.md` - the layer-2-only one-pager and the
  superseded synthesis draft this doc replaces.
- `LLM_PATTERNS_NEXT_SESSION.md`, `HANDOFF_NEXT_SESSION.md` - session bootstraps.
- `CONSOLIDATION_plan.md` - the cleanup plan that produced this consolidation.

Note: the `scripts/` tree (including `strategies/wave2/`, the `improved-det-v2`
deterministic floor) is intentionally NOT archived - an import audit confirmed it is
live (consumed by `round0b/` and `wave2_runner.py`). Only superseded prose moved.
