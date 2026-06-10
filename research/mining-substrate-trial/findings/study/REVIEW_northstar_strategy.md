# Adversarial strategic review: the mining spike vs its north star

Status: adversarial review, 2026-06-10. Reviewer brief: find where this spike has
gone wrong against the north star, not validate it. Scope: Phase 1 waves 1-4 + the
prior Wave 2 PoC, read against the design intent (`engine-knowledge-as-data.md`),
the runtime consumer (`config/models.py`, `study/library_resolution.py`), and the
runtime corpora actually shipped under `src/llenergymeasure/engines/<e>/`.

---

## THE SINGLE MOST IMPORTANT STRATEGIC CORRECTION

**Stop optimising single-version recall. The spike has been measuring the wrong
quantity for four consecutive waves.**

The north star is CROSS-BUMP CURRENCY: does the workflow keep knowledge correct
across an upstream version bump WITHOUT human intervention, cheaply, every bump.
Phase 1 waves 1-4 measured something else entirely: how many invariants a model can
mine from ONE frozen version (vllm 0.19.1, tensorrt 1.2.1), scored against a
single-version ground truth. That is a recall benchmark on a snapshot. The product
property - "does carried knowledge survive a bump, and does the workflow notice
when it doesn't?" - was never tested in any of waves 1-4.

This is not a subtle drift. The prior Wave 2 PoC **already measured the actual
property** and got the most important result in the entire corpus:

> vllm v0.7.3 -> v0.19.1: deterministic substrate recall COLLAPSED -0.366
> (0.513 -> 0.147), and **nothing in the deterministic output signalled the
> collapse** - it silently emitted fewer invariants.
> (`findings/wave2_bump_survivability.md`)

That finding - the silent cliff plus the absence of a self-degradation signal - is
the central risk to the entire architecture. Waves 1-4 then walked away from it and
spent four waves, a model-tier sweep (7B/12B/14B/16B/30B/32B/70B + Opus), a
construction-grounding strategy, a LangGraph agentic harness, two gate soundness
fixes, and an enormous documentation trail optimising recall on the two frozen
cells the cliff finding had already shown were the WRONG unit of analysis.

The kickoff flagged exactly this: the tensorrt 0.21->1.0 major-bump
self-update/degradation-signal binary is "the actual product property and nothing
has tested it ... more north-star-relevant than another coverage wave." It was
called the strongest un-run item on 2026-06-09 and is STILL un-run. The ground
truth for the bump endpoints already exists on disk
(`findings/study/ground_truth/tensorrt/v1_0_0/`). The experiment is cheap and
ready. It keeps being deferred behind another recall wave.

**Correction: the next wave must be the cross-bump degradation-signal binary, not
another point on the size x shape matrix.** Everything below elaborates why the
recall-maximisation framing is wrong and what to do instead.

---

## 1. Have we been approaching this wrong? YES, on the core objective

### 1a. Single-version recall is not the product property, and we have the evidence

The "size x shape matrix" that `LLM_PATTERNS_NEXT_SESSION.md` enshrines as the
CORE DELIVERABLE answers "what coverage does model M at workflow shape S achieve on
a frozen cell?" The north star asks a different question with a different failure
mode: "when upstream refactors imperative `raise` into declarative `Field(ge=...)`,
does the workflow keep up, and does it KNOW when it falls behind?" A model that
scores 69% recall on vllm 0.19.1 tells you nothing about whether it survives
0.19.1 -> 0.22.0. These are orthogonal axes, and we have spent four waves moving
along the wrong one.

The matrix is not worthless - it characterises the extraction rung. But it has been
mis-framed as the headline deliverable when it is, at best, a sub-component of one
workflow's extract step. The headline should always have been the per-bump-pair
self-update binary that Wave 2 started and that STUDY_DESIGN Section 5 itself marks
"the actual product property, so first-class."

### 1b. The cliff is the architecture's load-bearing risk and it is mechanical, not LLM-shaped

Wave 2.5 already found that the vllm cliff is recovered ~44% by a purely
deterministic Primitive 8 (declarative-`Field` extraction + subpackage globbing) -
no LLM. The cliff was caused by upstream moving from imperative to declarative
constraints; the fix was a deterministic parser for the new idiom. This is the
single highest-ROI, most north-star-aligned result in the whole study, and waves
1-4 built nothing on it. They went the other direction - toward more expensive LLM
strategies on cells that never bump.

### 1c. The two-cell freeze is self-reinforcing tunnel vision

Every wave 1-4 runs on the SAME two cells (vllm 0.19.1, tensorrt 1.2.1). Each wave
folds verified-real confirms back into THAT cell's GT, which raises the GT
denominator, which motivates the next recall wave, which folds more. This is a
closed loop that manufactures its own backlog. The GT-growth is real but it is
GT-internals churn on a frozen snapshot; it does not move the product property one
inch. The loop should have been broken by changing the cell (run a bump), not by
adding another model tier.

---

## 2. The LLM's actual value: largely unjustified under a cheap-CI north star

### 2a. The deterministic floor does almost all the load-bearing work

The data is consistent across every wave:
- Floor alone: vllm 44/80 (55%), tensorrt 35/61 (57%) - and the floor used in the
  study (`improved-det-v2`) is OLDER and NARROWER than the production
  pydantic-lift. The study's own Wave 4 residual analysis concludes the tensorrt
  "ceiling" is a STUDY-FLOOR ARTEFACT: production pydantic-lift already covers the
  20 PluginConfig Literal constraints the study floor misses. So production recall
  is materially higher than the headline numbers, and the LLM's apparent
  contribution is partly just "the study floor was sandbagged."
- LLM net-new (best OSS strategy, construction-grounded qwen-32B): vllm +11 GT
  keys, **tensorrt +1**. On tensorrt the LLM adds essentially nothing to GT recall.
- The LLM's tensorrt "20 verified-real confirms" are mostly GROWTH (real invariants
  not in the GT) - but Wave 4 itself then shows those same PluginConfig constraints
  are already covered by the production deterministic lift. So the LLM is
  re-deriving, at GPU cost and non-reproducibly, what a free deterministic pass
  already produces.

### 2b. The runtime DOES NOT CONSUME the comprehensive tail at all

This is the finding that most undercuts the LLM investment. I traced what the
runtime actually does with invariants (`config/models.py::_apply_invariants`):
exactly three consumer modes exist - `error` (raise), `warn`, and `dormant` (feeds
`study/library_resolution.py` config-dedup). The shipped corpora:

```
src/llenergymeasure/engines/tensorrt/invariants.validated.yaml   3 entries
src/llenergymeasure/engines/transformers/invariants.validated.yaml  41 entries
src/llenergymeasure/engines/vllm/invariants.validated.yaml       EMPTY
proposed (the active surface): tensorrt 3 error; transformers 22 error + 19 dormant;
                               vllm 9 error + 1 dormant
```

The runtime consumes a few dozen error/warn/dormant rules total. The cross-field
tail waves 2-4 spent Opus and 32B-GPU-hours to reach (`data_parallel_external_lb
requires dp>1`, `max_cpu_loras >= max_loras`, structured-outputs exactly-one, ...)
is NOT in any shipped runtime corpus and has no runtime consumer today. The
"comprehensive discovery, expose a subset" principle is being used to justify
mining a tail that the subset never draws from. The North-star reconciliation note
in Wave 3 waved this away as "comprehensive discovery IS the design" - but
comprehensive discovery is only worth its cost if (a) something downstream consumes
it or (b) it materially improves bump-currency. Neither is demonstrated. The
cross-field tail fails both tests.

### 2c. The honest answer to "is the LLM justified?"

For the SCHEMA and the error/dormant invariants the runtime actually consumes:
**deterministic miner + runtime gate, no LLM, is the right production default.** It
is free, reproducible, CI-affordable on every bump, and Wave 2.5 showed a
deterministic primitive recovers the bump cliff. The LLM's measured GT-recall lift
(+11 vllm, +1 tensorrt) is concentrated entirely in a cross-field tail with no
runtime consumer.

The genuinely-defensible LLM role is NOT extraction. It is the cheap READ roles the
prior PoC already validated and waves 1-4 ignored:
- **W-F / H9 diagnose** ("here is what changed and what to look at") - H9 scored 0
  fabrications across 8 diagnoses, ~50s/engine, cheapest-effective pattern in the
  whole corpus.
- **diff-review on a bump** ("v_new added these constraint surfaces the floor may
  have missed") - the one role that directly attacks the silent-cliff problem.

These are the north-star-relevant LLM uses, and they are exactly the ones the spike
de-prioritised in favour of extraction tier sweeps. The spike has been
over-invested in LLM-as-extractor (a role a better deterministic miner largely
subsumes) and under-invested in LLM-as-diff-reviewer (the role that addresses the
actual failure mode).

---

## 3. Gaps, errors, weaknesses that undermine the conclusions

Ranked by how much they damage the headline claims.

1. **No cross-bump test at all (fatal to the north-star claim).** Every wave 1-4
   conclusion is single-version. The synthesis claims a "production design" but has
   zero evidence it survives a bump - the one thing the design must do. The
   STUDY_SYNTHESIS "What is VALIDATED" list does not include bump-currency because
   it cannot. This is not a caveat; it is the whole point left untested.

2. **The study floor understates production (invalidates the recall deltas).** Wave
   4 admits `improved-det-v2` predates the production pydantic-lift and misses
   PluginConfig. Every "LLM lift over floor" number is therefore inflated against a
   sandbagged baseline. The correct comparison - production-det floor vs
   production-det + LLM - was never run. The headline "~69%/59% hybrid recall, LLM
   adds the tail" is measured against the wrong floor.

3. **N=2 cells, single-shot, directional - and the conclusions are stated as
   production design.** The synthesis says "Production design = floor + 32B-coder +
   gate" on the basis of 2 frozen cells. The caveats section concedes "N=2,
   directional" while the body recommends a production stack. The confidence stated
   exceeds the evidence.

4. **Construction-grounding is qwen2.5-coder-line-specific (kills generality).**
   The one OSS lever that breaks the tensorrt infra wall does NOT generalise to
   qwen3-coder (MoE) or deepseek (Wave 4 table: both regress on tensorrt). A
   production workflow pinned to a single model FAMILY's quirk is the opposite of
   "general" - and that model line will itself be deprecated within the cadence of
   the upstream bumps the workflow must track. This is a hidden version-currency
   problem inside the proposed solution.

5. **GT incompleteness is structural, not incidental.** The GT is grown by folding
   the same workflow's confirms back in. Recall-vs-GT is therefore partly
   measuring the workflow against its own prior output. The 20 tensorrt confirms
   that are "growth not recall" show the GT was missing real constraints; this means
   the recall denominators across waves are not comparable and the "ceiling"
   framing is unreliable.

6. **Per-bump COST was never measured directly.** The north star is "affordable on
   every bump" yet the study deliberately produces only an ORDINAL cost story
   (det < OSS < Opus). For a CI-affordability claim that is insufficient: the
   load-bearing question is the absolute wall-clock + GPU-minutes of the 32B-coder
   pass per bump per engine on the CI runner, and whether it fits the budget. "It is
   cheaper than Opus" does not establish "it is CI-affordable." No cell measured the
   actual per-bump cost of the recommended stack.

7. **Confounds in the tier sweep.** Opus ran whole-source single-call; OSS ran
   chunked 16k - acknowledged but it means the tier comparison is also a
   call-shape comparison. The llama-70B run swapped to a containerized ollama
   mid-study. These are logged but they blur the "code-tuned 32B > general 70B"
   claim that the synthesis leans on.

Which undermine the CONCLUSIONS most: #1 (no bump test) and #2 (sandbagged floor)
together mean the two headline conclusions - "the hybrid is the production design"
and "the LLM adds a valuable tail" - are both unproven. #6 means even "cheap" is
unestablished in absolute terms.

---

## 4. What to try instead, ranked by north-star value

### #1 (DO THIS NEXT) - The cross-bump degradation-signal binary

The single experiment that tests the product property. On a REAL bump-pair with
existing GT (tensorrt v0_21_0 -> v1_0_0, or vllm v0_18_1 -> v0_19_1 where the cliff
is already documented):
- Carry v_old's mined catalogue forward UNCHANGED to v_new's container.
- Run the runtime gate against v_new. Measure: how many carried invariants still
  confirm, how many now fail/skip/infra, how many NEW v_new constraints are missed.
- **The binary:** does gate-acceptance-rate DROP measurably when the surface churns
  (the degradation signal fires), and would a human reviewing only the gate's
  accept/reject delta know the catalogue went stale - with NO manual source diff?
- Score every workflow (W-G det-floor, W-G + LLM-diff-review, pure-LLM) on this
  binary, not on recall.

This is cheap (GT exists, gate exists, containers exist), it is the un-run item
flagged twice, and it is the ONLY experiment that can validate or kill the
architecture. Everything else is secondary until this runs.

### #2 - Production-det floor + Primitive-8, re-measured across the cliff

Replace the sandbagged `improved-det-v2` study floor with the actual production
pydantic-lift + Primitive 8, and re-run the vllm cliff (Wave 2.5 showed +44%
recovery). Then ask: with the REAL floor, how much does the LLM actually add on a
BUMP (not a frozen cell)? This directly tests claim 2 and likely shows the LLM tail
shrinks to near-zero on the consumed-invariant surface. Highest-ROI deterministic
engineering item, per Wave 2's own conclusion.

### #3 - The diff-based "what changed" extraction (D2/D3 + LLM diff-review)

The bump-survivability finding says robustness needs EITHER a declarative primitive
(#2) OR an LLM that reads the v_old->v_new diff and recovers the residual. Test the
diff-review role: give the LLM only the source DIFF between two versions and the
prior catalogue, ask "what constraint surfaces changed; what might the floor now
miss." Score on cliff-recovery and silent-collapse-caught, NOT recall. This is the
LLM role that attacks the actual failure mode, is cheap (diff << full source), and
plays to the H9-validated diagnose strength rather than the weak extract role.

### #4 - Directly measure per-bump cost of the recommended stack

One clean measurement: wall-clock + GPU-minutes for the production-det floor + one
construction-grounded 32B-coder pass + the full runtime-gate sweep, per engine, on
the CI runner profile. Turn the ordinal story into one absolute number per bump and
check it against the CI budget. Without this, "CI-affordable" is an assertion.

### #5 - The auto-PR-for-review workflow (W-F end to end)

The realistic production shape is not "fully autonomous re-mine"; it is "bump
triggers re-mine + gate + auto-open a PR with the catalogue delta for a
rubber-stamp." Test whether that PR's content needs SUBSTANTIVE review or is a
rubber-stamp (the W-F success criterion). This is the operational property the
service actually needs and it has never been exercised end to end.

### #6 (LOWER) - Finish the agentic re-diagnosis ONLY to close it out

The WAVE4_RECONCILIATION_MAP makes a clean case that h7's agentic-0 was a harness
artefact and proposes a fixed run. Fine - but agentic is a worse extract strategy
than construction-grounding by the spike's own data, and extract is the role we
should be de-emphasising. Run it only if cheap and only to bank the negative result
cleanly; do NOT let it become the next multi-wave investment. It is the least
north-star-relevant of the open items.

### Explicitly DOWN-RANK
- Another point on the size x shape matrix (more tiers, more shapes, self-consistency
  k-vote) on the frozen cells. This is the trap the spike is already in.
- Folding more verified-real confirms into the frozen-cell GT. It grows a number
  nobody downstream consumes.

---

## Bottom line

The spike built a rigorous, well-instrumented recall benchmark for LLM invariant
extraction on two frozen engine versions, and a genuinely useful gate + soundness
guards. But it has been optimising the wrong objective for four waves: single-version
recall, not cross-bump currency. The one experiment that tests the product property
- and the one deterministic result that actually moves it (the Primitive-8 cliff
recovery) - were found early by the prior PoC and then abandoned. The LLM-as-extractor
investment is largely unjustified: the deterministic floor (properly measured) does
the load-bearing work, the runtime consumes only a few dozen error/dormant rules, and
the comprehensive cross-field tail has no runtime consumer. The defensible LLM role is
diff-review/diagnose against a bump, which the spike de-prioritised. Next action: run
the cross-bump degradation-signal binary on a real bump-pair, against the production
floor, scored on "did the workflow stay current and notice when it didn't" - not on
recall.
