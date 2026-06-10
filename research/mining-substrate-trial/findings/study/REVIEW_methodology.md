# Methodological soundness review - Phase 1 mining study (waves 1-4)

Independent adversarial audit of HOW the wave 1-4 experiments were run, measured,
and scored. Scope: the runners (`scripts/phase1/*`, `scripts/wave2_llm_*`,
`scripts/study_gt_pilot.py`), the runtime gate (`scripts/validate_invariants.py`,
`scripts/_invariant_validation_common.py`), the identity/scoring helpers
(`scripts/gt_adapter.py`, `scripts/gt_scoring.py`), and the GT files. Paths in
`scripts/...` are repo-root; phase runners are under
`research/mining-substrate-trial/scripts/`.

Each finding is tagged **[INVALIDATES]** (undercuts a stated conclusion as written),
**[CAVEAT]** (the conclusion survives but the number/claim must be hedged), or
**[FRAGILITY]** (latent bug / silent-failure risk, not necessarily exercised). Items
are ordered by severity within each metric/confound/bug/repro group.

Note: several of these are already self-disclosed in the findings' own caveats or
in code comments. They are restated here because the *headline claims* and the
STUDY_SYNTHESIS production recommendation do not consistently carry the caveat, and
because the magnitudes were not previously quantified.

---

## 1. Metric soundness

### 1.1 [INVALIDATES, scoped] The type-coercion + cross-field soundness guards are pydantic-only; they silently do not protect the vLLM msgspec sampling-param surface

`SamplingParams` (and `BeamSearchParams`) in vLLM 0.19.1 are `msgspec.Struct`,
not pydantic (`/tmp/trial_vllm_v0_19_1_venv/src/vllm/sampling_params.py:156-163`).
Two consequences in the gate:

- `_extract_error_locs` (`_invariant_validation_common.py:292-325`) duck-types on
  `.errors()`. `msgspec.ValidationError` has no `.errors()` (verified), so it
  returns `None` -> `exception_locs = None`. The cross-field field-level rejection
  in `validate_invariants.py:832` (`elif _is_cross_field(invariant) and
  pos.exception_locs:`) therefore *can never fire* for a msgspec class. The Wave 2
  "attribute cross-field confirms by error locus" fix is pydantic-only.
- `_positive_is_type_coercion_artifact` (`validate_invariants.py:526-537`) parses
  error codes via `re.findall(r"type=([a-z_]+)", ...)`. A msgspec error message is
  `Expected \`int\`, got \`str\` - at \`$.n\`` (verified) - it contains no `type=`
  token, so `etypes` is empty and the guard returns `False` at line 533. The Wave 4
  type-coercion-artifact fix is also pydantic-only.

So for any vLLM SamplingParams invariant, a positive probe that confirms only
because it tripped a msgspec *type* error on a numeric-/literal-labelled rule (the
exact spurious class the Wave 4 fix was built to reject, e.g. the synth sentinel
`"__llem_invalid_probe_value__"` fed to a numeric field) is attributed via the
permissive substring path (`_raise_attributable_to` line 567) and is NOT caught by
either guard. The Wave 4 claim that the guard "makes raw strategy numbers
trustworthy without per-run manual verification" (PHASE1_WAVE4_FINDINGS.md:80) is
false for the sampling-param surface - which is a large share of the vLLM cross-field
finds (min_tokens<=max_tokens, repetition min<=max, structured-outputs exactly-one
all live in `sampling_params.py`).

Scope correctly: vLLM *config* classes (LoRAConfig, ParallelConfig, ...) and TRT-LLM
`*LlmArgs` ARE pydantic, so both guards work there. The gap is the msgspec sampling
surface only. This does not retract the verified-real counts (those went through
manual adversarial review), but it INVALIDATES the "guards make manual verification
unnecessary" claim and means every msgspec-surfaced confirm still requires the manual
backstop.

Fix: in `_extract_error_locs`, also handle `msgspec.ValidationError` by parsing the
`at \`$.field\`` locus out of the message (or import msgspec and special-case it);
in `_positive_is_type_coercion_artifact`, recognise the msgspec `Expected \`T\`, got`
message shape as a type-coercion signal. Add a regression probe with a msgspec class.

### 1.2 [CAVEAT, large] Recall-vs-GT is partly circular: the GT is built from the same deterministic + LLM sources whose recall it scores

`study_gt_pilot.configure()` unions sources `passA/passB` (Opus), `mech`
(improved-det-v2), `prod` (production static miner), `poc`, and `llm` (the folded
Wave 2 confirms) into the GT (`study_gt_pilot.py:80-104`). The code itself flags
this at `study_gt_pilot.py:96-100` ("deterministic recall vs GT is partly circular
and must be caveated"). Quantified against the committed GTs:

- vLLM 0.19.1: GT denom = 80 tolerant keys; 9 are deterministic-contributed, 8 are
  reachable ONLY via a deterministic source, 6 ONLY via the folded LLM source.
- tensorrt 1.2.1: GT denom = 61; 29 deterministic-contributed, 16 ONLY-deterministic,
  1 ONLY-LLM.

So the floor's "57% tensorrt recall" is scored against a denominator ~half of which
the floor (mech+prod) itself authored - it is partly measuring self-consistency, not
coverage of an independent oracle. Symmetrically, the 8 (vLLM) det-only keys are
keys the LLM strategies *structurally cannot reach* yet sit in their recall
denominator, deflating every LLM recall number by a fixed ~10%. Cross-strategy
*comparisons* among LLM strategies are roughly fair (same denominator, same handicap),
but any *absolute* recall number and any *floor-vs-LLM* comparison is biased by who
authored the denominator.

This directly weakens the Wave 4 "production floor pushes tensorrt materially higher"
resolution (PHASE1_WAVE4_FINDINGS.md:135-146): `prod` is simultaneously a GT source
AND the artefact whose recall is being celebrated. The PluginConfig recovery is real,
but "production hybrid recall is materially higher" is shown by adding the scoring
source to the scored set, which cannot raise recall above what that source defines.

Fix: report recall against a HELD-OUT oracle (a GT subset built only from sources NOT
under test), or at minimum publish the per-source provenance of the denominator next
to every recall number. Mark floor-vs-GT as a lower bound on a self-authored set, not
a coverage fraction.

### 1.3 [CAVEAT] Recall denominator = confirmed-only GT; the ~155 unverified entries per cell are excluded, so "recall" is recall-against-the-gate-confirmable-subset

`write_pilot_gt` writes `confirmed` (105 vLLM / 75 tensorrt) and `unverified`
(157 / 154) separately (`study_gt_pilot.py:335-374`); every recall metric keys only
on `confirmed` (`wave4_pure.gt_keys` reads `g.get("confirmed")`,
`wave4_pure.py:84-89`). The unverified entries are real mined constraints the gate
could not construct/probe (entangled classes, cross-field-without-kwargs, etc.). So
the headline "69% / 59% GT recall" is recall against the gate-CONFIRMABLE surface,
not the mined surface, and certainly not the true engine invariant surface. The
denominator is itself selected by gate-constructibility, which correlates with the
single-field/simple classes the floor already covers - inflating apparent recall.
The findings acknowledge "the GT is itself incomplete" (PHASE1_WAVE4_FINDINGS.md:112)
but the 69%/59% numbers are quoted without this denominator qualifier in
STUDY_SYNTHESIS.md:79.

Fix: state the denominator explicitly as "gate-confirmed GT (N=80/61)" everywhere the
percentage appears; never let "69% GT recall" stand unqualified.

### 1.4 [CAVEAT] Tolerant-key recall credits a confirmed invariant against a DIFFERENT GT constraint on the same field

Recall is `conf_tol & gt_tol` where the key is `(leaf_field, coarse_bucket)` and
`coarse_bucket` collapses all of `gt/lt/ge/le/range -> "numeric"`,
`in/not_in/not_equal -> "membership"`, etc. (`gt_scoring.py:148-159`,
`_coarse_from_scorer_pk`). So a confirmed `max_loras < 8` is credited against a GT
`max_loras >= 1` (same leaf, both `numeric`) even though they are different
constraints in different directions. The gate guarantees the confirmed entry is a
REAL rule on that field, but the recall match does not require it to be the SAME rule
as the GT entry it is credited against. Where a field carries multiple distinct
numeric/membership constraints, recall is over-credited. `study_gt_pilot` itself
counts GROWTH at the finer `ckey` (with canonical predicate value) grain precisely
because the tolerant grain "over-collapses" (`study_gt_pilot.py:121-127`,
`collapsed_tolerant_keys` metric) - but the wave4 *recall* metric uses the coarse
grain anyway.

Fix: report recall at the `ckey` (constraint-value) grain as the primary number, with
tolerant recall as a generous upper bound. The machinery already exists.

### 1.5 [CAVEAT] "Confirmed" precision is not comparable across strategies because the denominator (`gateable`) varies with construction reach, not extraction quality

`gate_confirmed_precision = confirmed / gateable` (e.g. `wave4_pure.py:121`).
`gateable` = "has a native_type" (`study_gt_pilot._gateable`, line 185-190), so a
strategy that proposes many unconstructible-but-typed candidates gets a low precision
that reflects construction reach, not extraction correctness; a strategy that proposes
fewer, simpler candidates scores higher. The cross-strategy precision table in
PHASE1_WAVE3_FINDINGS.md:42-51 (raw precision 0.018 .. 0.25) is being read as a
quality signal but is confounded by candidate constructibility and count. Use it only
directionally.

### 1.6 [INVALIDATES auditability] Verified-real counts are prose-only; the adversarial Opus verification is not a committed, reproducible artefact

The "verified-real" counts (8 of 9 Wave 2, 20 of 23 tensorrt Wave 4, etc.) are the
load-bearing soundness backstop for every cross-field confirm (the gate guards are
pydantic-only, see 1.1). But the verification exists only as hand-written tables in
the findings MD (e.g. PHASE1_WAVE4_FINDINGS.md:48-58); there is no committed Opus
verify prompt, transcript, or per-candidate machine-readable verdict file under
`phase1_wave{2,3,4}/`. The `*_CONFIRMED.yaml` dumps (`wave3_dump_confirmed.py`) carry
the raised error for the reviewer but not the reviewer's verdict + rationale. So the
single most important soundness claim in the study (which cross-field confirms are
REAL) is not independently auditable or reproducible from the repo - a re-run would
have to redo the manual review from scratch and might disagree.

Fix: commit, per confirmed cross-field candidate, the verify prompt, the model output,
and a structured `{id, verdict, source_file_line, rationale}` record. Treat the
verify as a gated, logged step like the gate itself.

---

## 2. Confounds

### 2.1 [INVALIDATES the cross-model comparison as a clean tier effect] OSS-chunked vs Opus-whole-source is a strategy confound entangled with the tier axis

The Opus rung sees the WHOLE validator source in one call; OSS rungs see greedily
packed `<=22k`-char chunks of validator BODIES only (`wave2_llm_source.chunk_validator_source`,
`wave1.gen_oss` vs the Opus `--proposed` path). The findings log this
(PHASE1_WAVE1_FINDINGS.md:88-92, PHASE1_WAVE3_FINDINGS.md:139-140) and correctly call
OSS counts "lower bounds". But the Wave 3 2x2 ("scale is the threshold; a code-tuned
32B beats a general 70B") draws a *tier* conclusion across rungs that differ in
call-shape AND context window AND chunking - all OSS models share the chunk handicap,
so the OSS-internal 2x2 (gemma/qwen-7b/qwen-32b/llama-70b) is reasonably controlled,
but the "Opus = ceiling" anchor is NOT on the same harness and cannot be read as a
pure tier point. The synthesis's ordinal "small < mid < Opus" conflates tier with
harness for the Opus rung specifically.

Material? For the OSS-internal scale-threshold finding: NO (all OSS share the harness).
For any Opus-vs-OSS gap attribution: YES - confounded.

### 2.2 [CAVEAT] Chunking sees only validator BODIES, structurally excluding pydantic Literal/Field type constraints - this is the real "ceiling", and it is a harness artefact not a model limit

`chunk_validator_source` emits only function/method bodies containing a trigger
substring (`raise`, `isinstance`, `model_validator`, ...,
`wave2_llm_source.py:19-37`). Pydantic `Literal[...]`/enum FIELD constraints with no
explicit `raise` are never in a chunk. The Wave 4 residual analysis discovers exactly
this for PluginConfig (PHASE1_WAVE4_FINDINGS.md:128-134) but frames it as a source-set
gap; it is more fundamentally a CHUNKER design choice that caps every LLM strategy's
reachable surface to raise-bearing validators. Any "LLM ceiling" conclusion is partly
a measurement of the chunker, not the model. (The construction-grounding AST pass
captures Literal values but, as the findings note, the prompt mines from the source
chunk not the signatures.)

### 2.3 [INVALIDATES clean comparison for affected runs] Mid-study ollama server change (shared :11434 vs containerized :11435) is an uncontrolled infra variable, and the default server differs BY RUNNER

`wave2_llm_cells.OLLAMA` defaults to `:11435` (containerized); `wave1/wave4_pure`
override to `:11434` (shared host) via `WAVE_OLLAMA`; `wave4_agentic` defaults to
`:11435`. So agentic ran against a different ollama than construct/pure/selfconsistency
by default, and Wave 3 explicitly switched the 70B to a containerized instance
mid-sweep because the shared one was memory-capped and 500'd
(PHASE1_WAVE3_FINDINGS.md:127-132). Different ollama builds/quantization defaults/
context handling across the same study is an uncontrolled variable. For the 70B the
findings note params were "held constant (num_ctx 16384, temp 0)" but the SERVER
differed - and the agentic "OSS emits 0" conclusion (a key Wave 4 claim) was produced
on the :11435 server while its construction-grounding comparator may have run on
:11434. The agentic-is-poor conclusion is plausible on mechanism (tool-call text
flakiness) but the head-to-head vs construction-grounding is not server-controlled.

Fix: pin one ollama (image digest + server) for the entire study; record the server
endpoint + ollama version in every result JSON.

### 2.4 [CAVEAT] temp=0 determinism vs the self-consistency temp - and self-consistency is still UNRUN

Single-shot runs use temp=0 (`ollama_generate` default, `wave2_llm_cells.py:67`);
self-consistency uses temp=0.7, k=3 (`wave4_selfconsistency.py`). That is the correct
design, BUT the self-consistency result is still a literal `<PLACEHOLDER - result
pending>` in STUDY_SYNTHESIS.md:69 while the synthesis already states the production
design as settled. A k-vote union at temp>0 could materially change the
recall/precision frontier (it is the obvious lever for the residual cross-field tail).
The synthesis presents a closed conclusion over an open experiment.

Also: temp=0 is not bit-reproducible across ollama/GPU/driver versions (sampling is
deterministic only given identical backend); combined with 2.3 this means even the
temp=0 runs are not guaranteed replayable.

### 2.5 [INVALIDATES generality] N=2 cells, single-shot, and construction-grounding is model-line-specific

Every wave is N=2 (vLLM 0.19.1 + tensorrt 1.2.1), mostly single-shot. The findings
flag this consistently ("directional, not a frontier point"). Construction-grounding
- the headline OSS lever - works for the qwen2.5-coder line but NOT qwen3-coder (MoE)
or deepseek (PHASE1_WAVE4_FINDINGS.md:35-46). So the production recommendation
("construction-grounded local ~32B code-model") rests on ONE model line, on TWO cells,
single-shot. That is a hypothesis-generating result, not a validated production
design. The STUDY_SYNTHESIS "production design = ..." framing (lines 73-85) overstates
the evidential weight; it should read as a recommended candidate to validate at N>2.

### 2.6 [CAVEAT] Study floor (improved-det-v2) vs production pydantic-lift - the floor under test is NOT the production deterministic path

The floor is `w2-a-improved-det-v2` (`wave2_llm_cells.py:33`,
`study_gt_pilot._MECH_TEMPLATES`), which the findings concede is "older/narrower than
the production pydantic-lift" and misses PluginConfig Literals
(PHASE1_WAVE4_FINDINGS.md:141). So the "floor recall" numbers (44/80, 35/61)
understate the deterministic path that production actually ships, and the
floor-vs-LLM delta is measured against a strawman floor. Combined with 1.2 (prod is
ALSO in the GT), the deterministic story is doubly entangled: scored against a partly
self-authored GT, using a floor that is not the production one. The "production recall
is higher" hand-wave is directionally true but unmeasured - it was never run with the
actual production lift as the floor.

---

## 3. Bugs / fragility

### 3.1 [FRAGILITY, material to construction-grounding] The AST signature extractor ignores inheritance, mishandles `Field(...)`/msgspec required markers, and truncates

`_class_signature` (`wave4_construct.py:35-55`) only reads `ast.AnnAssign` at the
top level of each `ClassDef` and classifies "required" as `item.value is None`. This
mis-models the actual construction contract:

- INHERITANCE: it never merges base-class fields. `SamplingParams(PydanticMsgspecMixin,
  msgspec.Struct, ...)` and the TRT-LLM `*LlmArgs` hierarchy have required fields in
  bases that are invisible. So the "constructor signature" injected for an inheriting
  class can OMIT required fields - the exact infra-wall construction-grounding claims
  to break. Construction-grounding's failure on some classes/models may be the
  extractor missing inherited required fields, not a model capability limit. This
  partially confounds the "construction-grounding is model-specific" conclusion.
- `x: int = Field(default=1, ge=1)` -> `item.value` is a Call node, so classified
  OPTIONAL with a default string `Field(default=1, ge=1)` truncated at 28 chars
  (`dflt[:28]`, line 54) - noisy/uninformative to the model.
- A field marked required via `= Field(...)` with no default (Ellipsis) would be
  mislabelled OPTIONAL.
- `format_sig_block` caps optional fields at 25 (`opt[:25]`, line 82), so large config
  classes get a partial signature.

Fix: resolve the MRO and merge base-class AnnAssign fields; detect `Field(...)` /
`msgspec.field(...)` and read the default/required marker from the call; stop
truncating required fields. At minimum, document that signatures are this-class-only.

### 3.2 [FRAGILITY] Tolerant-key dedup silently collapses distinct LLM constraints on the same (leaf, bucket)

`dedup_internal` / `dedup_vs_floor` key on `tolerant_key = (leaf, coarse_bucket)` and
drop on collision (`wave4_pure.py:59-70`, `wave1.py:72-83`). Two genuinely different
constraints the LLM found on the same field+bucket (e.g. `x > 0` and `x < 100`, both
`numeric`) collapse to one - the first wins, the second is discarded BEFORE gating. So
the LLM can be denied credit for a real second constraint, and the discarded one is
never gated. This caps LLM-measured coverage below its true output and interacts with
1.4 (recall also coarse). The GT pipeline deliberately moved to the finer `ckey` grain
for exactly this reason (`study_gt_pilot.py:121-127`); the wave runners' candidate
dedup did not follow.

Fix: dedup candidates at `ckey` grain (leaf, bucket, canonical predicate value), as
the GT does.

### 3.3 [FRAGILITY] `wave3_dump_confirmed._looks_cross_field` flags every standard-shape entry as cross-field

`_looks_cross_field` does `if len(match) > 1` (`wave3_dump_confirmed.py:35`), but
`match` is `{engine: ..., fields: {...}}`, so `len(match)` is always 2 for the normal
shape -> every entry is "cross-field". It should test `len(match.get("fields", {}))`.
Impact is limited (the script dumps the full inv for the human reviewer regardless, and
the reviewer is told to classify themselves), so this is a reviewer-hint bug, not a
scoring bug - but the cross-field COUNTS printed by this script are meaningless and
must not be quoted.

### 3.4 [FRAGILITY] `_raise_attributable_to` permissive fallback for non-pydantic raises admits substring-coincidence confirms

For `exception_locs is None` (non-pydantic: msgspec, plain ValueError, transformers
composed errors), attribution falls back to `leaf in haystack`
(`validate_invariants.py:567`) - bare substring presence of the field name anywhere in
the message. A cross-field rule whose message mentions the probed field in a
remediation clause confirms spuriously. The model-level (`locs == ()`) path has the
`_leaf_is_message_subject` "earliest-named field" tightening, but the `None` path does
not. Given 1.1 (all msgspec raises are `None`), this is the live attribution path for
the entire vLLM sampling surface. Combined with 1.1, vLLM sampling confirms rest
entirely on the permissive substring check + manual review.

### 3.5 [FRAGILITY] Internals-guard is applied in analysis, not in the miner/gate - so raw confirmed counts include internals

The internals-guard (drop underscore-private fields, type-trivia, observability,
launch-state) is applied by the human during verification, not by the gate or the
runner (PHASE1_WAVE3_FINDINGS.md:70-84, STUDY_SYNTHESIS.md:111 "applied in analysis,
not yet in the miner"). So every raw `confirmed`/`precision`/`recall_frac` number a
runner emits to its JSON includes internals (e.g. the llama-70b `_api_process_rank`
confirm) until a human subtracts them. Any automated reading of the result JSONs
overcounts. The verified-real columns are correct; the raw columns are not
internals-clean.

### 3.6 [FRAGILITY] Lenient YAML reparse (`_parse_per_entry`) and the qwen-7b key-recovery silently change what is scored

`parse_invariants` falls back to per-entry salvage on truncated/invalid YAML
(`wave2_llm_cells.py:119-184`), and qwen-7b's output under the wrong root key `i:` was
recovered by a separate lenient reparse (`wave3_reparse_lenient.py`,
PHASE1_WAVE3_FINDINGS.md:133-138). Both are defensible for a "capability lower bound",
but they mean the scored corpus is a post-processed derivative of the raw model output,
and the post-processing differs per model (only qwen-7b got the key-recovery). The
format-following failure is itself a capability signal that the lenient pass erases
from the headline count while reporting it separately - a reader comparing raw
confirmed counts across models is comparing differently-salvaged corpora.

### 3.7 [FRAGILITY] `_TRTLLM_MODEL_PLACEHOLDER` injection can mask construction failures as confirms/negatives

`_construct_trtllm` injects `model="/tmp/llem-validation-gate-model-placeholder"` for
any `*LlmArgs` class (`validate_invariants.py:275-278`). The comment argues
construction stops at validator-pass time before disk read. If a TRT-LLM version
moved the model-path check earlier (before the invariant's validator), the negative
would raise on the placeholder and the entry would be scored `failed`/`infra` rather
than confirmed - a silent version-fragility. Not necessarily wrong today, but it is an
injected value the negative-must-not-raise check depends on, untested across the
version axis the study cares about.

---

## 4. Reproducibility

### 4.1 [INVALIDATES exact replay] The source the LLM saw lives in ephemeral `/tmp` venvs, not the repo

`source_files_for` reads `/tmp/trial_{engine}_{vslug}_venv/src/...`
(`wave2_llm_source.py:122-163`). These venvs currently exist but are `/tmp`-ephemeral
(reaped on cleanup/reboot). The exact source bytes that were chunked and fed to every
model are NOT committed. The chunker output is also not committed. So the precise model
INPUT is unreconstructable from the repo once `/tmp` is cleared - only the engine
version pin lets you *re-install* and hope the bytes match. For a study whose finding
is "the chunker's validator-body extraction is the ceiling", the chunker input is
load-bearing and should be archived.

Fix: snapshot (or hash + commit) the exact chunked prompts per cell/model alongside the
result JSON. The committed `*_raw.txt` capture model OUTPUT, not the INPUT chunks.

### 4.2 [INVALIDATES result provenance] Most per-cell result JSONs live only in `/tmp` and are uncommitted

Every wave runner writes to `/tmp/phase1_*` (e.g. `wave4_pure.py:141`,
`wave1.py:171`). The committed `phase1_wave{2,3,4}/results/*.json` cover only a subset;
wave1's results and many wave4 cells exist only in `/tmp` (confirmed present now, but
ephemeral). The headline tables in the findings cannot all be traced to a committed
artefact. Two wave4 result files are currently untracked (`git status` shows
`w4c_qwen2_5_32b_vllm_v0_19_1*` as `??`). The study is not reconstructible end-to-end
from committed state.

### 4.3 [CAVEAT] Container reaping + in-place GT mutation make the GT a moving target

`study_gt_pilot.write_pilot_gt` OVERWRITES the committed `PILOT_GT.yaml` in place
(`study_gt_pilot.py:372`), and the Wave 2 fold-in mutated the GT mid-study (vLLM
98->105). The round0b gate was made deliberately non-destructive to avoid this
(`round0b/gate.py` docstring), which shows the destructive default was a recognised
hazard. Because recall is scored against whatever the GT currently says, a re-run of an
early wave against the post-fold-in GT yields different numbers than the wave's own
findings reported. There is no GT version pin in the result JSONs tying a recall number
to the GT revision it was scored against.

Fix: stamp every result JSON with the GT file's git SHA / content hash.

### 4.4 [POSITIVE] Prompt-locking discipline is sound

The locked-prompt discipline IS good and worth preserving: prompt bodies live in
committed `wave2_locked_prompts/*.md` and `phase1_wave4/*.md`, are hashed
(`7cd74960eab09e16`, gemma digest `f4031aab...`), and the preregs pin the hash +
model digests (PHASE1_WAVE3_PREREG.md:40,129-131). This half of reproducibility is
handled correctly; the gaps are in source/result/GT archival (4.1-4.3), not in prompt
control.

---

## Priority summary

INVALIDATES a conclusion as written (fix before relying on the claim):
- 1.1 guards are pydantic-only -> "no manual verification needed" is false for vLLM
  sampling; every msgspec confirm still needs the manual backstop.
- 1.6 verified-real review is not a committed artefact -> the core soundness claim is
  unauditable.
- 2.3 mid-study ollama server change, server differs by runner -> agentic-vs-grounding
  head-to-head not controlled.
- 2.5 N=2 / single model line -> "production design = construction-grounded 32B"
  overstated; it is a candidate to validate.
- 4.1 / 4.2 model input (chunks) and most results live only in `/tmp` -> not replayable.

CAVEAT (conclusion survives, number must be hedged):
- 1.2 recall-vs-GT is partly circular (quantified: tensorrt ~half the denom is
  det-authored).
- 1.3 recall denom = confirmed-only subset, not the invariant surface.
- 1.4 tolerant-key recall over-credits across distinct constraints.
- 1.5 precision denominator confounded by constructibility.
- 2.1 OSS-chunk vs Opus-whole-source confounds the Opus anchor.
- 2.2 chunker sees only validator bodies -> the "ceiling" is partly the harness.
- 2.4 self-consistency still UNRUN while synthesis presents a closed conclusion.
- 2.6 study floor != production floor; the delta is vs a strawman floor.
- 4.3 in-place GT mutation -> no GT-revision pin on recall numbers.

FRAGILITY (latent bug / silent mis-scoring risk):
- 3.1 AST extractor ignores inheritance + Field/msgspec markers (confounds
  construction-grounding's model-specificity).
- 3.2 candidate dedup at coarse grain drops distinct real constraints pre-gate.
- 3.3 `_looks_cross_field` always-true (reviewer-hint only).
- 3.4 permissive substring attribution on non-pydantic raises.
- 3.5 internals-guard not in the pipeline -> raw JSON counts include internals.
- 3.6 per-model-divergent lenient reparse changes the scored corpus.
- 3.7 TRT-LLM model-placeholder injection is an untested version-fragility.

The two cheapest high-value fixes: (a) make `_extract_error_locs` /
`_positive_is_type_coercion_artifact` msgspec-aware (closes 1.1, the only
INVALIDATES-class *code* bug), and (b) commit the chunked model inputs + the
adversarial-verify records (closes 4.1 and 1.6, the two reproducibility/auditability
INVALIDATES items). Neither requires re-running the GPU experiments.
