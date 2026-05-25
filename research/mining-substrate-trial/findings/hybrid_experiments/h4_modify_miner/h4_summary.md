# H4 cross-engine summary - LLM-modifies-miner

**Pattern:** Tier 1, user-priority. The LLM reads the AST static-miner
producer source + engine source excerpts + (a)'s current output, and
proposes structured walker patches (diagnoses + anchor/replacement
edits). Patches applied to isolated copies; patched walker re-run; output
diffed against canonical reference + today's baseline.

**Backend:** container Ollama @ port 11435, model `llama3.1:70b` (q4_K_M),
num_ctx=32768, temperature=0.

**Cells run:** transformers v4.57.3, vllm v0.7.3, tensorrt v0.21.0
(all active versions).

**Wall-clock:** 49.9 + 78.4 + 58.3 = 186.6s LLM time total + ~30s
subprocess overhead per engine.

## Top-level numbers

| Engine | Canon | Baseline | Patched | Diag | Patches | Applied | Recall lift vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| transformers v4.57.3 | 41 | 28 | 28 | 1 | 1 | 0 | 0% (no patch landed) |
| vllm v0.7.3          | 26 | 66 | 0 (crash) | 3 | 3 | 1 | -100% (patch broke walker) |
| tensorrt v0.21.0     | 35 | 38 | 0 (crash) | 2 | 1 | 1 | -100% (patch broke walker) |
| **Aggregate**        | 102 | 132 | — | **6** | **5** | **2** | — |

**Headline:** H4 does NOT lift (a)'s recall in any of the three cells.
Two of three engines crash after patch application; the third's patch
never found its anchor. As a STRATEGY-internal score, H4 is a clean
negative.

## What the diagnoses captured (cross-engine pattern)

The 6 diagnoses are EXCELLENT despite the patches' brokenness. Every
diagnosis MATCHED the manually-curated `post_trial_a_gap_closure.md`
gap inventory text. The LLM correctly identified:

| Gap category | Engines where surfaced | LLM's structural read |
|---|---|---|
| if/elif/else `raise` in `else:` branch not descended | vllm (G-vllm-3) | Correct |
| Local-variable aliases of `self.<field>` not tracked | vllm (G-vllm-2) | Correct |
| Normalisation-only (no-raise) classes not in `_CLASS_TARGETS` | vllm (G-vllm-1) | Correct |
| Type-blind probe-value synthesis | tensorrt (G-trt-1) | Correct |
| Nested Pydantic-config classes not in walk list | tensorrt (G-trt-3) | Correct |
| Defensive imports for version bumps | transformers (G-trf-1) | Correct (restated) |

**Cross-engine cluster pattern observed:**
- **Branch-descent gaps** appear in vllm walker (G-vllm-3); the
  tensorrt walker already has the descent + else-handling (line ~445).
  Patch-once-per-walker per engine.
- **Nested-config gaps** appear in tensorrt (G-trt-3) and vllm
  (CacheConfig/QuantConfig analogues). NOT present in transformers
  (the BNB companion is already inlined via the transformers walker's
  companion-walker). The cross-engine ABSTRACTION: a `_NestedConfigDispatch`
  mixin that scans field annotations for known config types and adds
  them to the walk list.
- **Type-blindness** (probe synth) appears in tensorrt. In vllm the
  walker has limited type info (mixes dataclass + bare classes + pydantic).
  Same pattern, different engines.
- **Defensive-import hardening** is a META concern - it applies to all
  three walkers because each does live engine imports somewhere. The
  transformers walker is the only one with the AST-only fallback that
  the tensorrt walker exhibits natively (since tensorrt has no live
  install at this version on the trial host).

## Patches sorted by likely mergeability for spike's vllm/tensorrt refactor

Below is the H4 patch backlog re-organised by H4-output mergeability into
the spike branch's planned ~1800 LoC mining refactor (per
`bakeoff_A_refactor_analysis.md`). Each row maps to a single
post-trial spike-branch issue.

### Tier A: directly mergeable as spike issues (DIAGNOSIS text + small implementation)

1. **vllm G-vllm-3 (else-branch descent)**: ~30 LoC walker patch.
   Diagnosis correctly identifies the line range (728-733). Spike
   implementation: inline the `_handle_if` recursion to handle non-If
   orelse statements with negated predicates.
2. **tensorrt G-trt-1 (type-aware probe synth)**: ~30-50 LoC walker
   patch. Diagnosis correctly identifies the `_value_satisfying`
   function + the missing `field_type` parameter. Spike implementation:
   add field-type extraction via class-body AST + thread through
   Predicate + caller updates. The LLM's incomplete patch-attempt
   confirms the scope; a competent human-or-larger-LLM implementer
   closes it.

### Tier B: mergeable but larger scope (broader refactor / new modules)

3. **vllm G-vllm-1 (EngineArgs normalisation-as-dormant)**: ~50 LoC.
   Diagnosis correctly identifies that the gap is structural (no
   raises). Spike implementation: add EngineArgs to `_CLASS_TARGETS`
   + ensure `_detect_self_assign` fires on the `if self.X is None:
   self.X = default` pattern + emit severity=dormant invariants.
4. **vllm G-vllm-2 (local-var alias tracking)**: ~50-100 LoC. Spike
   issue: implement `_TrackedAliases` frame; map locals from
   `tokenizer_mode = self.tokenizer_mode.lower()` to `self.tokenizer_mode`
   for downstream predicate extraction.

### Tier C: needs broader refactor (architectural; the LLM correctly tagged this)

5. **tensorrt G-trt-3 (nested-config dispatch)**: 200-400 LoC. LLM
   self-tagged `needs_broader_refactor` and emitted no patch. The
   diagnosis text confirms the gap. Spike implementation: design a
   `_NestedConfigWalker` mixin that scans Pydantic class annotations
   for nested-config types and adds them to the walk list, with
   appropriate namespace prefixes.

### Tier D: exploratory / no clear value-add

6. **transformers G-trf-1 (defensive imports)**: the proposed patch
   hallucinated the import block; the transformers walker is already
   pure-AST and doesn't have the gross import surface the LLM imagined.
   The genuine brittleness signal (Phase 3a.2: -22 invariants on minor
   bump) is real but the fix isn't a localised walker patch - it's
   either accepting brittleness as the signal OR a structural change to
   the walker's `import inspect`-based source resolution.

## Limitations of H4 (negative findings)

1. **70B-q4 isn't a competent walker-maintenance engineer at single-
   shot scope.** It diagnoses well but writes pseudocode for the fix.
   Examples:
   - Refers to `_handle_else` (undefined helper).
   - Refers to `negated_conditions` (undefined variable).
   - Adds required positional arg to `_value_satisfying` without
     updating any caller.
2. **Anchor-text brittleness.** 3 of 5 patches had `anchor_text` that
   didn't match the walker source verbatim. The LLM reconstructs
   anchors from memory rather than copy-pasting; single-character
   whitespace drift breaks the apply.
3. **Engine-excerpt vs walker-source confusion.** The transformers
   patch's anchor hallucinated an import block based on the engine
   excerpt I provided in the prompt. Prompt revision needed for future
   variants: clearly label which text is the patch target.
4. **Single-file refactor blindness.** Adding a parameter to a
   function should update all callers. The LLM didn't see this.
5. **No exploratory loop.** H4 is single-shot; a multi-pass agentic
   variant (H7) where the LLM gets to see the subprocess crash and
   re-attempt the patch could close the gap.

## What this means for the trial-internal score

H4 fails as a STRATEGY-internal pattern: no recall lift, two crashed
walkers, one non-applying patch. As a Phase 4 ship-with-defaults
candidate it's not viable in this form.

But H4 succeeds as a **diagnostic accelerator** for the post-trial
spike refactor: 6 correct gap diagnoses + 4 directly-mergeable spike
issues (with text already written) constitute material design input.

For Phase 4 synthesis: do NOT score H4 against the same recall metric
as pure strategies. Score it on diagnoses-correctness (6/6) and
patches-mergeability-via-human-polish (4/6 patches map to clean spike
issues; 2/6 are either too-broad-for-localised-patch or hallucinated).

## Recommendation: next Phase 3b pattern

Given the LLM's strong diagnostic ability but weak fix-implementation
ability, the natural Phase 3b follow-ups are:

1. **H9 (LLM diagnoses gaps, no output mutation)** - run the
   diagnoses-only variant of H4 across additional cells (bumped
   versions, particularly). The cost is low (~50s/cell) and the
   diagnoses are the highest-value H4 product. This would broaden the
   spike-issue backlog text.
2. **H7 (agentic loop with tool use)** - give the LLM tools to
   `read_file`, `run_miner`, `score_against`. Let it iterate: edit ->
   run -> see error -> edit again. This is the natural cure for the
   single-shot refactor blindness observed in H4.
3. **H3 (LLM proposes -> (a) runtime verifies)** - this validates LLM
   output via the deterministic harness. Closes the loop H4 lacks.

H4 is the right pattern to have run FIRST (per the priority brief): it
produces post-trial-useful artefacts regardless of internal score.
Subsequent patterns should build on the lesson that LLM produces
diagnostic intelligence well + needs deterministic feedback for fix
intelligence.

## Artefacts

Per-engine results: `h4_modify_miner/<engine>/h4_results.md` +
structured `h4_summary.json` + `proposed_patches/*.json` + raw LLM
trail in `raw_llm_outputs/`.

Aggregate machine-readable: `h4_modify_miner/h4_aggregate.json`.

## Commitments

- The 4 Tier-A + Tier-B patches above MUST be filed as spike-branch
  issues post-trial (per `post_trial_a_gap_closure.md`'s commitment
  text). Even though H4 didn't produce working code, the diagnoses
  + structural reads accelerate the spike refactor.
- The 1 Tier-C patch becomes the design input for the
  `_NestedConfigWalker` mixin (per `bakeoff_A_refactor_analysis.md`'s
  cross-engine abstraction target).
