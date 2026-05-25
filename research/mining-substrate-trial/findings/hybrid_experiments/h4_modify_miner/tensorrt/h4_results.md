# H4 results: tensorrt v0.21.0

**Pattern:** H4 LLM-modifies-miner. Llama3.1:70b q4 reads the
source-driven walker (1670 LoC), gap inventory, and two engine excerpts
(`_value_satisfying` site + nested-config classes), emits structured
diagnoses + patches.

Run wall-clock: 49.9s.

## Counts

| Metric | Value |
|---|---|
| Canonical reference count | 35 |
| Baseline count (unpatched walker, re-run today) | 38 |
| Patched count | 0 (subprocess crashed) |
| Patched run OK | False |
| Diagnoses produced | 2 |
| Patches proposed | 1 |
| Patches applied | 1 |

**Baseline-vs-canonical drift (38 vs 35):** the walker emits 3 more
invariants today than at canonical-mint. Likely cause: minor walker
improvements (e.g. nested if/else descent already present in this
walker) emit extra candidates the canonical was mined without. The 38 is
today's (a) floor.

## Diagnoses (both CORRECT and STRUCTURALLY ALIGNED with gap inventory)

### G-trt-1: type-blind probe-value synthesis

> `_value_satisfying` returns a fixed value (`"x"`) regardless of the
> field's declared type, which can produce invalid probes at runtime.

Exactly the gap from `post_trial_a_gap_closure.md` G-trt-1. The LLM
correctly identified that this is a runtime-validation failure mode
(invariant emitted correctly but the probe is type-mismatched).

Proposed fix: thread declared field type into `Predicate` + use it in
`_value_satisfying`. Conceptually correct. The LLM's actual patch
implementation is too aggressive (see below).

### G-trt-3: nested-config dispatch

> The walker doesn't recurse into SchedulerConfig / QuantConfig /
> KvCacheConfig via class-level type references.

Correct. The walker's `_CLASS_TARGETS` doesn't include these nested
Pydantic config classes; their `@field_validator` and `@model_validator`
decorated methods don't fire.

Proposed fix: scan class body for nested-config-typed fields; auto-add
to walk list. Tagged `needs_broader_refactor` in proposed_fix_summary by
the LLM itself - honest scope assessment.

## Patches

### P-trt-1: thread field_type into `_value_satisfying`

- **Status:** APPLIED. Crashed at subprocess time.
- **LLM's anchor (found uniquely at line ~722):**
  ```python
  _value_satisfying(op: str, rhs: Any) -> Any:
      if op == "present":
          return rhs if rhs is not True else "x"
  ```
- **LLM's `new_code`:**
  ```python
  _value_satisfying(op: str, rhs: Any, field_type: type) -> Any:
      if op == "present":
          if field_type == int:
              return 0
          elif field_type == bool:
              return True
          elif field_type == str:
              return "x"
          elif field_type == list:
              return []
  ```
- **Why crashed:** the patch ADDED a required positional arg
  `field_type` to `_value_satisfying`, but did not update the callers
  (`_value_for` at line 712 still calls `_value_satisfying(p.op, p.rhs)`
  with only two args). Runtime: `TypeError: _value_satisfying() missing
  1 required positional argument: 'field_type'`.
- **Verdict:** the LLM understands the gap correctly, picked the right
  function, and even wrote plausible type-dispatch branches. But it
  failed to recognise that adding a parameter breaks all callers + the
  Predicate dataclass doesn't yet have a `field_type` slot to source the
  value from. A correct patch needs:
  1. Add `field_type: type | None = None` to `_Predicate`.
  2. Populate `field_type` during predicate extraction (parse the class
     body annotations).
  3. Pass `field_type` through `_value_for` to `_value_satisfying`.
  4. Update `_value_satisfying`'s implementation to handle the new arg.
  The LLM did step 4 (partially) but not 1-3.

## What worked

- 2-for-2 correct diagnoses; both align with the manual gap inventory.
- The LLM correctly tagged G-trt-3 as `needs_broader_refactor` (no
  patch attempt) which is the right scope call - no hallucinated
  trivial patch for an architecturally hard gap.
- One unique-anchor patch landed cleanly; the harness applied it.

## What didn't work

- **Single-file refactor blindness**: the LLM doesn't see that adding a
  parameter to a function requires updating callers. This is a known
  limit of single-shot LLM editing without runtime feedback.
- **Pydantic type-extraction is not free**: the LLM proposed
  `field_type` as if it were trivially available, but extracting field
  types from Pydantic models via class-body AST is itself non-trivial
  (the walker would need a new `_extract_field_types_from_class_body`
  helper, then route them into predicate extraction). Not a localised
  ~30-LoC patch.
- **No exploratory patch for G-trt-3**: appropriate - the LLM
  correctly assessed it as needing broader refactor and emitted no
  patch.

## Spike-refactor value

The DIAGNOSES are directly usable as spike-branch issue text:

1. **G-trt-1 (type-aware probe synthesis)** is the highest-value spike
   target. Phase 1 Day 2 scoped it at ~30 LoC inside the walker; this
   H4 run independently confirms the same target. The LLM's sketch is
   directionally correct (type dispatch); the spike implementation
   would route field types through Predicate + update callers.
2. **G-trt-3 (nested-config dispatch)** is a larger spike target (200-
   400 LoC per the gap inventory). H4 didn't propose a working patch
   (correctly). The diagnosis text confirms the gap; the architectural
   sketch is already in `post_trial_a_gap_closure.md`.

## Negative findings

- LLM doesn't propagate function-signature changes to callers within a
  single-shot prompt. To get a working type-aware probe patch, we'd
  need either:
  - Multi-pass LLM (let it edit, run, see error, edit again - this is
    pattern H7's agentic loop).
  - Larger LLM (Claude Sonnet/Opus is much stronger at this kind of
    multi-site edit).
- The LLM correctly self-limited on G-trt-3 ("needs_broader_refactor"),
  meaning the prompt's scope-honesty rubric was understood.

## Artefacts

- `tensorrt/raw_llm_outputs/{prompt.txt, raw_response.txt,
  diagnoses.json, patches.json}` - full LLM trail.
- `tensorrt/proposed_patches/P-trt-1.json` - the one proposed patch.
- `tensorrt/patched_producer/static_invariant_miner.py` - patched
  walker (with the broken signature change).
- `tensorrt/patched_outputs/subprocess_stderr.txt` - the `TypeError`.
- `tensorrt/baseline_unpatched_run/invariants.proposed.yaml` - 38-
  invariant baseline.
- `tensorrt/h4_summary.json` - structured summary.
