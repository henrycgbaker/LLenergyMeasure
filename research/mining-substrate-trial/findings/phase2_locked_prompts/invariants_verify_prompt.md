# Invariants verify prompt (Phase 2.5 pass 2)

Added: Phase 2.5. Locked. This is PASS 2 of the multi-pass refinement
pipeline; PASS 1 uses the round-3-locked `invariants_prompt.md`.

Purpose: REVIEW pass-1 output against the source. Flag invariants that
look wrong. Conservative bias - prefer "confirm" when unsure.

Source: `research/mining-substrate-trial/scripts/strategies/prompts.py::INVARIANTS_VERIFY_PROMPT_TEMPLATE`.

```text
You are a code analyser doing PASS 2 (verification) of multi-pass
invariant extraction. PASS 1 already extracted invariants from
{engine} v{engine_version} for ONE chunk of source. Your job is to
REVIEW each emitted invariant against the source and flag any that
look WRONG.

INPUT 1 - PASS 1 EXTRACTED INVARIANTS (the candidate list):

{pass1_invariants}

INPUT 2 - THE SOURCE PASS 1 READ:

{source}

OUTPUT FORMAT: a YAML document with TWO sections:

confirmed:
- <id-of-pass1-invariant>  # one ID per line, no further detail needed.

flagged:
- id: <id-of-pass1-invariant>
  reason: <one-line: what looks wrong>
  fix: <one of: "drop", "correct_severity:<new-severity>", "correct_predicate:<new-kind>", "correct_kwargs_positive">

RULES:
1. Every PASS 1 invariant MUST appear in EITHER `confirmed` OR `flagged`
   (not both, not neither). If you're unsure, place in `confirmed` (the
   bar for flagging is "obviously wrong against the source").
2. Flag reasons must be CONCRETE - cite the source line that contradicts
   the invariant, or note the specific shape mismatch.
3. NO markdown code fences. NO commentary outside the YAML.
4. First character must be `c` (from `confirmed:`).

CRITERIA FOR FLAGGING:
- Severity wrong: source has `raise ValueError` but invariant says
  `severity: dormant`; or source has `minor_issues[...] = ...` but
  invariant says `severity: error`.
- Predicate wrong: source has `if X.field not in {a, b}: raise`
  but invariant emits `predicate_kind: exact`.
- Kwargs_positive wrong: source's predicate is "value < 0 raises" but
  invariant's kwargs_positive shows `field: 1` (which would NOT trigger).
- Hallucinated: invariant references a field name that does NOT appear
  in the source.

Emit the YAML now:
```

## Reconciliation contract

The orchestrator (`llm_b_oss.extract_invariants_multipass`) consumes the
output:

- Entries listed in `confirmed:` are kept as-is from pass 1.
- Entries listed in `flagged:` with `fix: drop` are REMOVED from the
  merged result. Other `fix:` values (`correct_severity:*`,
  `correct_predicate:*`, `correct_kwargs_positive`) are recorded as
  adjacent observations but DO NOT auto-modify the invariant - the
  pass-1 prompt is locked and we don't second-guess it programmatically.

## Why this exists

Phase 2 calibration's round-3 lock left ~14pp of invariant recall on
the table (60.7% vs 75% target; later remeasured at 41.0% under the
fixed rubric). Two root causes:

1. Per-field invariants collapsed: the LLM emitted ONE entry for
   "temperature dormant when do_sample=False" instead of one per field
   (top_p/top_k/min_p/typical_p/...).
2. Cross-field invariants with a wrong primary field (LLM keyed on
   `do_sample` for all 7 sampling-only-when-greedy dormancies; the
   reference has each keyed on the SECOND field).

Pass 2 alone doesn't fix these (it can flag wrongs but not invent
missing). Pass 3 is the recall lift. Pass 2 is the precision floor.

## Status

Locked at Phase 2.5. Not iterated against the calibration cell.
