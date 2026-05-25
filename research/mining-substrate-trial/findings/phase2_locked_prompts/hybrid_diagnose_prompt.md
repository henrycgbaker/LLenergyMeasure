# Locked hybrid diagnose prompt (Phase 2)

**Locked at:** Phase 2 calibration round 1 (initial baseline lock).
**Source:** `research/mining-substrate-trial/scripts/strategies/prompts.py` constant
`HYBRID_DIAGNOSE_PROMPT_TEMPLATE`.

## Purpose

A SEPARATE prompt from extension. Diagnose specifically WHY (a)
missed items the reference catalogue has. Phase 3 may run this
prompt OFFLINE (outside the trial cell) to produce a "walker-gap
report" that informs the deterministic miner refactor (Bake-off A's
~1800 LoC target).

Not invoked by Phase 2 (d-ab) executor by default; the extension
prompt subsumes basic diagnosis via the `missed_diagnosis` section.
This standalone prompt is reserved for richer diagnosis runs.

## Inputs

Three YAML/markdown payloads:
1. The deterministic-miner output.
2. The reference catalogue (the ground truth).
3. The source the miner read.

## Output

Single YAML envelope with `diagnoses: - {id, why_missed,
fix_suggestion}`.

## Full prompt template

```
You are a code analyser diagnosing GAPS in a deterministic miner.

The deterministic miner produced output X for {engine} v{engine_version}.
A reference catalogue has output Y (the ground truth). The reference
contains items the deterministic miner did NOT surface.

INPUT 1 - DETERMINISTIC OUTPUT:
{deterministic_output}

INPUT 2 - REFERENCE OUTPUT (ground truth):
{reference_output}

INPUT 3 - SOURCE THE DETERMINISTIC MINER READ:
{source}

Produce a YAML document listing each reference item the deterministic
miner missed, with a ONE-LINE diagnosis of WHY:

diagnoses:
- id: <reference-invariant-id>
  why_missed: <root-cause: AST walker limitation? kwargs.pop pattern?
    cross-method invariant? docstring-only specification?>
  fix_suggestion: <terse: "extend walker to recognise X" / "walker
    correctly skipped; this is documentation-only">

RULES:
- Limit each diagnosis to ONE LINE for `why_missed` and ONE LINE for
  `fix_suggestion`.
- NO markdown code fences. NO commentary outside the YAML.

Emit the YAML now:
```

## When to use this

- Phase 4 synthesis: feeds the "what should the deterministic
  refactor focus on" decision.
- Maintenance: when a new engine version surfaces NEW deterministic
  misses, run the diagnose prompt to triage AST vs walker
  enhancements.
