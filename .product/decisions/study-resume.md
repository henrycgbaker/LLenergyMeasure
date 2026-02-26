# Study Resume

**Status:** Proposed
**Date decided:** 2026-02-19
**Last updated:** 2026-02-25
**Research:** N/A

> **Note:** Status is `Proposed` rather than `Accepted` — this file was marked `DRAFT — TODO: needs to be reviewed` in its original form. The decisions below are Claude-recommended and pending user confirmation.

## Context

Long-running studies (multi-experiment sweeps, multi-backend Docker runs) can be interrupted
mid-run by hardware failures, power loss, or user cancellation. Without resume support,
interrupted studies must be restarted from scratch — wasting completed experiment time and
(for Docker/TRT) expensive compilation time.

The criticality of resume depends on study duration:
- **v2.0 local single-backend**: studies are short (minutes). Re-run cost is low. Resume
  adds implementation complexity that isn't warranted yet.
- **v2.2 Docker multi-backend**: TRT engine compilation alone is 10–30 minutes per model.
  A 5-backend × 8-config sweep could take hours. Interruption without resume is unacceptable.

The decision space splits into: (a) what to write during a study run regardless of resume
support, and (b) when to implement the `resume` command.

## Considered Options

### Sub-decision 1: Write a manifest file during every study

| Option | Pros | Cons |
|--------|------|------|
| **Always write `study_manifest.json` alongside results** | Foundation for resume at no cost to current users. Useful for post-hoc inspection even without resume. Written incrementally — survives interruption. | Small I/O overhead per experiment completion (negligible). |
| Only write manifest when resume is implemented | Less code now. | Cannot bootstrap resume from existing runs — users who already ran studies lose their history. |
| Write only a completion marker | Simpler. | Insufficient for resume — need per-experiment status, not just overall done/not-done. |

### Sub-decision 2: When to implement resume (`llem run --resume`)

| Option | Pros | Cons |
|--------|------|------|
| **Defer `llem run --resume` to v2.2** | Matches where resume becomes critical (Docker multi-backend). Avoids premature complexity in v2.0. Manifest data is already written — defer is low risk. | v2.0 users cannot resume interrupted studies. |
| Implement in v2.0 | Complete feature from the start. | Overengineering for short single-backend studies. Resume requires careful config-hash matching and edge case handling — adds scope to an already broad v2.0. |

### Sub-decision 3: Experiment identity for resume matching

| Option | Pros | Cons |
|--------|------|------|
| **Match by `config_hash` (resolved ExperimentConfig hash)** | Stable identity — independent of file paths or ordering. Hash mismatch detects when study.yaml changed between run and resume. | If study.yaml changes intentionally, user must confirm resume (hash mismatch warning). |
| Match by experiment index (position in sweep list) | Simple. | Brittle — inserting a new experiment shifts all subsequent indices. |
| Match by `config_summary` string | Human-readable. | Not unique — two experiments with same summary string but different params would collide. |

## Decision

We will write `study_manifest.json` to the study results directory at startup and update it
after each experiment completes, in every study run regardless of whether resume is
implemented. The `llem run --resume` command will be deferred to v2.2.

Experiment identity for resume matching will use `config_hash` (hash of the resolved
`ExperimentConfig`). If `study.yaml` has changed since the interrupted run, the tool warns
and asks for confirmation before resuming.

Rationale: writing the manifest costs nothing and provides useful post-hoc data even without
resume. Deferring the resume command to v2.2 matches the version where it becomes critical
(Docker/TRT compilation cost). Config hash is the most robust identity — it catches accidental
study.yaml changes and is independent of experiment ordering.

## Consequences

Positive:
- `study_manifest.json` data exists from v2.0 onward — no cold-start migration when v2.2
  implements resume.
- Users can inspect which experiments completed post-hoc even in v2.0.
- Resume matching via config hash is robust to study.yaml reordering.

Negative / Trade-offs:
- v2.0 users cannot resume interrupted studies — must restart from scratch.
- Hash mismatch detection adds a confirmation prompt that may surprise users who intentionally
  edited their study.yaml before resuming.

Neutral / Follow-up decisions triggered:
- v2.2 must implement `llem run --resume` — this decision creates that obligation.
- `study_manifest.json` schema must be treated as stable from v2.0 onward (or versioned
  like result files).

## `study_manifest.json` Schema

Written to the study results directory at startup, updated after each experiment:

```json
{
  "study_name": "batch-size-sweep-2026-02",
  "study_yaml_hash": "abc123def456",
  "started_at": "2026-02-19T14:30:00Z",
  "llenergymeasure_version": "2.0.0",
  "experiments": [
    {
      "config_hash": "def456",
      "config_summary": "pytorch / batch=1 / bf16",
      "cycle": 1,
      "status": "completed",
      "result_file": "llama-3.1-8b_pytorch_batch-1_2026-02-19T14-30.json",
      "completed_at": "2026-02-19T14:32:14Z"
    },
    {
      "config_hash": "ghi789",
      "config_summary": "pytorch / batch=4 / bf16",
      "cycle": 1,
      "status": "pending",
      "result_file": null,
      "completed_at": null
    }
  ]
}
```

## Resume Logic (v2.2)

```
llem run --resume study.yaml
  → load study_manifest.json from results directory (study YAML detected by llem run)
  → verify study_yaml_hash matches current study.yaml (warn if changed)
  → skip experiments where status = "completed"
  → run experiments where status = "pending" or "failed"
  → update manifest in place as experiments complete
```

If `study.yaml` has changed since the interrupted run, the tool warns and asks for
confirmation before resuming — hash mismatch means the sweep definition may differ.

## v2.0 Behaviour

Manifest is written (so the data exists) but the `--resume` flag for `llem run` is not
implemented in v2.0. Interrupted studies must be restarted from scratch. The manifest is
still useful for post-hoc inspection of which experiments ran.

## Related

- [result-schema-migration.md](result-schema-migration.md) — schema versioning strategy for result files
- [../designs/study-yaml.md](../designs/study-yaml.md) — study YAML format and sweep grammar
- [versioning-roadmap.md](versioning-roadmap.md) — v2.0 vs v2.2 scope boundary
