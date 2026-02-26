# Local Result Navigation

**Status:** Proposed
**Date decided:** 2026-02-19
**Last updated:** 2026-02-25
**Research:** N/A

## Context

After a run, users need to find and inspect their results. The question is how much CLI surface to provide for this. The target user (ML researcher) is comfortable with the terminal,
and the result format (JSON) is already widely understood. Building a dedicated results browser risks adding complexity that duplicates existing tools (`ls`, `jq`, `diff`).

Peer tools — lm-eval, MLflow, Optimum-Benchmark — do not ship a `results list` or `results show` subcommand. They rely on human-readable output directories and external
tooling (notebooks, jq, custom scripts).

A `llem results push` subcommand is planned for v2.4 (central DB upload). If that ships, a companion `llem results list` (to browse local results before pushing) becomes natural. The two should be designed together, not independently.

## Considered Options

### Result Discovery Mechanism

| Option | Pros | Cons |
|--------|------|------|
| **Human-readable filenames + `ls results/` — bold** | Zero new CLI surface. Standard shell tooling. Sufficient for target user. Already decided in `cli-ux.md`. No peer tool ships a richer alternative. | Requires user to know to look in `results/`. No sorting or filtering built in. |
| `llem results list` subcommand | Integrated UX; filterable | No peer tool has this. Adds maintenance burden. Duplicates `ls` + `grep`. |
| `llem results show <file>` subcommand | Formatted display | `cat` + `jq` are sufficient. YAGNI. |

### End-of-Run Feedback

| Option | Pros | Cons |
|--------|------|------|
| **Concise summary always printed to stdout — bold** | Stays in terminal scrollback. No need to reopen file. Human-readable at a glance. Shows filename of output so user knows where to look. | Duplicates data already in the JSON file. |
| No stdout summary | Cleaner output | User must open the file to see what happened; poor UX for interactive use. |
| Verbose stdout output | More detail | Noisy; the JSON file is already the full record. |

## Decision

We will use human-readable filenames as the primary result discovery mechanism, always print
a concise end-of-run summary to stdout, and ship no `llem results` subcommand at v2.0.

Rationale: No peer tool ships a results browser subcommand. Human-readable filenames + `ls`
is sufficient for the target user. The stdout summary bridges the UX gap without adding CLI
surface. The `llem results push` use case (v2.4) is the natural trigger to revisit this; a
local browser and a push command should be co-designed.

**Rejected (2026-02-19):** `llem results list` and `llem results show` subcommands — no peer
tool has this pattern; `ls` + `jq` is sufficient for the target research user. Revisit only
alongside `llem results push` at v2.4.

## Consequences

Positive:
- Zero additional CLI surface at v2.0
- Standard shell tooling (`ls`, `jq`, `diff`) works out of the box
- Stdout summary keeps users informed without requiring file inspection
- Consistent with no-magic, maximally-explicit UX principle from `cli-ux.md`

Negative / Trade-offs:
- No built-in filtering, sorting, or comparison of results
- Users unfamiliar with `jq` may struggle to query result fields
- `llem results push` at v2.4 may require a companion browser subcommand; those must be
  co-designed rather than retrofitted

Neutral / Follow-up decisions triggered:
- `llem results push` + optional `llem results list` deferred to v2.4
- jq patterns for power users to be documented in README (not new CLI surface)

## Filename Format

Confirmed in [`cli-ux.md`](cli-ux.md). Format: `{model-slug}_{backend}_{iso-timestamp}.json`

```
results/llama-3.1-8b_pytorch_2026-02-19T14-30.json
results/llama-3.1-8b_vllm_2026-02-19T15-45.json
```

For studies: `{study_name}_{timestamp}/` subdirectory containing `study_manifest.json` and
one JSON file per experiment.

```
results/batch-size-sweep-2026-02/
  study_manifest.json
  llama-3.1-8b_pytorch_batch-1_2026-02-19T14-30.json
  llama-3.1-8b_pytorch_batch-8_2026-02-19T14-35.json
  ...
```

## End-of-Run Summary Format

Always printed to stdout at the end of every run. Implementation detail belongs in
[`../designs/cli-commands.md`](../designs/cli-commands.md).

Single experiment example:
```
Result: results/llama-3.1-8b_pytorch_2026-02-19T14-30.json

  Energy      312.4 J    (3.12 J/request)
  Throughput  847 tok/s
  Latency     TTFT 142ms  ITL 28ms
  Duration    4m 32s
```

Study example:
```
Study complete: 12/12 ran, 0 failed, 0 skipped
Results: results/batch-size-sweep-2026-02/

  Best energy/request:  batch=32 / bf16  →  1.84 J/req
  Best throughput:      batch=32 / bf16  →  2,341 tok/s
  Worst energy/request: batch=1  / fp32  →  8.12 J/req
```

## Power User Patterns (README, not CLI)

jq patterns to be documented in the README. No new CLI surface.

```bash
# List all results with their energy
cat results/*.json | jq -r '[.model, .backend, .energy_joules] | @tsv'

# Compare two configs
diff <(jq . results/llama-3.1-8b_pytorch_2026-02-19T14-30.json) \
     <(jq . results/llama-3.1-8b_vllm_2026-02-19T15-45.json)
```

## Related

- [`cli-ux.md`](cli-ux.md) — human-readable filename format; maximally-explicit UX principle
- [`output-storage.md`](output-storage.md) — directory structure and collision policy
- [`../designs/cli-commands.md`](../designs/cli-commands.md) — stdout summary implementation
- [`../designs/result-schema.md`](../designs/result-schema.md) — JSON result fields
- [`open-questions.md`](open-questions.md) — `llem results push` (v2.4 blocker)
