---
title: Local knowledge production (operations guide)
description: How a maintainer produces and refreshes an engine's shipped validation rules locally, and how to debug the process.
---

# Local knowledge production (operations guide)

This page is the practical guide to producing an engine's shipped
validation rules (`rules.yaml`) on your own machine. Production is a local
maintainer task - it needs the engine source and, for the in-engine probes, the
engine's Docker image. CI never produces rules; it only verifies the committed
bytes (see [CI architecture](/explanation/architecture/ci-architecture)).

For the conceptual treatment - the two knowledge products and why they are
produced locally - see [Architecture overview](/explanation/architecture/architecture-overview).
For the schema-side counterpart, see [Schema refresh](/contributing/schema-refresh).

---

## Prerequisites

The cold-read stage reads the pinned engine source with a local LLM, so before
the first run:

- **A local Ollama daemon** reachable at `--ollama-host` (default
  `http://localhost:11434`, or the `OLLAMA_HOST` env var).
- **The analyst model pulled** into Ollama: `ollama pull qwen2.5-coder:32b`
  (the `--model` default; it needs a 32k context window).
- **A local copy of the engine's package source at the pinned version**, passed
  as `SRC` / `--source-root`. There is no downloader; unpack it yourself (e.g.
  `pip download --no-binary :all: --no-deps vllm==0.19.1` then extract the
  sdist).
- **An `analyst_clusters.yaml` manifest** for the engine (see below) - the
  three shipped engines already have one.

The in-engine probe tiers additionally need the engine's Docker image (and, for
the GPU engines, an NVIDIA GPU). The host-only stages (cold read, citation
check, coverage) do not.

## The absorb workflow

`make absorb` is the one command a maintainer runs per engine-version bump. It
drives the whole refresh loop for one engine's rules:

```bash
make absorb ENGINE=vllm SRC=engine-src/            # refresh the shipped rules
make absorb ENGINE=vllm SRC=engine-src/ ARGS='--dry-run'   # report the delta, write nothing
```

`SRC` is the engine's package source at the pinned version (the same tree the
`rules-coverage` CI check does a blobless sparse checkout of). The stages are:

1. **Cold read.** An assisted read of the pinned source proposes candidate
   rules (`make analyst-cold-read`, which prompts the local Ollama model per
   source cluster).
2. **Pool union.** The proposed candidates are merged with any manual seeds,
   then deduplicated into a version-scoped working file (the "pool"). Pools are
   never shipped.
3. **Recall interrogation.** For each shipped rule the fresh pool did not
   rediscover, absorb asks "is this constraint still present?" and annotates
   the answer.
4. **Verification ladder.** Every candidate (and every shipped rule's precise
   spec) is checked against the real engine. The engine is the arbiter.
5. **Promotion.** The shipped `rules.yaml` is regenerated from the confirmed
   candidates and the surviving shipped rules, byte-stably.
6. **Review delta.** An old-versus-new diff is written for the maintainer to
   review before committing.

Stages can be skipped to resume rather than redo a run:

- `--skip-cold-read` reuses the existing analyst pool file (no Ollama call).
- `--skip-interrogation` skips the recall-interrogation pass.
- `--skip-probe` runs the citation tier only (no engine container).
- `--clean-room` ignores the within-version verdict memory and re-probes
  everything.
- `--dry-run` reports the full delta and writes no shipped corpus.

The analyst endpoint and model are overridable with `--ollama-host` / `--model`
(see [Prerequisites](#prerequisites) for their defaults).

The only `src/` file absorb writes is the shipped `rules.yaml`, via promotion.

---

## The verification ladder

The ladder is why the rules are trustworthy: a candidate only ships if the
engine confirms it.

| Tier | Check | Where it runs |
|---|---|---|
| 1 | **Citation** - the candidate's cited `file:line` resolves in the pinned source | Host (`make check-citations`) |
| 2 | **Construction** - constructing the config with the offending values makes the engine raise (for `error` rules) | Engine Docker image (`make probe-candidates`) |
| 3 | **Effective-config identity** - the engine silently normalises the declared value (for `dormant` rules) | Engine Docker image (`make probe-candidates`) |

Survival is decided on the probe verdict's status, never the probe process exit
code (which is `0` even when every verdict is an infrastructure error). The
verdict vocabulary is closed - there is deliberately no `refuted`, because pure
config construction can never prove the engine accepts a config end-to-end:

| Verdict | Meaning | Disposition on promotion |
|---|---|---|
| `confirmed` | The probe proved the claim in the real engine | Ships |
| `unconfirmed` | The probe ran but neither proved nor disproved the claim | **Withheld** until signed off (fail-closed) |
| `unprobeable` | No deterministic probe could be derived or run | **Withheld** until signed off (fail-closed) |
| `infra_error` | A probe leg failed for infrastructure reasons | Kept untouched (not tested this run) |
| `unverified` | Not probed this run | Kept untouched (not tested this run) |

The residue semantics are **fail-closed**: an `unconfirmed` or `unprobeable`
rule is *withheld* from the promoted `rules.yaml` - it does not ship - until a
maintainer signs it off by hand. It is not retained-until-reviewed; an unmarked
residue rule silently disappears from the shipped corpus. Signing off means
adding the rule's id to the `residue:` list in
`engine_versions/<engine>/v<version>/outputs/absorb_signoff.yaml` with
`human_confirmed: true`; absorb reads that file and re-ships only the marked
rules (annotating them as human-confirmed). Rules that simply were not tested
this run (`unverified` / `infra_error`) are kept untouched, so an all-infra
failure promotes nothing new and drops nothing.

---

## Individual stages for debugging

Each stage has its own make target, so you can run and inspect them in
isolation:

| Target | Does | Produces |
|---|---|---|
| `make analyst-cold-read ENGINE=<e> SRC=<src>` | Cold-read the source into candidates | Candidate pool |
| `make check-citations CANDIDATES=<f> SRC=<src>` | Ladder tier 1: confirm each citation resolves | Per-candidate citation verdict |
| `make probe-candidates ENGINE=<e> CANDIDATES=<f>` | Ladder tiers 2-3: construction/identity probes in-engine | Per-candidate probe verdict |
| `make rules-coverage ENGINE=<e> SRC=<src>` | Advisory: report validator sites no shipped rule covers | Coverage report |

`make probe-candidates` and the in-engine ladder tiers need the engine's Docker
image; the host-only stages (cold read, citation check, coverage) do not.

---

## Where artifacts land

```
src/llenergymeasure/engines/<engine>/
└── rules.yaml                 Shipped validation rules (the only src/ output absorb writes)

engine_versions/<engine>/
├── current.yaml               Pin for the engine library version
├── analyst_clusters.yaml      Cold-read source clustering manifest (see below)
└── v<version>/
    ├── candidates/            Version-scoped working pool (gitignored)
    │   ├── analyst_cold_read.yaml   Cold-read proposals
    │   ├── manual_seeds.yaml        Hand-authored seeds
    │   ├── union.yaml               Merged, deduplicated pool
    │   └── ladder.yaml              Persisted probe verdicts (within-version memory)
    └── outputs/               Durable, git-tracked
        └── absorb_signoff.yaml      Human residue sign-off record (written by absorb; created on demand)
```

The pool and ladder verdicts under `candidates/` are working files and are
gitignored (`engine_versions/**/candidates/`). The `outputs/` sibling is
git-tracked: the sign-off file is a durable record a maintainer edits. Only
`rules.yaml` is a committed *shipped* artifact.

### The analyst_clusters.yaml manifest

Each engine has a per-engine manifest at
`engine_versions/<engine>/analyst_clusters.yaml`. It is a single `clusters:`
mapping of cluster name to a list of source files (relative to `SRC`); a path
with a `*` is a glob. The cold read packs each cluster's files into char-budget
chunks and prompts the model once per chunk, so the manifest is what scopes and
groups the read - a cluster name becomes the chunk id prefix. Example:

```yaml
# engine_versions/vllm/analyst_clusters.yaml
clusters:
  sampling:
    - sampling_params.py
  speculative:
    - config/speculative.py
  platforms:
    - platforms/*.py
```

Adding a source area to the cold read means adding files to a cluster (or a new
cluster) here; there are no per-cluster prompt hints, only the file list.

---

## The runtime-literal stage

Some engine type knowledge lives only in runtime validation code, not in the
static type surface. Transformers `early_stopping` is the canonical case: it
accepts `True`/`False`/`"never"` at runtime, but signature-based discovery
records `{"type": "bool"}`, so the generated typed config rejects the
upstream-valid `"never"`. The runtime-literal stage recovers these literals and
folds them into the discovered schema. It runs as a second stage of schema
discovery, inside the same engine container, over the just-discovered envelope:

```
static discovery -> candidate generation -> construction probe -> merged schema written once
```

**Candidate generation** unions three-plus-one sources, all string-valued only:

- the shipped rules corpus (equality / membership comparands and single-quoted
  `message_template` tokens);
- the engine's own validation source text (`self.<field> in {...}` / `== "..."`)
  and class docstrings;
- an optional LLM-proposed file (read-only; written by a separate analyst);
- plus the previous schema's recorded literals, carried forward so a still-valid
  literal survives even if its original evidence source moved.

A candidate is kept only when the field's static type cannot already express the
value.

**Construction probe** (the engine is the arbiter). Each candidate gets a
two-leg probe in the real engine: the literal value must BUILD the native
config, and a sentinel string must RAISE. A field that accepts both is not
string-validated at construction grain, so recording it would be unsound and it
is dropped. For transformers the probe builds `GenerationConfig(**kwargs)` then
calls `.validate()` WITHOUT `strict` - strict validation conflates a type-valid
literal with a beam-mode inertness complaint.

**Recording.** A verified literal is written under an in-schema `runtime_literals`
key on the field, with construction provenance and its evidence:

```json
"early_stopping": {
  "type": "bool",
  "default": null,
  "runtime_literals": [
    {"value": "never", "verified": "construction", "pin": "5.7.0",
     "evidence": ["rule:transformers_raises_early_stopping_not_in_set", "src:..."]}
  ]
}
```

Codegen unions this into the generated annotation:
`early_stopping: bool | Literal['never'] | None`.

**Staleness is auto-narrow with loud surfacing.** At a bump every recorded
literal is re-probed. One that no longer verifies is dropped from the schema
(no human gate) and a `NARROWED` line is emitted for the maintainer's diff, so
the removal is visible in review rather than silent.

**Standing check.** `make check-corpus-literals` reports any corpus rule literal
the discovered schema type cannot express (directly or via a `runtime_literals`
entry). It is the consistency tripwire between the two knowledge products; a
finding means a shipped rule references a value the generated typed config would
reject.

---

## Debugging patterns

### A candidate fails the citation check

The cited `file:line` does not resolve in the source tree you passed as `SRC`.
Confirm `SRC` points at the engine package at the pinned version (not a
different checkout), then re-run `make check-citations`.

### A candidate passes citation but fails the probe

The construction/identity probe ran the engine and the engine did not behave as
the candidate claimed - the constraint may be stale (the library relaxed it) or
the candidate's values do not actually trigger it. Inspect the probe verdict for
that candidate id; adjust or drop the candidate.

### The probe reports infrastructure errors for every candidate

The engine image is missing or the container could not start. Because survival
keys on verdict status (not exit code), an all-`infra_error` run promotes
nothing new and keeps the existing corpus. Fix the image, then re-run.

### A rule I expected to ship disappeared from `rules.yaml`

It was probed as `unconfirmed` or `unprobeable` and is not signed off, so it was
withheld (fail-closed). Check the review delta's residue list. If the constraint
is genuinely still real but not machine-probeable, add its id to the `residue:`
list in `engine_versions/<engine>/v<version>/outputs/absorb_signoff.yaml` with
`human_confirmed: true`, then re-run absorb - it will re-ship the marked rule.
If the constraint is genuinely gone, leave it unmarked and it stays dropped.

### `rules-coverage` flags a validator site with no rule

The rule set is recall-first, not exhaustive. Add the missing constraint by
running `make absorb` again after seeding the candidate, or hand-author a seed
for the pool. Coverage is advisory and never blocks a merge.

---

## See also

- [Architecture overview](/explanation/architecture/architecture-overview) - the two knowledge products
- [Parameter discovery](/explanation/architecture/parameter-discovery) - how the shipped rules are consumed at runtime
- [Schema refresh](/contributing/schema-refresh) - the schema-side operations guide
- [CI architecture](/explanation/architecture/ci-architecture) - what CI verifies
