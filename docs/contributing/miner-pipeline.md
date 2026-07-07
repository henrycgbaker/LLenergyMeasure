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
   rules (`make analyst-cold-read`, which uses a local model).
2. **Pool union.** The proposed candidates are merged with observed runtime
   collisions and any manual seeds, then deduplicated into a version-scoped
   working file (the "pool"). Pools are never shipped.
3. **Recall interrogation.** For each shipped rule the fresh pool did not
   rediscover, absorb asks "is this constraint still present?" and annotates
   the answer.
4. **Verification ladder.** Every candidate (and every shipped rule's precise
   spec) is checked against the real engine. The engine is the arbiter.
5. **Promotion.** The shipped `rules.yaml` is regenerated from the confirmed
   candidates and the surviving shipped rules, byte-stably.
6. **Review delta.** An old-versus-new diff is written for the maintainer to
   review before committing.

Every stage is skippable, so a re-run resumes rather than redoes. The only
`src/` file absorb writes is the shipped `rules.yaml`, via promotion.

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
code (which is `0` even when every verdict is an infrastructure error). A rule
that the probe could not exercise this run is residue; it keeps shipping until a
maintainer signs it off by hand. A rule never leaves the corpus automatically -
a recall interrogation that comes back "absent" is a retirement proposal, not a
drop.

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
├── rules.yaml                 Shipped validation rules (the only committed output)
└── _staging/                  Working files (gitignored)

engine_versions/<engine>/
└── current.yaml               Pin for the engine library version (Renovate-writable)
```

The pool, ladder verdicts, sign-off file, and review report are working files
under the staging area; only `rules.yaml` is committed.

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
