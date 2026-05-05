# Pipeline Architecture

This doc is the chain-diagram reference for the engine-coupling pipeline. SSOT-driven Renovate cycles flow through these stages: trigger, two per-concern CI workflows, vendored artefacts, and the human curation checkpoint before merge.

## Asymmetric engine architecture (locked design choice)

The three engines run different pipelines in CI for a load-bearing reason. **Don't undo this asymmetry without re-reading [`#518`](https://github.com/henrycgbaker/llenergymeasure/issues/518) — the conclusion has held across re-litigations 2026-04-30, 2026-05-01, and 2026-05-05.**

| Engine | Image source | CI flow on PR |
|---|---|---|
| **transformers** | First-party `docker/Dockerfile.transformers` (FA3-included; no upstream provides this) | `build-engine-image` (rebuild) → `update-engine-{invariants,schemas}` (probe + mine/introspect) → [merge] → `publish-engine-image` (mirror to production tag) |
| **vllm** | Upstream `vllm/vllm-openai:v<VER>` directly + bind-mount llem source | `update-engine-{invariants,schemas}` cells fire directly on `pull_request: paths` (no first-party build) |
| **tensorrt** | Upstream `nvcr.io/nvidia/tensorrt-llm/release:<VER>` directly + bind-mount llem source | Same shape as vllm |

**Why asymmetric.** vllm + tensorrt's upstream images empirically contain everything llem needs at runtime (PoC verified 2026-04-30: `pydantic`, `typer`, `pyarrow`, `rich`, `dotenv`, `pyyaml` all present transitively). Transformers' upstream images don't include FA3, which is non-negotiable for production-equivalent CI runs. So transformers gets a first-party Dockerfile; the others stay upstream-direct.

**Drift safety.** The only argument for first-party-everywhere is "what if upstream drops a transitive dep llem needs?" The migration cost from upstream-direct → first-party is bounded (~1 day, well-defined recipe per `#518`). The actual cost of running first-party-everywhere is the FA3 build for two extra engines that don't need it.

### Transformers PR-time CI flow (rebuild + probe/mine/introspect chain)

```text
PR opens (touches transformers paths: SSOT, Dockerfile, miner code, etc.)
  │
  │  build-engine-image.yml fires (paths trigger:
  │  engine_versions/transformers.yaml, docker/Dockerfile.transformers,
  │  .github/workflows/build-engine-image.yml)
  ▼
[Build transformers runtime image; cache hits ~10-15 min, cold FA3 ~60-90 min]
[Push to ghcr.io/<repo>/transformers-cache:transformers-<VER>]
  │
  │  workflow_run chain fires (Build engine image (transformers) success)
  ▼
update-engine-{invariants,schemas}.yml transformers cells run:
  pull transformers-cache image → probe → mine/introspect → vendor → writeback
  │
  │  Probe-fail → CI red. The 'accept-probe-fail' PR label bypasses
  │  the gate for known-drift cases (admin escalation; see #547).
  ▼
[CI green/red. PR ready to merge when green.]
  │
  │  PR merges to main (push event)
  ▼
build-engine-image.yml fires again (push trigger; cache hits)
  │
  │  publish-engine-image.yml chains via workflow_run
  │  (publish only fires on push event, NOT on pull_request)
  ▼
[Mirrors transformers-cache:transformers-<VER> to production tags
 transformers:<VER> + transformers:latest for end-user `pip install`]
```

### vllm + tensorrt PR-time CI flow (no rebuild; upstream-direct)

The diagram below applies to vllm + tensorrt only — `update-engine-*.yml` cells fire directly on `pull_request: paths` (no `build-engine-image` chain). They pull the upstream image at the SSOT-pinned version, bind-mount llem source, and probe/mine/introspect inside the upstream container.

```
================================================================================
LLenergyMeasure Engine-Coupling Pipeline (vllm + tensorrt)
Per-concern workflows (engine-invariants + engine-schemas) with sibling
coordination via wait-on-check-action.
================================================================================

LEGEND:  [auto]    fully automated, no human action
         [chk]     HUMAN CHECKPOINT — required dev input
         [info]    informational artefact, advisory
         { }       input
         [→...]    automated transition

   {Renovate scans upstream library releases on configured schedule}
                                │  [auto]
                                ▼
   Custom regex manager bumps:
     engine_versions/{engine}.yaml:library.current_version  (SSOT — canonical)
     docker/Dockerfile.{engine} ARG (derived; auto-templated from SSOT)
                                │  [auto]
                                ▼
   {Renovate opens PR: "fix(deps): bump vllm to 0.10.2"}
                                │  [auto] path-filtered triggers fan out
                                ▼              in PARALLEL to two workflows
   ┌────────────────────────────┴─────────────────────────────┐
   ▼                                                           ▼
┌──────────────────────────────┐         ┌──────────────────────────────┐
│  update-engine-invariants.yml│         │  update-engine-schemas.yml   │
│  (per-engine matrix)         │         │  (engines matrix)            │
│  Layers over: invariant-     │         │  Layers over: parameter-     │
│   miner + invalidity-miner + │         │   discovery + typed-schema-  │
│   lift modules + vendor-CI   │         │   discovery                  │
│                              │         │                              │
│  STEP 1 [auto]: PROBE (inline│         │  STEP 1 [auto]: PROBE (inline│
│   `python -m scripts._probe  │         │   `python -m scripts._probe  │
│    --producer invariants`)   │         │    --producer schemas`)      │
│   verdict: pass | fail       │         │   verdict: pass | fail       │
│                              │         │                              │
│  ── if probe == pass ──      │         │  ── if probe == pass ──      │
│  STEP 2 [auto]: MINE         │         │  STEP 2 [auto]: DISCOVER     │
│   build_corpus.py            │         │   engine_introspectors       │
│   → configs/engine_invariants│         │   → src/llenergymeasure/     │
│     /{engine}.proposed.yaml  │         │     config/discovered_       │
│                              │         │     schemas/{engine}.json    │
│  STEP 3 [auto]: VENDOR-REPLAY│         │                              │
│   vendor_rules.py + the      │         │  STEP 3 [auto]: DIFF vs HEAD │
│   compare_expected_vs_       │         │                              │
│   observed contract from     │         │  STEP 4 [auto]: REGENERATE   │
│   _invariant_vendor_common.py│         │   docs/generated/            │
│   replays kwargs_positive +  │         │     curation-{engine}.md     │
│   kwargs_negative against    │         │   (Parameters section —      │
│   live library; classifies   │         │    fact base for human       │
│   outcomes (positive_        │         │    curator; pre-existing     │
│   confirmed, negative_       │         │    behaviour preserved)      │
│   confirmed, divergence)     │         │                              │
│   → configs/engine_invariants│         │  STEP 5 [auto]: COMMENT      │
│     /{engine}.vendored.yaml  │         │   + LABEL (suppress on empty)│
│                              │         │                              │
│  STEP 4 [auto]: DIFF vs HEAD │         │  ── if probe == fail ──      │
│   for both proposed.yaml +   │         │  Post probe-fail comment     │
│   vendored.yaml artefacts    │         │  with 3 routes (per §3 of    │
│                              │         │   the design doc: patch      │
│  STEP 5 [auto]: REGENERATE   │         │   code / /approve-reuse /    │
│   docs/generated/            │         │   escalate). Apply           │
│     invariants-{engine}.md   │         │   probe-blocked label.       │
│   (Invariants section — fact │         │   exit 0 (not CI failure)    │
│   base; encompasses dormancy │         │                              │
│   + invalidity + walker      │         │                              │
│   output + introspection +   │         │                              │
│   runtime catch-all)         │         │                              │
│                              │         │                              │
│  STEP 6 [auto]: COMMENT      │         │                              │
│   + LABEL (suppress on empty)│         │                              │
│                              │         │                              │
│  ── if probe == fail ──      │         │                              │
│  Same 3-route handling as    │         │                              │
│  schemas-pipeline above.     │         │                              │
│  Apply probe-blocked label.  │         │                              │
│  exit 0.                     │         │                              │
└─────────────┬────────────────┘         └────────────┬─────────────────┘
              │                                       │
              │ Each workflow:                                                  │
              │  - uploads engine-step-diff-{engine}-{concern}.yaml             │
              │  - posts its OWN per-pipeline comment (suppress on empty)       │
              │  - applies its own per-pipeline label                           │
              │    (invariants/schemas-changed, invariants/schemas-breaking,    │
              │     corpus-changed, probe-blocked)                              │
              │  - WAITS for sibling pipeline to complete                       │
              │    (lewagon/wait-on-check-action; already-finished sibling      │
              │     exits immediately)                                          │
              │  - LAST-FINISHING workflow performs ATOMIC WRITEBACK in-line:   │
              │     git add configs/engine_invariants/{engine}.proposed.yaml    │
              │             configs/engine_invariants/{engine}.vendored.yaml    │
              │             src/llenergymeasure/config/discovered_schemas/      │
              │                  {engine}.json                                  │
              │             docs/generated/curation-{engine}.md                 │
              │             docs/generated/invariants-{engine}.md               │
              │             engine_versions/{engine}.compat.json                │
              │             engine_versions/{engine}.yaml (if /approve-reuse    │
              │                                            fired during cycle)  │
              │     git commit && git push --force-with-lease                   │
              │  - LAST-FINISHING workflow applies cross-pipeline rollup label  │
              │    (safe-bump | probe-blocked)                                  │
              │                                                                 │
              │ NO summariser workflow file. NO composite action.               │
              │ Cross-pipeline state lives on labels (GitHub-native primitive). │
              │ "Did the cycle run?" = check-status badge. "Anything change?"   │
              │ = per-pipeline comments + commits. "What's the rollup state?"   │
              │ = label.                                                        │
              ▼
            ┌────────────────────────────────────────────────┐
            │ PR after a Renovate cycle:                      │
            │  - 2 per-concern check statuses                 │
            │  - up to 2 comments per cycle (suppress-on-empty):│
            │     1. engine-invariants pipeline                │
            │     2. engine-schemas pipeline                   │
            │  - 1 atomic bot commit (all artefacts; written   │
            │    by whichever workflow finished last)          │
            │  - cross-pipeline rollup label                   │
            │    (safe-bump | probe-blocked)                   │
            └─────────────────────┬──────────────────────────┘
                                  │
                                  ▼
   ╔═══════════════════════════════════════════════════════════════════╗
   ║              HUMAN CURATION CHECKPOINT [chk]                       ║
   ║   The only crossing of the human-as-final-checkpoint boundary (P6)║
   ║   inside the otherwise-automated vendored half. Bots NEVER edit   ║
   ║   src/llenergymeasure/config/engine_configs.py.                    ║
   ║                                                                    ║
   ║   Dev consumes auto-generated digests:                             ║
   ║     docs/generated/curation-{engine}.md                            ║
   ║       Section 1: Parameters (discovered fields × Pydantic-curated  ║
   ║                  yes/no, deltas vs previous SSOT version)          ║
   ║     docs/generated/invariants-{engine}.md                          ║
   ║       Section 1: Invariants (corpus rules added/changed/removed,   ║
   ║                  classified by added_by; encompasses dormancy +    ║
   ║                  invalidity + walker output + introspection +      ║
   ║                  runtime catch-all)                                ║
   ║                                                                    ║
   ║   Dev manually edits engine_configs.py:                            ║
   ║     - which discovered params to expose in Pydantic                ║
   ║     - which Literal narrowings to pin                              ║
   ║     - which sub-config taxonomy to use                             ║
   ║     - which custom @model_validator decorators to add              ║
   ║                                                                    ║
   ║   Push -> triggers re-run of CI cycle -> updated summary comment   ║
   ║   supersedes prior (edited via comment-id key, no proliferation)   ║
   ║                                                                    ║
   ║   Decision routes after digest review:                             ║
   ║     safe-bump + green CI         -> squash-merge                   ║
   ║     corpus-changed + mechanical  -> squash-merge                   ║
   ║     invariants-breaking          -> edit engine_configs.py         ║
   ║     schemas-breaking             -> edit engine_configs.py         ║
   ║     probe-blocked                -> resolve via §3 routes:         ║
   ║                                     - patch producer code, OR      ║
   ║                                     - /approve-reuse (slash cmd)   ║
   ║                                     - escalate label               ║
   ║                                                                    ║
   ║   GUIDED CURATION UX (RFC-style YAML decision file + libcst        ║
   ║   applier) is DEFERRED to issue #475. Current redesign ships       ║
   ║   self-serve curation only — devs hand-edit engine_configs.py      ║
   ║   based on digest. After 2-3 Renovate cycles of operational data,  ║
   ║   #475 reactivation will evaluate whether guided UX pays off.      ║
   ╚═══════════════════════════════════════════════════════════════════╝
                                  │
                                  ▼
                          ┌──────────────┐
                          │ squash-merge │
                          └──────┬───────┘
                                 ▼
                   PR closes; engine version + all
                   vendored artefacts + curated Pydantic
                   pinned together at this commit.

================================================================================
PROBE-FAIL HUMAN CHECKPOINT [chk]
The OTHER human touchpoint (per P6). Inside the otherwise-automated CI half.
================================================================================

When a probe fails (inline step 1 of either workflow), three resolution routes:

  ┌─ ROUTE 1 [chk → auto]: Patch producer code
  │   Dev edits scripts/engine_miners/{engine}_*_miner.py or
  │   scripts/engine_introspectors/{engine}_introspector.py to fix the
  │   broken landmark (e.g. follow an upstream rename). Push commit ->
  │   workflow re-runs -> probe re-runs -> if pass, downstream stages
  │   proceed.
  │
  ├─ ROUTE 2 [chk → auto]: Approve reuse via slash command
  │   Dev posts `@llem-ci-bot /approve-reuse <engine> <producer>` as
  │   PR comment. Producer ∈ {invariants, schemas} (per-producer
  │   granularity — vllm invariants might be reusable while vllm
  │   schemas are not).
  │                            │  [auto]
  │                            ▼
  │   approve-reuse-bot.yml (issue_comment: created listener)
  │   - Validates dev approval rights
  │   - Updates engine_versions/{engine}.yaml miner_pins.{producer}
  │     to widen SpecifierSet to include the bumped version
  │   - Commits SSOT change via llem-ci-bot App token (cascades;
  │     GITHUB_TOKEN would not)
  │                            │  [auto]
  │                            ▼
  │   Probe re-runs against widened range -> verdict flips to PASS
  │   -> downstream stages proceed
  │
  └─ ROUTE 3 [chk]: Escalate / block
      Dev applies probe-blocked label. Renovate stops retrying this
      bump until the label is removed; route 1 or 2 must follow before
      merge.

NO OTHER SLASH COMMANDS. /rerun, /skip-probe, /force-merge explicitly
rejected as footguns. Deliberate scope: one binary approval gate per
(engine, producer), no escape hatches.

================================================================================
ADJACENT PIPELINES (independent of per-PR Renovate cycle)
================================================================================

   engine-versions-sweep.yml          {scheduled, e.g. weekly}     [auto info]
   └─ runs scripts/_probe.py over a curated version range
      (e.g. vllm v0.9..v0.12); updates engine_versions/{engine}.compat.json
      (probe cache + compat-matrix in one file; closes #470).
      Populates probe-result cache so per-PR probes hit warm cache.

   Runtime side-product               {study runtime, NOT CI; study-local}
   ├─ runtime_observations.jsonl      [info]
   │   - Producer: src/llenergymeasure/study/runtime_observations.py
   │     (warnings.catch_warnings + logger handler wrapping each worker
   │     body); wired in runner.py
   │   - Schema: schema_version=1; one record per (study_run_id,
   │     config_hash, cycle); outcome ∈ {success, exception,
   │     subprocess_died}
   │   - Consumer (today): llem report-gaps (--source runtime-warnings,
   │     the only wired source). Output: YAML fragment for manual
   │     append to corpus (`# TODO: human` markers on placeholder
   │     fields). PRESERVED as escape-hatch.
   │   - Consumer (long-term): subsume into curation digest Section 3
   │     ("Runtime gaps observed"). DEFERRED #475; reactivate after
   │     2-3 Renovate cycles of operational data.
   │
   └─ equivalence_groups.json         [info]
      - Detects observed_config_hash collisions across configs:
        configs Pydantic distinguishes (resolved_config_hash differs)
        but engine collapses (observed_config_hash matches). Flagged
        as gap_detected: true -> dormancy signal.
      - proposed_rule_id field is currently always None; consumer
        deferred until a researcher hits a real gap_detected: true
        group and asks for tooling. Tracked in #405 + #474.
================================================================================
```

For the full design rationale (including the resolution of the per-engine vs per-concern split, the wait-for-sibling coordination decision, and the rejected summariser-workflow alternative), see [`.product/designs/engine-coupling-architecture-2026-04-28.md`](../.product/designs/engine-coupling-architecture-2026-04-28.md).
