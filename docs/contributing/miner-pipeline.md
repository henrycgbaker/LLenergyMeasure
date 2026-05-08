---
title: Miner pipeline (debugging guide)
description: Practical guide to debugging the invariant-mining pipeline. For the conceptual treatment of how the pipeline works, see the Architecture page.
---

# Miner pipeline (debugging guide)

This page is a practical debugging reference for the invariant-mining
pipeline. For the conceptual treatment of how the pipeline works (and
how it parallels the schema-discovery pipeline), see
[engine introspection pipelines](/explanation/architecture/engine-introspection-pipelines).

For the format spec of the corpus YAMLs the pipeline produces, see
[invariants corpus format](/reference/invariants-corpus-format).

For step-by-step instructions on adding a new miner for a new engine,
see [extending miners](/contributing/extending-miners).

---

## Where artefacts land on disk

```
src/llenergymeasure/engines/{engine}/
├── invariants.proposed.yaml          Maintainer-seeded corpus, post-mining
└── invariants.validated.yaml         CI-validated overlay, post-validate-replay

src/llenergymeasure/engines/{engine}/_staging/   (gitignored, miner-only)
├── {engine}_static_miner.yaml        Per-miner staging output (not committed)
├── {engine}_dynamic_miner.yaml
└── _failed_validation_{engine}.yaml  Quarantined rules

scripts/engine_miners/
├── _base.py                          Shared AST primitives, detectors, filters
├── _ssot.py                          load_miner_pin() - resolves SpecifierSet from engine SSOT
├── _pydantic_lift.py                 Lift module for Pydantic models
├── _msgspec_lift.py                  Lift module for msgspec.Struct
├── _dataclass_lift.py                Lift module for stdlib @dataclass + Literal
├── {engine}_static_miner.py          Per-engine static miner
├── {engine}_dynamic_miner.py         Per-engine dynamic miner (when applicable)
├── build_corpus.py                   Orchestration: merge + dedup + validate
└── validate_invariants.py            Replays each rule against the live library

engine_versions/{engine}.yaml         SSOT for library version + miner_pins envelopes
```

The two committed YAML files form a lifecycle pair: the miners write
the proposed YAML, then `validate_invariants.py` replays each rule
inside the engine's Docker image and writes the validated YAML. The
runtime loader overlays validated observations onto the proposed
corpus, so consumers see CI-confirmed behaviour where available and the
declared shape elsewhere.

---

## How to read a probe-fail bot comment

When a producer's landmark check fails, the cell skips the rest of the
work and the bot posts a `probe-blocked` comment on the PR. The comment
identifies which engine and which producer (`invariants` or `schemas`)
the probe failed for, plus the symptom.

Three resolution routes:

1. **Patch producer code** - the dev edits the affected miner or
   introspector module to fix the broken landmark (for example, follow
   an upstream class rename). Pushing the commit re-runs the workflow;
   the probe re-runs; if it passes, downstream stages proceed.
2. **Approve reuse** - the dev posts `@llem-ci-bot /approve-reuse <engine> <producer>`
   as a PR comment. The slash command widens the `miner_pins.{producer}`
   `SpecifierSet` in the engine SSOT to include the bumped version.
   The probe re-runs against the widened range.
3. **Escalate** - the dev applies the `probe-blocked` label. Renovate
   stops retrying this bump until the label is removed.

Per-producer granularity matters: `vllm/invariants` might be reusable
while `vllm/schemas` is not, or vice versa.

---

## File locations to grep when investigating

| Symptom | Files to inspect first |
|---|---|
| Miner produces no rules for a new engine | `scripts/engine_miners/{engine}_*_miner.py` (does the file exist? imports succeed?); `engine_versions/{engine}.yaml` (is `miner_pins` populated?) |
| `MinerVersionMismatchError` raised at import time | `engine_versions/{engine}.yaml miner_pins.{static\|dynamic\|discovery}` vs the live library version (`importlib.metadata.version("{library}")`) |
| `MinerLandmarkMissingError` raised at import time | `scripts/engine_miners/{engine}_*_miner.py` (which `find_class` / `find_method` call returned None? compare against the live library source tree) |
| Validation gate fails on a previously-passing rule | `src/llenergymeasure/engines/{engine}/invariants.proposed.yaml` (locate the rule by id) and `_staging/_failed_validation_{engine}.yaml` (which check failed: `positive_raises`, `message_template_match`, or `negative_does_not_raise`) |
| Rule duplication or merge surprises | `scripts/engine_miners/build_corpus.py` (the merger; deduplication key is `(engine, severity, match_fields)`); look at `cross_validated_by` on the merged rule |
| Static miner missed a predicate | `scripts/engine_miners/_base.py` (which detector should have matched? did a filter drop the candidate?) |
| Dynamic miner inferred wrong template | `scripts/engine_miners/{engine}_dynamic_miner.py` (predicate-inference logic); the seven templates live in the same file or `_base.py` depending on engine |

The error classes (`MinerError`, `MinerVersionMismatchError`,
`MinerLandmarkMissingError`) live in `scripts/engine_miners/_base.py`
and are intentionally fail-loud: a previous extractor that swallowed
`ImportError` and returned `[]` silently degraded into "no rules
found", which masked broken extractors. Do not catch these without a
specific reason.

---

## Common debugging patterns

### Probe passes locally but fails in CI

The host has no engine libraries. Static analysis can run on the host
because miners read source via `inspect.getsource()`, but dynamic
miners and validation-replay must run inside the engine container. If
the probe passes on your laptop and fails in CI, the symptom is
usually a CUDA-aware import (the engine container has CUDA, your host
does not).

Run inside the container:

```bash
docker run --rm -v "$PWD":/workspace -w /workspace \
  llenergymeasure:{engine}-{version} \
  python -m scripts._probe --producer invariants
```

### Validation gate flips a previously-passing rule

The rule's `kwargs_positive` or `message_template` has drifted relative
to the live library's emission. Inspect
`_staging/_failed_validation_{engine}.yaml` to see which check
diverged:

- `positive_raises` failed - library no longer raises for the
  `kwargs_positive` shape. Either the library relaxed the constraint
  (rule is stale; remove or update) or the kwargs are now insufficient
  to trigger it (re-mine).
- `message_template_match` failed - library raises but the message
  template no longer matches. Update `message_template` to the new
  static fragment.
- `negative_does_not_raise` failed - library now raises for the
  `kwargs_negative` shape. The negative example is no longer valid;
  pick a different negative or remove the rule.

### Dynamic miner emits noisy false positives

Dynamic mining errs toward recall. The validation-CI gate is the
filter, not the miner. If a noisy candidate cluster appears, look at
`scripts/engine_miners/{engine}_dynamic_miner.py` for the cluster
definition and tighten the value sets so the Cartesian product is
smaller and more pointed.

### `manual_seed` rule lingers after the gap should have closed

`manual_seed` is pipeline-failure debt: each entry should close as
soon as the miner gains coverage for that pattern. Search for
`added_by: manual_seed` in the proposed YAML and check whether the
justification comment still applies. If the miner now covers the
pattern, the rule should be re-mined (and `added_by` updated to the
correct mechanical source) rather than left as `manual_seed`.

---

## See also

- Architecture: [engine introspection pipelines](/explanation/architecture/engine-introspection-pipelines) - how the pipeline works (conceptual)
- Reference: [invariants corpus format](/reference/invariants-corpus-format) - corpus YAML format spec
- Reference: [schema discovered format](/reference/schema-discovered-format) - the parallel pipeline's format spec
- Contributing: [extending miners](/contributing/extending-miners) - adding a new engine miner
- Contributing: [schema refresh (operations guide)](/contributing/schema-refresh) - the parallel pipeline's ops guide
- Architecture: [parameter discovery (runtime loader)](/explanation/architecture/parameter-discovery) - how the corpus is consumed at runtime
