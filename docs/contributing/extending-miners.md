# Extending the Invariant Miner: Adding a New Engine

This document is the practitioner's guide to adding invariant miner support for a new engine. It uses the transformers miner as the gold-standard reference throughout.

**Audience:** engine extenders. Assumes familiarity with the [miner-pipeline.md](/contributing/miner-pipeline) concepts.

---

## Before you start

1. Read [miner-pipeline.md](/contributing/miner-pipeline) to understand the static miner / dynamic miner / lift module split.
2. Read the [corpus format reference](/reference/invariants-corpus-format) to understand what rules look like.
3. Review `engine_versions/transformers/v4_57_3/producers/static_invariant_miner.py` and `engine_versions/transformers/v4_57_3/producers/dynamic_invariant_miner.py` as the gold standard. The comments in those files contain important design decisions. The `scripts/engine_producers/transformers_*_invariant_miner.py` modules are thin dispatcher shims that delegate to the per-version vendored producers.

---

## Step 0: Research the engine's validation surface

Before writing any code, answer these questions:

1. Which config classes does the engine validate? Where does validation happen - in `__init__`, in a separate `validate()` method, in validators decorated with `@model_validator`?

2. Which Python type system does each class use? (`pydantic.BaseModel` / `pydantic.dataclasses.dataclass`, `msgspec.Struct`, `@dataclasses.dataclass`, or something else?)

3. Does the engine constructor raise on invalid inputs, or silently normalise them? (Transformers and vLLM raise; TRT-LLM constructors are more permissive, so TRT-LLM has no dynamic miner.)

4. What is the CUDA / import dependency? Engines have no host install path - they are imported only inside their per-engine Docker images (see [development.md](/contributing/development)). Within the engine container, can the miner `import enginelib` on the CPU phase of the build, or does the import require a live CUDA runtime? (vLLM: importable inside `llenergymeasure:vllm-${VER}` without GPU at probe time; TRT-LLM: requires CUDA-aware import even inside the NGC container.)

5. What is a realistic post-validation-CI invariant count? (Transformers: 46; vLLM: 80-110; TRT-LLM: 20-28.) This helps plan the scope.

---

## The vendoring model

Each producer (static invariant miner, dynamic invariant miner, schema introspector) lives at two levels:

- **Per-version vendored module** at `engine_versions/<engine>/v<safe>/producers/{static_invariant_miner,dynamic_invariant_miner,schema_introspector}.py`. This is the real implementation. It pins its `LANDMARKS` tuple (the live class / method paths it expects) to the library version named in `engine_versions/<engine>/current.yaml:library.current_version`.

- **Dispatcher shim** at `scripts/engine_producers/<engine>_{static_invariant_miner,dynamic_invariant_miner,schema_introspector}.py`. ~14 lines via `scripts/engine_producers/_stub_factory.py`. Module-level `__getattr__` (PEP 562) resolves to `engine_versions.<engine>.<safe_version>.producers.<producer>` at attribute-access time. The orchestrator (`build_corpus.py`) imports the shim; the shim defers to the dispatcher.

When the SSOT bumps to a new library version, the dispatcher raises `ModuleNotFoundError` naming the exact file path to create. The maintainer copies the previous version's producer to the new path, adjusts `LANDMARKS` to match the new surface, and the cell re-runs green.

Step 1 of this guide walks through the shim contract; subsequent steps detail the per-version producer body.

---

## Step 1: Add the fail-loud import contract

Create `scripts/engine_producers/{engine}_static_invariant_miner.py` (and corresponding `_dynamic_invariant_miner` / `_schema_introspector` shims) using `_stub_factory.py`'s `make_static_stub` / `make_dynamic_stub` / `make_schema_stub`. The shim's job is to defer to the per-version producer module via the dispatcher; the fail-loud contract lives inside the per-version producer.

In `engine_versions/{engine}/v<safe>/producers/static_invariant_miner.py`,
declare the LANDMARKS tuple the dispatcher + probe gate on. The miner
does not perform a self-check on the installed library version: the
dispatcher resolves which `v<safe>/` archive to use from the
SSOT-pinned `library.current_version`, and the probe verifies the
LANDMARKS resolve under the live library before any mining runs.

Declare landmark checks for every class or method the miner will walk:

```python
import ast
import inspect
from scripts.engine_producers._base import find_class, find_method

from enginelib.config import SomeConfigClass

_source = inspect.getsource(SomeConfigClass)
_module = ast.parse(_source)
_cls = find_class(_module, "SomeConfigClass")
if _cls is None:
    raise MinerLandmarkMissingError(
        "SomeConfigClass",
        "expected in enginelib.config - check if the class was renamed"
    )
```

**Why this matters:** the Haiku-era TRT-LLM extractor imported `LlmConfig` - a class that does not exist in TRT-LLM 0.21.0. It caught the `ImportError` and silently returned `[]`. The fail-loud contract makes silent coverage loss impossible.

---

## Step 2: Apply the relevant lift module(s)

Based on your Step 0 research, apply one or more lift modules to extract constraints directly from type metadata.

All three lift modules expose a single function named `lift` with the same signature: `lift(target_type, *, namespace, today, source_path) -> list[InvariantCandidate]`. The engine/library is derived automatically from `target_type.__module__`. Import each lift under an alias to keep call sites readable.

### If the engine uses Pydantic v2

```python
from datetime import date
from scripts.engine_producers._pydantic_lift import lift as lift_pydantic
from enginelib.config import CacheConfig, SchedulerConfig

TODAY = date.today().isoformat()

def mine_pydantic_invariants():
    invariants = []
    for cls in [CacheConfig, SchedulerConfig]:
        invariants.extend(lift_pydantic(
            cls,
            namespace="myengine.config",
            today=TODAY,
            source_path="enginelib/config.py",
        ))
    return invariants
```

The lift emits one invariant per `Gt`, `Ge`, `Lt`, `Le`, `MultipleOf`, `MinLen`, `MaxLen` constraint and per `Literal[...]` allowlist found on any field.

### If the engine uses msgspec

```python
from scripts.engine_producers._msgspec_lift import lift as lift_msgspec
from enginelib.config import SamplingParams

def mine_msgspec_invariants():
    return lift_msgspec(
        SamplingParams,
        namespace="myengine.sampling",
        today=TODAY,
        source_path="enginelib/sampling.py",
    )
```

Note: if the class ships zero `Meta(ge=...)` annotations (common for msgspec classes), the lift returns `[]` - that is expected and not an error.

### If the engine uses stdlib dataclasses

```python
from scripts.engine_producers._dataclass_lift import lift as lift_dataclass
from enginelib.config import EngineArgs

def mine_dataclass_invariants():
    return lift_dataclass(
        EngineArgs,
        namespace="myengine.args",
        today=TODAY,
        source_path="enginelib/args.py",
    )
```

The dataclass lift is limited to `Literal[...]` value-allowlist invariants (no numeric bounds; stdlib dataclasses carry no bound metadata by default).

---

## Step 3: Write the static miner

Create `engine_versions/{engine}/v<safe>/producers/static_invariant_miner.py`. The static miner walks the AST of validator methods and emits rules for conditional raises, warnings, and silent normalisations.

### Pattern: walking a validator method

```python
import ast
import inspect
from scripts.engine_producers._base import (
    find_class, find_method, extract_condition_fields,
    filter_condition_references_self,
    ConditionalRaiseDetector, ConditionalSelfAssignDetector,
    ConditionalWarningsWarnDetector, ConditionalLoggerWarningDetector,
    InvariantCandidate, MinerSource,
)

def walk_validate_method(cls_source: str, cls_name: str) -> list[InvariantCandidate]:
    module = ast.parse(cls_source)
    cls_node = find_class(module, cls_name)
    if cls_node is None:
        raise MinerLandmarkMissingError(cls_name)

    validate = find_method(cls_node, "validate")
    if validate is None:
        raise MinerLandmarkMissingError(f"{cls_name}.validate")

    public_fields = frozenset(
        # derive from the class's dataclasses.fields() or __annotations__
    )

    detectors = (
        ConditionalRaiseDetector(),
        ConditionalSelfAssignDetector(),
        ConditionalWarningsWarnDetector(),
        ConditionalLoggerWarningDetector(),
    )

    candidates = []
    for node in ast.walk(validate):
        if not isinstance(node, ast.If):
            continue
        if not filter_condition_references_self(node.test, public_fields):
            continue
        for stmt in node.body:
            for detector in detectors:
                pattern = detector.detect(stmt)
                if pattern is not None:
                    # build InvariantCandidate from pattern + condition
                    candidate = _build_candidate(node.test, pattern, ...)
                    candidates.append(candidate)
    return candidates
```

### Per-engine detector customisation

The five default detectors cover the most common patterns. For engine-specific patterns, write a custom detector:

```python
# Example: engine uses self.errors.append(...) for error collection
class ErrorsAppendDetector:
    def detect(self, stmt: ast.stmt) -> DetectedPattern | None:
        if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
            return None
        path = call_func_path(stmt.value)
        if path != ["self", "errors", "append"]:
            return None
        return DetectedPattern(
            severity="error",
            emission_channel="none",
            affected_field=None,
            message_template=first_string_arg(stmt.value),
            detail="self.errors.append",
        )
```

### Important: the "revisit" comment

Per the transformers static miner's header, per-engine miners currently define their own `_detect_*` functions rather than using `_base.py`'s detector classes directly. This is because the `DetectedPattern` shape from `_base.py` doesn't carry the structured `FieldPredicate` data needed for cross-field corpus rules (operators like `not_divisible_by` and `@field_ref`). Once two or more engine miners exist and we can see whether the parallel detector logic is genuinely divergent or accidentally so, harmonise in a `_base.py` refactor.

---

## Step 4: Write the dynamic miner (if applicable)

Create `engine_versions/{engine}/v<safe>/producers/dynamic_invariant_miner.py` if the engine's constructors raise on invalid inputs.

**Skip this step if:** probing the engine's constructors yields zero raises. This is the case for TRT-LLM, where `TrtLlmArgs(**kwargs)` is extremely permissive at construction time; constraints are enforced in validator methods (covered by the static miner) or at build time.

### Cluster definition

Clusters group related fields for Cartesian probing:

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class _Cluster:
    name: str
    fields: list[str]
    values: dict[str, list[Any]]
    constructor: type  # e.g. SamplingParams
    validate_method: str | None = None  # e.g. "_verify_args"

CLUSTERS = [
    _Cluster(
        name="sampling_temperature",
        fields=["temperature", "top_p", "top_k"],
        values={
            "temperature": [0.0, 0.5, 1.0, 2.0, -0.1],
            "top_p": [0.0, 0.5, 1.0, 1.1],
            "top_k": [0, 1, 50, -1],
        },
        constructor=SamplingParams,
    ),
]
```

Cluster size rule: if `product(len(values[f]) for f in fields) > 200`, use Hypothesis as a supplement instead of Cartesian product:

```python
import itertools
import hypothesis.strategies as st
from hypothesis import given, settings

def probe_cluster(cluster: _Cluster) -> list[tuple[dict, str | None]]:
    size = 1
    for vs in cluster.values.values():
        size *= len(vs)

    if size <= 200:
        # Cartesian probe
        rows = []
        for combo in itertools.product(*[cluster.values[f] for f in cluster.fields]):
            kwargs = dict(zip(cluster.fields, combo))
            rows.append(_run_probe(kwargs, cluster))
        return rows
    else:
        # Hypothesis supplement (deterministic, fixed seed)
        return _hypothesis_probe(cluster)
```

**Important:** Hypothesis is used here as a deterministic value generator with a fixed seed - not as a property-based test runner. The pipeline must be deterministic: the same library version + miner code must produce the same corpus.

### Predicate inference

After probing, group error rows by message class and infer predicates:

```python
def infer_predicates(rows: list[tuple[dict, str | None]]) -> list[InvariantCandidate]:
    # Group by error message
    by_message: dict[str, list[dict]] = {}
    for kwargs, error in rows:
        if error is not None:
            by_message.setdefault(error, []).append(kwargs)

    candidates = []
    for message, trigger_kwargs in by_message.items():
        # Try templates in order of preference:
        # 1. cross-field divisibility: a % b != 0
        # 2. cross-field comparison: a > b
        # 3. type allowlist
        # 4. single-field range
        # 5. single-field equality
        # 6. value allowlist
        # Emit ALL plausible candidates (recall-first; validation CI prunes false positives)
        ...
    return candidates
```

---

## Step 5: Write the corpus orchestration entry

`engine_versions/{engine}/v<safe>/producers/static_invariant_miner.py` houses the orchestration. Lift modules, AST walkers, and cluster probes all live in this module (or its companion `dynamic_invariant_miner.py`); the dispatcher shim under `scripts/engine_producers/` only re-exports `mine` / `LANDMARKS` symbols. The pattern:

```python
def mine() -> list[InvariantCandidate]:
    candidates = []
    candidates.extend(mine_pydantic_invariants())
    candidates.extend(mine_dataclass_invariants())
    # Combinatorial cluster probes
    candidates.extend(mine_cluster_invariants())
    return candidates

if __name__ == "__main__":
    import yaml
    from scripts.engine_producers._base import candidate_to_dict
    results = mine()
    staging = {
        "schema_version": "1.0.0",
        "engine": ENGINE,
        "invariants": [candidate_to_dict(c) for c in results],
    }
    output_path = Path("src/llenergymeasure/engines/_staging/myengine_miner.yaml")
    output_path.write_text(yaml.dump(staging, allow_unicode=True))
    print(f"Wrote {len(results)} candidates to {output_path}")
```

---

## Step 6: Write fixpoint regression tests

Each per-engine miner ships with parametrised tests:

```python
# tests/unit/scripts/engine_producers/test_myengine_miner.py

import pytest
from scripts.engine_producers.myengine_miner import CLUSTERS

@pytest.mark.parametrize("cluster", CLUSTERS, ids=lambda c: c.name)
def test_cluster_probes_without_crashing(cluster):
    """Each cluster must complete probing without an unhandled exception."""
    rows = probe_cluster(cluster)
    assert isinstance(rows, list)

def test_version_envelope_resolves():
    """The miner pin loaded from the engine SSOT must be a non-empty SpecifierSet."""
    from packaging.specifiers import SpecifierSet
    from scripts.engine_producers._current import load_miner_pin
    envelope = load_miner_pin("myengine", "static")
    assert isinstance(envelope, SpecifierSet)
    assert str(envelope) != ""

def test_landmark_checks_raise_on_missing():
    """find_class returning None must raise MinerLandmarkMissingError."""
    from scripts.engine_producers._base import find_class, MinerLandmarkMissingError
    import ast
    module = ast.parse("class Unrelated: pass")
    cls = find_class(module, "SomeConfigClass")
    assert cls is None
    # Confirm caller raises (contract test)
    with pytest.raises(MinerLandmarkMissingError):
        if cls is None:
            raise MinerLandmarkMissingError("SomeConfigClass")
```

---

## Step 7: Add to CI

1. Decide the engine's CI shape:
   - **Upstream image consumer** (vllm / tensorrt pattern) - add a per-engine
     job to `engine-pipeline.yml` and `engine-pipeline.yml` mirroring the
     existing `invariants-vllm` / `schemas-vllm` (or `*-tensorrt`) jobs. The
     job pulls the upstream canonical image, then runs probe → mine →
     validate → doc-gen → atomic-writeback inline.
   - **First-party image (transformers pattern)** - split the build out into
     a pair of workflows modelled on `engine-pipeline.yml` (build + cache
     export, no runtime push) and `publish-engine-image.yml` (workflow_run-
     triggered, pulls cache + pushes runtime tags per parent event). The
     transformers cells in `engine-pipeline.yml` + `engine-pipeline.yml`
     then chain off Publish engine image success via `workflow_run`. The
     build/push split exists so push failures don't cost a full rebuild
     of the heavy from-source compile (e.g. ~30 min FA3 compile) on retry.

2. Set the runner: every engine miner runs inside its own Docker image
   (no host extras exist - see [development.md](/contributing/development)). For the
   upstream-image pattern, mirror `invariants-vllm` as the template for
   engines whose miners need a GPU only for `import`-time reasons; use
   `invariants-tensorrt` as the template for engines whose Python source
   layout shifts across image releases, that bundle source in non-
   introspectable ways (NGC-derived bases), or that require CUDA-aware
   imports: the cell downloads the upstream release tarball on the runner
   host and bind-mounts it into the container at a stable path, decoupling
   source resolution from the image's internals. For the first-party-image
   pattern, mirror `engine-pipeline.yml` + `publish-engine-image.yml` + the
   workflow_run-gated cell pair in engine-pipeline.yml / engine-pipeline.yml.

3. The validate step runs inside the engine's container in the same job as the miner - no separate validation workflow to update.

4. Add a Renovate `packageRule` so library bumps trigger the appropriate
   workflow via the `engine_versions/{engine}/current.yaml` path filter (or, for
   the first-party-image pattern, via `engine-pipeline.yml`'s filter -
   downstream `publish-engine-image.yml` and the workflow_run-gated cells fire
   automatically on its success).

---

## Step 8: Generate and review the corpus

Run the miner locally (inside the engine's Docker container if CUDA is required):

```bash
python scripts/engine_producers/build_corpus.py --engine myengine --producer static
# Runs the per-version vendored static miner via the dispatcher shim,
# writes src/llenergymeasure/engines/_staging/myengine_static_invariant_miner.yaml

python scripts/engine_producers/build_corpus.py --engine myengine
# Merges staging files, runs validation-CI gate, writes corpus

python scripts/validate_invariants.py \
  --engine myengine \
  --corpus src/llenergymeasure/engines/myengine.proposed.yaml \
  --out src/llenergymeasure/engines/myengine.validated.yaml
# Validates all rules against live library
```

Review the corpus manually:
- Do the `kwargs_positive` examples look right?
- Are there rules that fire too broadly (false positives)?
- Are there obvious constraints the miner missed (coverage gaps)?

If coverage gaps exist, extend the miner. Only add `manual_seed` rules as a last resort, with a justification comment.

---

## Transformers as the gold standard: key patterns

The transformers miner is the reference implementation. Key patterns to follow:

### The `find_class` / `find_method` / `MinerLandmarkMissingError` contract

Every class and method the miner walks must be guarded:

```python
cls_node = find_class(module, "GenerationConfig")
if cls_node is None:
    raise MinerLandmarkMissingError("GenerationConfig")

method_node = find_method(cls_node, "validate")
if method_node is None:
    raise MinerLandmarkMissingError("GenerationConfig.validate")
```

### The `public_fields` filter

Derive public fields from the class's dataclass fields or `__annotations__`, and use `filter_condition_references_self` to drop predicates that don't reference a public field:

```python
public_fields = frozenset(
    f.name for f in dataclasses.fields(GenerationConfig)
    if not f.name.startswith("_")
)
```

### Unparseable sub-clauses: log, don't drop

When the static miner encounters a condition sub-clause it cannot translate (e.g. an opaque function call), it logs the clause and emits the surrounding invariant with the parseable parts. The invariant is still useful; the validation-CI gate will confirm whether it fires correctly:

```python
# transformers_static_miner.py pattern:
if unparseable_clause:
    logger.debug(
        "static_miner: dropped sub-clause in %s.%s:%d: %s",
        cls_name, method_name, node.lineno, ast.unparse(sub_clause)
    )
    # Continue emitting the rule without the sub-clause
```

### Recall-first: emit all plausible candidates

Both static and dynamic miners err toward recall. The validation-CI gate is the prune step. Do not add extra filters "just in case" - if an invariant candidate is wrong, the validation-CI gate will quarantine it.

---

## Failure modes when libraries evolve

When Renovate bumps an engine library, the miner pipeline must catch behavioural drift before stale invariants ship. Failures fall into three categories: loud failures caught by the miner pipeline at mining time, loud failures caught by the validation gate at validation time, and one silent failure mode the YAML/JSON split was specifically designed to make visible.

### Loud failures caught by the probe + miner

The probe (`scripts._drift`) runs as the cell's first step and resolves every
declared LANDMARK against the live library. Failures surface as red CI on the
Renovate PR, blocking merge until the maintainer addresses them:

- **`MinerLandmarkMissingError`** - an expected class or method symbol is no longer present in the library source. Catches refactors where a class was renamed, moved to a different module, or an API was deprecated and removed.
  - Example: a hypothetical vLLM release dropping `vllm.sampling_params.StructuredOutputsParams` would raise `MinerLandmarkMissingError` at the landmark-check step before any AST walking begins.

- **`ImportError` / `AttributeError`** - propagated raw if the miner uses a library symbol that has been refactored without a landmark guard. The fail-loud principle requires letting these propagate; never wrap landmark imports in a `try/except` that returns `[]`. A previous TRT-LLM extractor was reverted specifically because it caught `ImportError` and silently degraded; the fail-loud contract prevents this class of regression.

The probe's verdict (`pass` | `fail`) is the runtime gate. The DriftReport JSON also carries diagnostic fields - `fingerprint_drift` and `landmarks_aliased` - that surface in the PR comment to direct maintainer attention on bumps that probe-pass but suggest the producer cut should advance.

### Loud failures caught by the validation gate

After mining completes and a YAML corpus is written, `validate_invariants.py --fail-on-divergence` replays each invariant's `kwargs_positive` and `kwargs_negative` against the live library inside the engine's Docker container:

- **`--fail-on-divergence`** flips the validation gate to non-zero exit when an existing invariant's declared `expected_outcome` no longer matches the library's actual behaviour. This catches three distinct kinds of behavioural drift:
  1. The library changed its validation behaviour for an existing rule (e.g. relaxed a numeric bound, changed an error type).
  2. The library dropped a rule entirely (the constraint no longer fires).
  3. The library added a new constraint path that the existing rule's `kwargs_negative` example now happens to trip.

All three engines (transformers, vLLM, TRT-LLM) have `--fail-on-divergence` operational as of PR #445. Gate-breaking divergences are P0 incidents - they block the Renovate PR from merging.

### Silent failure: recall regression

The validation gate above validates the invariants that *exist* in the corpus. It cannot tell you about invariants that *should* exist but no longer do, because the miner regressed and stopped finding them.

Concrete scenario: a refactor in `_pydantic_lift.py` changes how it walks `FieldInfo.metadata`, and the lift now finds 12 invariants where it previously found 30. The 18 lost invariants silently disappear from the corpus.

- The validation gate runs only on the 12 surviving invariants - every one of them passes.
- CI is green.
- The Renovate PR merges with a corpus that has 60% the recall it had before.
- Users hitting the lost validations get no constraint check at runtime.

**Mitigation: the proposed-vs-validated YAML pair (the trust seam).**

The engine-invariants pipeline (`engine-pipeline.yml`, with per-job `if:` gating selecting the right cell for each trigger source: `pull_request: paths` for vllm + tensorrt, `workflow_run` after Build engine image for transformers) mines the proposed corpus into `src/llenergymeasure/engines/{engine}/invariants.proposed.yaml` and then validates it into `src/llenergymeasure/engines/{engine}/invariants.validated.yaml` in the same job. Both YAMLs land in one atomic commit-back to the PR branch, and the per-pipeline diff comment includes both diffs.

Because the proposed-corpus diff is emitted alongside the validated diff, a miner refactor that silently drops 18 invariants shows up as 18 deletions in the proposed-corpus diff - a maintainer reading the PR notices the regression even when the validation gate's verdict on the surviving invariants is green.

The historical Stage-1 / Stage-2 split between `auto-mine.yml` and `invariant-miner.yml` forced the same property by serialising two workflows; the merger preserves the property by emitting two diffs from one workflow. Cross-reference: #450 (trust seam architecture decision), #465 (writeback contract).

### Tooling for diagnosis

The fail-loud probe and the YAML diff together cover the failure modes that trip on a routine library bump. One planned tool extends this for harder cases:

- **Compat-matrix sweep (#470).** Runs the probe + miner against every library version in a declared support range and reports per-version `(rule_count, divergences, errors)`. Surfaces "this miner mostly works on the new version but loses 3 rules" before a Renovate PR ever opens.

---

## Common mistakes

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| Catching `ImportError` on landmark imports | Silent degradation (returns `[]` on failure) | Let `ImportError` propagate; or raise `MinerLandmarkMissingError` explicitly |
| Cartesian-only probing with large clusters | Exponential probe count; CI timeouts | Add Hypothesis supplement for clusters > 200 combinations |
| Adding `manual_seed` rules for automatable constraints | Pipeline-failure debt | Extend the miner instead |
| Using Hypothesis as property-based test runner (not value generator) | Non-deterministic corpus | Use `hypothesis.strategies.from_type` with a fixed seed; never `@given` |
| Not calling `find_method` before walking | `AttributeError` on `None` if method renamed | Always guard: `if method is None: raise MinerLandmarkMissingError(...)` |

---

## See also

- [miner-pipeline.md](/contributing/miner-pipeline) - pipeline architecture reference
- [invariants-corpus-format.md](/reference/invariants-corpus-format) - corpus format
- [parameter-discovery.md](/explanation/architecture/parameter-discovery) - runtime validation
- [architecture-overview.md](/explanation/architecture/architecture-overview) - system overview
- `engine_versions/transformers/v4_57_3/producers/static_invariant_miner.py` - gold-standard static miner
- `engine_versions/transformers/v4_57_3/producers/dynamic_invariant_miner.py` - gold-standard dynamic miner
- `engine_versions/_dispatcher.py` and `scripts/engine_producers/_stub_factory.py` - per-version dispatch
- `scripts/_drift.py` - the drift tool that surfaces missing-vs-extra-vs-stable landmark state at probe time
- `scripts/engine_producers/_base.py` - shared infrastructure
