# Extending the Invariant Miner: Adding a New Engine

This document is the practitioner's guide to adding invariant miner support for a new engine. It uses the transformers miner as the gold-standard reference throughout.

**Audience:** engine extenders. Assumes familiarity with the [miner-pipeline.md](/contributing/miner-pipeline) concepts.

---

## Before you start

1. Read [miner-pipeline.md](/contributing/miner-pipeline) to understand the static miner / dynamic miner / lift module split.
2. Read the [corpus format reference](/reference/invariants-corpus-format) to understand what rules look like.
3. Review `engine_versions/transformers/v5_7_0/producers/static_invariant_miner.py` and `engine_versions/transformers/v5_7_0/producers/dynamic_invariant_miner.py` as the gold standard. The comments in those files contain important design decisions. The `scripts/engine_producers/transformers_*_invariant_miner.py` modules are thin dispatcher shims that delegate to the per-version vendored producers.

---

## Step 0: Research the engine's validation surface

Before writing any code, answer these questions:

1. Which config classes does the engine validate? Where does validation happen - in `__init__`, in a separate `validate()` method, in validators decorated with `@model_validator`?

2. Which Python type system does each class use? (`pydantic.BaseModel` / `pydantic.dataclasses.dataclass`, `msgspec.Struct`, `@dataclasses.dataclass`, or something else?)

3. Does the engine constructor raise on invalid inputs, or silently normalise them? (Transformers and vLLM raise; TRT-LLM constructors are more permissive, so TRT-LLM has no dynamic miner.)

4. What is the CUDA / import dependency? Engines have no host install path - they are imported only inside their per-engine Docker images (see [development.md](/contributing/development)). Within the engine container, can the miner `import enginelib` on the CPU phase of the build, or does the import require a live CUDA runtime? (vLLM: importable inside `llenergymeasure:vllm-${VER}` without GPU at probe time; TRT-LLM: requires CUDA-aware import even inside the NGC container.)

5. What is a realistic post-validation-CI invariant count? (Transformers: 46; vLLM: 80-110; TRT-LLM: 20-28.) This helps plan the scope.

---

## The new-engine contract

Adding an engine means producing seven things. Items 3-6 are the
mining code; items 1-2 decide its shape; item 7 wires it into dispatch.
The remaining steps of this guide expand each in order.

1. **An introspection mechanism.** Decide how the miner reaches the
   engine's validation source. Two patterns exist, plus one recovery
   technique:
   - *Import-driven framework reflection.* The library imports inside
     the engine's CPU build phase, so the miner imports the live package
     and reads source via `inspect.getsource()` (transformers, vLLM).
   - *Source-AST of an extracted tarball.* The library cannot import on
     the build host (TRT-LLM 0.21.0 needs a live CUDA runtime even inside
     the NGC container), so the miner AST-walks an extracted source tree
     and never imports the package (tensorrt).
   - *Docstring-type recovery.* When the config constructor is
     `__init__(self, **kwargs)` and carries no field annotations,
     introspection of the signature yields `type='unknown'`. Recover
     field types from the docstring Args block instead. The transformers
     schema introspector does this for `GenerationConfig` via
     `docstring_arg_types`.
2. **Per-version landmark DATA** at
   `engine_versions/<engine>/v<safe>/landmarks.yaml`. The data the miner
   walks with - probe landmarks, and (depending on mechanism) source-tree
   layout, class / method targets, StrEnum field map, or AST walk
   targets. See [Landmark data: `landmarks.yaml`](#landmark-data-landmarksyaml).
3. **A static miner** exposing the walk + emit + `main(--out)` contract
   that `build_corpus.py` invokes (Step 3).
4. **An engine-appropriate local detector set** and predicate vocabulary
   (Step 3). Do not wire a shared detector framework - it was removed.
   Each engine defines its own `_detect_*` functions.
5. **A schema `discover()` body** reusing the `_common` primitives
   (Step 5).
6. **Dispatch wiring** (Step 6): the `current.yaml` library pin, the
   vendored `producers/` directory, the `_stub_factory` shims, and
   registration in `scripts/_drift._PRODUCER_MODULES` plus
   `build_corpus._ENGINE_EXTRACTORS`.
7. **An optional dynamic miner** (Step 4) - only when the engine's
   constructors raise on invalid inputs. TRT-LLM has none by design.

---

## The dispatch model

Each producer (static invariant miner, dynamic invariant miner, schema introspector) lives at two levels, and its version-varying data lives in a third file:

- **Per-version vendored module** at `engine_versions/<engine>/v<safe>/producers/{static_invariant_miner,dynamic_invariant_miner,schema_introspector}.py`. This is the real implementation: the *walking algorithm*, which is version-stable. It does not pin a `LANDMARKS` tuple inline; instead it binds its data at import time from the externalised landmarks file (see below).

- **Per-version landmark data** at `engine_versions/<engine>/v<safe>/landmarks.yaml`. The data the algorithm walks with - probe landmarks plus optional source-tree / class-target / AST-target fields - keyed to the library version named in `engine_versions/<engine>/current.yaml:library.current_version`. The producer loads it via `load_landmarks(ENGINE, current_version(ENGINE))` and re-exposes the probe tuple as a module attribute named `LANDMARKS`, because `scripts/_drift._read_landmarks` reads `module.LANDMARKS`.

- **Dispatcher shim** at `scripts/engine_producers/<engine>_{static_invariant_miner,dynamic_invariant_miner,schema_introspector}.py`. ~5 lines via `scripts/engine_producers/_stub_factory.py`. Module-level `__getattr__` (PEP 562) resolves to `engine_versions.<engine>.<safe_version>.producers.<producer>` at attribute-access time. `build_corpus.py` invokes the shim; the shim defers to the dispatcher.

Both the code dispatcher (`engine_versions/_dispatcher.py`) and the landmark-data loader (`scripts/engine_producers/_landmarks.py`) resolve `(engine, version)` the same way: exact-match `v<safe(version)>/` wins; on a miss they fall back to the highest vendored version **at or below** the target that carries the required file. There is no fall-forward. When neither code nor data exists at or below the target, the resolver raises naming the exact path to create.

Because the algorithm is version-stable and only the data moves, one producer per engine matches its SSOT pin (exact). A patch-version bump whose landmarks still resolve is zero-touch: the dispatcher and loader fall back to the existing `v<safe>/`, and the probe verifies the landmarks resolve under the live library. A bump that genuinely shifts the surface needs a fresh `v<safe>/landmarks.yaml` (and, if the walk shape changed, a fresh `producers/` directory). The probe is the runtime gate either way.

Step 1 of this guide details the landmarks file; subsequent steps detail the per-version producer body.

---

## Step 1: Supply the landmark data and the fail-loud contract

The fail-loud import contract lives inside the per-version producer; the *data* it gates on lives in `landmarks.yaml`. Author both.

### Landmark data: `landmarks.yaml`

Create `engine_versions/{engine}/v<safe>/landmarks.yaml`. The data shape you supply depends on your engine's introspection mechanism (contract item 1). `scripts/engine_producers/_landmarks.py` parses the document into a typed `Landmarks` dataclass; `probe_landmarks` is the only required key. The three real shapes in the tree:

| Field | Required? | Carried by | Meaning |
|---|---|---|---|
| `probe_landmarks` | yes | all three engines | Dotted attribute paths `scripts/_drift.py` resolves under the installed library - the runtime drift gate. Re-exposed by the producer as `module.LANDMARKS`. |
| `ast_targets` | optional | vLLM (import-driven AST) | `{module_attr, method, namespace, native_type}` walk targets: the `module.Class` attribute path under the package, the validator method whose body is walked, the invariant namespace, and the native config type. |
| `source.root` | optional | tensorrt (source-driven) | Template for the extracted source tree; the `{version}` token is substituted with the requested library version. |
| `source.files` | optional | tensorrt | Named relative paths (`llm_args`, `builder`) the miner AST-walks under `source.root`. |
| `class_targets` | optional | tensorrt | `{class, file}` entries the miner fails loud on if absent from source. |
| `method_landmarks` | optional | tensorrt | `{class, method}` validator methods whose bodies are walked. |
| `strenum_fields` | optional | tensorrt | `{enum, field}` map of a StrEnum class to the field it constrains (StrEnum-typed allowlists). |

The three mechanisms map onto the schema as follows:

- **transformers** (import-driven, probe-only): carries only `probe_landmarks`. The miner imports the live package and hardcodes its walk targets (`GenerationConfig`, `WatermarkingConfig`, `SynthIDTextWatermarkingConfig`, `BitsAndBytesConfig`) in its function bodies, so no `source.*` / `ast_targets` are needed - those targets are algorithm structure, not version-varying data.
- **vLLM** (import-driven, AST-walk): carries `probe_landmarks` plus a four-field `ast_targets` list. The miner imports the live package and AST-walks the methods named in `ast_targets`; the flat-vs-subpackage layout and validator naming shift across versions, so the walk surface is data.
- **tensorrt** (source-driven): carries `probe_landmarks` plus `source.root` / `source.files`, `class_targets`, `method_landmarks`, and `strenum_fields`. The library needs CUDA to import, so the miner AST-walks an extracted tarball rather than importing - the source-tree layout is data.

### The fail-loud contract in the producer

The producer binds the data at module load and re-exposes the probe tuple as `LANDMARKS`:

```python
from scripts.engine_producers._current import current_version
from scripts.engine_producers._landmarks import load_landmarks

ENGINE = "myengine"

# Landmark DATA is externalised; the producer loads it for the SSOT-pinned
# version (with <=target fallback, mirroring the code dispatcher).
_LANDMARKS = load_landmarks(ENGINE, current_version(ENGINE))

# Kept as a module attribute because scripts/_drift._read_landmarks reads
# ``module.LANDMARKS`` for the probe.
LANDMARKS: tuple[str, ...] = _LANDMARKS.probe_landmarks
```

The miner does not self-check the installed library version: the dispatcher resolves which `v<safe>/` archive to use from the SSOT-pinned `library.current_version`, and the probe (`scripts/_drift`) verifies the `LANDMARKS` resolve under the live library before any mining runs. Inside the walk, guard every class or method you read with `find_class` / `find_method` and raise `MinerLandmarkMissingError` when one returns `None`:

```python
import ast
import inspect
from scripts.engine_producers._base import find_class, MinerLandmarkMissingError

from enginelib.config import SomeConfigClass

_module = ast.parse(inspect.getsource(SomeConfigClass))
_cls = find_class(_module, "SomeConfigClass")
if _cls is None:
    raise MinerLandmarkMissingError(
        "SomeConfigClass",
        "expected in enginelib.config - check if the class was renamed",
    )
```

**Why this matters:** a previous TRT-LLM extractor imported `LlmConfig` - a class that does not exist in TRT-LLM 0.21.0. It caught the `ImportError` and silently returned `[]`. The fail-loud contract makes silent coverage loss impossible.

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

There is **no shared detector framework** to import. Each engine defines its own local `_detect_*` functions and its own `DetectedBody`-style record, because the invariant shapes genuinely diverge per engine (see [Local detectors, not a shared framework](#local-detectors-not-a-shared-framework) below). `_base.py` provides only mechanical leaf primitives - the AST text helpers (`call_func_path`, `first_string_arg`, `render_joinedstr_template`, ...), `find_class` / `find_method`, the `InvariantCandidate` / `MinerSource` output types, and the `MinerLandmarkMissingError` fail-loud type. Compose those.

### Pattern: detectors over `if` bodies

A detector inspects one statement and returns a local detected-body record (or `None`). Define the set your engine needs:

```python
import ast
from dataclasses import dataclass
from scripts.engine_producers._base import call_func_path, first_string_arg


@dataclass
class DetectedBody:
    """Local to this engine - shape it to what your invariants need."""
    severity: str            # "error" | "warn" | "dormant"
    outcome: str
    emission_channel: str
    affected_field: str | None
    message_template: str | None
    detail: str


def _detect_raise(stmt: ast.stmt) -> DetectedBody | None:
    if not isinstance(stmt, ast.Raise) or stmt.exc is None:
        return None
    return DetectedBody(
        severity="error", outcome="error", emission_channel="none",
        affected_field=None,
        message_template=first_string_arg(stmt.exc) if isinstance(stmt.exc, ast.Call) else None,
        detail="raise",
    )


def _detect_logger_warning(stmt: ast.stmt) -> DetectedBody | None:
    if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
        return None
    path = call_func_path(stmt.value)
    if path != ["logger", "warning"]:
        return None
    return DetectedBody(
        severity="warn", outcome="warn", emission_channel="logger_warning",
        affected_field=None, message_template=first_string_arg(stmt.value),
        detail="logger.warning",
    )


_DETECTORS = (_detect_raise, _detect_logger_warning)


def _detect_body(stmt: ast.stmt) -> DetectedBody | None:
    for det in _DETECTORS:
        result = det(stmt)
        if result is not None:
            return result
    return None
```

Walk the guarded validator method, descend into `if` bodies, run `_detect_body` on each statement, and build an `InvariantCandidate` from the accumulated condition predicates plus the detected body. The condition predicate must reference a public field via `self.<field>`; drop predicates that do not.

### How many detectors does your engine need?

The detector set is engine-specific - size it to the validator patterns the engine actually emits. The three reference engines differ:

| Engine | Detectors | Set |
|---|---|---|
| tensorrt | 2 | `_detect_raise`, `_detect_logger_warning` |
| vLLM | 4 | `_detect_raise`, `_detect_self_assign`, `_detect_logger_warning`, `_detect_warnings_warn` |
| transformers | 6 | the above plus `_detect_assert` and `_detect_minor_issues` (HuggingFace's `minor_issues[key] = msg` announced-dormancy pattern) |

### Local detectors, not a shared framework

A shared `_base.py` detector framework (`ConditionalRaiseDetector`, `ConditionalSelfAssignDetector`, `ConditionalWarningsWarnDetector`, `ConditionalLoggerWarningDetector`, `MinorIssuesDictAssignDetector`, a `DetectedPattern` record, and `default_detectors`) was tried. With all three engines built, it turned out to be dead - none of the miners used it. It was removed.

The divergence is genuine, not accidental:

- The detector *set size* differs (2 / 4 / 6 above) because the engines emit different validator shapes.
- transformers needs structured `FieldPredicate` data carrying cross-field operators - `not_divisible_by` and `@field_ref` cross-field references for shapes like `num_beams % num_beam_groups != 0` - that vLLM and TRT-LLM never emit. A shared `DetectedPattern` record could not carry that without becoming the union of every engine's needs.

So the revisit is **resolved**: define your detectors and your detected-body record locally, in your engine's producer. Do not reach for a `_base.py` detector base - there isn't one, by decision.

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

## Step 5: Write the schema introspector

Create `engine_versions/{engine}/v<safe>/producers/schema_introspector.py` with a `discover(repo_root, image_ref) -> dict` body that reuses the shared envelope and per-field helpers from `scripts/engine_producers/_common.py`. The introspector reads the engine's typed API surface (`inspect.signature`, Pydantic `model_json_schema()`, `dataclasses.fields()`, msgspec) and emits the deterministic `schema.discovered.json` envelope - top-level metadata plus `engine_params` / `sampling_params` sections and a `discovery_limitations` list.

When a config class is `__init__(self, **kwargs)` with no field annotations (contract item 1, docstring-type recovery), the signature yields `type='unknown'`. Recover field types from the docstring Args block - the transformers introspector reads `GenerationConfig`'s docstring via `_common.docstring_arg_types` / `annotation_to_type_str`, and records the opaque `**kwargs` themselves as a `discovery_limitations` entry rather than inventing fields for them.

The full envelope shape is the [schema discovered format](/reference/schema-discovered-format); the conceptual treatment is in [engine introspection pipelines](/explanation/architecture/engine-introspection-pipelines).

---

## Step 6: Write the corpus orchestration entry and wire dispatch

### The `main(--out)` contract

Each miner is invoked by `build_corpus.py` as `python -m {module} --out {staging_path}`, so every static / dynamic producer must expose a `main(argv)` that accepts a single `--out` argument and writes a corpus-shaped staging YAML to that path. The lift modules, AST walkers, and cluster probes all live in the per-version producer (or its companion `dynamic_invariant_miner.py`); the dispatcher shim under `scripts/engine_producers/` carries no logic of its own. The pattern (mirroring the transformers static miner):

```python
import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("src/llenergymeasure/engines/myengine/_staging/myengine_static_invariant_miner.yaml"),
    )
    args = parser.parse_args(argv)

    candidates, engine_version, rel_path = walk_myengine()
    text = emit_yaml(candidates, engine_version=engine_version, rel_path=rel_path)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"Wrote {len(candidates)} candidate invariants to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### Wire dispatch

With the producers and `landmarks.yaml` in place, wire the engine into the four dispatch seams (contract item 6):

1. **Pin the library version.** Set `library.current_version` in `engine_versions/{engine}/current.yaml`. The dispatcher and the landmark-data loader both resolve `v<safe>/` from this pin.
2. **Vendor the producers.** The `engine_versions/{engine}/v<safe>/producers/` directory holds `static_invariant_miner.py`, the optional `dynamic_invariant_miner.py`, and `schema_introspector.py`, each with a sibling `__init__.py`.
3. **Add the `_stub_factory` shims.** Create `scripts/engine_producers/{engine}_static_invariant_miner.py` (and the `_dynamic_invariant_miner` / `_schema_introspector` shims) using `_stub_factory.make_static_stub` / `make_dynamic_stub` / `make_schema_stub`. ~5 lines each; the shim binds `(engine, producer)` and the dispatcher does the resolution.
4. **Register in both tables.** Add the producer module paths to `scripts/_drift._PRODUCER_MODULES` (keyed `(engine, "invariants")` and `(engine, "schemas")` - this is the probe seam) and add a `_Extractor` row per staging producer to `scripts/engine_producers/build_corpus._ENGINE_EXTRACTORS` (this is the mine seam). The two tables are the single places that enumerate each engine's producers.

---

## Step 7: Write fixpoint regression tests

Each per-engine miner ships with parametrised tests:

```python
# tests/unit/scripts/engine_producers/test_myengine_miner.py

import pytest
from engine_versions.myengine.v<safe>.producers.dynamic_invariant_miner import (
    CLUSTERS, probe_cluster,
)

@pytest.mark.parametrize("cluster", CLUSTERS, ids=lambda c: c.name)
def test_cluster_probes_without_crashing(cluster):
    """Each cluster must complete probing without an unhandled exception."""
    rows = probe_cluster(cluster)
    assert isinstance(rows, list)

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

## Step 8: Add to CI

1. Decide the engine's CI shape:
   - **Upstream image consumer** (vllm / tensorrt pattern) - add the engine
     to the `invariants-others` / `schemas-others` matrix in
     `engine-pipeline.yml`, which fans out the `_engine-rules-cell.yml` and
     `_engine-schemas-cell.yml` reusable cells per engine. The cell pulls the
     upstream canonical image, then runs probe → mine → validate → doc-gen →
     atomic-writeback inline.
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
   upstream-image pattern, mirror the vllm cell configuration as the template
   for engines whose miners need a GPU only for `import`-time reasons; use
   the tensorrt cell configuration as the template for engines whose Python source
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

## Step 9: Generate and review the corpus

Run the miner locally (inside the engine's Docker container if CUDA is required):

```bash
python scripts/engine_producers/build_corpus.py --engine myengine --producer static
# Runs the per-version vendored static miner via the dispatcher shim,
# writes src/llenergymeasure/engines/_staging/myengine_static_invariant_miner.yaml

python scripts/engine_producers/build_corpus.py --engine myengine
# Merges staging files, runs validation-CI gate, writes corpus

python scripts/validate_rules.py \
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

Derive public fields from the class's dataclass fields or `__annotations__`, and use `_base.extract_condition_fields` (which returns the `self.<field>` names a condition references) to drop predicates that don't reference a public field:

```python
from scripts.engine_producers._base import extract_condition_fields

public_fields = frozenset(
    f.name for f in dataclasses.fields(GenerationConfig)
    if not f.name.startswith("_")
)

# Skip an `if` whose condition references no public field.
if not (extract_condition_fields(node.test) & public_fields):
    continue
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

After mining completes and a YAML corpus is written, `validate_rules.py --fail-on-divergence` replays each invariant's `kwargs_positive` and `kwargs_negative` against the live library inside the engine's Docker container:

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

The engine-rules pipeline (`engine-pipeline.yml`, with per-job `if:` gating selecting the right cell for each trigger source: `pull_request: paths` for vllm + tensorrt, `workflow_run` after Build engine image for transformers) mines the proposed corpus into `src/llenergymeasure/engines/{engine}/rules.proposed.yaml` and then validates it into `src/llenergymeasure/engines/{engine}/rules.validated.yaml` in the same job. Both YAMLs land in one atomic commit-back to the PR branch, and the per-pipeline diff comment includes both diffs.

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
| Reaching for a shared `_base.py` detector framework | There isn't one - it was removed as dead | Define your `_detect_*` functions and detected-body record locally in your producer |
| Pinning a `LANDMARKS` tuple inline in the producer | Drifts from the externalised data; the probe reads `module.LANDMARKS` | Put the data in `v<safe>/landmarks.yaml`; bind via `load_landmarks(...)` and re-export `LANDMARKS` |

---

## See also

- [miner-pipeline.md](/contributing/miner-pipeline) - pipeline architecture reference
- [invariants-corpus-format.md](/reference/invariants-corpus-format) - corpus format
- [parameter-discovery.md](/explanation/architecture/parameter-discovery) - runtime validation
- [architecture-overview.md](/explanation/architecture/architecture-overview) - system overview
- `engine_versions/transformers/v5_7_0/producers/static_invariant_miner.py` - gold-standard static miner
- `engine_versions/transformers/v5_7_0/producers/dynamic_invariant_miner.py` - gold-standard dynamic miner
- `engine_versions/transformers/v5_7_0/landmarks.yaml` - probe-only landmark data (import-driven example)
- `engine_versions/tensorrt/v0_21_0/landmarks.yaml` - source-driven landmark data (extracted-tarball example)
- `engine_versions/vllm/v0_7_3/landmarks.yaml` - import-driven AST landmark data (`ast_targets` example)
- `scripts/engine_producers/_landmarks.py` - the `load_landmarks` loader + `Landmarks` schema
- `engine_versions/_dispatcher.py` and `scripts/engine_producers/_stub_factory.py` - per-version dispatch
- `scripts/_drift.py` - the drift tool that surfaces missing-vs-extra-vs-stable landmark state at probe time
- `scripts/engine_producers/_base.py` - shared leaf primitives (AST helpers, `InvariantCandidate`, `find_class` / `find_method`)
