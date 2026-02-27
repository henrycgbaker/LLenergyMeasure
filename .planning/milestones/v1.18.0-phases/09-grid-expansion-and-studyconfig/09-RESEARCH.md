# Phase 9: Grid Expansion and StudyConfig - Research

**Researched:** 2026-02-27
**Domain:** YAML sweep parsing, Cartesian grid expansion, Pydantic config modelling, hash computation
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Phase boundary:**
- Scope is pure data transformation: YAML in, resolved `list[ExperimentConfig]` out
- No subprocess execution, no manifest writing, no CLI wiring in this phase

**Pre-flight display:**
- Count summary only: `Study [hash]: 4 configs x 3 cycles = 12 runs`
- No time estimates — just experiment count and total run count
- Hash displayed in pre-flight (git/Docker style — researchers reference it in papers)
- Display ordering mode name only ("Order: interleaved"), not the full execution sequence
- `--dry-run` detailed config listing deferred to Phase 12 (CLI concern)

**Invalid combination handling:**
- Count + per-skip log line with reason: "Skipping 3/15 (pytorch, fp32, batch=32): [Pydantic message]"
- Light wrapping: prepend which config failed, let Pydantic provide the "why"
- Warn prominently if >50% of generated configs are invalid
- Hard error (exit 1) if ALL configs are invalid
- Skipped configs + validation reasons persisted in study metadata for post-hoc review

**Base config resolution:**
- `base:` field included in Phase 9 scope (optional DRY convenience)
- Hard error at parse time if base file does not exist
- Path resolution: relative to the study.yaml file's directory
- `base:` accepts experiment config files only — not study files; one level of inheritance, no chaining

**Cycle ordering:**
- Shuffle seed: derived from `study_design_hash` by default (same study = same shuffle, reproducible)
- Optional `shuffle_seed:` field in `execution:` block for explicit override
- Shuffle seed is NOT part of `study_design_hash`

### Claude's Discretion

- Confirmation prompt threshold for large sweeps (whether to auto-proceed or pause above N experiments)
- `base:` field scoping details (how to detect experiment vs non-experiment YAML)
- Exact format of pre-flight display (spacing, colours, alignment)
- Internal structure of the grid expander (function signatures, module placement)

### Deferred Ideas (OUT OF SCOPE)

- Documentation write-up phase for M1, M2, M3 — future milestone phases
- `--dry-run` flag showing full resolved config list — Phase 12 (CLI integration)
- VRAM estimation in dry-run mode — Phase 12 or later
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| CFG-11 | `StudyConfig` = resolved container: `list[ExperimentConfig]` + `ExecutionConfig` | `StudyConfig` model exists as M1 stub in `config/models.py`; requires expansion with `sweep:`, `execution:`, `base:` support |
| CFG-12 | Sweep resolution at YAML parse time, before Pydantic validation | `load_study_config()` function must exist in `config/loader.py`; resolution happens pre-Pydantic at raw dict level |
| CFG-13 | Dotted notation sweep keys: `pytorch.batch_size: [1, 8]` — backend-scoped grid | `_unflatten()` in `loader.py` already parses dotted keys; grid expander extends this to sweep dimensions |
| CFG-14 | Three modes: grid sweep (Cartesian), explicit `experiments:` list, combined | `itertools.product` for Cartesian; list concat for explicit; both modes resolved before Pydantic sees the data |
| CFG-15 | `ExecutionConfig`: `n_cycles` (default=1), `cycle_order`, `config_gap_seconds`, `cycle_gap_seconds` | New Pydantic model alongside `StudyConfig`; `shuffled` ordering requires `shuffle_seed` derived from `study_design_hash` |
| CFG-16 | `study_design_hash` = SHA-256[:16] of sweep+experiments only (execution block excluded) | Pattern identical to `compute_measurement_config_hash()` in `domain/experiment.py`; hash computed over `ExperimentConfig` list after expansion |
</phase_requirements>

---

## Summary

Phase 9 is pure data transformation with no side effects, no subprocess involvement, and no I/O beyond YAML file reading. The work is: parse study YAML, expand sweep dimensions into a flat `list[ExperimentConfig]`, apply n_cycles and cycle ordering, compute `study_design_hash`, and display a pre-flight count summary. Every component uses Python stdlib plus existing project dependencies (Pydantic, PyYAML, hashlib).

The codebase already has all necessary building blocks: `ExperimentConfig` with full backend sections exists in `config/models.py`; `StudyConfig` exists as an M1 stub (`experiments: list[ExperimentConfig]` + `name`); `_unflatten()` in `config/loader.py` already handles dotted key expansion for CLI overrides; `compute_measurement_config_hash()` in `domain/experiment.py` shows the exact hash pattern to replicate for `study_design_hash`. The module placement is specified in `.product/designs/architecture.md`: new files go in a `study/` package (`study/grid.py`).

The primary complexity is the multi-mode sweep grammar (grid / explicit / combined / backend-scoped) and the cycle ordering logic (interleaved round-robin vs sequential vs shuffled). Both are well-specified in `.product/designs/study-yaml.md`. Invalid combination handling requires collecting all Pydantic `ValidationError` instances upfront before surfacing them, then deciding continue/warn/hard-error based on skip rate. The `base:` inheritance adds one pre-processing step at parse time (load base file, merge, then resolve sweep on top).

**Primary recommendation:** Implement `study/grid.py` as a pure module with `expand_grid(raw_study_dict, study_yaml_path) -> tuple[list[ExperimentConfig], list[SkippedConfig]]` plus extend `StudyConfig` and add `ExecutionConfig` to `config/models.py`. Add `load_study_config()` to `config/loader.py`.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `itertools.product` | stdlib | Cartesian grid expansion | Used identically by Hydra, W&B, Ray Tune, lm-eval — universal ML sweep pattern |
| `hashlib.sha256` | stdlib | `study_design_hash` computation | Identical to `compute_measurement_config_hash()` already in codebase |
| `pydantic` | 2.x (existing) | `StudyConfig`, `ExecutionConfig`, `ExperimentConfig` validation | Already the project's validation framework |
| `yaml.safe_load` | PyYAML (existing) | Study YAML parsing | Already used in `config/loader.py` |
| `pathlib.Path` | stdlib | File path resolution for `base:` | Project standard per CLAUDE.md |
| `json.dumps` / `sort_keys=True` | stdlib | Canonical serialisation for hash | Identical pattern already in `domain/experiment.py` |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `random.shuffle` with seeded `Random` | stdlib | `shuffled` cycle ordering | Seeded from `study_design_hash` for reproducibility |
| `difflib.get_close_matches` | stdlib | "did you mean?" for unknown sweep keys | Already used via Levenshtein in `loader.py` — consistency |
| Rich / `typer.echo` | existing | Pre-flight count display | Display only; no Rich dependency introduced in `grid.py` |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `itertools.product` | Hand-rolled nested loops | No reason to hand-roll — `product` is standard, readable, correct |
| SHA-256[:16] | MD5, UUID | SHA-256 already used for `measurement_config_hash` — consistency |
| Pure function in `study/grid.py` | Method on `StudyConfig` | Pure function is easier to test and matches lm-eval's "expand first" pattern |

**Installation:** No new dependencies. All stdlib + existing project deps.

---

## Architecture Patterns

### Recommended Project Structure

```
src/llenergymeasure/
├── config/
│   ├── models.py          # ADD: ExecutionConfig, expand StudyConfig (M1 stub → full)
│   └── loader.py          # ADD: load_study_config()
└── study/                 # NEW package
    ├── __init__.py        # minimal exports
    └── grid.py            # expand_grid(), SkippedConfig, CycleOrder enum
```

Phase 9 creates `study/` package (grid only). `study/runner.py` and `study/manifest.py` come in Phases 10-11.

### Pattern 1: Expand-First, Validate-Second

**What:** Fully resolve the raw dict Cartesian product before any Pydantic construction. Collect all `ValidationError` instances. Then decide on continue/warn/error.

**When to use:** Always — this is the industry pattern (Hydra BasicLauncher, lm-eval task loading). Never interleave grid expansion with validation.

**Example (from `.planning/research/SUMMARY.md` and `.product/designs/study-yaml.md`):**

```python
# study/grid.py

import itertools
from pydantic import ValidationError
from llenergymeasure.config.models import ExperimentConfig

def expand_grid(
    raw_study: dict,
    study_yaml_path: Path | None = None,
) -> tuple[list[ExperimentConfig], list[SkippedConfig]]:
    """Expand sweep dimensions into a flat list of ExperimentConfig.

    Resolution order:
      1. Built-in ExperimentConfig defaults
      2. base: experiment.yaml (optional DRY inheritance)
      3. Inline study fields (model, backend, dataset, n, etc.)
      4. sweep: block — generates Cartesian product
      5. experiments: list — appended explicitly
    """
    # 1. Resolve base: (optional DRY inheritance)
    base_dict = _load_base(raw_study.get("base"), study_yaml_path)

    # 2. Build the "fixed" dict (non-sweep, non-experiments keys)
    fixed = {**base_dict, **_extract_fixed(raw_study)}

    # 3. Expand sweep: block into raw config dicts
    sweep_configs = _expand_sweep(raw_study.get("sweep", {}), fixed)

    # 4. Append explicit experiments: list
    explicit_configs = [
        {**fixed, **exp} for exp in raw_study.get("experiments", [])
    ]

    all_raw = sweep_configs + explicit_configs

    # 5. Pydantic validation — collect all errors upfront
    valid, skipped = [], []
    for raw in all_raw:
        try:
            valid.append(ExperimentConfig(**raw))
        except ValidationError as e:
            skipped.append(SkippedConfig(raw_config=raw, reason=e))

    return valid, skipped


def _expand_sweep(sweep: dict, fixed: dict) -> list[dict]:
    """Cartesian product of sweep dimensions.

    Dotted keys (pytorch.batch_size) are backend-scoped and generate
    independent grids per backend. Non-dotted keys are universal.
    """
    if not sweep:
        return [fixed] if fixed.get("model") else []

    universal_dims = {}   # key -> list[value]
    scoped_dims = {}      # backend -> {key -> list[value]}

    for key, values in sweep.items():
        if "." in key:
            backend, param = key.split(".", 1)
            scoped_dims.setdefault(backend, {})[param] = values
        else:
            universal_dims[key] = values

    # Determine backends: from sweep scopes or from fixed backend field
    backends = list(scoped_dims.keys()) if scoped_dims else [fixed.get("backend", "pytorch")]

    configs = []
    for backend in backends:
        backend_specific = scoped_dims.get(backend, {})
        all_dims = {**universal_dims, **backend_specific}

        keys = list(all_dims.keys())
        for combo in itertools.product(*[all_dims[k] for k in keys]):
            overrides = dict(zip(keys, combo))
            # Backend-scoped params go into the backend section dict
            config_dict = {**fixed, "backend": backend}
            for k, v in overrides.items():
                if k in backend_specific:
                    config_dict.setdefault(backend, {})[k] = v
                else:
                    config_dict[k] = v
            configs.append(config_dict)

    return configs
```

### Pattern 2: Hash Computation (Exclude Execution Block)

**What:** Compute `study_design_hash` over the resolved `list[ExperimentConfig]` only — execution block excluded by design.

**When to use:** After `expand_grid()` returns `valid: list[ExperimentConfig]`. Hash the ordered list, not the raw YAML dict (normalised form).

**Example:**

```python
# study/grid.py

import hashlib
import json

def compute_study_design_hash(experiments: list[ExperimentConfig]) -> str:
    """SHA-256[:16] of the resolved experiment list.

    Execution block (n_cycles, cycle_order, gaps) excluded — same design
    with different rigor settings gets the same hash. This enables 'topping
    up' a study without a new identity.

    Identical pattern to compute_measurement_config_hash() in domain/experiment.py.
    """
    canonical = json.dumps(
        [exp.model_dump() for exp in experiments],
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
```

### Pattern 3: Cycle Ordering

**What:** Apply `n_cycles` and `cycle_order` to the base experiment list to produce the final ordered execution sequence.

**When to use:** After `expand_grid()`, before returning `StudyConfig.experiments`.

```python
# study/grid.py

import random
from enum import StrEnum

class CycleOrder(StrEnum):
    SEQUENTIAL = "sequential"
    INTERLEAVED = "interleaved"
    SHUFFLED = "shuffled"


def apply_cycles(
    experiments: list[ExperimentConfig],
    n_cycles: int,
    cycle_order: CycleOrder,
    study_design_hash: str,
    shuffle_seed: int | None = None,
) -> list[ExperimentConfig]:
    """Return the ordered execution sequence for n_cycles repetitions.

    sequential:   A,A,A → B,B,B → C,C,C  (grouped by config)
    interleaved:  A,B,C → A,B,C → A,B,C  (round-robin — default)
    shuffled:     random order each cycle, seeded from study_design_hash
    """
    if cycle_order == CycleOrder.SEQUENTIAL:
        return [exp for exp in experiments for _ in range(n_cycles)]

    if cycle_order == CycleOrder.INTERLEAVED:
        return experiments * n_cycles

    # shuffled: reproducible from study_design_hash unless overridden
    seed = shuffle_seed if shuffle_seed is not None else int(study_design_hash, 16)
    rng = random.Random(seed)
    result = []
    for _ in range(n_cycles):
        cycle = list(experiments)
        rng.shuffle(cycle)
        result.extend(cycle)
    return result
```

### Pattern 4: base: Inheritance (Pre-processing Step)

**What:** Load an experiment YAML as the base config dict, then merge study-level fields on top.

**When to use:** In `expand_grid()` before any sweep expansion.

```python
def _load_base(base_path_str: str | None, study_yaml_path: Path | None) -> dict:
    """Load base experiment config as a raw dict.

    Path resolved relative to the study.yaml directory (matches Hydra, Docker Compose
    conventions). Hard error if file doesn't exist. One level only — base: inside
    a base file is ignored.
    """
    if base_path_str is None:
        return {}

    base_path = Path(base_path_str)
    if not base_path.is_absolute() and study_yaml_path is not None:
        base_path = study_yaml_path.parent / base_path

    if not base_path.exists():
        raise ConfigError(
            f"base: file not found: {base_path}\n"
            f"(resolved relative to study.yaml directory: {study_yaml_path.parent if study_yaml_path else 'cwd'})"
        )

    raw = yaml.safe_load(base_path.read_text())
    if not isinstance(raw, dict):
        raise ConfigError(f"base: file must be a YAML mapping: {base_path}")

    # Strip study-only keys (sweep:, experiments:, execution:, base:) — these
    # are not valid in experiment.yaml. Presence is not an error (researcher
    # may accidentally point at a study file) but they are silently dropped.
    # This is how we "detect" experiment vs study YAML: drop study-only keys.
    study_only_keys = {"sweep", "experiments", "execution", "base", "name"}
    return {k: v for k, v in raw.items() if k not in study_only_keys}
```

### Pattern 5: Extended StudyConfig and ExecutionConfig (Pydantic models)

**What:** Expand the M1 stub `StudyConfig` to include `execution:` block. Add `ExecutionConfig` as a new model.

**Location:** `config/models.py` — alongside `ExperimentConfig`.

```python
# config/models.py (additions)

from typing import Literal

class ExecutionConfig(BaseModel):
    """Study execution orchestration settings.

    These fields are EXCLUDED from study_design_hash — they control HOW
    the study runs, not WHAT is measured. Same design + more cycles = same hash.

    Stored as measurement_protocol in StudyResult (RES-13).
    """
    model_config = {"extra": "forbid"}

    n_cycles: int = Field(
        default=1,
        ge=1,
        description="Number of times to repeat the full experiment set. CLI default: 3."
    )
    cycle_order: Literal["sequential", "interleaved", "shuffled"] = Field(
        default="sequential",
        description="Order of experiments across cycles: sequential (A,A,B,B), "
                    "interleaved (A,B,A,B — default for thermal fairness), shuffled."
    )
    config_gap_seconds: float | None = Field(
        default=None,
        ge=0.0,
        description="Thermal gap between experiments (None = use user config machine default)."
    )
    cycle_gap_seconds: float | None = Field(
        default=None,
        ge=0.0,
        description="Thermal gap between full cycles (None = use user config machine default)."
    )
    shuffle_seed: int | None = Field(
        default=None,
        description="Explicit shuffle seed for cycle_order=shuffled. "
                    "None = derived from study_design_hash (reproducible, zero config)."
    )


class StudyConfig(BaseModel):
    """Thin resolved container for a study (list of experiments + execution config).

    Full M2 implementation. The resolved experiment list is the OUTPUT of expand_grid(),
    not an input from YAML directly. Users author sweep:, experiments:, and execution:
    blocks in YAML; the loader resolves them into this model.
    """
    model_config = {"extra": "forbid"}

    experiments: list[ExperimentConfig] = Field(
        ..., min_length=1, description="Resolved list of experiments (ordered for execution)"
    )
    name: str | None = Field(
        default=None, description="Study name (used in output directory naming)"
    )
    execution: ExecutionConfig = Field(
        default_factory=ExecutionConfig,
        description="Execution orchestration settings (excluded from study_design_hash)"
    )
    study_design_hash: str | None = Field(
        default=None,
        description="SHA-256[:16] of resolved experiments only (execution excluded). "
                    "Set by load_study_config() after grid expansion."
    )
    skipped_configs: list[dict] = Field(
        default_factory=list,
        description="Invalid combinations skipped during grid expansion (for post-hoc review)."
    )
```

### Pattern 6: load_study_config() — the full resolution entry point

**What:** Public function in `config/loader.py` that takes a file path and returns a resolved `StudyConfig`.

```python
# config/loader.py (addition)

def load_study_config(
    path: Path | str,
    cli_overrides: dict[str, Any] | None = None,
) -> StudyConfig:
    """Load, expand, and validate a study YAML file.

    Resolution order:
      1. base: experiment.yaml (optional)
      2. Inline study fields
      3. sweep: → Cartesian grid expansion (pre-Pydantic)
      4. experiments: → appended explicitly
      5. Pydantic validation of each ExperimentConfig
      6. study_design_hash computed over valid experiments
      7. apply_cycles() for cycle ordering

    Raises:
        ConfigError: File not found, parse error, base file missing, ALL configs invalid.
        ValidationError: Pydantic field-level errors pass through (shouldn't happen here
            since we catch them during grid expansion, but structural errors surface).
    """
    path = Path(path)
    raw = _load_file(path)      # reuse existing _load_file

    # Apply CLI overrides on execution block (--cycles, --order, --no-gaps are Phase 12)
    if cli_overrides:
        raw = deep_merge(raw, cli_overrides)

    # Strip version key (same as experiment loader)
    raw.pop("version", None)

    # Extract study-level metadata
    name = raw.get("name")

    # Parse execution block (Pydantic validates it)
    execution = ExecutionConfig(**(raw.get("execution") or {}))

    # Expand sweep → list[ExperimentConfig], collect skipped
    valid_experiments, skipped = expand_grid(raw, study_yaml_path=path)

    # Guard: all invalid → hard error
    total = len(valid_experiments) + len(skipped)
    if total == 0:
        raise ConfigError("Study produced no experiments (empty sweep and no experiments: list).")
    if not valid_experiments:
        raise ConfigError(
            f"All {total} generated configs are invalid. "
            "Nothing to run. Check sweep dimensions against backend constraints.\n"
            + "\n".join(f"  {s['short_label']}: {s['reason']}" for s in skipped[:5])
        )

    # Compute study_design_hash before applying cycles
    study_hash = compute_study_design_hash(valid_experiments)

    # Apply cycle ordering to produce execution sequence
    ordered = apply_cycles(
        valid_experiments,
        n_cycles=execution.n_cycles,
        cycle_order=CycleOrder(execution.cycle_order),
        study_design_hash=study_hash,
        shuffle_seed=execution.shuffle_seed,
    )

    return StudyConfig(
        experiments=ordered,
        name=name,
        execution=execution,
        study_design_hash=study_hash,
        skipped_configs=[s.to_dict() for s in skipped],
    )
```

### Pattern 7: Pre-flight Count Display

**What:** Display study summary before proceeding. Called from CLI after `load_study_config()`.

**Scope note:** `grid.py` returns data. Display logic belongs in `study/preflight.py` or the CLI layer. Phase 9 delivers the display function — the CLI wiring happens in Phase 12.

```python
# study/grid.py (or study/preflight.py)

def format_preflight_summary(
    study_config: StudyConfig,
    skipped: list[SkippedConfig],
) -> str:
    """Return pre-flight display string.

    Format:
        Study [abc123de]: 4 configs x 3 cycles = 12 runs
        Order: interleaved
        Skipping 3/15 (reason per skip):
          ✗ pytorch, fp32, batch=32: [Pydantic message]
    """
    n_configs = len(study_config.experiments) // study_config.execution.n_cycles
    n_cycles = study_config.execution.n_cycles
    n_runs = len(study_config.experiments)
    hash_display = study_config.study_design_hash or "unknown"[:8]

    lines = [
        f"Study [{hash_display}]: {n_configs} configs x {n_cycles} cycles = {n_runs} runs",
        f"Order: {study_config.execution.cycle_order}",
    ]

    if skipped:
        total_generated = n_configs + len(skipped)
        lines.append(f"Skipping {len(skipped)}/{total_generated}:")
        for s in skipped:
            lines.append(f"  ✗ {s.short_label}: {s.reason_short}")

        skip_rate = len(skipped) / total_generated
        if skip_rate > 0.5:
            lines.append(
                f"  WARNING: {skip_rate:.0%} of sweep configs are invalid — check your sweep dimensions."
            )

    return "\n".join(lines)
```

### Anti-Patterns to Avoid

- **Interleaving expansion with validation:** Never try to `ExperimentConfig(**dict)` inside `itertools.product()`. Collect all raw dicts first, validate in a separate loop. Reason: error collection and skip-rate logic require knowing the total.
- **Storing execution block in hash:** The hash must exclude `execution:` — run with n_cycles=1 vs n_cycles=3 must produce the same hash (enables "topping up").
- **Using `set_start_method('spawn')` globally:** Not a Phase 9 concern, but note it for Phase 11 — always `get_context('spawn')` scoped.
- **Resolving backend-scoped keys too early:** Parse `pytorch.batch_size` as a dotted sweep key, not as a nested Pydantic path. The expansion happens at raw dict level, not post-Pydantic.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Cartesian grid expansion | Nested for-loops | `itertools.product` | Correct, readable, universal ML tooling pattern |
| Canonical JSON for hash | Custom serialiser | `json.dumps(model_dump(), sort_keys=True)` | Identical to existing `compute_measurement_config_hash()` |
| YAML loading with anchors | Custom parser | `yaml.safe_load` | Already in codebase; native anchor support |
| Fuzzy field suggestion | Levenshtein from scratch | `_did_you_mean()` from `config/loader.py` | Already implemented; reuse |
| Seeded shuffle | Custom RNG | `random.Random(seed).shuffle()` | Standard library; `Random` object is stateful but not global |

**Key insight:** Everything in Phase 9 is a composition of standard Python stdlib primitives and existing project patterns. The design docs have fully specified the grammar; the code is mostly mechanical transcription.

---

## Common Pitfalls

### Pitfall 1: Sweep Combinatorial Explosion (CP-5)

**What goes wrong:** A researcher writes `sweep: batch_size: [1, 2, 4, 8, 16, 32]` and `precision: [fp16, bf16, fp32]` and `n: [50, 100, 200]` — producing 54 experiments x 3 cycles = 162 runs x 60s gaps = 3+ hours unattended. Tool silently starts.

**Why it happens:** No pre-flight count display; no cap.

**How to avoid:** Always display count summary before proceeding (Phase 9 delivers this). The confirmation prompt threshold (Claude's discretion) should default to 50 experiments — above this, display a warning and optionally pause. The CONTEXT.md leaves this as Claude's discretion; recommend a WARNING at >50 experiments with auto-proceed (no prompt), and only a hard cap at >500 that requires `--yes` flag.

**Warning signs:** Pre-flight summary shows a large number; researcher does not respond to it.

### Pitfall 2: Backend-Scoped Key Routing (sweep grammar edge case)

**What goes wrong:** `pytorch.batch_size: [1, 8]` in sweep is incorrectly routed into the top-level config dict as `{"pytorch.batch_size": [1, 8]}` instead of being recognised as a sweep dimension for the `pytorch` backend section.

**Why it happens:** The existing `_unflatten()` in `loader.py` handles dotted keys for CLI overrides (producing `{"pytorch": {"batch_size": ...}}`). The sweep grammar is different: dotted keys in `sweep:` are backend-scoped *dimensions*, not direct field assignments.

**How to avoid:** The sweep parser must handle dotted keys differently from `_unflatten()`. In `_expand_sweep()`, split on first dot to extract `(backend, param)`. Then for each combo, place the param into the backend sub-dict of the config, not at the top level.

**Warning signs:** `ExperimentConfig(**raw)` raises `ValidationError` for `pytorch.batch_size` as an unknown top-level field.

### Pitfall 3: Hash Instability from Non-Deterministic Dict Ordering

**What goes wrong:** `study_design_hash` changes between runs even with identical configs because `model_dump()` returns dicts with non-deterministic ordering in some edge cases.

**Why it happens:** Python dicts are insertion-ordered from 3.7+, but nested dicts from `model_dump()` rely on field declaration order. Adding/removing optional fields with `None` values can shift things.

**How to avoid:** Always `json.dumps(..., sort_keys=True)`. Existing `compute_measurement_config_hash()` already uses this pattern — copy exactly.

**Warning signs:** Same YAML produces different hash on second run.

### Pitfall 4: ExperimentConfig Validation with Backend Section Mismatch

**What goes wrong:** A sweep produces `{"backend": "pytorch", "vllm": {"max_num_seqs": 256}}` — a config with a vllm section but pytorch backend. `ExperimentConfig` validator rejects this with a confusing error.

**Why it happens:** When combining universal sweep dimensions with backend-scoped dimensions, the config dict can accumulate backend sections from other backends.

**How to avoid:** In `_expand_sweep()`, when building a config for backend X, only include the X backend section. Omit sections for other backends. The grid expansion is backend-separated by design.

**Warning signs:** `ValidationError: vllm: config section provided but backend=pytorch` during grid expansion.

### Pitfall 5: base: File Points at a study.yaml (Not experiment.yaml)

**What goes wrong:** A researcher writes `base: study.yaml` instead of `base: experiment.yaml`. The base file has `sweep:` and `execution:` blocks. These get merged into the fixed dict and passed to the sweep resolver, causing unexpected behavior or errors.

**Why it happens:** No validation distinguishes experiment YAML from study YAML by content.

**How to avoid:** In `_load_base()`, strip study-only keys (`sweep`, `experiments`, `execution`, `base`, `name`) from the loaded dict before using it as the base. This silently recovers from the mistake — the study-only keys in the base file are ignored, and the remaining experiment-config fields are used. Optionally emit a warning: "base: file contains study-only keys (sweep, execution) — these are ignored."

**Warning signs:** Unexpected experiments appearing from sweep defined in base file.

### Pitfall 6: Empty Sweep Generates Zero Experiments

**What goes wrong:** A study YAML with only `execution:` block and no `sweep:` and no `experiments:` list raises `min_length=1` error on `StudyConfig.experiments` with a confusing message.

**Why it happens:** `expand_grid()` returns an empty list when sweep is empty and no experiments are listed.

**How to avoid:** Guard explicitly in `load_study_config()` before constructing `StudyConfig`: if `total == 0`, raise `ConfigError("Study produced no experiments...")`.

---

## Code Examples

Verified patterns from existing codebase and product designs:

### Hash Pattern (from domain/experiment.py — replicate exactly)

```python
# Source: src/llenergymeasure/domain/experiment.py::compute_measurement_config_hash
import hashlib
import json

def compute_study_design_hash(experiments: list[ExperimentConfig]) -> str:
    """SHA-256[:16] of resolved experiments (execution block excluded)."""
    canonical = json.dumps(
        [exp.model_dump() for exp in experiments],
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
```

### Existing _unflatten pattern (DO NOT reuse for sweep — create a separate function)

```python
# Source: src/llenergymeasure/config/loader.py::_unflatten
# This handles CLI overrides: "pytorch.batch_size=8" → {"pytorch": {"batch_size": 8}}
# The sweep grammar uses dotted keys differently (as dimensions, not direct assignments)
# Create _parse_sweep_dims() separately to avoid conflating the two semantics.
```

### Existing StudyConfig stub (M1 — to be replaced in Phase 9)

```python
# Source: src/llenergymeasure/config/models.py
class StudyConfig(BaseModel):
    """M1 stub — Phase 9 replaces this with full implementation."""
    model_config = {"extra": "forbid"}
    experiments: list[ExperimentConfig] = Field(..., min_length=1)
    name: str | None = Field(default=None)
    # Phase 9 adds: execution, study_design_hash, skipped_configs
```

### Seeded shuffle for reproducible cycle ordering

```python
# Source: Python stdlib — standard pattern
import random

seed = int(study_design_hash, 16) & 0xFFFFFFFF  # convert hex to int, cap to 32-bit
rng = random.Random(seed)
cycle = list(experiments)
rng.shuffle(cycle)
# Same study_design_hash always produces same shuffle order
```

### pre-flight summary format (aligned with git/Docker style)

```
Study [a3f9bc12]: 4 configs x 3 cycles = 12 runs
Order: interleaved
Skipping 2/6: (2 invalid combinations)
  ✗ pytorch, fp32: pytorch does not support precision='fp32'
  ✗ vllm, bf16, batch=64: vllm: config section provided but backend=pytorch
```

---

## State of the Art

| Old Approach (v1.x) | Current Approach (v2.0) | When Changed | Impact |
|---------------------|------------------------|--------------|--------|
| `CampaignExecutionConfig` with `cycles`, `structure`, `config_gap_seconds` | `ExecutionConfig` with `n_cycles`, `cycle_order`, nested under `execution:` in YAML | M2 design phase 2026-02-25 | Cleaner YAML; separated from experiment identity |
| `StudyConfig` as M1 stub (`experiments` + `name` only) | Full `StudyConfig` with `execution`, `study_design_hash`, `skipped_configs` | Phase 9 | First usable study model |
| No sweep grammar | `sweep:` block with dotted notation backend-scoped dims | M2 design phase | Researchers can express multi-dimensional sweeps |
| No `study_design_hash` | SHA-256[:16] excluding execution block | Phase 9 | Study identity for reproducibility tracking and paper citation |

**Deprecated/outdated:**

- `CampaignExecutionConfig` (v1.x): replaced by `ExecutionConfig`. Do not import or reference.
- `campaign` terminology: all references updated to `study`. No aliases.

---

## Open Questions

1. **Large sweep confirmation threshold**
   - What we know: CONTEXT.md leaves the exact threshold as Claude's discretion
   - What's unclear: Whether to hard-stop above N and require `--yes`, or just warn
   - Recommendation: Warn at >50 experiments (auto-proceed); display WARNING at >200 experiments; no hard block in Phase 9 (hard block with `--yes` is Phase 12 CLI concern). Phase 9 delivers the count and `format_preflight_summary()` function; the CLI decides what to do with it.

2. **SkippedConfig model shape**
   - What we know: Must capture `raw_config` dict and Pydantic `ValidationError`; persisted in `StudyConfig.skipped_configs` for post-hoc review
   - What's unclear: Whether to store the full `ValidationError` json or a simplified string
   - Recommendation: Store `{"raw_config": ..., "reason": str(error), "errors": error.errors()}` — structured enough for post-hoc analysis, simple enough to serialise without a full Pydantic model for `SkippedConfig` itself.

3. **cycle_order default: "sequential" vs "interleaved"**
   - What we know: REQUIREMENTS.md says `cycle_order` Pydantic default=1 for `n_cycles`; CONTEXT.md shows default "sequential" in Pydantic; CLI effective default=3 for `n_cycles`; study-yaml.md shows "interleaved" described as "default" for thermal fairness
   - What's unclear: The study-yaml.md and REQUIREMENTS.md slightly conflict (study-yaml.md says interleaved is default for thermal fairness; REQUIREMENTS.md implementation note says "sequential | interleaved")
   - Recommendation: Use `sequential` as the Pydantic model default (conservative, matches REQUIREMENTS.md CFG-15 explicit note), and document `interleaved` as the recommended setting for multi-cycle studies. Phase 12 CLI can set effective default to `interleaved` via the `--order` flag default.

4. **`study/` package vs `config/` placement for grid.py**
   - What we know: Architecture design specifies `study/grid.py`; `ExecutionConfig` and expanded `StudyConfig` belong in `config/models.py`
   - Recommendation: Follow architecture design exactly. `study/grid.py` for pure expansion functions; `config/models.py` for `ExecutionConfig` and `StudyConfig`; `config/loader.py` for `load_study_config()`.

---

## Sources

### Primary (HIGH confidence)

- `.product/designs/study-yaml.md` — sweep grammar, execution block, base: resolution, cycle_order semantics (project source of truth)
- `.product/designs/experiment-config.md` — `ExperimentConfig` full schema, backend sections, SSOT pattern
- `.product/designs/architecture.md` — module placement: `study/grid.py`, `config/models.py`, `config/loader.py`
- `src/llenergymeasure/config/models.py` — existing M1 `StudyConfig` stub, `ExperimentConfig` full implementation
- `src/llenergymeasure/config/loader.py` — existing `_unflatten()`, `deep_merge()`, `load_experiment_config()` patterns
- `src/llenergymeasure/domain/experiment.py` — `compute_measurement_config_hash()` hash pattern to replicate
- `.planning/research/SUMMARY.md` — M2 architecture research: `itertools.product` pattern, no new deps, expand-first pattern
- Python stdlib docs: `itertools.product`, `hashlib.sha256`, `random.Random`, `json.dumps(sort_keys=True)`

### Secondary (MEDIUM confidence)

- Hydra BasicLauncher: expand full sweep before first job dispatched (expand-first pattern confirmed)
- lm-eval: task loading "expand first, iterate second" pattern
- W&B Sweeps `method: grid` + `parameters:` block: identical grammar semantics to our `sweep:` block

### Tertiary (LOW confidence)

- None — all architecture decisions are project-confirmed and backed by primary sources

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new deps; all stdlib + existing; verified against existing codebase
- Architecture: HIGH — module placement specified in architecture.md; patterns from existing code
- Pitfalls: HIGH — identified from design docs and existing codebase constraints; CP-5 from M2 research

**Research date:** 2026-02-27
**Valid until:** Indefinite — all sources are project-internal or Python stdlib (stable)
