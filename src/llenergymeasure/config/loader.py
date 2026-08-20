"""YAML/JSON configuration loader for experiment and study configs.

Implements the loading contract:
- Collect ALL errors before raising (not one-at-a-time)
- ConfigError with file path + did-you-mean for unknown fields
- Native YAML anchor support via yaml.safe_load

This layer parses; it does not resolve. Layering settings from the tool-wide user
config, the environment and the call site belongs to
``llenergymeasure.study.loading.resolve_study``, the single entry point every
study passes through (#886).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from llenergymeasure.config._dict_utils import deep_merge
from llenergymeasure.config.grid import SkippedConfig, expand_grid
from llenergymeasure.config.models import (
    RETIRED_HARNESS_KEY_MSG,
    SERVING_MODE_REQUIRED_MSG,
    ExecutionConfig,
    ExperimentConfig,
    OutputConfig,
)
from llenergymeasure.utils.exceptions import ConfigError

__all__ = [
    "LoadedStudyRaw",
    "deep_merge",
    "load_experiment_config",
    "load_study_config",
]


# =============================================================================
# Public API
# =============================================================================


def load_experiment_config(path: Path | str) -> ExperimentConfig:
    """Load and validate a single-experiment configuration file.

    A pure parse of one file: experiment semantics live in the YAML, not in
    layered overrides.

    Args:
        path: Path to YAML or JSON config file.

    Returns:
        Validated ExperimentConfig.

    Raises:
        ConfigError: File not found, parse error, unknown fields, or structural validation failure.
            Includes all errors collected at once (not one-at-a-time).
        ValidationError: Pydantic field-level validation errors pass through unchanged.
            (Bad values like n=-1 are Pydantic's domain; unknown keys become ConfigError.)
    """
    merged = _load_file(path)  # raises ConfigError on missing/parse error

    # Strip optional version field - not an ExperimentConfig field
    merged.pop("version", None)

    # Retired top-level ``harness:`` key: fail with the migration message naming
    # the new location, before the generic unknown-field did-you-mean fires.
    if "harness" in merged:
        raise ConfigError(f"{RETIRED_HARNESS_KEY_MSG} (in {path})")

    # Required serving_mode (no default): fail with the friendly migration message
    # naming both modes, with file context, rather than Pydantic's bare
    # "Field required".
    if "serving_mode" not in merged:
        raise ConfigError(f"{SERVING_MODE_REQUIRED_MSG} (in {path})")

    # Collect unknown field errors before handing to Pydantic
    known_fields = set(ExperimentConfig.model_fields.keys())
    unknown = set(merged.keys()) - known_fields
    if unknown:
        errors = []
        for key in sorted(unknown):
            suggestion = _did_you_mean(key, known_fields)
            msg = f"Unknown field '{key}'"
            if suggestion:
                msg += f" - did you mean '{suggestion}'?"
            errors.append(f"{msg} (in {path})")
        raise ConfigError("\n".join(errors))

    # Construct ExperimentConfig - let ValidationError pass through unchanged
    try:
        return ExperimentConfig(**merged)
    except ValidationError:
        raise  # Pydantic field-level errors are not our domain to wrap
    except Exception as e:
        raise ConfigError(f"Config construction failed (in {path}): {e}") from e


@dataclass(frozen=True)
class LoadedStudyRaw:
    """Parsed + sweep-expanded study material, before dedup/hash/cycle resolution.

    Output of :func:`load_study_config`. Carries everything the single
    study-resolution entry point (``llenergymeasure.study.loading.resolve_study``)
    needs to build a resolved :class:`~llenergymeasure.config.models.StudyConfig`, without
    the config layer reaching upward into ``study`` for the library-resolution /
    dedup mechanism.

    Attributes:
        valid_experiments: Sweep-expanded, Pydantic-validated declared configs
            (pre-dedup, pre-cycle). At least one entry (guards raise otherwise).
        skipped: Grid points that failed validation during expansion.
        study_name: Optional study name (output directory naming).
        output: Parsed study-level output configuration.
        execution: Parsed execution block (cycles, ordering, dedup toggle).
        runners: Per-engine runner config, or None.
        images: Per-engine Docker image overrides, or None.
    """

    valid_experiments: list[ExperimentConfig]
    skipped: list[SkippedConfig]
    study_name: str | None
    output: OutputConfig
    execution: ExecutionConfig
    runners: dict[str, str] | None
    images: dict[str, str] | None


def load_study_config(
    path: Path | str,
    cli_overrides: dict[str, Any] | None = None,
) -> LoadedStudyRaw:
    """Load, expand, and validate a study YAML file (pure config-layer parse).

    Resolution order:
      1. Load YAML file
      2. Apply CLI overrides on execution block
      3. Parse output + execution blocks (Pydantic validates them)
      4. expand_grid() - Cartesian product + Pydantic validation of each ExperimentConfig
      5. Guard: empty or all-invalid -> ConfigError

    The contract: sweep resolution at YAML parse time, before Pydantic sees the
    individual ExperimentConfig objects.

    Dedup, study_design_hash, cycle ordering, and equivalence-group serialisation
    are deliberately NOT done here - they require the study-layer
    library-resolution mechanism, and the config layer must not import upward.
    Those steps live in ``llenergymeasure.study.loading.resolve_study``, which
    consumes the :class:`LoadedStudyRaw` returned here and produces the resolved
    :class:`~llenergymeasure.config.models.StudyConfig`. Use
    ``llenergymeasure.api.load_study`` to run both steps in one call.

    Args:
        path: Path to study YAML file.
        cli_overrides: Optional dict of overrides deep-merged onto the parsed
            file, at the highest priority (e.g. {"study_execution":
            {"n_cycles": 5}}).

    Returns:
        :class:`LoadedStudyRaw` - sweep-expanded experiments plus the parsed
        metadata the study-layer resolution step needs.

    Raises:
        ConfigError: File not found, parse error, base file missing, ALL configs invalid,
            or empty study (no sweep and no experiments).
        ValidationError: Pydantic structural errors on ExecutionConfig pass through.
    """
    path = Path(path)
    raw = _load_file(path)  # reuse existing _load_file - raises ConfigError on missing/parse error

    # Apply CLI overrides (highest priority, deep-merged onto the parsed file)
    if cli_overrides:
        raw = deep_merge(raw, cli_overrides)

    # Strip version key (same as experiment loader)
    raw.pop("version", None)

    # Extract study-level metadata
    name = raw.get("study_name")
    # runners: per-engine runner config (e.g. {"transformers": "process", "vllm": "container"})
    # None if not specified in YAML - caller uses user config / auto-detection.
    runners: dict[str, str] | None = raw.get("runners") or None
    # images: per-engine Docker image overrides (orthogonal to runners)
    # e.g. {"vllm": "ghcr.io/org/vllm:v1.0"}. None = smart default resolution.
    images: dict[str, str] | None = raw.get("images") or None

    # Parse output block - Pydantic validates it
    output = OutputConfig(**(raw.get("output") or {}))

    # Parse execution block - Pydantic validates it
    execution = ExecutionConfig(**(raw.get("study_execution") or {}))

    # Expand sweep → list[ExperimentConfig], collect skipped
    # Sweep resolution at YAML parse time, before Pydantic
    valid_experiments, skipped = expand_grid(raw, study_yaml_path=path)

    # Guard: empty study - expand_grid already raises if all_raw_configs is empty,
    # but we also need to handle the degenerate case where expand_grid itself
    # returns no valid experiments and raises. If we reach here, valid_experiments
    # has at least one entry (expand_grid raises ConfigError if all invalid).
    # The "empty study" case (no model, no sweep, no experiments) is already
    # caught inside expand_grid(). We add an extra guard here for clarity:
    total = len(valid_experiments) + len(skipped)
    if total == 0:
        raise ConfigError("Study produced no experiments (empty sweep and no experiments: list).")
    if not valid_experiments:
        skip_details = "\n".join(f"  {s.short_label}: {s.reason}" for s in skipped[:5])
        raise ConfigError(
            f"All {total} generated configs are invalid. "
            "Nothing to run. Check sweep dimensions against engine constraints.\n" + skip_details
        )

    return LoadedStudyRaw(
        valid_experiments=valid_experiments,
        skipped=skipped,
        study_name=name,
        output=output,
        execution=execution,
        runners=runners,
        images=images,
    )


# =============================================================================
# Private helpers
# =============================================================================


def _load_file(path: Path | str) -> dict[str, Any]:
    """Load YAML or JSON config file into a dict.

    Args:
        path: Path to config file.

    Returns:
        Parsed config dictionary.

    Raises:
        ConfigError: If file not found, unsupported format, parse error, or not a mapping.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        content = path.read_text()
        if path.suffix in (".yaml", ".yml"):
            result = yaml.safe_load(content)  # native YAML anchors (&/*) handled automatically
        elif path.suffix == ".json":
            result = json.loads(content)
        else:
            raise ConfigError(f"Unsupported config format '{path.suffix}': use .yaml or .json")
        if not isinstance(result, dict):
            raise ConfigError(f"Config must be a mapping (got {type(result).__name__}): {path}")
        return result
    except (yaml.YAMLError, json.JSONDecodeError) as e:
        raise ConfigError(f"Parse error in {path}: {e}") from e


def _did_you_mean(key: str, candidates: set[str], max_distance: int = 3) -> str | None:
    """Return the closest candidate if within max_distance edits, else None.

    Args:
        key: Unknown key to find a suggestion for.
        candidates: Set of valid field names.
        max_distance: Maximum Levenshtein distance to suggest (default 3).

    Returns:
        Closest candidate string, or None if nothing is close enough.
    """
    best: str | None = None
    best_dist = max_distance + 1
    for candidate in candidates:
        dist = _levenshtein(key, candidate)
        if dist < best_dist:
            best_dist = dist
            best = candidate
    return best if best_dist <= max_distance else None


def _levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        return _levenshtein(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]
