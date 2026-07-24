"""Canonical serialisation and SHA-256 hashing primitives.

Pure, dependency-free primitives shared by both the resolved-config (library-resolution
mechanism output) and observed-config (library-observed) hashing pipelines.  Neither
hash requires imports from upper layers, so these live at Layer 0 (domain).

``build_resolved_view`` - the one function that needs ``ExperimentConfig`` -
stays in :mod:`llenergymeasure.study.hashing` where it belongs (Layer 4).

Normalisation is strict (over-normalising would hide library-enforced semantics,
e.g. ``None`` vs missing in vLLM) with one deliberate unification: integral
numerics fold onto their ``int`` form, superseding the int-vs-float split
sweep-dedup.md §9.Q3 previously locked (the design doc records the reversal).
The declared-config hash family folds the opposite way - int -> float via pydantic
``mode="json"`` (PR #822) - which is harmless because the declared and
resolved/observed hash namespaces never intersect.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

_FLOAT_SIG_DIGITS = 12
"""Float rounding precision (significant digits) for hash stability.

Upstream float arithmetic can produce bit-level jitter in the last 1-2
digits that does not reflect an actual configuration difference.  Rounding
at 12 significant digits removes that jitter without compressing any two
values a researcher would write differently.
"""


# ---------------------------------------------------------------------------
# Canonical serialisation (shared by resolved-config and observed-config hashes)
# ---------------------------------------------------------------------------


def _normalise(value: Any) -> Any:
    """Normalise a value for deterministic JSON serialisation.

    Canonicalisation rules; the integral-fold below supersedes the int-vs-float
    split sweep-dedup.md §9.Q3 previously locked:

    - ``NaN`` -> string ``"NaN"`` (NaN != NaN breaks dict hashing otherwise)
    - ``+/-Infinity`` -> string literal (stable across platforms)
    - integral numerics unify on their ``int`` form: a float that is exactly
      integral (``0.0``, ``1.0``, ``-1.0``) folds to ``int``, so a value written
      as an int and the same value carried as a float hash identically. Folding
      toward int (rather than int -> float) keeps genuine integer identity fields
      - seeds, token counts - bit-exact, so no large distinct ints collide via a
      lossy float round-trip.
    - non-integral float -> rounded to 12 significant digits (stable across
      minor arithmetic jitter)
    - tuple -> list (incidental immutability choice, not semantic)
    - bool is preserved as bool (not folded into int even though
      ``True == 1`` in Python)
    - dict -> dict with recursively normalised values (key sorting happens
      at ``json.dumps(sort_keys=True)`` time)
    - None and missing keys stay distinguishable (by never inserting a
      sentinel for missing)
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            # Preserve infinity as a string literal for hash stability
            return "Infinity" if value > 0 else "-Infinity"
        if value.is_integer():
            # Integral float (incl. 0.0) folds to int so it hashes identically
            # to the int form; also sidesteps log10(0) below.
            return int(value)
        # Round to sig-figs rather than decimal places.
        mag = math.floor(math.log10(abs(value)))
        factor = 10 ** (_FLOAT_SIG_DIGITS - 1 - mag)
        rounded = round(value * factor) / factor
        # Rounding can land on an exact integer (float jitter around N.0); fold
        # that onto int too so it matches the int / integral-float forms.
        return int(rounded) if rounded.is_integer() else rounded
    if isinstance(value, (set, frozenset)):
        # Sort for determinism; normalise elements recursively.
        return [_normalise(v) for v in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [_normalise(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalise(v) for k, v in value.items()}
    return value


def canonical_serialise(obj: Any) -> bytes:
    """Serialise ``obj`` to canonical-JSON bytes, ready for hashing.

    Applies :func:`_normalise` then ``json.dumps(sort_keys=True)``.  Uses a
    separators tuple with no whitespace for compactness and stability
    across Python versions.
    """
    normalised = _normalise(obj)
    return json.dumps(
        normalised,
        sort_keys=True,
        separators=(",", ":"),
        default=str,  # Pydantic enums and dates
        ensure_ascii=False,
    ).encode("utf-8")


# ---------------------------------------------------------------------------
# Hashed-field schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfigHashView:
    """Fixed-schema view of the fields hashed into resolved or observed config hash.

    The hashed-field set is:

    - ``task`` - model, prompt source, batch shape
    - ``serving_mode`` - offline-batch vs online-serving discriminator. A
      conditioning identity axis, so an offline and a server run of the same
      task never collapse under dedup.
    - ``observed_engine_params`` - engine state (library-resolution mechanism output for
      resolved-config-hash, live library observation for observed-config-hash)
    - ``observed_sampling_params`` - sampling state (same sources as above)
    - ``passthrough_kwargs`` - user-attached overrides
    - ``llem_execution`` - the active engine's llem-owned execution knobs
      (batch_size, torch_compile, allow_tf32, autocast). These drive execution
      but have no engine-native API, so they must join the config identity or an
      llem-execution-knob sweep collapses to one run under dedup.
    - ``measurement`` - measurement methodology (warmup, baseline, energy sampler,
      windowing). Sweeping methodology creates distinct runs, so these join the
      identity too; dedup then collapses only true duplicates.

    Excluded: ``ExecutionConfig`` (runner/parallelism), ``experiment_id``.
    """

    engine: str
    task: dict[str, Any]
    serving_mode: str = "offline"
    observed_engine_params: dict[str, Any] = field(default_factory=dict)
    observed_sampling_params: dict[str, Any] = field(default_factory=dict)
    passthrough_kwargs: dict[str, Any] = field(default_factory=dict)
    llem_execution: dict[str, Any] = field(default_factory=dict)
    measurement: dict[str, Any] = field(default_factory=dict)


def hash_config(view: ConfigHashView) -> str:
    """Return SHA-256 hex digest of ``view`` via :func:`canonical_serialise`.

    Both resolved-config-hash and observed-config-hash route through this function;
    they differ only in how the :class:`ConfigHashView` is populated.
    """
    payload = asdict(view)
    return hashlib.sha256(canonical_serialise(payload)).hexdigest()


# ---------------------------------------------------------------------------
# Observed-config view construction - from library-observed effective params
# ---------------------------------------------------------------------------


def build_observed_view(
    *,
    engine: str,
    task: dict[str, Any],
    observed_engine_params: dict[str, Any],
    observed_sampling_params: dict[str, Any],
    serving_mode: str = "offline",
    passthrough_kwargs: dict[str, Any] | None = None,
    llem_execution: dict[str, Any] | None = None,
    measurement: dict[str, Any] | None = None,
) -> ConfigHashView:
    """Assemble an observed-config view from per-engine ``extract_observed_params`` output.

    Callers live in the harness/sidecar path - they read ``task`` from the
    same config that ran and pair it with the native-object dumps the engine
    returned. ``serving_mode``, ``llem_execution`` and ``measurement`` come from
    the same config (they are mode/execution/methodology dials, not
    library-observable) so that the observed hash covers the same identity
    dimensions as the resolved hash; keeping them aligned stops the
    observed-collision analysis from flagging a pure execution/measurement sweep
    as a false library-resolution gap.
    """
    return ConfigHashView(
        engine=engine,
        task=task,
        serving_mode=serving_mode,
        observed_engine_params=dict(observed_engine_params),
        observed_sampling_params=dict(observed_sampling_params),
        passthrough_kwargs=dict(passthrough_kwargs or {}),
        llem_execution=dict(llem_execution or {}),
        measurement=dict(measurement or {}),
    )
