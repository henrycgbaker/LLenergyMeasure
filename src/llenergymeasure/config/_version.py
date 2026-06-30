"""Shared semver-major parsing for the config schema/rules loaders.

Both :mod:`llenergymeasure.config.schema_loader` and
:mod:`llenergymeasure.config.engine_rules.loader` need to read the major
version out of a ``"1.0.0"``-style envelope string. The mechanical parse is
identical; only the typed error each loader raises on failure differs (callers
catch by the module-local exception type, so those stay separate). This module
owns the parse; each loader wraps it to raise its own exception.
"""

from __future__ import annotations


def parse_major(version: str) -> int | None:
    """Return the integer major component of a semver-ish string, or ``None``.

    ``"1.0.0"`` -> ``1``; ``"1.7.3"`` -> ``1``. Returns ``None`` when the
    leading component is not an integer (``"dev"``, ``"not-semver"``) or the
    input is not a string - the caller turns that into its own typed
    unsupported-version error with a context-appropriate message.
    """
    try:
        return int(version.split(".", 1)[0])
    except (ValueError, AttributeError):
        return None
