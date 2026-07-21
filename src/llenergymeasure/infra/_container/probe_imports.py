"""Runtime-dependency import probe for the container entrypoint.

Runs INSIDE the dispatch container (invoked by ``container_entrypoint.sh``
against the bind-mounted package source at ``/llem-src``). It diffs the runtime
requirements list at ``/llem-requirements.txt`` against what actually imports in
the container's interpreter and prints the space-separated specs that are
missing, so the entrypoint can pip-install only those into the persistent deps
cache.

A dist-info presence check is not sufficient: metadata can be present while the
package fails to import (a compiled extension built for the wrong ABI, or an
otherwise broken install). So each present requirement is verified by importing
its resolved module(s), and flagged missing if the import raises. Absent metadata
short-circuits as missing with no import attempt.

Stdlib-only and self-contained: it is executed as a plain script file
(``python3 probe_imports.py /llem-requirements.txt``), NOT as a package module,
so it never imports the ``llenergymeasure`` package and stays runnable in a bare
engine image. It is a real module (not an inline heredoc) so lint/mypy cover it
and its unit tests import it directly.
"""

from __future__ import annotations

import functools
import importlib
import importlib.metadata
import sys
from pathlib import Path

# Distributions whose top-level import name differs from the distribution name.
# Keys are canonical (lowercase, dashes) distribution names; values are the
# module to import. Authoritative: consulted before top_level.txt so noisy or
# absent top-level metadata cannot mis-resolve these known cases.
IMPORT_NAME_OVERRIDES = {
    "nvidia-ml-py": "pynvml",
    "pyyaml": "yaml",
    "python-dotenv": "dotenv",
}


def bare_name(spec: str) -> str:
    """Strip version bounds / extras to the bare distribution name.

    The full spec is kept for the pip install; this bare form is the metadata
    lookup key.
    """
    name = spec
    for sep in (">=", "<=", "==", "!=", "~=", "<", ">", "["):
        if sep in name:
            name = name.split(sep)[0]
    return name.strip()


def canonical(name: str) -> str:
    """Return the canonical (lowercase, dashes) form of a distribution name."""
    return name.strip().lower().replace("_", "-")


@functools.cache
def reverse_packages() -> dict[str, list[str]]:
    """Invert ``packages_distributions()`` into dist name -> import name(s).

    ``importlib.metadata.packages_distributions()`` maps import name -> dist
    names; inverting it lets a distribution's top-level module(s) be recovered
    even when its ``top_level.txt`` is absent.
    """
    mapping: dict[str, list[str]] = {}
    try:
        items = importlib.metadata.packages_distributions().items()
    except Exception:
        return mapping
    for import_name, dists in items:
        for dist_name in dists:
            mapping.setdefault(canonical(dist_name), []).append(import_name)
    return mapping


def import_names(dist_name: str, dist: importlib.metadata.Distribution) -> list[str]:
    """Resolve the top-level import name(s) for an installed distribution."""
    key = canonical(dist_name)
    if key in IMPORT_NAME_OVERRIDES:
        return [IMPORT_NAME_OVERRIDES[key]]
    try:
        top_level = dist.read_text("top_level.txt")
    except Exception:
        top_level = None
    if top_level:
        names = [ln.strip() for ln in top_level.splitlines() if ln.strip()]
        if names:
            return names
    resolved = reverse_packages().get(key)
    if resolved:
        return resolved
    # Fall back to the normalised distribution name (dashes to underscores).
    return [dist_name.replace("-", "_")]


def find_missing(requirements_file: str | Path) -> list[str]:
    """Return the requirement specs whose module(s) are absent or fail to import."""
    missing: list[str] = []
    for line in Path(requirements_file).read_text().splitlines():
        spec = line.strip()
        if not spec or spec.startswith("#"):
            continue
        name = bare_name(spec)
        if not name:
            continue
        try:
            dist = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            # Absent metadata: definitely missing, no import attempt needed.
            missing.append(spec)
            continue
        try:
            for import_name in import_names(name, dist):
                importlib.import_module(import_name)
        except Exception:
            # Present metadata but the module does not import (wrong-ABI
            # extension, broken install): prime it so the container's pip
            # reinstalls an ABI-correct copy into the deps cache.
            missing.append(spec)
    return missing


def main(argv: list[str] | None = None) -> int:
    """Print the space-separated missing specs for the given requirements file.

    The requirements list is bind-mounted at ``/llem-requirements.txt`` inside
    the container; an optional argv override (defaulting to the mount path) lets
    the probe run against a synthetic requirements file without changing
    container behaviour.
    """
    args = list(sys.argv if argv is None else argv)
    requirements_file = args[1] if len(args) > 1 else "/llem-requirements.txt"
    print(" ".join(find_missing(requirements_file)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
