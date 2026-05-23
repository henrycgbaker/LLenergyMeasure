#!/usr/bin/env python3
"""Read one dotted key path from a TOML file and print its value.

Shell-friendly stdlib-only replacement for ``yq '.<key>' <file>`` after the
per-engine SSOT migration from YAML to TOML. Used by GitHub Actions cells
to extract ``library.current_version`` and ``source_tarball.url_template``
out of ``engine_versions/<engine>/current.toml``.

Usage::

    python3 scripts/ci/read_toml_value.py <path> <dotted.key.path>

Examples::

    python3 scripts/ci/read_toml_value.py engine_versions/vllm/current.toml \\
        library.current_version
    python3 scripts/ci/read_toml_value.py engine_versions/tensorrt/current.toml \\
        source_tarball.url_template

Exit codes:
    0 = key resolved, value printed to stdout (no trailing newline beyond ``print``)
    1 = file missing, malformed TOML, or key path does not resolve
"""

from __future__ import annotations

import sys
from pathlib import Path

import tomllib


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(
            "Usage: read_toml_value.py <path> <dotted.key.path>",
            file=sys.stderr,
        )
        return 1
    path = Path(argv[0])
    key_path = argv[1]
    try:
        with path.open("rb") as fh:
            data: object = tomllib.load(fh)
    except FileNotFoundError:
        print(f"File not found: {path}", file=sys.stderr)
        return 1
    except tomllib.TOMLDecodeError as exc:
        print(f"Malformed TOML in {path}: {exc}", file=sys.stderr)
        return 1

    cursor: object = data
    for segment in key_path.split("."):
        if not isinstance(cursor, dict) or segment not in cursor:
            print(
                f"Key path {key_path!r} not found in {path} (failed at {segment!r}).",
                file=sys.stderr,
            )
            return 1
        cursor = cursor[segment]

    print(cursor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
