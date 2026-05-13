"""Update the ``last_probe:`` block in an engine current.yaml from a ProbeReport JSON.

Helper for the engine-coupling probe-writeback workflow. Reads a
``ProbeReport`` JSON document on stdin (the shape emitted by
``scripts._probe``) and rewrites the four mutable fields under
``last_probe:`` in ``engine_versions/{engine}.yaml``:

    verdict
    version_inside_envelope
    fingerprint
    fingerprint_drift

Determinism contract
--------------------
For a given (current.yaml contents, ProbeReport JSON) input pair, the rewrite
output is a pure function of the inputs. Re-running with no change to
input emits ``changed=false`` to ``$GITHUB_OUTPUT`` and exits 0 without
mutating the file. The compare-before-write short-circuit is what
keeps the bot-writeback step from looping when nothing material has
changed since the previous run.

The mutation is line-surgical: only the four mutable lines under
``last_probe:`` are rewritten. Header comments, string-quoting style,
key order, blank lines, and every other line are preserved
byte-for-byte. Roundtripping the YAML through ``yaml.safe_load`` +
``yaml.safe_dump`` would strip comments and re-quote scalars, breaking
the determinism contract on subsequent runs.

Mirrors the line-surgical pattern in ``scripts/widen_miner_pin.py``;
share its ``_PIN_LINE_RE``-style block-scoped key matching but adapt
for the fact that ``last_probe:`` values include lists, booleans, and
``null``.

Outputs
-------
When invoked with ``$GITHUB_OUTPUT`` set, emits two keys for the
workflow to consume:

    changed=<true|false>
    verdict=<pass|fail>

CLI
---
    python scripts/update_last_probe.py --engine transformers < probe.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.engine_producers._current import current_path, safe_version  # noqa: E402

# Match `  verdict: pass`, `  verdict: "pass"`, `  fingerprint_drift: []`,
# `  version_inside_envelope: null`, etc. Captures indent + key + the rest
# of the value line so we can reconstruct with the original indentation.
_LAST_PROBE_LINE_RE = re.compile(r"^(?P<indent> {2,})(?P<key>[a-z_]+):\s*(?P<value>.*)$")

# The four mutable fields under ``last_probe:``. Order intentional -
# matches the canonical current.yaml layout so the regenerated block is
# byte-identical when nothing changes.
_MUTABLE_FIELDS: tuple[str, ...] = (
    "verdict",
    "version_inside_envelope",
    "fingerprint",
    "fingerprint_drift",
)


def _format_scalar(value: Any) -> str:
    """Render *value* in the current.yaml's canonical scalar style.

    The current.yaml uses unquoted YAML scalars for ``verdict`` (bare words like
    ``pass``, ``fail``, ``unrun``), unquoted booleans / ``null`` for
    diagnostics, and a flow-style empty list for ``fingerprint_drift``.
    Strings that look like fingerprints are rendered with double quotes
    to keep them stable through YAML's natural-language scalar coercion
    (a 64-char hex digest could otherwise be reinterpreted as
    something exotic). Lists of strings render in flow style for
    compactness - drift lists are short.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        if not value:
            return "[]"
        rendered = ", ".join(_format_scalar(item) for item in value)
        return f"[{rendered}]"
    if isinstance(value, str):
        # ``verdict`` is a bare-word enum (pass/fail/unrun); render it
        # unquoted to match the canonical seed shape. Everything else
        # quoted - fingerprints (hex) and arbitrary strings round-trip
        # safely double-quoted.
        if value in {"pass", "fail", "unrun"}:
            return value
        return f'"{value}"'
    # int / float fall through to repr (no current.yaml field uses them
    # under last_probe but defensive).
    return str(value)


def _replace_last_probe_block(text: str, updates: dict[str, Any]) -> tuple[str, bool]:
    """Replace mutable fields under ``last_probe:``; return (new_text, changed).

    ``changed`` is ``False`` iff every targeted line is byte-identical
    before and after the rewrite - this is the determinism short-circuit
    that prevents the workflow looping on no-op writebacks. Lines whose
    keys are not in ``updates`` are left untouched.

    Raises :class:`ValueError` if the ``last_probe:`` block cannot be
    located, or if any field in ``updates`` is missing from the block.
    """
    lines = text.splitlines(keepends=True)
    in_block = False
    block_indent: str | None = None
    new_lines = list(lines)
    seen: set[str] = set()
    changed = False

    for idx, line in enumerate(lines):
        if line.startswith("last_probe:"):
            in_block = True
            continue
        if not in_block:
            continue

        # Block boundary: a line that doesn't start with the block
        # indent (and is non-empty / non-comment) ends ``last_probe:``.
        stripped_no_nl = line.rstrip("\n")
        if stripped_no_nl == "" or stripped_no_nl.startswith("#"):
            continue
        match = _LAST_PROBE_LINE_RE.match(stripped_no_nl)
        if match is None:
            # Dedent or unexpected shape - block ended.
            break

        indent = match.group("indent")
        if block_indent is None:
            block_indent = indent
        elif indent != block_indent:
            # Deeper-indented child of a nested mapping; skip.
            continue

        key = match.group("key")
        if key not in updates:
            continue
        seen.add(key)

        new_value = _format_scalar(updates[key])
        newline = "\n" if line.endswith("\n") else ""
        new_line = f"{indent}{key}: {new_value}{newline}"
        if new_line != line:
            changed = True
        new_lines[idx] = new_line

    if not in_block or block_indent is None:
        raise ValueError(
            "last_probe: block not found in current.yaml (or block was empty). "
            "Every supported engine current.yaml must have a last_probe: block; "
            "see engine_versions/transformers.yaml for the canonical shape."
        )

    missing = set(updates) - seen
    if missing:
        raise ValueError(
            f"last_probe: block missing required fields: {sorted(missing)}. "
            "Add them to the current.yaml's last_probe: block before running the "
            "writeback step."
        )

    return "".join(new_lines), changed


def _emit_outputs(*, changed: bool, verdict: str) -> None:
    """Write step outputs to ``$GITHUB_OUTPUT`` if running under Actions."""
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    with Path(output_path).open("a", encoding="utf-8") as fh:
        fh.write(f"changed={'true' if changed else 'false'}\n")
        fh.write(f"verdict={verdict}\n")


def _write_archive_last_probe(*, engine: str, version: str, report: dict[str, Any]) -> bool:
    """Mirror the last_probe block to the per-version archive.

    Each version's archive at ``engine_versions/<engine>/<safe>/outputs/``
    holds a frozen ``last_probe.yaml`` capturing the verdict + diagnostics
    of the last successful pipeline run at that version. The file is a
    derivative of the current.yaml's ``last_probe:`` block; it is NOT the
    source of truth. The current.yaml remains authoritative for downstream
    consumers.

    Gated on the engine being vendored at this version (per-engine
    ``__init__.py`` present). When an engine is not yet vendored, this
    function returns ``False`` (no-op); the per-engine vendor PR adds the
    package marker and bootstraps the archive. Without this gate, a
    half-baked archive directory (``outputs/`` without ``__init__.py``)
    would be created on every pipeline run for engines awaiting vendoring.

    Idempotent: returns ``True`` when the file was created or rewritten
    with new content, ``False`` if the existing file already matches the
    payload OR the engine is not vendored at this version.
    """
    safe = safe_version(version)
    archive_root = _PROJECT_ROOT / "engine_versions" / engine / safe
    if not (archive_root / "__init__.py").is_file():
        # Engine not yet vendored at this version; archive mirror is a no-op.
        return False
    archive_dir = archive_root / "outputs"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / "last_probe.yaml"

    # Build payload: same fields, same scalar formatting as the current.yaml block.
    rendered_lines = [f"{field}: {_format_scalar(report[field])}\n" for field in _MUTABLE_FIELDS]
    payload = "".join(rendered_lines)

    if archive_path.exists() and archive_path.read_text() == payload:
        return False
    archive_path.write_text(payload)
    return True


def update(*, engine: str, report: dict[str, Any]) -> int:
    """Apply *report*'s last_probe-relevant fields to the engine current.yaml.

    Returns 0 on success (whether or not a change was written), 2 on
    infrastructure failure (current.yaml missing / malformed, ProbeReport
    missing required fields).

    Also mirrors the last_probe block as a standalone YAML to the
    versioned archive at
    ``engine_versions/<engine>/<safe>/outputs/last_probe.yaml``.
    The archive write happens after the current.yaml update succeeds; on archive-
    write failure the current.yaml update is preserved (no rollback). Both paths
    are idempotent so subsequent runs converge.
    """
    required = {
        "verdict",
        "version_inside_envelope",
        "fingerprint",
        "fingerprint_drift",
        "library_version",
    }
    missing = required - report.keys()
    if missing:
        print(
            f"ProbeReport JSON missing required fields: {sorted(missing)}.",
            file=sys.stderr,
        )
        return 2

    path = current_path(engine)
    try:
        text = path.read_text()
    except FileNotFoundError:
        print(
            f"Engine current.yaml not found at {path}. "
            f"Every supported engine must have a last_probe: block before "
            f"the writeback helper is wired up for it.",
            file=sys.stderr,
        )
        return 2

    updates = {field: report[field] for field in _MUTABLE_FIELDS}
    try:
        new_text, changed = _replace_last_probe_block(text, updates)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    verdict = str(report["verdict"])
    if changed:
        path.write_text(new_text)
        print(f"Updated {path} last_probe block (verdict={verdict}).")
    else:
        print(f"No change to {path} last_probe block (verdict={verdict}).")

    # Mirror to the versioned archive. Failure here is non-fatal; the
    # current.yaml is authoritative and the archive will be regenerated on the
    # next run. We still surface the failure to stderr so it's visible
    # in CI logs.
    library_version = str(report["library_version"])
    try:
        archive_changed = _write_archive_last_probe(
            engine=engine, version=library_version, report=report
        )
    except (OSError, ValueError) as exc:
        print(
            f"Warning: could not mirror last_probe to archive for {engine} "
            f"v{library_version}: {exc!r}. current.yaml update preserved.",
            file=sys.stderr,
        )
    else:
        if archive_changed:
            print(f"Mirrored last_probe to archive at {engine}/v{library_version}.")

    _emit_outputs(changed=changed, verdict=verdict)
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="scripts.update_last_probe",
        description=(
            "Update the last_probe: block in engine_versions/{engine}.yaml "
            "from a ProbeReport JSON read on stdin."
        ),
    )
    parser.add_argument(
        "--engine",
        required=True,
        choices=("transformers", "vllm", "tensorrt"),
        help="Engine whose current.yaml to mutate.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    raw = sys.stdin.read()
    if not raw.strip():
        print("No ProbeReport JSON received on stdin.", file=sys.stderr)
        return 2
    try:
        report = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"Could not parse stdin as JSON: {exc}.", file=sys.stderr)
        return 2
    if not isinstance(report, dict):
        print("ProbeReport JSON must be an object at the top level.", file=sys.stderr)
        return 2
    return update(engine=args.engine, report=report)


if __name__ == "__main__":
    raise SystemExit(main())
