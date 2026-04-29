"""Widen ``miner_pins.{producer}`` SpecifierSet in an engine's SSOT.

Helper for the ``/approve-reuse`` slash-command workflow. Reads the
per-engine SSOT (``engine_versions/{engine}.yaml``), inspects the current
``library.current_version`` and the ``miner_pins.{ssot_key}`` SpecifierSet
for the requested producer, and widens the SpecifierSet so the bumped
version falls inside the envelope.

Determinism contract
--------------------
For a given (SSOT contents, engine, producer) input, the widening output
is a pure function of the inputs. Re-running with no change to the SSOT
emits ``changed=false`` and exits 0 without mutating the file. The
"already widened" short-circuit prevents the workflow from looping when
the same slash command runs twice.

The mutation is line-surgical: only the ``miner_pins.{key}: <range>``
line is rewritten. Header comments, string-quoting style, key order, and
every other line are preserved byte-for-byte. Roundtripping the YAML
through ``yaml.safe_load`` + ``yaml.safe_dump`` would strip comments and
re-quote scalars, breaking the determinism contract on subsequent runs.

User-facing producer naming
---------------------------
``producer`` is ``invariants | schemas`` — matching the probe primitive's
user-facing producer kinds and the suggestion text emitted in probe-fail
PR comments. The mapping to the SSOT's ``miner_pins`` keys
(``static | dynamic | discovery``) is the same one the probe uses
(``_SSOT_PIN_FOR_PRODUCER``):

    invariants -> static
    schemas    -> discovery

Outputs
-------
When invoked with ``$GITHUB_OUTPUT`` set, emits three keys for the
workflow to consume:

    changed=<true|false>
    old_range=<existing SpecifierSet, e.g. ">=4.56,<4.57">
    new_range=<post-widening SpecifierSet>

CLI
---
    python scripts/widen_miner_pin.py --engine transformers --producer invariants
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import yaml
from packaging.specifiers import SpecifierSet
from packaging.version import Version

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._probe import _SSOT_PIN_FOR_PRODUCER, ProducerKind  # noqa: E402
from scripts.engine_miners._ssot import ssot_path  # noqa: E402


def _widen_range_string(existing: str, version: Version) -> str:
    """Return a ``>=A,<B``-shaped range string that includes ``version``.

    Recognised shape: ``>=A,<B`` (the convention in every
    ``engine_versions/*.yaml`` miner_pin today). When ``version >= B``,
    the upper bound is widened to the next minor above ``version``
    (e.g. ``4.58.0`` -> ``<4.59``).

    Falls back to appending ``,<=current_version`` for unrecognised
    shapes — loud-but-functional; the workflow will still commit a
    deterministic result. The returned string preserves the order
    callers expect to see in the SSOT (lower bound first).
    """
    parts = [p.strip() for p in existing.split(",") if p.strip()]
    lower: str | None = None
    upper: str | None = None
    extras: list[str] = []
    for part in parts:
        if part.startswith((">=", ">")):
            lower = part
        elif part.startswith(("<=", "<")):
            upper = part
        else:
            extras.append(part)

    if lower is None or upper is None or extras:
        # Unrecognised shape — preserve every original constraint and
        # add a ``<=current_version`` ceiling so the bumped version is
        # covered. Ordering: original parts in original order, then the
        # new ceiling appended.
        return ",".join([*parts, f"<={version}"])

    next_minor = f"{version.major}.{version.minor + 1}"
    return f"{lower},<{next_minor}"


def _emit_outputs(*, changed: bool, old_range: str, new_range: str) -> None:
    """Write step outputs to ``$GITHUB_OUTPUT`` if running under Actions."""
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    with Path(output_path).open("a", encoding="utf-8") as fh:
        fh.write(f"changed={'true' if changed else 'false'}\n")
        fh.write(f"old_range={old_range}\n")
        fh.write(f"new_range={new_range}\n")


# Match `  static: ">=4.56,<4.57"` (or single-quoted, or unquoted). Captures
# the indent + key prefix, the optional opening quote, the value, and the
# optional closing quote. Anchored to BOL so we never substitute inside a
# nested mapping where the same key might appear at greater indentation.
_PIN_LINE_RE = re.compile(
    r"^(?P<prefix>(?P<indent> {2,})(?P<key>[a-z]+):\s*)"
    r"(?P<oq>['\"])?(?P<value>[^'\"\n]+?)(?P=oq)?\s*$"
)


def _replace_pin_line(text: str, pin_key: str, new_value: str) -> tuple[str, str | None]:
    """Replace the ``miner_pins.{pin_key}`` line; return (new_text, old_value).

    ``old_value`` is ``None`` if no matching line was found inside the
    ``miner_pins:`` block. The block is identified by the literal
    ``miner_pins:`` line at the start of a line (zero indent). Mutation is
    confined to the indented child lines that follow.
    """
    lines = text.splitlines(keepends=True)
    in_block = False
    block_indent: str | None = None
    old_value: str | None = None
    for idx, line in enumerate(lines):
        if line.startswith("miner_pins:"):
            in_block = True
            continue
        if in_block:
            stripped = line.lstrip(" ")
            if stripped == line and stripped != "\n" and stripped != "":
                # Dedent below `miner_pins:` -> block ended without a hit.
                break
            match = _PIN_LINE_RE.match(line.rstrip("\n"))
            if match is None:
                continue
            if block_indent is None:
                block_indent = match.group("indent")
            elif match.group("indent") != block_indent:
                # Deeper-indented child of a nested mapping; skip.
                continue
            if match.group("key") != pin_key:
                continue
            old_value = match.group("value")
            quote = match.group("oq") or '"'
            newline = "\n" if line.endswith("\n") else ""
            lines[idx] = f"{match.group('prefix')}{quote}{new_value}{quote}{newline}"
            return "".join(lines), old_value
    return text, old_value


def widen(*, engine: str, producer: ProducerKind) -> int:
    """Widen ``miner_pins.{producer}`` for ``engine``; return exit status."""
    if producer not in _SSOT_PIN_FOR_PRODUCER:
        print(
            f"Unknown producer {producer!r}; expected one of {sorted(_SSOT_PIN_FOR_PRODUCER)}.",
            file=sys.stderr,
        )
        return 2

    pin_key = _SSOT_PIN_FOR_PRODUCER[producer]
    path = ssot_path(engine)
    text = path.read_text()

    # Parse for value extraction only; never write the parsed dict back.
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        print(f"SSOT at {path} is not a mapping.", file=sys.stderr)
        return 2

    library = data.get("library") or {}
    current_version_raw = library.get("current_version")
    if not current_version_raw:
        print(f"library.current_version missing from {path}.", file=sys.stderr)
        return 2

    miner_pins = data.get("miner_pins") or {}
    if pin_key not in miner_pins:
        print(
            f"miner_pins.{pin_key} missing from {path}; present keys: {sorted(miner_pins)}.",
            file=sys.stderr,
        )
        return 2

    version = Version(str(current_version_raw))
    old_range = str(miner_pins[pin_key])

    if version in SpecifierSet(old_range):
        print(
            f"{engine} miner_pins.{pin_key} = {old_range!r} already covers "
            f"library.current_version={version}; no change."
        )
        _emit_outputs(changed=False, old_range=old_range, new_range=old_range)
        return 0

    new_range = _widen_range_string(old_range, version)
    if new_range == old_range or version not in SpecifierSet(new_range):
        # Either widening was a no-op or the result still doesn't cover the
        # version (defensive: the SpecifierSet round-trip should always
        # succeed for the recognised shape, but fail loud rather than
        # commit a wrong-result widening).
        print(
            f"Refusing to widen {engine} miner_pins.{pin_key}: computed "
            f"{new_range!r} does not cover {version} (input was {old_range!r}).",
            file=sys.stderr,
        )
        return 2

    new_text, old_line_value = _replace_pin_line(text, pin_key, new_range)
    if old_line_value is None:
        # Parser saw the key but the regex missed it — inconsistent shape.
        print(
            f"Could not locate miner_pins.{pin_key} line in {path}.",
            file=sys.stderr,
        )
        return 2

    path.write_text(new_text)
    print(
        f"Widened {engine} miner_pins.{pin_key}: {old_range!r} -> {new_range!r} "
        f"(library.current_version={version})."
    )
    _emit_outputs(changed=True, old_range=old_range, new_range=new_range)
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="scripts.widen_miner_pin",
        description="Widen miner_pins.{producer} SpecifierSet in an engine SSOT.",
    )
    parser.add_argument(
        "--engine",
        required=True,
        choices=("transformers", "vllm", "tensorrt"),
        help="Engine whose SSOT to mutate.",
    )
    parser.add_argument(
        "--producer",
        required=True,
        choices=("invariants", "schemas"),
        help="User-facing producer kind (matches the probe primitive).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return widen(engine=args.engine, producer=args.producer)


if __name__ == "__main__":
    raise SystemExit(main())
