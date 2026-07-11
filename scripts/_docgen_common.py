"""Shared argparse + write-markdown boilerplate for the docs generators.

Each ``scripts/generate_*.py`` builds a Markdown document from a live SSOT and
either writes it to a fixed path or exposes a ``--output`` flag. The rendering
(content) logic stays in each generator; this module owns only the plumbing:
the standard output option, the stdout-or-file emit, and the fixed-path write.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def add_output_option(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the standard ``--output/-o`` optional path flag (stdout when omitted)."""
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Write output to this file path (default: stdout)",
    )
    return parser


def emit_markdown(markdown: str, output: Path | None) -> None:
    """Write *markdown* to *output*, or print it to stdout when *output* is None.

    When a path is given, parent directories are created and the destination is
    logged to stderr (keeping stdout clean for piping); otherwise the document
    is printed to stdout.
    """
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown)
        print(f"Written to {output}", file=sys.stderr)
    else:
        print(markdown)


def write_doc(path: Path, content: str) -> None:
    """Create parent directories, write *content* to *path*, and log the location."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print(f"Generated: {path}")
