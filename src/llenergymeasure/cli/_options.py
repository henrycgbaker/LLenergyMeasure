"""Shared Typer option definitions reused across llem CLI commands."""

from __future__ import annotations

from typing import Annotated

import typer

#: Standard ``-v/--verbose`` count option. 0 = WARNING, 1 = INFO, 2+ = DEBUG.
VerboseOption = Annotated[
    int,
    typer.Option("--verbose", "-v", count=True, help="Increase verbosity (-v=INFO, -vv=DEBUG)"),
]
