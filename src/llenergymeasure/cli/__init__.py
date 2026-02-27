"""Command-line interface for llem.

Skeleton CLI — run and config commands will be added in Phase 7.
"""

from __future__ import annotations

from typing import Annotated

import typer

from llenergymeasure import __version__

app = typer.Typer(
    name="llem",
    help="LLM inference efficiency measurement framework",
    add_completion=False,
)


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        print(f"llem v{__version__}")
        raise typer.Exit()


@app.callback()  # type: ignore[misc]
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=version_callback, is_eager=True),
    ] = False,
) -> None:
    """LLM inference efficiency measurement framework."""


__all__ = ["app"]

if __name__ == "__main__":
    app()
