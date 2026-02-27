"""Command-line interface for llem."""

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


# Register commands — deferred imports inside command functions keep startup fast
from llenergymeasure.cli.run import run as _run_cmd  # noqa: E402

app.command(name="run", help="Run an LLM efficiency experiment")(_run_cmd)

from llenergymeasure.cli.config_cmd import config_command as _config_cmd  # noqa: E402

app.command(name="config", help="Show environment and configuration status")(_config_cmd)

__all__ = ["app"]

if __name__ == "__main__":
    app()
