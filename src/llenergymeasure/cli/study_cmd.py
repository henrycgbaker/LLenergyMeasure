"""llem study - study-authoring helper commands.

Currently hosts ``build-skeleton``, which emits a commented starting study YAML
with every auto-expandable curated knob for an engine pre-expanded. CLI stays
thin (Layer 6): it parses flags and delegates to the public API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

study_app = typer.Typer(
    name="study",
    help="Study-authoring helpers",
    add_completion=False,
)


@study_app.command(name="build-skeleton")  # type: ignore[misc]
def build_skeleton(
    engine: Annotated[
        str,
        typer.Option(
            "--engine", help="Engine to build a skeleton for (transformers/vllm/tensorrt)"
        ),
    ],
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Write skeleton to this path (prints to stdout if omitted)"),
    ] = None,
) -> None:
    """Emit a commented starting study YAML with every auto-expandable knob."""
    from llenergymeasure.api import build_study_skeleton
    from llenergymeasure.utils.exceptions import ConfigError

    try:
        skeleton = build_study_skeleton(engine)
    except ConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    if out is None:
        typer.echo(skeleton)
    else:
        out.write_text(skeleton)
        typer.echo(f"Wrote study skeleton to {out}")
