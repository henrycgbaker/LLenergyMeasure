"""``llem study`` - tools for writing study files.

The ``study`` group holds commands that help you author a study file. A study
file is the complete, explicit specification of what runs; ``llem run`` stays
the sole executor. Today the group has one command, ``init``, which writes a
starter study file for a model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

__all__ = ["study_app"]

study_app = typer.Typer(
    name="study",
    help="Write and prepare study files (llem run stays the executor).",
    add_completion=False,
    no_args_is_help=True,
)


@study_app.command("init")  # type: ignore[misc, untyped-decorator]
def study_init(
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Model id or HuggingFace path for the study."),
    ],
    engine: Annotated[
        str | None,
        typer.Option(
            "--engine",
            "-e",
            help="Engine to write (transformers, vllm, tensorrt). Omit to write all engines.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Where to write the study file (default: study.yaml). Will not overwrite.",
        ),
    ] = None,
    defaults: Annotated[
        bool,
        typer.Option(
            "--defaults",
            help="Set every field to its default (a runnable baseline) instead of "
            "leaving them commented out.",
        ),
    ] = False,
    bounds: Annotated[
        bool,
        typer.Option(
            "--bounds",
            help="Emit every field with a derivable value range as an explicit "
            "value list under sweep: - a maximal grid to prune before running.",
        ),
    ] = False,
) -> None:
    """Write a study file for a model, ready to edit and run with ``llem run``.

    Without ``--defaults`` the tuning fields are present but commented out, each
    annotated with its type, range, and default - uncomment the ones you want.
    With ``--defaults`` those fields are filled in at their defaults, giving a
    reproducible baseline you can run as-is. With ``--bounds`` every field with
    a derivable value range becomes an explicit value list under ``sweep:``,
    forming a maximal grid you prune down to the study you want.
    """
    from llenergymeasure.config.scaffold import (
        all_engine_names,
        render_study_bounds,
        render_study_scaffold,
    )
    from llenergymeasure.config.ssot import Engine

    if defaults and bounds:
        raise typer.BadParameter(
            "--defaults and --bounds are mutually exclusive; pick one.",
            param_hint="--bounds",
        )

    if not model.strip():
        raise typer.BadParameter("Model must not be empty.", param_hint="--model")

    if engine is not None:
        valid = [e.value for e in Engine]
        if engine not in valid:
            raise typer.BadParameter(
                f"Unknown engine {engine!r}. Choose one of: {', '.join(valid)}.",
                param_hint="--engine",
            )
        engines = [engine]
    else:
        engines = all_engine_names()

    if bounds:
        text = render_study_bounds(model, engines)
    else:
        text = render_study_scaffold(model, engines, defaults=defaults)

    out_path = output if output is not None else Path("study.yaml")
    if out_path.exists():
        typer.echo(
            f"Refusing to overwrite existing file: {out_path}. "
            "Pass -o <path> to write elsewhere, or remove it first.",
            err=True,
        )
        raise typer.Exit(code=1)

    out_path.write_text(text)
    typer.echo(f"Wrote study file: {out_path}")
    if bounds:
        typer.echo(f"Prune the sweep value lists, then run: llem run {out_path}")
    else:
        typer.echo(f"Edit it, then run: llem run {out_path}")


@study_app.command("plan")  # type: ignore[misc, untyped-decorator]
def study_plan(
    study_file: Annotated[
        Path,
        typer.Argument(help="Study YAML file to preview (as llem run would expand it)."),
    ],
) -> None:
    """Preview what ``llem run <study_file>`` would execute, without running it.

    Prints a read-only funnel - declared grid points, then known-invalid points
    pruned by engine rules, then duplicate configs merged away, then experiments
    times cycles equals total runs - plus a wall-clock lower bound from the
    thermal gaps alone. No experiments run, nothing is written, the GPU is never
    touched. Exit 0 on a valid plan (even with skips); nonzero when the file is
    missing, unparseable, or produces no experiments.
    """
    from pydantic import ValidationError

    from llenergymeasure.api import load_study
    from llenergymeasure.cli._study_plan import build_funnel, render_funnel
    from llenergymeasure.utils.exceptions import ConfigError

    try:
        study = load_study(study_file)
    except (ConfigError, ValidationError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from None

    funnel = build_funnel(study)
    title = study.study_name or str(study_file)
    typer.echo(render_funnel(funnel, title))
