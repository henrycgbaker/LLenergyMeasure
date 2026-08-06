"""``llem doctor`` - the environment health check.

Runs a sectioned readiness check (GPU/driver, engines, energy samplers, Docker,
credentials, resolved configuration, and the image schema handshake) and prints
each line prefixed ``[ok]``/``[warn]``/``[fail]`` with an actionable fix hint on
anything that is not OK.

Exit codes:
  * default: 0, unless a hard failure is present (e.g. an image schema MISMATCH),
    which exits 1 so CI can gate on it.
  * ``--check``: 0 = all ok, 1 = warnings present, 2 = errors present.

``--json`` emits the full report as machine-readable JSON.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Annotated

import typer

if TYPE_CHECKING:
    from llenergymeasure.api.health import HealthReport

__all__ = ["doctor_command"]

_STATUS_PREFIX = {"ok": "[ok]  ", "warn": "[warn]", "fail": "[fail]"}


def doctor_command(
    check: Annotated[
        bool,
        typer.Option(
            "--check",
            help="Exit 0=ok, 1=warnings, 2=errors (for CI/scripting). Output is unchanged.",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit the full report as machine-readable JSON."),
    ] = False,
) -> None:
    """Check whether this host is ready to run measurements."""
    from llenergymeasure.api.health import build_health_report

    report = build_health_report()

    if json_output:
        typer.echo(json.dumps(report.to_dict(), indent=2))
    else:
        _render(report)

    # Default invocation stays green on warnings but fails hard on errors
    # (e.g. an image schema mismatch). --check
    # grades every severity (0=ok, 1=warnings, 2=errors).
    default_code = 1 if report.worst == "fail" else 0
    exit_code = report.check_exit_code if check else default_code

    if exit_code != 0:
        raise typer.Exit(code=exit_code)


def _render(report: HealthReport) -> None:
    typer.echo("Environment health check")
    typer.echo("=" * 24)

    for section in report.sections:
        typer.echo("")
        typer.echo(section.title)
        for line in section.lines:
            typer.echo(f"  {_STATUS_PREFIX[line.status]} {line.message}")
            if line.fix:
                typer.echo(f"           -> {line.fix}")

    counts = report.counts
    typer.echo("")
    typer.echo(
        f"Summary: {counts['ok']} ok, {counts['warn']} warning(s), {counts['fail']} error(s)"
    )
