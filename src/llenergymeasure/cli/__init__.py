"""Command-line interface for lem.

Provides commands for:
- Running experiments
- Aggregating raw results
- Validating configurations
- Listing and inspecting results
"""

from __future__ import annotations

# Load .env file BEFORE any llenergymeasure imports (constants reads env vars at import time)
from dotenv import load_dotenv

load_dotenv()  # Loads from .env in current directory or parents

# ruff: noqa: E402 - imports must come after load_dotenv()
import os
from typing import Annotated

import typer

from llenergymeasure.cli.config import config_app
from llenergymeasure.cli.display import console
from llenergymeasure.cli.results import results_app
from llenergymeasure.constants import SCHEMA_VERSION
from llenergymeasure.logging import setup_logging

app = typer.Typer(
    name="lem",
    help="LLM inference efficiency measurement framework",
    add_completion=False,
)

# Register subcommand groups
app.add_typer(config_app, name="config")
app.add_typer(results_app, name="results")


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        console.print(f"lem v{SCHEMA_VERSION}")
        raise typer.Exit()


@app.callback()  # type: ignore[misc]
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=version_callback, is_eager=True),
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable full logs with timestamps")
    ] = False,
    quiet: Annotated[
        bool, typer.Option("--quiet", "-q", help="Minimal output (warnings only)")
    ] = False,
    json_output: Annotated[
        bool, typer.Option("--json", help="Output results as JSON (machine-readable)")
    ] = False,
) -> None:
    """LLM inference efficiency measurement framework."""
    from llenergymeasure.logging import VerbosityType

    # Determine verbosity level
    verbosity: VerbosityType
    if quiet:
        verbosity = "quiet"
    elif verbose:
        verbosity = "verbose"
    else:
        verbosity = "normal"

    # Set environment variable for subprocesses and progress module
    os.environ["LLM_ENERGY_VERBOSITY"] = verbosity
    # Set JSON output mode flag for subcommands to check
    os.environ["LLM_ENERGY_JSON_OUTPUT"] = "true" if json_output else "false"
    setup_logging(verbosity=verbosity)


# Import and register commands after app is defined
# This must happen after app definition to avoid circular imports
def _register_commands() -> None:
    """Register all commands with the app.

    This is done in a function to control import order and avoid
    circular imports that can occur with typer.

    Commands are imported directly from their defining modules to avoid
    unnecessary re-exports and keep module responsibilities clear.
    """
    from llenergymeasure.cli import (
        batch,
        campaign,
        doctor,
        experiment,
        init_cmd,
        listing,
        resume,
        schedule,
    )

    # Core experiment commands
    app.command("experiment")(experiment.experiment_cmd)
    app.command("aggregate")(experiment.aggregate_cmd)

    # Listing commands (informational)
    app.command("datasets")(listing.list_datasets_cmd)
    app.command("presets")(listing.list_presets_cmd)
    app.command("gpus")(listing.list_gpus_cmd)
    app.command("doctor")(doctor.doctor_cmd)

    # Execution modes
    app.command("batch")(batch.batch_run_cmd)
    app.command("schedule")(schedule.schedule_experiment_cmd)
    app.command("campaign")(campaign.campaign_cmd)

    # Setup / configuration commands
    app.command("init")(init_cmd.init_cmd)
    app.command("resume")(resume.resume_cmd)


# Register commands
_register_commands()


# Export commonly used items for backwards compatibility
__all__ = ["app", "console", "main"]

if __name__ == "__main__":
    app()
