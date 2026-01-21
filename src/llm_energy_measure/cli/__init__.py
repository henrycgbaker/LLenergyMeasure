"""Command-line interface for llm-energy-measure.

Provides commands for:
- Running experiments
- Aggregating raw results
- Validating configurations
- Listing and inspecting results
"""

from __future__ import annotations

# Load .env file BEFORE any llm_energy_measure imports (constants reads env vars at import time)
from dotenv import load_dotenv

load_dotenv()  # Loads from .env in current directory or parents

# ruff: noqa: E402 - imports must come after load_dotenv()
import os
from typing import Annotated

import typer

from llm_energy_measure.cli.config import config_app
from llm_energy_measure.cli.display import console
from llm_energy_measure.cli.results import results_app
from llm_energy_measure.constants import SCHEMA_VERSION
from llm_energy_measure.logging import setup_logging

app = typer.Typer(
    name="llm-energy-measure",
    help="LLM inference efficiency measurement framework",
    add_completion=False,
)

# Register subcommand groups
app.add_typer(config_app, name="config")
app.add_typer(results_app, name="results")


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        console.print(f"llm-energy-measure v{SCHEMA_VERSION}")
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
) -> None:
    """LLM inference efficiency measurement framework."""
    from llm_energy_measure.logging import VerbosityType

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
    setup_logging(verbosity=verbosity)


# Import and register commands after app is defined
# This must happen after app definition to avoid circular imports
def _register_commands() -> None:
    """Register all commands with the app.

    This is done in a function to control import order and avoid
    circular imports that can occur with typer.
    """
    from llm_energy_measure.cli import experiment

    # Register experiment commands with their original signatures
    app.command("run")(experiment.run_cmd)  # Legacy command
    app.command("experiment")(experiment.experiment_cmd)
    app.command("aggregate")(experiment.aggregate_cmd)
    app.command("datasets")(experiment.list_datasets_cmd)
    app.command("presets")(experiment.list_presets_cmd)
    app.command("gpus")(experiment.list_gpus_cmd)
    app.command("batch")(experiment.batch_run_cmd)
    app.command("schedule")(experiment.schedule_experiment_cmd)


# Register commands
_register_commands()


# Export commonly used items for backwards compatibility
__all__ = ["app", "console", "main"]

if __name__ == "__main__":
    app()
