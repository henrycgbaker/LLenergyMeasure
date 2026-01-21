"""CLI display package.

This package provides display and formatting helpers for CLI output,
organised into focused modules:

- console: Rich console setup and duration formatting
- tables: Table field formatting utilities
- summaries: Config and state summary rendering
- results: Result display functions
"""

from llm_energy_measure.cli.display.console import console, format_duration
from llm_energy_measure.cli.display.results import show_aggregated_result, show_raw_result
from llm_energy_measure.cli.display.summaries import (
    display_config_summary,
    display_incomplete_experiment,
    show_effective_config,
)
from llm_energy_measure.cli.display.tables import (
    add_section_header,
    format_dict_field,
    format_field,
    print_value,
)

__all__ = [
    "add_section_header",
    "console",
    "display_config_summary",
    "display_incomplete_experiment",
    "format_dict_field",
    "format_duration",
    "format_field",
    "print_value",
    "show_aggregated_result",
    "show_effective_config",
    "show_raw_result",
]
