"""Table formatting utilities for CLI output.

This module provides functions for formatting table fields and values
with consistent styling.
"""

from __future__ import annotations

from typing import Any

from rich.table import Table

from llm_energy_measure.cli.display.console import console


def format_field(
    name: str,
    value: Any,
    is_default: bool,
    nested: bool = False,
) -> tuple[str, str]:
    """Format field name and value with appropriate styling.

    Args:
        name: Field name.
        value: Field value.
        is_default: Whether value is the default (dim if True).
        nested: Whether this is a nested field (indented).

    Returns:
        Tuple of (formatted_name, formatted_value) for table row.
    """
    indent = "  " if nested else ""
    if is_default:
        return f"[dim]{indent}{name}[/dim]", f"[dim]{value}[/dim]"
    else:
        style = "cyan" if nested else "green"
        return f"[{style}]{indent}{name}[/{style}]", str(value)


def add_section_header(table: Table, name: str) -> None:
    """Add a bold section header row to the table."""
    table.add_row(f"[bold]{name}[/bold]", "")


def print_value(
    name: str, value: Any, is_default: bool, indent: int = 2, show_defaults: bool = True
) -> None:
    """Print a config value with dim styling for defaults.

    Args:
        name: Field name.
        value: Field value.
        is_default: Whether this is the default value.
        indent: Indentation level.
        show_defaults: If False, skip printing default values.
    """
    if not show_defaults and is_default:
        return

    spaces = " " * indent
    if is_default:
        console.print(f"[dim]{spaces}{name}: {value}[/dim]")
    else:
        console.print(f"{spaces}[cyan]{name}[/cyan]: {value}")


def format_dict_field(
    name: str,
    value: Any,
    default: Any,
    nested: bool = False,
) -> tuple[str, str]:
    """Format field from dict config with default comparison.

    Args:
        name: Field name.
        value: Current value.
        default: Default value for comparison.
        nested: Whether this is a nested field.

    Returns:
        Tuple of (formatted_name, formatted_value) for table row.
    """
    is_default = value == default
    indent = "  " if nested else ""
    if is_default:
        return f"[dim]{indent}{name}[/dim]", f"[dim]{value}[/dim]"
    else:
        style = "cyan" if nested else "green"
        return f"[{style}]{indent}{name}[/{style}]", str(value)
