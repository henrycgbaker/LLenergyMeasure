"""Resume command for discovering and continuing interrupted campaigns.

Provides `lem resume` which scans for interrupted campaign manifests,
shows an interactive selection menu, and guides users to resume execution.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated

import questionary
import typer
from rich.table import Table

from llenergymeasure.cli.display import console
from llenergymeasure.orchestration.manifest import CampaignManifest, ManifestManager


def resume_cmd(
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show what would be resumed without executing")
    ] = False,
    wipe: Annotated[bool, typer.Option("--wipe", help="Clear all campaign state files")] = False,
) -> None:
    """Discover and resume interrupted campaigns.

    Scans for campaign manifests in .state/ directory, presents an interactive
    menu to select which campaign to resume, and shows the resume command.

    Examples:

        lem resume               # Interactive selection
        lem resume --dry-run     # Preview what would be resumed
        lem resume --wipe        # Clear all state files
    """
    state_dir = Path(".state")

    # Handle --wipe flag first
    if wipe:
        if not state_dir.exists():
            console.print("[dim]No state directory found. Nothing to clear.[/dim]")
            return

        typer.confirm("Delete ALL state files in .state/?", abort=True)
        shutil.rmtree(state_dir)
        console.print("[green]Cleared all state files.[/green]")
        return

    # Check if state directory exists
    if not state_dir.exists():
        console.print("[dim]No interrupted work found.[/dim]")
        console.print("Run `lem campaign <config.yaml>` to start a campaign.")
        raise typer.Exit(1)

    # Discover manifests
    manifest_files = list(state_dir.glob("**/campaign_manifest.json"))
    if not manifest_files:
        console.print("[dim]No interrupted campaigns found.[/dim]")
        console.print("Run `lem campaign <config.yaml>` to start a campaign.")
        raise typer.Exit(1)

    # Load and filter to incomplete campaigns
    incomplete_campaigns: list[tuple[Path, CampaignManifest]] = []
    for manifest_path in manifest_files:
        manifest_mgr = ManifestManager(manifest_path)
        manifest = manifest_mgr.load()
        if manifest is not None and not manifest.is_complete:
            incomplete_campaigns.append((manifest_path, manifest))

    if not incomplete_campaigns:
        console.print("[dim]No interrupted campaigns found.[/dim]")
        console.print("All discovered campaigns have completed.")
        raise typer.Exit(1)

    # Sort by updated_at descending (most recent first)
    incomplete_campaigns.sort(key=lambda x: x[1].updated_at, reverse=True)

    # Single campaign auto-select
    if len(incomplete_campaigns) == 1:
        manifest_path, manifest = incomplete_campaigns[0]
        console.print(f"Found: [cyan]{manifest.campaign_name}[/cyan]")
    else:
        # Multiple campaigns - show table and menu
        table = Table(title="Interrupted Campaigns")
        table.add_column("ID", style="dim", width=10)
        table.add_column("Name")
        table.add_column("Progress", justify="right")
        table.add_column("Last Activity")

        for _, m in incomplete_campaigns:
            completed = m.completed_count
            total = m.total_experiments
            failed = m.failed_count
            progress = f"{completed}/{total}"
            if failed > 0:
                progress += f" [red]({failed} failed)[/red]"
            table.add_row(
                m.campaign_id[:8],
                m.campaign_name,
                progress,
                m.updated_at.strftime("%Y-%m-%d %H:%M"),
            )

        console.print(table)
        console.print()

        # Build choices for questionary
        choices = [
            questionary.Choice(
                title=f"{m.campaign_name} ({m.completed_count}/{m.total_experiments})",
                value=idx,
            )
            for idx, (_, m) in enumerate(incomplete_campaigns)
        ]

        selection = questionary.select(
            "Select campaign to resume:",
            choices=choices,
        ).ask()

        if selection is None:
            # User cancelled (Ctrl+C)
            raise typer.Abort()

        manifest_path, manifest = incomplete_campaigns[selection]

    # Type narrowing: manifest is guaranteed to be set at this point
    # (either from single campaign auto-select or from user selection above)
    assert manifest is not None

    # Handle --dry-run (show info without prompting)
    if dry_run:
        pending = manifest.pending_count
        console.print("\n[cyan]Dry run - would resume:[/cyan]")
        console.print(f"  Campaign: {manifest.campaign_name}")
        console.print(f"  Campaign ID: {manifest.campaign_id}")
        console.print(f"  Completed: {manifest.completed_count}/{manifest.total_experiments}")
        console.print(f"  Pending: {pending}")
        if manifest.failed_count > 0:
            console.print(f"  Failed: {manifest.failed_count}")
            console.print(f"  To execute (with retry): {pending + manifest.failed_count}")
            console.print(f"  To execute (without retry): {pending}")
        else:
            console.print(f"  To execute: {pending}")
        return

    # Ask about retrying failed experiments
    retry_failed = False
    if manifest.failed_count > 0:
        retry_failed = typer.confirm(
            f"Retry {manifest.failed_count} failed experiments?",
            default=True,
        )

    # Calculate what would be resumed
    pending = manifest.pending_count
    to_resume = pending
    if retry_failed:
        to_resume += manifest.failed_count

    # Show resume instructions
    console.print(f"\n[bold]Resuming: {manifest.campaign_name}[/bold]")
    console.print(f"  State: {manifest_path.parent}")
    console.print(f"  To resume: {to_resume} experiments")
    console.print()

    # Guide user to the resume command
    # The campaign command with --resume flag handles the actual resumption
    console.print("[dim]To resume, run the original campaign command with --resume:[/dim]")
    console.print()
    console.print("  [bold]lem campaign <your-campaign.yaml> --resume[/bold]")
    console.print()
    console.print("[dim]Note: The campaign config must match the interrupted campaign.[/dim]")


__all__ = ["resume_cmd"]
