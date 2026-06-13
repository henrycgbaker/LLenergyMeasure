"""``llem diagnose-bump`` - maintainer-invokable Stage-1 LLM bump diagnose.

Thin Typer command that delegates everything interesting to
:mod:`llenergymeasure.api.diagnose`. CLI-boundary contract: imports only from
``llenergymeasure.api``; never reaches into ``study``/``harness``/etc.

Runs on the maintainer's box (local GPU + local ollama, same venue as the vllm
gate cells) and proposes GATE-CONFIRMED rule entries for review. Two modes:

- ``--mode carried`` (1a): triage the decay alarm's post-Rung-0 RESIDUAL.
  Reads a re-gate report (``scripts/validate_rules.py --carried`` output, with a
  ``reconciliation.residual`` block), the previous pin's carried corpus, and the
  diff-scoped new-pin source for the files the residual rules cite.
- ``--mode gap`` (1b): flag what mining missed. Reads the fresh
  ``schema.discovered.json`` and the new-pin config-surface source.

The model proposes; the EXISTING gate (run inside the engine container)
confirms. Only gate-confirmed entries are written to ``--out`` with provenance
``llm_diagnose``; everything else is reported to stderr, never written.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml

from llenergymeasure.api import (
    DiagnoseError,
    DiagnoseResult,
    diagnose_carried_failures,
    diagnose_gaps,
)

__all__ = ["diagnose_bump_cmd"]


def _load_yaml(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise typer.BadParameter(f"{label} not found: {path}")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise typer.BadParameter(f"{label} is not a mapping: {path}")
    return data


def _residual_ids_from_report(report: dict[str, Any]) -> list[str]:
    """Pull the post-Rung-0 residual ids from a --carried re-gate report."""
    recon = report.get("reconciliation")
    if not isinstance(recon, dict):
        raise typer.BadParameter(
            "re-gate report has no 'reconciliation' block; run validate_rules "
            "--carried with --fresh-validated/--fresh-proposed first (Rung 0)."
        )
    residual = recon.get("residual") or []
    return [str(e.get("id", "")) for e in residual if isinstance(e, dict) and e.get("id")]


def _print_report(result: DiagnoseResult, out: Path | None) -> None:
    """Summarise the run to stderr; the YAML (confirmed only) goes to --out."""
    print(
        f"[{result.engine} {result.engine_version}] diagnose ({result.mode}): "
        f"{len(result.confirmed)} gate-confirmed, "
        f"{len(result.unconfirmed)} unconfirmed (not written), "
        f"{len(result.not_construction_confirmable)} for review (silent dormancy), "
        f"{len(result.dropped_malformed)} dropped (malformed kwargs)",
        file=sys.stderr,
    )
    for rec in result.not_construction_confirmable:
        print(f"  review (silent dormancy): {rec['rule_id']}", file=sys.stderr)
    for rec in result.dropped_malformed:
        print(f"  dropped (malformed kwargs): {rec['rule_id']} {rec['malformed']}", file=sys.stderr)

    if result.confirmed and result.proposed_yaml is not None:
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(result.proposed_yaml)
            print(
                f"Wrote {len(result.confirmed)} gate-confirmed entry(ies) "
                f"(added_by: llm_diagnose) to {out}.",
                file=sys.stderr,
            )
        else:
            print(result.proposed_yaml)
    else:
        print("No gate-confirmed entries; nothing written.", file=sys.stderr)


def diagnose_bump_cmd(
    engine: Annotated[
        str,
        typer.Option("--engine", help="Engine to diagnose (transformers/vllm/tensorrt)."),
    ],
    image: Annotated[
        str,
        typer.Option("--image", help="Engine cache image to run the gate inside (new pin)."),
    ],
    new_version: Annotated[
        str,
        typer.Option("--new-version", help="The new (current) engine version being diagnosed."),
    ],
    source_root: Annotated[
        Path,
        typer.Option("--source-root", help="Root of the new pin's engine source tree."),
    ],
    mode: Annotated[
        str,
        typer.Option("--mode", help="'carried' (1a residual triage) or 'gap' (1b gap-diagnose)."),
    ] = "carried",
    # Mode 1a inputs.
    regate_report: Annotated[
        Path | None,
        typer.Option(
            "--regate-report",
            help="[carried] validate_rules --carried JSON report (with reconciliation block).",
        ),
    ] = None,
    carried_corpus: Annotated[
        Path | None,
        typer.Option(
            "--carried-corpus",
            help="[carried] previous pin's rules.proposed.yaml (the executable carried catalogue).",
        ),
    ] = None,
    old_version: Annotated[
        str | None,
        typer.Option("--old-version", help="[carried] the previous engine version."),
    ] = None,
    cited_file: Annotated[
        list[str] | None,
        typer.Option(
            "--cited-file",
            help="[carried] source file (relative to --source-root) the residual rules cite. Repeatable.",
        ),
    ] = None,
    # Mode 1b inputs.
    schema: Annotated[
        Path | None,
        typer.Option("--schema", help="[gap] the fresh schema.discovered.json."),
    ] = None,
    surface_file: Annotated[
        list[str] | None,
        typer.Option(
            "--surface-file",
            help="[gap] config-surface source file (relative to --source-root). Repeatable.",
        ),
    ] = None,
    model_name: Annotated[
        str,
        typer.Option("--model", help="Local ollama model to use as the proposer."),
    ] = "qwen2.5-coder:32b",
    out: Annotated[
        Path | None,
        typer.Option(
            "--out", help="Where to write gate-confirmed rules.proposed.yaml (else stdout)."
        ),
    ] = None,
    workdir: Annotated[
        Path,
        typer.Option("--workdir", help="Scratch dir for gate proposal/verdict JSON."),
    ] = Path("/tmp/llem-diagnose"),
) -> None:
    """Propose gate-confirmed rule entries from an upstream bump (local GPU)."""
    # Deferred: live model + container gate are only constructed when invoked, so
    # --help stays import-light and the heavy paths never load in tests.
    from llenergymeasure.api.diagnose import ContainerGateRunner, OllamaDiagnoseModel

    repo = Path.cwd()
    runner = ContainerGateRunner(image=image, repo=repo, workdir=workdir)
    proposer = OllamaDiagnoseModel(model=model_name)

    try:
        if mode == "carried":
            if regate_report is None or carried_corpus is None or old_version is None:
                raise typer.BadParameter(
                    "--mode carried requires --regate-report, --carried-corpus, --old-version."
                )
            if not cited_file:
                raise typer.BadParameter("--mode carried requires at least one --cited-file.")
            report = json.loads(regate_report.read_text())
            residual_ids = _residual_ids_from_report(report)
            if not residual_ids:
                print(
                    "Reconciliation left no residual; Rung 0 healed everything. "
                    "Nothing to diagnose (1a is dormant on this bump).",
                    file=sys.stderr,
                )
                return
            corpus = _load_yaml(carried_corpus, "--carried-corpus")
            result = diagnose_carried_failures(
                engine=engine,
                old_version=old_version,
                new_version=new_version,
                residual_ids=residual_ids,
                carried_corpus=corpus,
                source_root=source_root,
                cited_files=cited_file,
                model=proposer,
                gate_runner=runner,
            )
        elif mode == "gap":
            if schema is None:
                raise typer.BadParameter("--mode gap requires --schema.")
            if not surface_file:
                raise typer.BadParameter("--mode gap requires at least one --surface-file.")
            schema_data = json.loads(schema.read_text())
            result = diagnose_gaps(
                engine=engine,
                new_version=new_version,
                schema=schema_data,
                source_root=source_root,
                config_surface_files=surface_file,
                model=proposer,
                gate_runner=runner,
            )
        else:
            raise typer.BadParameter(f"--mode must be 'carried' or 'gap', got {mode!r}.")
    except DiagnoseError as exc:
        print(f"llem diagnose-bump: {exc}", file=sys.stderr)
        raise typer.Exit(code=2) from None

    _print_report(result, out)
