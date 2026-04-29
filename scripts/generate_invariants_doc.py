#!/usr/bin/env python3
"""Generate the invariants digest doc for one engine.

Produces ``docs/generated/invariants-{engine}.md`` from the corpus + vendor
artefacts under ``configs/engine_invariants/`` and the previous corpus state
in git history. Per the engine-coupling design doc §6, this digest is
section 2 of the per-engine curation digest (sections 1 + 3 are produced
by ``generate_curation_doc.py`` + an on-demand runtime-gaps renderer).

Run::

    python scripts/generate_invariants_doc.py --engine transformers \\
        --out docs/generated/invariants-transformers.md
"""

from __future__ import annotations

import argparse
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

_ENGINE_DISPLAY_NAMES = {
    "transformers": "Transformers",
    "vllm": "vLLM",
    "tensorrt": "TensorRT-LLM",
}

_ADDED_BY_DISPLAY = {
    "static_miner": "Static miner (AST analysis)",
    "dynamic_miner": "Dynamic miner (constructor probing)",
    "pydantic_lift": "Pydantic field lift",
    "msgspec_lift": "msgspec field lift",
    "dataclass_lift": "dataclass field lift",
    "runtime_warning": "Runtime warning observation",
}

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict[str, Any]:
    """Read + parse a YAML file; return ``{}`` if missing or empty."""
    if not path.exists():
        return {}
    text = path.read_text()
    if not text.strip():
        return {}
    data = yaml.safe_load(text)
    return data if isinstance(data, dict) else {}


def _previous_proposed(engine: str) -> dict[str, Any]:
    """Read the previous-revision corpus from ``HEAD~1`` for delta computation.

    Returns ``{}`` on any failure (no previous version, malformed YAML,
    not a git checkout). Deltas degrade gracefully — a fresh corpus
    reports every rule as "added".
    """
    rel_path = f"configs/engine_invariants/{engine}.proposed.yaml"
    try:
        text = subprocess.check_output(
            ["git", "show", f"HEAD~1:{rel_path}"],
            cwd=_PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except subprocess.CalledProcessError:
        return {}
    if not text.strip():
        return {}
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def _rule_summary(rule: dict[str, Any]) -> str:
    """One-line summary of a single rule for the digest list."""
    rule_id = rule.get("id", "<unknown>")
    severity = rule.get("severity", "?")
    under_test = rule.get("rule_under_test") or rule.get("message_template") or ""
    under_test = str(under_test).replace("\n", " ").strip()
    if len(under_test) > 100:
        under_test = under_test[:97] + "..."
    return f"- `{rule_id}` [{severity}] — {under_test}"


def _group_by_added_by(rules: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group rules by the ``added_by`` field, preserving rule order within groups."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rule in rules:
        key = str(rule.get("added_by") or "unknown")
        groups[key].append(rule)
    return dict(groups)


def _delta_section(current_ids: set[str], previous_ids: set[str]) -> list[str]:
    """Render the added-vs-removed rule-ID delta against the previous corpus."""
    added = sorted(current_ids - previous_ids)
    removed = sorted(previous_ids - current_ids)
    lines: list[str] = ["## Delta vs previous SSOT version", ""]
    if not previous_ids:
        lines.append("_No previous corpus on `HEAD~1`; treating every rule as new._")
        lines.append("")
        return lines
    if not added and not removed:
        lines.append("_No invariants added or removed since the previous revision._")
        lines.append("")
        return lines
    if added:
        lines.append(f"**Added ({len(added)}):**")
        lines.append("")
        for rule_id in added:
            lines.append(f"- `{rule_id}`")
        lines.append("")
    if removed:
        lines.append(f"**Removed ({len(removed)}):**")
        lines.append("")
        for rule_id in removed:
            lines.append(f"- `{rule_id}`")
        lines.append("")
    return lines


def _render(engine: str) -> str:
    """Build the digest Markdown for ``engine``."""
    proposed = _load_yaml(
        _PROJECT_ROOT / "configs" / "engine_invariants" / f"{engine}.proposed.yaml"
    )
    vendored = _load_yaml(
        _PROJECT_ROOT / "configs" / "engine_invariants" / f"{engine}.vendored.yaml"
    )
    ssot = _load_yaml(_PROJECT_ROOT / "engine_versions" / f"{engine}.yaml")

    library = ssot.get("library") if isinstance(ssot.get("library"), dict) else {}
    library_version = str(library.get("current_version", "<unknown>"))
    proposed_rules = proposed.get("rules") or []
    if not isinstance(proposed_rules, list):
        proposed_rules = []
    vendored_cases = vendored.get("cases") or []
    if not isinstance(vendored_cases, list):
        vendored_cases = []

    previous = _previous_proposed(engine)
    previous_rules = previous.get("rules") or []
    previous_ids = {
        str(r.get("id")) for r in previous_rules if isinstance(r, dict) and r.get("id") is not None
    }
    current_ids = {
        str(r.get("id")) for r in proposed_rules if isinstance(r, dict) and r.get("id") is not None
    }

    display = _ENGINE_DISPLAY_NAMES.get(engine, engine.title())
    lines: list[str] = [
        f"# {display} Engine Invariants",
        "",
        "<!-- Auto-generated by scripts/generate_invariants_doc.py -- do not edit manually -->",
        "",
        f"Library version: **{library_version}**  ",
        f"Mined at: {proposed.get('mined_at', '<unknown>')}  ",
        f"Vendored at: {vendored.get('vendored_at', '<unknown>')}",
        "",
        f"**Summary:** {len(proposed_rules)} proposed rules, "
        f"{len(vendored_cases)} vendor-confirmed cases.",
        "",
    ]

    lines.append("## Invariants by extraction source")
    lines.append("")
    groups = _group_by_added_by(proposed_rules)
    if not groups:
        lines.append("_No invariants in the proposed corpus._")
        lines.append("")
    else:
        for added_by in sorted(groups):
            display_label = _ADDED_BY_DISPLAY.get(added_by, added_by)
            rules = groups[added_by]
            lines.append(f"### {display_label} ({len(rules)})")
            lines.append("")
            for rule in rules:
                lines.append(_rule_summary(rule))
            lines.append("")

    lines.extend(_delta_section(current_ids, previous_ids))
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--engine",
        required=True,
        choices=tuple(_ENGINE_DISPLAY_NAMES),
        help="Engine name.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output Markdown path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    text = _render(args.engine)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
