#!/usr/bin/env python3
"""Generate curation docs showing discovered vs curated fields per engine.

For each engine, produces a Markdown file with a four-column table showing
every discovered parameter and whether llem's Pydantic layer curates it.

The header surfaces a delta-vs-previous block. The previous SSOT version
is read from ``engine_versions/{engine}.yaml``'s
``last_probe.version_inside_envelope`` field if present; on a fresh SSOT
where ``last_probe.verdict`` is ``unrun`` the block degrades gracefully
to a "deferred until first probe-pass cycle" placeholder. HEAD~1 is
deliberately NOT consulted because it can be a bot writeback commit,
which would make the comparison anchor non-deterministic.

Output: docs/reference/engines/curation-{engine}.md

Run: python scripts/generate_curation_doc.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from llenergymeasure.config.introspection import get_engine_params  # noqa: E402
from llenergymeasure.config.schema_loader import SchemaLoader  # noqa: E402
from llenergymeasure.config.ssot import Engine  # noqa: E402
from scripts.engine_miners._ssot import ssot_path  # noqa: E402

ENGINES = tuple(e.value for e in Engine)

_ENGINE_DISPLAY_NAMES = {
    "transformers": "Transformers",
    "vllm": "vLLM",
    "tensorrt": "TensorRT-LLM",
}


def _get_curated_names(engine: str) -> set[str]:
    """Get the set of leaf field names curated by llem for this engine."""
    params = get_engine_params(engine)
    return {meta["name"] for meta in params.values()}


def _read_previous_ssot_version(engine: str) -> str | None:
    """Return the previous library version recorded in the SSOT, or ``None``.

    Reads ``engine_versions/{engine}.yaml`` and returns the value of
    ``last_probe.version_inside_envelope`` only when (a) the SSOT file
    exists, (b) ``last_probe.verdict`` is one of the post-run verdicts
    (``pass``/``fail``), and (c) the recorded version differs from the
    current one. On any other shape (``unrun``, missing, malformed) the
    delta block degrades to a deferred placeholder rather than fabricate
    a comparison.
    """
    path = ssot_path(engine)
    if not path.is_file():
        return None
    try:
        data: Any = yaml.safe_load(path.read_text())
    except yaml.YAMLError:
        return None
    if not isinstance(data, dict):
        return None
    last_probe = data.get("last_probe")
    if not isinstance(last_probe, dict):
        return None
    if last_probe.get("verdict") not in {"pass", "fail"}:
        return None
    version = last_probe.get("version_inside_envelope")
    if not isinstance(version, str) or not version.strip():
        return None
    return version


def _render_delta_block(engine: str, current_version: str) -> list[str]:
    """Render the "Delta vs previous" header block.

    On a fresh SSOT (``last_probe.verdict: unrun``) this degrades to an
    honest placeholder rather than fabricating a comparison from
    HEAD~1 (which can be a bot writeback commit).
    """
    previous = _read_previous_ssot_version(engine)
    if previous is None:
        return [
            "**Delta vs previous:** _deferred until first probe-pass cycle._",
            "",
        ]
    if previous == current_version:
        return [
            f"**Delta vs previous:** _no version change since `{previous}`._",
            "",
        ]
    return [
        f"**Delta vs previous:** `{previous}` -> `{current_version}` "
        "(field-level diff lands once probe-pass cycles populate the previous-snapshot cache).",
        "",
    ]


def _render_section(
    section_name: str,
    fields: dict[str, dict],
    curated: set[str],
) -> list[str]:
    """Render a table for one section (engine_params or sampling_params)."""
    lines = [
        f"### {section_name}",
        "",
        "| Field | Type | Default | Curated? |",
        "|-------|------|---------|----------|",
    ]
    for name, meta in sorted(fields.items()):
        type_str = meta.get("type", "unknown").replace("|", "\\|")
        default = meta.get("default")
        default_str = f"`{default}`" if default is not None else "`null`"
        is_curated = "yes" if name in curated else "-"
        lines.append(f"| `{name}` | {type_str} | {default_str} | {is_curated} |")
    lines.append("")
    return lines


def generate_engine_doc(engine: str, loader: SchemaLoader | None = None) -> str:
    """Generate curation doc for one engine."""
    if loader is None:
        loader = SchemaLoader()
    schema = loader.load_schema(engine)
    curated = _get_curated_names(engine)

    engine_total = len(schema.engine_params)
    sampling_total = len(schema.sampling_params)
    all_discovered = {**schema.engine_params, **schema.sampling_params}
    curated_count = sum(1 for name in all_discovered if name in curated)
    total = engine_total + sampling_total

    display_name = _ENGINE_DISPLAY_NAMES.get(engine, engine.title())
    lines = [
        f"# {display_name} Parameter Curation",
        "",
        "<!-- Auto-generated by scripts/generate_curation_doc.py -- do not edit manually -->",
        "",
        f"Engine version: **{schema.engine_version}**  ",
        f"Discovered at: {schema.discovered_at}  ",
        f"Discovery method: {schema.discovery_method}",
        "",
        f"**Summary:** {curated_count}/{total} parameters curated "
        f"({engine_total} engine + {sampling_total} sampling discovered)",
        "",
    ]
    lines.extend(_render_delta_block(engine, schema.engine_version))

    if schema.engine_params:
        lines.extend(_render_section("Engine Parameters", schema.engine_params, curated))

    if schema.sampling_params:
        lines.extend(_render_section("Sampling Parameters", schema.sampling_params, curated))

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Generate curation docs for one engine or all engines."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--engine",
        choices=ENGINES,
        help="Generate the digest for one engine. Omit to render all three.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output Markdown path. Requires --engine; omit for default location.",
    )
    args = parser.parse_args(argv)

    if args.out and not args.engine:
        parser.error("--out requires --engine (single-engine mode)")

    default_dir = _PROJECT_ROOT / "docs" / "reference" / "engines"
    loader = SchemaLoader()

    targets = (args.engine,) if args.engine else ENGINES
    for engine in targets:
        content = generate_engine_doc(engine, loader)
        output_path = args.out if args.out else default_dir / f"curation-{engine}.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)
        print(f"Generated: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
