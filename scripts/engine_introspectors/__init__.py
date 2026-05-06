"""Per-engine schema introspectors.

Runs inside an environment where the target engine package is installed
(typically a Docker container). For each engine, introspects the native
Python API surface and writes a JSON schema file with a common envelope.

Expected discovery targets:
    vllm         -> inside vllm/vllm-openai:<tag>
    tensorrt     -> inside nvcr.io/nvidia/tensorrt-llm/release:<tag>
    transformers -> inside llenergymeasure:transformers

Usage::

    python -m scripts.engine_introspectors --engine vllm
    python -m scripts.engine_introspectors --engine tensorrt
    python -m scripts.engine_introspectors --engine transformers
    python -m scripts.engine_introspectors --all
    python -m scripts.engine_introspectors --engine vllm --output /tmp/vllm.json
    python -m scripts.engine_introspectors --engine vllm --image-ref vllm/vllm-openai:v0.7.3
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from scripts.engine_introspectors._common import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SCHEMA_FILENAME,
    jsonable,
)
from scripts.engine_introspectors.tensorrt_introspector import discover as discover_tensorrt
from scripts.engine_introspectors.transformers_introspector import (
    discover as discover_transformers,
)
from scripts.engine_introspectors.vllm_introspector import discover as discover_vllm

DiscoveryFn = Callable[[Path, str | None], dict[str, Any]]

DISCOVERY_FUNCTIONS: dict[str, DiscoveryFn] = {
    "vllm": discover_vllm,
    "tensorrt": discover_tensorrt,
    "transformers": discover_transformers,
}


def _resolve_output_path(
    *, engine: str, output_arg: Path | None, multi: bool, repo_root: Path
) -> Path:
    if output_arg is None:
        return repo_root / DEFAULT_OUTPUT_DIR / engine / DEFAULT_SCHEMA_FILENAME
    if multi or output_arg.is_dir() or output_arg.suffix == "":
        return output_arg / engine / DEFAULT_SCHEMA_FILENAME
    return output_arg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Discover engine parameter schemas and write discovered JSON files."
    )
    parser.add_argument(
        "--engine",
        action="append",
        choices=list(DISCOVERY_FUNCTIONS),
        default=None,
        help="Engine to introspect (vllm, tensorrt, transformers). May be passed "
        "multiple times. Omit when using --all.",
    )
    parser.add_argument("--all", action="store_true", help="Introspect all known engines.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file (single engine) or directory (multiple engines). "
        f"Default: {DEFAULT_OUTPUT_DIR}/<engine>/{DEFAULT_SCHEMA_FILENAME} relative to repo root.",
    )
    parser.add_argument(
        "--image-ref",
        default=None,
        help="Image reference to record in envelope.image_ref. Defaults to the "
        "Dockerfile FROM tag (also recorded as base_image_ref).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repo root (for Dockerfile lookup). Defaults to the parent of the scripts/ directory.",
    )

    args = parser.parse_args(argv)
    requested: list[str] = list(DISCOVERY_FUNCTIONS) if args.all else (args.engine or [])
    if not requested:
        parser.error("Specify --engine at least once, or use --all.")

    repo_root = args.repo_root or Path(__file__).resolve().parents[2]

    succeeded: list[str] = []
    failed: list[tuple[str, str]] = []

    for engine in requested:
        try:
            envelope = DISCOVERY_FUNCTIONS[engine](repo_root, args.image_ref)
        except ImportError as exc:
            print(f"[{engine}] SKIPPED (not importable): {exc}", file=sys.stderr)
            failed.append((engine, "not importable"))
            continue
        except Exception as exc:
            print(f"[{engine}] FAILED: {exc!r}", file=sys.stderr)
            failed.append((engine, repr(exc)))
            continue

        out_path = _resolve_output_path(
            engine=engine,
            output_arg=args.output,
            multi=len(requested) > 1,
            repo_root=repo_root,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(envelope, indent=2, sort_keys=False, default=jsonable) + "\n"
        )
        print(
            f"[{engine}] wrote {out_path} "
            f"(version={envelope['engine_version']}, "
            f"engine_params={len(envelope['engine_params'])}, "
            f"sampling_params={len(envelope['sampling_params'])})"
        )
        succeeded.append(engine)

    if not succeeded:
        print(f"\nAll engines failed: {failed}", file=sys.stderr)
        return 1
    if failed:
        print(
            f"\nPartial success: {succeeded} ok, {[e for e, _ in failed]} skipped/failed.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
