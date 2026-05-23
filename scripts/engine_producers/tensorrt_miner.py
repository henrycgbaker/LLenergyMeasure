"""TensorRT-LLM miner orchestrator - single-stage (static-only).

Per the locked invariant-miner design (decision #8 of the adversarial
review, 2026-04-26), TRT-LLM ships **without** a dynamic miner: probing
``TrtLlmArgs(...)`` with combinatorial inputs yields near-zero raises
because the constructor is permissive - all real cross-field validity
invariants fire later at engine compile time, inside C++ ``Builder.build_engine``.
The static miner alone covers the surface that matters at config-validation
time.

Pipeline this orchestrator drives:

1. Verify the source tree is present (canonical location:
   ``/tmp/trt-llm-<library.current_version>/tensorrt_llm/``). The TRT-LLM
   library version comes from ``engine_versions/tensorrt/current.toml``;
   the per-version dispatcher
   (:mod:`engine_versions._dispatcher.load_producer`) selects which
   archived static miner runs, falling back to the most-recent prior
   vendored version when no exact-match archive exists at the bumped
   version. Probe verdict (``scripts._drift``) is the runtime gate.
2. Read ``tensorrt_llm/version.py`` from the source tree (no import).
3. Run :mod:`scripts.engine_producers.tensorrt_static_invariant_miner` and emit the
   staging YAML.

This orchestrator never imports ``tensorrt_llm``. The host has 1.1.0
installed and importing it would mine the wrong source - exactly the
silent-degradation failure mode that the Haiku-era extractor PRs (#415,
#416, #417, all reverted in #423) tripped on. AST-walk over a known
extracted source tree is the only safe path.

Usage::

    PYTHONPATH=. python3 scripts/engine_producers/tensorrt_miner.py --out path/to/tensorrt.yaml

Or via the canonical corpus builder::

    PYTHONPATH=. python3 scripts/engine_producers/build_corpus.py --engine tensorrt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# NOTE: this script's parent dir contains a sibling ``transformers_*.py``
# that would shadow the real ``transformers`` package on import. Strip the
# script directory before any third-party imports - same defensive measure
# as the static miner.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if Path(p).resolve() != Path(_SCRIPT_DIR).resolve()]
sys.path[:] = [p for p in sys.path if p != ""]

from scripts.engine_producers.tensorrt_static_invariant_miner import (  # noqa: E402
    _resolve_producer,
)


def main(argv: list[str] | None = None) -> int:
    # Load the per-version archive at entry time (not at module import) so that
    # the miner can be imported on the host without triggering the archive.
    _archive = _resolve_producer()
    _DEFAULT_SOURCE_ROOT = _archive._DEFAULT_SOURCE_ROOT
    emit_yaml = _archive.emit_yaml
    walk_tensorrt = _archive.walk_tensorrt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Write extracted YAML to this path.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=_DEFAULT_SOURCE_ROOT,
        help=(
            "Path to the extracted tensorrt_llm 0.21.0 source tree (default: "
            f"{_DEFAULT_SOURCE_ROOT})"
        ),
    )
    args = parser.parse_args(argv)

    candidates, source_version, rel_path = walk_tensorrt(args.source_root)

    text = emit_yaml(candidates, engine_version=source_version, rel_path=rel_path)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(
        f"Wrote {len(candidates)} tensorrt_llm invariants to {args.out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
