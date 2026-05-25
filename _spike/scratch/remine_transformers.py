"""Re-mine transformers v4.57.3 schema with the Move 1 walker enabled.

Calls the per-version introspector's ``discover()`` against the locally
installed ``transformers==4.57.3`` + ``torch==2.5.1+cpu``, then writes
the resulting envelope to both the archive (engine_versions) and the
shadow (src/<e>/) atomically.

Spike-only. The supported way to re-mine is the cells workflow; this
script just lets us iterate locally during Phase 2-T Move-1 deepening
without spinning up the container.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from engine_versions.transformers.v4_57_3.producers.schema_introspector import (
    discover,
)

ARCHIVE = (
    _PROJECT_ROOT
    / "engine_versions"
    / "transformers"
    / "v4_57_3"
    / "outputs"
    / "schema.discovered.json"
)
SHADOW = (
    _PROJECT_ROOT
    / "src"
    / "llenergymeasure"
    / "engines"
    / "transformers"
    / "schema.discovered.json"
)


def main() -> int:
    print("calling discover()...", file=sys.stderr)
    env = discover(_PROJECT_ROOT, "llenergymeasure:transformers-4.57.3")
    n_engine = len(env.get("engine_params", {}))
    n_sampling = len(env.get("sampling_params", {}))
    n_limitations = len(env.get("discovery_limitations", []))
    print(
        f"  engine_params={n_engine}, sampling_params={n_sampling}, limitations={n_limitations}",
        file=sys.stderr,
    )

    text = json.dumps(env, indent=2) + "\n"
    ARCHIVE.write_text(text, encoding="utf-8")
    print(f"wrote {ARCHIVE}", file=sys.stderr)
    SHADOW.write_text(text, encoding="utf-8")
    print(f"wrote {SHADOW}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
