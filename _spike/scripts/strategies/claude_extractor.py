"""Strategy (c) "pure Claude API" executor - contract-compatible stub.

Per Phase 2 plan:
- ANTHROPIC_API_KEY is NOT available in this environment; trial_runner
  catches ``KeyAbsentError`` and skips (c) cells.
- When the key arrives, switching strategy (c) ON requires NO code
  change beyond setting the env var.
- Uses the SAME prompt templates as (b) (so (b)/(c) cells are directly
  comparable; the only axis is model identity).
- Uses Anthropic SDK with prompt caching (cache_control: ephemeral on
  the SOURCE blocks; the cache hit on retries saves 90% input tokens).
- Per-cell cost cap enforced post-hoc by AnthropicBackend.

CONTRACT: ``run_c_on_transformers_active(out_dir, ...)`` mirrors
``run_b_on_transformers_active``. If ANTHROPIC_API_KEY absent, raises
``KeyAbsentError`` immediately on backend instantiation.

For (c) cells: trial_runner can probe by calling the function and
catching ``KeyAbsentError``; the cell record then sets failure_modes=
['key_absent'] and proceeds.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Local imports (sys.path adjusted above; ruff E402 deliberate here).
from _spike.scripts.strategies.llm_b_oss import (  # noqa: E402
    StrategyBOutputs,
    run_b_on_transformers_active,
)
from _spike.scripts.strategies.llm_extractor import (  # noqa: E402
    AnthropicBackend,
    KeyAbsentError,
)

# Re-export so callers can catch the right exception type.
__all__ = ["KeyAbsentError", "run_c_on_transformers_active"]


def run_c_on_transformers_active(
    *,
    out_dir: Path,
    engine_version: str = "4.57.3",
    model: str = "claude-sonnet-4-5",
    max_retries: int = 2,
) -> StrategyBOutputs:
    """Run strategy (c) for transformers active version.

    Raises ``KeyAbsentError`` if ANTHROPIC_API_KEY is not set (the
    trial_runner handles by skipping the cell with a recorded
    observation).

    Same output shape as ``run_b_on_transformers_active`` - the only
    difference is the backend (and consequently the model identity).
    """
    backend = AnthropicBackend(model=model)
    # Reuses (b)'s extract_schema + extract_invariants verbatim via
    # run_b_on_transformers_active's backend injection.
    return run_b_on_transformers_active(
        out_dir=out_dir,
        engine_version=engine_version,
        backend=backend,
        max_retries=max_retries,
    )
