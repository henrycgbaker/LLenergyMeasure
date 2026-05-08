"""Static-miner LANDMARKS for Transformers 4.57.3.

Read by the probe before the invariants miner orchestrator
(``scripts.engine_miners.transformers_miner``) runs. Covers the symbols
the orchestrator's ``_check_landmarks()`` verifies plus the symbols the
delegated dynamic miner (``transformers_dynamic_miner``) imports.
"""

from __future__ import annotations

LANDMARKS: tuple[str, ...] = (
    "transformers.GenerationConfig",
    "transformers.GenerationConfig.validate",
    "transformers.BitsAndBytesConfig",
    "transformers.BitsAndBytesConfig.post_init",
)
