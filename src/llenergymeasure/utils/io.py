"""Shared filesystem IO helpers (Layer 0 - importable from any layer)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: Path | str) -> Any:
    """Read and parse a UTF-8 JSON file.

    Lets ``OSError`` (missing/unreadable file) and ``json.JSONDecodeError``
    (malformed content) propagate to the caller; callers that tolerate either
    keep their own try/except.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))
