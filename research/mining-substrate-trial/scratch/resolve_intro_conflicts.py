"""Resolve schema_introspector.py merge conflicts.

Rules (per DECISIONS_LOG resume entry, 2026-05-24):
- Empty HEAD vs imports/code in theirs (#666) -> take theirs.
- HEAD has content, theirs contains "discovery_method" -> take ours
  (PR #667 already dropped discovery_method from canonical envelope).
- Otherwise -> leave a marker for manual review.
"""

import re
import sys
from pathlib import Path

CONFLICT_RE = re.compile(
    r"<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> [^\n]+\n",
    re.DOTALL,
)


def resolve(text: str) -> str:
    def replace(m: re.Match[str]) -> str:
        ours, theirs = m.group(1), m.group(2)
        if "discovery_method" in theirs:
            return ours
        if not ours.strip():
            return theirs
        return f"# CONFLICT - MANUAL REVIEW:\n{ours}# vs:\n{theirs}# end-conflict\n"

    return CONFLICT_RE.sub(replace, text)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        path = Path(p)
        original = path.read_text(encoding="utf-8")
        resolved = resolve(original)
        path.write_text(resolved, encoding="utf-8")
        markers_left = resolved.count("<<<<<<<") + resolved.count(">>>>>>>")
        manual = resolved.count("# CONFLICT - MANUAL REVIEW:")
        print(f"{path}: markers_left={markers_left} manual_review={manual}")
