"""Renovate regex regression test.

The customManagers regex in ``renovate.json`` matches the ``pep503_name``
+ ``current_version`` pair on adjacent lines of every
``engine_versions/<engine>/current.toml``. Inserting a comment, blank line,
or new field between those two lines silently breaks Renovate detection
(``\\s*\\n\\s*`` allows whitespace only). This test catches that drift in
PR review by asserting the regex matches every current.toml.

Per the second adversarial pass on PR-0 (FM 5 mitigation).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_regex_pattern() -> str:
    """Read the customManagers regex from renovate.json.

    Renovate uses JavaScript-flavoured regex (named groups written as
    ``(?<name>...)``); Python's ``re`` module requires ``(?P<name>...)``.
    Translate at load time so the test exercises the same line-shape
    requirements against the TOML, just via a Python-compatible engine.
    """
    cfg = json.loads((_REPO_ROOT / "renovate.json").read_text())
    managers = cfg.get("customManagers", [])
    for mgr in managers:
        if mgr.get("customType") == "regex":
            patterns = mgr.get("matchStrings", [])
            if patterns:
                # JS (?<name>) -> Python (?P<name>). Single regex pass since
                # ``)`` -terminated.
                return re.sub(r"\(\?<", "(?P<", str(patterns[0]))
    pytest.fail("No customManagers[].matchStrings entry found in renovate.json")


@pytest.mark.parametrize("engine", ["transformers", "vllm", "tensorrt"])
def test_renovate_regex_matches_current_toml(engine: str) -> None:
    """engine_versions/<engine>/current.toml MUST match Renovate's regex.

    If this fails, Renovate has stopped detecting library bumps for
    ``engine``. Likely cause: a comment / blank line / new field has been
    interleaved between ``pep503_name`` and ``current_version``. The
    regex requires only whitespace between them.
    """
    pattern = _load_regex_pattern()
    text = (_REPO_ROOT / "engine_versions" / engine / "current.toml").read_text()
    match = re.search(pattern, text)
    assert match is not None, (
        f"engine_versions/{engine}/current.toml does not match Renovate's pep503_name+"
        f"current_version regex. Did someone interleave a comment/blank line "
        f"between those two fields? Renovate's regex is `\\s*\\n\\s*` between "
        f"fields - strict whitespace only."
    )
    # depName + currentValue must both have been captured.
    assert match.group("depName"), (
        f"Renovate regex matched {engine}/current.toml but did not capture depName."
    )
    assert match.group("currentValue"), (
        f"Renovate regex matched {engine}/current.toml but did not capture currentValue."
    )
