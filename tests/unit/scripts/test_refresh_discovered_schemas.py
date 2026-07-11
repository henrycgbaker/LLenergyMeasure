"""Guard test: the schema refresh script must not write the src copy directly.

The refresh script's single discovery write target is the versioned outputs/
snapshot; the packaged src copy is written only by ``promote_schemas.py``.
Actually running the script needs a GPU container, so this asserts the write
path structurally by reading the committed script text.
"""

from __future__ import annotations

from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "refresh_discovered_schemas.sh"


def _output_rel_assignments(text: str) -> list[str]:
    """Lines assigning OUTPUT_REL - the discovery (docker --output) write target."""
    return [line for line in text.splitlines() if line.lstrip().startswith("OUTPUT_REL=")]


class TestRefreshWritePath:
    def test_discovery_target_is_not_the_src_copy(self):
        assignments = _output_rel_assignments(SCRIPT.read_text())
        assert assignments, "OUTPUT_REL (the discovery write target) assignment not found"
        for line in assignments:
            assert "src/llenergymeasure" not in line, (
                "discovery must write the versioned outputs/ snapshot, not the src copy"
            )

    def test_discovery_target_derives_from_the_mangling_locus(self):
        # v<safe> must come from the one name-mangling locus, _outputs.py.
        assert "_outputs" in SCRIPT.read_text()

    def test_docker_output_flag_targets_the_snapshot(self):
        assert '--output "/repo/$OUTPUT_REL"' in SCRIPT.read_text()

    def test_promotion_writes_the_src_copy(self):
        assert "promote_schemas.py" in SCRIPT.read_text()
