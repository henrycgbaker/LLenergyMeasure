"""Tests for scripts/generate_config_docs.py section labels staying schema-accurate."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3] / "scripts"))
sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

from generate_config_docs import _DEF_TO_SECTION, _SECTION_ORDER, render_markdown

from llenergymeasure.config.models import ExperimentConfig


def test_section_order_uses_transformers_key_not_pytorch() -> None:
    """The engine section label must reflect the real `transformers:` YAML key, not pytorch."""
    titles = [title for _, title in _SECTION_ORDER]
    assert any("transformers" in t.lower() for t in titles)
    assert not any("pytorch" in t.lower() for t in titles)


def test_section_order_drops_defunct_decoder_section() -> None:
    """DecoderConfig/decoder: was removed; no section may reference it."""
    keys = [key for key, _ in _SECTION_ORDER]
    titles = [title for _, title in _SECTION_ORDER]
    assert "decoder" not in keys
    assert not any("decoder" in t.lower() for t in titles)


def test_def_to_section_drops_decoder_config() -> None:
    """DecoderConfig is no longer a real schema $def and must not be mapped."""
    assert "DecoderConfig" not in _DEF_TO_SECTION


def test_rendered_doc_has_no_pytorch_or_decoder_labels() -> None:
    """End-to-end render must not emit the stale pytorch:/decoder: labels."""
    markdown = render_markdown(ExperimentConfig.model_json_schema())
    assert "`pytorch:`" not in markdown
    assert "`decoder:`" not in markdown
    assert "Transformers Engine (`transformers.engine_params:`)" in markdown
