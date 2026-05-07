# Sphinx config for the Python API reference renderer.
#
# Produces a single Docusaurus-compatible markdown page at
# docs/api/llenergymeasure.md by walking llenergymeasure.__all__ via
# autodoc + autodoc_pydantic, and converting the Sphinx output to
# markdown via sphinx-markdown-builder.
#
# Invoked indirectly via scripts/generate_api_docs.py, which runs
# `sphinx-build -b markdown -c docs/sphinx docs/sphinx <build-dir>`
# and then post-processes the result to land at the historical
# docs/api/llenergymeasure.md path the Docusaurus `api` plugin expects.

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# Project metadata. Sphinx reads __version__ from the package directly
# so the rendered "release" line stays in sync with pyproject.toml +
# src/llenergymeasure/_version.py.
import llenergymeasure  # noqa: E402

project = "LLenergyMeasure"
author = "LLenergyMeasure contributors"
release = llenergymeasure.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinxcontrib.autodoc_pydantic",
    "sphinx_markdown_builder",
]

# Autodoc configuration.
# - "members" pulls in everything; the index.rst :members: list filters
#   down to llenergymeasure.__all__ for the top-level module.
# - "show-inheritance" surfaces base classes (BaseModel for Pydantic
#   configs), useful for navigating Pydantic v2 surfaces.
# - typehints render in a separate description block, not inline in
#   the signature heading. Two reasons: (1) sphinx-markdown-builder
#   drops the bare `*` keyword-only separator when serialising a
#   signature into a heading, leaving a stray comma between positional
#   and keyword-only args (e.g. `(config = None, , model = None)`).
#   (2) `**kwargs` gets emphasis-escaped to `\*\*kwargs` because
#   markdown's `**` is bold. Pulling types into a structured Parameters
#   list below the heading sidesteps both rendering quirks - types
#   render correctly per-parameter, and the heading reduces to a clean
#   `func(config=None, *, model=None, ...)`.
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_typehints_description_target = "documented_params"
autodoc_member_order = "bysource"

# autodoc_pydantic: render Pydantic v2 model fields with the metadata
# the existing stdlib renderer surfaces (defaults, descriptions, types).
# Suppress the high-noise blocks (config table, validator hyperlinks,
# JSON schema collapsibles) that don't translate cleanly to flat
# Docusaurus markdown.
autodoc_pydantic_model_show_json = False
autodoc_pydantic_model_show_config_summary = False
autodoc_pydantic_model_show_validator_summary = False
autodoc_pydantic_model_show_validator_members = False
autodoc_pydantic_model_show_field_summary = False
autodoc_pydantic_model_member_order = "bysource"
autodoc_pydantic_field_list_validators = False
autodoc_pydantic_field_show_constraints = True
autodoc_pydantic_field_show_default = True

# Napoleon: docstrings in this codebase mix Google and reST styles;
# enabling both lets Sphinx parse either without manual rewriting.
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# Suppress warnings that don't matter for a single-page reference: the
# project doesn't use cross-references between pages, so missing anchor
# warnings would be noise rather than signal.
suppress_warnings = ["myst.xref_missing", "ref.python"]

# sphinx-markdown-builder: emit GitHub-flavoured markdown so Docusaurus
# (which reads CommonMark + GFM tables) parses field tables correctly.
markdown_anchor_sections = False
markdown_anchor_signatures = False
markdown_docinfo = False

# Disable Sphinx's "smart" typography: Docusaurus pages must contain
# only ASCII punctuation per project convention (no em-dashes,
# en-dashes, or curly quotes in committed markdown). Sphinx's default
# smartquotes transform converts ASCII -- to en-dash and "" to curly
# quotes; turning it off keeps the rendered docstrings byte-identical
# to the source.
smartquotes = False


import re


# autodoc-process-signature hook: scrub Markdown-writer rendering
# artefacts before the signature reaches the heading.
#
# sphinx-markdown-builder serialises signatures into Markdown
# headings without escaping asterisks the way reST/HTML output does:
#   - the bare `*` keyword-only separator gets dropped, leaving
#     a stray `, ,` between positional and keyword-only args
#   - `**kwargs` becomes `\*\*kwargs` (visible escape) because
#     `**` is Markdown bold
#   - `*args` similarly mangles
#
# The cleanest fix is to strip these markers from the heading
# entirely - the per-parameter description block below the heading
# already documents which parameters are keyword-only, varargs,
# or **kwargs via prose, so heading omission is lossless.
def _scrub_signature(app, what, name, obj, options, signature, return_annotation):
    if signature is None:
        return None
    cleaned = re.sub(r", \*,", ",", signature)  # keyword-only separator
    cleaned = re.sub(r"\*\*(\w+)", r"\1", cleaned)  # **kwargs -> kwargs
    cleaned = re.sub(r"(?<![*\w])\*(\w+)", r"\1", cleaned)  # *args -> args
    return (cleaned, return_annotation)


def setup(app):
    app.connect("autodoc-process-signature", _scrub_signature)
