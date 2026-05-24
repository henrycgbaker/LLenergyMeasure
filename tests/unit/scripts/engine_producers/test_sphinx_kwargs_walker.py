"""Unit tests for ``parse_sphinx_kwargs`` (Move 1 docstring walker).

Lifts ``name (`type`, *optional*[, defaults to <expr>]):`` blocks out
of a class / method docstring. Used by per-engine schema introspectors
to recover ``**kwargs`` params that are documented only in the docstring
(HuggingFace's ``from_pretrained`` being the canonical case).
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.engine_producers._common import parse_sphinx_kwargs  # noqa: E402


class TestParseSphinxKwargsBasics:
    def test_empty_docstring_returns_empty(self) -> None:
        assert parse_sphinx_kwargs("") == {}

    def test_simple_block_extracted(self) -> None:
        doc = textwrap.dedent(
            """
            Description preamble.

            Args:
                temperature (`float`, *optional*, defaults to 1.0):
                    Sampling temperature.
            """
        )
        out = parse_sphinx_kwargs(doc)
        assert "temperature" in out
        assert out["temperature"]["type"] == "number"
        assert out["temperature"]["default"] == 1.0
        assert out["temperature"]["x-source"] == "kwargs_docstring"

    def test_block_without_default(self) -> None:
        doc = textwrap.dedent(
            """
            Args:
                attn_implementation (`str`, *optional*):
                    Attention impl.
            """
        )
        out = parse_sphinx_kwargs(doc)
        assert "attn_implementation" in out
        assert out["attn_implementation"]["type"] == "string"
        assert "default" not in out["attn_implementation"]

    def test_multiple_blocks_all_extracted(self) -> None:
        doc = textwrap.dedent(
            """
            Args:
                dtype (`str`, *optional*, defaults to `auto`):
                    Model dtype.
                use_cache (`bool`, *optional*, defaults to True):
                    Whether to use cache.
                top_k (`int`, *optional*, defaults to 50):
                    Sampling top-k.
            """
        )
        out = parse_sphinx_kwargs(doc)
        assert set(out.keys()) == {"dtype", "use_cache", "top_k"}
        assert out["dtype"]["default"] == "auto"
        assert out["use_cache"]["default"] is True
        assert out["top_k"]["default"] == 50

    def test_type_omitted_for_complex_unions(self) -> None:
        # torch.dtype isn't a recognised primitive - walker omits the
        # type key rather than guess. Caller renders as Any | None.
        doc = textwrap.dedent(
            """
            Args:
                bnb_4bit_compute_dtype (`torch.dtype`, *optional*, defaults to None):
                    Compute dtype.
            """
        )
        out = parse_sphinx_kwargs(doc)
        assert "bnb_4bit_compute_dtype" in out
        assert "type" not in out["bnb_4bit_compute_dtype"]
        assert out["bnb_4bit_compute_dtype"]["default"] is None


class TestParseSphinxKwargsSkipNames:
    def test_skip_names_filters_signature_params(self) -> None:
        doc = textwrap.dedent(
            """
            Args:
                in_signature (`str`, *optional*):
                    Already in inspect.signature.
                kwargs_only (`str`, *optional*):
                    Documented but not in signature.
            """
        )
        out = parse_sphinx_kwargs(doc, skip_names={"in_signature"})
        assert "in_signature" not in out
        assert "kwargs_only" in out

    def test_empty_skip_names_returns_all(self) -> None:
        doc = textwrap.dedent(
            """
            Args:
                a (`int`, *optional*):
                    .
                b (`str`, *optional*):
                    .
            """
        )
        out = parse_sphinx_kwargs(doc, skip_names=set())
        assert set(out.keys()) == {"a", "b"}


class TestParseSphinxKwargsTypeMapping:
    def test_recognises_primitive_python_types(self) -> None:
        doc = textwrap.dedent(
            """
            Args:
                a (`bool`, *optional*):
                    .
                b (`int`, *optional*):
                    .
                c (`float`, *optional*):
                    .
                d (`str`, *optional*):
                    .
                e (`list`, *optional*):
                    .
                f (`dict`, *optional*):
                    .
            """
        )
        out = parse_sphinx_kwargs(doc)
        assert out["a"]["type"] == "boolean"
        assert out["b"]["type"] == "integer"
        assert out["c"]["type"] == "number"
        assert out["d"]["type"] == "string"
        assert out["e"]["type"] == "array"
        assert out["f"]["type"] == "object"

    def test_strips_generic_parameters(self) -> None:
        # dict[str, int] -> dict -> object
        doc = textwrap.dedent(
            """
            Args:
                config (`dict[str, Any]`, *optional*):
                    .
            """
        )
        out = parse_sphinx_kwargs(doc)
        assert out["config"]["type"] == "object"

    def test_strips_dotted_qualifier(self) -> None:
        # typing.dict is hypothetical; collections.OrderedDict more realistic
        doc = textwrap.dedent(
            """
            Args:
                ordered (`collections.OrderedDict`, *optional*):
                    .
            """
        )
        out = parse_sphinx_kwargs(doc)
        # OrderedDict isn't in our map; expect type omitted
        assert "type" not in out["ordered"]


class TestParseSphinxKwargsAltTypes:
    def test_backticked_alternative_types_match(self) -> None:
        # HF often writes `str` or `os.PathLike`
        doc = textwrap.dedent(
            """
            Args:
                cache_dir (`str` or `os.PathLike`, *optional*):
                    .
            """
        )
        out = parse_sphinx_kwargs(doc)
        assert "cache_dir" in out
        assert out["cache_dir"]["type"] == "string"

    def test_bare_alternative_type_matches(self) -> None:
        # BitsAndBytesConfig: `torch.dtype` or str
        doc = textwrap.dedent(
            """
            Args:
                compute_dtype (`torch.dtype` or str, *optional*, defaults to `torch.float32`):
                    .
            """
        )
        out = parse_sphinx_kwargs(doc)
        assert "compute_dtype" in out
        # Primary type (torch.dtype) doesn't map; bare alt is informational
        # only - walker keeps the primary's mapping (omit since untyped).
        assert "type" not in out["compute_dtype"]
        assert out["compute_dtype"]["default"] == "torch.float32"


class TestParseSphinxKwargsProvenance:
    def test_x_source_always_set(self) -> None:
        doc = textwrap.dedent(
            """
            Args:
                foo (`int`, *optional*):
                    .
            """
        )
        out = parse_sphinx_kwargs(doc)
        assert out["foo"]["x-source"] == "kwargs_docstring"

    def test_source_ref_propagates_when_provided(self) -> None:
        doc = textwrap.dedent(
            """
            Args:
                foo (`int`, *optional*):
                    .
            """
        )
        out = parse_sphinx_kwargs(
            doc, source_ref="transformers.PreTrainedModel.from_pretrained.__doc__"
        )
        assert (
            out["foo"]["x-source-ref"]
            == "transformers.PreTrainedModel.from_pretrained.__doc__"
        )

    def test_source_ref_omitted_when_none(self) -> None:
        doc = textwrap.dedent(
            """
            Args:
                foo (`int`, *optional*):
                    .
            """
        )
        out = parse_sphinx_kwargs(doc, source_ref=None)
        assert "x-source-ref" not in out["foo"]


class TestParseSphinxKwargsDefaults:
    def test_quoted_string_default(self) -> None:
        doc = textwrap.dedent(
            """
            Args:
                dtype (`str`, *optional*, defaults to `auto`):
                    .
            """
        )
        out = parse_sphinx_kwargs(doc)
        assert out["dtype"]["default"] == "auto"

    def test_integer_default(self) -> None:
        doc = textwrap.dedent(
            """
            Args:
                top_k (`int`, *optional*, defaults to 50):
                    .
            """
        )
        out = parse_sphinx_kwargs(doc)
        assert out["top_k"]["default"] == 50

    def test_none_default(self) -> None:
        doc = textwrap.dedent(
            """
            Args:
                max_memory (`dict`, *optional*, defaults to None):
                    .
            """
        )
        out = parse_sphinx_kwargs(doc)
        assert out["max_memory"]["default"] is None

    def test_unparseable_default_kept_as_string(self) -> None:
        doc = textwrap.dedent(
            """
            Args:
                policy (`str`, *optional*, defaults to whatever-engine-chooses):
                    .
            """
        )
        out = parse_sphinx_kwargs(doc)
        assert out["policy"]["default"] == "whatever-engine-chooses"
