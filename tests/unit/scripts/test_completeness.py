"""Tests for the drift-completeness substrate (scripts.engine_producers._completeness).

Covers the raise-validator detector (widened naming conventions + inherited /
no-source surfacing) and the uncovered-computation's same-class helper-follow
coverage model (a raise the miner reaches via ``self.helper()`` from a covered
method must NOT be flagged as a gap).
"""

from __future__ import annotations

import ast
import dataclasses
import textwrap

from scripts.engine_producers import _completeness


# Module-scope synthetic config classes (inspect.getsource needs a real source).
@dataclasses.dataclass
class _Leaf:
    """A flat-hoisted config: __post_init__ delegates to a non-conventionally
    named ``verify_*`` helper (no leading underscore) that raises."""

    alpha: int = 0
    beta: int = 0
    gamma: int = 0

    def __post_init__(self) -> None:
        self.verify_relation()

    def verify_relation(self) -> None:
        if self.alpha > self.beta:
            raise ValueError("alpha must not exceed beta")


@dataclasses.dataclass
class _Unreached:
    """A flat-hoisted config whose raise-validator is NOT called from any
    covered method (mirrors ModelConfig.verify_with_parallel_config)."""

    alpha: int = 0
    beta: int = 0
    gamma: int = 0

    def __post_init__(self) -> None:  # does NOT call the validator below
        pass

    def verify_external(self) -> None:
        if self.alpha != self.beta:
            raise ValueError("alpha must equal beta")


class _Parent:
    def _validate_inherited(self) -> None:
        if self.foo:  # type: ignore[attr-defined]
            raise ValueError("inherited validator")


@dataclasses.dataclass
class _Child(_Parent):
    foo: int = 0
    bar: int = 0
    baz: int = 0


_HOIST = {"alpha", "beta", "gamma", "foo", "bar", "baz"}


def test_widened_filter_detects_bare_verify_prefix() -> None:
    """A ``verify_*`` method with no leading underscore is detected (not dropped)."""
    raisers, unanalyzable = _completeness.find_raise_validators(_Leaf)
    names = {m for m, _ in raisers}
    assert "verify_relation" in names
    assert unanalyzable == []


def test_inherited_validator_surfaced_unanalyzable() -> None:
    """A validator defined on a PARENT is surfaced to unanalyzable on the child,
    never silently dropped (the false-negative guard)."""
    raisers, unanalyzable = _completeness.find_raise_validators(_Child)
    assert all(m != "_validate_inherited" for m, _ in raisers)
    assert "_validate_inherited" in unanalyzable
    # ...and analyzable on the parent itself.
    parent_raisers, _ = _completeness.find_raise_validators(_Parent)
    assert any(m == "_validate_inherited" for m, _ in parent_raisers)


def test_helper_follow_covers_reached_validator() -> None:
    """A validator reached via self.helper() from a COVERED method is treated as
    covered (mirrors the miner's depth follow) and is NOT flagged."""
    covered = {("_Leaf", "__post_init__")}
    uncovered, _ = _completeness.compute_uncovered_validators(
        roots=[_Leaf], covered=covered, hoist_fields=_HOIST, seed_names={"_Leaf"}
    )
    assert uncovered == []


def test_unreached_validator_is_flagged() -> None:
    """A raise-validator NOT in covered and NOT reachable via helper-follow is a
    genuine gap and IS flagged."""
    covered = {("_Unreached", "__post_init__")}
    uncovered, _ = _completeness.compute_uncovered_validators(
        roots=[_Unreached], covered=covered, hoist_fields=_HOIST, seed_names={"_Unreached"}
    )
    assert uncovered == ["_Unreached.verify_external"]


def test_directly_covered_validator_not_flagged() -> None:
    """A validator that is itself a covered ast_target is not flagged."""
    covered = {("_Unreached", "__post_init__"), ("_Unreached", "verify_external")}
    uncovered, _ = _completeness.compute_uncovered_validators(
        roots=[_Unreached], covered=covered, hoist_fields=_HOIST, seed_names={"_Unreached"}
    )
    assert uncovered == []


def test_blob_only_excluded() -> None:
    """A non-seed class sharing too few fields with the hoist surface is excluded
    as a blob even with an uncovered raise-validator."""
    uncovered, _ = _completeness.compute_uncovered_validators(
        roots=[_Unreached],
        covered={("_Unreached", "__post_init__")},
        hoist_fields=set(),
        seed_names=set(),
    )
    assert uncovered == []


# ---------------------------------------------------------------------------
# Plain-class fallback (the flag predicate when the class exposes no declared
# field set - e.g. transformers' GenerationConfig, which assigns knobs in
# __init__ rather than as class fields).
# ---------------------------------------------------------------------------


class _PlainSeed:
    """A plain (non-dataclass/pydantic/msgspec) seed config: settable-field
    introspection returns empty, yet its public self.X attributes ARE knobs."""

    def __init__(self) -> None:
        self.alpha = 0
        self.beta = 0

    def validate_plain(self) -> None:
        if self.alpha > self.beta:
            raise ValueError("alpha must not exceed beta")


def test_plain_seed_class_flag_not_zeroed() -> None:
    """A seed plain class with no introspectable field set still flags its
    uncovered raise-validator via the non-private self.X fallback (without it the
    tripwire would be silently blind to GenerationConfig-shaped classes)."""
    assert _completeness._settable_fields(_PlainSeed) == set()
    uncovered, _ = _completeness.compute_uncovered_validators(
        roots=[_PlainSeed], covered=set(), hoist_fields=set(), seed_names={"_PlainSeed"}
    )
    assert uncovered == ["_PlainSeed.validate_plain"]


# ---------------------------------------------------------------------------
# AST-source twins (for engines that cannot import their library - tensorrt).
# ---------------------------------------------------------------------------

_AST_SRC = textwrap.dedent(
    """
    class TrtLlmArgs(BaseLlmArgs):
        alpha: int = 0
        beta: int = 0
        gamma: int = 0
        delta: int = 0

        @model_validator(mode="after")
        def validate_covered(self):
            if self.alpha > self.beta:
                raise ValueError("covered")
            return self

        @model_validator(mode="after")
        def validate_relocated(self):
            if self.gamma > self.delta:
                raise ValueError("relocated - not in covered")
            return self

    class BaseLlmArgs(BaseModel):
        alpha: int = 0
        beta: int = 0
        gamma: int = 0

        def verify_bare(self):
            # bare verify_ (no underscore), uncovered, public field -> flagged
            if self.alpha:
                raise ValueError("bare")

        def validate_no_raise(self):
            return self.beta  # no raise -> not a raise-validator

    class _InternalHelper:
        x: int = 0

        def validate_internal(self):
            if self._private:  # only a private field -> excluded
                raise ValueError("private")

    class Mixin(SomeExternalBase):
        alpha: int = 0
        beta: int = 0
        gamma: int = 0

        def validate_mixin(self):
            if self.alpha:
                raise ValueError("x")
    """
)


def _classdefs(src: str) -> dict[str, ast.ClassDef]:
    return {n.name: n for n in ast.parse(src).body if isinstance(n, ast.ClassDef)}


def test_ast_find_raise_validators_decorator_and_name() -> None:
    """The AST detector finds decorator-tagged AND naming-convention raise
    validators, and skips a validator-named method with no raise."""
    cds = _classdefs(_AST_SRC)
    names = {m for m, _ in _completeness.find_raise_validators_ast(cds["TrtLlmArgs"])}
    assert names == {"validate_covered", "validate_relocated"}
    base_names = {m for m, _ in _completeness.find_raise_validators_ast(cds["BaseLlmArgs"])}
    assert "verify_bare" in base_names
    assert "validate_no_raise" not in base_names


def test_ast_compute_uncovered_flags_relocated_and_bare() -> None:
    """AST-source uncovered computation flags uncovered seed validators (incl.
    the bare verify_), honours covered, blob-excludes the non-seed helper, and
    surfaces an out-of-source base to unanalyzable."""
    cds = _classdefs(_AST_SRC)
    uncovered, unanalyzable = _completeness.compute_uncovered_validators_ast(
        classdefs=cds,
        covered={("TrtLlmArgs", "validate_covered")},
        hoist_fields={"alpha", "beta", "gamma", "delta"},
        seed_names={"TrtLlmArgs", "BaseLlmArgs"},
    )
    # validate_covered is covered; validate_relocated + verify_bare are gaps;
    # Mixin shares >=3 hoist fields so it is in-scope too; _InternalHelper is a
    # blob (not seed, no shared fields) and is excluded.
    assert "TrtLlmArgs.validate_relocated" in uncovered
    assert "BaseLlmArgs.verify_bare" in uncovered
    assert "TrtLlmArgs.validate_covered" not in uncovered
    assert "_InternalHelper.validate_internal" not in uncovered
    # Mixin's base is not in the parsed source -> surfaced, never assumed clean.
    assert any("SomeExternalBase" in u for u in unanalyzable)


def test_ast_helper_follow_covers_reached_validator() -> None:
    """A raise reached via self.helper() from a covered method is not flagged
    (AST-source helper-follow, mirroring the live path)."""
    src = textwrap.dedent(
        """
        class Cfg(BaseModel):
            alpha: int = 0
            beta: int = 0
            gamma: int = 0

            def __post_init__(self):
                self.verify_rel()

            def verify_rel(self):
                if self.alpha > self.beta:
                    raise ValueError("rel")
        """
    )
    cds = _classdefs(src)
    uncovered, _ = _completeness.compute_uncovered_validators_ast(
        classdefs=cds,
        covered={("Cfg", "__post_init__")},
        hoist_fields={"alpha", "beta", "gamma"},
        seed_names={"Cfg"},
    )
    assert uncovered == []


# ---------------------------------------------------------------------------
# Constructor-knob settable fields (BitsAndBytesConfig shape): a class whose
# real knobs are __init__ params, not declared fields, must not zero the flag.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _BnbShape:
    """Mirrors transformers BitsAndBytesConfig: ONE declared dataclass field, but
    the real user knobs are explicit __init__ params set as instance attrs."""

    quant_method: str = "x"

    def __init__(self, load_in_4bit: bool = False, quant_method: str = "x") -> None:
        self.quant_method = quant_method
        self.load_in_4bit = load_in_4bit

    def post_init(self) -> None:
        if not isinstance(self.load_in_4bit, bool):
            raise ValueError("load_in_4bit must be bool")


def test_settable_fields_includes_init_params() -> None:
    """_settable_fields unions declared fields with __init__ knob params, so a
    BNB-shaped class's real knobs are captured (not just the one declared field)."""
    assert _completeness._settable_fields(_BnbShape) == {"quant_method", "load_in_4bit"}


def test_bnb_shape_validator_flagged_when_uncovered() -> None:
    """A relocation/rename on a BNB-shaped class IS caught: post_init reads an
    __init__-knob (load_in_4bit); without the constructor-knob union this would
    be zeroed and the tripwire would stay silently green."""
    uncovered, _ = _completeness.compute_uncovered_validators(
        roots=[_BnbShape], covered=set(), hoist_fields=set(), seed_names={"_BnbShape"}
    )
    assert uncovered == ["_BnbShape.post_init"]


# ---------------------------------------------------------------------------
# AST inheritance-aware hoist + transitive base-class false-negative guard.
# ---------------------------------------------------------------------------

_AST_INHERIT = textwrap.dedent(
    """
    class BaseArgs(StrictBaseModel):
        alpha: int = 0
        beta: int = 0
        gamma: int = 0
        delta: int = 0

    class TopArgs(BaseArgs):
        epsilon: int = 0

    class BuildCfg(BaseModel):
        alpha: int = 0
        beta: int = 0
        gamma: int = 0
        zeta: int = 0

        def check_sizes(self):
            if self.alpha > self.beta:
                raise ValueError("sizes")
    """
)


def test_ast_inheritance_aware_hoist_reaches_engine_arg_shaped_class() -> None:
    """The AST hoist set includes inherited fields, so a non-seed engine-arg-shaped
    config (BuildCfg, sharing >=3 INHERITED-hoist fields) is scanned and its
    uncovered validator flagged. Without inheritance-awareness the hoist would be
    TopArgs's single own field and BuildCfg would be wrongly blob-excluded."""
    cds = _classdefs(_AST_INHERIT)
    hoist = {
        f
        for f in _completeness._settable_fields_ast_inherited("TopArgs", cds)
        if not f.startswith("_")
    }
    assert {"alpha", "beta", "gamma", "delta", "epsilon"} <= hoist
    uncovered, _ = _completeness.compute_uncovered_validators_ast(
        classdefs=cds, covered=set(), hoist_fields=hoist, seed_names={"TopArgs", "BaseArgs"}
    )
    assert "BuildCfg.check_sizes" in uncovered


def test_ast_transitive_base_guard_surfaces_grandparent() -> None:
    """A flat seed re-parented through a parsed-but-non-flat intermediate onto an
    out-of-source grandparent surfaces the grandparent (never silently dropped)."""
    src = textwrap.dedent(
        """
        class MidBase(HiddenUpstream):
            solo: int = 0

        class FlatSeed(MidBase):
            alpha: int = 0
            beta: int = 0
            gamma: int = 0
        """
    )
    cds = _classdefs(src)
    _, unanalyzable = _completeness.compute_uncovered_validators_ast(
        classdefs=cds,
        covered=set(),
        hoist_fields={"alpha", "beta", "gamma"},
        seed_names={"FlatSeed"},
    )
    assert any("HiddenUpstream" in u for u in unanalyzable)
