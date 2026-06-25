"""Tests for the drift-completeness substrate (scripts.engine_producers._completeness).

Covers the raise-validator detector (widened naming conventions + inherited /
no-source surfacing) and the uncovered-computation's same-class helper-follow
coverage model (a raise the miner reaches via ``self.helper()`` from a covered
method must NOT be flagged as a gap).
"""

from __future__ import annotations

import dataclasses

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
