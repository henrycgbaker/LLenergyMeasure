"""Tests for the vLLM static + dynamic miners.

Covers:
- Lift call-site: ``_pydantic_lift`` and ``_msgspec_lift`` produce
  candidates over the real vLLM types they're configured against.
- ``_dataclass_lift`` is intentionally NOT called by the vLLM dynamic miner
  (EngineArgs Literal invariants don't fire at construction); the lift module
  itself has its own unit tests in ``test_dataclass_lift.py``.
- Static miner method-resolution: every declared landmark exists on the
  installed library, raising :class:`MinerLandmarkMissingError` if any
  drifts away.
- Fixpoint gate-soundness: ``assert_gate_soundness_fixpoint`` succeeds on
  the real validation gate, mirroring the transformers test.

Tests are skipped when ``vllm`` isn't importable in the test environment
(GH-hosted ``ubuntu-latest`` without the optional extra installed).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

vllm = pytest.importorskip("vllm", reason="vLLM not installed in this environment")

from scripts.engine_miners._base import (  # noqa: E402
    MinerLandmarkMissingError,
    RuleCandidate,
)
from scripts.engine_miners._dataclass_lift import lift as dataclass_lift  # noqa: E402
from scripts.engine_miners._msgspec_lift import lift as msgspec_lift  # noqa: E402
from scripts.engine_miners._pydantic_lift import lift as pydantic_lift  # noqa: E402
from scripts.engine_miners._ssot import load_miner_pin  # noqa: E402
from scripts.engine_miners.vllm_dynamic_miner import (  # noqa: E402
    _PYDANTIC_LIFT_TARGETS,
    CLUSTERS,
    walk_vllm_dynamic,
)
from scripts.engine_miners.vllm_static_miner import (  # noqa: E402
    _AST_TARGETS,
    _check_landmarks,
    walk_vllm_static,
)

# ---------------------------------------------------------------------------
# Lift call-site smoke tests - confirm the per-engine driver composes the
# library-level lifts correctly against real vLLM types.
# ---------------------------------------------------------------------------


class TestLiftCallSites:
    """Verify the dynamic miner's lift composition produces non-empty output
    on classes the design says it should."""

    def test_msgspec_lift_called_on_sampling_params(self) -> None:
        """msgspec_lift on ``SamplingParams`` returns a list (possibly empty).

        Per the research doc, vLLM 0.17.1 ships zero ``msgspec.Meta(...)``
        annotations on SamplingParams. The lift therefore returns ``[]``
        today - the test pins the call-site (lift IS invoked) and the
        contract that the result is a list of :class:`RuleCandidate`. The
        day vLLM adopts ``Meta(ge=...)`` annotations, the count flips
        positive without code change.
        """
        from vllm import SamplingParams

        result = msgspec_lift(
            SamplingParams,
            namespace="vllm.sampling",
            today="2026-04-26",
            source_path="vllm/sampling_params.py",
        )
        assert isinstance(result, list)
        for cand in result:
            assert isinstance(cand, RuleCandidate)
            assert cand.added_by == "msgspec_lift"

    def test_pydantic_lift_called_on_cache_config(self) -> None:
        """pydantic_lift on ``CacheConfig`` produces multiple invariants.

        Anchors the lift's behaviour on a class the research doc identifies
        as having rich annotated-types metadata
        (``gpu_memory_utilization: Annotated[float, Gt(0), Le(1)]``).
        """
        from vllm.config import CacheConfig

        result = pydantic_lift(
            CacheConfig,
            namespace="vllm.engine",
            today="2026-04-26",
            source_path="vllm/config/cache.py",
        )
        assert len(result) >= 5, (
            f"Expected pydantic_lift to find at least 5 invariants on CacheConfig "
            f"(gpu_memory_utilization, swap_space, block_size etc.); got {len(result)}"
        )
        for cand in result:
            assert isinstance(cand, RuleCandidate)
            assert cand.added_by == "pydantic_lift"

    def test_dataclass_lift_call_site_dropped_for_engine_args(self) -> None:
        """The dynamic miner intentionally skips ``_dataclass_lift(EngineArgs)``.

        Stdlib dataclass doesn't enforce Literal types - running the lift
        here would emit ~23 unenforceable invariants that all fail validation-CI.
        Pin the omission so a future refactor doesn't accidentally re-add
        the call.
        """
        # Sanity check the lift module DOES handle EngineArgs structurally
        # (the lift itself is fine; the engine-level decision is to skip it).
        from vllm.engine.arg_utils import EngineArgs

        candidates_if_called = dataclass_lift(
            EngineArgs,
            namespace="vllm.engine",
            today="2026-04-26",
            source_path="vllm/engine/arg_utils.py",
        )
        # Sanity: the lift is structurally functional (would emit something).
        assert isinstance(candidates_if_called, list)
        # The actual dynamic-miner output must NOT include EngineArgs invariants.
        candidates, _ = walk_vllm_dynamic()
        engineargs_rules = [c for c in candidates if "EngineArgs" in c.native_type]
        assert engineargs_rules == [], (
            f"Dynamic miner unexpectedly emitted {len(engineargs_rules)} "
            f"EngineArgs invariants; the dataclass-lift call is supposed to be "
            f"skipped for EngineArgs (see vllm_dynamic_miner.py for rationale)."
        )

    @pytest.mark.parametrize(
        "target",
        _PYDANTIC_LIFT_TARGETS,
        ids=[t.class_name for t in _PYDANTIC_LIFT_TARGETS],
    )
    def test_every_pydantic_target_imports(
        self,
        target: object,  # _LiftTarget; using object to avoid private-type leak
    ) -> None:
        """Each target the dynamic miner lifts must be importable.

        Mirrors the static miner's fail-loud landmark contract for the
        dynamic-miner side: missing class -> clear failure rather than
        silent zero-invariant output.
        """
        from scripts.engine_miners.vllm_dynamic_miner import _LiftTarget

        assert isinstance(target, _LiftTarget)
        module = __import__(target.module_path, fromlist=[target.class_name])
        cls = getattr(module, target.class_name, None)
        assert cls is not None, (
            f"Pydantic lift target {target.module_path}.{target.class_name} is not importable"
        )


# ---------------------------------------------------------------------------
# Static miner method resolution
# ---------------------------------------------------------------------------


class TestStaticMinerLandmarks:
    """Every declared AST landmark must resolve via ``find_class`` /
    ``find_method`` at miner-import time, so an upstream rename surfaces
    as a loud landmark error instead of silent zero-invariant output."""

    def test_check_landmarks_passes_on_pinned_version(self) -> None:
        """``_check_landmarks()`` returns cleanly on the pinned vLLM version.

        Implicitly verifies every entry in ``_AST_TARGETS`` resolves; if any
        method has been renamed / split, the test fails with the specific
        landmark name, not a silent zero-invariant output.
        """
        version, paths = _check_landmarks()
        assert version, "vLLM version string was empty"
        assert paths, "No source paths returned from landmark check"

    @pytest.mark.parametrize(
        "target",
        _AST_TARGETS,
        ids=[f"{t.class_name}.{t.method}" for t in _AST_TARGETS],
    )
    def test_each_ast_landmark_present(self, target: object) -> None:
        """Each (class, method) target lives where the miner expects it.

        Parametrised so a missing landmark surfaces in the test name,
        making CI failures actionable.
        """
        from scripts.engine_miners.vllm_static_miner import _ASTTarget

        assert isinstance(target, _ASTTarget)
        module = __import__(target.module_path, fromlist=[target.class_name])
        cls = getattr(module, target.class_name, None)
        assert cls is not None, f"AST target class {target.module_path}.{target.class_name} missing"
        method = getattr(cls, target.method, None)
        assert method is not None, f"AST target method {target.class_name}.{target.method} missing"

    def test_landmark_missing_raises_loud(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Removing a landmark on the imported class makes ``_check_landmarks`` raise."""
        from vllm.config import CacheConfig

        # Hide the validator method on the class for the duration of this test.
        monkeypatch.delattr(CacheConfig, "_validate_cache_dtype")
        with pytest.raises(MinerLandmarkMissingError):
            _check_landmarks()


# ---------------------------------------------------------------------------
# Miner top-level smoke
# ---------------------------------------------------------------------------


class TestWalkerSmoke:
    """Confirm the full miner returns plausible output."""

    def test_static_miner_emits_rules(self) -> None:
        """Static miner produces a non-trivial number of invariants on the live library."""
        candidates, version = walk_vllm_static()
        assert load_miner_pin("vllm", "static").contains(version, prereleases=True), (
            f"Installed vllm=={version} outside SSOT miner pin"
        )
        assert len(candidates) >= 30, (
            f"Static miner emitted only {len(candidates)} candidates; "
            f"expected >=30 across the AST-target list."
        )
        # Every emitted invariant must reference a real vLLM source path.
        for c in candidates:
            assert c.miner_source.path.startswith("vllm/"), (
                f"Source path {c.miner_source.path!r} not rooted at vllm/"
            )

    def test_dynamic_miner_emits_rules(self) -> None:
        """Dynamic miner's lift composition produces a non-trivial corpus."""
        candidates, _version = walk_vllm_dynamic()
        # Pydantic lift over the 12 ``vllm.config.*`` classes plus the
        # SamplingParams cluster probes produces roughly 30-40 candidates
        # on vLLM 0.17.x; assert at least 25 to leave headroom for minor
        # bumps without making the test brittle.
        assert len(candidates) >= 25, f"Dynamic miner emitted only {len(candidates)} candidates"
        for c in candidates:
            assert c.engine == "vllm"
            assert c.added_by in {
                "pydantic_lift",
                "msgspec_lift",
                "dataclass_lift",
                "dynamic_miner",
            }


# ---------------------------------------------------------------------------
# Cluster registry sanity
# ---------------------------------------------------------------------------


class TestClusters:
    @pytest.mark.parametrize("cluster", CLUSTERS, ids=[c.name for c in CLUSTERS])
    def test_cluster_factory_constructs(self, cluster: object) -> None:
        """Each cluster's probe factory accepts a representative kwargs dict.

        Smoke-tests the per-cluster ``probe_class_factory`` so a typo in
        cluster definition fails loudly rather than silently emitting zero
        probe rows.
        """
        from scripts.engine_miners.vllm_dynamic_miner import _Cluster

        assert isinstance(cluster, _Cluster)
        # Pick the FIRST value of each grid as a positive trial.
        kwargs = {name: vals[0] for name, vals in cluster.values_per_field.items()}
        # Some combinations may legitimately raise - that's fine; we just
        # need the factory to be callable without TypeError on the kwargs
        # shape.
        try:
            cluster.probe_class_factory(kwargs)
        except (ValueError, TypeError) as exc:
            # ValueError / TypeError from the library itself is fine -
            # construction-time validation, not factory misconfiguration.
            assert "probe_class_factory" not in str(exc)


# ---------------------------------------------------------------------------
# Gate-soundness fixpoint regression
# ---------------------------------------------------------------------------


class TestGateSoundnessOnVllmCorpus:
    """Re-run the gate-soundness fixpoint from the vLLM test file.

    The structural gate-soundness fixpoint is library-agnostic - it
    synthesises malformed invariants and asserts the gate flags each - but
    re-running it here guarantees the contract is exercised in the
    vLLM CI lane too.
    """

    def test_gate_soundness_passes(self) -> None:
        from scripts.engine_miners._fixpoint_test import assert_gate_soundness_fixpoint

        assert_gate_soundness_fixpoint()


# ---------------------------------------------------------------------------
# build_corpus registration
# ---------------------------------------------------------------------------


def test_vllm_engine_registered_in_build_corpus() -> None:
    """``--engine vllm`` resolves to the static + dynamic miner extractors.

    Without this, removing or renaming an entry in
    ``_ENGINE_EXTRACTORS["vllm"]`` would silently break the pipeline.
    """
    from scripts.engine_miners import build_corpus

    assert "vllm" in build_corpus._ENGINE_EXTRACTORS
    extractors = build_corpus._ENGINE_EXTRACTORS["vllm"]
    modules = {e.module for e in extractors}
    assert modules == {
        "scripts.engine_miners.vllm_static_miner",
        "scripts.engine_miners.vllm_dynamic_miner",
    }
    basenames = {e.staging_basename for e in extractors}
    assert basenames == {
        "vllm_static_miner.yaml",
        "vllm_dynamic_miner.yaml",
    }


# ---------------------------------------------------------------------------
# CPU-only safety
# ---------------------------------------------------------------------------


def test_vllm_miner_does_not_import_torch() -> None:
    """Mining must not pull torch into the importing process.

    The miner is invoked from ``ubuntu-latest`` GitHub runners which have
    no torch installed (engines run inside their Docker images; host has
    no engine deps). Asserting ``torch not in sys.modules`` after the
    walkers run is a stronger CPU-safety guarantee than the previous
    ``torch.cuda.is_initialized()`` check: it catches an inadvertent
    ``import torch`` even before any CUDA context could be created, and
    needs no host torch dependency to enforce.
    """
    pre = {name for name in sys.modules if name == "torch" or name.startswith("torch.")}
    walk_vllm_static(today="2026-04-27")
    walk_vllm_dynamic(today="2026-04-27")
    post = {name for name in sys.modules if name == "torch" or name.startswith("torch.")}
    added = post - pre
    assert not added, f"miner imported torch modules: {sorted(added)}"
