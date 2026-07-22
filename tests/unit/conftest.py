"""Shared fixtures for the unit-test suite."""

from __future__ import annotations

import importlib.metadata
from unittest.mock import patch

import pytest

# Canned runtime-dependency specs used to keep dispatch-command construction
# hermetic. Command-construction and container-lifecycle tests only need a
# non-empty spec list so _materialise_dispatch_assets can write a real
# requirements file; they never assert on the specs themselves.
_STUB_RUNTIME_REQUIREMENTS = ("filelock>=3.12", "numpy>=1.24", "pyarrow>=15", "pydantic>=2")


@pytest.fixture(autouse=True)
def stub_runtime_requirements(request):
    """Keep dispatch-command construction independent of an installed distribution.

    ``append_package_dispatch`` -> ``_materialise_dispatch_assets`` ->
    ``_runtime_requirements`` reads ``importlib.metadata.requires`` for the
    llenergymeasure distribution. That metadata exists only when the package is
    pip-installed; it is absent when the suite runs from a source tree on
    ``PYTHONPATH`` with no install (the GPU-CI container mounts the repo and sets
    ``PYTHONPATH`` deliberately, without installing). Without a stub every test
    that builds a docker command raises ``PackageNotFoundError`` - the docker
    runner (``tests/unit/docker``), the baseline-container dispatch
    (``tests/unit/study``), and the engine docker paths (``tests/unit/engines``)
    all reach it, so the stub is autouse across the whole unit suite.

    These are unit tests of command construction and container lifecycle, so the
    metadata lookup is stubbed with a canned spec list. Tests that assert on the
    real distribution metadata mark themselves ``needs_dist_metadata``; they are
    left unstubbed and skipped when the distribution is not installed.

    The stub is a ``patch.object`` attribute swap, so it takes effect without
    touching the ``functools.cache``-wrapped resolvers. The process-lifetime
    ``_materialise_dispatch_assets`` cache is deliberately left intact (it is
    warmed once and reused, the behaviour the prefix-aware mkdtemp router is
    built to tolerate); the ``needs_dist_metadata`` tests clear it themselves via
    their ``_fresh`` helper when they need a live re-materialisation.
    """
    from llenergymeasure.infra.docker import command

    if "needs_dist_metadata" in request.keywords:
        try:
            importlib.metadata.distribution(command._DISPATCH_DIST_NAME)
        except importlib.metadata.PackageNotFoundError:
            pytest.skip(
                "llenergymeasure distribution metadata is not installed "
                "(source tree on PYTHONPATH, no install); real-metadata "
                "assertions are not applicable"
            )
        yield
        return

    with patch.object(command, "_runtime_requirements", return_value=_STUB_RUNTIME_REQUIREMENTS):
        yield
