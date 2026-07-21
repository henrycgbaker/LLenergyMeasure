"""Shared fixtures and factories for docker/ tests."""

from __future__ import annotations

import importlib.metadata
from unittest.mock import MagicMock, patch

import pytest

# Canned runtime-dependency specs used to keep dispatch-command construction
# hermetic. The command-construction and container-lifecycle tests only need a
# non-empty spec list so _materialise_dispatch_assets can write a real
# requirements file; they never assert on the specs themselves.
_STUB_RUNTIME_REQUIREMENTS = ("filelock>=3.12", "numpy>=1.24", "pyarrow>=15", "pydantic>=2")


def make_subprocess_result(returncode: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    """Return a mock subprocess.CompletedProcess.

    Replaces the identical _make_proc (test_docker_runner) and
    _make_subprocess_result (test_docker_preflight) factories.
    """
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdout = stdout
    proc.stderr = stderr
    return proc


@pytest.fixture(autouse=True)
def stub_runtime_requirements(request):
    """Keep dispatch-command construction independent of an installed distribution.

    ``append_package_dispatch`` -> ``_materialise_dispatch_assets`` ->
    ``_runtime_requirements`` reads ``importlib.metadata.requires`` for the
    llenergymeasure distribution. That metadata exists only when the package is
    pip-installed; it is absent when the suite runs from a source tree on
    ``PYTHONPATH`` with no install (the GPU-CI container mounts the repo and sets
    ``PYTHONPATH`` deliberately, without installing). Without a stub every test
    that builds a docker command raises ``PackageNotFoundError``.

    These are unit tests of command construction and container lifecycle, so the
    metadata lookup is stubbed with a canned spec list. Tests that assert on the
    real distribution metadata mark themselves ``needs_dist_metadata``; they are
    left unstubbed and skipped when the distribution is not installed.

    The two ``functools.cache``-wrapped resolvers are cleared around every test
    so the stub (or the real lookup) is what actually runs, regardless of which
    value an earlier test warmed the cache with.
    """
    from llenergymeasure.infra.docker import command

    command._runtime_requirements.cache_clear()
    command._materialise_dispatch_assets.cache_clear()

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
    else:
        with patch.object(
            command, "_runtime_requirements", return_value=_STUB_RUNTIME_REQUIREMENTS
        ):
            yield

    command._runtime_requirements.cache_clear()
    command._materialise_dispatch_assets.cache_clear()
