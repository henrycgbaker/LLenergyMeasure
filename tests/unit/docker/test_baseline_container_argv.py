"""Unit tests for the idle-baseline container argv builder (pure, no docker).

The baseline container is one of the three shapes this module builds. It runs to
completion like the offline experiment dispatch and shares its package-dispatch
bootstrap, but points the in-container entrypoint at the baseline entry module,
mounts no HuggingFace cache, requests no shared memory, and stays anonymous
(ownership labels, no ``--name``). These tests pin those choices flag by flag;
``test_container_argv_shapes.py`` pins the whole argv.
"""

from __future__ import annotations

from pathlib import Path

from llenergymeasure.config.ssot import ENV_BASELINE_SPEC_PATH, ENV_ENGINE, ENV_ENTRY_MODULE
from llenergymeasure.infra.docker.command import build_baseline_container_argv


def test_argv_contains_spec_env_and_gpu_filter(tmp_path: Path):
    argv = build_baseline_container_argv(
        image="ghcr.io/foo/bar:v1",
        exchange_dir=str(tmp_path),
        gpu_indices=[0, 2],
        engine="vllm",
    )
    assert argv[0] == "docker"
    assert "run" in argv
    assert "--rm" in argv
    assert "--gpus" in argv
    # env vars
    assert any(f"{ENV_BASELINE_SPEC_PATH}=" in part for part in argv)
    assert any("CUDA_VISIBLE_DEVICES=0,2" in part for part in argv)
    # image is the LAST arg (the entrypoint script invokes the module itself)
    assert argv[-1] == "ghcr.io/foo/bar:v1"
    # default --gpus request is "all"
    assert argv[argv.index("--gpus") + 1] == "all"


def test_argv_honours_llem_docker_gpus(tmp_path: Path, monkeypatch):
    """LLEM_DOCKER_GPUS overrides the --gpus value (shared-host pinning)."""
    monkeypatch.setenv("LLEM_DOCKER_GPUS", "device=2")
    argv = build_baseline_container_argv(
        image="ghcr.io/foo/bar:v1",
        exchange_dir=str(tmp_path),
        gpu_indices=[0],
        engine="vllm",
    )
    assert argv[argv.index("--gpus") + 1] == "device=2"


def test_argv_scopes_gpus_from_config_indices(tmp_path: Path, monkeypatch):
    """config_gpu_indices scopes the baseline --gpus to the same physical
    devices as the experiment container (single -> device=N)."""
    monkeypatch.delenv("LLEM_DOCKER_GPUS", raising=False)
    argv = build_baseline_container_argv(
        image="img:latest",
        exchange_dir=str(tmp_path),
        gpu_indices=[0],
        engine="vllm",
        config_gpu_indices=[2],
    )
    assert argv[argv.index("--gpus") + 1] == "device=2"


def test_argv_scopes_gpus_from_config_indices_multi(tmp_path: Path, monkeypatch):
    """Multi config indices are quoted for docker's CSV parser."""
    monkeypatch.delenv("LLEM_DOCKER_GPUS", raising=False)
    argv = build_baseline_container_argv(
        image="img:latest",
        exchange_dir=str(tmp_path),
        gpu_indices=[0, 1],
        engine="vllm",
        config_gpu_indices=[2, 3],
    )
    assert argv[argv.index("--gpus") + 1] == '"device=2,3"'


def test_env_overrides_config_gpu_indices(tmp_path: Path, monkeypatch):
    """LLEM_DOCKER_GPUS still wins over config_gpu_indices for the baseline."""
    monkeypatch.setenv("LLEM_DOCKER_GPUS", "device=5")
    argv = build_baseline_container_argv(
        image="img:latest",
        exchange_dir=str(tmp_path),
        gpu_indices=[0],
        engine="vllm",
        config_gpu_indices=[2, 3],
    )
    assert argv[argv.index("--gpus") + 1] == "device=5"


def test_argv_makes_package_importable(tmp_path: Path):
    """The baseline argv must mount the host package + route through the
    entrypoint script so the upstream engine image can import the package.

    Without this, every Docker baseline run fails with ModuleNotFoundError:
    upstream engine images do not ship the llenergymeasure package.
    """
    argv = build_baseline_container_argv(
        image="img:latest",
        exchange_dir=str(tmp_path),
        gpu_indices=[0],
        engine="vllm",
    )
    # Package source bind-mount (makes the package importable via PYTHONPATH).
    # The package dir is mounted at the nested target so /llem-src exposes
    # only llenergymeasure, never a host site-packages sibling.
    assert any(arg.endswith(":/llem-src/llenergymeasure:ro") for arg in argv)
    # Dispatch routes through the shared in-container bootstrap script.
    ep_idx = argv.index("--entrypoint")
    assert argv[ep_idx + 1] == "/llem-entry.sh"
    # The script exec's the baseline module (not the experiment one).
    assert f"{ENV_ENTRY_MODULE}=llenergymeasure.entrypoints.baseline_measure" in argv
    # Engine is propagated so the script can route tensorrt correctly.
    assert f"{ENV_ENGINE}=vllm" in argv
    # The old, broken trailing "python3 -m ..." form must be gone.
    assert "python3" not in argv


def test_argv_tensorrt_engine_propagated(tmp_path: Path):
    argv = build_baseline_container_argv(
        image="img:latest",
        exchange_dir=str(tmp_path),
        gpu_indices=[0],
        engine="tensorrt",
    )
    assert f"{ENV_ENGINE}=tensorrt" in argv


def test_argv_empty_gpu_indices(tmp_path: Path):
    argv = build_baseline_container_argv(
        image="img:latest",
        exchange_dir=str(tmp_path),
        gpu_indices=[],
        engine="transformers",
    )
    assert any("CUDA_VISIBLE_DEVICES=" in part for part in argv)


def test_nccl_vars_forwarded(tmp_path: Path, monkeypatch):
    """Host NCCL_* vars are forwarded into the baseline container, matching
    the experiment path (e.g. NCCL_P2P_DISABLE=1 on PCIe hosts without P2P)."""
    monkeypatch.setenv("NCCL_P2P_DISABLE", "1")
    monkeypatch.setenv("NCCL_IB_DISABLE", "1")
    argv = build_baseline_container_argv(
        image="img:latest",
        exchange_dir=str(tmp_path),
        gpu_indices=[0, 1],
        engine="vllm",
    )
    assert "NCCL_P2P_DISABLE=1" in argv
    assert "NCCL_IB_DISABLE=1" in argv


def test_non_nccl_var_not_forwarded(tmp_path: Path, monkeypatch):
    """A non-NCCL host var must not be forwarded into the baseline container."""
    monkeypatch.setenv("SOME_UNRELATED_VAR", "leak")
    argv = build_baseline_container_argv(
        image="img:latest",
        exchange_dir=str(tmp_path),
        gpu_indices=[0],
        engine="vllm",
    )
    assert not any("SOME_UNRELATED_VAR" in part for part in argv)


def test_nccl_vars_sorted(tmp_path: Path, monkeypatch):
    """NCCL_* vars are emitted in sorted key order (deterministic argv)."""
    monkeypatch.setenv("NCCL_SOCKET_IFNAME", "eth0")
    monkeypatch.setenv("NCCL_DEBUG", "INFO")
    monkeypatch.setenv("NCCL_P2P_DISABLE", "1")
    argv = build_baseline_container_argv(
        image="img:latest",
        exchange_dir=str(tmp_path),
        gpu_indices=[0],
        engine="vllm",
    )
    # Filter to the three we set so a stray host NCCL_* var can't perturb
    # the assertion; their relative order must be alphabetical.
    expected = ["NCCL_DEBUG=INFO", "NCCL_P2P_DISABLE=1", "NCCL_SOCKET_IFNAME=eth0"]
    mine = [p for p in argv if p in set(expected)]
    assert mine == expected


def test_argv_carries_ownership_labels(tmp_path: Path):
    """Ownership labels ride before the image so cleanup and the reaper see it.

    A baseline container holds the GPU while it samples; unlabelled, it is
    invisible to the study-scoped cleanup and to the orphan reaper.
    """
    argv = build_baseline_container_argv(
        image="img:latest",
        exchange_dir=str(tmp_path),
        gpu_indices=[0],
        engine="vllm",
        labels={"llem.study_id": "abcdef12", "llem.parent_pid": "4242"},
    )

    for value in ("llem.study_id=abcdef12", "llem.parent_pid=4242"):
        idx = argv.index(value)
        assert argv[idx - 1] == "--label"
        assert idx < argv.index("img:latest")


def test_argv_without_labels_emits_none(tmp_path: Path):
    argv = build_baseline_container_argv(
        image="img:latest",
        exchange_dir=str(tmp_path),
        gpu_indices=[0],
        engine="vllm",
    )

    assert "--label" not in argv
