"""Unit tests for the engine-server container argv builder (pure, no docker).

The long-lived server container is one of the container shapes this module
builds, and it diverges from the offline dispatch on purpose: detached, on the
host network, and deliberately without ``--rm`` so a crash-on-startup leaves its
logs recoverable. These tests pin those choices, the ownership labels the study
cleanup selects on, the HF cache mount that stops a launched server
re-downloading the weights, and the host ``NCCL_*`` forwarding that a
tensor-parallel server needs as much as a multi-GPU offline run does. One test
pins the whole argv so a reordering or a stray extra flag cannot slip through.
"""

from __future__ import annotations

import os

import pytest

from llenergymeasure.infra.docker.command import build_server_container_argv


@pytest.fixture
def no_host_nccl(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear host ``NCCL_*`` vars so they cannot perturb an exact-argv assertion.

    The builder forwards them, so a machine that exports ``NCCL_P2P_DISABLE=1``
    (the reason the forwarding exists) would otherwise add ``-e`` pairs the
    literal below does not name.
    """
    for key in list(os.environ):
        if key.startswith("NCCL_"):
            monkeypatch.delenv(key, raising=False)


def test_container_argv_has_ruled_flags():
    """docker run argv carries image, --gpus, --network host, and the port."""
    argv = build_server_container_argv(
        image="vllm/vllm-openai:v0.19.1",
        container_name="llem-vllm-server-abc",
        gpu_indices=None,
        serve_args=["Qwen/Qwen2.5-0.5B", "--port", "8123"],
        shm_size="8g",
    )

    assert argv[:2] == ["docker", "run"]
    assert "-d" in argv
    # No --rm: a crashed container must survive so `docker logs` can recover the
    # startup diagnostic (the failure-artefact hand-off); leak-freeness is explicit in shutdown.
    assert "--rm" not in argv
    # --network host is unconditional and adjacent.
    assert argv[argv.index("--network") + 1] == "host"
    assert argv[argv.index("--gpus") + 1] == "all"
    assert argv[argv.index("--name") + 1] == "llem-vllm-server-abc"
    # Image precedes the serve args; the port lives in the serve args (host net,
    # so no -p publish is emitted).
    img_idx = argv.index("vllm/vllm-openai:v0.19.1")
    assert argv[img_idx + 1 :] == ["Qwen/Qwen2.5-0.5B", "--port", "8123"]
    assert "-p" not in argv


def test_container_argv_carries_ownership_labels():
    """Ownership labels are emitted before the image, so the study owns the server.

    The study-scoped cleanup filters on ``llem.study_id`` and the orphan reaper on
    ``llem.parent_pid``; an unlabelled server container is invisible to both.
    """
    argv = build_server_container_argv(
        image="vllm/vllm-openai:v0.19.1",
        container_name="llem-vllm-server-abc",
        gpu_indices=None,
        serve_args=["m", "--port", "8123"],
        shm_size="8g",
        labels={"llem.study_id": "abcdef12", "llem.parent_pid": "4242"},
    )

    assert "llem.study_id=abcdef12" in argv
    assert "llem.parent_pid=4242" in argv
    for value in ("llem.study_id=abcdef12", "llem.parent_pid=4242"):
        idx = argv.index(value)
        assert argv[idx - 1] == "--label"
        # docker run options must precede the image name.
        assert idx < argv.index("vllm/vllm-openai:v0.19.1")


def test_container_argv_without_labels_emits_none():
    """No labels supplied (e.g. a direct non-study launch) emits no --label flags."""
    argv = build_server_container_argv(
        image="img:v1",
        container_name=None,
        gpu_indices=None,
        serve_args=["m"],
    )

    assert "--label" not in argv


def test_container_argv_mounts_hf_cache(monkeypatch):
    """The server container binds the HF cache + sets HF_HOME (else weights re-download).

    Same LLEM_DOCKER_HF_CACHE-driven mount the offline docker dispatch uses; the
    mount/env precede the image (docker run options come before the image name).
    """
    monkeypatch.setenv("LLEM_DOCKER_HF_CACHE", "/data/hf")
    argv = build_server_container_argv(
        image="vllm/vllm-openai:v0.19.1",
        container_name="llem-vllm-server-abc",
        gpu_indices=None,
        serve_args=["m", "--port", "8123"],
        shm_size="8g",
    )
    target = "/root/.cache/huggingface"
    # -v <host>:<target> present, and it precedes the image.
    assert f"/data/hf:{target}" in argv
    mount_idx = argv.index(f"/data/hf:{target}")
    assert argv[mount_idx - 1] == "-v"
    assert mount_idx < argv.index("vllm/vllm-openai:v0.19.1")
    # HF_HOME points at the in-container target.
    assert f"HF_HOME={target}" in argv


def test_container_argv_forwards_host_nccl_env(monkeypatch):
    """Host ``NCCL_*`` vars reach the server container, in sorted ``-e`` form.

    The engine's tensor-parallel workers run inside this container, so a PCIe
    host whose topology lacks functional GPU peer-to-peer needs
    ``NCCL_P2P_DISABLE=1`` here exactly as the offline and baseline containers
    need it: without it the server hangs at its first NCCL collective and never
    becomes ready. Same ``-e KEY=VALUE`` form and sorted key order as the other
    two shapes, and before the image (docker options precede the image name).
    """
    monkeypatch.setenv("NCCL_P2P_DISABLE", "1")
    monkeypatch.setenv("NCCL_DEBUG", "INFO")

    argv = build_server_container_argv(
        image="vllm/vllm-openai:v0.19.1",
        container_name="llem-vllm-server-abc",
        gpu_indices=[0, 1],
        serve_args=["m", "--port", "8123", "--tensor-parallel-size", "2"],
        shm_size="8g",
    )

    for value in ("NCCL_DEBUG=INFO", "NCCL_P2P_DISABLE=1"):
        position = argv.index(value)
        assert argv[position - 1] == "-e"
        assert position < argv.index("vllm/vllm-openai:v0.19.1")
    # Sorted key order keeps the argv deterministic (other tests pin it whole).
    assert argv.index("NCCL_DEBUG=INFO") < argv.index("NCCL_P2P_DISABLE=1")


def test_container_argv_carries_no_nccl_when_host_sets_none(no_host_nccl):
    """No host ``NCCL_*`` var means no NCCL flags: the argv is otherwise untouched.

    The other half of the forwarding pair - forwarding is opt-in through the host
    environment, never an unconditional flag the shape now always carries.
    """
    argv = build_server_container_argv(
        image="vllm/vllm-openai:v0.19.1",
        container_name="llem-vllm-server-abc",
        gpu_indices=None,
        serve_args=["m", "--port", "8123"],
        shm_size="8g",
    )

    assert not any("NCCL" in token for token in argv)


def test_non_nccl_host_var_not_forwarded(monkeypatch):
    """A non-NCCL host var must not ride into the server container.

    ``NCCLX_FAKE`` is the near-miss half: an unrelated var is caught by any
    forwarding filter at all, but only a prefix that keeps the underscore
    (``NCCL_``, not ``NCCL``) excludes a var whose name merely starts with the
    same four letters.
    """
    monkeypatch.setenv("SOME_UNRELATED_VAR", "leak")
    monkeypatch.setenv("NCCLX_FAKE", "1")

    argv = build_server_container_argv(
        image="img:v1",
        container_name=None,
        gpu_indices=None,
        serve_args=["m"],
    )

    assert not any("SOME_UNRELATED_VAR" in token for token in argv)
    assert not any("NCCLX_FAKE" in token for token in argv)


def test_container_argv_is_pinned_exactly(monkeypatch, no_host_nccl):
    """The whole argv is pinned, flag order included, for one fully-specified launch.

    The other argv tests assert individual flags and their adjacency; this one
    fixes the complete list so a reordering or an accidental extra flag cannot
    slip through. Every env-driven input is pinned so the expectation is stable
    on any host.
    """
    monkeypatch.delenv("LLEM_DOCKER_GPUS", raising=False)
    monkeypatch.setenv("LLEM_DOCKER_HF_CACHE", "/data/hf")

    argv = build_server_container_argv(
        image="vllm/vllm-openai:v0.19.1",
        container_name="llem-vllm-server-abc",
        gpu_indices=[2, 3],
        serve_args=["Qwen/Qwen2.5-0.5B", "--port", "8123"],
        shm_size="8g",
        labels={"llem.study_id": "abcdef12", "llem.parent_pid": "4242"},
    )

    assert argv == [
        "docker",
        "run",
        "-d",
        "--network",
        "host",
        "--gpus",
        '"device=2,3"',
        "--name",
        "llem-vllm-server-abc",
        "--label",
        "llem.study_id=abcdef12",
        "--label",
        "llem.parent_pid=4242",
        "--shm-size",
        "8g",
        "-v",
        "/data/hf:/root/.cache/huggingface",
        "-e",
        "HF_HOME=/root/.cache/huggingface",
        "vllm/vllm-openai:v0.19.1",
        "Qwen/Qwen2.5-0.5B",
        "--port",
        "8123",
    ]
