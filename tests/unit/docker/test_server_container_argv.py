"""Unit tests for the engine-server container argv builder (pure, no docker).

The long-lived server container is one of the container shapes this module
builds, and it diverges from the offline dispatch on purpose: detached, on the
host network, and deliberately without ``--rm`` so a crash-on-startup leaves its
logs recoverable. These tests pin those choices, the ownership labels the study
cleanup selects on, and the HF cache mount that stops a launched server
re-downloading the weights. One test pins the whole argv so a reordering or a
stray extra flag cannot slip through.
"""

from __future__ import annotations

from llenergymeasure.infra.docker.command import build_server_container_argv


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


def test_container_argv_is_pinned_exactly(monkeypatch):
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
