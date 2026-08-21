"""Whole-argv pins for the three container shapes the shared core assembles.

Every ``docker run`` llenergymeasure launches comes out of
``infra.docker.command.build_container_argv``; the offline experiment dispatch,
the idle-baseline measurement, and the long-lived engine server are
parameterisations of it. Their divergences are load-bearing:

- server: no ``--rm`` (a crash-on-startup must survive for ``docker logs``),
  ``-d``, ``--network host``, no package-dispatch bootstrap, HF-cache mount only;
- baseline: blocking, ``--rm``, package dispatch with its own entry module,
  anonymous (labels but no ``--name``);
- offline: blocking, ``--rm``, package dispatch, ``LLEM_*`` forwarding, the full
  mount set.

Host ``NCCL_*`` forwarding is NOT one of the divergences: all three shapes carry
it. The server shape used to omit it, which was an omission rather than a
decision - the engine's tensor-parallel workers run inside the server container,
so a host needing ``NCCL_P2P_DISABLE=1`` for multi-GPU needs it there exactly as
the offline and baseline containers do.

Sibling files pin individual flags with targeted assertions. These tests instead
pin the ENTIRE argv, element for element, so a refactor of the shared core cannot
quietly reorder, drop, or add a flag in any shape: the whole emitted command is
the assertion. Everything host-dependent (package dir, materialised dispatch
assets, deps cache, uid/gid, ``$HOME``, and the ``LLEM_*``/``NCCL_*``
environment) is pinned by the ``pinned_host`` fixture so the literals below are
the real, complete commands.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from llenergymeasure.config.ssot import Engine
from llenergymeasure.infra.docker import command as cmd

_PKG_DIR = Path("/fake/site-packages/llenergymeasure")
_ENTRY_SCRIPT = Path("/fake/dispatch/container_entrypoint.sh")
_REQUIREMENTS = Path("/fake/dispatch/requirements.txt")
_DEPS_CACHE = Path("/fake/cache/deps")

_LABELS = {
    "llem.study_id": "abcdef1234",
    "llem.parent_pid": "4242",
    "llem.started_at": "2026-08-21T09:00:00+00:00",
}
_LABEL_ARGS = [
    "--label",
    "llem.study_id=abcdef1234",
    "--label",
    "llem.parent_pid=4242",
    "--label",
    "llem.started_at=2026-08-21T09:00:00+00:00",
]
# The four bind-mounts + env the in-container entrypoint script needs, shared by
# the offline and baseline shapes and deliberately absent from the server shape.
_PACKAGE_DISPATCH = [
    "-v",
    "/fake/site-packages/llenergymeasure:/llem-src/llenergymeasure:ro",
    "-v",
    "/fake/dispatch/requirements.txt:/llem-requirements.txt:ro",
    "-v",
    "/fake/dispatch/container_entrypoint.sh:/llem-entry.sh:ro",
    "-v",
    "/fake/cache/deps:/llem-runtime-deps",
    "-e",
    "PYTHONDONTWRITEBYTECODE=1",
]
_HF_MOUNT = [
    "-v",
    "/fake/home/.cache/huggingface:/root/.cache/huggingface",
    "-e",
    "HF_HOME=/root/.cache/huggingface",
]


class _Cfg:
    """Minimal stand-in for ExperimentConfig: the builder only reads ``engine``."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine


@pytest.fixture
def pinned_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin every host-dependent input the builders read.

    Without this the argv carries this machine's site-packages path, a
    per-process tempdir for the materialised dispatch assets, the real uid/gid,
    and whatever ``LLEM_*``/``NCCL_*`` happens to be exported - none of which can
    appear in a literal.
    """
    monkeypatch.setattr(cmd, "_resolve_package_dir", lambda: _PKG_DIR)
    monkeypatch.setattr(cmd, "_materialise_dispatch_assets", lambda: (_ENTRY_SCRIPT, _REQUIREMENTS))
    monkeypatch.setattr(cmd, "_ensure_deps_cache_dir", lambda: _DEPS_CACHE)
    monkeypatch.setattr(cmd.os, "getuid", lambda: 1000)
    monkeypatch.setattr(cmd.os, "getgid", lambda: 2000)
    for key in list(os.environ):
        if key.startswith(("LLEM_", "NCCL_")):
            monkeypatch.delenv(key, raising=False)
    # Drives docker_hf_cache_dir() and trt_build_cache_host_dir(), both of which
    # fall back to $HOME, plus the flashinfer JIT cache mount.
    monkeypatch.setenv("HOME", "/fake/home")


# ---------------------------------------------------------------------------
# Offline experiment dispatch: blocking, --rm, package dispatch, LLEM_*
# forwarding, full mount set, identity last.
# ---------------------------------------------------------------------------


def test_offline_argv_is_pinned_whole(pinned_host: None):
    """The plain transformers dispatch, every element."""
    argv = cmd.build_docker_cmd(
        image="llem-transformers:0.7.0",
        config=_Cfg(Engine.TRANSFORMERS),
        config_hash="cafebabe",
        exchange_dir="/tmp/llem-x",
        env_path=None,
        extra_mounts=[],
        container_name="llem-abcdef12-0001",
        labels=_LABELS,
        gpu_indices=None,
    )

    assert argv == [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "-v",
        "/tmp/llem-x:/run/llem",
        "-e",
        "LLEM_CONFIG_PATH=/run/llem/cafebabe_config.json",
        "--shm-size",
        "8g",
        *_HF_MOUNT,
        *_PACKAGE_DISPATCH,
        "-e",
        "LLEM_ENGINE=transformers",
        "-e",
        "LLEM_HOST_UID=1000",
        "-e",
        "LLEM_HOST_GID=2000",
        "--entrypoint",
        "/llem-entry.sh",
        "--name",
        "llem-abcdef12-0001",
        *_LABEL_ARGS,
        "llem-transformers:0.7.0",
    ]


def test_offline_argv_is_pinned_whole_with_env_file_mounts_and_pins(
    pinned_host: None, monkeypatch: pytest.MonkeyPatch
):
    """The tensorrt dispatch with every optional leg engaged.

    Exercises the secrets env-file, the TRT build-cache and flashinfer mounts,
    a user mount that COLLIDES with the auto HF-cache target (so the auto mount
    is suppressed and only the user's own appears), the quoted multi-device
    ``--gpus`` selector, host ``LLEM_*``/``NCCL_*`` forwarding, and the reserved
    exchange key that forwarding must skip.
    """
    monkeypatch.setenv("NCCL_P2P_DISABLE", "1")
    monkeypatch.setenv("LLEM_TRANSFORMERS_DEFAULT_DEVICE_MAP", "auto")
    # Reserved: the dispatch sets LLEM_CONFIG_PATH itself, and docker -e is
    # last-wins, so a host copy must NOT be forwarded.
    monkeypatch.setenv("LLEM_CONFIG_PATH", "/host/should/be/ignored")

    argv = cmd.build_docker_cmd(
        image="nvcr.io/nvidia/tritonserver:trtllm",
        config=_Cfg(Engine.TENSORRT),
        config_hash="d00dfeed",
        exchange_dir="/tmp/llem-y",
        env_path=Path("/tmp/llem-env"),
        extra_mounts=[("/host/hf", "/root/.cache/huggingface"), ("/data", "/data")],
        container_name=None,
        labels={},
        gpu_indices=[1, 3],
    )

    assert argv == [
        "docker",
        "run",
        "--rm",
        "--gpus",
        '"device=1,3"',
        "-v",
        "/tmp/llem-y:/run/llem",
        "-e",
        "LLEM_CONFIG_PATH=/run/llem/d00dfeed_config.json",
        "--shm-size",
        "8g",
        "--env-file",
        "/tmp/llem-env",
        "-v",
        "/fake/home/.cache/trt-llm:/root/.cache/trt-llm",
        "-e",
        "LLEM_TRT_BUILD_CACHE_PATH=/root/.cache/trt-llm",
        "-v",
        "/fake/home/.cache/flashinfer:/root/.cache/flashinfer",
        "-e",
        "LLEM_TRANSFORMERS_DEFAULT_DEVICE_MAP=auto",
        "-e",
        "NCCL_P2P_DISABLE=1",
        "-v",
        "/host/hf:/root/.cache/huggingface",
        "-v",
        "/data:/data",
        *_PACKAGE_DISPATCH,
        "-e",
        "LLEM_ENGINE=tensorrt",
        "-e",
        "LLEM_HOST_UID=1000",
        "-e",
        "LLEM_HOST_GID=2000",
        "--entrypoint",
        "/llem-entry.sh",
        "nvcr.io/nvidia/tritonserver:trtllm",
    ]


# ---------------------------------------------------------------------------
# Idle-baseline measurement: blocking, --rm, package dispatch pointed at the
# baseline entry module, NCCL forwarding, no shm-size, no HF mount, anonymous.
# ---------------------------------------------------------------------------


def test_baseline_argv_is_pinned_whole(pinned_host: None):
    """The baseline container, every element, with ownership labels."""
    argv = cmd.build_baseline_container_argv(
        image="llem-transformers:0.7.0",
        exchange_dir="/tmp/llem-b",
        gpu_indices=[0],
        engine="transformers",
        config_gpu_indices=None,
        labels=_LABELS,
    )

    assert argv == [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "-v",
        "/tmp/llem-b:/run/llem",
        "-e",
        "LLEM_BASELINE_SPEC_PATH=/run/llem/baseline_spec.json",
        "-e",
        "CUDA_VISIBLE_DEVICES=0",
        *_PACKAGE_DISPATCH,
        "-e",
        "LLEM_ENGINE=transformers",
        "-e",
        "LLEM_HOST_UID=1000",
        "-e",
        "LLEM_HOST_GID=2000",
        "-e",
        "LLEM_ENTRY_MODULE=llenergymeasure.entrypoints.baseline_measure",
        "--entrypoint",
        "/llem-entry.sh",
        *_LABEL_ARGS,
        "llem-transformers:0.7.0",
    ]
    # No --name: the baseline shape is deliberately anonymous, so the labels are
    # the only thing making it attributable to its study.
    assert "--name" not in argv
    # No --shm-size (it starts no shared-memory dataloader workers) and no HF
    # cache mount (it loads no weights).
    assert "--shm-size" not in argv
    assert "HF_HOME=/root/.cache/huggingface" not in argv


def test_baseline_argv_is_pinned_whole_with_gpu_pin_and_nccl(
    pinned_host: None, monkeypatch: pytest.MonkeyPatch
):
    """The two GPU params are distinct: host ``--gpus`` vs in-container CUDA ids.

    ``config_gpu_indices`` scopes the container to physical devices 2 and 3;
    ``gpu_indices`` are the logical in-container ids the samplers address, which
    re-enumerate from 0. Also pins NCCL forwarding and the unlabelled case.
    """
    monkeypatch.setenv("NCCL_DEBUG", "INFO")

    argv = cmd.build_baseline_container_argv(
        image="vllm/vllm-openai:v0.19.1",
        exchange_dir="/tmp/llem-b",
        gpu_indices=[0, 1],
        engine="vllm",
        config_gpu_indices=[2, 3],
        labels=None,
    )

    assert argv == [
        "docker",
        "run",
        "--rm",
        "--gpus",
        '"device=2,3"',
        "-v",
        "/tmp/llem-b:/run/llem",
        "-e",
        "LLEM_BASELINE_SPEC_PATH=/run/llem/baseline_spec.json",
        "-e",
        "CUDA_VISIBLE_DEVICES=0,1",
        "-e",
        "NCCL_DEBUG=INFO",
        *_PACKAGE_DISPATCH,
        "-e",
        "LLEM_ENGINE=vllm",
        "-e",
        "LLEM_HOST_UID=1000",
        "-e",
        "LLEM_HOST_GID=2000",
        "-e",
        "LLEM_ENTRY_MODULE=llenergymeasure.entrypoints.baseline_measure",
        "--entrypoint",
        "/llem-entry.sh",
        "vllm/vllm-openai:v0.19.1",
    ]


# ---------------------------------------------------------------------------
# Engine server: detached, NOT --rm, --network host, no package dispatch, HF
# mount plus host NCCL forwarding, identity ahead of the resource flags, engine
# command after image.
# ---------------------------------------------------------------------------


def test_server_argv_is_pinned_whole(pinned_host: None):
    """The named server container, every element, default shm size."""
    argv = cmd.build_server_container_argv(
        image="vllm/vllm-openai:v0.19.1",
        container_name="llem-vllm-server-abc123def456",
        gpu_indices=None,
        serve_args=["Qwen/Qwen2.5-0.5B", "--port", "8123"],
        shm_size=None,
        labels=_LABELS,
    )

    assert argv == [
        "docker",
        "run",
        "-d",
        "--network",
        "host",
        "--gpus",
        "all",
        "--name",
        "llem-vllm-server-abc123def456",
        *_LABEL_ARGS,
        "--shm-size",
        "8g",
        *_HF_MOUNT,
        "vllm/vllm-openai:v0.19.1",
        "Qwen/Qwen2.5-0.5B",
        "--port",
        "8123",
    ]
    # NOT --rm: a container that crashes during startup must survive so
    # `docker logs` can recover the diagnostic. Removal is the serving layer's
    # explicit job, not docker's.
    assert "--rm" not in argv
    # No package-dispatch bootstrap: the engine server runs the image's own
    # server, never a llenergymeasure entry module.
    assert "--entrypoint" not in argv
    # Host network namespace, so the port is selected in serve_args, not published.
    assert "-p" not in argv
    # The no-NCCL half of the forwarding pair: the fixture exports no NCCL_* var,
    # so the forwarding adds nothing and the argv is exactly what it always was.
    assert not any("NCCL" in token for token in argv)


def test_server_argv_is_pinned_whole_anonymous_with_gpu_pin(pinned_host: None):
    """The unlabelled, unnamed server with an explicit shm size and a GPU pin."""
    argv = cmd.build_server_container_argv(
        image="nvcr.io/nvidia/tritonserver:trtllm",
        container_name=None,
        gpu_indices=[2, 3],
        serve_args=["trtllm-serve", "m", "--port", "9000"],
        shm_size="16g",
        labels=None,
    )

    assert argv == [
        "docker",
        "run",
        "-d",
        "--network",
        "host",
        "--gpus",
        '"device=2,3"',
        "--shm-size",
        "16g",
        *_HF_MOUNT,
        "nvcr.io/nvidia/tritonserver:trtllm",
        "trtllm-serve",
        "m",
        "--port",
        "9000",
    ]


def test_server_argv_is_pinned_whole_with_host_nccl(
    pinned_host: None, monkeypatch: pytest.MonkeyPatch
):
    """Host ``NCCL_*`` forwarding, pinned in place inside the whole server argv.

    The set half of the forwarding pair. The engine's tensor-parallel workers run
    inside this container, so a PCIe host without functional GPU peer-to-peer
    needs ``NCCL_P2P_DISABLE=1`` here just as the offline and baseline containers
    do; without it the server hangs at its first NCCL collective and never
    becomes ready. Emitted as ``-e KEY=VALUE`` in sorted key order, the same form
    the other two shapes use, after the HF mount and still before the image.
    """
    monkeypatch.setenv("NCCL_P2P_DISABLE", "1")
    monkeypatch.setenv("NCCL_DEBUG", "INFO")

    argv = cmd.build_server_container_argv(
        image="vllm/vllm-openai:v0.19.1",
        container_name="llem-vllm-server-abc123def456",
        gpu_indices=[0, 1],
        serve_args=["Qwen/Qwen2.5-0.5B", "--port", "8123", "--tensor-parallel-size", "2"],
        shm_size=None,
        labels=_LABELS,
    )

    assert argv == [
        "docker",
        "run",
        "-d",
        "--network",
        "host",
        "--gpus",
        '"device=0,1"',
        "--name",
        "llem-vllm-server-abc123def456",
        *_LABEL_ARGS,
        "--shm-size",
        "8g",
        *_HF_MOUNT,
        "-e",
        "NCCL_DEBUG=INFO",
        "-e",
        "NCCL_P2P_DISABLE=1",
        "vllm/vllm-openai:v0.19.1",
        "Qwen/Qwen2.5-0.5B",
        "--port",
        "8123",
        "--tensor-parallel-size",
        "2",
    ]


# ---------------------------------------------------------------------------
# Cross-shape invariants the core enforces for all three at once.
# ---------------------------------------------------------------------------


def _all_three_shapes(labels: dict[str, str] | None = None) -> dict[str, list[str]]:
    return {
        "offline": cmd.build_docker_cmd(
            image="llem-transformers:0.7.0",
            config=_Cfg(Engine.TRANSFORMERS),
            config_hash="cafebabe",
            exchange_dir="/tmp/llem-x",
            env_path=None,
            extra_mounts=[],
            container_name="llem-abcdef12-0001",
            labels=labels or {},
            gpu_indices=None,
        ),
        "baseline": cmd.build_baseline_container_argv(
            image="llem-transformers:0.7.0",
            exchange_dir="/tmp/llem-b",
            gpu_indices=[0],
            engine="transformers",
            labels=labels,
        ),
        "server": cmd.build_server_container_argv(
            image="vllm/vllm-openai:v0.19.1",
            container_name="llem-vllm-server-abc",
            gpu_indices=None,
            serve_args=["m", "--port", "8123"],
            labels=labels,
        ),
    }


def test_every_shape_opens_with_docker_run_and_one_gpus_selector(pinned_host: None):
    """No shape may skip the GPU selector or bury the subcommand."""
    for shape, argv in _all_three_shapes().items():
        assert argv[:2] == ["docker", "run"], shape
        assert argv.count("--gpus") == 1, shape


def test_every_shape_puts_ownership_labels_before_the_image(pinned_host: None):
    """Labels after the image reference would be swallowed as container args.

    The study-scoped cleanup filters on ``llem.study_id`` and the orphan reaper
    on ``llem.parent_pid``; a label docker never saw makes the container
    invisible to both.
    """
    images = {
        "offline": "llem-transformers:0.7.0",
        "baseline": "llem-transformers:0.7.0",
        "server": "vllm/vllm-openai:v0.19.1",
    }
    for shape, argv in _all_three_shapes(labels=_LABELS).items():
        image_index = argv.index(images[shape])
        assert argv.count("--label") == len(_LABELS), shape
        for position, token in enumerate(argv):
            if token == "--label":
                assert position < image_index, shape


def test_every_shape_forwards_host_nccl_env(pinned_host: None, monkeypatch: pytest.MonkeyPatch):
    """No shape may skip host ``NCCL_*`` forwarding.

    The server shape's omission is the defect this guards against recurring: the
    engine's tensor-parallel workers run inside the server container, so a host
    that needs ``NCCL_P2P_DISABLE=1`` for multi-GPU needs it in every shape that
    runs a multi-GPU process, not only the offline ones.
    """
    monkeypatch.setenv("NCCL_P2P_DISABLE", "1")

    for shape, argv in _all_three_shapes().items():
        position = argv.index("NCCL_P2P_DISABLE=1")
        assert argv[position - 1] == "-e", shape


def test_removal_policy_is_exclusive_per_shape(pinned_host: None):
    """``--rm`` and ``-d`` are the two policies and never coexist."""
    shapes = _all_three_shapes()
    for shape in ("offline", "baseline"):
        assert "--rm" in shapes[shape] and "-d" not in shapes[shape], shape
    assert "-d" in shapes["server"] and "--rm" not in shapes["server"]
