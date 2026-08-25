"""Unit tests for the TensorRT-LLM ServerCapable adapter (host-only, no GPU, no docker).

Mirrors the vLLM adapter coverage matrix: command construction is asserted pure
(incl. the explicit ``trtllm-serve`` verb the un-baked NGC image requires and the
HF-cache mount on the container leg); launch routing is asserted with the infra
launchers mocked; and the FULL protocol is driven through the plugin's own
ServerCapable methods against the asyncio stub server (process leg).
"""

from __future__ import annotations

import sys
from pathlib import Path

from llenergymeasure.engines.protocol import EnginePlugin, ServerCapable
from llenergymeasure.engines.tensorrt import _serving
from llenergymeasure.engines.tensorrt.plugin import TensorRTEngine
from llenergymeasure.infra.docker.command import build_server_container_argv
from llenergymeasure.serving import lifecycle as sl
from llenergymeasure.serving.types import ServerHandle, ServerPlacement
from tests.conftest import make_config

STUB_SERVER = Path(__file__).parent / "_stub_server.py"


# ---------------------------------------------------------------------------
# Protocol conformance (offline contract intact; server extension present)
# ---------------------------------------------------------------------------


def test_tensorrt_engine_satisfies_both_protocols():
    engine = TensorRTEngine()
    assert isinstance(engine, EnginePlugin)  # offline single-call contract untouched
    assert isinstance(engine, ServerCapable)  # additive server-lifecycle extension


def test_offline_inference_method_still_present():
    """The additive extension did not remove or rename the offline surface."""
    engine = TensorRTEngine()
    for method in ("load_model", "run_inference", "run_warmup_prompt", "cleanup"):
        assert callable(getattr(engine, method))


# ---------------------------------------------------------------------------
# TRT-LLM command construction (pure)
# ---------------------------------------------------------------------------


def test_serve_command_includes_the_trtllm_serve_verb():
    """Unlike vllm, the NGC image is NOT entrypoint-baked, so the verb leads the command."""
    assert _serving.serve_command("Qwen/Qwen2.5-0.5B", 8000) == [
        "trtllm-serve",
        "Qwen/Qwen2.5-0.5B",
        "--port",
        "8000",
    ]
    assert _serving.serve_command("m", 8000, ["--tp_size", "2"])[-2:] == ["--tp_size", "2"]


def test_completions_probe_defaults():
    probe = _serving.build_completions_probe("m")
    assert probe.path == "/v1/completions"
    assert probe.method == "POST"
    assert probe.payload == {"model": "m", "prompt": "ready?", "max_tokens": 1, "temperature": 0.0}


def test_container_argv_carries_ruled_flags_and_the_explicit_verb():
    """The container leg's docker argv: image, --gpus, --network host, HF mount, verb+port."""
    argv = build_server_container_argv(
        image="nvcr.io/nvidia/tensorrt-llm/release:1.2.1",
        container_name="llem-tensorrt-server-1",
        gpu_indices=[2],
        serve_args=_serving.serve_command("Qwen/Qwen2.5-0.5B", 8000),
        shm_size="8g",
    )
    assert "nvcr.io/nvidia/tensorrt-llm/release:1.2.1" in argv
    assert "--rm" not in argv  # crashed container must survive for docker-logs diagnostics
    assert argv[argv.index("--network") + 1] == "host"
    assert argv[argv.index("--gpus") + 1] == "device=2"
    assert "HF_HOME=/root/.cache/huggingface" in argv
    # The image is followed by the FULL command (verb included) - no baked entrypoint.
    img_idx = argv.index("nvcr.io/nvidia/tensorrt-llm/release:1.2.1")
    assert argv[img_idx + 1 :] == ["trtllm-serve", "Qwen/Qwen2.5-0.5B", "--port", "8000"]


# ---------------------------------------------------------------------------
# launch() routing (infra launchers mocked - no docker, no subprocess)
# ---------------------------------------------------------------------------


def test_launch_routes_to_container_leg(monkeypatch):
    captured: dict = {}

    def fake_launch_container(argv, *, base_url, engine, container_name):
        captured.update(argv=argv, base_url=base_url, engine=engine)
        return ServerHandle(base_url=base_url, engine=engine, container_name=container_name)

    monkeypatch.setattr(sl, "allocate_free_port", lambda: 9111)
    monkeypatch.setattr(sl, "launch_container_server", fake_launch_container)
    monkeypatch.setattr(
        "llenergymeasure.infra.image_registry.get_default_image",
        lambda engine: "nvcr.io/nvidia/tensorrt-llm/release:1.2.1",
    )

    handle = TensorRTEngine().launch(
        make_config(engine="tensorrt", model="Qwen/Qwen2.5-0.5B"),
        ServerPlacement(mode="container", image=None, gpu_indices=[2]),
    )

    argv = captured["argv"]
    assert captured["engine"] == "tensorrt"
    assert "nvcr.io/nvidia/tensorrt-llm/release:1.2.1" in argv
    assert argv[argv.index("--network") + 1] == "host"
    assert argv[argv.index("--gpus") + 1] == "device=2"
    img_idx = argv.index("nvcr.io/nvidia/tensorrt-llm/release:1.2.1")
    assert argv[img_idx + 1 :] == ["trtllm-serve", "Qwen/Qwen2.5-0.5B", "--port", "9111"]
    assert handle.base_url == "http://127.0.0.1:9111"


def test_launch_container_uses_explicit_image_override(monkeypatch):
    captured: dict = {}

    def fake_launch_container(argv, *, base_url, engine, container_name):
        captured["argv"] = argv
        return ServerHandle(base_url=base_url, engine=engine, container_name=container_name)

    monkeypatch.setattr(sl, "allocate_free_port", lambda: 9111)
    monkeypatch.setattr(sl, "launch_container_server", fake_launch_container)

    TensorRTEngine().launch(
        make_config(engine="tensorrt", model="m"),
        ServerPlacement(mode="container", image="my/trt:tag", gpu_indices=None),
    )
    assert "my/trt:tag" in captured["argv"]  # no registry resolution when pinned


def test_launch_routes_to_process_leg(monkeypatch):
    captured: dict = {}

    def fake_launch_process(argv, *, base_url, engine, log_path, gpu_indices=None):
        captured.update(argv=argv, base_url=base_url, engine=engine, gpu_indices=gpu_indices)
        return ServerHandle(base_url=base_url, engine=engine, log_path=log_path)

    monkeypatch.setattr(sl, "allocate_free_port", lambda: 9222)
    monkeypatch.setattr(sl, "launch_process_server", fake_launch_process)

    handle = TensorRTEngine().launch(
        make_config(engine="tensorrt", model="m"),
        ServerPlacement(mode="process"),
    )
    assert captured["argv"] == ["trtllm-serve", "m", "--port", "9222"]
    assert handle.base_url == "http://127.0.0.1:9222"


def test_launch_scopes_the_process_leg_to_the_placement_gpus(monkeypatch):
    """The placement's physical devices reach the process launch, not just the container."""
    captured: dict = {}

    def fake_launch_process(argv, *, base_url, engine, log_path, gpu_indices=None):
        captured.update(gpu_indices=gpu_indices)
        return ServerHandle(base_url=base_url, engine=engine, log_path=log_path)

    monkeypatch.setattr(sl, "allocate_free_port", lambda: 9222)
    monkeypatch.setattr(sl, "launch_process_server", fake_launch_process)

    TensorRTEngine().launch(
        make_config(engine="tensorrt", model="m"),
        ServerPlacement(mode="process", gpu_indices=[2, 3]),
    )
    assert captured["gpu_indices"] == [2, 3]


# ---------------------------------------------------------------------------
# Full protocol through the plugin surface (process leg, stub server)
# ---------------------------------------------------------------------------


def test_full_lifecycle_through_plugin(monkeypatch, tmp_path):
    """Drive launch / await_ready / shutdown through TensorRTEngine's ServerCapable API."""
    # Substitute the stub server for `trtllm-serve` (host-only, no TRT-LLM install).
    monkeypatch.setattr(
        _serving,
        "serve_command",
        lambda model, port, extra=None: [sys.executable, str(STUB_SERVER), "--port", str(port)],
    )
    monkeypatch.setattr(
        sl, "default_server_log_path", lambda engine, port: tmp_path / f"{engine}-{port}.log"
    )

    engine = TensorRTEngine()
    config = make_config(engine="tensorrt", model="stub-model")
    handle = engine.launch(config, ServerPlacement(mode="process"))
    try:
        engine.await_ready(handle, _serving.build_completions_probe("stub-model"), timeout=20.0)
        assert "stub server listening" in handle.read_logs()
    finally:
        engine.shutdown(handle)

    assert handle.process is not None
    assert handle.process.poll() is not None  # reaped by shutdown, no orphan
