"""Unit tests for the vLLM ServerCapable adapter (host-only, no GPU, no docker).

Command construction is asserted pure; launch routing is asserted with the
infra launchers mocked; and the FULL protocol is driven through the plugin's
own ServerCapable methods against the asyncio stub server (process leg).
"""

from __future__ import annotations

import sys
from pathlib import Path

from llenergymeasure.engines.protocol import EnginePlugin, ServerCapable
from llenergymeasure.engines.vllm import _serving
from llenergymeasure.engines.vllm.plugin import VLLMEngine
from llenergymeasure.infra import server_lifecycle as sl
from llenergymeasure.infra.server_lifecycle import ServerHandle, ServerPlacement
from tests.conftest import make_config

STUB_SERVER = Path(__file__).parent / "_stub_server.py"


# ---------------------------------------------------------------------------
# Protocol conformance (offline contract intact; server extension present)
# ---------------------------------------------------------------------------


def test_vllm_engine_satisfies_both_protocols():
    engine = VLLMEngine()
    assert isinstance(engine, EnginePlugin)  # offline single-call contract untouched
    assert isinstance(engine, ServerCapable)  # additive server-lifecycle extension


def test_offline_inference_method_still_present():
    """The additive extension did not remove or rename the offline surface."""
    engine = VLLMEngine()
    for method in ("load_model", "run_inference", "run_warmup_prompt", "cleanup"):
        assert callable(getattr(engine, method))


# ---------------------------------------------------------------------------
# vLLM command construction (pure)
# ---------------------------------------------------------------------------


def test_serve_args_shape():
    assert _serving.serve_args("Qwen/Qwen2.5-0.5B", 8000) == [
        "Qwen/Qwen2.5-0.5B",
        "--port",
        "8000",
    ]
    assert _serving.serve_args("m", 8000, ["--max-model-len", "2048"])[-2:] == [
        "--max-model-len",
        "2048",
    ]


def test_process_argv_is_vllm_serve():
    assert _serving.process_argv("m", 8000) == ["vllm", "serve", "m", "--port", "8000"]


def test_completions_probe_defaults():
    probe = _serving.build_completions_probe("m")
    assert probe.path == "/v1/completions"
    assert probe.method == "POST"
    assert probe.payload == {"model": "m", "prompt": "ready?", "max_tokens": 1, "temperature": 0.0}


def test_container_argv_carries_ruled_flags():
    """The container leg's docker argv: image, --gpus, --network host, port."""
    argv = sl.build_server_container_argv(
        image="vllm/vllm-openai:v0.19.1",
        container_name="llem-vllm-server-1",
        gpu_indices=[2],
        serve_args=_serving.serve_args("Qwen/Qwen2.5-0.5B", 8000),
        shm_size="8g",
    )
    assert "vllm/vllm-openai:v0.19.1" in argv
    assert "--rm" not in argv  # crashed container must survive for docker-logs diagnostics
    assert argv[argv.index("--network") + 1] == "host"
    assert argv[argv.index("--gpus") + 1] == "device=2"
    # The HF cache mount rides every server-container launch (retrofit): HF_HOME
    # set so the launched server reuses downloaded weights instead of re-pulling.
    assert "HF_HOME=/root/.cache/huggingface" in argv
    # The entrypoint (vllm serve) supplies the verb; the image is followed by
    # the serve args only, with the port inside them.
    img_idx = argv.index("vllm/vllm-openai:v0.19.1")
    assert argv[img_idx + 1 :] == ["Qwen/Qwen2.5-0.5B", "--port", "8000"]


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
        lambda engine: "vllm/vllm-openai:v0.19.1",
    )

    handle = VLLMEngine().launch(
        make_config(engine="vllm", model="Qwen/Qwen2.5-0.5B"),
        ServerPlacement(mode="container", image=None, gpu_indices=[2]),
    )

    argv = captured["argv"]
    assert captured["engine"] == "vllm"
    assert "vllm/vllm-openai:v0.19.1" in argv
    assert argv[argv.index("--network") + 1] == "host"
    assert argv[argv.index("--gpus") + 1] == "device=2"
    img_idx = argv.index("vllm/vllm-openai:v0.19.1")
    assert argv[img_idx + 1 :] == ["Qwen/Qwen2.5-0.5B", "--port", "9111"]
    assert handle.base_url == "http://127.0.0.1:9111"


def test_launch_container_uses_explicit_image_override(monkeypatch):
    captured: dict = {}

    def fake_launch_container(argv, *, base_url, engine, container_name):
        captured["argv"] = argv
        return ServerHandle(base_url=base_url, engine=engine, container_name=container_name)

    monkeypatch.setattr(sl, "allocate_free_port", lambda: 9111)
    monkeypatch.setattr(sl, "launch_container_server", fake_launch_container)

    VLLMEngine().launch(
        make_config(engine="vllm", model="m"),
        ServerPlacement(mode="container", image="my/custom:tag", gpu_indices=None),
    )
    assert "my/custom:tag" in captured["argv"]  # no registry resolution when pinned


def test_launch_routes_to_process_leg(monkeypatch):
    captured: dict = {}

    def fake_launch_process(argv, *, base_url, engine, log_path):
        captured.update(argv=argv, base_url=base_url, engine=engine)
        return ServerHandle(base_url=base_url, engine=engine, log_path=log_path)

    monkeypatch.setattr(sl, "allocate_free_port", lambda: 9222)
    monkeypatch.setattr(sl, "launch_process_server", fake_launch_process)

    handle = VLLMEngine().launch(
        make_config(engine="vllm", model="m"),
        ServerPlacement(mode="process"),
    )
    assert captured["argv"] == ["vllm", "serve", "m", "--port", "9222"]
    assert handle.base_url == "http://127.0.0.1:9222"


# ---------------------------------------------------------------------------
# Full protocol through the plugin surface (process leg, stub server)
# ---------------------------------------------------------------------------


def test_full_lifecycle_through_plugin(monkeypatch, tmp_path):
    """Drive launch / await_ready / shutdown through VLLMEngine's ServerCapable API."""
    # Substitute the stub server for `vllm serve` (host-only, no vLLM install).
    monkeypatch.setattr(
        _serving,
        "process_argv",
        lambda model, port, extra=None: [sys.executable, str(STUB_SERVER), "--port", str(port)],
    )
    monkeypatch.setattr(
        sl, "default_server_log_path", lambda engine, port: tmp_path / f"{engine}-{port}.log"
    )

    engine = VLLMEngine()
    config = make_config(engine="vllm", model="stub-model")
    handle = engine.launch(config, ServerPlacement(mode="process"))
    try:
        engine.await_ready(handle, _serving.build_completions_probe("stub-model"), timeout=20.0)
        assert "stub server listening" in handle.read_logs()
    finally:
        engine.shutdown(handle)

    assert handle.process is not None
    assert handle.process.poll() is not None  # reaped by shutdown, no orphan
