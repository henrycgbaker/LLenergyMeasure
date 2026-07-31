"""Tests for the server-mode call-site wiring in the study orchestrator (SM9 PR-B).

Covers the server-capable-entry-path contract: the orchestration choke point
overlays the user-config server warmup onto every server experiment (so the
ServerSession reads the overlay-resolved protocol on the bypass paths that skip
api.load_study), leaves offline configs untouched, and is idempotent.
"""

from __future__ import annotations

from llenergymeasure.config.models import (
    ExperimentConfig,
    ServerWarmupConfig,
    StudyConfig,
)
from llenergymeasure.config.user_config import UserConfig, UserServerConfig
from llenergymeasure.domain.experiment import compute_declared_config_hash
from llenergymeasure.study.orchestration import _apply_server_warmup_overlay_to_study


def _server_exp() -> ExperimentConfig:
    return ExperimentConfig(
        task={"model": "gpt2"},
        engine="vllm",
        serving_mode="server",
        server={"traffic": {"rate": 10, "window_seconds": 60}},
    )


def _offline_exp() -> ExperimentConfig:
    return ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        serving_mode="offline",
    )


def _user(mode: str = "fixed") -> UserConfig:
    return UserConfig(server=UserServerConfig(warmup=ServerWarmupConfig(mode=mode)))


def test_overlay_applies_to_server_leaves_offline_untouched() -> None:
    server, offline = _server_exp(), _offline_exp()
    study = StudyConfig(experiments=[server, offline])
    offline_declared_before = compute_declared_config_hash(offline)

    _apply_server_warmup_overlay_to_study(study, _user("fixed"))

    # Server experiment now resolves the overlaid (fixed) protocol...
    resolved = server.resolved_server_warmup()
    assert resolved is not None and resolved.mode == "fixed"
    # ...while the offline experiment is untouched (no server warmup, hash stable).
    assert offline.resolved_server_warmup() is None
    assert compute_declared_config_hash(offline) == offline_declared_before


def test_overlay_is_idempotent() -> None:
    server = _server_exp()
    study = StudyConfig(experiments=[server])
    user = _user("fixed")

    _apply_server_warmup_overlay_to_study(study, user)
    first = server.resolved_server_warmup()
    _apply_server_warmup_overlay_to_study(study, user)
    second = server.resolved_server_warmup()

    assert first is not None and second is not None
    assert first.mode == second.mode == "fixed"


def test_overlay_noop_when_no_server_experiments() -> None:
    offline = _offline_exp()
    study = StudyConfig(experiments=[offline])
    # No server experiment: the helper returns early (a plain UserConfig with no
    # server section is harmless either way).
    _apply_server_warmup_overlay_to_study(study, UserConfig())
    assert offline.resolved_server_warmup() is None


def test_overlay_noop_when_user_config_has_no_warmup_layer() -> None:
    # A server experiment but no user-config warmup layer: resolved falls back to
    # the declared server.warmup (byte-identical to no-overlay behaviour).
    server = _server_exp()
    study = StudyConfig(experiments=[server])
    _apply_server_warmup_overlay_to_study(study, UserConfig())
    resolved = server.resolved_server_warmup()
    assert resolved is not None and resolved.mode == "composite"  # the built-in default
