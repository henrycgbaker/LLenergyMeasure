"""Tests for the built-in Docker image registry and runner value parsing."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from llenergymeasure.config.ssot import ENV_IMAGE_PREFIX, Engine

# ---------------------------------------------------------------------------
# parse_runner_value
# ---------------------------------------------------------------------------


class TestParseRunnerValue:
    def test_local_returns_local_none(self):
        from llenergymeasure.infra.image_registry import parse_runner_value

        assert parse_runner_value("local") == ("local", None)

    def test_docker_returns_docker_none(self):
        from llenergymeasure.infra.image_registry import parse_runner_value

        assert parse_runner_value("docker") == ("docker", None)

    def test_docker_colon_image_returns_docker_with_image(self):
        from llenergymeasure.infra.image_registry import parse_runner_value

        assert parse_runner_value("docker:custom/img:v1") == ("docker", "custom/img:v1")

    def test_docker_colon_ghcr_image(self):
        from llenergymeasure.infra.image_registry import parse_runner_value

        result = parse_runner_value("docker:ghcr.io/org/vllm:1.19.0-cuda12")
        assert result == ("docker", "ghcr.io/org/vllm:1.19.0-cuda12")

    def test_docker_colon_empty_string_raises(self):
        from llenergymeasure.infra.image_registry import parse_runner_value

        with pytest.raises(ValueError, match="empty image name"):
            parse_runner_value("docker:")

    def test_unknown_runner_type_raises(self):
        from llenergymeasure.infra.image_registry import parse_runner_value

        with pytest.raises(ValueError, match="Unrecognised runner value"):
            parse_runner_value("kubernetes")


# ---------------------------------------------------------------------------
# get_default_image
# ---------------------------------------------------------------------------


class TestGetDefaultImage:
    def test_prefers_local_image_when_available(self):
        from llenergymeasure.infra.image_registry import get_default_image

        with patch("llenergymeasure.infra.image_registry._image_exists_locally", return_value=True):
            image = get_default_image("vllm")

        assert image == "llenergymeasure:vllm"

    def test_falls_back_to_ghcr_when_no_local_image(self):
        from llenergymeasure.infra.image_registry import get_default_image

        with patch(
            "llenergymeasure.infra.image_registry._image_exists_locally", return_value=False
        ):
            image = get_default_image("vllm")

        assert image.startswith("ghcr.io/henrycgbaker/llenergymeasure/vllm:v")

    def test_fallback_to_latest_when_version_empty(self):
        from llenergymeasure.infra.image_registry import get_default_image

        with (
            patch("llenergymeasure.infra.image_registry._image_exists_locally", return_value=False),
            patch("llenergymeasure._version.__version__", ""),
        ):
            image = get_default_image("transformers")

        assert image.endswith(":vlatest")

    def test_engine_name_included_in_image(self):
        from llenergymeasure.infra.image_registry import get_default_image

        for engine in Engine:
            image = get_default_image(engine)
            assert engine in image, f"Expected engine {engine!r} in image {image!r}"

    def test_ghcr_image_includes_package_version(self):
        from llenergymeasure import __version__
        from llenergymeasure.infra.image_registry import get_default_image

        with patch(
            "llenergymeasure.infra.image_registry._image_exists_locally", return_value=False
        ):
            image = get_default_image("vllm")

        assert f"v{__version__}" in image


# ---------------------------------------------------------------------------
# show_image_resolution
# ---------------------------------------------------------------------------


class TestShowImageResolution:
    def test_prints_all_engines(self, capsys):
        from llenergymeasure.infra.image_registry import show_image_resolution

        with patch("llenergymeasure.infra.image_registry._image_exists_locally", return_value=True):
            show_image_resolution()

        output = capsys.readouterr().out
        assert "transformers" in output
        assert "vllm" in output
        assert "tensorrt" in output

    def test_shows_local_source(self, capsys):
        from llenergymeasure.infra.image_registry import show_image_resolution

        with patch("llenergymeasure.infra.image_registry._image_exists_locally", return_value=True):
            show_image_resolution()

        output = capsys.readouterr().out
        assert "(local_build)" in output

    def test_shows_registry_source(self, capsys):
        from llenergymeasure.infra.image_registry import show_image_resolution

        with patch(
            "llenergymeasure.infra.image_registry._image_exists_locally", return_value=False
        ):
            show_image_resolution()

        output = capsys.readouterr().out
        assert "(registry)" in output


# ---------------------------------------------------------------------------
# resolve_image
# ---------------------------------------------------------------------------


class TestResolveImage:
    def test_env_var_takes_highest_precedence(self, monkeypatch):
        from llenergymeasure.infra.image_registry import resolve_image

        monkeypatch.setenv(f"{ENV_IMAGE_PREFIX}VLLM", "custom/env-image:v1")

        image, source = resolve_image(
            "vllm",
            spec_image="spec-image:v1",
            yaml_images={"vllm": "yaml-image:v1"},
            user_config_images={"vllm": "uc-image:v1"},
        )

        assert image == "custom/env-image:v1"
        assert source == "env"

    def test_yaml_images_second_precedence(self):
        from llenergymeasure.infra.image_registry import resolve_image

        image, source = resolve_image(
            "vllm",
            spec_image="spec-image:v1",
            yaml_images={"vllm": "yaml-image:v1"},
            user_config_images={"vllm": "uc-image:v1"},
        )

        assert image == "yaml-image:v1"
        assert source == "yaml"

    def test_spec_image_third_precedence(self):
        from llenergymeasure.infra.image_registry import resolve_image

        image, source = resolve_image(
            "vllm",
            spec_image="spec-image:v1",
            user_config_images={"vllm": "uc-image:v1"},
        )

        assert image == "spec-image:v1"
        assert source == "runner_override"

    def test_user_config_images_fourth_precedence(self):
        from llenergymeasure.infra.image_registry import resolve_image

        with patch(
            "llenergymeasure.infra.image_registry._image_exists_locally", return_value=False
        ):
            image, source = resolve_image(
                "vllm",
                user_config_images={"vllm": "uc-image:v1"},
            )

        assert image == "uc-image:v1"
        assert source == "user_config"

    def test_smart_default_local_build(self):
        from llenergymeasure.infra.image_registry import resolve_image

        with patch("llenergymeasure.infra.image_registry._image_exists_locally", return_value=True):
            image, source = resolve_image("vllm")

        assert image == "llenergymeasure:vllm"
        assert source == "local_build"

    def test_smart_default_registry_fallback(self):
        from llenergymeasure.infra.image_registry import resolve_image

        with patch(
            "llenergymeasure.infra.image_registry._image_exists_locally", return_value=False
        ):
            image, source = resolve_image("vllm")

        assert image.startswith("ghcr.io/henrycgbaker/llenergymeasure/vllm:v")
        assert source == "registry"

    def test_env_var_case_insensitive_engine(self, monkeypatch):
        from llenergymeasure.infra.image_registry import resolve_image

        monkeypatch.setenv(f"{ENV_IMAGE_PREFIX}TRANSFORMERS", "my/pytorch:v1")
        image, source = resolve_image("transformers")
        assert image == "my/pytorch:v1"
        assert source == "env"

    def test_yaml_images_ignores_other_engines(self):
        from llenergymeasure.infra.image_registry import resolve_image

        with patch("llenergymeasure.infra.image_registry._image_exists_locally", return_value=True):
            image, source = resolve_image(
                "vllm",
                yaml_images={"transformers": "pytorch-image:v1"},
            )

        assert image == "llenergymeasure:vllm"
        assert source == "local_build"
