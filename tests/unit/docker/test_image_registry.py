"""Tests for the built-in Docker image registry and runner value parsing."""

from __future__ import annotations

import logging
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

    def test_transformers_default_is_ghcr_at_package_version(self):
        """Transformers keeps the first-party GHCR image at the package version."""
        from llenergymeasure import __version__
        from llenergymeasure.infra.image_registry import get_default_image

        with patch(
            "llenergymeasure.infra.image_registry._image_exists_locally", return_value=False
        ):
            image = get_default_image("transformers")

        assert image == f"ghcr.io/henrycgbaker/llenergymeasure/transformers:v{__version__}"

    def test_transformers_falls_back_to_latest_when_version_empty(self):
        from llenergymeasure.infra.image_registry import get_default_image

        with (
            patch("llenergymeasure.infra.image_registry._image_exists_locally", return_value=False),
            patch("llenergymeasure._version.__version__", ""),
        ):
            image = get_default_image("transformers")

        assert image == "ghcr.io/henrycgbaker/llenergymeasure/transformers:vlatest"

    def test_vllm_default_is_upstream_openai_at_pinned_version(self):
        """vLLM resolves to the upstream Docker Hub image at the pinned engine version."""
        from llenergymeasure.infra.image_registry import get_default_image
        from llenergymeasure.infra.version_handshake import read_bundled_engine_version

        expected_version = read_bundled_engine_version("vllm")
        assert expected_version, "vllm bundled engine version must resolve in-repo"

        with patch(
            "llenergymeasure.infra.image_registry._image_exists_locally", return_value=False
        ):
            image = get_default_image("vllm")

        assert image == f"vllm/vllm-openai:v{expected_version}"

    def test_tensorrt_default_is_upstream_ngc_at_pinned_version(self):
        """TensorRT-LLM resolves to the upstream NGC image (no ``v`` prefix)."""
        from llenergymeasure.infra.image_registry import get_default_image
        from llenergymeasure.infra.version_handshake import read_bundled_engine_version

        expected_version = read_bundled_engine_version("tensorrt")
        assert expected_version, "tensorrt bundled engine version must resolve in-repo"

        with patch(
            "llenergymeasure.infra.image_registry._image_exists_locally", return_value=False
        ):
            image = get_default_image("tensorrt")

        assert image == f"nvcr.io/nvidia/tensorrt-llm/release:{expected_version}"

    def test_engine_name_included_in_image(self):
        from llenergymeasure.infra.image_registry import get_default_image

        with patch(
            "llenergymeasure.infra.image_registry._image_exists_locally", return_value=False
        ):
            for engine in Engine:
                image = get_default_image(engine)
                assert engine in image, f"Expected engine {engine!r} in image {image!r}"

    def test_hard_error_when_pinned_version_unavailable(self):
        """A partial/broken wheel yields an actionable error, never a 404 tag."""
        from llenergymeasure.infra.image_registry import get_default_image
        from llenergymeasure.utils.exceptions import ConfigError

        with (
            patch("llenergymeasure.infra.image_registry._image_exists_locally", return_value=False),
            patch(
                "llenergymeasure.infra.version_handshake.read_bundled_engine_version",
                return_value=None,
            ),
            pytest.raises(ConfigError, match=r'runners\.vllm to "docker:'),
        ):
            get_default_image("vllm")


# ---------------------------------------------------------------------------
# Local-tag shadow warning
# ---------------------------------------------------------------------------


class TestLocalImageShadowWarning:
    """A local bare tag winning resolution warns once, naming the bypassed default."""

    @pytest.fixture(autouse=True)
    def _clear_shadow_dedup(self):
        """``_warn_local_shadow`` is @lru_cache'd for once-per-process dedup; clear
        it so a warning from one test does not suppress the next."""
        from llenergymeasure.infra.image_registry import _warn_local_shadow

        _warn_local_shadow.cache_clear()
        yield
        _warn_local_shadow.cache_clear()

    def test_warns_and_names_local_tag_and_bypassed_default(self, caplog):
        from llenergymeasure.infra.image_registry import get_default_image
        from llenergymeasure.infra.version_handshake import read_bundled_engine_version

        bypassed = f"vllm/vllm-openai:v{read_bundled_engine_version('vllm')}"

        with (
            patch("llenergymeasure.infra.image_registry._image_exists_locally", return_value=True),
            caplog.at_level(logging.WARNING, logger="llenergymeasure.infra.image_registry"),
        ):
            image = get_default_image("vllm")

        # Resolution result is unchanged: local tag still wins.
        assert image == "llenergymeasure:vllm"

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        text = warnings[0].getMessage()
        assert "llenergymeasure:vllm" in text  # (1) the local tag used
        assert bypassed in text  # (2) the version-pinned default it bypassed
        assert "docker rmi llenergymeasure:vllm" in text  # (3) the remedy

    def test_no_warning_when_local_tag_absent(self, caplog):
        from llenergymeasure.infra.image_registry import get_default_image
        from llenergymeasure.infra.version_handshake import read_bundled_engine_version

        with (
            patch("llenergymeasure.infra.image_registry._image_exists_locally", return_value=False),
            caplog.at_level(logging.WARNING, logger="llenergymeasure.infra.image_registry"),
        ):
            image = get_default_image("vllm")

        # Resolution result is unchanged: pinned default wins, no shadow.
        assert image == f"vllm/vllm-openai:v{read_bundled_engine_version('vllm')}"
        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []

    def test_deduplicated_across_repeated_calls(self, caplog):
        from llenergymeasure.infra.image_registry import get_default_image

        with (
            patch("llenergymeasure.infra.image_registry._image_exists_locally", return_value=True),
            caplog.at_level(logging.WARNING, logger="llenergymeasure.infra.image_registry"),
        ):
            get_default_image("vllm")
            get_default_image("vllm")
            get_default_image("vllm")

        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


class TestShadowedDefaultImage:
    """``shadowed_default_image`` names the bypassed default only for the local tag."""

    def test_returns_bypassed_default_for_local_tag(self):
        from llenergymeasure.infra.image_registry import shadowed_default_image
        from llenergymeasure.infra.version_handshake import read_bundled_engine_version

        result = shadowed_default_image("vllm", "llenergymeasure:vllm")
        assert result == f"vllm/vllm-openai:v{read_bundled_engine_version('vllm')}"

    def test_none_when_resolved_image_is_not_the_local_tag(self):
        from llenergymeasure.infra.image_registry import shadowed_default_image

        assert shadowed_default_image("vllm", "vllm/vllm-openai:v0.19.1") is None


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

        assert image.startswith("vllm/vllm-openai:v")
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


# ---------------------------------------------------------------------------
# resolve_image_digest
# ---------------------------------------------------------------------------

_MISSING = object()


def _fake_inspect(*, repo_digests: object, returncode: int = 0):
    """Build a fake ``docker image inspect`` CompletedProcess with RepoDigests."""
    import json
    import subprocess
    from unittest.mock import MagicMock

    body: list[dict[str, object]] = [{"Id": "sha256:local-config-digest"}]
    if repo_digests is not _MISSING:
        body[0]["RepoDigests"] = repo_digests
    result = MagicMock(spec=subprocess.CompletedProcess)
    result.returncode = returncode
    result.stdout = json.dumps(body).encode("utf-8")
    return result


class TestResolveImageDigest:
    """resolve_image_digest reads RepoDigests and degrades to None, never raising."""

    @pytest.fixture(autouse=True)
    def _clear_digest_cache(self):
        """resolve_image_digest is @lru_cache'd; clear it so mocked inspects don't
        leak a cached digest across tests that reuse the same image reference."""
        from llenergymeasure.infra.image_registry import resolve_image_digest

        resolve_image_digest.cache_clear()
        yield
        resolve_image_digest.cache_clear()

    def test_returns_first_repo_digest(self):
        from llenergymeasure.infra.image_registry import resolve_image_digest

        fake = _fake_inspect(
            repo_digests=[
                "ghcr.io/acme/vllm@sha256:aaaa",
                "ghcr.io/acme/vllm@sha256:bbbb",
            ]
        )
        with patch("llenergymeasure.infra.image_registry.inspect_image", return_value=fake):
            assert resolve_image_digest("ghcr.io/acme/vllm:1.0") == "ghcr.io/acme/vllm@sha256:aaaa"

    def test_none_when_repo_digests_empty(self):
        """Locally-built image (no registry digest) resolves to None, not the local Id."""
        from llenergymeasure.infra.image_registry import resolve_image_digest

        fake = _fake_inspect(repo_digests=[])
        with patch("llenergymeasure.infra.image_registry.inspect_image", return_value=fake):
            assert resolve_image_digest("localbuild:dev") is None

    def test_none_when_repo_digests_absent(self):
        from llenergymeasure.infra.image_registry import resolve_image_digest

        fake = _fake_inspect(repo_digests=_MISSING)
        with patch("llenergymeasure.infra.image_registry.inspect_image", return_value=fake):
            assert resolve_image_digest("localbuild:dev") is None

    def test_none_when_docker_unavailable(self):
        """inspect_image returns None (docker missing / timeout) -> digest None."""
        from llenergymeasure.infra.image_registry import resolve_image_digest

        with patch("llenergymeasure.infra.image_registry.inspect_image", return_value=None):
            assert resolve_image_digest("ghcr.io/acme/vllm:1.0") is None

    def test_none_when_nonzero_returncode(self):
        """Image not pulled yet (non-zero exit) -> digest None."""
        from llenergymeasure.infra.image_registry import resolve_image_digest

        fake = _fake_inspect(repo_digests=["x@sha256:aaaa"], returncode=1)
        with patch("llenergymeasure.infra.image_registry.inspect_image", return_value=fake):
            assert resolve_image_digest("ghcr.io/acme/vllm:1.0") is None

    def test_none_when_malformed_json(self):
        import subprocess
        from unittest.mock import MagicMock

        from llenergymeasure.infra.image_registry import resolve_image_digest

        bad = MagicMock(spec=subprocess.CompletedProcess)
        bad.returncode = 0
        bad.stdout = b"not json"
        with patch("llenergymeasure.infra.image_registry.inspect_image", return_value=bad):
            assert resolve_image_digest("ghcr.io/acme/vllm:1.0") is None
