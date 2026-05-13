"""Tests for ``llenergymeasure.infra.version_handshake``."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from llenergymeasure.infra.version_handshake import (
    ENV_SKIP_IMAGE_CHECK,
    LABEL_IMAGE_VERSION,
    LABEL_SCHEMA_FINGERPRINT,
    ImageStamp,
    SchemaStatus,
    classify_engine_version,
    classify_stamp,
    compute_expconf_fingerprint,
    inspect_image_stamp,
    probe_image_engine_version,
    read_ssot_engine_version,
    skip_check_enabled,
)

# ---------------------------------------------------------------------------
# compute_expconf_fingerprint
# ---------------------------------------------------------------------------


class TestComputeExpConfFingerprint:
    def test_deterministic_across_calls(self) -> None:
        assert compute_expconf_fingerprint() == compute_expconf_fingerprint()

    def test_hex_length(self) -> None:
        fp = compute_expconf_fingerprint()
        assert len(fp) == 64
        int(fp, 16)  # must be valid hex

    def test_sensitive_to_model_changes(self) -> None:
        """Adding a field to a throwaway model must change its fingerprint.

        We don't mutate ExperimentConfig itself (too invasive for a unit
        test), but we verify the serialiser is sensitive to schema shape by
        hashing two synthetic pydantic models with and without an extra field.
        """
        import hashlib

        from pydantic import BaseModel

        class _A(BaseModel):
            x: int = 0

        class _B(BaseModel):
            x: int = 0
            y: int = 0

        def _hash(model_cls: type[BaseModel]) -> str:
            payload = json.dumps(
                model_cls.model_json_schema(),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            return hashlib.sha256(payload).hexdigest()

        assert _hash(_A) != _hash(_B)


# ---------------------------------------------------------------------------
# inspect_image_stamp
# ---------------------------------------------------------------------------


def _fake_inspect_result(*, labels: dict[str, str] | None, returncode: int = 0) -> MagicMock:
    body = [{"Id": "sha256:abc", "Config": {"Labels": labels}}] if labels is not None else [{}]
    result = MagicMock(spec=subprocess.CompletedProcess)
    result.returncode = returncode
    result.stdout = json.dumps(body).encode("utf-8")
    result.stderr = b""
    return result


class TestInspectImageStamp:
    def test_valid_labels(self) -> None:
        labels = {
            LABEL_SCHEMA_FINGERPRINT: "a" * 64,
            LABEL_IMAGE_VERSION: "0.9.0",
        }
        with patch("subprocess.run", return_value=_fake_inspect_result(labels=labels)):
            stamp = inspect_image_stamp("my/img:latest")
        assert stamp == ImageStamp(pkg_version="0.9.0", expconf_fingerprint="a" * 64)

    def test_missing_labels(self) -> None:
        with patch("subprocess.run", return_value=_fake_inspect_result(labels={})):
            stamp = inspect_image_stamp("my/img:latest")
        assert stamp.pkg_version is None
        assert stamp.expconf_fingerprint is None

    def test_partial_labels(self) -> None:
        labels = {LABEL_IMAGE_VERSION: "0.9.0"}
        with patch("subprocess.run", return_value=_fake_inspect_result(labels=labels)):
            stamp = inspect_image_stamp("my/img:latest")
        assert stamp.pkg_version == "0.9.0"
        assert stamp.expconf_fingerprint is None

    def test_timeout_returns_empty_stamp(self) -> None:
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired("docker", 5),
        ):
            stamp = inspect_image_stamp("my/img:latest", timeout=1.0)
        assert stamp == ImageStamp(pkg_version=None, expconf_fingerprint=None)

    def test_docker_not_installed(self) -> None:
        with patch("subprocess.run", side_effect=FileNotFoundError("docker")):
            stamp = inspect_image_stamp("my/img:latest")
        assert stamp == ImageStamp(pkg_version=None, expconf_fingerprint=None)

    def test_nonzero_returncode(self) -> None:
        bad = MagicMock(spec=subprocess.CompletedProcess)
        bad.returncode = 1
        bad.stdout = b""
        bad.stderr = b"No such image"
        with patch("subprocess.run", return_value=bad):
            stamp = inspect_image_stamp("my/img:latest")
        assert stamp == ImageStamp(pkg_version=None, expconf_fingerprint=None)

    def test_malformed_json(self) -> None:
        bad = MagicMock(spec=subprocess.CompletedProcess)
        bad.returncode = 0
        bad.stdout = b"not json"
        bad.stderr = b""
        with patch("subprocess.run", return_value=bad):
            stamp = inspect_image_stamp("my/img:latest")
        assert stamp == ImageStamp(pkg_version=None, expconf_fingerprint=None)


# ---------------------------------------------------------------------------
# skip_check_enabled
# ---------------------------------------------------------------------------


class TestSkipCheckEnabled:
    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "Yes"])
    def test_truthy(self, value: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_SKIP_IMAGE_CHECK, value)
        assert skip_check_enabled() is True

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "random"])
    def test_falsy(self, value: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_SKIP_IMAGE_CHECK, value)
        assert skip_check_enabled() is False

    def test_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ENV_SKIP_IMAGE_CHECK, raising=False)
        assert skip_check_enabled() is False


# ---------------------------------------------------------------------------
# classify_stamp - "unknown" fingerprint now classifies as UNVERIFIED, not
# MISMATCH (closes #602)
# ---------------------------------------------------------------------------


class TestClassifyStampUnknown:
    def test_unknown_fingerprint_classifies_as_unverified(self) -> None:
        """A literal "unknown" fingerprint is a legacy build marker, not a
        positive disagreement signal. Treating it as MISMATCH was the bug
        in #602; treating it as UNVERIFIED lets the engine-version probe
        take over.
        """
        stamp = ImageStamp(pkg_version="dev", expconf_fingerprint="unknown")
        assert classify_stamp(stamp, "real_host_fingerprint") is SchemaStatus.UNVERIFIED

    def test_real_fingerprint_mismatch_still_classifies_as_mismatch(self) -> None:
        stamp = ImageStamp(pkg_version="0.9.0", expconf_fingerprint="abcd1234ef56")
        assert classify_stamp(stamp, "different_host_fp") is SchemaStatus.MISMATCH

    def test_real_fingerprint_match_classifies_as_ok(self) -> None:
        stamp = ImageStamp(pkg_version="0.9.0", expconf_fingerprint="match_me")
        assert classify_stamp(stamp, "match_me") is SchemaStatus.OK

    def test_none_fingerprint_unchanged(self) -> None:
        stamp = ImageStamp(pkg_version="0.9.0", expconf_fingerprint=None)
        assert classify_stamp(stamp, "anything") is SchemaStatus.UNVERIFIED

    def test_empty_stamp_is_unreachable(self) -> None:
        stamp = ImageStamp(pkg_version=None, expconf_fingerprint=None)
        assert classify_stamp(stamp, "anything") is SchemaStatus.UNREACHABLE


# ---------------------------------------------------------------------------
# classify_engine_version - SSOT engine-version probe result classification
# ---------------------------------------------------------------------------


class TestClassifyEngineVersion:
    def test_exact_match(self) -> None:
        assert classify_engine_version("0.7.3", "0.7.3") is SchemaStatus.OK

    def test_v_prefix_normalisation(self) -> None:
        """``v0.7.3`` from the image matches ``0.7.3`` in SSOT."""
        assert classify_engine_version("v0.7.3", "0.7.3") is SchemaStatus.OK
        assert classify_engine_version("0.7.3", "v0.7.3") is SchemaStatus.OK

    def test_mismatch(self) -> None:
        assert classify_engine_version("0.7.4", "0.7.3") is SchemaStatus.MISMATCH
        assert classify_engine_version("0.21.0", "0.7.3") is SchemaStatus.MISMATCH

    @pytest.mark.parametrize(
        "probed,expected",
        [
            (None, "0.7.3"),
            ("0.7.3", None),
            (None, None),
        ],
    )
    def test_unreachable_when_either_side_missing(
        self, probed: str | None, expected: str | None
    ) -> None:
        assert classify_engine_version(probed, expected) is SchemaStatus.UNREACHABLE


# ---------------------------------------------------------------------------
# read_ssot_engine_version - reads engine_versions/{engine}.yaml
# ---------------------------------------------------------------------------


class TestReadSsotEngineVersion:
    def test_known_engine(self) -> None:
        """The committed SSOT files vllm/tensorrt/transformers all resolve."""
        # Exact values are subject to Renovate bumps; assert non-empty + str.
        for engine in ("vllm", "tensorrt", "transformers"):
            version = read_ssot_engine_version(engine)
            assert isinstance(version, str)
            assert version  # non-empty

    def test_unknown_engine_returns_none(self) -> None:
        assert read_ssot_engine_version("nonexistent_engine_xyz") is None


# ---------------------------------------------------------------------------
# probe_image_engine_version - subprocess-mocked; live tests are integration
# ---------------------------------------------------------------------------


class TestProbeImageEngineVersion:
    def test_parses_marker_line(self) -> None:
        """The probe code emits a marker-prefixed version line; the parser strips it."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Some banner output\n---LLEM_VER:0.7.3\nSome trailing line\n"
        with patch(
            "llenergymeasure.infra.version_handshake.subprocess.run",
            return_value=mock_result,
        ):
            # Use a unique image string to bypass the lru_cache
            assert probe_image_engine_version("test/unique-img-marker:tag", "vllm") == "0.7.3"

    def test_returns_none_on_nonzero_exit(self) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "ImportError: No module named 'vllm'"
        with patch(
            "llenergymeasure.infra.version_handshake.subprocess.run",
            return_value=mock_result,
        ):
            assert probe_image_engine_version("test/unique-img-nonzero:tag", "vllm") is None

    def test_returns_none_on_timeout(self) -> None:
        with patch(
            "llenergymeasure.infra.version_handshake.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["docker"], timeout=60.0),
        ):
            assert probe_image_engine_version("test/unique-img-timeout:tag", "vllm") is None

    def test_returns_none_on_docker_not_installed(self) -> None:
        with patch(
            "llenergymeasure.infra.version_handshake.subprocess.run",
            side_effect=FileNotFoundError("docker not found"),
        ):
            assert probe_image_engine_version("test/unique-img-no-docker:tag", "vllm") is None

    def test_unknown_engine_returns_none_without_invoking_docker(self) -> None:
        """An unrecognised engine short-circuits before docker subprocess."""
        with patch("llenergymeasure.infra.version_handshake.subprocess.run") as mock_run:
            result = probe_image_engine_version("any:img", "no_such_engine")
            assert result is None
            mock_run.assert_not_called()

    def test_tensorrt_uses_nvidia_entrypoint(self) -> None:
        """TRT-LLM probe routes through /opt/nvidia/nvidia_entrypoint.sh
        so libnvinfer's LD_LIBRARY_PATH is set up before import."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "---LLEM_VER:0.21.0\n"
        with patch(
            "llenergymeasure.infra.version_handshake.subprocess.run",
            return_value=mock_result,
        ) as mock_run:
            probe_image_engine_version("test/unique-trt-entry:tag", "tensorrt")

        called_cmd = mock_run.call_args[0][0]
        assert "/opt/nvidia/nvidia_entrypoint.sh" in called_cmd
        assert "--entrypoint" in called_cmd
        ep_idx = called_cmd.index("--entrypoint")
        assert called_cmd[ep_idx + 1] == "/opt/nvidia/nvidia_entrypoint.sh"

    def test_vllm_uses_python3_entrypoint(self) -> None:
        """vLLM probe uses --entrypoint python3 (upstream image's default
        entrypoint is api_server.py which would interpret -c as a flag)."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "---LLEM_VER:0.7.3\n"
        with patch(
            "llenergymeasure.infra.version_handshake.subprocess.run",
            return_value=mock_result,
        ) as mock_run:
            probe_image_engine_version("test/unique-vllm-entry:tag", "vllm")

        called_cmd = mock_run.call_args[0][0]
        ep_idx = called_cmd.index("--entrypoint")
        assert called_cmd[ep_idx + 1] == "python3"
