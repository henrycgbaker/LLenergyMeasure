"""Unit tests for the deterministic cross-field constraint extractor.

The extractor's per-engine :data:`TARGETS` table names concrete
class/method/file landmarks. These tests build synthetic source trees that hit
those landmarks (rather than depending on a real engine install), so they cover
the four constraint shapes the recall check proved the extractor must surface:
a cross-field raise, a single-field raise, a self-assign dormancy, and an
``isinstance``/``is not None`` type guard.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.cross_field_extractor as cfe  # noqa: E402

_VLLM_SAMPLING_PARAMS = """
class SamplingParams:
    def _verify_greedy_sampling(self) -> None:
        if self.n > 1:
            raise ValueError(f"n must be 1 when using greedy sampling, got {self.n}.")

    def __post_init__(self) -> None:
        if 0 < self.temperature < _MAX_TEMP:
            logger.warning("temperature too low")
            self.temperature = max(self.temperature, _MAX_TEMP)
"""

_VLLM_COMPILATION = """
class CompilationConfig:
    def __post_init__(self) -> None:
        if (
            self.cudagraph_mm_encoder
            and self.encoder_cudagraph_max_images_per_batch < 0
        ):
            raise ValueError("encoder_cudagraph_max_images_per_batch must be non-negative")
"""

_TFM_QUANT = """
class BitsAndBytesConfig:
    def post_init(self):
        if self.bnb_4bit_compute_dtype is not None and not isinstance(
            self.bnb_4bit_compute_dtype, torch.dtype
        ):
            raise TypeError("bnb_4bit_compute_dtype must be torch.dtype")
"""


def _write(root: Path, rel: str, body: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def _by_field(cands: list[dict[str, Any]], leaf: str) -> dict[str, Any] | None:
    for cand in cands:
        if any(path.endswith(f".{leaf}") for path in cand["match"]["fields"]):
            return cand
    return None


def _vllm_source(root: Path) -> Path:
    src = root / "vllm"
    _write(src, "sampling_params.py", _VLLM_SAMPLING_PARAMS)
    _write(src, "config/compilation.py", _VLLM_COMPILATION)
    return src


def test_single_field_raise_becomes_error_candidate(tmp_path: Path) -> None:
    cands = cfe.extract("vllm", _vllm_source(tmp_path), "0.19.1", "2026-07-13")
    greedy = _by_field(cands, "n")
    assert greedy is not None
    assert greedy["severity"] == "error"
    assert greedy["match"]["fields"] == {"vllm.sampling.n": {">": 1}}
    assert greedy["provenance"]["source"] == "deterministic_extractor"


def test_cross_field_raise_keeps_both_operands(tmp_path: Path) -> None:
    cands = cfe.extract("vllm", _vllm_source(tmp_path), "0.19.1", "2026-07-13")
    cross = _by_field(cands, "encoder_cudagraph_max_images_per_batch")
    assert cross is not None
    assert cross["severity"] == "error"
    assert cross["match"]["fields"] == {
        "vllm.engine.cudagraph_mm_encoder": {"present": True},
        "vllm.engine.encoder_cudagraph_max_images_per_batch": {"<": 0},
    }


def test_self_assign_clamp_becomes_dormant_candidate(tmp_path: Path) -> None:
    # ``if 0 < self.temperature < _MAX_TEMP: self.temperature = max(...)`` - the
    # non-literal bound (_MAX_TEMP) drops, leaving temperature>0 dormant.
    cands = cfe.extract("vllm", _vllm_source(tmp_path), "0.19.1", "2026-07-13")
    clamp = next(
        c for c in cands if c["severity"] == "dormant" and "temperature" in str(c["match"])
    )
    assert clamp["match"]["fields"] == {"vllm.sampling.temperature": {">": 0}}
    assert clamp["normalised_fields"] == ["temperature"]


def test_isinstance_and_is_not_none_type_guard(tmp_path: Path) -> None:
    src = tmp_path / "transformers"
    _write(src, "utils/quantization_config.py", _TFM_QUANT)
    cands = cfe.extract("transformers", src, "5.7.0", "2026-07-13")
    bnb = _by_field(cands, "bnb_4bit_compute_dtype")
    assert bnb is not None
    assert bnb["severity"] == "error"
    assert bnb["match"]["fields"] == {
        "transformers.quant.bnb_4bit_compute_dtype": {"present": True, "type_is_not": "dtype"},
    }


def test_output_is_byte_stable_and_sorted(tmp_path: Path) -> None:
    src = _vllm_source(tmp_path)
    first = cfe.extract("vllm", src, "0.19.1", "2026-07-13")
    second = cfe.extract("vllm", src, "0.19.1", "2026-07-13")
    assert first == second
    assert [c["id"] for c in first] == sorted(c["id"] for c in first)


def test_missing_target_file_is_skipped_not_fatal(tmp_path: Path) -> None:
    # Only one of the vllm target files exists; the rest are absent.
    src = tmp_path / "vllm"
    _write(src, "sampling_params.py", _VLLM_SAMPLING_PARAMS)
    cands = cfe.extract("vllm", src, "0.19.1", "2026-07-13")
    assert _by_field(cands, "n") is not None  # extracted from the file that exists
    assert _by_field(cands, "encoder_cudagraph_max_images_per_batch") is None  # file absent


def test_id_digest_collapses_re_reached_claim(tmp_path: Path) -> None:
    cands = cfe.extract("vllm", _vllm_source(tmp_path), "0.19.1", "2026-07-13")
    ids = [c["id"] for c in cands]
    assert len(ids) == len(set(ids))  # no duplicate ids within one engine run
