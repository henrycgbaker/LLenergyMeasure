"""Tests for distributed computing utilities."""

from unittest.mock import MagicMock

from llm_bench.core.distributed import (
    get_original_generate_method,
    get_persistent_unique_id,
)


class TestGetPersistentUniqueId:
    """Tests for get_persistent_unique_id."""

    def test_creates_new_id_file(self, tmp_path):
        id_file = tmp_path / "subdir" / "id.txt"
        result = get_persistent_unique_id(id_file)

        assert result == "0001"
        assert id_file.exists()
        assert id_file.read_text() == "1"

    def test_increments_existing_id(self, tmp_path):
        id_file = tmp_path / "id.txt"
        id_file.write_text("42")

        result = get_persistent_unique_id(id_file)

        assert result == "0043"
        assert id_file.read_text() == "43"

    def test_handles_invalid_content(self, tmp_path):
        id_file = tmp_path / "id.txt"
        id_file.write_text("not a number")

        result = get_persistent_unique_id(id_file)

        assert result == "0001"  # Resets to 1 on invalid

    def test_zero_pads_correctly(self, tmp_path):
        id_file = tmp_path / "id.txt"
        id_file.write_text("8")

        result = get_persistent_unique_id(id_file)

        assert result == "0009"

    def test_large_numbers(self, tmp_path):
        id_file = tmp_path / "id.txt"
        id_file.write_text("9999")

        result = get_persistent_unique_id(id_file)

        assert result == "10000"  # 5 digits when exceeding 4


class TestGetOriginalGenerateMethod:
    """Tests for get_original_generate_method."""

    def test_finds_generate_method(self):
        model = MagicMock()
        model.generate = MagicMock()

        result = get_original_generate_method(model)

        assert result is model.generate

    def test_finds_nested_generate_method(self):
        inner_model = MagicMock()
        inner_model.generate = MagicMock()

        wrapper = MagicMock()
        wrapper.generate = None  # Wrapper doesn't have generate
        del wrapper.generate  # Remove the attribute entirely
        wrapper.module = inner_model

        # Create proper spec
        wrapper_no_generate = MagicMock(spec=["module"])
        wrapper_no_generate.module = inner_model

        result = get_original_generate_method(wrapper_no_generate)

        assert result is inner_model.generate

    def test_returns_none_when_no_generate(self):
        model = MagicMock(spec=[])  # No generate method

        result = get_original_generate_method(model)

        assert result is None

    def test_handles_deeply_nested_wrappers(self):
        inner = MagicMock()
        inner.generate = MagicMock()

        mid = MagicMock(spec=["module"])
        mid.module = inner

        outer = MagicMock(spec=["module"])
        outer.module = mid

        result = get_original_generate_method(outer)

        assert result is inner.generate
