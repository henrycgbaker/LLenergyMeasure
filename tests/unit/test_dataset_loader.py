"""Tests for dataset loading utilities."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llenergymeasure.config.models import (
    BUILTIN_DATASETS,
    FilePromptSource,
    HuggingFacePromptSource,
)
from llenergymeasure.core.dataset_loader import (
    _auto_detect_column,
    _extract_from_conversation,
    _extract_prompts,
    list_builtin_datasets,
    load_prompts_from_file,
    load_prompts_from_hf,
    load_prompts_from_source,
)
from llenergymeasure.exceptions import ConfigurationError


class TestFilePromptSource:
    """Tests for file-based prompt loading."""

    def test_load_from_file(self, tmp_path: Path):
        """Load prompts from a text file."""
        prompts_file = tmp_path / "prompts.txt"
        prompts_file.write_text("prompt 1\nprompt 2\n\nprompt 3")

        source = FilePromptSource(path=str(prompts_file))
        prompts = load_prompts_from_source(source)

        assert prompts == ["prompt 1", "prompt 2", "prompt 3"]

    def test_load_from_file_direct(self, tmp_path: Path):
        """Load prompts using direct function."""
        prompts_file = tmp_path / "prompts.txt"
        prompts_file.write_text("hello\nworld")

        prompts = load_prompts_from_file(prompts_file)
        assert prompts == ["hello", "world"]

    def test_file_not_found(self):
        """Raise error for missing file."""
        source = FilePromptSource(path="/nonexistent/file.txt")
        with pytest.raises(ConfigurationError, match="not found"):
            load_prompts_from_source(source)

    def test_empty_file(self, tmp_path: Path):
        """Raise error for empty file."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")

        source = FilePromptSource(path=str(empty_file))
        with pytest.raises(ConfigurationError, match="No prompts found"):
            load_prompts_from_source(source)

    def test_whitespace_only_file(self, tmp_path: Path):
        """Raise error for whitespace-only file."""
        whitespace_file = tmp_path / "whitespace.txt"
        whitespace_file.write_text("   \n\n  \n")

        source = FilePromptSource(path=str(whitespace_file))
        with pytest.raises(ConfigurationError, match="No prompts found"):
            load_prompts_from_source(source)

    def test_strips_whitespace(self, tmp_path: Path):
        """Strip whitespace from prompts."""
        prompts_file = tmp_path / "prompts.txt"
        prompts_file.write_text("  hello  \n  world  ")

        prompts = load_prompts_from_file(prompts_file)
        assert prompts == ["hello", "world"]


class TestHuggingFacePromptSource:
    """Tests for HuggingFace dataset prompt source config."""

    def test_builtin_alias_resolution(self):
        """Built-in aliases resolve to full paths."""
        source = HuggingFacePromptSource(dataset="alpaca")
        assert source.dataset == "tatsu-lab/alpaca"
        assert source.column == "instruction"

    def test_custom_dataset_no_alias(self):
        """Custom datasets don't get modified."""
        source = HuggingFacePromptSource(dataset="custom/dataset", column="text")
        assert source.dataset == "custom/dataset"
        assert source.column == "text"

    def test_builtin_gsm8k_resolution(self):
        """GSM8K alias resolves correctly."""
        source = HuggingFacePromptSource(dataset="gsm8k")
        assert source.dataset == "gsm8k"
        assert source.column == "question"
        assert source.subset == "main"

    def test_explicit_column_not_overwritten(self):
        """Explicit column overrides builtin default."""
        source = HuggingFacePromptSource(dataset="alpaca", column="output")
        assert source.column == "output"

    def test_sample_size_validation(self):
        """Sample size must be positive."""
        with pytest.raises(ValueError):
            HuggingFacePromptSource(dataset="alpaca", sample_size=0)

    def test_default_values(self):
        """Check default values."""
        source = HuggingFacePromptSource(dataset="custom/ds")
        assert source.split == "train"
        assert source.shuffle is False
        assert source.seed == 42
        assert source.sample_size is None


class TestHuggingFaceLoading:
    """Tests for actual HF dataset loading (mocked)."""

    @patch("datasets.load_dataset")
    def test_load_simple_dataset(self, mock_load):
        """Load simple text column dataset."""
        mock_ds = MagicMock()
        mock_ds.column_names = ["text", "label"]
        mock_ds.__iter__ = lambda self: iter([{"text": "hello"}, {"text": "world"}])
        mock_ds.__len__ = lambda self: 2
        mock_load.return_value = mock_ds

        source = HuggingFacePromptSource(dataset="test/ds", column="text")
        prompts = load_prompts_from_hf(source)

        assert prompts == ["hello", "world"]
        mock_load.assert_called_once()

    @patch("datasets.load_dataset")
    def test_load_with_sampling(self, mock_load):
        """Load with sample size limit."""
        mock_ds = MagicMock()
        mock_ds.column_names = ["text"]
        mock_ds.__len__ = lambda self: 100

        # Mock select to return limited dataset
        limited_ds = MagicMock()
        limited_ds.column_names = ["text"]
        limited_ds.__iter__ = lambda self: iter([{"text": f"prompt {i}"} for i in range(10)])
        mock_ds.select.return_value = limited_ds

        mock_load.return_value = mock_ds

        source = HuggingFacePromptSource(dataset="test/ds", column="text", sample_size=10)
        prompts = load_prompts_from_hf(source)

        assert len(prompts) == 10
        mock_ds.select.assert_called_once()

    @patch("datasets.load_dataset")
    def test_load_with_shuffle(self, mock_load):
        """Load with shuffle enabled."""
        mock_ds = MagicMock()
        mock_ds.column_names = ["text"]
        mock_ds.__len__ = lambda self: 5
        mock_ds.__iter__ = lambda self: iter([{"text": "prompt"}])

        shuffled_ds = MagicMock()
        shuffled_ds.column_names = ["text"]
        shuffled_ds.__iter__ = lambda self: iter([{"text": "prompt"}])
        shuffled_ds.__len__ = lambda self: 5
        mock_ds.shuffle.return_value = shuffled_ds

        mock_load.return_value = mock_ds

        source = HuggingFacePromptSource(dataset="test/ds", column="text", shuffle=True, seed=123)
        load_prompts_from_hf(source)

        mock_ds.shuffle.assert_called_once_with(seed=123)

    @patch("datasets.load_dataset")
    def test_missing_column_error(self, mock_load):
        """Raise error for missing column."""
        mock_ds = MagicMock()
        mock_ds.column_names = ["different_column"]
        mock_load.return_value = mock_ds

        source = HuggingFacePromptSource(dataset="test/ds", column="text")
        with pytest.raises(ConfigurationError, match="Column 'text' not found"):
            load_prompts_from_hf(source)


class TestAutoDetectColumn:
    """Tests for column auto-detection."""

    def test_detects_text(self):
        """Detect 'text' column."""
        assert _auto_detect_column(["id", "text", "label"]) == "text"

    def test_detects_prompt(self):
        """Detect 'prompt' column."""
        assert _auto_detect_column(["id", "prompt", "response"]) == "prompt"

    def test_detects_question(self):
        """Detect 'question' column."""
        assert _auto_detect_column(["question", "answer"]) == "question"

    def test_detects_instruction(self):
        """Detect 'instruction' column."""
        assert _auto_detect_column(["instruction", "output"]) == "instruction"

    def test_priority_order(self):
        """'text' should take priority over later options."""
        # text comes before instruction in AUTO_DETECT_COLUMNS
        assert _auto_detect_column(["instruction", "text"]) == "text"

    def test_no_match_raises(self):
        """Raise error when no suitable column found."""
        with pytest.raises(ConfigurationError, match="Could not auto-detect"):
            _auto_detect_column(["id", "label", "output"])


class TestExtractPrompts:
    """Tests for prompt extraction from various formats."""

    def test_simple_strings(self):
        """Extract simple string values."""
        dataset = [{"text": "hello"}, {"text": "world"}]
        assert _extract_prompts(dataset, "text") == ["hello", "world"]

    def test_skips_empty_strings(self):
        """Skip empty or whitespace-only strings."""
        dataset = [{"text": ""}, {"text": "valid"}, {"text": "   "}]
        assert _extract_prompts(dataset, "text") == ["valid"]

    def test_strips_whitespace(self):
        """Strip whitespace from extracted prompts."""
        dataset = [{"text": "  hello  "}, {"text": "\nworld\n"}]
        assert _extract_prompts(dataset, "text") == ["hello", "world"]


class TestExtractFromConversation:
    """Tests for conversation format extraction."""

    def test_sharegpt_format(self):
        """Extract from ShareGPT format."""
        messages = [
            {"from": "human", "value": "Hello there"},
            {"from": "assistant", "value": "Hi!"},
        ]
        assert _extract_from_conversation(messages) == "Hello there"

    def test_openai_format(self):
        """Extract from OpenAI-style format."""
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
        assert _extract_from_conversation(messages) == "What is 2+2?"

    def test_no_human_message(self):
        """Return None if no human message found."""
        messages = [
            {"from": "assistant", "value": "Hello"},
            {"from": "system", "value": "You are helpful"},
        ]
        assert _extract_from_conversation(messages) is None

    def test_empty_human_message(self):
        """Return None if human message is empty."""
        messages = [
            {"from": "human", "value": ""},
            {"from": "assistant", "value": "Hi"},
        ]
        assert _extract_from_conversation(messages) is None


class TestListBuiltinDatasets:
    """Tests for built-in dataset listing."""

    def test_returns_all_builtins(self):
        """Return all built-in datasets."""
        builtins = list_builtin_datasets()
        assert "alpaca" in builtins
        assert "gsm8k" in builtins
        assert "mmlu" in builtins
        assert "sharegpt" in builtins

    def test_returns_copy(self):
        """Return a copy, not the original dict."""
        builtins = list_builtin_datasets()
        builtins["custom"] = {}
        assert "custom" not in BUILTIN_DATASETS


class TestPromptSourceConfigModels:
    """Tests for prompt source config discriminated union."""

    def test_file_source_from_dict(self):
        """Create FilePromptSource from dict."""
        data = {"type": "file", "path": "/some/path.txt"}
        source = FilePromptSource(**data)
        assert source.type == "file"
        assert source.path == "/some/path.txt"

    def test_hf_source_from_dict(self):
        """Create HuggingFacePromptSource from dict."""
        data = {"type": "huggingface", "dataset": "alpaca", "sample_size": 100}
        source = HuggingFacePromptSource(**data)
        assert source.type == "huggingface"
        assert source.sample_size == 100
