"""
Unit tests for data processing utilities.

Tests prompt loading, filtering, and batching.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from llm_efficiency.data.prompts import (
    load_prompts_from_dataset,
    filter_prompts_by_length,
    sort_prompts_by_length,
    tokenize_prompts,
    create_prompt_batches,
    split_prompts_across_processes,
)


class TestLoadPromptsFromDataset:
    """Tests for dataset loading."""

    @patch('llm_efficiency.data.prompts.load_dataset')
    def test_load_all_prompts(self, mock_load_dataset):
        """Test loading all prompts from dataset."""
        mock_dataset = {"train": [{"prompt": f"prompt_{i}"} for i in range(10)]}
        mock_load_dataset.return_value = mock_dataset
        
        prompts = load_prompts_from_dataset(
            dataset_name="test/dataset",
            split="train",
            num_prompts=None,
        )
        
        assert len(prompts) == 10
        assert prompts[0] == "prompt_0"

    @patch('llm_efficiency.data.prompts.load_dataset')
    def test_load_limited_prompts(self, mock_load_dataset):
        """Test loading limited number of prompts."""
        mock_dataset = {"train": [{"prompt": f"prompt_{i}"} for i in range(100)]}
        mock_load_dataset.return_value = mock_dataset
        
        prompts = load_prompts_from_dataset(
            dataset_name="test/dataset",
            split="train",
            num_prompts=10,
        )
        
        assert len(prompts) == 10

    @patch('llm_efficiency.data.prompts.load_dataset')
    def test_custom_text_column(self, mock_load_dataset):
        """Test loading with custom text column."""
        mock_dataset = {"train": [{"text": f"text_{i}"} for i in range(5)]}
        mock_load_dataset.return_value = mock_dataset
        
        prompts = load_prompts_from_dataset(
            dataset_name="test/dataset",
            text_column="text",
        )
        
        assert len(prompts) == 5
        assert prompts[0] == "text_0"


class TestFilterPromptsByLength:
    """Tests for prompt length filtering."""

    def test_filter_short_prompts(self):
        """Test filtering prompts by maximum length."""
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {"input_ids": [[1, 2, 3], [1, 2, 3, 4, 5, 6]]}
        
        prompts = ["short", "longer prompt that exceeds limit"]
        
        filtered = filter_prompts_by_length(prompts, mock_tokenizer, max_tokens=5)
        
        assert len(filtered) == 1
        assert filtered[0] == "short"

    def test_filter_min_length(self):
        """Test filtering prompts by minimum length."""
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": [[1], [1, 2, 3], [1, 2, 3, 4, 5]]
        }
        
        prompts = ["a", "abc", "abcde"]
        
        filtered = filter_prompts_by_length(
            prompts, mock_tokenizer, min_tokens=2, max_tokens=10
        )
        
        assert len(filtered) == 2
        assert "a" not in filtered

    def test_no_filtering(self):
        """Test when all prompts pass filter."""
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {"input_ids": [[1, 2], [1, 2, 3], [1, 2, 3, 4]]}
        
        prompts = ["a", "b", "c"]
        
        filtered = filter_prompts_by_length(prompts, mock_tokenizer, max_tokens=10)
        
        assert len(filtered) == 3


class TestSortPromptsByLength:
    """Tests for prompt sorting."""

    def test_sort_ascending(self):
        """Test sorting prompts by length ascending."""
        mock_tokenizer = Mock()
        mock_tokenizer.side_effect = [
            {"input_ids": [[1, 2, 3, 4, 5]]},
            {"input_ids": [[1, 2]]},
            {"input_ids": [[1, 2, 3]]},
        ]
        
        prompts = ["long", "short", "medium"]
        
        sorted_prompts = sort_prompts_by_length(prompts, mock_tokenizer, ascending=True)
        
        assert sorted_prompts == ["short", "medium", "long"]

    def test_sort_descending(self):
        """Test sorting prompts by length descending."""
        mock_tokenizer = Mock()
        mock_tokenizer.side_effect = [
            {"input_ids": [[1, 2, 3, 4, 5]]},
            {"input_ids": [[1, 2]]},
            {"input_ids": [[1, 2, 3]]},
        ]
        
        prompts = ["long", "short", "medium"]
        
        sorted_prompts = sort_prompts_by_length(prompts, mock_tokenizer, ascending=False)
        
        assert sorted_prompts == ["long", "medium", "short"]


class TestCreatePromptBatches:
    """Tests for batch creation."""

    def test_create_even_batches(self):
        """Test creating batches with even division."""
        prompts = [f"prompt_{i}" for i in range(10)]
        
        batches = create_prompt_batches(prompts, batch_size=5)
        
        assert len(batches) == 2
        assert len(batches[0]) == 5
        assert len(batches[1]) == 5

    def test_create_uneven_batches(self):
        """Test creating batches with uneven division."""
        prompts = [f"prompt_{i}" for i in range(10)]
        
        batches = create_prompt_batches(prompts, batch_size=3)
        
        assert len(batches) == 4
        assert len(batches[0]) == 3
        assert len(batches[1]) == 3
        assert len(batches[2]) == 3
        assert len(batches[3]) == 1  # Remainder

    def test_single_batch(self):
        """Test when all prompts fit in one batch."""
        prompts = [f"prompt_{i}" for i in range(5)]
        
        batches = create_prompt_batches(prompts, batch_size=10)
        
        assert len(batches) == 1
        assert len(batches[0]) == 5


class TestSplitPromptsAcrossProcesses:
    """Tests for multi-process prompt splitting."""

    def test_split_evenly(self):
        """Test splitting prompts evenly across processes."""
        prompts = [f"prompt_{i}" for i in range(12)]
        
        split = split_prompts_across_processes(
            prompts, process_index=0, num_processes=3
        )
        
        assert len(split) == 4
        assert split[0] == "prompt_0"

    def test_split_different_process(self):
        """Test getting split for different process."""
        prompts = [f"prompt_{i}" for i in range(12)]
        
        split = split_prompts_across_processes(
            prompts, process_index=1, num_processes=3
        )
        
        assert len(split) == 4
        assert split[0] == "prompt_4"

    def test_split_uneven(self):
        """Test splitting with uneven distribution."""
        prompts = [f"prompt_{i}" for i in range(10)]
        
        split_0 = split_prompts_across_processes(prompts, 0, 3)
        split_1 = split_prompts_across_processes(prompts, 1, 3)
        split_2 = split_prompts_across_processes(prompts, 2, 3)
        
        # First two processes get 4 each, last gets 2
        assert len(split_0) == 4
        assert len(split_1) == 3
        assert len(split_2) == 3
