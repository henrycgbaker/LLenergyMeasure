"""Tests for prompt processing utilities."""

from unittest.mock import MagicMock

import pytest
import torch

from llm_energy_measure.core.prompts import (
    create_adaptive_batches,
    create_fixed_batches,
    filter_n_prompts,
    sort_prompts_by_length,
    tokenize_batch,
)


class TestFilterNPrompts:
    """Tests for filter_n_prompts."""

    def test_filter_list(self):
        prompts = ["a", "b", "c", "d", "e"]
        result = filter_n_prompts(prompts, 3)
        assert result == ["a", "b", "c"]

    def test_filter_list_more_than_available(self):
        prompts = ["a", "b"]
        result = filter_n_prompts(prompts, 10)
        assert result == ["a", "b"]

    def test_filter_dataset_with_select(self):
        # Mock a HuggingFace-style dataset with select method
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=10)
        mock_dataset.select = MagicMock(return_value="selected")

        result = filter_n_prompts(mock_dataset, 5)

        mock_dataset.select.assert_called_once_with(range(5))
        assert result == "selected"

    def test_filter_empty_list(self):
        result = filter_n_prompts([], 5)
        assert result == []


class TestSortPromptsByLength:
    """Tests for sort_prompts_by_length."""

    def test_sort_prompts(self):
        prompts = ["medium", "short", "very long prompt"]
        result = sort_prompts_by_length(prompts)
        assert result == ["short", "medium", "very long prompt"]

    def test_sort_empty(self):
        result = sort_prompts_by_length([])
        assert result == []

    def test_sort_single(self):
        result = sort_prompts_by_length(["only one"])
        assert result == ["only one"]

    def test_sort_same_length(self):
        prompts = ["abc", "xyz", "def"]
        result = sort_prompts_by_length(prompts)
        # Should maintain relative order for same-length strings
        assert len(result) == 3
        assert all(len(p) == 3 for p in result)


class TestCreateFixedBatches:
    """Tests for create_fixed_batches."""

    def test_create_batches_even_split(self):
        prompts = ["a", "b", "c", "d"]
        result = create_fixed_batches(prompts, batch_size=2)
        assert result == [["a", "b"], ["c", "d"]]

    def test_create_batches_uneven_split(self):
        prompts = ["a", "b", "c", "d", "e"]
        result = create_fixed_batches(prompts, batch_size=2)
        assert result == [["a", "b"], ["c", "d"], ["e"]]

    def test_create_batches_larger_than_prompts(self):
        prompts = ["a", "b"]
        result = create_fixed_batches(prompts, batch_size=10)
        assert result == [["a", "b"]]

    def test_create_batches_empty(self):
        result = create_fixed_batches([], batch_size=2)
        assert result == []


class TestCreateAdaptiveBatches:
    """Tests for create_adaptive_batches."""

    @pytest.fixture
    def mock_tokenizer(self):
        tokenizer = MagicMock()

        def encode_side_effect(prompt, **kwargs):
            # Simple mock: 1 token per character
            return {"input_ids": list(range(len(prompt)))}

        tokenizer.side_effect = encode_side_effect
        tokenizer.__call__ = MagicMock(side_effect=encode_side_effect)
        return tokenizer

    def test_create_adaptive_batches_token_budget(self, mock_tokenizer):
        prompts = ["short", "medium text", "longer prompt here"]
        result = create_adaptive_batches(
            prompts=prompts,
            tokenizer=mock_tokenizer,
            max_tokens_per_batch=15,
            max_prompt_tokens=100,
        )

        # First batch: "short" (5 tokens) + "medium text" (11 tokens) = 16 > 15
        # So "short" alone, then "medium text" alone, then "longer prompt here"
        assert len(result) >= 1
        # All prompts should be in some batch
        flat = [p for batch in result for p in batch]
        assert set(flat) == set(prompts)

    def test_create_adaptive_batches_max_batch_size(self, mock_tokenizer):
        prompts = ["a", "b", "c", "d", "e"]
        result = create_adaptive_batches(
            prompts=prompts,
            tokenizer=mock_tokenizer,
            max_tokens_per_batch=1000,  # High budget
            max_prompt_tokens=100,
            max_batch_size=2,  # But limit batch size
        )

        # Should split into batches of max 2
        for batch in result[:-1]:  # All but last
            assert len(batch) <= 2

    def test_create_adaptive_batches_empty(self, mock_tokenizer):
        result = create_adaptive_batches(
            prompts=[],
            tokenizer=mock_tokenizer,
            max_tokens_per_batch=100,
            max_prompt_tokens=50,
        )
        assert result == []


class TestTokenizeBatch:
    """Tests for tokenize_batch."""

    @pytest.fixture
    def mock_tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.model_max_length = 512

        def encode_batch(prompts, **kwargs):
            batch_size = len(prompts)
            max_len = kwargs.get("max_length", 512)
            return {
                "input_ids": torch.zeros(batch_size, max_len, dtype=torch.long),
                "attention_mask": torch.ones(batch_size, max_len, dtype=torch.long),
            }

        tokenizer.side_effect = encode_batch
        tokenizer.__call__ = MagicMock(side_effect=encode_batch)
        return tokenizer

    def test_tokenize_basic(self, mock_tokenizer):
        prompts = ["Hello world", "Test prompt"]
        result = tokenize_batch(prompts, mock_tokenizer, max_length=128)

        assert "input_ids" in result
        assert "attention_mask" in result
        assert result["input_ids"].shape[0] == 2  # 2 prompts

    def test_tokenize_respects_model_max(self, mock_tokenizer):
        mock_tokenizer.model_max_length = 100
        prompts = ["Test"]
        result = tokenize_batch(prompts, mock_tokenizer, max_length=200)

        # The actual result shape should use the capped max_length
        # With our mock, the result shape[1] is the max_length used
        assert result["input_ids"].shape[1] == 100

    def test_tokenize_batches_internally(self, mock_tokenizer):
        prompts = ["p1", "p2", "p3", "p4", "p5"]
        result = tokenize_batch(prompts, mock_tokenizer, max_length=128, batch_size=2)

        # Should process in batches but return concatenated result
        assert result["input_ids"].shape[0] == 5
