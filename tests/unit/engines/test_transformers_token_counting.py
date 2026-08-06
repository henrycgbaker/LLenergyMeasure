"""Tests for PyTorch engine token counting with padded batches."""

import pytest

torch = pytest.importorskip("torch")


def _count_tokens(
    inputs: dict,
    outputs: "torch.Tensor",  # type: ignore[name-defined]  # torch via importorskip
    n_inputs: int | None = None,
) -> tuple[int, int]:
    """Replicate the token counting logic from TransformersEngine._run_batch().

    ``n_inputs`` is the number of input prompts (the batch list length); it
    defaults to the attention_mask row count for the n=1 case but must be passed
    explicitly when ``outputs`` has ``n_inputs * num_return_sequences`` rows.
    """
    input_token_count = int(inputs["attention_mask"].sum().item())
    input_lengths = [int(x) for x in inputs["attention_mask"].sum(dim=1).tolist()]
    if n_inputs is None:
        n_inputs = len(input_lengths)
    n_rows = int(outputs.shape[0])
    n_return = n_rows // n_inputs if n_inputs > 0 and n_rows % n_inputs == 0 else 1
    output_token_count = int(
        sum(
            max(0, int(outputs.shape[1]) - input_lengths[min(j // n_return, n_inputs - 1)])
            for j in range(n_rows)
        )
    )
    return input_token_count, output_token_count


def _old_count_tokens(inputs: dict, outputs: "torch.Tensor") -> tuple[int, int]:  # type: ignore[name-defined]  # torch via importorskip
    """Previous (buggy) logic for comparison."""
    input_token_count = inputs["input_ids"].shape[1] * len(inputs["input_ids"])
    tokens_per_seq = outputs.shape[1] - inputs["input_ids"].shape[1]
    output_token_count = max(0, tokens_per_seq) * len(inputs["input_ids"])
    return input_token_count, output_token_count


class TestUniformBatch:
    """Uniform-length sequences: both methods should agree."""

    def test_input_tokens_uniform(self) -> None:
        # 2 sequences, length 4, all real tokens (no padding)
        input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        attention_mask = torch.ones_like(input_ids)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        outputs = torch.zeros((2, 6), dtype=torch.long)  # 2 new tokens each

        new_count, _ = _count_tokens(inputs, outputs)
        old_count, _ = _old_count_tokens(inputs, outputs)

        assert new_count == 8
        assert new_count == old_count, "Uniform batch: new and old methods must agree"

    def test_output_tokens_uniform(self) -> None:
        # 2 sequences, input length 4, output length 6 → 2 new tokens each
        input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        attention_mask = torch.ones_like(input_ids)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        outputs = torch.zeros((2, 6), dtype=torch.long)

        _, new_out = _count_tokens(inputs, outputs)
        _, old_out = _old_count_tokens(inputs, outputs)

        assert new_out == 4
        assert new_out == old_out, "Uniform batch: output counts must agree"


class TestPaddedBatch:
    """Padded batch: sequences of different lengths padded to the same width."""

    def test_input_tokens_padded_returns_real_tokens_only(self) -> None:
        # Sequence 1: length 3 tokens, padded to 5  → [1, 2, 3, 0, 0], mask [1,1,1,0,0]
        # Sequence 2: length 5 tokens, no padding   → [4, 5, 6, 7, 8], mask [1,1,1,1,1]
        # Total real input tokens = 3 + 5 = 8 (NOT 5*2 = 10)
        input_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]])
        attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        outputs = torch.zeros((2, 7), dtype=torch.long)  # dummy outputs

        new_count, _ = _count_tokens(inputs, outputs)
        old_count, _ = _old_count_tokens(inputs, outputs)

        assert new_count == 8, f"Expected 8 real input tokens, got {new_count}"
        assert old_count == 10, "Old method overcounts padded input tokens (expected 10)"
        assert new_count != old_count, "Fix should differ from old method on padded batch"

    def test_output_tokens_padded_per_sequence(self) -> None:
        # Sequence 1: input length 3, padded to 5; total output length 7
        #   → generated tokens for seq1 = 7 - 3 = 4
        # Sequence 2: input length 5 (no padding); total output length 7
        #   → generated tokens for seq2 = 7 - 5 = 2
        # Total output tokens = 4 + 2 = 6 (NOT (7-5)*2 = 4)
        input_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]])
        attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        # output tensor shape (batch=2, total_seq_len=7)
        outputs = torch.zeros((2, 7), dtype=torch.long)

        _, new_out = _count_tokens(inputs, outputs)
        _, old_out = _old_count_tokens(inputs, outputs)

        assert new_out == 6, f"Expected 6 output tokens (per-sequence), got {new_out}"
        assert old_out == 4, "Old method undercounts output tokens on padded batch (expected 4)"
        assert new_out != old_out, "Fix should differ from old method on padded output"

    def test_output_tokens_no_new_tokens_clamped_to_zero(self) -> None:
        # Edge case: for a uniform batch where output length equals input length → no new tokens
        # Both sequences are length 4 (no padding), output also length 4
        input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        attention_mask = torch.ones_like(input_ids)  # all real tokens
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        # Output same width as input → no generation happened
        outputs = input_ids  # shape (2, 4)

        _, new_out = _count_tokens(inputs, outputs)

        assert new_out == 0, f"Expected 0 output tokens when no generation, got {new_out}"


class TestNumReturnSequences:
    """EN4: outputs.shape[0] == batch * num_return_sequences; count ALL rows.

    HF returns N rows per input prompt when num_return_sequences=N (incl. beam
    search). The old loop counted only ``batch`` rows -> N-fold undercount.
    """

    def test_n1_unchanged_baseline(self) -> None:
        # n=1: 2 prompts, input length 4, output length 6 -> 2 new tokens each = 4.
        input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        attention_mask = torch.ones_like(input_ids)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        outputs = torch.zeros((2, 6), dtype=torch.long)  # 2 rows (n=1)

        _, out_n1 = _count_tokens(inputs, outputs, n_inputs=2)
        assert out_n1 == 4

    def test_n3_counts_all_rows(self) -> None:
        # 2 prompts, num_return_sequences=3 -> outputs has 6 rows. Each output row
        # is length 6, every input length 4 -> 2 new tokens per row * 6 rows = 12.
        # The n=1 count for the same inputs/output width would be 4, so n=3 == 3x.
        input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        attention_mask = torch.ones_like(input_ids)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        outputs = torch.zeros((6, 6), dtype=torch.long)  # 2 inputs * 3 returns

        _, out_n3 = _count_tokens(inputs, outputs, n_inputs=2)
        assert out_n3 == 12, f"Expected 12 (all 6 rows), got {out_n3}"

        # Equivalence to n=1 * N.
        outputs_n1 = torch.zeros((2, 6), dtype=torch.long)
        _, out_n1 = _count_tokens(inputs, outputs_n1, n_inputs=2)
        assert out_n3 == 3 * out_n1

    def test_n2_padded_maps_rows_to_source_input(self) -> None:
        # Padded inputs: seq1 real length 3, seq2 real length 5. n=2 -> 4 rows.
        # Output width 7. Rows map j//2: rows 0,1 -> input0 (len 3 -> 4 new),
        # rows 2,3 -> input1 (len 5 -> 2 new). Total = 4+4+2+2 = 12.
        input_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]])
        attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        outputs = torch.zeros((4, 7), dtype=torch.long)

        _, out = _count_tokens(inputs, outputs, n_inputs=2)
        assert out == 12, f"Expected 12 with per-source-input mapping, got {out}"

    def test_non_multiple_row_count_falls_back(self) -> None:
        # Defensive guard: 5 output rows for 2 inputs is not an exact multiple, so
        # n_return falls back to 1 and rows map j//1 clamped to last input.
        # Rows 0,1 -> input0 (len 4 -> 2 new), rows 2,3,4 -> input1 (clamped).
        # input1 len 4 -> 2 new. Total = 5 rows * 2 = 10.
        input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        attention_mask = torch.ones_like(input_ids)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        outputs = torch.zeros((5, 6), dtype=torch.long)

        _, out = _count_tokens(inputs, outputs, n_inputs=2)
        assert out == 10


def _count_padding(inputs: dict) -> int:
    """Replicate the padding-token math from TransformersEngine._run_batch()."""
    input_token_count = int(inputs["attention_mask"].sum().item())
    return int(inputs["input_ids"].numel()) - input_token_count


class TestPaddingTokenCount:
    """Padding tokens = input_ids.numel() - attention_mask.sum()."""

    def test_no_padding_uniform(self) -> None:
        input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        attention_mask = torch.ones_like(input_ids)
        assert _count_padding({"input_ids": input_ids, "attention_mask": attention_mask}) == 0

    def test_padded_batch(self) -> None:
        # 2x5 = 10 positions, 8 real tokens -> 2 padding tokens
        input_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]])
        attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
        assert _count_padding({"input_ids": input_ids, "attention_mask": attention_mask}) == 2

    def test_single_sequence_no_padding(self) -> None:
        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.ones_like(input_ids)
        assert _count_padding({"input_ids": input_ids, "attention_mask": attention_mask}) == 0
