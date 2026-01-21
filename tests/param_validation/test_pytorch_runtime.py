"""GPU-based runtime tests for PyTorch parameters.

These tests verify that PyTorch/Transformers parameters affect model
behaviour at runtime. Requires GPU.

Run with: pytest tests/param_validation/test_pytorch_runtime.py -v
"""

from __future__ import annotations

import gc

import pytest

from .conftest import requires_ampere, requires_flash_attn, requires_gpu


def cleanup_gpu() -> None:
    """Clean up GPU memory."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


@requires_gpu
class TestPyTorchAttention:
    """Test PyTorch attention implementation parameters."""

    @pytest.fixture(autouse=True)
    def cleanup(self) -> None:
        """Clean up GPU between tests."""
        yield
        cleanup_gpu()

    def test_sdpa_attention(self) -> None:
        """Verify SDPA attention implementation is applied."""
        import torch
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            torch_dtype=torch.float16,
            device_map="cuda",
            attn_implementation="sdpa",
        )

        # Check config has correct implementation
        attn_impl = getattr(model.config, "_attn_implementation", None)
        assert attn_impl == "sdpa", f"Expected sdpa, got {attn_impl}"

        del model
        cleanup_gpu()

    def test_eager_attention(self) -> None:
        """Verify eager attention implementation is applied."""
        import torch
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            torch_dtype=torch.float16,
            device_map="cuda",
            attn_implementation="eager",
        )

        attn_impl = getattr(model.config, "_attn_implementation", None)
        assert attn_impl == "eager", f"Expected eager, got {attn_impl}"

        del model
        cleanup_gpu()

    @requires_flash_attn
    def test_flash_attention_2(self) -> None:
        """Verify Flash Attention 2 implementation is applied."""
        import torch
        from transformers import AutoModelForCausalLM

        # Note: Not all models support flash_attention_2
        # Use a compatible model
        try:
            model = AutoModelForCausalLM.from_pretrained(
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                torch_dtype=torch.float16,
                device_map="cuda",
                attn_implementation="flash_attention_2",
            )

            attn_impl = getattr(model.config, "_attn_implementation", None)
            assert attn_impl == "flash_attention_2", f"Expected flash_attention_2, got {attn_impl}"

            del model
        except ValueError as e:
            pytest.skip(f"Flash Attention 2 not supported for this model: {e}")
        finally:
            cleanup_gpu()


@requires_gpu
class TestPyTorchPrecision:
    """Test PyTorch precision/dtype parameters."""

    @pytest.fixture(autouse=True)
    def cleanup(self) -> None:
        """Clean up GPU between tests."""
        yield
        cleanup_gpu()

    def test_float16_precision(self) -> None:
        """Verify float16 precision is applied."""
        import torch
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            torch_dtype=torch.float16,
            device_map="cuda",
        )

        assert model.dtype == torch.float16, f"Expected float16, got {model.dtype}"

        del model
        cleanup_gpu()

    @requires_ampere
    def test_bfloat16_precision(self) -> None:
        """Verify bfloat16 precision is applied (Ampere+ GPU required)."""
        import torch
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )

        assert model.dtype == torch.bfloat16, f"Expected bfloat16, got {model.dtype}"

        del model
        cleanup_gpu()

    def test_float32_precision(self) -> None:
        """Verify float32 precision is applied."""
        import torch
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            torch_dtype=torch.float32,
            device_map="cuda",
        )

        assert model.dtype == torch.float32, f"Expected float32, got {model.dtype}"

        del model
        cleanup_gpu()


@requires_gpu
class TestPyTorchGeneration:
    """Test PyTorch generation parameters."""

    @pytest.fixture(autouse=True)
    def cleanup(self) -> None:
        """Clean up GPU between tests."""
        yield
        cleanup_gpu()

    def test_use_cache_enabled(self) -> None:
        """Verify use_cache affects generation behaviour."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            torch_dtype=torch.float16,
            device_map="cuda",
        )
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        inputs = tokenizer("Hello, world!", return_tensors="pt").to("cuda")

        # With cache
        outputs_cache = model.generate(
            **inputs,
            max_new_tokens=32,
            use_cache=True,
            do_sample=False,
        )

        # Without cache
        outputs_no_cache = model.generate(
            **inputs,
            max_new_tokens=32,
            use_cache=False,
            do_sample=False,
        )

        # Both should produce same output (deterministic)
        assert torch.equal(outputs_cache, outputs_no_cache), "Cache should not affect output"

        del model
        cleanup_gpu()

    def test_output_scores(self) -> None:
        """Verify output_scores returns scores when enabled."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            torch_dtype=torch.float16,
            device_map="cuda",
        )
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        inputs = tokenizer("Hello", return_tensors="pt").to("cuda")

        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=False,
        )

        assert hasattr(outputs, "scores"), "Output should have scores attribute"
        assert outputs.scores is not None, "Scores should not be None"
        assert len(outputs.scores) > 0, "Should have at least one score tensor"

        del model
        cleanup_gpu()

    def test_return_dict_in_generate(self) -> None:
        """Verify return_dict_in_generate returns GenerateOutput."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            torch_dtype=torch.float16,
            device_map="cuda",
        )
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        inputs = tokenizer("Hello", return_tensors="pt").to("cuda")

        # With return_dict
        outputs_dict = model.generate(
            **inputs,
            max_new_tokens=10,
            return_dict_in_generate=True,
            do_sample=False,
        )

        assert hasattr(outputs_dict, "sequences"), "Should have sequences attribute"

        # Without return_dict
        outputs_tensor = model.generate(
            **inputs,
            max_new_tokens=10,
            return_dict_in_generate=False,
            do_sample=False,
        )

        assert isinstance(outputs_tensor, torch.Tensor), "Should return tensor directly"

        del model
        cleanup_gpu()


@requires_gpu
class TestPyTorchSampling:
    """Test PyTorch sampling parameters."""

    @pytest.fixture(autouse=True)
    def cleanup(self) -> None:
        """Clean up GPU between tests."""
        yield
        cleanup_gpu()

    def test_temperature_zero_deterministic(self) -> None:
        """Verify temperature=0 produces deterministic output."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            torch_dtype=torch.float16,
            device_map="cuda",
        )
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        inputs = tokenizer("The capital of France is", return_tensors="pt").to("cuda")

        outputs = []
        for _ in range(3):
            out = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,  # Greedy = deterministic
            )
            outputs.append(tokenizer.decode(out[0], skip_special_tokens=True))

        # All outputs should be identical
        assert all(o == outputs[0] for o in outputs), "Greedy sampling should be deterministic"

        del model
        cleanup_gpu()

    def test_temperature_high_varied(self) -> None:
        """Verify high temperature produces varied output."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            torch_dtype=torch.float16,
            device_map="cuda",
        )
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        inputs = tokenizer("The capital of France is", return_tensors="pt").to("cuda")

        outputs = []
        for i in range(5):
            torch.manual_seed(i)  # Different seed each time
            out = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=True,
                temperature=1.0,
            )
            outputs.append(tokenizer.decode(out[0], skip_special_tokens=True))

        unique = set(outputs)
        # With high temperature and different seeds, we should get some variation
        assert len(unique) >= 2, f"Expected varied outputs, got {len(unique)} unique"

        del model
        cleanup_gpu()

    def test_top_p_sampling(self) -> None:
        """Verify top_p (nucleus) sampling works."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            torch_dtype=torch.float16,
            device_map="cuda",
        )
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        inputs = tokenizer("Once upon a time", return_tensors="pt").to("cuda")

        # Should not raise error with top_p
        out = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
        )

        text = tokenizer.decode(out[0], skip_special_tokens=True)
        assert len(text) > len("Once upon a time"), "Should generate additional text"

        del model
        cleanup_gpu()

    def test_top_k_sampling(self) -> None:
        """Verify top_k sampling works."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            torch_dtype=torch.float16,
            device_map="cuda",
        )
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        inputs = tokenizer("The weather today is", return_tensors="pt").to("cuda")

        # Should not raise error with top_k
        out = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.8,
            top_k=50,
        )

        text = tokenizer.decode(out[0], skip_special_tokens=True)
        assert len(text) > len("The weather today is"), "Should generate additional text"

        del model
        cleanup_gpu()

    def test_repetition_penalty(self) -> None:
        """Verify repetition_penalty affects output."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            torch_dtype=torch.float16,
            device_map="cuda",
        )
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        inputs = tokenizer("I love I love I love", return_tensors="pt").to("cuda")

        # Without repetition penalty (greedy)
        out_no_penalty = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            repetition_penalty=1.0,
        )
        text_no_penalty = tokenizer.decode(out_no_penalty[0], skip_special_tokens=True)

        # With repetition penalty (greedy)
        out_with_penalty = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            repetition_penalty=1.5,
        )
        text_with_penalty = tokenizer.decode(out_with_penalty[0], skip_special_tokens=True)

        # Texts should differ due to penalty
        assert text_no_penalty != text_with_penalty, "Repetition penalty should change output"

        del model
        cleanup_gpu()
