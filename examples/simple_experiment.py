#!/usr/bin/env python3
"""
Simple experiment example using v2.0 API.

This demonstrates the basic workflow for running a single experiment
with the modernized v2.0 API.
"""

import torch
from llm_efficiency.config import ExperimentConfig, BatchingConfig, DecoderConfig
from llm_efficiency.core.model_loader import load_model_and_tokenizer
from llm_efficiency.metrics import FLOPsCalculator, get_gpu_memory_stats
from llm_efficiency.utils.logging import setup_logging


def main():
    """Run a simple experiment."""

    # Setup logging
    setup_logging(level="INFO", rich_output=True)

    print("=" * 60)
    print("LLM Efficiency Measurement Tool v2.0 - Simple Example")
    print("=" * 60)

    # Create configuration using Pydantic models
    config = ExperimentConfig(
        # Model
        model_name="hf-internal-testing/tiny-random-gpt2",  # Tiny model for demo
        precision="float16",
        # Inference settings
        num_input_prompts=5,
        max_input_tokens=32,
        max_output_tokens=16,
        # Batching
        batching=BatchingConfig(batch_size=2),
        # Decoder
        decoder=DecoderConfig(mode="greedy"),
    )

    print(f"\n📋 Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Precision: {config.precision}")
    print(f"  Batch size: {config.batching.batch_size}")
    print(f"  Decoder: {config.decoder.mode}")

    # Load model and tokenizer
    print(f"\n🔄 Loading model...")
    model, tokenizer = load_model_and_tokenizer(config)

    device = next(model.parameters()).device
    print(f"  Device: {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Calculate FLOPs using v2.0 FLOPs calculator
    print(f"\n🔢 Computing FLOPs...")
    calculator = FLOPsCalculator()

    sequence_length = config.max_input_tokens
    is_quantized = config.quantization.enabled

    flops = calculator.get_flops(
        model=model,
        model_name=config.model_name,
        sequence_length=sequence_length,
        device=device,
        is_quantized=is_quantized,
    )

    print(f"  FLOPs (seq_len={sequence_length}): {flops:,}")

    # Test inference
    print(f"\n🚀 Running test inference...")
    test_prompts = [
        "Hello, how are you?",
        "What is the meaning of life?",
    ]

    inputs = tokenizer(
        test_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.max_input_tokens,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_output_tokens,
            do_sample=config.decoder.do_sample,
            temperature=config.decoder.temperature,
        )

    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    print(f"\n📝 Generated outputs:")
    for i, text in enumerate(generated_texts):
        print(f"  [{i+1}] {text[:100]}...")

    # Memory stats
    if torch.cuda.is_available():
        memory_stats = get_gpu_memory_stats(device)
        print(f"\n💾 Memory Usage:")
        print(f"  Allocated: {memory_stats['gpu_current_memory_allocated_bytes'] / 1e9:.2f} GB")
        print(
            f"  Peak: {memory_stats['gpu_max_memory_allocated_bytes'] / 1e9:.2f} GB"
        )

    print(f"\n✅ Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
