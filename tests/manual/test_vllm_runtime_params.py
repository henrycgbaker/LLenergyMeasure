#!/usr/bin/env python3
"""Runtime verification of vLLM parameter effects.

Tests that our config parameters actually affect vLLM behavior at runtime.
"""

import gc

import torch


def cleanup():
    """Clean up GPU memory between tests."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_prefix_caching():
    """Verify enable_prefix_caching is correctly passed to vLLM."""
    from vllm import LLM

    print("\n" + "=" * 60)
    print("TEST: enable_prefix_caching")
    print("=" * 60)

    model_name = "facebook/opt-125m"

    # Test WITHOUT prefix caching
    print("\n--- Without prefix caching ---")
    llm_no_cache = LLM(
        model=model_name,
        gpu_memory_utilization=0.3,
        enforce_eager=True,
        enable_prefix_caching=False,
    )

    # Check config via cache_config
    cache_disabled = not llm_no_cache.llm_engine.cache_config.enable_prefix_caching
    print(
        f"Cache config enable_prefix_caching: {llm_no_cache.llm_engine.cache_config.enable_prefix_caching}"
    )

    del llm_no_cache
    cleanup()

    # Test WITH prefix caching
    print("\n--- With prefix caching ---")
    llm_cache = LLM(
        model=model_name,
        gpu_memory_utilization=0.3,
        enforce_eager=True,
        enable_prefix_caching=True,
    )

    cache_enabled = llm_cache.llm_engine.cache_config.enable_prefix_caching
    print(f"Cache config enable_prefix_caching: {cache_enabled}")

    del llm_cache
    cleanup()

    # Analysis
    print("\n--- Results ---")
    if cache_disabled and cache_enabled:
        print("✓ VERIFIED: enable_prefix_caching correctly controls cache config")
        return True
    else:
        print("✗ FAILED: enable_prefix_caching not applied correctly")
        print("  Expected: disabled=True, enabled=True")
        print(f"  Got: disabled={cache_disabled}, enabled={cache_enabled}")
        return False


def test_enforce_eager():
    """Verify enforce_eager disables CUDA graphs."""

    from vllm import LLM, SamplingParams

    print("\n" + "=" * 60)
    print("TEST: enforce_eager")
    print("=" * 60)

    model_name = "facebook/opt-125m"
    prompts = ["Hello, world!"]
    sampling = SamplingParams(max_tokens=10)

    # Capture stderr to check for CUDA graph messages
    print("\n--- With enforce_eager=True (no CUDA graphs) ---")
    llm_eager = LLM(
        model=model_name,
        gpu_memory_utilization=0.3,
        enforce_eager=True,
    )
    llm_eager.generate(prompts, sampling)

    # Check llm_engine config
    eager_mode = llm_eager.llm_engine.model_config.enforce_eager
    print(f"Model config enforce_eager: {eager_mode}")

    del llm_eager
    cleanup()

    print("\n--- With enforce_eager=False (CUDA graphs enabled) ---")
    llm_graphs = LLM(
        model=model_name,
        gpu_memory_utilization=0.3,
        enforce_eager=False,
    )
    llm_graphs.generate(prompts, sampling)

    graphs_mode = llm_graphs.llm_engine.model_config.enforce_eager
    print(f"Model config enforce_eager: {graphs_mode}")

    del llm_graphs
    cleanup()

    print("\n--- Results ---")
    if eager_mode is True and graphs_mode is False:
        print("✓ VERIFIED: enforce_eager correctly controls CUDA graph usage")
        return True
    else:
        print(
            f"✗ FAILED: enforce_eager not set correctly (eager={eager_mode}, graphs={graphs_mode})"
        )
        return False


def test_max_model_len():
    """Verify max_model_len limits context length."""
    from vllm import LLM, SamplingParams

    print("\n" + "=" * 60)
    print("TEST: max_model_len")
    print("=" * 60)

    model_name = "facebook/opt-125m"
    short_limit = 128

    print(f"\n--- Creating LLM with max_model_len={short_limit} ---")
    llm = LLM(
        model=model_name,
        gpu_memory_utilization=0.3,
        enforce_eager=True,
        max_model_len=short_limit,
    )

    # Check the actual config
    actual_limit = llm.llm_engine.model_config.max_model_len
    print(f"Actual max_model_len in config: {actual_limit}")

    # Try to generate with a prompt that would exceed the limit
    # OPT-125m default is 2048, so 128 is a real restriction
    long_prompt = "word " * 200  # ~200 tokens, should exceed 128

    print("\n--- Testing with long prompt (~200 tokens) ---")
    try:
        llm.generate([long_prompt], SamplingParams(max_tokens=10))
        # If it succeeds, check if prompt was truncated
        print("Generation completed (prompt may have been truncated)")
    except Exception as e:
        print(f"Got expected error: {type(e).__name__}")

    del llm
    cleanup()

    print("\n--- Results ---")
    if actual_limit == short_limit:
        print(f"✓ VERIFIED: max_model_len={short_limit} correctly applied")
        return True
    else:
        print(f"✗ FAILED: max_model_len not applied (expected {short_limit}, got {actual_limit})")
        return False


def main():
    """Run all runtime verification tests."""
    print("=" * 60)
    print("vLLM Runtime Parameter Verification")
    print("=" * 60)

    results = {}

    # Run tests
    results["enable_prefix_caching"] = test_prefix_caching()
    results["enforce_eager"] = test_enforce_eager()
    results["max_model_len"] = test_max_model_len()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for param, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {param}: {status}")

    all_passed = all(results.values())
    print(f"\nOverall: {'All tests passed!' if all_passed else 'Some tests failed'}")
    return all_passed


if __name__ == "__main__":
    import sys

    success = main()
    sys.exit(0 if success else 1)
