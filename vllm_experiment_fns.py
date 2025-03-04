import os
import json
import sys
import time
import logging
from datetime import datetime
from transformers import AutoTokenizer
from codecarbon import EmissionsTracker
import vllm
import torch


# -----------------------------------------------------------------------------
# Model Loading using vLLM
# -----------------------------------------------------------------------------
def load_model_tokenizer_vllm(model_name):
    """Loads the model and tokenizer for vLLM.
    
    Note: vLLM expects the model name (or path) as its tokenizer argument.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Pass model_name (string) as the tokenizer argument so that vLLM initializes correctly.
    model = vllm.LLM(model_name, tokenizer=model_name)
    return model, tokenizer

# -----------------------------------------------------------------------------
# Energy Tracking Functions
# -----------------------------------------------------------------------------
def start_energy_tracking():
    """Starts CodeCarbon energy tracking."""
    logging.getLogger("codecarbon").setLevel(logging.ERROR)
    tracker = EmissionsTracker(measure_power_secs=1)
    tracker.start()
    return tracker

def stop_energy_tracking(tracker):
    """Stops energy tracking and returns the final emissions data."""
    tracker.stop()
    return tracker.final_emissions_data

# -----------------------------------------------------------------------------
# Inference Function for vLLM with Metrics
# -----------------------------------------------------------------------------
def run_gen_inference_with_metrics_vllm(model, tokenizer, prompts, 
                                        max_input_tokens=512, max_output_tokens=50, batch_size=8):
    """
    Runs generation with vLLM on a list of prompts and measures performance.
    This implementation loops over prompts; adjust batching as needed.
    """
    latencies = []
    total_tokens = 0
    total_input_tokens = 0
    outputs_list = []

    for prompt in prompts:
        # (Optionally, you could truncate the prompt at the token level)
        truncated_prompt = prompt[:max_input_tokens]
        start_time = time.perf_counter()
        # vLLM uses a generate() method; parameters may vary per model.
        output = model.generate(truncated_prompt, max_tokens=max_output_tokens, do_sample=False)
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000.0)
        outputs_list.append(output)

        # Count tokens: tokenize prompt and output separately
        input_ids = tokenizer(truncated_prompt, return_tensors="pt").input_ids
        total_input_tokens += input_ids.numel()
        # Estimate generated tokens by tokenizing the output text.
        gen_ids = tokenizer(output, return_tensors="pt").input_ids
        # Here we assume the generated token count is the difference in token counts.
        gen_tokens = max(gen_ids.numel() - input_ids.numel(), 0)
        total_tokens += gen_tokens

    avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0
    total_time_sec = sum(latencies) / 1000.0
    throughput_qps = len(prompts) / total_time_sec if total_time_sec > 0 else 0.0
    tokens_per_sec = total_tokens / total_time_sec if total_time_sec > 0 else 0.0

    return {
        "avg_latency_ms": avg_latency_ms,
        "throughput_qps": throughput_qps,
        "tokens_per_sec": tokens_per_sec,
        "total_generated_tokens": total_tokens,
        "num_runs": len(prompts),
        "total_time": total_time_sec,
        "total_input_tokens": total_input_tokens,
        "outputs": outputs_list
    }

# -----------------------------------------------------------------------------
# Experiment Extraction and Result Saving
# -----------------------------------------------------------------------------
def extract_experiment_setup(model_name, codecarbon_data, task_type):
    """Extracts configuration and environment details."""
    return {
        "model": model_name,
        "task_type": task_type,
        "date": datetime.today().strftime("%Y-%m-%d"),
        # Using CodeCarbon properties (ensure these attributes exist on your final_emissions_data)
        "cpu_count": codecarbon_data.cpu_count,
        "cpu_model": codecarbon_data.cpu_model,
        "gpu_count": codecarbon_data.gpu_count,
        "gpu_model": codecarbon_data.gpu_model,
        "os": codecarbon_data.os,
        "python_version": sys.version,
        "accelerate_config": "vLLM",  # Note: We're not using Accelerate here.
        "country": codecarbon_data.country_name,
        "region": codecarbon_data.region,
    }

def extract_experiment_results(metrics, codecarbon_data):
    """Extracts performance and energy results."""
    energy_kwh = codecarbon_data.energy_consumed
    energy_joules = energy_kwh * 3.6e6  # Convert kWh to Joules
    tokens_per_joule = (metrics["total_generated_tokens"] / energy_joules) if energy_joules > 0 else 0
    task_specific_performance = {}

    return {
        "inference_performance": {
            "total_inference_time_sec": metrics["total_time"],
            "average_latency_ms": metrics["avg_latency_ms"],
            "throughput_queries_per_sec": metrics["throughput_qps"],
            "throughput_tokens_per_sec": metrics["tokens_per_sec"],
            "total_tokens_generated": metrics["total_generated_tokens"],
            "num_runs": metrics["num_runs"]
        },
        "energy_performance": {
            "cpu_power": codecarbon_data.cpu_power,
            "gpu_power": codecarbon_data.gpu_power,
            "ram_power": codecarbon_data.ram_power,
            "cpu_energy": codecarbon_data.cpu_energy,
            "gpu_energy": codecarbon_data.gpu_energy,
            "ram_energy": codecarbon_data.ram_energy,
            "total_energy_consumed_kwh": energy_kwh,
            "total_energy_consumed_joules": energy_joules,
            "energy_efficiency_tokens_per_joule": tokens_per_joule,
            "final_emissions": codecarbon_data.emissions
        },
        "task-specific_performance": task_specific_performance
    }

def save_results(task_type, benchmark_results):
    """Saves benchmark results to a JSON file (appending if the file exists)."""
    output_dir = "benchmark_results"
    os.makedirs(output_dir, exist_ok=True)
    output_json_path = os.path.join(output_dir, f"{task_type}_results.json")
    
    if os.path.exists(output_json_path):
        with open(output_json_path, "r") as json_file:
            try:
                existing_data = json.load(json_file)
                if not isinstance(existing_data, list):
                    existing_data = [existing_data]
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []
    
    existing_data.append(benchmark_results)
    with open(output_json_path, "w") as json_file:
        json.dump(existing_data, json_file, indent=4)
    
    return output_json_path

# -----------------------------------------------------------------------------
# Experiment Runner for vLLM
# -----------------------------------------------------------------------------
def run_experiment_vllm(model_name, prompts, task_type, 
                        max_input_tokens=512, max_output_tokens=50, batch_size=8):
    model, tokenizer = load_model_tokenizer_vllm(model_name)
    tracker = start_energy_tracking()
    metrics = run_gen_inference_with_metrics_vllm(model, tokenizer, prompts, max_input_tokens, max_output_tokens, batch_size)
    codecarbon_data = stop_energy_tracking(tracker)
    experiment_setup = extract_experiment_setup(model_name, codecarbon_data, task_type)
    experiment_results = extract_experiment_results(metrics, codecarbon_data)
    benchmark_results = {
        "experiment_setup": experiment_setup,
        "experiment_variables": {
            "input_tokens": metrics["total_input_tokens"],
            "output_tokens": metrics["total_generated_tokens"]
        },
        "experiment_results": experiment_results
    }
    output_json_path = save_results(task_type, benchmark_results)
    print(json.dumps({
        "model": model_name,
        "energy_consumed_kwh": experiment_results["energy_performance"]["total_energy_consumed_kwh"],
        "energy_efficiency_tokens_per_joule": experiment_results["energy_performance"]["energy_efficiency_tokens_per_joule"]
    }, indent=4))
    return benchmark_results

# -----------------------------------------------------------------------------
# Main Entry Point (when running as a script)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from datasets import load_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--task_type", type=str, default="text_generation")
    parser.add_argument("--max_input_tokens", type=int, default=512)
    parser.add_argument("--max_output_tokens", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    # Load a dataset sample for testing
    ds = load_dataset("lighteval/pile_helm", "arxiv")["test"]
    ds = ds.select(range(10))
    prompts = [sample["text"] for sample in ds]
    
    run_experiment_vllm(
        model_name=args.model_name,
        prompts=prompts,
        task_type=args.task_type,
        max_input_tokens=args.max_input_tokens,
        max_output_tokens=args.max_output_tokens,
        batch_size=args.batch_size
    )
