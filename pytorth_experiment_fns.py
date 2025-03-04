import os
import json
import sys
import time
import logging
from datetime import datetime
from contextlib import redirect_stdout
from io import StringIO
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator, notebook_launcher
from codecarbon import EmissionsTracker
import torch
from optimum_benchmark import Benchmark, BenchmarkConfig, TorchrunConfig, InferenceConfig, PyTorchConfig
from optimum_benchmark.logging_utils import setup_logging


# -----------------------------------------------------------------------------
# Backend model loading: choose between PyTorch, vLLM, ONNX or TensorRT.
# -----------------------------------------------------------------------------
def load_model_tokenizer_backend(model_name, backend="pytorch"):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    )
    model.eval()

    return model, tokenizer

# -----------------------------------------------------------------------------
# Accelerator 
# -----------------------------------------------------------------------------

def prep_distributed_env(model, tokenizer):
    """Prepares model and tokenizer for FSDP distributed inference using Accelerate."""
    accelerator = Accelerator()
    
    model, tokenizer = accelerator.prepare(model, tokenizer)

    accelerator.print(f"Using {accelerator.num_processes} GPUs")
    print(f"Model is on {next(model.parameters()).device}") 
    return model, tokenizer, accelerator

# -----------------------------------------------------------------------------
# energy tracking functions 
# -----------------------------------------------------------------------------

def start_energy_tracking():
    """Starts CodeCarbon energy tracking."""
    tracker = EmissionsTracker(measure_power_secs=1, allow_multiple_runs=True)
    tracker.start()
    return tracker

def stop_energy_tracking(tracker):
    """Stops energy tracking and returns the final emissions data."""
    tracker.stop()  
    codecarbon_data = tracker.final_emissions_data
    return codecarbon_data

# -----------------------------------------------------------------------------
# Inference function that measures performance metrics.
# -----------------------------------------------------------------------------
def run_gen_inference_with_metrics(model, tokenizer, accelerator, prompts, 
                                   max_input_tokens=512, max_output_tokens=50, batch_size=8):
    """
    Runs inference and returns performance metrics.
    """
    truncated_prompts = [
        tokenizer.decode(
            tokenizer(p, truncation=True, max_length=max_input_tokens, return_tensors="pt").input_ids[0],
            skip_special_tokens=True
        )
        for p in prompts
    ]
    
    # Sort prompts by token length for efficient batching
    sorted_prompts = sorted(truncated_prompts, key=lambda x: len(tokenizer.tokenize(x)))
    latencies = []
    ttft_values = []  # (time-to-first-token, not currently measured)
    total_tokens = 0
    total_input_tokens = 0  # Track input tokens
    device = accelerator.device
    num_batches = (len(sorted_prompts) + batch_size - 1) // batch_size

    for i in range(num_batches):
        batch = sorted_prompts[i * batch_size: (i + 1) * batch_size]

        # Tokenize batch
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_input_tokens)
        input_ids = encoded.input_ids.to(device)
        total_input_tokens += input_ids.numel()  # Count input tokens

    
        # Generate outputs with DistributedDataParallel fix
        start_time = time.perf_counter()
        if hasattr(model, "module"):
            outputs = model.module.generate(input_ids, max_new_tokens=max_output_tokens, do_sample=False)
        else:
            outputs = model.generate(input_ids, max_new_tokens=max_output_tokens, do_sample=False)
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000.0)

        # Count generated tokens per prompt
        for j in range(len(batch)):
            prompt_len = input_ids[j].shape[0]
            gen_len = outputs[j].shape[0] - prompt_len
            total_tokens += gen_len

    avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0
    avg_ttft_ms = sum(ttft_values) / len(ttft_values) if ttft_values else 0.0
    total_time_sec = sum(latencies) / 1000.0
    throughput_qps = len(sorted_prompts) / total_time_sec if total_time_sec > 0 else 0.0
    tokens_per_sec = total_tokens / total_time_sec if total_time_sec > 0 else 0.0

    return {
        "avg_latency_ms": avg_latency_ms,
        "avg_ttft_ms": avg_ttft_ms,
        "throughput_qps": throughput_qps,
        "tokens_per_sec": tokens_per_sec,
        "total_generated_tokens": total_tokens,
        "num_runs": len(sorted_prompts),
        "total_time": total_time_sec,
        "total_input_tokens": total_input_tokens  
    }

# -----------------------------------------------------------------------------
# Experiment extraction and result saving functions.
# -----------------------------------------------------------------------------
def extract_experiment_setup(model_name, codecarbon_data, accelerator, task_type):
    """Extracts experiment configuration and environment details."""
    return {
        "model": model_name,
        "task_type": task_type,
        "date": datetime.today().strftime("%Y-%m-%d"),
        "cpu_count": codecarbon_data.cpu_count,
        "cpu_model": codecarbon_data.cpu_model,
        "gpu_count": codecarbon_data.gpu_count,
        "gpu_model": codecarbon_data.gpu_model,
        "os": codecarbon_data.os,
        "python_version": sys.version,
        "accelerate_config": {
            "distributed_type": str(accelerator.distributed_type),
            "num_processes": accelerator.num_processes,
            "local_process_index": accelerator.local_process_index
        },
        "country": codecarbon_data.country_name,
        "region": codecarbon_data.region,
    }

def extract_experiment_results(metrics, codecarbon_data):
    """
    Extracts performance and energy results from the inference metrics and
    CodeCarbon energy data.
    """
    energy_kwh = codecarbon_data.energy_consumed
    energy_joules = energy_kwh * 3.6e6  # kWh to Joules
    tokens_per_joule = (metrics["total_generated_tokens"] / energy_joules) if energy_joules > 0 else 0

    # Placeholder for task-specific performance (e.g., text-generation vs summarization)
    task_specific_performance = {}

    return {
        "inference_performance": {
            "total_inference_time_sec": metrics["total_time"],
            "average_latency_ms_per_batch": metrics["avg_latency_ms"],
            "average_ttft_ms": metrics["avg_ttft_ms"],
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
    """
    Saves the benchmark results to a JSON file. Appends new experiments if the file exists.
    """
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
# Experiment runner with optional optimum_benchmark integration.
# -----------------------------------------------------------------------------
def run_experiment(model_name, prompts, inference_fn, task_type, 
                   backend="pytorch", use_optimum=False, **inference_kwargs):
    """
    Runs an experiment in one of two modes:
      1. If use_optimum is True, it uses the optimum_benchmark package to run the benchmark.
      2. Otherwise, it loads the model via the specified backend, runs energy tracking, 
         inference and extracts results as before.
    """
    if use_optimum:
        # Set up logging (adjust level and handlers as needed)
        setup_logging(level="INFO")

        launcher_config = TorchrunConfig(nproc_per_node=1)
        scenario_config = InferenceConfig(
            latency=True, 
            memory=True, 
            input_shapes={"sequence_length": 128} 
        )
        backend_config = PyTorchConfig(model=model_name, device="cuda", device_ids="0", no_weights=True)

        benchmark_config = BenchmarkConfig(
            name=f"{backend}_{model_name}",
            scenario=scenario_config,
            launcher=launcher_config,
            backend=backend_config,
        )

        # Launch the benchmark using the new API.
        benchmark_report = Benchmark.launch(benchmark_config)

        # Convert benchmark artifacts to a dictionary.
        benchmark_results = benchmark_report.to_dict()

        print(json.dumps({
            "model": model_name,
            "optimum_benchmark_results": benchmark_results
        }, indent=4))
        return benchmark_results
    else:
        # Standard run using my prev energy tracking and inference measurement
        model, tokenizer = load_model_tokenizer_backend(model_name, backend=backend)
        model, tokenizer, accelerator = prep_distributed_env(model, tokenizer)
        tracker = start_energy_tracking()
        inference_metrics = inference_fn(model, tokenizer, accelerator, prompts, **inference_kwargs)
        codecarbon_data = stop_energy_tracking(tracker)
        experiment_setup = extract_experiment_setup(model_name, codecarbon_data, accelerator, task_type)
        experiment_results = extract_experiment_results(inference_metrics, codecarbon_data)
        benchmark_results = {
            "experiment_setup": experiment_setup,
            "experiment_variables": {
                "input_tokens": inference_metrics["total_input_tokens"], 
                "output_tokens": inference_metrics["total_generated_tokens"]
            },
            "experiment_results": experiment_results
        }
        output_json_path = save_results(task_type, benchmark_results)
        results_summary = {
            "model": benchmark_results["experiment_setup"]["model"],
            "energy_consumed_kwh": benchmark_results["experiment_results"]["energy_performance"]["total_energy_consumed_kwh"],
            "energy_efficiency_tokens_per_joule": benchmark_results["experiment_results"]["energy_performance"]["energy_efficiency_tokens_per_joule"]
        }
        print(json.dumps(results_summary, indent=4))
        return benchmark_results
