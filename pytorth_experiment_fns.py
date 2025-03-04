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
import psutil
from fvcore.nn import FlopCountAnalysis
import torch.nn as nn

# -----------------------------------------------------------------------------
# for the FLOP counter:

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, input_ids):
        return self.model(input_ids=input_ids)
    
# -----------------------------------------------------------------------------
# Backend model loading
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
def prep_distributed_env(model, tokenizer, placement):
    """Prepares model and tokenizer for FSDP distributed inference using Accelerate."""
    accelerator = Accelerator(device_placement=True)
    device = torch.device("cuda", accelerator.process_index % 2)  
    print(f"Using device: {device}")

    
    model, tokenizer = accelerator.prepare(model, tokenizer)
    accelerator.print(f"Using {accelerator.num_processes} GPUs")
    print(f"Model is on {next(model.parameters()).device}") 
    return model, tokenizer, accelerator



def prep_distributed_env(model, tokenizer, gpu_list=[0, 1]):
    """Prepares model and tokenizer for FSDP distributed inference using Accelerate, 
    """
    
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_list))

    accelerator = Accelerator(device_placement=True)

    local_rank = accelerator.local_process_index  # Get local process index
    selected_device = gpu_list[local_rank % len(gpu_list)]  # Cycle through provided GPUs
    device = torch.device(f"cuda:{selected_device}")

    accelerator.print(f"Using device: {device} (Local Rank: {local_rank})")

    model, tokenizer = accelerator.prepare(model, tokenizer)

    accelerator.print(f"Using {accelerator.num_processes} GPUs: {gpu_list}")
    print(f"Model is on {next(model.parameters()).device}") 

    return model, tokenizer, accelerator


# -----------------------------------------------------------------------------
# Energy tracking functions 
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
# Compute performance metrics collection.
# -----------------------------------------------------------------------------
def get_compute_performance_metrics(model=None, tokenizer=None, device=None, input_length=128):
    """
    Returns a dictionary of low-level compute metrics.
    If model, tokenizer, and device are provided, attempts to compute FLOPs using fvcore.
    Also reports GPU memory usage, GPU utilization, and CPU usage.
    """
    compute_metrics = {}

    # GPU memory metrics (if CUDA is available)
    if torch.cuda.is_available() and device is not None:
        torch.cuda.reset_peak_memory_stats(device)
        compute_metrics["current_memory_allocated_bytes"] = torch.cuda.memory_allocated(device)
        compute_metrics["max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        compute_metrics["current_memory_reserved_bytes"] = torch.cuda.memory_reserved(device)
        compute_metrics["max_memory_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        # GPU utilization using nvidia-smi
        import subprocess
        try:
            result = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]
            )
            gpu_util = float(result.decode("utf-8").strip())
            compute_metrics["gpu_utilization_percent"] = gpu_util
        except Exception as e:
            compute_metrics["gpu_utilization_percent"] = f"Error: {str(e)}"
    else:
        compute_metrics["gpu"] = "CUDA not available or device not provided"

    # CPU metrics using psutil
    try:
        process = psutil.Process(os.getpid())
        # Measure CPU usage over a short interval
        cpu_usage = process.cpu_percent(interval=1.0)
        compute_metrics["cpu_usage_percent"] = cpu_usage
        cpu_mem = process.memory_info().rss
        compute_metrics["cpu_memory_usage_bytes"] = cpu_mem
    except ImportError:
        compute_metrics["cpu_usage_percent"] = "psutil not installed"

    # FLOPs estimation fvcore 
    if model is not None and tokenizer is not None and device is not None:
        try:
            from fvcore.nn import FlopCountAnalysis
            # Use the underlying model if it is wrapped (e.g. in DistributedDataParallel)
            model_to_trace = model.module if hasattr(model, "module") else model
            # Set to eval mode to disable any training-specific branches
            model_to_trace.eval()
            # Wrap the model in our custom wrapper
            wrapped_model = ModelWrapper(model_to_trace)
            # Create a dummy input tensor of shape (1, input_length)
            dummy_input_ids = torch.ones((1, input_length), dtype=torch.long).to(device)
            # Run FLOP analysis on the wrapped model.
            flops_analyzer = FlopCountAnalysis(wrapped_model, dummy_input_ids)
            compute_metrics["flops_forward_pass"] = flops_analyzer.total()
        except ImportError:
            compute_metrics["flops_forward_pass"] = "fvcore not installed"
        except Exception as e:
            compute_metrics["flops_forward_pass"] = f"Error: {str(e)}"
    else:
        compute_metrics["flops_forward_pass"] = "model, tokenizer, or device not provided"

    return compute_metrics

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

def extract_experiment_results(metrics, codecarbon_data, model=None, tokenizer=None, device=None):
    """
    Extracts performance and energy results from the inference metrics and
    CodeCarbon energy data. Also extracts low-level compute performance metrics.
    """
    energy_kwh = codecarbon_data.energy_consumed
    energy_joules = energy_kwh * 3.6e6  # kWh to Joules
    tokens_per_joule = (metrics["total_generated_tokens"] / energy_joules) if energy_joules > 0 else 0

    # Placeholder for task-specific performance (e.g., text-generation vs summarization)
    task_specific_performance = {}

    # Gather compute performance metrics (pass model, tokenizer, and accelerator device if available)
    compute_metrics = get_compute_performance_metrics(model=model, tokenizer=tokenizer, device=device)

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
        "compute_performance": compute_metrics,
        "task-specific_performance": task_specific_performance
    }

def save_results(task_type, benchmark_results):
    """
    Saves the benchmark results to a JSON file. Appends new experiments if the file exists.
    """
    # output directory
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
    
    # save it to that
    existing_data.append(benchmark_results)
    with open(output_json_path, "w") as json_file:
        json.dump(existing_data, json_file, indent=4)
    
    return output_json_path


def aggregate_process_results(results_list):
    """
    Aggregate a list of experiment_results (one per process) into a single dictionary.
    
    For example, this function sums total inference time, total tokens, and energy,
    and averages the latency. Adjust as needed for your metrics.
    """
    aggregated = {
        "inference_performance": {},
        "energy_performance": {},
        "compute_performance": {}
    }
    
    # Aggregate inference performance
    total_time = sum(r["inference_performance"].get("total_inference_time_sec", 0) for r in results_list)
    total_tokens = sum(r["inference_performance"].get("total_tokens_generated", 0) for r in results_list)
    total_runs = sum(r["inference_performance"].get("num_runs", 0) for r in results_list)
    avg_latency = sum(r["inference_performance"].get("average_latency_ms_per_batch", 0) for r in results_list) / len(results_list)
    total_throughput = sum(r["inference_performance"].get("throughput_tokens_per_sec", 0) for r in results_list)
    
    aggregated["inference_performance"] = {
        "total_inference_time_sec": total_time,
        "total_tokens_generated": total_tokens,
        "num_runs": total_runs,
        "average_latency_ms_per_batch": avg_latency,
        "throughput_tokens_per_sec": total_throughput
    }
    
    # Aggregate energy performance: sum energy consumed and recalc tokens per joule.
    total_energy_kwh = sum(r["energy_performance"].get("total_energy_consumed_kwh", 0) for r in results_list)
    total_energy_joules = sum(r["energy_performance"].get("total_energy_consumed_joules", 0) for r in results_list)
    tokens_per_joule = total_tokens / total_energy_joules if total_energy_joules > 0 else 0
    
    aggregated["energy_performance"] = {
        "total_energy_consumed_kwh": total_energy_kwh,
        "total_energy_consumed_joules": total_energy_joules,
        "energy_efficiency_tokens_per_joule": tokens_per_joule
    }
    
    # For compute performance, you might choose to average metrics or simply take the first process’ values.
    aggregated["compute_performance"] = results_list[0].get("compute_performance", {})
    
    return aggregated

# -----------------------------------------------------------------------------
# Experiment runner with optional optimum_benchmark integration.
# -----------------------------------------------------------------------------

def run_experiment(model_name, prompts, inference_fn, task_type, 
                   backend="pytorch", use_optimum=False, **inference_kwargs):
    """
    Runs an experiment in one of two modes.
    
    In the standard mode (use_optimum=False), this function:
      1. Loads the model via the specified backend.
      2. Prepares the distributed environment.
      3. Starts energy tracking, runs inference, and stops tracking.
      4. Extracts per-process experiment results.
      5. Aggregates the per-process results.
      6. Saves the aggregated benchmark results.
    
    The final saved JSON structure nests the aggregated results (experiment_results) along
    with common experiment_setup and experiment_variables.
    """
    if use_optimum:
        # --- Optimum benchmark branch (unchanged) ---
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

        benchmark_report = Benchmark.launch(benchmark_config)
        benchmark_results = benchmark_report.to_dict()

        print(json.dumps({
            "model": model_name,
            "optimum_benchmark_results": benchmark_results
        }, indent=4))
        return benchmark_results
    else:
        # --- Standard experiment branch ---
        model, tokenizer = load_model_tokenizer_backend(model_name, backend=backend)
        model, tokenizer, accelerator = prep_distributed_env(model, tokenizer)
        tracker = start_energy_tracking()
        
        # Run inference (assumed to be executed in a distributed fashion)
        inference_metrics = inference_fn(model, tokenizer, accelerator, prompts, **inference_kwargs)
        
        codecarbon_data = stop_energy_tracking(tracker)
        
        # Each process computes its own local result.
        local_result = extract_experiment_results(
            inference_metrics, codecarbon_data, model=model, tokenizer=tokenizer, device=accelerator.device
        )
        # Gather the local_result from all processes into a list.
        # accelerator.gather will collect the data from each process.
        all_results = accelerator.gather_object(local_result)

        
        # Now aggregate the actual results from each process.
        aggregated_results = aggregate_process_results(all_results)
        
        # Extract common experimental setup and variables
        experiment_setup = extract_experiment_setup(model_name, codecarbon_data, accelerator, task_type)
        experiment_variables = {
            "input_tokens": inference_metrics["total_input_tokens"],
            "output_tokens": inference_metrics["total_generated_tokens"]
        }
        
        # Compose the final benchmark_results structure
        benchmark_results = {
            "experiment_setup": experiment_setup,
            "experiment_variables": experiment_variables,
            "experiment_results": aggregated_results
        }
        
        # Save the aggregated results using your save_results function
        output_json_path = save_results(task_type, benchmark_results)
        
        results_summary = {
            "model": benchmark_results["experiment_setup"]["model"],
            "energy_consumed_kwh": benchmark_results["experiment_results"]["energy_performance"]["total_energy_consumed_kwh"],
            "energy_efficiency_tokens_per_joule": benchmark_results["experiment_results"]["energy_performance"]["energy_efficiency_tokens_per_joule"]
        }
        print(json.dumps(results_summary, indent=4))
        return benchmark_results