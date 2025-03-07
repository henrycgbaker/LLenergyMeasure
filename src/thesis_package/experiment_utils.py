import os
import json
import sys
import time
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
import uuid
from datetime import datetime
import psutil

from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
import yaml

def load_experiment_config(config_path="experiment_configs.yaml"):
    """Loads experiment configuration settings from a YAML file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def load_model_tokenizer_backend(model_name, backend="pytorch", fp_precision="float16"):
    if fp_precision == "float32":
        dtype = torch.float32
    else:
        dtype = torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.eval()
    return model, tokenizer
   

def prep_distributed_env(model, tokenizer, gpu_list=[0, 1]):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_list))
    accelerator = Accelerator(device_placement=True)
    local_rank = accelerator.local_process_index
    selected_device = gpu_list[local_rank % len(gpu_list)]
    device = torch.device(f"cuda:{selected_device}")
    accelerator.print(f"Using device: {device} (Local Rank: {local_rank})")
    model, tokenizer = accelerator.prepare(model, tokenizer)
    accelerator.print(f"Using {accelerator.num_processes} GPUs: {gpu_list}")
    print(f"Model is on {next(model.parameters()).device}")
    return model, tokenizer, accelerator


def extract_experiment_setup(model_name, codecarbon_data, accelerator, task_type):
    return {
        "model": model_name,
        "task_type": task_type,
        "date": datetime.now().strftime("%B %d, %Y at %I:%M:%S %p"),
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
    energy_kwh = codecarbon_data.energy_consumed
    energy_joules = energy_kwh * 3.6e6  # Convert kWh to Joules
    tokens_per_joule = (metrics["total_generated_tokens"] / energy_joules) if energy_joules > 0 else 0

    # Placeholder for task-specific performance
    task_specific_performance = {}

    # Import the metrics function from your metrics module
    from metrics import get_compute_performance_metrics  
    compute_metrics = get_compute_performance_metrics(model=model, tokenizer=tokenizer, device=device)

    return {
        "inference_performance": {
            "total_inference_time_sec": metrics["total_time"],
            "average_latency_ms_per_batch": metrics["avg_latency_ms"],
            "throughput_queries_per_sec": metrics["throughput_qps"],
            "throughput_tokens_per_sec": metrics["tokens_per_sec"],
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


def aggregate_experiments(results):
    aggregated = {}

    aggregated["experiment_setup"] = results[0]["experiment_setup"].copy()
    if "accelerate_config" in aggregated["experiment_setup"]:
        aggregated["experiment_setup"]["accelerate_config"].pop("local_process_index", None)
    aggregated["experiment_variables"] = results[0]["experiment_variables"]

    # Aggregate inference performance by averaging
    inf_keys = ["total_inference_time_sec", "average_latency_ms_per_batch", "throughput_queries_per_sec", "throughput_tokens_per_sec"]
    agg_inference = {key: sum(proc["experiment_results"]["inference_performance"][key] for proc in results) / len(results)
                     for key in inf_keys}

    # Aggregate energy performance (averaging some keys, summing others)
    energy_keys_avg = ["cpu_power", "gpu_power", "ram_power", "energy_efficiency_tokens_per_joule"]
    energy_keys_sum = ["cpu_energy", "gpu_energy", "ram_energy", "total_energy_consumed_kwh", "total_energy_consumed_joules", "final_emissions"]
    agg_energy = {}
    for key in energy_keys_avg:
        agg_energy[key] = sum(proc["experiment_results"]["energy_performance"][key] for proc in results) / len(results)
    for key in energy_keys_sum:
        agg_energy[key] = sum(proc["experiment_results"]["energy_performance"][key] for proc in results)

    # Aggregate compute performance.
    compute_setup = results[0]["experiment_results"]["compute_performance"]
    agg_compute = {
        "gpu": compute_setup.get("gpu", "N/A"),
        "flops_forward_pass": compute_setup.get("flops_forward_pass", "N/A"),
    }
    numeric_compute_keys = [
        "cpu_usage_percent",
        "cpu_memory_usage_bytes",
        "gpu_utilization_percent",
        "current_memory_allocated_bytes",
        "max_memory_allocated_bytes",
        "current_memory_reserved_bytes",
        "max_memory_reserved_bytes"
    ]
    for key in numeric_compute_keys:
        values = []
        for proc in results:
            val = proc["experiment_results"]["compute_performance"].get(key)
            if isinstance(val, list):
                values.append(val)
            elif isinstance(val, (int, float)):
                values.append(val)
        if values:
            if isinstance(values[0], list):
                list_length = len(values[0])
                averaged = [sum(v[i] for v in values) / len(values) for i in range(list_length)]
                agg_compute[key] = averaged
            else:
                agg_compute[key] = sum(values) / len(values)
    
    for extra_key in ["cpu_vendor", "AMD_CONSTANT_POWER"]:
        if extra_key in compute_setup:
            agg_compute[extra_key] = compute_setup[extra_key]
    
    task_perf = results[0]["experiment_results"].get("task-specific_performance", {})

    aggregated["experiment_results"] = {
        "inference_performance": agg_inference,
        "energy_performance": agg_energy,
        "compute_performance": agg_compute,
        "task-specific_performance": task_perf
    }
    return aggregated

### ---

def get_persistent_unique_id():
    ID_FILE = "experiment_id.txt"
    if os.path.exists(ID_FILE):
        with open(ID_FILE, "r") as f:
            last_id = int(f.read().strip())
    else:
        last_id = 0
    new_id = last_id + 1
    with open(ID_FILE, "w") as f:
        f.write(str(new_id))
    return f"{new_id:04d}"

### ---

def save_results(task_type, benchmark_results):
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
