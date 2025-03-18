import sys
from datetime import datetime
import json
import os

def make_json_serializable(obj):
    """Recursively convert non-JSON-serializable objects to strings."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    else:
        try:
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            return str(obj)

def aggregate_experiments(all_results):
    """
    Aggregates results from multiple processes.
    
    Parameters:
      - all_results: List of per-process results.
      
    Returns:
      A dictionary of aggregated experiment results.
    """
    # Use the experiment setup and variables from the first process as a baseline.
    aggregated = {
        "experiment_setup": all_results[0].get("experiment_setup", {}),
        "experiment_variables": all_results[0].get("experiment_variables", {}),
    }
    
    # Aggregate inference performance metrics (averaging for performance, summing counts).
    inf_keys = ["total_inference_time_sec", "average_latency_ms_per_batch", "throughput_queries_per_sec", "throughput_tokens_per_sec"]
    agg_inference = {}
    for key in inf_keys:
        values = [proc["experiment_results"]["inference_performance"].get(key, 0) for proc in all_results if "experiment_results" in proc]
        agg_inference[key] = sum(values) / len(values) if values else 0
    
    # Aggregate energy metrics: sum counts where appropriate and average others.
    energy_keys_avg = ["cpu_power", "gpu_power", "ram_power", "energy_efficiency_tokens_per_joule"]
    energy_keys_sum = ["cpu_energy", "gpu_energy", "ram_energy", "total_energy_kwh", "total_energy_joules"]
    agg_energy = {}
    for key in energy_keys_avg:
        values = [proc["experiment_results"]["energy_performance"].get(key, 0) for proc in all_results if "experiment_results" in proc]
        agg_energy[key] = sum(values) / len(values) if values else 0
    for key in energy_keys_sum:
        agg_energy[key] = sum(proc["experiment_results"]["energy_performance"].get(key, 0) for proc in all_results if "experiment_results" in proc)
    # Include final emissions as a list.
    agg_energy["final_emissions"] = [proc["experiment_results"]["energy_performance"].get("final_emissions") for proc in all_results if "experiment_results" in proc]
    
    # Aggregate compute metrics (for numeric values, take averages).
    compute_keys = ["FLOPs", "cpu_usage_percent", "gpu_utilization_percent", "current_memory_allocated_bytes", "max_memory_allocated_bytes",
                    "current_memory_reserved_bytes", "max_memory_reserved_bytes"]
    agg_compute = {}
    for key in compute_keys:
        values = []
        for proc in all_results:
            val = proc["experiment_results"]["compute_performance"].get(key)
            if isinstance(val, list):
                values.extend(val)
            elif isinstance(val, (int, float)):
                values.append(val)
        agg_compute[key] = sum(values) / len(values) if values else 0
    
    # Optionally, include any task-specific performance metrics from the first process.
    task_perf = all_results[0]["experiment_results"].get("task-specific_performance", {})

    aggregated["experiment_results"] = {
        "inference_performance": agg_inference,
        "energy_performance": agg_energy,
        "compute_performance": agg_compute,
        "task_specific_performance": task_perf,
    }

    return make_json_serializable(aggregated)
