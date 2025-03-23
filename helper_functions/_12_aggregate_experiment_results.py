import os
import glob
import json


def load_local_energy_results(experiment_id):
    """
    Loads per-process energy results from JSON files in results/raw_results/<experiment_id>/.
    Expects filenames like: <experiment_id>_6_local_energy_results_#<pid>.json
    Returns a dict keyed by process index.
    """
    results = {}
    folder = os.path.join(os.getcwd(), "results", "raw_results", experiment_id)
    pattern = os.path.join(folder, f"{experiment_id}_6_local_energy_results_#*.json")
    for filepath in glob.glob(pattern):
        basename = os.path.basename(filepath)
        try:
            pid_str = basename.split("_")[-1].split(".json")[0]  # yields "#1"
            pid = int(pid_str.lstrip("#"))  # remove '#' then convert to int (cleaner for downstream key-value in dict)
        except Exception as e:
            continue
        with open(filepath, "r") as f:
            results[pid] = json.load(f)
    return results


# DEGRADED: -----

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
    aggregated = {}
    
    if all_results[0].get("experiment_setup"):
        aggregated["experiment_setup"] = all_results[0]["experiment_setup"]
    if all_results[0].get("experiment_variables"):
        aggregated["experiment_variables"] = all_results[0]["experiment_variables"]
    
    # Inference: take process 0's values (they are constant)
    agg_inference = all_results[0]["experiment_results"]["inference_performance"]
    
    # Energy: average rate metrics and sum count metrics.
    energy_keys_avg = ["cpu_power", "gpu_power", "ram_power", "energy_efficiency_tokens_per_joule"]
    energy_keys_sum = ["cpu_energy", "gpu_energy", "ram_energy", "total_energy_kwh", "total_energy_joules"]
    agg_energy = {}
    for key in energy_keys_avg:
        values = [proc["experiment_results"]["energy_performance"].get(key, 0)
                  for proc in all_results if "experiment_results" in proc]
        agg_energy[key] = sum(values) / len(values) if values else 0
    for key in energy_keys_sum:
        agg_energy[key] = sum(proc["experiment_results"]["energy_performance"].get(key, 0)
                              for proc in all_results if "experiment_results" in proc)
    agg_energy["final_emissions"] = [proc["experiment_results"]["energy_performance"].get("final_emissions")
                                      for proc in all_results if "experiment_results" in proc]
    
    # Compute: use process 0's FLOPs and average the rest.
    agg_compute = {}
    flops_value = all_results[0]["experiment_results"]["compute_performance"].get("FLOPs", 0)
    agg_compute["FLOPs"] = flops_value
    compute_keys_avg = [
        "cpu_usage_percent", "gpu_utilization_percent", "current_memory_allocated_bytes",
        "max_memory_allocated_bytes", "current_memory_reserved_bytes", "max_memory_reserved_bytes"
    ]
    for key in compute_keys_avg:
        values = []
        for proc in all_results:
            val = proc["experiment_results"]["compute_performance"].get(key)
            if isinstance(val, list):
                values.extend(val)
            elif isinstance(val, (int, float)):
                values.append(val)
        agg_compute[key] = sum(values) / len(values) if values else 0

    task_perf = all_results[0]["experiment_results"].get("task_specific_performance", {})
    
    aggregated["experiment_results"] = {
        "inference_performance": agg_inference,
        "energy_performance": agg_energy,
        "compute_performance": agg_compute,
        "task_specific_performance": task_perf,
    }
    
    return make_json_serializable(aggregated)


