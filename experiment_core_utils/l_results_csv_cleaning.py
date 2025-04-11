# NB THIS HAS DUPLICATE FUNCTIONS (FROM JSON-> CSV) FROM METRICS_SAVING... 
# TO DO: HAVE ONLY IN METRICS_SAVING AND IMPORT FROM THERE
# NB THE EXPERIMENT_RUNNER ALREADY SAVES AND UPDATES CSV FILES AS IT GOES... 
# THIS IS THE STATIC VERSION THAT CONVERTS FINAL JSON TO A CSV 
# (REASON: ISSUES WITH THE AD-HOC CSV UPDATING PROCESS... BUT WHEN REFACTORING, REMOVE/SEPARATE THIS WORKFLOW,
# TO REMOVE: THE CSV TO JSON PARTS (JUST HAVE IN EXPERIMENT WORKFLOW)
# TO KEEP: THE CSV CLEANING AND REORDERING (CALL IN THE ANALYSIS)

#!/usr/bin/env python3
import os
import json
import csv
import re
import logging
import pandas as pd


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Helpers for flattening and ordering nested JSON structures
# -------------------------------------------------------------------

def flatten_dict(d, prefix=""):
    """
    Recursively flattens a nested dictionary.
    Each nested level’s key is appended using underscores.
    """
    items = {}
    for key, value in d.items():
        new_prefix = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            items.update(flatten_dict(value, prefix=new_prefix))
        elif isinstance(value, list):
            # Iterate over list elements.
            for i, elem in enumerate(value):
                if isinstance(elem, dict):
                    items.update(flatten_dict(elem, prefix=f"{new_prefix}_{i}"))
                else:
                    items[f"{new_prefix}_{i}"] = elem
        else:
            items[new_prefix] = value
    return items

def get_ordered_keys_from_json(nested, prefix=""):
    """
    Performs a depth-first search (DFS) on nested JSON.
    Returns a list of flattened keys (with concatenated prefix) in the order encountered.
    """
    keys = []
    if isinstance(nested, dict):
        for key, value in nested.items():
            new_prefix = f"{prefix}_{key}" if prefix else key
            if isinstance(value, dict):
                keys.extend(get_ordered_keys_from_json(value, new_prefix))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    item_prefix = f"{new_prefix}_{i}"
                    if isinstance(item, dict):
                        keys.extend(get_ordered_keys_from_json(item, item_prefix))
                    else:
                        keys.append(item_prefix)
            else:
                keys.append(new_prefix)
    elif isinstance(nested, list):
        for i, item in enumerate(nested):
            keys.extend(get_ordered_keys_from_json(item, f"{prefix}_{i}" if prefix else str(i)))
    return keys

def enforce_column_order(union_keys, ordering_json):
    """
    Orders the union of keys based on a DFS traversal of the provided ordering_json.
    Keys encountered in the ordering_json appear first (in DFS order);
    any extra keys are appended at the end.
    """
    ideal_order = get_ordered_keys_from_json(ordering_json)
    ordered = [key for key in ideal_order if key in union_keys]
    remaining = [key for key in union_keys if key not in ordered]
    return ordered + remaining

# -------------------------------------------------------------------
# Cleaning column names (removing unwanted prefixes)
# -------------------------------------------------------------------

def clean_column(col: str) -> str:
    """
    Cleans a column name by removing messy prefixes.
    For per-process metrics, uses regex to match keys (e.g. "gpu_energy_process_0").
    Otherwise, searches for known tokens (which you can update as needed) and returns the
    substring starting at the token.
    """
    # Remove extra whitespace
    col = col.strip()

    # Try to match per-process metrics first.
    per_process_patterns = [
        r'(cpu_power_process_\d+)',
        r'(gpu_power_process_\d+)',
        r'(ram_power_process_\d+)',
        r'(cpu_energy_process_\d+)',
        r'(gpu_energy_process_\d+)',
        r'(ram_energy_process_\d+)',
        r'(total_energy_kwh_process_\d+)',
        r'(total_energy_joules_process_\d+)'
    ]
    for pattern in per_process_patterns:
        match = re.search(pattern, col)
        if match:
            return match.group(1)
    
    # Tokens to search for in non-process metrics.
    tokens = [ 
        "config_name", "experiment_id", "date_time", "model", "is_encoder_decoder",
        "task_type", "available_gpu_count", "gpu_model", "available_cpu_count", "cpu_model",
        "os", "python_version", "country", "region", "fsdp_use_orig_params", "fsdp_cpu_offload",
        "sharding_strategy", "distributed_type", "num_processes", "max_input_tokens", "max_output_tokens",
        "number_input_prompts", "decode_token_to_text", "decoder_temperature", "decoder_top_k", "decoder_top_p",
        "query_rate", "latency_simulate", "latency_delay_min", "latency_delay_max", "latency_simulate_burst",
        "latency_burst_interval", "latency_burst_size", "fp_precision", "quantization", "load_in_8bit",
        "load_in_4bit", "cached_flops_for_quantised_models", "batch_size___fixed_batching", "adaptive_batching",
        "adaptive_max_tokens", "max_batch_size___adaptive_batching", "inference_type", "backend", "total_params",
        "architecture", "total_input_tokens", "total_generated_tokens", "total_inference_time_sec", 
        "average_latency_ms_per_batch", "throughput_queries_per_sec", "throughput_tokens_per_sec", "flops",
        "gpu_current_memory_allocated_bytes", "gpu_max_memory_allocated_bytes", "gpu_current_memory_reserved_bytes",
        "gpu_max_memory_reserved_bytes", "gpu_utilization_percent", "cpu_usage_percent", "cpu_memory_usage_bytes",
        "cpu_power_avg", "gpu_power_avg", "ram_power_avg", "cpu_energy_total", "gpu_energy_total", "ram_energy_total",
        "total_energy_kwh", "total_energy_joules", "tokens_per_joule", "joules_per_token", "flops_per_joule", "joules_per_flop",
        "per-process_emissions"
    ]
    
    for token in tokens:
        if token in col:
            idx = col.find(token)
            return col[idx:]
    
    return col

def clean_columns_in_row(row: dict) -> dict:
    """
    Given a dictionary representing a flattened row, returns a new dict with all keys
    cleaned (i.e. removing messy prefixes). Assumes cleaned keys remain unique.
    """
    new_row = {}
    for key, value in row.items():
        new_key = clean_column(key)
        new_row[new_key] = value
    return new_row

# -------------------------------------------------------------------
# Flattening a single experiment’s JSON structure
# -------------------------------------------------------------------

def flatten_configuration_run_json(run_json):
    """
    Given a nested JSON for a single experiment, flattens it into a single dictionary.
    Assumes the top-level object has a single key (e.g. "CONFIGURATION_RUN_#<id>").
    Processes everything—including local energy results (which may vary in number).
    """
    # Get the single top-level key’s value.
    config_run_data = next(iter(run_json.values()))
    flattened = {}

    # Flatten keys outside of "results".
    for key, value in config_run_data.items():
        if key != "results":
            flattened.update(flatten_dict({key: value}))

    # In "results", process keys other than local_energy_results.
    if "results" in config_run_data:
        results_section = config_run_data["results"]
        for subkey, subvalue in results_section.items():
            if subkey != "local_energy_results":
                flattened.update(flatten_dict({subkey: subvalue}))

    # Process local_energy_results dynamically.
    if "results" in config_run_data and "local_energy_results" in config_run_data["results"]:
        local_energy = config_run_data["results"]["local_energy_results"]
        for proc_key, proc_value in local_energy.items():
            # Flatten this process’ block.
            proc_flat = flatten_dict({proc_key: proc_value})
            flattened.update(proc_flat)
    return flattened

# -------------------------------------------------------------------
# Main processing: Convert the JSON file to a CSV file
# -------------------------------------------------------------------

def process_json_to_csv(input_json_file, output_csv_file):
    """
    Reads the given JSON file (assumed to be a list of experiments),
    flattens each experiment into a single dictionary row, checks for duplicates
    (using experiment_id as a primary key), and writes out a CSV file with the union
    of keys (missing values are filled with "NA") and cleaned/ordered column names.
    """
    # Load the complete JSON.
    with open(input_json_file, "r") as f:
        try:
            experiments = json.load(f)
        except json.JSONDecodeError as e:
            logger.error("Failed to decode JSON: %s", e)
            return

    flattened_rows = []
    seen_experiment_ids = set()
    
    # Process each experiment.
    for exp in experiments:
        flat_row = flatten_configuration_run_json(exp)
        # Clean the keys in the flattened row.
        flat_row = clean_columns_in_row(flat_row)
        # Use experiment_id as the primary key; skip duplicates.
        exp_id = flat_row.get("experiment_id")
        if not exp_id:
            logger.warning("Skipping a row due to missing experiment_id: %s", flat_row)
            continue
        if exp_id in seen_experiment_ids:
            logger.info("Duplicate found for experiment_id %s; skipping...", exp_id)
            continue
        seen_experiment_ids.add(exp_id)
        flattened_rows.append(flat_row)

    if not flattened_rows:
        logger.error("No experiments to process after flattening.")
        return

    # Compute the union of keys across all rows.
    all_keys = set()
    for row in flattened_rows:
        all_keys.update(row.keys())

    # Normalize each row: add missing keys with "NA".
    for row in flattened_rows:
        for key in all_keys:
            if key not in row:
                row[key] = "NA"

    # Determine column order.
    # Use the first experiment’s original JSON as the ordering guide.
    ordering_json = experiments[0]
    ordered_keys = enforce_column_order(all_keys, ordering_json)

    # Write the CSV file.
    with open(output_csv_file, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=ordered_keys)
        writer.writeheader()
        for row in flattened_rows:
            writer.writerow(row)

    logger.info("CSV file written to: %s", os.path.abspath(output_csv_file))