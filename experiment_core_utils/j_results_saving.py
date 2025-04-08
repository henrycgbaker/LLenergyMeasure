import os
import json
import torch
import csv
import logging

logger = logging.getLogger(__name__)

# -----------------------
# Basic JSON I/O Helpers
# -----------------------

def save_raw_results_json(experiment_id, type, results, pid=None):
    """Save raw results to a JSON file."""
    output_dir = os.path.join(os.getcwd(), f"results/raw_results/{experiment_id}")
    os.makedirs(output_dir, exist_ok=True)
    
    if pid is not None:
        output_json_path = os.path.join(output_dir, f"{experiment_id}_{type}_{pid}.json")
    else:
        output_json_path = os.path.join(output_dir, f"{experiment_id}_{type}.json")
    
    with open(output_json_path, "w") as json_file:
        json.dump(results, json_file, indent=2)
    
    return output_json_path

def make_json_serializable(obj):
    """Recursively converts objects to JSON-serializable objects."""
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, torch.dtype):
        return str(obj)
    else:
        return obj

def save_final_results_json(task_type, benchmark_results):
    """Append benchmark results to a JSON file."""
    output_dir = "results"
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
    serializable_results = make_json_serializable(existing_data)
    
    with open(output_json_path, "w") as json_file:
        json.dump(serializable_results, json_file, indent=4, default=str)
    
    return output_json_path

# -----------------------------
# Functions for JSON-to-CSV
# -----------------------------

def get_ordered_keys_from_json(nested, prefix=""):
    """
    Performs a depth-first search (DFS) on the nested JSON and returns a list of
    flattened keys in the order they appear.
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

def enforce_column_order(union_keys, original_json):
    """
    Orders the given union_keys based on a DFS traversal of the original JSON.
    Any extra keys not in the DFS order are appended at the end.
    """
    ideal_order = get_ordered_keys_from_json(original_json)
    ordered = [key for key in ideal_order if key in union_keys]
    remaining = [key for key in union_keys if key not in ordered]
    return ordered + remaining

def flatten_dict(d, prefix=""):
    """
    Recursively flattens a nested dictionary. Keys at each level are concatenated
    with underscores.
    """
    items = {}
    for key, value in d.items():
        new_prefix = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            items.update(flatten_dict(value, prefix=new_prefix))
        elif isinstance(value, list):
            for i, elem in enumerate(value):
                if isinstance(elem, dict):
                    items.update(flatten_dict(elem, prefix=f"{new_prefix}_{i}"))
                else:
                    items[f"{new_prefix}_{i}"] = elem
        else:
            items[new_prefix] = value
    return items

def flatten_configuration_run_json(run_json):
    """
    Flattens the configuration run JSON into a single dictionary for tabular output.
    This version dynamically iterates over the keys in local_energy_results rather than
    assuming a fixed number (e.g., 4).
    """
    # Assume run_json is structured like: { "CONFIGURATION_RUN_#<id>": { ... } }
    config_run_data = next(iter(run_json.values()))
    flattened = {}

    # Process all keys outside "results"
    for key, value in config_run_data.items():
        if key != "results":
            flattened.update(flatten_dict({key: value}))

    # Process the "results" section, except for local_energy_results.
    if "results" in config_run_data:
        results_section = config_run_data["results"]
        for subkey, subvalue in results_section.items():
            if subkey != "local_energy_results":
                flattened.update(flatten_dict({subkey: subvalue}))

    # Process local energy results dynamically.
    if "results" in config_run_data and "local_energy_results" in config_run_data["results"]:
        local_energy = config_run_data["results"]["local_energy_results"]
        for proc_key, proc_value in local_energy.items():
            proc_flat = flatten_dict({proc_key: proc_value})
            flattened.update(proc_flat)
    
    return flattened

def save_final_results_tabular(task_type, new_row, ordering_json=None, experiment_suite=None):
    """
    Reads any existing rows from the CSV file for a given task type, appends the new row,
    computes the union of all keys, normalizes each row so that every key is present, and
    then rewrites the CSV file with a consistent header. This ensures that—even if some
    experiments did not run all processes—the expected process columns appear with "NA".
    """
    results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    filename = f"{experiment_suite}_results.csv" if experiment_suite else f"{task_type}_results.csv"
    file_path = os.path.join(results_dir, filename)
    
    # Read existing rows.
    rows = []
    if os.path.isfile(file_path):
        with open(file_path, mode="r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for r in reader:
                rows.append(r)
    
    # Append the new row.
    rows.append(new_row)
    
    # Compute the union of keys from all rows.
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    
    # Normalize each row so that every key is present.
    for row in rows:
        for key in all_keys:
            if key not in row:
                row[key] = "NA"
    
    # Determine the column order based on ordering_json, if provided.
    if ordering_json is not None:
        ordered_keys = enforce_column_order(all_keys, ordering_json)
    else:
        ordered_keys = list(all_keys)
    
    # Write out the new CSV with the ordered keys as header.
    with open(file_path, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=ordered_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    
    return os.path.abspath(file_path)

# -----------------------------------------
# Wrapper Function Called in Experiment Run
# -----------------------------------------

def save_configuration_run_results_tabular(self):
    """
    This wrapper function is called during the experiment run.
    It:
      - Retrieves the nested JSON run results.
      - Flattens the JSON structure into a single-row dictionary.
      - (Optionally) Enforces expected process columns. Here, we explicitly add keys for
        process_0 through process_3 if they are missing.
      - Saves (or updates) the CSV file with the union of keys from all experiments.
    """
    logger.info("Saving configuration run tabular results")
    configuration_run_id = self.configuration_run_id
    configuration_run_title = f"CONFIGURATION_RUN_#{configuration_run_id}"
    
    # Get the nested JSON structure.
    run_results_json = self.save_configuration_run_results_json()
    
    # Flatten the configuration run JSON.
    flattened_row = flatten_configuration_run_json(run_results_json)
    
    # Optional: Explicitly enforce expected process columns (e.g., process_0 to process_3).
    expected_processes = ["process_0", "process_1", "process_2", "process_3"]
    # Here we assume that each process block should include some energy reading (e.g., "process_i_energy").
    # If no key starting with the expected prefix is found, add a placeholder.
    for proc in expected_processes:
        if not any(key.startswith(f"{proc}_") for key in flattened_row):
            flattened_row[f"{proc}_energy"] = "NA"
    
    # Save the normalized flattened row to CSV.
    experiment_suite = self.config.get("suite", self.config.task_type)
    output_tabular_path = save_final_results_tabular(
        self.config.task_type,
        flattened_row,
        ordering_json=run_results_json,
        experiment_suite=experiment_suite
    )
    logger.info(f"Configuration run tabular results saved to {output_tabular_path}")
    
    return flattened_row
