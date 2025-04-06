import os
import json
import torch
import csv
import logging

logger = logging.getLogger(__name__)


def save_raw_results_json(experiment_id, type, results, pid=None):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "results", "raw_results", str(experiment_id))
    os.makedirs(output_dir, exist_ok=True)
    
    if pid is not None:
        output_json_path = os.path.join(output_dir, f"{experiment_id}_{type}_#{pid}.json")  

    else:
        output_json_path = os.path.join(output_dir, f"{experiment_id}_{type}.json") 

    with open(output_json_path, "w") as json_file:
        json.dump(results, json_file, indent=2) 

    return output_json_path


def make_json_serializable(obj):
    if isinstance(obj, dict):
        # Ensure keys are strings and process values recursively.
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, torch.dtype):
        return str(obj)
    else:
        return obj
  
    
def save_final_results_json(task_type, benchmark_results):
    
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
    
    #make serialisable
    serializable_results = make_json_serializable(existing_data)
    
    with open(output_json_path, "w") as json_file:
        json.dump(serializable_results, json_file, indent=4, default=str)
    
    return output_json_path

# FOR JSON -> CSV _____________

def get_ordered_keys_from_json(nested, prefix=""):
    """
    Performs a depth-first search (DFS) on the nested JSON to return a list of 
    flattened keys in the order they appear. This ensures that keys that belong
    together in the hierarchy are grouped together.

    Parameters:
        nested (dict or list): The nested JSON.
        prefix (str): The prefix built up from parent keys.

    Returns:
        list: A list of flattened keys in DFS order.
    """
    keys = []
    if isinstance(nested, dict):
        for key, value in nested.items():
            # Build the new prefix for the current key.
            new_prefix = f"{prefix}_{key}" if prefix else key
            # If the value is a dict, dive deeper.
            if isinstance(value, dict):
                keys.extend(get_ordered_keys_from_json(value, new_prefix))
            # If the value is a list, iterate over elements.
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    # For each element, construct a prefix with the index.
                    item_prefix = f"{new_prefix}_{i}"
                    # If the element is a dict, dive deeper.
                    if isinstance(item, dict):
                        keys.extend(get_ordered_keys_from_json(item, item_prefix))
                    else:
                        keys.append(item_prefix)
            else:
                # If it is a leaf value, add the key.
                keys.append(new_prefix)
    elif isinstance(nested, list):
        # If the top-level object is a list.
        for i, item in enumerate(nested):
            keys.extend(get_ordered_keys_from_json(item, f"{prefix}_{i}" if prefix else str(i)))
    return keys

def enforce_column_order(union_keys, original_json):
    """
    Orders the given union_keys based on a DFS traversal of the original JSON.
    Keys found in the original JSON will be ordered according to their DFS order,
    and any extra keys in union_keys will be appended at the end.

    Parameters:
        union_keys (set): The set of all keys that appear in the flattened rows.
        original_json (dict): A representative nested JSON to determine the desired order.

    Returns:
        list: A list of keys ordered according to the JSON structure.
    """
    # Get the ideal order from the JSON structure.
    ideal_order = get_ordered_keys_from_json(original_json)
    # Keep only those keys that are in the union.
    ordered = [key for key in ideal_order if key in union_keys]
    # Append any remaining keys that weren't in the DFS order.
    remaining = [key for key in union_keys if key not in ordered]
    return ordered + remaining



def flatten_dict(d, prefix=""):
    """
    Recursively flattens a nested dictionary.
    Every level’s key is included in the final key names, concatenated by underscores.
    
    Parameters:
        d (dict): The dictionary to flatten.
        prefix (str): The prefix built up from parent keys.
        
    Returns:
        dict: A flattened dictionary with keys reflecting the full hierarchy.
    """
    items = {}
    for key, value in d.items():
        new_prefix = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            items.update(flatten_dict(value, prefix=new_prefix))
        elif isinstance(value, list):
            # Optionally: If you have lists, you can flatten them too.
            for i, elem in enumerate(value):
                if isinstance(elem, dict):
                    items.update(flatten_dict(elem, prefix=f"{new_prefix}_{i}"))
                else:
                    items[f"{new_prefix}_{i}"] = elem
        else:
            items[new_prefix] = value
    return items


def flatten_configuration_run_json(run_json):

    # Drop the top-level key. (Assumes there is only one key like "CONFIGURATION_RUN_#123".)
    config_run_data = next(iter(run_json.values()))
    flattened = {}
    
    # Process all second-level keys except "results" first.
    for key, value in config_run_data.items():
        if key != "results":
            # We drop this level’s key (e.g. "setup", "variables", "model_architecture")
            # and flatten their contents so that the keys from their inner dictionaries become columns.
            flattened.update(flatten_dict({key: value}))
    
    # Process "results" separately.
    if "results" in config_run_data and "local_energy_results" in config_run_data["results"]:
        local_energy = config_run_data["results"]["local_energy_results"]
        for proc_key, proc_data in local_energy.items():
            proc_flat = flatten_dict({proc_key: proc_data})
            flattened.update(proc_flat)
    
    # Handle local energy results separately.
    # We expect local_energy_results to be a dict with keys like "process_0", "process_1", etc.
    if "results" in config_run_data and "local_energy_results" in config_run_data["results"]:
        local_energy = config_run_data["results"]["local_energy_results"]
        # For each expected process (max 4), flatten if present; otherwise add a fixed column with "NA".
        for i in range(4):
            process_key = f"process_{i}"
            if process_key in local_energy:
                # Instead of flattening all the nested metrics, you may choose to keep them as one JSON string,
                # or—as shown here—flatten them so that each nested metric becomes its own column.
                proc_flat = flatten_dict({process_key: local_energy[process_key]})
                flattened.update(proc_flat)
            else:
                # If the process did not run, add a placeholder column.
                # (If multiple energy metrics are expected per process, you could add one combined column.)
                flattened[f"{process_key}_energy"] = "NA"
    
    # (Optionally, you could enforce a fixed set of columns here if needed.)
    return flattened

def save_final_results_tabular(task_type, new_row, ordering_json=None):
    """
    Reads any existing rows from the CSV file for this task_type,
    appends the new row, computes the union of all keys, normalizes each row,
    and then rewrites the CSV file with a consistent header.

    If ordering_json is provided, the CSV header order will reflect the JSON structure.

    Parameters:
        task_type (str): A string indicating the task type, used to name the file.
        new_row (dict): A dictionary representing a new row of tabular data.
        ordering_json (dict): A representative nested JSON to determine the desired column order.

    Returns:
        str: The absolute path to the CSV file where the rows were saved.
    """
    import csv
    import os

    # Ensure the results directory exists.
    results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Construct the full path to the CSV file.
    filename = f"{task_type}_results.csv"
    file_path = os.path.join(results_dir, filename)
    
    # Read existing rows if the file exists.
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
    
    # Determine column order.
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


# then the wrapper function to call in the experiment runner:

def save_configuration_run_results_tabular(self):
    logger.info("Saving configuration run tabular results")
    configuration_run_id = self.configuration_run_id
    configuration_run_title = f"CONFIGURATION_RUN_#{configuration_run_id}"
    
    # First, get the nested JSON structure.
    run_results_json = self.save_configuration_run_results_json()
    
    # Flatten the nested JSON into a single dictionary.
    flattened = flatten_configuration_run_json(run_results_json)
    
    # (Optional) Enforce fixed columns for local energy results if desired.
    for i in range(4):
        prefix = f"process_{i}_"
        if not any(key.startswith(prefix) for key in flattened.keys()):
            flattened[f"process_{i}_energy"] = "NA"
    
    # Save the normalized flattened row to CSV.
    # Use run_results_json as the representative JSON for ordering.
    output_tabular_path = save_final_results_tabular(self.config.task_type, flattened, ordering_json=run_results_json)
    logger.info(f"Configuration run tabular results saved to {output_tabular_path}")
    
    return flattened

