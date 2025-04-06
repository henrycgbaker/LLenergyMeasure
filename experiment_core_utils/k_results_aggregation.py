import os
import glob
import json

def load_local_energy_results(experiment_id):
    """
    Loads per-process energy results from JSON files in results/raw_results/<experiment_id>/.
    Expects filenames like: <experiment_id>_6_local_energy_results_process_<pid>.json
    Returns a dictionary keyed by process identifier, e.g., "process_0", "process_1", etc.
    """
    results = {}
    folder = os.path.join(os.getcwd(), "results", "raw_results", experiment_id)
    pattern = os.path.join(folder, f"{experiment_id}_6_local_energy_results_process_*.json")
    
    for filepath in glob.glob(pattern):
        basename = os.path.basename(filepath)
        try:
            # Extract the part after "process_" and before ".json"
            pid_part = basename.split("process_")[-1].split(".json")[0]
            key = f"process_{pid_part}"
        except Exception as e:
            continue
        with open(filepath, "r") as f:
            results[key] = json.load(f)
    return results
