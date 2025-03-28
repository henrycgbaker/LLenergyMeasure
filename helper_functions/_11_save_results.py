import os
import json
import torch

def save_raw_results(experiment_id, type, results, pid=None):
    output_dir = os.path.join(os.getcwd(), f"results/raw_results/{experiment_id}")    
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
    
    
def save_final_results(task_type, benchmark_results):
    """
    Saves benchmark results as a JSON log.
    
    Parameters:
      - task_type: A string representing the type of task.
      - benchmark_results: A dictionary of results to save.
      
    Returns:
      The file path where the results were saved.
    """
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
