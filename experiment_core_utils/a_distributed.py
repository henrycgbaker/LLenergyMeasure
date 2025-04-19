import os
import yaml
from accelerate import Accelerator
import torch.distributed as dist
import threading
import sys
import torch

def delete_accelerator_cli_configs():
    """
    can delete this i think - originally included as necessary override accelerate launcher configs - but I sorted that out with a workaround
    """
    config_path = os.path.expanduser("~/.cache/huggingface/accelerate/default_config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        # Remove GPU-related settings if they exist
        config.pop("gpu_list", None)
        config.pop("num_processes", None)
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        print("Modified Accelerate CLI config to remove gpu_list and num_processes.")
    else:
        print("No Accelerate CLI config file found.")

def get_accelerator(gpu_list=None, num_processes=None):
    
    #delete_accelerator_cli_configs()
    
    os.environ["ACCELERATE_CONFIG_FILE"] = ""   # disables the CLI config so the script settings take priority
    
    if gpu_list is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_list)
        
        if num_processes is not None:
            available = len(gpu_list)
            if num_processes > available:
                print(f"Warning: num_processes ({num_processes}) exceeds available GPUs ({available}). Using {available} processes instead.")
                num_processes = available
            os.environ["ACCELERATE_NUM_PROCESSES"] = str(num_processes)
    
    from accelerate import Accelerator  # Import AFTER setting the environment variables
    accelerator = Accelerator(device_placement=True)
    return accelerator

def get_persistent_unique_id():
    """
    Retrieves a persistent unique ID from a local file and increments it.
    """
    ID_FILE = "persistent_progress_trackers/experiment_id.txt"
    if os.path.exists(ID_FILE):
        with open(ID_FILE, "r") as f:
            try:
                last_id = int(f.read().strip())
            except ValueError:
                last_id = 0
    else:
        last_id = 0
    new_id = last_id + 1
    with open(ID_FILE, "w") as f:
        f.write(str(new_id))
    return f"{new_id:04d}"

def get_shared_unique_id(accelerator):
    """
    Generate a unique ID on the main process and broadcast it to all workers.
    Uses torch.distributed.broadcast_object_list to ensure all processes
    get the same unique ID.
    """
    unique_id_list = [""]
    if accelerator.is_main_process:
        unique_id_list[0] = get_persistent_unique_id()  # Your function that returns a unique string.
    # Ensure the distributed group is initialized and all processes call this.
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(unique_id_list, src=0)
    return unique_id_list[0]


def get_original_generate_method(model):
    """
    Recursively searches for a callable 'generate' method within a model,
    checking through wrappers like DataParallel or FSDP.
    
    NB: this is needed BEFORE the model is wrapped up and distributed
    
    Returns:
      The original generate method if found, or None otherwise.
    """
    if hasattr(model, "generate") and callable(model.generate):
        return model.generate
    elif hasattr(model, "module"):
        return get_original_generate_method(model.module)
    else:
        return None
    

def check_failed_flag(accelerator):
    flag = torch.tensor([0], device=accelerator.device)
    # rank 0’s flag will be the “true” one if any rank broadcast 1
    dist.broadcast(flag, src=0)
    if flag.item() == 1:
        accelerator.print("Detected failure on another rank – exiting.")
        # clean up and exit
        torch.cuda.empty_cache()
        dist.destroy_process_group()
        sys.exit(1)

def safe_wait(accelerator, description="", timeout=600):
    accelerator.print(f"Entering wait barrier: {description}")
    
    # wrap the blocking call.
    def wait_func():
        try:
            accelerator.wait_for_everyone()
        except Exception as e:
            accelerator.print(f"Error during wait_for_everyone at {description}: {e}")
    
    # run the wait function in a daemon thread.
    t = threading.Thread(target=wait_func, daemon=True)
    t.start()
    t.join(timeout)
    
    if t.is_alive():
        accelerator.print(f"Timeout reached: wait_for_everyone did not finish within {timeout} seconds for {description}.")
    else:
        accelerator.print(f"wait_for_everyone completed within {timeout} seconds for {description}.")
    
    accelerator.print(f"Exiting wait barrier: {description}")