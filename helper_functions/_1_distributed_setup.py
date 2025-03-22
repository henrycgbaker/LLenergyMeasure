import os
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.distributed as dist
    
def get_accelerator(gpu_list=None):
    """
    Sets up a distributed environment using Accelerate 
    """
    # Set CUDA_VISIBLE_DEVICES if a gpu_list is provided.
    if gpu_list is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_list)
    
    accelerator = Accelerator(device_placement=True)
    
    return accelerator

def get_persistent_unique_id():
    """
    Retrieves a persistent unique ID from a local file and increments it.
    """
    ID_FILE = "experiment_id.txt"
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


def load_model_tokenizer(model_name: str, backend, fp_precision: str = "float32"):
    """
    Loads a model and tokenizer from Hugging Face, setting the model's precision according to fp_precision.
    
    Parameters:
        model_name: The model identifier.
        fp_precision: Desired precision ("float8", "float16", "bfloat16", or "float32").
        
    Returns:
        A tuple (model, tokenizer).
    """
    
    # ADD BACKEND
    
    # chose precision    
    if fp_precision == "float8":
        dtype = torch.float8
    elif fp_precision == "float16":
        dtype = torch.float16
    elif fp_precision == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32  
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    
    return model, tokenizer

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

class ModelWrapper(torch.nn.Module):
    """
    Models loaded from Hugging Face transformers lib, not always in a standard nn.Module format.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, input_ids):
        return self.model(input_ids=input_ids)
    