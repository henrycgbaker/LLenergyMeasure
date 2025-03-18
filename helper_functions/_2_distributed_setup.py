import os
from accelerate import Accelerator

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

def distributed_env_torch(model, tokenizer, gpu_list=None):
    """
    Sets up a distributed environment using Accelerate and prepares the model and tokenizer.
    
    Parameters:
        model: The model to distribute.
        tokenizer: The associated tokenizer.
        gpu_list: Optional list of GPU IDs to set via CUDA_VISIBLE_DEVICES.
        
    Returns:
        A tuple (model, tokenizer, accelerator).
    """
    # Set CUDA_VISIBLE_DEVICES if a gpu_list is provided.
    if gpu_list is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_list)
    
    accelerator = Accelerator(device_placement=True)
    model, tokenizer = accelerator.prepare(model, tokenizer)
        
    return model, tokenizer, accelerator

