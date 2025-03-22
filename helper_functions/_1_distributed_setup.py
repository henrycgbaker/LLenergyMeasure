import os
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class ModelWrapper(torch.nn.Module):
    """
    Models loaded from Hugging Face transformers lib, not always in a standard nn.Module format.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, input_ids):
        return self.model(input_ids=input_ids)
    

    
def get_accelerator(gpu_list=None):
    """
    Sets up a distributed environment using Accelerate 
    """
    # Set CUDA_VISIBLE_DEVICES if a gpu_list is provided.
    if gpu_list is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_list)
    
    accelerator = Accelerator(device_placement=True)
    
    return accelerator


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

