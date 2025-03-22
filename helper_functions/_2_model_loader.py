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


def load_model_tokenizer(model_name: str, backend, fp_precision: str = "float32", device=None):
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
    
    # assign to correct device
    if device is not None:
        model.to(device)
    
    model.eval()
    
    return model, tokenizer
