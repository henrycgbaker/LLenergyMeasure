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


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import BitsAndBytesConfig


class ModelWrapper(torch.nn.Module):
    """
    Models loaded from Hugging Face transformers lib, not always in a standard nn.Module format.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, input_ids):
        return self.model(input_ids=input_ids)


def load_model_tokenizer(configs):
    """
    Loads a model and tokenizer from Hugging Face, setting the model's precision according to fp_precision.
    """
    model_name = configs.model_name      
    fp_precision = configs.fp_precision
    backend = configs.backend
    quant_config_dict = configs.quantization_config
    
    # Choose precision.
    if fp_precision == "float8":
        dtype = torch.float8
    elif fp_precision == "float16":
        dtype = torch.float16
    elif fp_precision == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32  
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if quant_config_dict and quant_config_dict.get("quantization", False):        
            # Prepare arguments for BitsAndBytesConfig
            bnb_kwargs = quant_config_dict.copy()
            
            if quant_config_dict.get("load_in_4bit", False):
                bnb_kwargs["bnb_4bit_compute_dtype"] = torch.float16 
                bnb_kwargs["bnb_4bit_quant_type"] = "nf4"
            
            if quant_config_dict.get("load_in_8bit", False):
                bnb_kwargs["bnb_8bit_compute_dtype"] = torch.float16
                bnb_kwargs["bnb_8bit_quant_type"] = "nf8"
            
            bnb_config = BitsAndBytesConfig(**bnb_kwargs)
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
        )
    
    return model, tokenizer


