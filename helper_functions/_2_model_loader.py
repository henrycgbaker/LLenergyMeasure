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



def load_model_tokenizer(model_name: str, backend, fp_precision: str = "float32", quantization_config=None):
    """
    Loads a model and tokenizer from Hugging Face, setting the model's precision according to fp_precision.
    
    If quantisation is enabled (i.e. quantisation is truthy), then a BitsAndBytesConfig is used via the 
    quantization_config argument.
    
    Parameters:
        model_name: The model identifier.
        fp_precision: Desired precision ("float8", "float16", "bfloat16", or "float32").
        quantisation: A flag (or value) indicating whether to quantize the model.
        quantization_config: (Optional) A dictionary with parameters to pass to BitsAndBytesConfig.
        
    Returns:
        A tuple (model, tokenizer).
    """
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
    

    if quantization_config and quantization_config.get("quantization", False):
        # If a quantization_config dictionary is provided, build a BitsAndBytesConfig from it.
        if quantization_config is not None:
            bnb_config = BitsAndBytesConfig(**quantization_config)
        else:
            # Provide default values if none are specified.
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=False,
                llm_int8_has_fp16_weight=False,
            )
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


