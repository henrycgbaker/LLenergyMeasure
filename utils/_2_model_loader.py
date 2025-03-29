from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import importlib
import warnings
from packaging import version

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


def detect_supported_quant_types():
    try:
        bnb = importlib.import_module("bitsandbytes")
        bnb_version = getattr(bnb, "__version__", "0.0.0")
    except ImportError:
        warnings.warn("bitsandbytes is not installed. Defaulting to no quantisation.")
        return {
            "supports_4bit": False,
            "supports_8bit": False,
            "default_4bit_quant_type": None,
            "default_8bit_quant_type": None
        }

    parsed_version = version.parse(bnb_version)

    supports_4bit = parsed_version >= version.parse("0.39.0")  # QLoRA-level support
    supports_8bit = parsed_version >= version.parse("0.38.0")

    quant_type_4bit = "nf4" if supports_4bit else None
    quant_type_8bit = "fp8" if supports_8bit else "int8"  # nf8 not officially supported

    return {
        "supports_4bit": supports_4bit,
        "supports_8bit": supports_8bit,
        "default_4bit_quant_type": quant_type_4bit,
        "default_8bit_quant_type": quant_type_8bit
    }


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
    
    qsupport = detect_supported_quant_types()
    
    if quant_config_dict and quant_config_dict.get("quantization", False):        
            # Prepare arguments for BitsAndBytesConfig
            bnb_kwargs = quant_config_dict.copy()
            
            if quant_config_dict.get("load_in_4bit", False) and qsupport["supports_4bit"]:
                bnb_kwargs["bnb_4bit_compute_dtype"] = torch.float16
                bnb_kwargs["bnb_4bit_quant_type"] = qsupport["default_4bit_quant_type"]

            if quant_config_dict.get("load_in_8bit", False) and qsupport["supports_8bit"]:
                bnb_kwargs["bnb_8bit_compute_dtype"] = torch.float16
                bnb_kwargs["bnb_8bit_quant_type"] = qsupport["default_8bit_quant_type"]
            
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


