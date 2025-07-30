import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Tuple
import logging
import os
import json
from tqdm import tqdm

HF_token = "YOUR_HUGGING_FACE_TOKEN_HERE"

logger = logging.getLogger(__name__)


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Tuple
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_model_multiGPU(
    model_name: str,
    local_rank: int,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    
    torch.cuda.set_device(local_rank)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_auth_token=HF_token,
        truncation=True,
        return_tensors='pt',
        trust_remote_code=True,
    )

    model_kwargs = {
        "use_auth_token": HF_token,
        "torch_dtype": torch.float16,
        "torch_dtype": "auto",                    # fp16 automatically
        "device_map": {"": local_rank},           # 1-GPU *per rank*
        "low_cpu_mem_usage": True,
        "trust_remote_code":True
    }

    if load_in_8bit:
        model_kwargs["load_in_8bit"] = True
    elif load_in_4bit:
        model_kwargs["load_in_4bit"] = True
        
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    return model, tokenizer



def load_model_singleGPU(
    model_name: str,
    local_rank: int,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    
    torch.cuda.set_device(local_rank)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_auth_token=HF_token,
        truncation=True,
        return_tensors='pt',
        trust_remote_code=True,
    )

    model_kwargs = {
        "use_auth_token": HF_token,
        "torch_dtype": torch.float16,
        "torch_dtype": "auto",                    # fp16 automatically
        "device_map": "auto",           # 1-GPU *per rank*
        "low_cpu_mem_usage": True,
        "trust_remote_code":True
    }

    if load_in_8bit:
        model_kwargs["load_in_8bit"] = True
    elif load_in_4bit:
        model_kwargs["load_in_4bit"] = True
        
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    return model, tokenizer


def load_model(
    model_name: str,
    device: Optional[str] = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Loading model: {model_name} on {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_auth_token=HF_token,
        truncation=True,
        return_tensors='pt',
        trust_remote_code=True
    )

    model_kwargs = {
        "use_auth_token": HF_token,
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32
    }

    if load_in_8bit:
        model_kwargs["load_in_8bit"] = True
    elif load_in_4bit:
        model_kwargs["load_in_4bit"] = True

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    try:
        model.to(device)
    except Exception as e:
        logger.error(f"Failed to load model {model_name} on {device}: {e}")
        print(f"Failed to load model {model_name} on {device}: {e}")
    
    return model, tokenizer

# # List of models that fit on a single H100 without quantization
# model_names = [
#     "meta-llama/Meta-Llama-3-8B-Instruct",
#     "mistralai/Mistral-7B-v0.1",
#     "google/gemma-7b",
#     "mistralai/Mixtral-8x7B-v0.1",
#     "01-ai/Yi-34B",
#     "deepseek-ai/deepseek-coder-33b-instruct",
#     "Qwen/Qwen-32B"
# ]

# # Example loop to load models (you can comment out others if just testing one at a time)
# for name in model_names:
#     try:
#         model, tokenizer = load_model(name)
#         logger.info(f"Successfully loaded: {name}")
#         # Optional: delete model after loading to free VRAM
#         del model
#         del tokenizer
#         torch.cuda.empty_cache()
#     except Exception as e:
#         logger.error(f"Failed to load {name}: {e}")




def load_llama2(
    model_name: str,
    device: Optional[str] = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load model and tokenizer
    
    Args:
        model_name: Name or path of the model
        device: Device to load model on
        load_in_8bit: Whether to load in 8-bit precision
        load_in_4bit: Whether to load in 4-bit precision
        
    Returns:
        Tuple of (model, tokenizer)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Loading model {model_name} on {device}")
    
    # Load tokenizer
    max_new_tokens = 1024
    tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                              use_auth_token=HF_token,
                                              truncation=True, return_tensors='pt')
    
    # Configure model loading
    model_kwargs = {
        "use_auth_token": HF_token,
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32
    }
    
    if load_in_8bit:
        model_kwargs["load_in_8bit"] = True
    elif load_in_4bit:
        model_kwargs["load_in_4bit"] = True
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.to(device)
    
    return model, tokenizer 



# ----- for vicuna

def load_vicuna(
    model_name: str = "lmsys/vicuna-13b-v1.5",
    device: Optional[str] = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load Vicuna 13B v1.5 and its tokenizer.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading Vicuna {model_name} on {device}")

    # Tokenizer (needs trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_auth_token=HF_token,
        trust_remote_code=True,
        truncation=True,
        return_tensors="pt"
    )

    # Model kwargs
    model_kwargs = {
        "use_auth_token": HF_token,
        "trust_remote_code": True,
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
    }
    if load_in_8bit:
        model_kwargs["load_in_8bit"] = True
    elif load_in_4bit:
        model_kwargs["load_in_4bit"] = True

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.to(device)

    # Vicuna doesn’t define pad_token_id by default
    model.config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer
