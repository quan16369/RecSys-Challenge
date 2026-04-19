from .llama import LLAMA_MODEL
from .template import TEMPLATE_MODEL

def load_lm_module(lm_type, device, attn_implementation, dtype):
    if lm_type == "template":
        return TEMPLATE_MODEL(model_name=lm_type, device=device, attn_implementation=attn_implementation, dtype=dtype)
    return LLAMA_MODEL(model_name=lm_type, device=device, attn_implementation=attn_implementation, dtype=dtype)
