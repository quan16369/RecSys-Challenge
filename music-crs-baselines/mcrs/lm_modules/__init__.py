from .llama import LLAMA_MODEL
from .template import TEMPLATE_MODEL

def load_lm_module(lm_type, device, attn_implementation, dtype, max_input_tokens=1536, max_new_tokens=48):
    if lm_type == "template":
        return TEMPLATE_MODEL(model_name=lm_type, device=device, attn_implementation=attn_implementation, dtype=dtype)
    return LLAMA_MODEL(
        model_name=lm_type,
        device=device,
        attn_implementation=attn_implementation,
        dtype=dtype,
        max_input_tokens=max_input_tokens,
        max_new_tokens=max_new_tokens,
    )
