import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLAMA_MODEL:
    def __init__(self, model_name="meta-llama/Llama-3.2-1B-Instruct", device="cuda", attn_implementation="eager", dtype=torch.bfloat16):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.lm, self.tokenizer = self._load_model()
        self.lm.eval()
        if getattr(self.lm, "hf_device_map", None) is None:
            self.lm.to(device=self.device, dtype=self.dtype)

    def _load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="left",
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model_kwargs = {
            "attn_implementation": self.attn_implementation,
            "torch_dtype": self.dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"
        lm = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        )
        return lm, tokenizer

    def _format_chat_history(self, sys_prompt, chat_history: list, recommend_item: str):
        chat_data = [{"role": "system", "content": sys_prompt}]
        chat_data += chat_history
        chat_data += [{"role": "assistant", "content": recommend_item}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            kwargs = {"tokenize": False, "add_generation_prompt": True}
            if "qwen3" in self.model_name.lower():
                kwargs["enable_thinking"] = False
            return self.tokenizer.apply_chat_template(chat_data, **kwargs)

        lines = []
        for message in chat_data:
            lines.append(f"{message['role']}: {message['content']}")
        lines.append("assistant:")
        return "\n".join(lines)

    def _postprocess_generated_text(self, text: str) -> str:
        text = re.sub(r"<think>\s*.*?\s*</think>\s*", "", text, flags=re.DOTALL)
        return text.strip()

    def response_generation(self, sys_prompt: str, chat_history: list, recommend_item: str,max_new_tokens=512, response_format=None):
        chat_history = self._format_chat_history(sys_prompt, chat_history, recommend_item)
        token_inputs = self.tokenizer(chat_history, return_tensors="pt")
        target_device = self.lm.device
        input_ids = token_inputs.input_ids.to(target_device)
        attention_mask = token_inputs.attention_mask.to(target_device)
        with torch.no_grad():
            outputs = self.lm.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False,
            )
        generated_text = self.tokenizer.batch_decode(outputs[:,input_ids.shape[1]:], skip_special_tokens=True)[0]
        generated_text = self._postprocess_generated_text(generated_text)
        return generated_text

    def batch_response_generation(self, sys_prompts: list[str], chat_histories: list[list], recommend_items: list[str], max_new_tokens=64):
        """Generate responses for multiple inputs in batch.

        Args:
            sys_prompts: List of system prompts.
            chat_histories: List of chat history lists.
            recommend_items: List of recommended items.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            List of generated response texts.
        """
        # Format all chat histories
        formatted_chats = [
            self._format_chat_history(sys_prompt, chat_history, recommend_item)
            for sys_prompt, chat_history, recommend_item in zip(sys_prompts, chat_histories, recommend_items)
        ]

        # Tokenize with padding
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        token_inputs = self.tokenizer(formatted_chats, return_tensors="pt", padding=True, truncation=True)
        target_device = self.lm.device
        input_ids = token_inputs.input_ids.to(target_device)
        attention_mask = token_inputs.attention_mask.to(target_device)

        with torch.no_grad():
            outputs = self.lm.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False,
            )

        # Decode only the newly generated tokens
        generated_texts = self.tokenizer.batch_decode(outputs[:,input_ids.shape[1]:], skip_special_tokens=True)
        return [self._postprocess_generated_text(text) for text in generated_texts]
