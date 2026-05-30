from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from .base import ModelClient


class PaperClipClient(ModelClient):

    def __init__(self, config):
        super().__init__(config)

        self.tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
        self.model = AutoModelForCausalLM.from_pretrained(config["model_id"])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # 🔥 critical fix for GPT2-style / causal LMs
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # -----------------------
    def generate_chat_response(self, prompt, **kwargs):

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True
        ).to(self.device)

        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=250,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_tokens = outputs[0][input_len:]

        text = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

        return text if text else "I couldn't generate a response."

    # -----------------------
    def generate_summary(self, text, **kwargs):

        prompt = f"""Summarize the following academic paper clearly and concisely:

{text}
"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True
        ).to(self.device)

        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=500,
                do_sample=True,
                temperature=0.3,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_tokens = outputs[0][input_len:]

        return self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()