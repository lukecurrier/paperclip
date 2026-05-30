from .base import ModelClient

class HuggingFaceClient(ModelClient):

    def generate_text(self, prompt, **kwargs):
        # keep your existing implementation here
        return "hf output"

    def generate_summary(self, text, **kwargs):
        prompt = f"Summarize:\n{text}"
        return self.generate_text(prompt, **kwargs)

    def generate_chat_response(self, query, paper_id, **kwargs):
        prompt = f"{query}\nAnswer based on paper context"
        return self.generate_text(prompt)