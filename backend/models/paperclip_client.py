from .base import ModelClient

class PaperClipClient(ModelClient):

    def generate_text(self, prompt, **kwargs):
        return "paperclip output"

    def generate_summary(self, text, **kwargs):
        return self.generate_text(f"Summarize:\n{text}")

    def generate_chat_response(self, query, paper_id, **kwargs):
        return self.generate_text(query)