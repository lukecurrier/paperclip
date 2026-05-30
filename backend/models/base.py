class ModelClient:
    def __init__(self, config):
        self.config = config

    def generate_text(self, prompt, **kwargs):
        raise NotImplementedError

    def generate_summary(self, text, **kwargs):
        raise NotImplementedError

    def generate_chat_response(self, query, paper_id, **kwargs):
        raise NotImplementedError