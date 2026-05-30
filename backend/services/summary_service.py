from ..models.model_client_factory import ModelClientFactory

class SummaryService:

    def summarize(self, markdown_content, model_id=None):
        client = ModelClientFactory.get_client(model_id)
        return client.generate_summary(markdown_content)