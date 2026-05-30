from .openai_client import OpenAIClient
from .huggingface_client import HuggingFaceClient
from .paperclip_client import PaperClipClient
from ..models_config import get_model_config

class ModelClientFactory:

    @staticmethod
    def get_client(model_id=None):
        config = get_model_config(model_id)

        provider = config["provider"]

        if provider == "openai":
            return OpenAIClient(config)
        elif provider == "huggingface":
            return HuggingFaceClient(config)
        elif provider == "paperclip":
            return PaperClipClient(config)

        raise ValueError(f"Unknown provider: {provider}")