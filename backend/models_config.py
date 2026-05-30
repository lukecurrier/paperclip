MODEL_CONFIGS = {
    "gpt-4o-mini": {
        "name": "GPT-4o Mini",
        "description": "OpenAI-hosted large language model",
        "provider": "openai",
        "model_id": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
        "max_tokens": 750,
        "temperature": 0.4
    },
     "llama-3.2-1b": {
         "name": "Llama 3.2 1B",
         "description": "Local model via Transformers",
         "provider": "huggingface",
         "model_id": "meta-llama/Llama-3.2-1b",
         "api_key_env": "HUGGINGFACE_API_TOKEN",
        "max_tokens": 750,
        "temperature": 0.7
    },
    "paperclip": {
        "name": "PaperClip (Fine-Tuned Model)",
        "description": "CPU fine-tuned summarization model",
        "provider": "paperclip",
        "model_id": "backend/pipeline/finetuning/cpu_output/final_model",
        "max_tokens": 750,
        "temperature": 0.7
    }
}

DEFAULT_MODEL = "gpt-4o-mini"

def get_model_config(model_id=None):
    if model_id is None:
        model_id = DEFAULT_MODEL
    return MODEL_CONFIGS.get(model_id, MODEL_CONFIGS[DEFAULT_MODEL])


def get_available_models():
    return [
        {
            "id": model_id,
            "name": config["name"],
            "description": config["description"],
            "provider": config["provider"]
        }
        for model_id, config in MODEL_CONFIGS.items()
    ]