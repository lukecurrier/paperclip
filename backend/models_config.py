MODEL_CONFIGS = {
    "llama-3.1-8b": {
        "name": "Llama 3.1 8B",
        "description": "OpenAI-hosted large Llama model",
        "provider": "openai",
        "model_id": "llama3p1-8b-instruct",
        "api_key_env": "OPENAI_API_KEY",
        "max_tokens": 750,
        "temperature": 0.4
    },
    "llama-3.1-8b2": {
        "name": "Llama 3.2 1B",
        "description": "HF-hosted small Llama model",
        "provider": "openai",
        "model_id": "llama3p1-8b-instruct",
        "api_key_env": "OPENAI_API_KEY",
        "max_tokens": 750,
        "temperature": 0.9
    },
    "llama-3.1-8b3": {
        "name": "PaperClip (Fine-Tuned Llama)",
        "description": "Paper-specialized fine-tuned model",
        "provider": "openai",
        "model_id": "llama3p1-8b-instruct",
        "api_key_env": "OPENAI_API_KEY",
        "max_tokens": 750,
        "temperature": 0.9
    },
#     "llama-3.2-1b": {
#         "name": "Llama 3.2 1B",
#         "description": "Local model via Transformers",
#         "provider": "huggingface",
#         "model_id": "meta-llama/Llama-3.2-1b",
#         "api_key_env": "HUGGINGFACE_API_TOKEN",
#         "max_tokens": 750,
#         "temperature": 0.7
#     },
#     "paperclip": {
#         "name": "Fine-Tuned Llama 1B",
#         "description": "Paper-specialized fine-tuned model",
#         "provider": "paperclip",
#         "base_model_id": "meta-llama/Llama-3.2-1B",
#         "model_id": "neharavuri/paperclip-Llama-3.2-1B-finetuned",
#         "api_key_env": "HUGGINGFACE_API_TOKEN",
#         "max_tokens": 750,
#         "temperature": 0.7
#     }
}

DEFAULT_MODEL = "llama-3.1-8b"

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