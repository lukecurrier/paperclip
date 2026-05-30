import os
from dotenv import load_dotenv
from openai import OpenAI
from .base import ModelClient
import re


class OpenAIClient(ModelClient):

    def __init__(self, config):
        load_dotenv()
        super().__init__(config)

        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("BASE_URL")
        )

    # -----------------------
    def generate_chat_response(self, prompt, **kwargs):

        response = self.client.chat.completions.create(
            model=self.config["model_id"],
            messages=[
                {
                    "role": "system",
                    "content": "You answer strictly based on provided context."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            max_tokens=400
        ).choices[0].message.content

        return self.clean(response)
    
    def generate_summary(self, text, **kwargs):
        prompt = f"""
    Summarize the following academic paper clearly and concisely:

    {text}
    """

        response = self.client.chat.completions.create(
            model=self.config["model_id"],
            messages=[
                {"role": "system", "content": "You summarize academic papers."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )

        return response.choices[0].message.content.strip()

    # -----------------------
    def clean(self, text):
        text = re.sub(r'```.*?```', ' ', text, flags=re.DOTALL)
        text = re.sub(r'`.*?`', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text