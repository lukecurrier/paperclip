import os
from dotenv import load_dotenv
import torch
from openai import OpenAI

def summarize(document_content):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("BASE_URL")
    
    if hasattr(torch.backends, 'mps'):
        print(f"MPS available: {torch.backends.mps.is_available()}")
    
    client = OpenAI(api_key=api_key, base_url=base_url)

    SYSTEM_PROMPT = """You are a helpful assistant that summarizes scientific papers."""
    
    USER_PROMPT = f"""Please briefly summarize the following markdown content:
    
    {document_content}

    Summary:"""
    
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT}
        ],
        model="llama3p1-8b-instruct",
        temperature=0,
        max_tokens=300
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    # take input from the command line
    import sys
    document_content = sys.argv[1]
    print(document_content)
    print(summarize(document_content))