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

    SYSTEM_PROMPT = """You are a helpful assistant that summarizes scientific papers. 
You have a conversational but professional tone, and are trying to synthesize information in the most accessible way possible.
    
When writing a summary, make sure to add line breaks and formatting to make things possible to read quickly and easily. 
If using technical terms or abbreviations, give context or a brief explanation.
Keep your summaries to no more than a few short paragraphs."""
    
    USER_PROMPT = f"""Please summarize the following markdown content:
    
    {document_content}

    Summary:"""
    
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT}
        ],
        model="llama3p1-8b-instruct",
        temperature=0,
        max_tokens=500
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    # take input to a markdown file path from the command line
    import sys
    document_path = sys.argv[1]
    with open(document_path, 'r', encoding='utf-8') as f:
        document_content = f.read()
    print(summarize(document_content))