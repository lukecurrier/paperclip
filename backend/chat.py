import os
from dotenv import load_dotenv
from openai import OpenAI
import sys
import re
from typing import Tuple

def clean_markdown(text: str) -> str:
    """
    Clean markdown text by removing code blocks and other formatting that might confuse the model
    """
    # Remove code blocks
    text = re.sub(r'```.*?```', ' ', text, flags=re.DOTALL)
    # Remove inline code
    text = re.sub(r'`.*?`', ' ', text)
    # Remove headers
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    # Remove lists
    text = re.sub(r'^[-*]\s+', '', text, flags=re.MULTILINE)
    # Remove emphasis
    text = re.sub(r'\*\*|__', ' ', text)
    # Remove links
    text = re.sub(r'\[.*?\]\(.*?\)', ' ', text)
    # Remove images
    text = re.sub(r'!\[.*?\]\(.*?\)', ' ', text)
    # Remove horizontal rules
    text = re.sub(r'^[-=]{3,}$', ' ', text, flags=re.MULTILINE)
    # Remove blockquotes
    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def chat(query: str, paper_id: str) -> Tuple[str, str]:
    """
    Returns both the simple response and the entire formatted document/context/query/response for recalling the function
    """
    try:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("BASE_URL")

        # Get context
        context = get_context(paper_id)
        
        # Clean markdown in context
        clean_context = clean_markdown(context)

        client = OpenAI(api_key=api_key, base_url=base_url)

        SYSTEM_PROMPT = """You are a helpful assistant that summarizes scientific papers. 
Your task is to answer questions about the paper, including giving context to specific parts of the text and extrapolating for the user.
Answer with friendly, conversational language, and make the user feel comfortable talking about anything related to the paper.
If you don't know the answer, say so - don't make things up."""
        
        USER_PROMPT = f"""Paper Content:
{clean_context}

User Question: {query}"""
        
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT}
            ],
            model="llama3p1-8b-instruct",
            temperature=0,
            max_tokens=500
        )
        
        # Clean the response
        clean_response = clean_markdown(response.choices[0].message.content)
        
        new_context = format_context(context, query, clean_response)
        save_context(paper_id, new_context)
        
        return clean_response, new_context
        
    except Exception as e:
        print(f"Error in chat function: {str(e)}", file=sys.stderr)
        raise

def get_context(paper_id):
    """
    Get paper content and context from the papers directory
    """
    papers_dir = os.path.join(os.path.dirname(__file__), 'papers')
    paper_path = os.path.join(papers_dir, paper_id, f'{paper_id}.md')
    context_path = os.path.join(papers_dir, paper_id, 'context.txt')
    
    if not os.path.exists(paper_path):
        raise FileNotFoundError(f"Paper not found: {paper_path}")
    
    if not os.path.exists(context_path):
        print("No context path, making new file")
        os.makedirs(os.path.dirname(context_path), exist_ok=True)
        with open(paper_path, 'r', encoding='utf-8') as f:
            paper_content = f.read().strip()
        with open(context_path, 'w', encoding='utf-8') as f:
            f.write(paper_content + "\n---------------------------------------------------------------------")

    with open(context_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def save_context(paper_id, new_context):
    """
    Save the updated context for a paper
    """
    papers_dir = os.path.join(os.path.dirname(__file__), 'papers')
    context_path = os.path.join(papers_dir, paper_id, 'context.txt')
    
    os.makedirs(os.path.dirname(context_path), exist_ok=True)
    with open(context_path, 'w', encoding='utf-8') as f:
        print("Saving context!")
        f.write(new_context)

def format_context(existing_context, query, response):
    return f"{existing_context}\n\nQuery: {query}\n\nResponse: {response}"

if __name__ == "__main__":
    paper_id = sys.argv[1]
    
    while True:
        query = input("User: ")
        if query == "":
            break
        response, _ = chat(query, paper_id)
        print(response)