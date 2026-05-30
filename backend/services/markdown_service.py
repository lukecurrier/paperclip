import re

def clean_markdown(text: str) -> str:
    text = re.sub(r'```.*?```', ' ', text, flags=re.DOTALL)
    text = re.sub(r'`.*?`', ' ', text)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[-*]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*|__', ' ', text)
    text = re.sub(r'\[.*?\]\(.*?\)', ' ', text)
    text = re.sub(r'!\[.*?\]\(.*?\)', ' ', text)
    text = re.sub(r'^[-=]{3,}$', ' ', text, flags=re.MULTILINE)
    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text