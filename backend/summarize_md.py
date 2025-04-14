from model_client import ModelClientFactory

def summarize(markdown_content, model_id=None):
    try:
        client = ModelClientFactory.get_client(model_id)
        
        summary = client.generate_summary(markdown_content)
        
        return summary
    except Exception as e:
        print(f"Error generating summary: {e}")
        return "Error generating summary"

if __name__ == "__main__":
    import sys
    path_to_markdown = sys.argv[1]
    model_id = sys.argv[2] if len(sys.argv) > 2 else None
    with open(path_to_markdown, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    print(summarize(markdown_content, model_id))