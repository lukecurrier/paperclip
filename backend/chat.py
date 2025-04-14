# chat.py
import os
import sys
from model_client import ModelClientFactory

def chat(query, paper_id, model_id=None):
    """
    Returns both the simple response and the entire formatted document/context/query/response for recalling the function
    """
    try:
        # Get the appropriate model client
        client = ModelClientFactory.get_client(model_id)
        
        # Generate chat response using the selected model
        # This will automatically handle context management
        response = client.generate_chat_response(query, paper_id)
        
        # Get updated context for returning
        context = client.get_context(paper_id)
        
        return response, context
    except Exception as e:
        print(f"Error in chat function: {str(e)}", file=sys.stderr)
        raise

if __name__ == "__main__":
    # use the function, taking arguments from the command line
    query = sys.argv[1]
    paper_id = sys.argv[2]
    model_id = sys.argv[3] if len(sys.argv) > 3 else None
    response, context = chat(query, paper_id, model_id)
    print(f"Response: {response}")
    print(f"Context: {context}")