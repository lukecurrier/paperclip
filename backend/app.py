from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import tempfile
import uuid

from convert_pdf import convert_pdf_to_markdown

app = Flask(__name__)
CORS(app)  # This allows your frontend to call this API

# Create a directory to store conversions
CONVERSIONS_DIR = os.path.join(os.path.dirname(__file__), "conversions")
os.makedirs(CONVERSIONS_DIR, exist_ok=True)

@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({'message': 'Backend server is working!'})

@app.route('/api/process-pdf', methods=['POST'])
def process_pdf():
    """
    Endpoint to process uploaded PDF files
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        # Save the file temporarily
        temp_dir = tempfile.gettempdir()
        temp_filename = f"{uuid.uuid4().hex}_{file.filename}"
        file_path = os.path.join(temp_dir, temp_filename)
        file.save(file_path)
        
        try:
            # Create a unique output directory for this conversion
            output_path = os.path.join(CONVERSIONS_DIR, uuid.uuid4().hex)
            
            # Call your PDF conversion function
            success = convert_pdf_to_markdown(file_path, output_path)
            
            if not success:
                return jsonify({'error': 'Failed to convert PDF'}), 500
            
            # Get the path to the generated markdown file
            markdown_filename = os.path.splitext(os.path.basename(file_path))[0] + ".md"
            markdown_path = os.path.join(output_path, markdown_filename)
            
            # Read the markdown content
            with open(markdown_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            
            # Create a simple placeholder summary
            # First, try to extract a title from the markdown
            lines = markdown_content.split('\n')
            title = "Untitled Document"
            for line in lines:
                if line.startswith('# '):
                    title = line[2:].strip()
                    break
            
            # Create a basic summary with word count and section count
            section_count = sum(1 for line in lines if line.startswith('# ') or line.startswith('## '))
            word_count = len(markdown_content.split())
            
            summary = f"# Summary of {os.path.basename(file.filename)}\n\n"
            summary += f"## Document: {title}\n\n"
            summary += f"This document contains approximately {word_count} words "
            summary += f"organized across {section_count} major sections.\n\n"
            summary += "## Key Points\n\n"
            summary += "- The document has been successfully converted to markdown format\n"
            summary += "- You can ask questions about the content in the discussion tab\n"
            summary += "- A more detailed summary will be available when summarization is implemented\n"
            
            # Clean up the temporary file
            os.remove(file_path)
            
            return jsonify({
                'success': True,
                'extractedText': markdown_content,
                'summary': summary,
                'markdownPath': markdown_path
            })
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(file_path):
                os.remove(file_path)
            
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Endpoint to handle chat interactions about the paper
    """
    try:
        data = request.json
        question = data.get('question')
        paper_content = data.get('paperContent')
        
        if not question or not paper_content:
            return jsonify({'error': 'Missing required parameters'}), 400
        
        # Call your existing chat function
        # Replace this with your actual function
        # answer = your_chat_module.generate_response(question, paper_content)
        
        # For testing, we'll use a placeholder response
        answer = f"This is a response to your question: '{question}'"
        
        return jsonify({
            'success': True,
            'answer': answer
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-audio', methods=['POST'])
def generate_audio():
    """
    Endpoint to generate audio from text
    """
    try:
        data = request.json
        text = data.get('text')
        
        if not text:
            return jsonify({'error': 'Missing text parameter'}), 400
        
        # Call your existing audio generation function
        # Replace this with your actual function
        # audio_file_path = your_audio_generator.create_audio(text)
        
        # For testing, return a placeholder URL
        audio_url = "/api/audio/placeholder.mp3"
        
        return jsonify({
            'success': True,
            'audioUrl': audio_url
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# Add an endpoint to serve audio files if needed
@app.route('/api/audio/<filename>', methods=['GET'])
def serve_audio(filename):
    audio_dir = os.path.join(os.path.dirname(__file__), "audio")
    return send_from_directory(audio_dir, filename)

# Add an endpoint to serve markdown files if needed
@app.route('/api/markdown/<path:filepath>', methods=['GET'])
def serve_markdown(filepath):
    return send_from_directory(CONVERSIONS_DIR, filepath)

if __name__ == '__main__':
    app.run(debug=True, port=5000)