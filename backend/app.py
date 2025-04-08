from flask import Flask, request, jsonify, send_from_directory, send_file, Response
from flask_cors import CORS
import os
import tempfile
import uuid
import shutil
import json
import time
import threading
import queue

from convert_pdf import convert_pdf_to_markdown, create_paper_directory
from summarize_md import summarize
from chat import chat

app = Flask(__name__)
CORS(app)  # allows your frontend to call this API

PAPERS_DIR = os.path.join(os.path.dirname(__file__), "papers")
os.makedirs(PAPERS_DIR, exist_ok=True)

@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({'message': 'Backend server is working!'})

@app.route('/api/check-paper/<paper_id>', methods=['GET'])
def check_paper(paper_id):
    """
    Check if a paper already exists in the system
    """
    paper_dir = os.path.join(PAPERS_DIR, paper_id)
    md_path = os.path.join(paper_dir, f"{paper_id}.md")
    pdf_path = os.path.join(paper_dir, f"{paper_id}.pdf")
    
    # Check if both markdown and PDF files exist
    exists = os.path.exists(md_path) and os.path.exists(pdf_path)
    
    return jsonify({
        'exists': exists,
        'markdownPath': md_path if exists else None,
        'pdfPath': pdf_path if exists else None
    })

@app.route('/api/paper/<paper_id>', methods=['GET'])
def get_paper(paper_id):
    """
    Get paper content and summary for a specific paper ID
    """
    paper_dir = os.path.join(PAPERS_DIR, paper_id)
    md_path = os.path.join(paper_dir, f"{paper_id}.md")
    
    if not os.path.exists(md_path):
        return jsonify({'error': 'Paper not found'}), 404
    
    try:
        # Read the markdown content
        with open(md_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        # Generate summary if it doesn't exist
        summary_path = os.path.join(paper_dir, f"{paper_id}_summary.txt")
        if os.path.exists(summary_path):
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = f.read()
        else:
            try:
                summary = summarize(markdown_content)
                # Save the summary
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write(summary)
            except Exception as e:
                print(f"Error generating summary: {e}")
                summary = "Error generating summary. Please try again later."
        
        return jsonify({
            'success': True,
            'content': markdown_content,
            'summary': summary
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/regenerate-summary/<paper_id>', methods=['POST'])
def regenerate_summary(paper_id):
    """
    Regenerate the summary for a paper
    """
    paper_dir = os.path.join(PAPERS_DIR, paper_id)
    md_path = os.path.join(paper_dir, f"{paper_id}.md")
    summary_path = os.path.join(paper_dir, f"{paper_id}_summary.txt")
    
    if not os.path.exists(md_path):
        return jsonify({'error': 'Paper not found'}), 404
    
    try:
        # Read the markdown content
        with open(md_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        # Generate a new summary
        new_summary = summarize(markdown_content)
        
        # Save the new summary
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(new_summary)
        
        return jsonify({
            'success': True,
            'summary': new_summary
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/check-progress/<paper_id>', methods=['GET'])
def check_progress(paper_id):
    """
    Check if a paper has been processed yet (used for progress monitoring)
    """
    paper_dir = os.path.join(PAPERS_DIR, paper_id)
    md_path = os.path.join(paper_dir, f"{paper_id}.md")
    summary_path = os.path.join(paper_dir, f"{paper_id}_summary.txt")
    
    # Check if both markdown and summary files exist
    if os.path.exists(md_path) and os.path.exists(summary_path):
        return jsonify({
            'complete': True,
            'message': 'Processing complete'
        })
    elif os.path.exists(md_path):
        return jsonify({
            'complete': False,
            'progress': 0.8,
            'message': 'PDF converted, generating summary...'
        })
    else:
        return jsonify({
            'complete': False,
            'progress': 0.5,
            'message': 'Processing in progress...'
        })


@app.route('/api/process-pdf', methods=['POST'])
def process_pdf():
    """
    Endpoint to process uploaded PDF files
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    paper_id = request.form.get('paperId')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not paper_id:
        return jsonify({'error': 'No paper ID provided'}), 400
    
    if file:
        # Save the file temporarily
        temp_dir = tempfile.gettempdir()
        temp_filename = f"{uuid.uuid4().hex}_{file.filename}"
        file_path = os.path.join(temp_dir, temp_filename)
        file.save(file_path)

        # Process the PDF in a background thread
        def process_thread():
            try:
                paper_dir = create_paper_directory(paper_id)
                
                # Also save a copy of the original PDF
                pdf_path = os.path.join(paper_dir, f"{paper_id}.pdf")
                shutil.copy(file_path, pdf_path)
                
                # Convert PDF to markdown
                success = convert_pdf_to_markdown(file_path, paper_id)
                
                if not success:
                    print(f"Failed to convert PDF for {paper_id}")
                    return
                
                # Get the path to the generated markdown file
                markdown_path = os.path.join(paper_dir, f"{paper_id}.md")
                
                # Read the markdown content
                with open(markdown_path, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
                
                # Summarize the markdown
                summary = summarize(markdown_content)
                
                # Save the summary
                summary_path = os.path.join(paper_dir, f"{paper_id}_summary.txt")
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write(summary)
                
                # Clean up the temporary file
                if os.path.exists(file_path):
                    os.remove(file_path)
                    
                print(f"Successfully processed {paper_id}")
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error processing PDF: {str(e)}")
                # Clean up on error
                if os.path.exists(file_path):
                    os.remove(file_path)
        
        # Start the processing thread
        threading.Thread(target=process_thread).start()
        
        # Return success immediately without waiting for processing to complete
        return jsonify({
            'success': True,
            'message': 'PDF processing started',
            'paperId': paper_id
        })
        
    return jsonify({'error': 'Unknown error'}), 500

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    """
    Endpoint to handle chat interactions
    """
    try:
        data = request.json
        query = data.get('query')
        paper_id = data.get('paperId')
        
        if not query or not paper_id:
            return jsonify({
                'error': 'Missing required parameters',
                'details': 'Both query and paperId are required'
            }), 400
        
        # Call the chat function
        try:
            response, _ = chat(query, paper_id)
            return jsonify({
                'success': True,
                'response': response
            })
        except FileNotFoundError as e:
            return jsonify({
                'error': 'Paper not found',
                'details': str(e)
            }), 404
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({
                'error': 'Internal server error',
                'details': str(e)
            }), 500
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': 'Invalid request',
            'details': str(e)
        }), 400

@app.route('/api/pdf/<paper_id>', methods=['GET'])
def serve_pdf(paper_id):
    """
    Serve the PDF file for a specific paper
    """
    pdf_path = os.path.join(PAPERS_DIR, paper_id, f"{paper_id}.pdf")
    
    if not os.path.exists(pdf_path):
        return jsonify({'error': 'PDF not found'}), 404
    
    return send_file(pdf_path, mimetype='application/pdf')

# Add an endpoint to serve markdown files if needed
@app.route('/api/markdown/<path:filepath>', methods=['GET'])
def serve_markdown(filepath):
    return send_from_directory(PAPERS_DIR, filepath)

if __name__ == '__main__':
    app.run(debug=True, port=5000)