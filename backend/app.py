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

# Dictionary to store progress information for each paper processing task
processing_progress = {}

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
        # Check if this paper is still being processed
        if paper_id in processing_progress:
            return jsonify({
                'success': False,
                'status': 'processing',
                'message': 'Paper is still being processed'
            }), 202
        
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

def process_pdf_task(file_path, paper_id, progress_queue):
    """
    Background task to process a PDF file
    """
    try:
        paper_dir = create_paper_directory(paper_id)
        
        # Also save a copy of the original PDF
        pdf_path = os.path.join(paper_dir, f"{paper_id}.pdf")
        shutil.copy(file_path, pdf_path)
        
        progress_queue.put({"progress": 0.2, "message": "PDF saved, starting conversion..."})
        
        # Convert PDF to markdown
        progress_queue.put({"progress": 0.3, "message": "Extracting text from PDF..."})
        success = convert_pdf_to_markdown(file_path, paper_id)
        
        if not success:
            progress_queue.put({"progress": 0, "message": "Failed to convert PDF", "error": True})
            return None, None, "Failed to convert PDF"
        
        progress_queue.put({"progress": 0.6, "message": "PDF converted to text, reading content..."})
        
        # Get the path to the generated markdown file
        markdown_path = os.path.join(paper_dir, f"{paper_id}.md")
        
        # Read the markdown content
        with open(markdown_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        progress_queue.put({"progress": 0.7, "message": "Generating summary..."})
        
        # Summarize the markdown
        summary = summarize(markdown_content)
        
        progress_queue.put({"progress": 0.9, "message": "Saving summary..."})
        
        # Save the summary
        summary_path = os.path.join(paper_dir, f"{paper_id}_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        # Clean up the temporary file
        if os.path.exists(file_path):
            os.remove(file_path)
        
        progress_queue.put({"progress": 1.0, "message": "Processing complete!"})
        
        return markdown_content, summary, None
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Clean up on error
        if os.path.exists(file_path):
            os.remove(file_path)
        progress_queue.put({"progress": 0, "message": f"Error: {str(e)}", "error": True})
        return None, None, str(e)

@app.route('/api/process-progress/<paper_id>', methods=['GET'])
def get_processing_progress(paper_id):
    """
    SSE endpoint to stream progress updates
    """
    def generate():
        if paper_id not in processing_progress:
            yield f"data: {json.dumps({'progress': 0, 'message': 'No processing task found'})}\n\n"
            return
        
        q = processing_progress[paper_id]
        last_progress = None
        
        while True:
            try:
                # Non-blocking get with timeout
                progress_data = q.get(timeout=0.1)
                q.task_done()
                
                # If we got an error, return it and stop
                if progress_data.get("error", False):
                    yield f"data: {json.dumps(progress_data)}\n\n"
                    break
                
                # Send progress update
                yield f"data: {json.dumps(progress_data)}\n\n"
                last_progress = progress_data
                
                # If we reached 100%, stop streaming
                if progress_data["progress"] >= 1.0:
                    break
            except queue.Empty:
                # If no updates, check if the queue still exists
                if paper_id not in processing_progress:
                    if last_progress and last_progress["progress"] < 1.0:
                        yield f"data: {json.dumps({'progress': 0, 'message': 'Processing stopped unexpectedly'})}\n\n"
                    break
                time.sleep(0.5)  # Short sleep to prevent CPU spinning
    
    return Response(generate(), mimetype="text/event-stream")

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

        # Create a queue for progress updates
        progress_queue = queue.Queue()
        processing_progress[paper_id] = progress_queue
        
        # Initial progress update
        progress_queue.put({"progress": 0.1, "message": "Starting PDF processing..."})
        
        # Process the PDF in a background thread
        def process_thread():
            markdown_content, summary, error = process_pdf_task(file_path, paper_id, progress_queue)
            # Keep the queue in the dictionary for a short while to ensure all messages are read
            time.sleep(5)
            # Clean up
            if paper_id in processing_progress:
                del processing_progress[paper_id]
        
        threading.Thread(target=process_thread).start()
        
        # Return success immediately without waiting for processing to complete
        return jsonify({
            'success': True,
            'message': 'PDF processing started',
            'paperid': paper_id
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