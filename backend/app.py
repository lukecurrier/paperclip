from flask import Flask, request, jsonify, send_from_directory, send_file, Response
from flask_cors import CORS
import os
import tempfile
import uuid
import shutil
import threading
from models_config import get_available_models
from summarize_md import summarize
from chat import chat
from convert_pdf import create_paper_directory, convert_pdf_to_markdown

app = Flask(__name__)
CORS(app) 

PAPERS_DIR = os.path.join(os.path.dirname(__file__), "papers")
os.makedirs(PAPERS_DIR, exist_ok=True)

@app.route('/api/models', methods=['GET'])
def get_models():
    models = get_available_models()
    print(models)
    return jsonify(models)

@app.route('/api/check-paper/<paper_id>', methods=['GET'])
def check_paper(paper_id):
    paper_dir = os.path.join(PAPERS_DIR, paper_id)
    md_path = os.path.join(paper_dir, f"{paper_id}.md")
    pdf_path = os.path.join(paper_dir, f"{paper_id}.pdf")
    
    exists = os.path.exists(md_path) and os.path.exists(pdf_path)
    
    return jsonify({
        'exists': exists,
        'markdownPath': md_path if exists else None,
        'pdfPath': pdf_path if exists else None
    })

@app.route('/api/paper/<paper_id>', methods=['GET'])
def get_paper(paper_id):    
    paper_dir = os.path.join(PAPERS_DIR, paper_id)
    md_path = os.path.join(paper_dir, f"{paper_id}.md")
    
    if not os.path.exists(md_path):
        return jsonify({'error': 'Paper not found'}), 404
    
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        summary_path = os.path.join(paper_dir, f"{paper_id}_summary.txt")
        if os.path.exists(summary_path):
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = f.read()
        else:
            try:
                summary = summarize(markdown_content)
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

@app.route('/api/check-progress/<paper_id>', methods=['GET'])
def check_progress(paper_id):
    paper_dir = os.path.join(PAPERS_DIR, paper_id)
    md_path = os.path.join(paper_dir, f"{paper_id}.md")
    summary_path = os.path.join(paper_dir, f"{paper_id}_summary.txt")
    
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
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    paper_id = request.form.get('paperId')
    model_id = request.form.get('modelId')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not paper_id:
        return jsonify({'error': 'No paper ID provided'}), 400
    
    if file:
        temp_dir = tempfile.gettempdir()
        temp_filename = f"{uuid.uuid4().hex}_{file.filename}"
        file_path = os.path.join(temp_dir, temp_filename)
        file.save(file_path)

        def process_thread():
            try:
                paper_dir = create_paper_directory(paper_id)
                
                pdf_path = os.path.join(paper_dir, f"{paper_id}.pdf")
                shutil.copy(file_path, pdf_path)
                
                success = convert_pdf_to_markdown(file_path, paper_id)
                
                if not success:
                    print(f"Failed to convert PDF for {paper_id}")
                    return
                
                markdown_path = os.path.join(paper_dir, f"{paper_id}.md")
                
                with open(markdown_path, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
                
                summary = summarize(markdown_content, model_id)
                
                summary_path = os.path.join(paper_dir, f"{paper_id}_summary.txt")
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write(summary)
                
                if os.path.exists(file_path):
                    os.remove(file_path)
                    
                print(f"Successfully processed {paper_id}")
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error processing PDF: {str(e)}")
                if os.path.exists(file_path):
                    os.remove(file_path)
        
        threading.Thread(target=process_thread).start()
        
        return jsonify({
            'success': True,
            'message': 'PDF processing started',
            'paperId': paper_id
        })
        
    return jsonify({'error': 'Unknown error'}), 500

@app.route('/api/regenerate-summary/<paper_id>', methods=['POST'])
def regenerate_summary(paper_id):
    model_id = request.json.get('modelId')
    paper_dir = os.path.join(PAPERS_DIR, paper_id)
    md_path = os.path.join(paper_dir, f"{paper_id}.md")
    
    if not os.path.exists(md_path):
        return jsonify({'error': 'Paper not found'}), 404
    
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        new_summary = summarize(markdown_content, model_id)
        
        summary_path = os.path.join(paper_dir, f"{paper_id}_summary.txt")
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

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    try:
        data = request.json
        query = data.get('query')
        paper_id = data.get('paperId')
        model_id = data.get('modelId')
        
        if not query or not paper_id:
            return jsonify({
                'error': 'Missing required parameters',
                'details': 'Both query and paperId are required'
            }), 400
        
        response, _ = chat(query, paper_id, model_id)
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

@app.route('/api/pdf/<paper_id>', methods=['GET'])
def serve_pdf(paper_id):
    pdf_path = os.path.join(PAPERS_DIR, paper_id, f"{paper_id}.pdf")
    
    if not os.path.exists(pdf_path):
        return jsonify({'error': 'PDF not found'}), 404
    
    return send_file(pdf_path, mimetype='application/pdf')

@app.route('/api/markdown/<path:filepath>', methods=['GET'])
def serve_markdown(filepath):
    return send_from_directory(PAPERS_DIR, filepath)

if __name__ == '__main__':
    app.run(debug=True, port=8000)