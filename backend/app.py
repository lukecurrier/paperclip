from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import os
import tempfile
import uuid
import shutil
import threading
import traceback
import json
import logging

from .models_config import get_model_config, get_available_models
from .services.chat_service import ChatService
from .services.summary_service import SummaryService
from .services.paper_service import PaperService
from openai import OpenAI
from sentence_transformers import SentenceTransformer

def get_embedding(text: str):
    return embedding_model.encode(text).tolist()

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("paperclip")

PAPERS_DIR = os.path.join(os.path.dirname(__file__), "papers")
os.makedirs(PAPERS_DIR, exist_ok=True)

chat_service = ChatService(embedding_fn=get_embedding)
summary_service = SummaryService()
paper_service = PaperService(embedding_fn=get_embedding)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def status_path(paper_id):
    return os.path.join(PAPERS_DIR, paper_id, "status.json")

def write_status(paper_id, status, progress=0.0, message=""):
    try:
        path = status_path(paper_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump({
                "status": status,
                "progress": progress,
                "message": message
            }, f)

        os.replace(tmp_path, path)
    except Exception as e:
        logger.exception(f"[{paper_id}] Failed writing status: {e}")

def read_status(paper_id):
    path = status_path(paper_id)
    try:
        if not os.path.exists(path):
            return {
                "status": "not_found",
                "progress": 0.0,
                "message": "not started"
            }
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {
            "status": "error",
            "progress": 0.0,
            "message": "corrupted status file"
        }


@app.route('/api/models', methods=['GET'])
def get_models():
    return jsonify(get_available_models())

@app.route('/api/check-progress/<paper_id>', methods=['GET'])
def check_progress(paper_id):
    return jsonify(read_status(paper_id))

@app.route('/api/check-paper/<paper_id>', methods=['GET'])
def check_paper(paper_id):
    paper_dir = os.path.join(PAPERS_DIR, paper_id)
    md_path = os.path.join(paper_dir, f"{paper_id}.md")
    pdf_path = os.path.join(paper_dir, f"{paper_id}.pdf")

    return jsonify({
        "exists": os.path.exists(md_path) and os.path.exists(pdf_path),
        "markdownPath": md_path,
        "pdfPath": pdf_path
    })

@app.route('/api/paper/<paper_id>', methods=['GET'])
def get_paper(paper_id):
    try:
        content, summary = paper_service.get_paper_with_summary(
            paper_id,
            summary_service
        )

        return jsonify({
            "success": True,
            "content": content,
            "summary": summary
        })

    except FileNotFoundError:
        return jsonify({"error": "Paper not found"}), 404

    except Exception as e:
        logger.exception(e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/process-pdf', methods=['POST'])
def process_pdf():
    try:
        file = request.files.get("file")
        paper_id = request.form.get("paperId")

        if not file or not paper_id:
            return jsonify({"error": "Missing file or paperId"}), 400

        paper_dir = os.path.join(PAPERS_DIR, paper_id)
        md_path = os.path.join(paper_dir, f"{paper_id}.md")
        pdf_path = os.path.join(paper_dir, f"{paper_id}.pdf")

        if os.path.exists(paper_dir):
            shutil.rmtree(paper_dir)

        os.makedirs(paper_dir, exist_ok=True)

        write_status(paper_id, "processing", 0.3, "upload received")

        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, f"{uuid.uuid4().hex}_{file.filename}")
        file.save(file_path)

        def worker():
            try:
                write_status(paper_id, "processing", 0.4, "parsing PDF")

                paper_service.process_pdf(file_path, paper_id)

                write_status(paper_id, "complete", 1.0, "done")

                logger.info(f"[{paper_id}] processing complete")

            except Exception as e:
                logger.exception(f"[{paper_id}] worker failed")
                write_status(paper_id, "failed", 0.0, str(e))

            finally:
                if os.path.exists(file_path):
                    os.remove(file_path)

        threading.Thread(target=worker, daemon=True).start()

        return jsonify({
            "success": True,
            "skipped": False,
            "message": "Processing started",
            "paperId": paper_id
        }), 200

    except Exception as e:
        logger.exception(e)
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    try:
        data = request.json or {}

        query = data.get("query")
        paper_id = data.get("paperId")
        model_id = data.get("modelId", "gpt-4o-mini")
        history = data.get("history", [])

        if not query or not paper_id:
            return jsonify({"error": "Missing query or paperId"}), 400

        status = read_status(paper_id)

        if status["status"] != "complete":
            return jsonify({
                "error": "Paper not ready",
                "status": status
            }), 409

        result = chat_service.chat(
            query=query,
            paper_id=paper_id,
            history=history,
            model_id=model_id
        )

        return jsonify({
            "success": True,
            "response": result["response"],
            "model_used": model_id
        })

    except Exception as e:
        logger.exception(e)
        return jsonify({"error": str(e)}), 500

@app.route('/api/pdf/<paper_id>', methods=['GET'])
def serve_pdf(paper_id):
    path = os.path.join(PAPERS_DIR, paper_id, f"{paper_id}.pdf")

    if not os.path.exists(path):
        return jsonify({"error": "PDF not found"}), 404

    return send_file(path, mimetype="application/pdf", as_attachment=False, download_name=f"{paper_id}.pdf")


@app.route('/api/markdown/<path:filepath>', methods=['GET'])
def serve_markdown(filepath):
    return send_from_directory(PAPERS_DIR, filepath)
    
if __name__ == "__main__":
    app.run(debug=True, port=8000)