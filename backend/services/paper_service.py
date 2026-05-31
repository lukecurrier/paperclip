from pathlib import Path
import shutil
import fitz
import re
import json
import logging
import numpy as np
class PaperService:

    BASE_DIR = Path(__file__).parent.parent / "papers"

    def __init__(self, embedding_fn=None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.embedding_fn = embedding_fn  # IMPORTANT

    def create_paper_directory(self, paper_id):
        paper_dir = self.BASE_DIR / paper_id
        paper_dir.mkdir(parents=True, exist_ok=True)
        return paper_dir

    def convert_pdf_to_text(self, pdf_path):
        doc = fitz.open(pdf_path)
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                pages.append(f"\n\n## Page {i+1}\n\n{text}")
        return "\n".join(pages)

    def clean_text(self, text: str) -> str:
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def chunk_text(self, text, max_words=250):
        sentences = text.split(". ")
        chunks = []
        current = []
        for s in sentences:
            current.append(s)
            if len(current) >= max_words:
                chunks.append(". ".join(current))
                current = []
        if current:
            chunks.append(". ".join(current))
        return chunks

    # Core processing pipeline PDF -> text -> chunks + embeddings
    def process_pdf(self, temp_pdf_path, paper_id):
        paper_dir = self.create_paper_directory(paper_id)
        pdf_path = paper_dir / f"{paper_id}.pdf"
        shutil.copy(temp_pdf_path, pdf_path)

        raw = self.convert_pdf_to_text(pdf_path)
        clean = self.clean_text(raw)

        chunks = self.chunk_text(clean)

        # Save the markdown of the paper and the chunks that were created
        (paper_dir / f"{paper_id}.md").write_text(clean, encoding="utf-8")
        (paper_dir / "chunks.json").write_text(json.dumps(chunks, indent=2))

        embeddings = []

        # Create and save the embeddings for the previously created chunks
        if self.embedding_fn is None:
            raise ValueError("embedding_fn not provided to PaperService")

        for c in chunks:
            emb = self.embedding_fn(c)
            embeddings.append(emb)

        np.save(paper_dir / "embeddings.npy", np.array(embeddings, dtype="float32"))
        self.logger.info(f"[{paper_id}] processed {len(chunks)} chunks + embeddings")
        return {"paper_id": paper_id, "chunks": len(chunks)}

    def load_chunks(self, paper_id):
        path = self.BASE_DIR / paper_id / "chunks.json"
        return json.loads(path.read_text())

    def load_embeddings(self, paper_id):
        path = self.BASE_DIR / paper_id / "embeddings.npy"
        return np.load(path)
    
    def get_paper_with_summary(self, paper_id, summary_service):
        paper_dir = self.BASE_DIR / paper_id

        md_path = paper_dir / f"{paper_id}.md"
        summary_path = paper_dir / f"{paper_id}_summary.txt"

        if not md_path.exists():
            raise FileNotFoundError("Markdown not found")

        content = md_path.read_text(encoding="utf-8")

        if summary_path.exists():
            summary = summary_path.read_text(encoding="utf-8")
        else:
            summary = summary_service.summarize(content)
            summary_path.write_text(summary, encoding="utf-8")

        return content, summary