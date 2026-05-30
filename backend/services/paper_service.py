from pathlib import Path
import shutil
import fitz
import re
import json
import logging


class PaperService:

    BASE_DIR = Path(__file__).parent.parent / "papers"

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def create_paper_directory(self, paper_id):
        paper_dir = self.BASE_DIR / paper_id
        paper_dir.mkdir(parents=True, exist_ok=True)
        return paper_dir

    # -----------------------
    def convert_pdf_to_text(self, pdf_path):
        doc = fitz.open(pdf_path)

        pages = []
        for i, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                pages.append(f"\n\n## Page {i+1}\n\n{text}")

        return "\n".join(pages)

    # -----------------------
    def clean_text(self, text: str) -> str:
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    # -----------------------
    def chunk_text(self, text: str, max_size=1500):
        paragraphs = text.split("\n\n")

        chunks = []
        current = ""

        for p in paragraphs:
            if len(current) + len(p) > max_size:
                chunks.append(current.strip())
                current = p
            else:
                current += "\n\n" + p

        if current.strip():
            chunks.append(current.strip())

        return chunks

    # -----------------------
    def process_pdf(self, temp_pdf_path, paper_id):
        paper_dir = self.create_paper_directory(paper_id)

        pdf_path = paper_dir / f"{paper_id}.pdf"
        shutil.copy(temp_pdf_path, pdf_path)

        raw = self.convert_pdf_to_text(pdf_path)
        clean = self.clean_text(raw)

        chunks = self.chunk_text(clean)

        # save markdown
        (paper_dir / f"{paper_id}.md").write_text(clean, encoding="utf-8")

        # save chunks
        (paper_dir / "chunks.json").write_text(
            json.dumps(chunks, indent=2),
            encoding="utf-8"
        )

        self.logger.info(f"[{paper_id}] processed with {len(chunks)} chunks")

        return {
            "paper_id": paper_id,
            "chunks": len(chunks)
        }

    # -----------------------
    def load_chunks(self, paper_id):
        path = self.BASE_DIR / paper_id / "chunks.json"

        if not path.exists():
            raise FileNotFoundError("Paper not processed yet")

        return json.loads(path.read_text())
    
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