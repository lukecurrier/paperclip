from pathlib import Path

PAPERS_DIR = Path(__file__).parent.parent / "papers"

class PaperRepository:

    @staticmethod
    def get_paper_dir(paper_id: str) -> Path:
        return PAPERS_DIR / paper_id

    @staticmethod
    def markdown_path(paper_id: str) -> Path:
        return PaperRepository.get_paper_dir(paper_id) / f"{paper_id}.md"

    @staticmethod
    def pdf_path(paper_id: str) -> Path:
        return PaperRepository.get_paper_dir(paper_id) / f"{paper_id}.pdf"

    @staticmethod
    def summary_path(paper_id: str) -> Path:
        return PaperRepository.get_paper_dir(paper_id) / f"{paper_id}_summary.txt"

    @staticmethod
    def context_path(paper_id: str) -> Path:
        return PaperRepository.get_paper_dir(paper_id) / "context.txt"