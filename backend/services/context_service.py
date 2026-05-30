import os

class ContextService:
    def __init__(self, base_dir="papers"):
        self.base_dir = base_dir

    def _context_path(self, paper_id):
        return os.path.join(self.base_dir, paper_id, "context.txt")

    def load(self, paper_id):
        path = self._context_path(paper_id)
        if not os.path.exists(path):
            return ""
        return open(path, "r", encoding="utf-8").read()

    def save(self, paper_id, context):
        path = self._context_path(paper_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(context)

    def append(self, existing, query, response):
        return f"{existing}\n\nQuery: {query}\nResponse: {response}"