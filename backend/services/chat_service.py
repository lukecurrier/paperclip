import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..models.model_client_factory import ModelClientFactory
from .paper_service import PaperService

class ChatService:

    def __init__(self, embedding_fn):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.paper_service = PaperService()
        self.embedding_fn = embedding_fn

    def chat(self, query, paper_id, model_id=None):
        try:
            client = ModelClientFactory.get_client(model_id)
            chunks = self.paper_service.load_chunks(paper_id)
            embeddings = self.paper_service.load_embeddings(paper_id)
            if chunks is None or embeddings is None:
                raise ValueError("Missing chunks or embeddings")
            if len(chunks) == 0 or len(embeddings) == 0:
                raise ValueError("Empty chunks or embeddings")
            context = self.retrieve(query, chunks, embeddings)
            prompt = self.build_prompt(query, context)
            response = client.generate_chat_response(prompt)
            return {
                "response": response,
                "context_used": context
            }
        except Exception as e:
            self.logger.exception(f"[{paper_id}] chat failed: {e}")
            raise

    # Uses the embeddings to retrive the top k most relavant chunks for the query
    def retrieve(self, query, chunks, embeddings, k=5):
        # Embed the user query and compute cosine similarity with chunk embeddings
        query_emb = np.array(self.embedding_fn(query))
        embeddings = np.array(embeddings)
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        scores = embeddings @ query_emb

        # Get the indices of the top k most similar chunks
        top_idx = np.argsort(scores)[::-1]

        # Filter out chunks below a similarity threshold (e.g., 0.25) and take top k
        filtered = []
        for i in top_idx:
            if scores[i] < 0.25:
                continue
            filtered.append(chunks[i])
            if len(filtered) == k:
                break

        # Fallback if nothing passes threshold
        if not filtered:
            filtered = [chunks[i] for i in top_idx[:k]]

        return "\n\n---\n\n".join(filtered)

    def build_prompt(self, query, context):
        return f"""
You are PaperClip, an expert research assistant.

Your job:
- Use ONLY the provided context.
- If the answer is partially in context, you MUST still answer.
- If the context is weak, infer reasonable explanations from it.
- Do NOT say "not enough information" unless context is completely unrelated.

---

CONTEXT:
{context}

---

QUESTION:
{query}

---

INSTRUCTIONS:
Explain clearly, simply, and directly.

---

ANSWER:
"""