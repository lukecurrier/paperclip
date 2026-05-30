import logging
from ..models.model_client_factory import ModelClientFactory
from .paper_service import PaperService


class ChatService:

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.paper_service = PaperService()

    # -----------------------
    def chat(self, query, paper_id, model_id=None):

        try:
            client = ModelClientFactory.get_client(model_id)

            # load chunks
            chunks = self.paper_service.load_chunks(paper_id)

            # retrieve relevant chunks (simple RAG)
            context = self.retrieve(query, chunks)

            prompt = self.build_prompt(query, context)

            response = client.generate_chat_response(prompt)

            return {
                "response": response,
                "context_used": context
            }

        except Exception as e:
            self.logger.exception(f"[{paper_id}] chat failed: {e}")
            raise

    # -----------------------
    def retrieve(self, query, chunks, k=3):
        q = set(query.lower().split())

        scored = []
        for c in chunks:
            score = len(q & set(c.lower().split()))

            # boost important sections
            if "abstract" in c.lower():
                score += 5
            if "conclusion" in c.lower():
                score += 4
            if "introduction" in c.lower():
                score += 3

            scored.append((score, c))

        scored.sort(reverse=True, key=lambda x: x[0])

        return "\n\n".join([c for _, c in scored[:k]])

    # -----------------------
    def build_prompt(self, query, context):
        return f"""
You are an expert assistant answering questions about an academic paper.

Use ONLY the provided context.

If the answer is not in the context, say:
"The paper does not contain enough information to answer this."

Context:
{context}

Question:
{query}
"""