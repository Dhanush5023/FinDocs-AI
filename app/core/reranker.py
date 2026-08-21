from typing import List, Dict, Any, Optional

class CrossEncoderReRanker:
    """
    Contextual Re-ranking Layer using Cross-Encoder models (e.g., ms-marco-MiniLM-L-6-v2).
    Refines Hybrid RRF candidates into top-k high-precision chunks, filtering noise.
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None
        self._initialized = False

    def _lazy_init(self):
        if not self._initialized:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except Exception:
                self._model = None
            self._initialized = True

    def rerank(self, query: str, candidate_chunks: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        if not candidate_chunks:
            return []

        self._lazy_init()

        if self._model is not None:
            pairs = [[query, chunk["content"]] for chunk in candidate_chunks]
            scores = self._model.predict(pairs)
            
            for i, chunk in enumerate(candidate_chunks):
                chunk["rerank_score"] = float(scores[i])
            
            sorted_chunks = sorted(candidate_chunks, key=lambda x: x["rerank_score"], reverse=True)
            return sorted_chunks[:top_k]

        # Fast fallback: use hybrid RRF score with length normalization
        for chunk in candidate_chunks:
            # Table chunks get a slight confidence boost for numerical queries
            is_table = chunk.get("chunk_type") == "financial_table"
            boost = 1.15 if is_table and any(char.isdigit() for char in query) else 1.0
            chunk["rerank_score"] = round(chunk.get("rrf_score", 0.0) * boost, 5)

        sorted_chunks = sorted(candidate_chunks, key=lambda x: x["rerank_score"], reverse=True)
        return sorted_chunks[:top_k]
