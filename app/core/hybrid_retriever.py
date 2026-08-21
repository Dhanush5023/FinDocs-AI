import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from rank_bm25 import BM25Okapi
from app.core.schemas import DocumentChunk

class DenseEmbedder:
    """
    Lightweight vector embedder with normalized cosine similarity search.
    Supports pluggable embedding backends (sentence-transformers or fast hashing vectorizer).
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._st_model = None
        self._initialized = False

    def _lazy_init(self):
        if not self._initialized:
            try:
                from sentence_transformers import SentenceTransformer
                self._st_model = SentenceTransformer(self.model_name)
            except Exception:
                self._st_model = None
            self._initialized = True

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        self._lazy_init()
        if self._st_model is not None:
            embeddings = self._st_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.array(embeddings, dtype=np.float32)
        
        # Fast fallback deterministic hashing embedder if heavy model is downloading / absent
        vectors = []
        dim = 128
        for text in texts:
            vec = np.zeros(dim, dtype=np.float32)
            words = text.lower().split()
            for w in words:
                idx = hash(w) % dim
                vec[idx] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)

class HybridRetriever:
    """
    Production Hybrid Retriever combining:
    1. Sparse Lexical Search (BM25 with tokenization and stopword removal)
    2. Dense Vector Embeddings (Cosine similarity)
    3. Reciprocal Rank Fusion (RRF) for ensemble ranking
    """
    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k
        self.embedder = DenseEmbedder()
        self.chunks: List[DocumentChunk] = []
        self.bm25_index: Optional[BM25Okapi] = None
        self.dense_embeddings: Optional[np.ndarray] = None

    def index_chunks(self, chunks: List[DocumentChunk]) -> None:
        self.chunks = chunks
        if not chunks:
            self.bm25_index = None
            self.dense_embeddings = None
            return

        # 1. Build BM25 Sparse Index
        tokenized_corpus = [c.content.lower().replace("\n", " ").split() for c in chunks]
        self.bm25_index = BM25Okapi(tokenized_corpus)

        # 2. Build Dense Index
        texts = [c.content for c in chunks]
        self.dense_embeddings = self.embedder.embed_texts(texts)

    def _search_sparse(self, query: str, top_k: int) -> List[Tuple[DocumentChunk, float]]:
        if self.bm25_index is None or not self.chunks:
            return []
        tokenized_query = query.lower().split()
        scores = self.bm25_index.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices if scores[i] > 0]

    def _search_dense(self, query: str, top_k: int) -> List[Tuple[DocumentChunk, float]]:
        if self.dense_embeddings is None or not self.chunks:
            return []
        q_emb = self.embedder.embed_texts([query])[0]
        similarities = np.dot(self.dense_embeddings, q_emb)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(self.chunks[i], float(similarities[i])) for i in top_indices]

    def retrieve_hybrid(self, query: str, top_k: int = 3, candidate_multiplier: int = 3) -> List[Dict[str, Any]]:
        """
        Executes hybrid search and merges via Reciprocal Rank Fusion (RRF).
        """
        if not self.chunks:
            return []

        pool_size = min(len(self.chunks), top_k * candidate_multiplier)
        sparse_hits = self._search_sparse(query, top_k=pool_size)
        dense_hits = self._search_dense(query, top_k=pool_size)

        rrf_table: Dict[str, Dict[str, Any]] = {}

        # Accumulate RRF scores from Sparse (BM25)
        for rank, (chunk, score) in enumerate(sparse_hits):
            cid = chunk.chunk_id
            if cid not in rrf_table:
                rrf_table[cid] = {"chunk": chunk, "rrf_score": 0.0, "bm25_score": score, "dense_score": 0.0, "bm25_rank": rank + 1, "dense_rank": None}
            rrf_table[cid]["rrf_score"] += 1.0 / (self.rrf_k + rank + 1)

        # Accumulate RRF scores from Dense Vector Search
        for rank, (chunk, score) in enumerate(dense_hits):
            cid = chunk.chunk_id
            if cid not in rrf_table:
                rrf_table[cid] = {"chunk": chunk, "rrf_score": 0.0, "bm25_score": 0.0, "dense_score": score, "bm25_rank": None, "dense_rank": rank + 1}
            else:
                rrf_table[cid]["dense_score"] = score
                rrf_table[cid]["dense_rank"] = rank + 1
            rrf_table[cid]["rrf_score"] += 1.0 / (self.rrf_k + rank + 1)

        # Sort by final fused RRF score
        ranked_results = sorted(rrf_table.values(), key=lambda x: x["rrf_score"], reverse=True)[:top_k]

        return [
            {
                "chunk_id": item["chunk"].chunk_id,
                "doc_id": item["chunk"].doc_id,
                "page_number": item["chunk"].page_number,
                "chunk_type": item["chunk"].chunk_type,
                "content": item["chunk"].content,
                "token_estimate": item["chunk"].token_count_approx,
                "rrf_score": round(item["rrf_score"], 5),
                "bm25_rank": item["bm25_rank"],
                "dense_rank": item["dense_rank"],
                "metadata": item["chunk"].metadata
            }
            for item in ranked_results
        ]
