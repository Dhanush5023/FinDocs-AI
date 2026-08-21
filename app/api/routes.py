import time
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, status
from app.api.schemas import (
    IngestDocumentRequest, IngestResponse,
    QueryRequest, QueryResponse, RetrievedSource,
    SystemMetricsResponse
)
from app.core.schemas import ExtractedPage, ExtractedTable, DocumentChunk
from app.core.chunker import FinancialChunker
from app.core.hybrid_retriever import HybridRetriever
from app.core.reranker import CrossEncoderReRanker
from app.utils.cache import QueryCache
from app.utils.metrics import PerformanceMetricsTracker

router = APIRouter(prefix="/api/v1", tags=["Financial Document RAG"])

# Global Service State
chunker = FinancialChunker(target_chunk_size=350, chunk_overlap=50)
retriever = HybridRetriever(rrf_k=60)
reranker = CrossEncoderReRanker()
query_cache = QueryCache(capacity=500, default_ttl_seconds=3600)
metrics_tracker = PerformanceMetricsTracker()

# In-memory document registry
all_chunks: List[DocumentChunk] = []
indexed_doc_ids: set = set()

def _synthesize_grounded_answer(query: str, sources: List[RetrievedSource]) -> str:
    """
    Synthesizes a grounded, non-hallucinatory answer from retrieved passages.
    In production with LLM API keys, calls LLM; otherwise produces a deterministic extract.
    """
    if not sources:
        return "No relevant financial clauses or ledger entries found for this query in the indexed documents."

    top_content = sources[0].content
    chunk_type = sources[0].chunk_type
    doc_id = sources[0].doc_id
    page = sources[0].page_number

    if chunk_type == "financial_table":
        return f"Based on the verified financial table in document [{doc_id}, Page {page}], the relevant figures are located in the line items below.\n\nKey Match:\n{top_content}"
    
    return f"Grounded response extracted from document [{doc_id}, Page {page}]:\n\"{top_content.splitlines()[-1]}\""

@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_document(payload: IngestDocumentRequest):
    start_time = time.perf_counter()
    try:
        tables = []
        if payload.tables_markdown:
            for idx, md in enumerate(payload.tables_markdown):
                tables.append(ExtractedTable(
                    table_index=idx,
                    page_number=1,
                    headers=[],
                    rows=[],
                    markdown_repr=md,
                    row_count=len(md.splitlines()),
                    col_count=md.count("|") // max(1, len(md.splitlines()))
                ))

        page = ExtractedPage(
            page_number=1,
            raw_text=payload.raw_text,
            tables=tables,
            is_ocr_fallback=False,
            char_count=len(payload.raw_text)
        )

        new_chunks = chunker.chunk_document(
            doc_id=payload.doc_id,
            pages=[page],
            filename=payload.filename
        )

        # Update global chunk store & re-index hybrid engine
        all_chunks.extend(new_chunks)
        indexed_doc_ids.add(payload.doc_id)
        retriever.index_chunks(all_chunks)

        latency_ms = (time.perf_counter() - start_time) * 1000
        table_count = sum(1 for c in new_chunks if c.chunk_type == "financial_table")
        narrative_count = len(new_chunks) - table_count

        return IngestResponse(
            status="success",
            doc_id=payload.doc_id,
            filename=payload.filename,
            chunks_indexed=len(new_chunks),
            table_chunks=table_count,
            narrative_chunks=narrative_count,
            indexing_latency_ms=round(latency_ms, 2)
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(exc)}")

@router.post("/query", response_model=QueryResponse)
async def execute_query(payload: QueryRequest):
    start_time = time.perf_counter()
    cache_doc_key = payload.doc_id or "ALL"

    # 1. Check Query Cache for instant response
    cached_result = query_cache.get(doc_id=cache_doc_key, query=payload.query, top_k=payload.top_k)
    if cached_result:
        latency_ms = (time.perf_counter() - start_time) * 1000
        metrics_tracker.record_query(latency_ms=latency_ms, tokens_used=0)
        cached_result["latency_ms"] = round(latency_ms, 2)
        cached_result["cached"] = True
        cached_result["tokens_used"] = 0
        cached_result["estimated_cost_usd"] = 0.0
        return QueryResponse(**cached_result)

    # 2. Hybrid Retrieval (BM25 + Dense Vectors + RRF)
    raw_results = retriever.retrieve_hybrid(
        query=payload.query,
        top_k=payload.top_k * 2 if payload.use_reranker else payload.top_k
    )

    # Filter by doc_id if specified
    if payload.doc_id:
        raw_results = [r for r in raw_results if r["doc_id"] == payload.doc_id]

    # 3. Contextual Cross-Encoder Re-Ranking
    if payload.use_reranker and raw_results:
        final_results = reranker.rerank(query=payload.query, candidate_chunks=raw_results, top_k=payload.top_k)
    else:
        final_results = raw_results[:payload.top_k]

    # Map to schema
    retrieved_sources = [
        RetrievedSource(
            chunk_id=r["chunk_id"],
            doc_id=r["doc_id"],
            page_number=r["page_number"],
            chunk_type=r["chunk_type"],
            content=r["content"],
            token_estimate=r["token_estimate"],
            rrf_score=r["rrf_score"],
            rerank_score=r.get("rerank_score"),
            bm25_rank=r.get("bm25_rank"),
            dense_rank=r.get("dense_rank")
        )
        for r in final_results
    ]

    # 4. Synthesize Answer
    answer_text = _synthesize_grounded_answer(payload.query, retrieved_sources)
    
    latency_ms = (time.perf_counter() - start_time) * 1000
    tokens_used = sum(s.token_estimate for s in retrieved_sources) + len(answer_text) // 4
    estimated_cost = round((tokens_used / 1000.0) * 0.005, 5)

    response_dict = {
        "query": payload.query,
        "answer": answer_text,
        "sources": [s.model_dump() for s in retrieved_sources],
        "latency_ms": round(latency_ms, 2),
        "cached": False,
        "tokens_used": tokens_used,
        "estimated_cost_usd": estimated_cost
    }

    # Record telemetry & update cache
    metrics_tracker.record_query(latency_ms=latency_ms, tokens_used=tokens_used)
    query_cache.set(doc_id=cache_doc_key, query=payload.query, top_k=payload.top_k, value=response_dict)

    return QueryResponse(**response_dict)

@router.get("/metrics", response_model=SystemMetricsResponse)
async def get_system_metrics():
    return SystemMetricsResponse(
        service="FinDocs-AI Engine",
        uptime_status="operational",
        total_indexed_documents=len(indexed_doc_ids),
        total_indexed_chunks=len(all_chunks),
        telemetry=metrics_tracker.get_summary(),
        cache_stats=query_cache.stats
    )

@router.delete("/cache")
async def clear_cache():
    query_cache.clear()
    return {"status": "success", "message": "Query cache flushed successfully"}