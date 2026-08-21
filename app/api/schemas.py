from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class IngestDocumentRequest(BaseModel):
    doc_id: str = Field(..., json_schema_extra={"example": "doc_fin_2026_01"})
    filename: str = Field(..., json_schema_extra={"example": "Apex_Q2_Financial_Report.pdf"})
    raw_text: str = Field(..., json_schema_extra={"example": "Revenue: $2.4M. Net margin: 18%. AWS server cost: $45,000."})
    tables_markdown: Optional[List[str]] = Field(default=None, json_schema_extra={"example": ["| Quarter | Revenue | EBITDA |\n| --- | --- | --- |\n| Q1 | $1.8M | $320K |\n| Q2 | $2.4M | $480K |"]})

class IngestResponse(BaseModel):
    status: str
    doc_id: str
    filename: str
    chunks_indexed: int
    table_chunks: int
    narrative_chunks: int
    indexing_latency_ms: float

class QueryRequest(BaseModel):
    doc_id: Optional[str] = Field(None, description="Filter search to a specific document, or search across all indexed docs if omitted", json_schema_extra={"example": "doc_fin_2026_01"})
    query: str = Field(..., min_length=3, max_length=500, json_schema_extra={"example": "What was the total compute and cloud cost?"})
    top_k: int = Field(default=3, ge=1, le=10)
    use_reranker: bool = Field(default=True)

class RetrievedSource(BaseModel):
    chunk_id: str
    doc_id: str
    page_number: int
    chunk_type: str
    content: str
    token_estimate: int
    rrf_score: float
    rerank_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    dense_rank: Optional[int] = None

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[RetrievedSource]
    latency_ms: float
    cached: bool
    tokens_used: int
    estimated_cost_usd: float

class SystemMetricsResponse(BaseModel):
    service: str
    uptime_status: str
    total_indexed_documents: int
    total_indexed_chunks: int
    telemetry: Dict[str, Any]
    cache_stats: Dict[str, Any]