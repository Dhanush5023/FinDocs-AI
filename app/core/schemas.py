from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class ExtractedTable(BaseModel):
    table_index: int
    page_number: int
    headers: List[str] = Field(default_factory=list)
    rows: List[List[str]] = Field(default_factory=list)
    markdown_repr: str
    row_count: int
    col_count: int

class ExtractedPage(BaseModel):
    page_number: int
    raw_text: str
    tables: List[ExtractedTable] = Field(default_factory=list)
    is_ocr_fallback: bool = False
    char_count: int

class DocumentChunk(BaseModel):
    chunk_id: str
    doc_id: str
    page_number: int
    chunk_type: str = Field(..., description="'narrative_text', 'financial_table', or 'metadata_kv'")
    content: str
    token_count_approx: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ExtractionResult(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    total_pages: int
    total_chunks: int
    total_tables: int
    extraction_latency_ms: float
    pages: List[ExtractedPage]
    chunks: List[DocumentChunk]
