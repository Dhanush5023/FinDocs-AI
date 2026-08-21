import re
from typing import List, Dict, Any
from app.core.schemas import ExtractedPage, DocumentChunk

class FinancialChunker:
    """
    Structure-aware chunker designed for financial statements, invoices, and contracts.
    Preserves whole tables as distinct units and chunks narrative text by semantic paragraphs.
    """
    def __init__(self, target_chunk_size: int = 400, chunk_overlap: int = 50):
        self.target_chunk_size = target_chunk_size
        self.chunk_overlap = chunk_overlap

    def _estimate_tokens(self, text: str) -> int:
        """Approximates token count (avg ~4 chars per token)."""
        return max(1, len(text) // 4)

    def chunk_document(self, doc_id: str, pages: List[ExtractedPage], filename: str) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        chunk_counter = 0

        for page in pages:
            page_num = page.page_number
            
            # 1. Process Extracted Tables first as standalone high-priority chunks
            for table in page.tables:
                table_content = (
                    f"### [DOCUMENT: {filename}] [PAGE {page_num}] [FINANCIAL TABLE]\n"
                    f"{table.markdown_repr}\n"
                    f"(Table dimensions: {table.row_count} rows, {table.col_count} columns)"
                )
                chunk_counter += 1
                chunks.append(DocumentChunk(
                    chunk_id=f"{doc_id}_p{page_num}_tbl{table.table_index}_{chunk_counter}",
                    doc_id=doc_id,
                    page_number=page_num,
                    chunk_type="financial_table",
                    content=table_content,
                    token_count_approx=self._estimate_tokens(table_content),
                    metadata={
                        "filename": filename,
                        "page_number": page_num,
                        "table_index": table.table_index,
                        "row_count": table.row_count,
                        "col_count": table.col_count
                    }
                ))

            # 2. Process Narrative Text (paragraphs & key-values)
            raw_text = page.raw_text
            if not raw_text:
                continue

            # Split text by double newlines or paragraph headers
            paragraphs = [p.strip() for p in re.split(r'\n{2,}', raw_text) if p.strip()]
            
            current_buffer = []
            current_tokens = 0

            for para in paragraphs:
                para_tokens = self._estimate_tokens(para)

                if current_tokens + para_tokens > self.target_chunk_size and current_buffer:
                    # Flush current chunk
                    combined_text = "\n\n".join(current_buffer)
                    chunk_counter += 1
                    formatted_content = f"### [DOCUMENT: {filename}] [PAGE {page_num}]\n{combined_text}"
                    chunks.append(DocumentChunk(
                        chunk_id=f"{doc_id}_p{page_num}_txt_{chunk_counter}",
                        doc_id=doc_id,
                        page_number=page_num,
                        chunk_type="narrative_text",
                        content=formatted_content,
                        token_count_approx=self._estimate_tokens(formatted_content),
                        metadata={
                            "filename": filename,
                            "page_number": page_num,
                            "is_ocr": page.is_ocr_fallback
                        }
                    ))
                    # Retain overlap from previous buffer if available
                    current_buffer = [current_buffer[-1]] if len(current_buffer) > 1 else []
                    current_tokens = self._estimate_tokens(current_buffer[0]) if current_buffer else 0

                current_buffer.append(para)
                current_tokens += para_tokens

            # Flush any remaining text for the page
            if current_buffer:
                combined_text = "\n\n".join(current_buffer)
                chunk_counter += 1
                formatted_content = f"### [DOCUMENT: {filename}] [PAGE {page_num}]\n{combined_text}"
                chunks.append(DocumentChunk(
                    chunk_id=f"{doc_id}_p{page_num}_txt_{chunk_counter}",
                    doc_id=doc_id,
                    page_number=page_num,
                    chunk_type="narrative_text",
                    content=formatted_content,
                    token_count_approx=self._estimate_tokens(formatted_content),
                    metadata={
                        "filename": filename,
                        "page_number": page_num,
                        "is_ocr": page.is_ocr_fallback
                    }
                ))

        return chunks
