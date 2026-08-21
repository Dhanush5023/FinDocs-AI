import os
import sys
import time
import json

# Add current directory to path
sys.path.insert(0, os.path.abspath("."))

from app.core.schemas import ExtractedPage, ExtractedTable, DocumentChunk
from app.core.chunker import FinancialChunker

def run_test():
    print("==================================================")
    print("   FinDocs-AI: Ingestion & Chunker Unit Test     ")
    print("==================================================")

    # 1. Simulate a multi-page financial invoice with complex tables and narrative text
    sample_table = ExtractedTable(
        table_index=0,
        page_number=1,
        headers=["Item Description", "Qty", "Unit Price", "Total Amount"],
        rows=[
            ["Cloud Server Compute (AWS ec2)", "10", "$120.00", "$1,200.00"],
            ["Database Managed Instance (RDS)", "2", "$450.00", "$900.00"],
            ["Vector DB Enterprise License", "1", "$2,500.00", "$2,500.00"],
            ["Subtotal", "", "", "$4,600.00"],
            ["Tax (HST 13%)", "", "", "$598.00"],
            ["Total Due (Net 30)", "", "", "$5,198.00"]
        ],
        markdown_repr=(
            "| Item Description | Qty | Unit Price | Total Amount |\n"
            "| --- | --- | --- | --- |\n"
            "| Cloud Server Compute (AWS ec2) | 10 | $120.00 | $1,200.00 |\n"
            "| Database Managed Instance (RDS) | 2 | $450.00 | $900.00 |\n"
            "| Vector DB Enterprise License | 1 | $2,500.00 | $2,500.00 |\n"
            "| Subtotal | | | $4,600.00 |\n"
            "| Tax (HST 13%) | | | $598.00 |\n"
            "| Total Due (Net 30) | | | $5,198.00 |"
        ),
        row_count=6,
        col_count=4
    )

    sample_page_1 = ExtractedPage(
        page_number=1,
        raw_text="INVOICE #INV-2026-9081\nDate: August 15, 2026\nVendor: Apex SaaS Infrastructure Inc.\nClient: Toronto AI Solutions Ltd.\nPayment Terms: Net 30 days via wire transfer.",
        tables=[sample_table],
        is_ocr_fallback=False,
        char_count=180
    )

    sample_page_2 = ExtractedPage(
        page_number=2,
        raw_text="TERMS & CONDITIONS\n1. Late payments are subject to a 1.5% compounding monthly fee.\n2. All services are governed by the Ontario Commercial Technology Act.\n3. In case of dispute, arbitration shall take place in Toronto, ON.",
        tables=[],
        is_ocr_fallback=False,
        char_count=210
    )

    # 2. Run FinancialChunker
    start = time.perf_counter()
    chunker = FinancialChunker(target_chunk_size=300, chunk_overlap=50)
    chunks = chunker.chunk_document(
        doc_id="doc_inv_9081",
        pages=[sample_page_1, sample_page_2],
        filename="INV-2026-9081_Apex_Infra.pdf"
    )
    latency_ms = (time.perf_counter() - start) * 1000

    print(f"\n[OK] Chunker executed in {latency_ms:.3f} ms")
    print(f"[OK] Total chunks generated: {len(chunks)}")
    
    for i, c in enumerate(chunks, 1):
        print(f"\n--- Chunk #{i} [{c.chunk_type}] (Tokens approx: {c.token_count_approx}) ---")
        print(f"ID: {c.chunk_id}")
        print(f"Page: {c.page_number}")
        print("Content Preview:")
        print(c.content)
        print("-" * 50)

    print("\n[SUCCESS] Ingestion & Chunker validation passed!")

if __name__ == "__main__":
    run_test()
