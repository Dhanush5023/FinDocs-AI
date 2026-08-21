import os
import sys
import time
from tabulate import tabulate

# Add repo root to path
sys.path.insert(0, os.path.abspath("."))

from app.core.schemas import ExtractedPage, ExtractedTable, DocumentChunk
from app.core.chunker import FinancialChunker
from app.core.hybrid_retriever import HybridRetriever
from app.core.reranker import CrossEncoderReRanker

# Benchmark ground-truth evaluation dataset
BENCHMARK_DOCUMENTS = [
    {
        "doc_id": "doc_apex_q2_2026",
        "filename": "Apex_Technologies_Q2_2026.pdf",
        "pages": [
            ExtractedPage(
                page_number=1,
                raw_text="Apex Technologies Inc. Q2 2026 Financial Report.\nTotal Revenue reached $14.2M, representing a 28% YoY growth.\nGross margin stood at 74.5% compared to 71.0% in Q1.\nOperating cash flow increased to $3.1M.",
                tables=[],
                is_ocr_fallback=False,
                char_count=210
            ),
            ExtractedPage(
                page_number=2,
                raw_text="Operating Expense Breakdown for Q2 2026:\nResearch & Development: $4.2M\nSales & Marketing: $3.8M\nGeneral & Administrative: $1.9M\nTotal Operating Expenses: $9.9M.",
                tables=[
                    ExtractedTable(
                        table_index=0,
                        page_number=2,
                        headers=["Department", "Q1 Expense", "Q2 Expense", "Variance %"],
                        rows=[
                            ["Cloud Infrastructure (AWS/GCP)", "$820,000", "$940,000", "+14.6%"],
                            ["AI Model Training Compute", "$450,000", "$680,000", "+51.1%"],
                            ["Personnel & Payroll", "$2,100,000", "$2,350,000", "+11.9%"],
                            ["Office & Facilities", "$150,000", "$140,000", "-6.6%"],
                            ["Total Core Operations", "$3,520,000", "$4,110,000", "+16.7%"]
                        ],
                        markdown_repr=(
                            "| Department | Q1 Expense | Q2 Expense | Variance % |\n"
                            "| --- | --- | --- | --- |\n"
                            "| Cloud Infrastructure (AWS/GCP) | $820,000 | $940,000 | +14.6% |\n"
                            "| AI Model Training Compute | $450,000 | $680,000 | +51.1% |\n"
                            "| Personnel & Payroll | $2,100,000 | $2,350,000 | +11.9% |\n"
                            "| Office & Facilities | $150,000 | $140,000 | -6.6% |\n"
                            "| Total Core Operations | $3,520,000 | $4,110,000 | +16.7% |"
                        ),
                        row_count=5,
                        col_count=4
                    )
                ],
                is_ocr_fallback=False,
                char_count=320
            ),
            ExtractedPage(
                page_number=3,
                raw_text="LEGAL & DEBT OBLIGATIONS\nCredit facility with Silicon Valley Bank: $5.0M principal at SOFR + 2.5%.\nMaturity date is December 31, 2028. No covenants breached during Q2.",
                tables=[],
                is_ocr_fallback=False,
                char_count=190
            )
        ]
    }
]

BENCHMARK_QUERIES = [
    {
        "query": "How much was spent on AI model training compute in Q2?",
        "expected_keywords": ["$680,000", "model training compute"],
        "target_chunk_type": "financial_table"
    },
    {
        "query": "What was the total revenue and YoY growth in Q2 2026?",
        "expected_keywords": ["$14.2m", "28%"],
        "target_chunk_type": "narrative_text"
    },
    {
        "query": "What are the terms and interest rate for the Silicon Valley Bank credit facility?",
        "expected_keywords": ["sofr + 2.5%", "$5.0m", "december 31, 2028"],
        "target_chunk_type": "narrative_text"
    },
    {
        "query": "What was the variance percentage for Cloud Infrastructure AWS GCP?",
        "expected_keywords": ["+14.6%", "$940,000"],
        "target_chunk_type": "financial_table"
    },
    {
        "query": "What were the total operating expenses and R&D spend?",
        "expected_keywords": ["$9.9m", "$4.2m"],
        "target_chunk_type": "narrative_text"
    }
]

def run_benchmark():
    print("==================================================================")
    print("      FinDocs-AI: Production RAG Performance Benchmark Suite     ")
    print("==================================================================")

    # 1. Ingestion & Chunking
    chunker = FinancialChunker(target_chunk_size=350, chunk_overlap=50)
    all_chunks = []
    for doc in BENCHMARK_DOCUMENTS:
        chunks = chunker.chunk_document(doc_id=doc["doc_id"], pages=doc["pages"], filename=doc["filename"])
        all_chunks.extend(chunks)

    print(f"\n[INFO] Indexed {len(BENCHMARK_DOCUMENTS)} document(s) into {len(all_chunks)} chunks.")

    retriever = HybridRetriever(rrf_k=60)
    retriever.index_chunks(all_chunks)
    reranker = CrossEncoderReRanker()

    # Trackers for 3 architectures
    results = {
        "BM25 Sparse Only": {"hits_top1": 0, "hits_top3": 0, "latencies": [], "tokens": 0},
        "Dense Vector Only": {"hits_top1": 0, "hits_top3": 0, "latencies": [], "tokens": 0},
        "FinDocs-AI (Hybrid + RRF + ReRank)": {"hits_top1": 0, "hits_top3": 0, "latencies": [], "tokens": 0}
    }

    for item in BENCHMARK_QUERIES:
        q = item["query"]
        expected = [k.lower() for k in item["expected_keywords"]]

        # 1. BM25 Only
        t0 = time.perf_counter()
        bm25_res = retriever._search_sparse(q, top_k=3)
        lat_bm25 = (time.perf_counter() - t0) * 1000
        results["BM25 Sparse Only"]["latencies"].append(lat_bm25)
        if bm25_res:
            c0 = bm25_res[0][0].content.lower()
            if any(k in c0 for k in expected):
                results["BM25 Sparse Only"]["hits_top1"] += 1
            if any(any(k in r[0].content.lower() for k in expected) for r in bm25_res):
                results["BM25 Sparse Only"]["hits_top3"] += 1
            results["BM25 Sparse Only"]["tokens"] += sum(r[0].token_count_approx for r in bm25_res)

        # 2. Dense Vector Only
        t0 = time.perf_counter()
        dense_res = retriever._search_dense(q, top_k=3)
        lat_dense = (time.perf_counter() - t0) * 1000
        results["Dense Vector Only"]["latencies"].append(lat_dense)
        if dense_res:
            c0 = dense_res[0][0].content.lower()
            if any(k in c0 for k in expected):
                results["Dense Vector Only"]["hits_top1"] += 1
            if any(any(k in r[0].content.lower() for k in expected) for r in dense_res):
                results["Dense Vector Only"]["hits_top3"] += 1
            results["Dense Vector Only"]["tokens"] += sum(r[0].token_count_approx for r in dense_res)

        # 3. FinDocs-AI Hybrid + ReRanker
        t0 = time.perf_counter()
        hybrid_candidates = retriever.retrieve_hybrid(q, top_k=6)
        final_res = reranker.rerank(q, hybrid_candidates, top_k=3)
        lat_hybrid = (time.perf_counter() - t0) * 1000
        results["FinDocs-AI (Hybrid + RRF + ReRank)"]["latencies"].append(lat_hybrid)
        if final_res:
            c0 = final_res[0]["content"].lower()
            if any(k in c0 for k in expected):
                results["FinDocs-AI (Hybrid + RRF + ReRank)"]["hits_top1"] += 1
            if any(any(k in r["content"].lower() for k in expected) for r in final_res):
                results["FinDocs-AI (Hybrid + RRF + ReRank)"]["hits_top3"] += 1
            results["FinDocs-AI (Hybrid + RRF + ReRank)"]["tokens"] += sum(r["token_estimate"] for r in final_res)

    # Compile Table
    total_q = len(BENCHMARK_QUERIES)
    table_data = []

    for name, data in results.items():
        top1_acc = f"{(data['hits_top1'] / total_q) * 100:.1f}%"
        top3_acc = f"{(data['hits_top3'] / total_q) * 100:.1f}%"
        avg_lat = f"{sum(data['latencies']) / len(data['latencies']):.2f} ms"
        p95_lat = f"{sorted(data['latencies'])[int(len(data['latencies']) * 0.95)]:.2f} ms"
        cost_per_1k = f"${(data['tokens'] / (total_q * 1000)) * 0.005 * 1000:.3f}"
        
        table_data.append([
            name,
            top1_acc,
            top3_acc,
            avg_lat,
            p95_lat,
            cost_per_1k
        ])

    headers = ["Architecture", "Top-1 Precision", "Top-3 Recall", "Avg Latency", "P95 Latency", "Cost / 1k Queries"]
    print("\n" + tabulate(table_data, headers=headers, tablefmt="github"))
    print("\n[SUCCESS] Benchmark completed successfully.")

if __name__ == "__main__":
    run_benchmark()
