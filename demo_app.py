import time
import streamlit as st
from app.core.schemas import ExtractedPage, ExtractedTable
from app.core.chunker import FinancialChunker
from app.core.hybrid_retriever import HybridRetriever
from app.core.reranker import CrossEncoderReRanker
from app.utils.cache import QueryCache
from app.utils.metrics import PerformanceMetricsTracker

st.set_page_config(
    page_title="FinDocs-AI: Production Financial RAG Engine",
    page_icon="📊",
    layout="wide"
)

# Initialize Session State Singletons
if "retriever" not in st.session_state:
    st.session_state.retriever = HybridRetriever(rrf_k=60)
    st.session_state.reranker = CrossEncoderReRanker()
    st.session_state.cache = QueryCache(capacity=200)
    st.session_state.metrics = PerformanceMetricsTracker()
    st.session_state.chunker = FinancialChunker(target_chunk_size=350)
    st.session_state.indexed_docs = {}

st.title("📊 FinDocs-AI: Production Financial Document RAG Engine")
st.caption("Low-Latency Hybrid Search (BM25 + Dense Vectors + RRF + Cross-Encoder) for Financial Statements & Invoices")

# Sidebar: Document Ingestion & System Telemetry
with st.sidebar:
    st.header("⚡ System Telemetry & HUD")
    summary = st.session_state.metrics.get_summary()
    c_stats = st.session_state.cache.stats

    col_m1, col_m2 = st.columns(2)
    col_m1.metric("P95 Latency", f"{summary.get('p95_latency_ms', 0.0)} ms")
    col_m2.metric("Cache Hit Rate", f"{c_stats.get('hit_rate_pct', 0.0)}%")

    col_m3, col_m4 = st.columns(2)
    col_m3.metric("Cost Savings", f"${summary.get('estimated_savings_vs_naive_usd', 0.0)}")
    col_m4.metric("Token Reduction", f"{summary.get('token_reduction_pct', 0.0)}%")

    st.markdown("---")
    st.header("📥 Ingest Sample Invoices")
    
    if st.button("Load Apex Q2 Report & Invoices"):
        sample_pages = [
            ExtractedPage(
                page_number=1,
                raw_text="Apex Technologies Inc. Q2 2026 Financial Report.\nTotal Revenue: $14.2M (+28% YoY).\nGross Margin: 74.5%.\nOperating cash flow: $3.1M.",
                tables=[],
                char_count=180
            ),
            ExtractedPage(
                page_number=2,
                raw_text="Operating Expense Breakdown for Q2 2026:\nR&D: $4.2M, Sales: $3.8M, G&A: $1.9M.",
                tables=[
                    ExtractedTable(
                        table_index=0,
                        page_number=2,
                        headers=["Department", "Q1 Expense", "Q2 Expense", "Variance %"],
                        rows=[
                            ["Cloud Infrastructure (AWS)", "$820,000", "$940,000", "+14.6%"],
                            ["AI Model Training Compute", "$450,000", "$680,000", "+51.1%"],
                            ["Personnel & Payroll", "$2,100,000", "$2,350,000", "+11.9%"]
                        ],
                        markdown_repr=(
                            "| Department | Q1 Expense | Q2 Expense | Variance % |\n"
                            "| --- | --- | --- | --- |\n"
                            "| Cloud Infrastructure (AWS) | $820,000 | $940,000 | +14.6% |\n"
                            "| AI Model Training Compute | $450,000 | $680,000 | +51.1% |\n"
                            "| Personnel & Payroll | $2,100,000 | $2,350,000 | +11.9% |"
                        ),
                        row_count=3,
                        col_count=4
                    )
                ],
                char_count=290
            )
        ]
        chunks = st.session_state.chunker.chunk_document("doc_apex_q2", sample_pages, "Apex_Q2_Report.pdf")
        st.session_state.retriever.index_chunks(chunks)
        st.session_state.indexed_docs["Apex_Q2_Report.pdf"] = len(chunks)
        st.success(f"Indexed {len(chunks)} chunks into Hybrid Index!")

    st.write(f"**Indexed Documents:** {len(st.session_state.indexed_docs)}")
    for name, cnt in st.session_state.indexed_docs.items():
        st.text(f"• {name} ({cnt} chunks)")

# Main Query Interface
st.subheader("🔍 Financial Query & Clause Search")
sample_queries = [
    "How much was spent on AI model training compute in Q2?",
    "What was the total revenue and YoY growth in Q2 2026?",
    "What was the variance percentage for Cloud Infrastructure AWS?"
]

selected_query = st.selectbox("Or choose a pre-set benchmark query:", [""] + sample_queries)
user_query = st.text_input("Enter your financial / ledger query:", value=selected_query)

col_ctrl1, col_ctrl2 = st.columns([1, 1])
top_k = col_ctrl1.slider("Top-K Passages", min_value=1, max_value=5, value=2)
use_reranker = col_ctrl2.checkbox("Enable Cross-Encoder Re-Ranker", value=True)

if st.button("Execute Search", type="primary"):
    if not user_query.strip():
        st.warning("Please enter a valid query.")
    elif not st.session_state.indexed_docs:
        st.error("Please click 'Load Apex Q2 Report & Invoices' in the sidebar first.")
    else:
        t0 = time.perf_counter()
        
        # Check Cache
        cached_result = st.session_state.cache.get("ALL", user_query, top_k)
        if cached_result:
            lat = (time.perf_counter() - t0) * 1000
            st.session_state.metrics.record_query(lat, 0)
            st.info(f"⚡ Instant Response from LRU Cache (Latency: {lat:.2f} ms)")
            results = cached_result["sources"]
            answer = cached_result["answer"]
        else:
            raw_res = st.session_state.retriever.retrieve_hybrid(user_query, top_k=top_k * 2 if use_reranker else top_k)
            if use_reranker:
                results = st.session_state.reranker.rerank(user_query, raw_res, top_k=top_k)
            else:
                results = raw_res[:top_k]

            lat = (time.perf_counter() - t0) * 1000
            tokens = sum(r["token_estimate"] for r in results)
            st.session_state.metrics.record_query(lat, tokens)
            
            top_content = results[0]["content"] if results else "No matches found."
            answer = f"Relevant financial extract:\n\n{top_content}"
            st.session_state.cache.set("ALL", user_query, top_k, {"sources": results, "answer": answer})
            st.success(f"Retrieved {len(results)} grounded passages in {lat:.2f} ms")

        st.markdown("### 📋 Grounded Extracted Result")
        st.markdown(answer)

        st.markdown("### 📑 Retrieved Source Passages & Ranking Scores")
        for i, src in enumerate(results, 1):
            with st.expander(f"Source #{i} [{src.get('chunk_type')}] — RRF Score: {src.get('rrf_score', 0):.4f} (Page {src.get('page_number')})"):
                st.markdown(src.get("content"))
                st.caption(f"BM25 Rank: {src.get('bm25_rank')} | Dense Rank: {src.get('dense_rank')} | Approx Tokens: {src.get('token_estimate')}")