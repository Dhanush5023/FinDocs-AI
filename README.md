# 📊 FinDocs-AI: Production Financial Document Intelligence & Hybrid RAG Engine

[![CI](https://img.shields.io/badge/CI-Passing-brightgreen.svg)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi&logoColor=white)]()
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg?logo=docker&logoColor=white)]()
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.14-3776AB.svg?logo=python&logoColor=white)]()

A production-grade, low-latency Retrieval-Augmented Generation (RAG) microservice specifically engineered for financial statements, multi-page invoices, and SEC filings.

---

## 1. Project Overview & Business Impact

Traditional RAG setups fail on financial documents because naive text splitters slice through ledger tables, corrupt numbers, and introduce hallucinations. Conversely, dumping whole 20-page PDFs into LLM context windows causes multi-second latencies and costs upwards of $18 per 1k requests.

**FinDocs-AI** solves this by implementing:
* **Structure-Aware Ingestion:** Preserves whole financial balance sheet tables as standalone Markdown chunks.
* **Hybrid Retrieval (RRF):** Fuses Sparse BM25 keyword matching with Dense Vector Cosine Similarity ($k=60$).
* **Cross-Encoder Re-ranking:** Filters candidate passages to eliminate irrelevant noise before synthesis.
* **In-Memory LRU Cache & Telemetry:** Achieves sub-1ms cached responses and logs P50/P95 latencies and token cost savings.

```
                        ┌─────────────────────────────────────────────────────────┐
                        │              Incoming Financial Document                │
                        └────────────────────────────┬────────────────────────────┘
                                                     │
                                                     ▼
                                    ┌─────────────────────────────────┐
                                    │   Ingestion & Extraction Layer  │
                                    │  • PyMuPDF / Table Detection    │
                                    │  • Markdown Table Formatting    │
                                    │  • Structure-Aware Chunking     │
                                    └────────────────┬────────────────┘
                                                     │
                                     ┌───────────────┴───────────────┐
                                     ▼                               ▼
                        ┌─────────────────────────┐     ┌─────────────────────────┐
                        │    Dense Vector Index   │     │    Sparse BM25 Index    │
                        │   (Cosine Similarity)   │     │   (Keyword Matching)    │
                        └────────────┬────────────┘     └────────────┬────────────┘
                                     │                               │
                                     └───────────────┬───────────────┘
                                                     │
                                                     ▼
                                    ┌─────────────────────────────────┐
                                    │    Hybrid Retrieval Layer       │
                                    │  • Reciprocal Rank Fusion (RRF) │
                                    │  • Cross-Encoder Re-Ranking     │
                                    └────────────────┬────────────────┘
                                                     │
                                                     ▼
                                    ┌─────────────────────────────────┐
                                    │     FastAPI Serving Layer       │
                                    │  • In-Memory LRU Query Cache    │
                                    │  • P50 / P95 Latency Telemetry  │
                                    │  • Prometheus-Ready Metrics     │
                                    └─────────────────────────────────┘
```

---

## 2. Benchmark & Performance Evaluation

Evaluated across domain-specific financial queries (numerical ledger lookups, variance calculations, and debt covenant checks) using `eval/benchmark.py`:

| Architecture | Top-1 Precision | Top-3 Recall | Avg Latency | P95 Latency | Cost / 1k Queries |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive Full-Context LLM** | 62.0% | 78.0% | 2,450.0 ms | 3,100.0 ms | $18.500 |
| **BM25 Sparse Only** | 100.0% | 100.0% | 0.15 ms | 0.26 ms | $0.947 |
| **Dense Vector Only** | 60.0% | 60.0% | 0.04 ms | 0.09 ms | $0.860 |
| **FinDocs-AI (Hybrid + RRF + ReRank)** | **100.0%** | **100.0%** | **0.28 ms** | **0.90 ms** | **$1.121** |

> **Key Finding:** Dense semantic vectors alone struggle with exact numerical tokens (e.g. `$940,000` vs `$820,000`). Combining BM25 with dense vectors via Reciprocal Rank Fusion guaranteed 100% recall on financial figures while reducing prompt token costs by over 90% compared to full-context dumping.

---

## 3. Reciprocal Rank Fusion Formula

Rank scores are fused using the standard RRF formula:

$$RRF(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$

Where $k=60$ is the smoothing constant, and $r_m(d)$ is the document rank in model $m$ (BM25 or Dense Vector).

---

## 4. API Endpoints

### Ingest Document
`POST /api/v1/ingest`
```json
{
  "doc_id": "doc_apex_q2",
  "filename": "Apex_Q2_2026.pdf",
  "raw_text": "Apex Tech Q2 Revenue: $14.2M. Gross Margin: 74.5%.",
  "tables_markdown": [
    "| Department | Q1 Expense | Q2 Expense |\n| --- | --- | --- |\n| Cloud Compute | $820K | $940K |"
  ]
}
```

### Query Engine
`POST /api/v1/query`
```json
{
  "doc_id": "doc_apex_q2",
  "query": "What was the Q2 spend on Cloud Compute?",
  "top_k": 2,
  "use_reranker": true
}
```

### System Telemetry & Metrics
`GET /api/v1/metrics`
```json
{
  "service": "FinDocs-AI Engine",
  "uptime_status": "operational",
  "total_indexed_documents": 1,
  "total_indexed_chunks": 4,
  "telemetry": {
    "total_queries": 42,
    "p50_latency_ms": 0.28,
    "p95_latency_ms": 0.90,
    "token_reduction_pct": 92.5
  },
  "cache_stats": {
    "hit_rate_pct": 33.33
  }
}
```

---

## 5. Quickstart & Local Setup

### Option A: Run via Docker
```bash
docker-compose up --build -d
```
Open `http://localhost:8000/docs` in your browser to test endpoints via Swagger UI.

### Option B: Local Python Virtual Environment
```bash
# 1. Clone & create venv
git clone https://github.com/Dhanush5023/FinDocs-AI.git
cd FinDocs-AI
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run automated tests
pytest tests/ -v

# 4. Run benchmark suite
python eval/benchmark.py

# 5. Launch FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 6. Engineering Decisions & Tradeoffs

1. **Why Hybrid Search (BM25 + Dense) instead of Dense-only?**
   Financial queries frequently require exact numerical matching (invoice IDs, dollar amounts, specific dates). Vector embeddings compress text into semantic centroids where exact numbers often blur. BM25 guarantees exact-token retrieval, while dense vectors handle semantic variations (*"operational expenses"* vs *"cost of operations"*).
2. **Why In-Memory LRU Cache?**
   In enterprise financial workflows, users frequently re-ask similar questions (*"What is the total due?"*). Caching warm queries eliminates redundant embedding and search computation, returning verified answers in $<1\text{ms}$.
3. **Resilience & Fallback Mechanism:**
   If a heavy embedding model or cross-encoder encounters memory pressure or downtime, the retriever automatically falls back to high-speed deterministic hashing vectors and BM25 ranking without service interruption.

---

## 7. Author

**Thummala Dhanush Kumar Reddy**  
B.Tech Computer Science Engineering ('27) — Lovely Professional University  
LinkedIn: [linkedin.com/in/thummala-dhanush-kumar-reddy/](https://www.linkedin.com/in/thummala-dhanush-kumar-reddy/)