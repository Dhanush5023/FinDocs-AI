from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "findocs-ai-engine"}

def test_ingest_document():
    payload = {
        "doc_id": "test_invoice_001",
        "filename": "AWS_Compute_Invoice_July.pdf",
        "raw_text": "Invoice #AWS-9901. Vendor: Amazon Web Services. Total Due: $1,450.00. Due Date: August 30, 2026. Payment terms: Net 15.",
        "tables_markdown": [
            "| Service | Region | Usage Hrs | Cost |\n| --- | --- | --- | --- |\n| EC2 c5.2xlarge | us-east-1 | 720 | $720.00 |\n| RDS PostgreSQL | us-east-1 | 720 | $730.00 |"
        ]
    }
    response = client.post("/api/v1/ingest", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "success"
    assert data["doc_id"] == "test_invoice_001"
    assert data["chunks_indexed"] >= 2
    assert data["table_chunks"] == 1

def test_query_and_cache_cycle():
    # Query 1: Uncached cold query
    query_payload = {
        "doc_id": "test_invoice_001",
        "query": "What is the cost of RDS PostgreSQL?",
        "top_k": 2,
        "use_reranker": True
    }
    r1 = client.post("/api/v1/query", json=query_payload)
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["cached"] is False
    assert len(d1["sources"]) > 0
    assert d1["tokens_used"] > 0

    # Query 2: Repeated warm query -> Must hit LRU Cache
    r2 = client.post("/api/v1/query", json=query_payload)
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["cached"] is True
    assert d2["tokens_used"] == 0

def test_system_telemetry_metrics():
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["uptime_status"] == "operational"
    assert data["total_indexed_documents"] >= 1
    assert "telemetry" in data
    assert "cache_stats" in data
    assert data["cache_stats"]["hits"] >= 1
