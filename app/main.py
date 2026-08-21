from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as api_router

app = FastAPI(
    title="FinDocs-AI: Production Financial Document RAG Engine",
    description="Low-latency hybrid search (BM25 + Dense Vectors + RRF + Cross-Encoder) for financial statements, invoices, and contracts.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

@app.get("/", tags=["Root"])
async def root():
    return {
        "service": "FinDocs-AI Engine",
        "status": "online",
        "docs": "/docs",
        "health": "/health",
        "version": "1.0.0"
    }

@app.get("/health", tags=["Health"], status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "healthy", "service": "findocs-ai-engine"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
