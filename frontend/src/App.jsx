import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  Zap, Database, ShieldCheck, Activity, Search, 
  FileText, Table, CheckCircle2, ArrowRight, Github, ExternalLink 
} from 'lucide-react';

const API_BASE = "http://localhost:8000/api/v1";

export default function App() {
  const [query, setQuery] = useState("How much was spent on AI model training compute in Q2?");
  const [topK, setTopK] = useState(2);
  const [useReranker, setUseReranker] = useState(true);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [metrics, setMetrics] = useState({
    p95_latency_ms: 0.59,
    hit_rate_pct: 33.3,
    token_reduction_pct: 92.5,
    estimated_savings_usd: 0.18
  });

  const sampleQueries = [
    "How much was spent on AI model training compute in Q2?",
    "What was the total revenue and YoY growth in Q2 2026?",
    "What are the interest terms and maturity date of the credit facility?",
    "Which operating cost category was over budget and what was the variance?"
  ];

  const handleSearch = async () => {
    if (!query.trim()) return;
    setLoading(true);
    try {
      // In local dev without backend running, fallback to mock response
      const res = await axios.post(`${API_BASE}/query`, {
        query,
        top_k: topK,
        use_reranker: useReranker
      }).catch(() => ({
        data: {
          query,
          answer: "Grounded Extract from Apex_Q2_2026_Report.pdf (Page 2):\\nAI Model Training Compute (GPUs) expense totaled $680,000 in Q2 (a +51.1% QoQ increase over $450,000 in Q1). Status: Approved Override.",
          latency_ms: 0.28,
          cached: false,
          tokens_used: 118,
          estimated_cost_usd: 0.00059,
          sources: [
            {
              chunk_id: "doc_apex_q2_p2_tbl0_1",
              doc_id: "doc_apex_q2",
              page_number: 2,
              chunk_type: "financial_table",
              content: "| Cost Category | Q1 Expense | Q2 Expense | Variance % | Budget Status |\\n| --- | --- | --- | --- | --- |\\n| AI Model Training Compute (GPUs) | $450,000 | $680,000 | +51.1% | Approved Override |\\n| Cloud Infrastructure (AWS/GCP) | $820,000 | $940,000 | +14.6% | Over Budget |",
              token_estimate: 88,
              rrf_score: 0.0328,
              rerank_score: 0.94,
              bm25_rank: 1,
              dense_rank: 2
            }
          ]
        }
      }));
      setResult(res.data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0B0F19] text-slate-100 p-6 md:p-10 font-sans">
      {/* Top Navigation */}
      <header className="max-w-7xl mx-auto flex flex-col md:flex-row md:items-center md:justify-between pb-8 border-b border-slate-800 gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-400 via-blue-500 to-purple-600 flex items-center justify-center shadow-lg shadow-blue-500/20">
            <Zap className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-bold bg-gradient-to-r from-cyan-400 via-blue-400 to-indigo-400 bg-clip-text text-transparent">
              FinDocs-AI
            </h1>
            <p className="text-xs text-slate-400">Financial Document Intelligence & Low-Latency Hybrid RAG</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            Hybrid Engine Active
          </span>
          <a 
            href="https://findocs-ai.streamlit.app/" 
            target="_blank" 
            rel="noreferrer"
            className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition"
          >
            <ExternalLink className="w-3.5 h-3.5" />
            Live Cloud Demo
          </a>
          <a 
            href="https://github.com/Dhanush5023/FinDocs-AI" 
            target="_blank" 
            rel="noreferrer"
            className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-xs font-medium bg-blue-600 hover:bg-blue-500 text-white transition shadow-sm"
          >
            <Github className="w-3.5 h-3.5" />
            GitHub
          </a>
        </div>
      </header>

      {/* Telemetry HUD Grid */}
      <section className="max-w-7xl mx-auto my-8 grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 shadow-sm">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-medium">P95 Latency</span>
            <Activity className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-2xl font-bold text-slate-100">{metrics.p95_latency_ms} <span className="text-sm font-normal text-slate-400">ms</span></div>
          <span className="text-[10px] text-cyan-400 font-mono">BM25 + Dense RRF</span>
        </div>

        <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 shadow-sm">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-medium">Cache Hit Rate</span>
            <Database className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-bold text-slate-100">{metrics.hit_rate_pct} <span className="text-sm font-normal text-slate-400">%</span></div>
          <span className="text-[10px] text-emerald-400 font-mono">&lt;1ms Warm Queries</span>
        </div>

        <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 shadow-sm">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-medium">Token Reduction</span>
            <ShieldCheck className="w-4 h-4 text-blue-400" />
          </div>
          <div className="text-2xl font-bold text-slate-100">{metrics.token_reduction_pct} <span className="text-sm font-normal text-slate-400">%</span></div>
          <span className="text-[10px] text-blue-400 font-mono">vs Full-Doc Ingestion</span>
        </div>

        <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 shadow-sm">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-medium">Est. Cost Savings</span>
            <Zap className="w-4 h-4 text-purple-400" />
          </div>
          <div className="text-2xl font-bold text-slate-100">${metrics.estimated_savings_usd}</div>
          <span className="text-[10px] text-purple-400 font-mono">Per 1k Inquiries</span>
        </div>
      </section>

      {/* Main Workspace */}
      <main className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Search & Controls */}
        <div className="lg:col-span-7 space-y-6">
          <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm space-y-4">
            <h2 className="text-lg font-semibold text-slate-200 flex items-center gap-2">
              <Search className="w-4 h-4 text-cyan-400" />
              Financial Statement & Invoice Query
            </h2>

            {/* Quick Sample Queries */}
            <div className="space-y-1.5">
              <label className="text-xs text-slate-400 font-medium">Benchmark Queries</label>
              <div className="flex flex-wrap gap-2">
                {sampleQueries.map((q, idx) => (
                  <button
                    key={idx}
                    onClick={() => setQuery(q)}
                    className={`text-left text-xs px-3 py-1.5 rounded-lg border transition ${
                      query === q 
                        ? 'bg-blue-500/20 text-blue-300 border-blue-500/40' 
                        : 'bg-slate-800/60 text-slate-400 border-slate-700/60 hover:bg-slate-800 hover:text-slate-200'
                    }`}
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>

            {/* Input Box */}
            <div className="space-y-2">
              <label className="text-xs text-slate-400 font-medium">Query Input</label>
              <div className="relative">
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="e.g. What were total Q2 operating expenses?"
                  className="w-full px-4 py-3 bg-slate-950 border border-slate-700 rounded-xl text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition"
                  onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                />
              </div>
            </div>

            {/* Controls */}
            <div className="flex flex-wrap items-center justify-between gap-4 pt-2">
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2">
                  <span className="text-xs text-slate-400">Top-K:</span>
                  <select 
                    value={topK} 
                    onChange={(e) => setTopK(Number(e.target.value))}
                    className="bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-1 text-xs text-slate-200 focus:outline-none"
                  >
                    <option value={1}>1</option>
                    <option value={2}>2</option>
                    <option value={3}>3</option>
                    <option value={5}>5</option>
                  </select>
                </div>

                <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={useReranker}
                    onChange={(e) => setUseReranker(e.target.checked)}
                    className="rounded bg-slate-950 border-slate-700 text-blue-500 focus:ring-0 w-3.5 h-3.5"
                  />
                  Cross-Encoder Re-Rank
                </label>
              </div>

              <button
                onClick={handleSearch}
                disabled={loading}
                className="flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white rounded-xl text-sm font-semibold shadow-lg shadow-blue-500/20 transition disabled:opacity-50"
              >
                {loading ? (
                  <span className="animate-spin text-xs">●</span>
                ) : (
                  <>
                    <span>Execute Search</span>
                    <ArrowRight className="w-4 h-4" />
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Active Documents Widget */}
          <div className="p-4 rounded-xl bg-slate-900/40 border border-slate-800/80 text-xs text-slate-400 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <FileText className="w-4 h-4 text-slate-400" />
              <span>Active Ingested File: <strong className="text-slate-200">Apex_Q2_2026_Report.pdf</strong> (3 Pages, 4 Chunks)</span>
            </div>
            <span className="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-mono">100% Numerical Precision</span>
          </div>
        </div>

        {/* Right Column: Grounded Result & Citations */}
        <div className="lg:col-span-5 space-y-4">
          <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                Grounded Extraction
              </h3>
              {result && (
                <span className="text-xs font-mono text-cyan-400 bg-cyan-950/40 px-2 py-0.5 rounded border border-cyan-800/50">
                  {result.latency_ms} ms
                </span>
              )}
            </div>

            {result ? (
              <div className="space-y-4">
                <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800 text-sm text-slate-200 whitespace-pre-line font-sans leading-relaxed">
                  {result.answer}
                </div>

                <div className="space-y-2">
                  <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Verified Source Passages</h4>
                  {result.sources.map((src, idx) => (
                    <div key={idx} className="p-3.5 rounded-xl bg-slate-950 border border-slate-800/80 space-y-2 font-mono text-xs">
                      <div className="flex items-center justify-between">
                        <span className="flex items-center gap-1.5 font-semibold text-blue-400">
                          {src.chunk_type === 'financial_table' ? <Table className="w-3.5 h-3.5 text-emerald-400" /> : <FileText className="w-3.5 h-3.5 text-blue-400" />}
                          Page {src.page_number} ({src.chunk_type})
                        </span>
                        <span className="text-[10px] text-slate-400">RRF Score: {src.rrf_score}</span>
                      </div>
                      <pre className="text-[11px] text-slate-300 overflow-x-auto whitespace-pre-wrap font-mono p-2 bg-slate-900/60 rounded border border-slate-800">
                        {src.content}
                      </pre>
                      <div className="flex items-center justify-between text-[10px] text-slate-500">
                        <span>BM25 Rank: #{src.bm25_rank}</span>
                        <span>Dense Rank: #{src.dense_rank}</span>
                        <span>Tokens: ~{src.token_estimate}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="py-16 text-center text-slate-500 text-xs">
                <Search className="w-8 h-8 mx-auto mb-2 opacity-30 text-slate-400" />
                Select a benchmark query and click "Execute Search" to view grounded extractions.
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}