import time
import math
from typing import List, Dict, Any

class PerformanceMetricsTracker:
    """
    In-flight performance and cost observability tracker.
    Calculates P50, P90, P95, P99 latencies, token consumption, and cost savings.
    """
    def __init__(self, full_doc_avg_tokens: int = 4000, cost_per_1k_input_tokens: float = 0.005):
        self.latencies: List[float] = []
        self.total_tokens_used: int = 0
        self.total_queries: int = 0
        self.full_doc_avg_tokens = full_doc_avg_tokens
        self.cost_per_1k = cost_per_1k_input_tokens

    def record_query(self, latency_ms: float, tokens_used: int) -> None:
        self.latencies.append(latency_ms)
        self.total_tokens_used += tokens_used
        self.total_queries += 1

    def _percentile(self, data: List[float], p: float) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_data[int(k)]
        d0 = sorted_data[int(f)] * (c - k)
        d1 = sorted_data[int(c)] * (k - f)
        return d0 + d1

    def get_summary(self) -> Dict[str, Any]:
        if not self.latencies:
            return {
                "total_queries": 0,
                "avg_latency_ms": 0.0,
                "p50_latency_ms": 0.0,
                "p90_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
                "total_tokens_consumed": 0,
                "estimated_cost_usd": 0.0,
                "estimated_savings_vs_naive_usd": 0.0,
                "token_reduction_pct": 0.0
            }

        avg_lat = sum(self.latencies) / len(self.latencies)
        actual_cost = (self.total_tokens_used / 1000.0) * self.cost_per_1k
        naive_tokens = self.total_queries * self.full_doc_avg_tokens
        naive_cost = (naive_tokens / 1000.0) * self.cost_per_1k
        savings = max(0.0, naive_cost - actual_cost)

        return {
            "total_queries": self.total_queries,
            "avg_latency_ms": round(avg_lat, 2),
            "p50_latency_ms": round(self._percentile(self.latencies, 50), 2),
            "p90_latency_ms": round(self._percentile(self.latencies, 90), 2),
            "p95_latency_ms": round(self._percentile(self.latencies, 95), 2),
            "p99_latency_ms": round(self._percentile(self.latencies, 99), 2),
            "total_tokens_consumed": self.total_tokens_used,
            "estimated_cost_usd": round(actual_cost, 4),
            "estimated_savings_vs_naive_usd": round(savings, 4),
            "token_reduction_pct": round(((naive_tokens - self.total_tokens_used) / naive_tokens * 100) if naive_tokens > 0 else 0.0, 1)
        }