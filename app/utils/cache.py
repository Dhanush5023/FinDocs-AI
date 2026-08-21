import time
import hashlib
from collections import OrderedDict
from typing import Optional, Dict, Any, Tuple

class QueryCache:
    """
    In-memory LRU Query Cache with TTL and hit/miss telemetry.
    Eliminates redundant retrieval and LLM calls for repeated financial queries.
    """
    def __init__(self, capacity: int = 500, default_ttl_seconds: int = 3600):
        self.capacity = capacity
        self.default_ttl = default_ttl_seconds
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self.hits: int = 0
        self.misses: int = 0

    def _hash_key(self, doc_id: str, query: str, top_k: int) -> str:
        raw = f"{doc_id}::{query.strip().lower()}::{top_k}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, doc_id: str, query: str, top_k: int) -> Optional[Any]:
        key = self._hash_key(doc_id, query, top_k)
        if key not in self._cache:
            self.misses += 1
            return None

        val, expire_at = self._cache[key]
        if time.time() > expire_at:
            del self._cache[key]
            self.misses += 1
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        self.hits += 1
        return val

    def set(self, doc_id: str, query: str, top_k: int, value: Any, ttl: Optional[int] = None) -> None:
        key = self._hash_key(doc_id, query, top_k)
        ttl = ttl if ttl is not None else self.default_ttl
        expire_at = time.time() + ttl

        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (value, expire_at)

        if len(self._cache) > self.capacity:
            self._cache.popitem(last=False)

    @property
    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0.0
        return {
            "capacity": self.capacity,
            "current_size": len(self._cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_pct": round(hit_rate, 2)
        }

    def clear(self) -> None:
        self._cache.clear()
        self.hits = 0
        self.misses = 0
