import asyncio
import hashlib
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

from app.core.config import settings


class AsyncTTLCache:
    """
    Asynchronous in-memory LRU + TTL cache keyed by normalized SHA-256 digests.
    Ensures repeated or canonicalized requests resolve in sub-millisecond latency
    without hitting downstream RAG or LLM providers.
    """

    def __init__(self, max_size: int = None, ttl_seconds: int = None):
        self.max_size = max_size if max_size is not None else settings.cache_max_size
        self.ttl_seconds = ttl_seconds if ttl_seconds is not None else settings.cache_ttl_seconds
        self._store: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = asyncio.Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    @staticmethod
    def build_key(canonical_text: str, threshold: float, force_stage2: bool = False, include_pos: bool = True) -> str:
        payload = f"{canonical_text.lower().strip()}|t={threshold:.3f}|f={int(force_stage2)}|p={int(include_pos)}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self.misses += 1
                return None

            expires_at, value = entry
            if time.perf_counter() > expires_at:
                del self._store[key]
                self.misses += 1
                self.evictions += 1
                return None

            self._store.move_to_end(key)
            self.hits += 1
            return value

    async def set(self, key: str, value: Any) -> None:
        async with self._lock:
            expires_at = time.perf_counter() + self.ttl_seconds
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (expires_at, value)
            while len(self._store) > self.max_size:
                self._store.popitem(last=False)
                self.evictions += 1

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()
            self.hits = 0
            self.misses = 0
            self.evictions = 0

    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {
            "current_size": len(self._store),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate_pct": round((self.hits / total) * 100.0, 2) if total > 0 else 0.0,
        }


_SHARED_CACHE: Optional[AsyncTTLCache] = None


def get_cache() -> AsyncTTLCache:
    global _SHARED_CACHE
    if _SHARED_CACHE is None:
        _SHARED_CACHE = AsyncTTLCache()
    return _SHARED_CACHE
