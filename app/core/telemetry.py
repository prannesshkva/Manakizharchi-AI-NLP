from collections import Counter, deque
from typing import Any, Callable, Deque, Dict
import numpy as np

from app.core.config import settings

try:
    from langsmith import traceable as _ls_traceable
    LANGSMITH_INSTALLED = True
except ImportError:
    LANGSMITH_INSTALLED = False

    def _ls_traceable(*args, **kwargs):
        def decorator(fn: Callable) -> Callable:
            return fn
        return decorator


def traceable(name: str = None, run_type: str = "chain"):
    """Wrapper for LangSmith @traceable with graceful zero-overhead no-op fallback."""
    if LANGSMITH_INSTALLED and settings.langsmith_tracing and settings.langsmith_api_key:
        return _ls_traceable(name=name, run_type=run_type)

    def noop(fn: Callable) -> Callable:
        return fn

    return noop


class PipelineTelemetry:
    """Tracks live latency percentiles, Stage-1 vs Stage-2 routing split, and ambiguity resolutions."""

    def __init__(self, max_samples: int = 5000):
        self.total_requests: int = 0
        self.cache_hits: int = 0
        self.stage1_fast_path_count: int = 0
        self.stage2_deep_semantic_count: int = 0
        self.ambiguities_resolved: int = 0
        self.ambiguity_breakdown: Counter = Counter()
        self.sentiment_breakdown: Counter = Counter()
        self.llm_live_calls: int = 0
        self.llm_fallback_calls: int = 0
        self.pre_noise_chars_total: int = 0
        self.post_noise_chars_total: int = 0
        self.latencies_ms: Deque[float] = deque(maxlen=max_samples)

    def record(
        self,
        routing_path: str,
        sentiment_label: str,
        latency_ms: float,
        ambiguity_detected: bool,
        ambiguity_reasons: list,
        original_chars: int,
        cleaned_chars: int,
        llm_provider: str,
    ) -> None:
        self.total_requests += 1
        if routing_path == "CACHE_HIT":
            self.cache_hits += 1
        elif routing_path == "STAGE1_VADER_FAST":
            self.stage1_fast_path_count += 1
        elif routing_path == "STAGE2_RAG_LLM_DEEP":
            self.stage2_deep_semantic_count += 1

        if ambiguity_detected and routing_path == "STAGE2_RAG_LLM_DEEP":
            self.ambiguities_resolved += 1
            for reason in ambiguity_reasons:
                self.ambiguity_breakdown[reason] += 1

        self.sentiment_breakdown[sentiment_label] += 1
        self.pre_noise_chars_total += max(0, original_chars)
        self.post_noise_chars_total += max(0, cleaned_chars)
        self.latencies_ms.append(float(latency_ms))

        if llm_provider in {"groq", "gemini"}:
            self.llm_live_calls += 1
        elif llm_provider == "rag_semantic_synthesizer":
            self.llm_fallback_calls += 1

    def snapshot(self) -> Dict[str, Any]:
        lats = list(self.latencies_ms)
        if lats:
            arr = np.asarray(lats, dtype=np.float64)
            mean_ms = float(np.mean(arr))
            p50_ms = float(np.percentile(arr, 50))
            p95_ms = float(np.percentile(arr, 95))
            p99_ms = float(np.percentile(arr, 99))
        else:
            mean_ms = p50_ms = p95_ms = p99_ms = 0.0

        noise_stripped_pct = 0.0
        if self.pre_noise_chars_total > 0:
            diff = max(0, self.pre_noise_chars_total - self.post_noise_chars_total)
            noise_stripped_pct = round((diff / self.pre_noise_chars_total) * 100.0, 2)

        routed_non_cache = self.stage1_fast_path_count + self.stage2_deep_semantic_count
        stage1_share = (
            round((self.stage1_fast_path_count / routed_non_cache) * 100.0, 2)
            if routed_non_cache > 0
            else 0.0
        )

        return {
            "total_requests": self.total_requests,
            "routing_counts": {
                "cache_hits": self.cache_hits,
                "stage1_vader_fast": self.stage1_fast_path_count,
                "stage2_rag_llm_deep": self.stage2_deep_semantic_count,
                "stage1_fast_share_pct": stage1_share,
            },
            "ambiguities_resolved": self.ambiguities_resolved,
            "ambiguity_breakdown": dict(self.ambiguity_breakdown),
            "sentiment_breakdown": dict(self.sentiment_breakdown),
            "llm_telemetry": {
                "live_api_calls": self.llm_live_calls,
                "deterministic_rag_calls": self.llm_fallback_calls,
                "langsmith_installed": LANGSMITH_INSTALLED,
                "langsmith_tracing": settings.langsmith_tracing,
            },
            "preprocessing": {
                "noise_chars_stripped": max(0, self.pre_noise_chars_total - self.post_noise_chars_total),
                "avg_noise_reduction_pct": noise_stripped_pct,
            },
            "latency_ms": {
                "mean": round(mean_ms, 3),
                "p50": round(p50_ms, 3),
                "p95": round(p95_ms, 3),
                "p99": round(p99_ms, 3),
            },
        }


_SHARED_TELEMETRY = PipelineTelemetry()


def get_telemetry() -> PipelineTelemetry:
    return _SHARED_TELEMETRY
