import asyncio
import time
from collections import Counter
from typing import Any, Dict, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from app.agents.sentiment_agent import LANGGRAPH_AVAILABLE, get_orchestrator
from app.core.cache import get_cache
from app.core.config import settings
from app.core.embeddings import get_embedder
from app.core.telemetry import get_telemetry
from app.models.schemas import (
    AnalyzeRequest,
    BatchAnalyzeRequest,
    BatchAnalyzeResponse,
    IngestDocumentRequest,
    IngestResponse,
    RoutingPath,
    SentimentResponse,
    SocialAnalyzeRequest,
)
from app.rag.qdrant_store import get_qdrant_store
from app.services.social_analyzer import analyze_social_engagement

router = APIRouter()


@router.get("/health", tags=["Health"])
@router.get("/api/health", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    store = get_qdrant_store()
    await store.initialize()
    embedder = get_embedder()
    cache = get_cache()

    llm_mode = "groq" if settings.groq_api_key else ("gemini" if settings.gemini_api_key else "deterministic_rag_fallback")
    return {
        "status": "online",
        "service": settings.app_name,
        "version": settings.app_version,
        "langgraph_enabled": LANGGRAPH_AVAILABLE,
        "qdrant_collection": store.collection_name,
        "qdrant_points": store.corpus_count(),
        "embedding_backend": embedder.active_backend,
        "embedding_dim": embedder.dim,
        "llm_mode": llm_mode,
        "confidence_threshold": settings.confidence_threshold,
        "cache": cache.stats(),
    }


@router.post("/api/v2/analyze", response_model=SentimentResponse, tags=["Sentiment Engine v2"])
async def analyze_text_v2(req: AnalyzeRequest) -> SentimentResponse:
    orchestrator = get_orchestrator()
    return await orchestrator.analyze(req)


@router.post("/api/v2/analyze/batch", response_model=BatchAnalyzeResponse, tags=["Sentiment Engine v2"])
async def analyze_batch_v2(req: BatchAnalyzeRequest) -> BatchAnalyzeResponse:
    orchestrator = get_orchestrator()
    await orchestrator.store.initialize()

    t_wall_start = time.perf_counter()
    sem = asyncio.Semaphore(req.concurrency)

    async def _run_item(txt: str) -> SentimentResponse:
        async with sem:
            return await orchestrator.analyze(
                AnalyzeRequest(
                    text=txt,
                    include_pos_tags=req.include_pos_tags,
                    include_plot=False,
                    confidence_threshold=req.confidence_threshold,
                )
            )

    results = await asyncio.gather(*[_run_item(t) for t in req.texts])
    wall_ms = round((time.perf_counter() - t_wall_start) * 1000.0, 3)

    latencies = [r.latency_ms for r in results]
    arr = np.asarray(latencies, dtype=np.float64) if latencies else np.asarray([0.0])

    s1_count = sum(1 for r in results if r.routing_path == RoutingPath.STAGE1_VADER_FAST)
    s2_count = sum(1 for r in results if r.routing_path == RoutingPath.STAGE2_RAG_LLM_DEEP)
    cache_count = sum(1 for r in results if r.routing_path == RoutingPath.CACHE_HIT)
    non_cache = s1_count + s2_count
    ambiguities_resolved = sum(1 for r in results if r.ambiguity_detected and r.routing_path == RoutingPath.STAGE2_RAG_LLM_DEEP)
    avg_noise = round(sum(r.features.noise_reduction_pct for r in results) / len(results), 2) if results else 0.0
    dist = dict(Counter(r.sentiment_label.value for r in results))

    return BatchAnalyzeResponse(
        total_inputs=len(results),
        stage1_fast_count=s1_count,
        stage2_deep_count=s2_count,
        cache_hit_count=cache_count,
        stage1_fast_pct=round((s1_count / non_cache) * 100.0, 2) if non_cache > 0 else 0.0,
        ambiguities_resolved=ambiguities_resolved,
        mean_latency_ms=round(float(np.mean(arr)), 3),
        p50_latency_ms=round(float(np.percentile(arr, 50)), 3),
        p95_latency_ms=round(float(np.percentile(arr, 95)), 3),
        total_wall_time_ms=wall_ms,
        avg_noise_reduction_pct=avg_noise,
        sentiment_distribution=dist,
        items=results if req.include_items else [],
    )


@router.post("/api/v2/ingest", response_model=IngestResponse, tags=["Qdrant Hybrid KB"])
async def ingest_corpus_item(req: IngestDocumentRequest) -> IngestResponse:
    store = get_qdrant_store()
    chunks_added = await store.ingest_document(
        text=req.text,
        emotion_label=req.emotion_label,
        sentiment=req.sentiment.value,
        category=req.category,
        source=req.source,
    )
    if chunks_added == 0:
        raise HTTPException(status_code=400, detail="No valid semantic chunks extracted from input.")
    return IngestResponse(
        status="indexed",
        chunks_indexed=chunks_added,
        total_corpus_size=store.corpus_count(),
        category=req.category,
    )


@router.get("/api/v2/corpus", tags=["Qdrant Hybrid KB"])
async def list_corpus(
    category: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
) -> Dict[str, Any]:
    store = get_qdrant_store()
    await store.initialize()
    items = store.list_corpus(category=category, limit=limit)
    return {
        "total_indexed": store.corpus_count(),
        "returned": len(items),
        "items": items,
    }


@router.get("/api/v2/metrics", tags=["Observability"])
async def get_metrics() -> Dict[str, Any]:
    store = get_qdrant_store()
    await store.initialize()
    telemetry = get_telemetry().snapshot()
    cache_stats = get_cache().stats()
    return {
        "telemetry": telemetry,
        "cache": cache_stats,
        "qdrant_corpus_size": store.corpus_count(),
    }


@router.post("/api/v2/cache/clear", tags=["Observability"])
async def clear_cache() -> Dict[str, Any]:
    cache = get_cache()
    await cache.clear()
    return {"status": "cleared", "cache": cache.stats()}


@router.post("/api/v2/social/analyze", tags=["Social Engagement"])
async def analyze_social_v2(req: SocialAnalyzeRequest) -> Dict[str, Any]:
    return await analyze_social_engagement(req, orchestrator=get_orchestrator())


# ------------------------------------------------------------------
# Backward-Compatible v1 Endpoints
# ------------------------------------------------------------------
@router.post("/api/text-analysis", tags=["Legacy Compatibility"])
async def legacy_text_analysis(payload: Dict[str, Any]) -> Dict[str, Any]:
    raw_text = (payload.get("text") or "").strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Text is required")

    req = AnalyzeRequest(text=raw_text, include_plot=True, include_pos_tags=True)
    res = await get_orchestrator().analyze(req)

    return {
        "scores": {
            "positive": f"{res.vader_scores.positive * 100:.2f}%",
            "neutral": f"{res.vader_scores.neutral * 100:.2f}%",
            "negative": f"{res.vader_scores.negative * 100:.2f}%",
            "compound": res.vader_scores.compound,
        },
        "plot": res.plot_base64,
        "interpretation": res.interpretation,
        "v2": res.model_dump(mode="json"),
    }


@router.post("/api/insta-analysis", tags=["Legacy Compatibility"])
async def legacy_insta_analysis(payload: Dict[str, Any]) -> Dict[str, Any]:
    req = SocialAnalyzeRequest(
        post_url=payload.get("postUrl") or payload.get("post_url"),
        caption=payload.get("caption"),
        comments=payload.get("comments"),
        session_id=payload.get("sessionId") or payload.get("session_id"),
        include_plot=True,
    )
    return await analyze_social_engagement(req, orchestrator=get_orchestrator())
