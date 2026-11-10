import os
import sys
import pytest
from httpx import ASGITransport, AsyncClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agents.sentiment_agent import SentimentOrchestrator
from app.core.cache import AsyncTTLCache
from app.core.telemetry import PipelineTelemetry
from app.main import app
from app.models.schemas import (
    AnalyzeRequest,
    RoutingPath,
    SentimentLabel,
)
from app.nlp.preprocessor import (
    clean_raw_text,
    extract_features,
    measure_inconsistency_reduction,
    semantic_chunk_text,
)
from app.nlp.vader_engine import evaluate_stage1_vader
from app.rag.qdrant_store import HybridQdrantStore


def test_noise_stripping_and_inconsistency_reduction():
    raw = "<div>THIS IS SOOOOO GOOOOD!! ❤️ https://example.com/track?id=123 @friend #BestProductEver</div>"
    cleaned, tags, emojis = clean_raw_text(raw)

    assert "<div" not in cleaned
    assert "https://" not in cleaned
    assert "@friend" not in cleaned
    assert "Best Product Ever" in cleaned
    assert "html_markup" in tags
    assert "urls_and_handles" in tags
    assert "mentions" in tags
    assert "char_elongation" in tags

    # Measure inconsistency reduction across noisy variant pairs
    noisy_pairs = [
        ("<div>Awesome app!! https://x.co/1 @user</div>", "awesome app!!"),
        ("SOOOOO GOOOOOD experience   #FastDelivery", "soo good experience fast delivery"),
        ("<p>Terrible crash... http://err.local</p>", "terrible crash."),
    ]
    metrics = measure_inconsistency_reduction(noisy_pairs)
    assert metrics["cleaned_inconsistency_rate"] < metrics["raw_inconsistency_rate"]
    assert metrics["relative_reduction_pct"] > 50.0


def test_pos_tagging_and_semantic_chunking():
    text = "The camera display looks crisp and vibrant, however the battery drains terribly fast."
    feats = extract_features(text, include_pos_tags=True)

    assert len(feats.pos_tags_sample) > 5
    assert feats.pos_distribution["nouns"] >= 1
    assert feats.pos_distribution["adjectives"] >= 1
    assert "however" in feats.contrast_markers
    assert len(feats.semantic_chunks) >= 2
    assert any("camera display" in c.lower() for c in feats.semantic_chunks)
    assert any("battery" in c.lower() for c in feats.semantic_chunks)


def test_stage1_vader_confidence_gating():
    # Clear unambiguous positive should take fast path
    clear_text = "Absolutely loved this product! Fantastic craftsmanship, ultra reliable, and prompt shipping."
    clear_feats = extract_features(clear_text)
    clear_s1 = evaluate_stage1_vader(clear_text, clear_feats, confidence_threshold=0.68)
    assert clear_s1.initial_label == SentimentLabel.POSITIVE
    assert clear_s1.confidence >= 0.68
    assert clear_s1.requires_stage2 is False

    # Slang ambiguity ('im literally dead') must trigger Stage-2 despite 'dead' negative score
    slang_text = "im literally dead 💀😂 this beat drop is straight fire no cap"
    slang_feats = extract_features(slang_text)
    slang_s1 = evaluate_stage1_vader(slang_text, slang_feats, confidence_threshold=0.68)
    assert slang_s1.requires_stage2 is True
    assert "slang_lexicon_gap" in slang_s1.ambiguity_reasons

    # Sarcasm must trigger Stage-2
    sarcasm_text = "Oh great, another 4-hour server crash right before the demo!! 🙄"
    sarc_feats = extract_features(sarcasm_text)
    sarc_s1 = evaluate_stage1_vader(sarcasm_text, sarc_feats, confidence_threshold=0.68)
    assert sarc_s1.requires_stage2 is True
    assert "sarcasm_or_irony_cue" in sarc_s1.ambiguity_reasons


@pytest.mark.asyncio
async def test_qdrant_hybrid_rag_and_custom_ingestion():
    store = HybridQdrantStore()
    count_before = await store.initialize()
    assert count_before >= 30

    matches = await store.hybrid_search("im literally dead skull emoji", top_k=3)
    assert len(matches) >= 1
    assert matches[0].sentiment == "POSITIVE"

    chunks_added = await store.ingest_document(
        text="ultra clean microservice tuning zero tail spikes",
        emotion_label="Deeply Impressed",
        sentiment="POSITIVE",
        category="engineering_slang",
    )
    assert chunks_added == 1
    assert store.corpus_count() == count_before + 1

    custom_hits = await store.hybrid_search("zero tail spikes microservice tuning", top_k=2)
    assert any("tail spikes" in h.text for h in custom_hits)


@pytest.mark.asyncio
async def test_two_stage_langgraph_routing_and_caching():
    orch = SentimentOrchestrator(
        store=HybridQdrantStore(),
        cache=AsyncTTLCache(max_size=128, ttl_seconds=600),
        telemetry=PipelineTelemetry(),
    )

    # 1. Clear positive -> Stage-1 Fast Exit
    fast_res = await orch.analyze(
        AnalyzeRequest(text="Wonderful experience! Top quality build and super fast delivery.")
    )
    assert fast_res.routing_path == RoutingPath.STAGE1_VADER_FAST
    assert fast_res.sentiment_label == SentimentLabel.POSITIVE
    assert fast_res.cache_hit is False
    assert fast_res.retrieved_context == []

    # 2. Repeat exact same canonical input (even with extra HTML/whitespace) -> Cache Hit
    cache_res = await orch.analyze(
        AnalyzeRequest(text="<div>Wonderful experience! Top quality build and super fast delivery.</div>  ")
    )
    assert cache_res.routing_path == RoutingPath.CACHE_HIT
    assert cache_res.cache_hit is True
    assert cache_res.sentiment_label == SentimentLabel.POSITIVE

    # 3. Gen-Z Slang -> Escalates to Stage-2 Qdrant RAG & flips naive VADER negative to POSITIVE
    slang_res = await orch.analyze(
        AnalyzeRequest(text="im literally dead 💀😂 this edit is straight fire tbh")
    )
    assert slang_res.routing_path == RoutingPath.STAGE2_RAG_LLM_DEEP
    assert slang_res.sentiment_label == SentimentLabel.POSITIVE
    assert len(slang_res.retrieved_context) >= 1

    # 4. Sarcasm -> Escalates to Stage-2 -> SARCASTIC
    sarc_res = await orch.analyze(
        AnalyzeRequest(text="Oh great, another bug in production just what I needed 🙄")
    )
    assert sarc_res.routing_path == RoutingPath.STAGE2_RAG_LLM_DEEP
    assert sarc_res.sentiment_label == SentimentLabel.SARCASTIC

    # 5. Mixed Contrast Clauses -> Escalates to Stage-2 -> MIXED
    mixed_res = await orch.analyze(
        AnalyzeRequest(text="The camera resolution is brilliant and crisp, however the battery drains terribly fast.")
    )
    assert mixed_res.routing_path == RoutingPath.STAGE2_RAG_LLM_DEEP
    assert mixed_res.sentiment_label == SentimentLabel.MIXED

    # 6. Tanglish Appreciation -> Escalates to Stage-2 -> POSITIVE
    tanglish_res = await orch.analyze(
        AnalyzeRequest(text="Bro sema mass performance!! Vera level energy kalakkal")
    )
    assert tanglish_res.routing_path == RoutingPath.STAGE2_RAG_LLM_DEEP
    assert tanglish_res.sentiment_label == SentimentLabel.POSITIVE

    # 7. Log-tag prefix [ALERT] stripping prevents false positive VADER drift on neutral/negative inputs
    alert_neu = await orch.analyze(
        AnalyzeRequest(text="[ALERT] Updated headset firmware version 2.1 yesterday afternoon.")
    )
    assert alert_neu.sentiment_label == SentimentLabel.NEUTRAL
    assert "log_metadata_tags" in alert_neu.features.noise_tags_removed

    alert_neg = await orch.analyze(
        AnalyzeRequest(text="[ALERT] Avoid this keyboard at all costs — slow, frustrating, and unusable.")
    )
    assert alert_neg.sentiment_label == SentimentLabel.NEGATIVE

    # 8. OOV expressive praise & Tanglish negative critique avoid false sarcasm drift
    oov_pos = await orch.analyze(
        AnalyzeRequest(text="The laptop works smoothly and exceeded every single expectation I had.")
    )
    assert oov_pos.sentiment_label == SentimentLabel.POSITIVE

    tanglish_neg = await orch.analyze(
        AnalyzeRequest(text="its giving broken laptop, konjam overa trash tbh")
    )
    assert tanglish_neg.sentiment_label == SentimentLabel.NEGATIVE

    mixed_hinge = await orch.analyze(
        AnalyzeRequest(text="Fast delivery of the laptop, yet the hinge feels worryingly fragile.")
    )
    assert mixed_hinge.sentiment_label == SentimentLabel.MIXED


@pytest.mark.asyncio
async def test_fastapi_endpoints_end_to_end():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Health
        h = await client.get("/health")
        assert h.status_code == 200
        h_json = h.json()
        assert h_json["status"] == "online"
        assert h_json["qdrant_points"] >= 30

        # Analyze v2
        r = await client.post(
            "/api/v2/analyze",
            json={"text": "this movie is so mid womp womp"},
        )
        assert r.status_code == 200
        r_json = r.json()
        assert r_json["sentiment_label"] == "NEGATIVE"
        assert r_json["routing_path"] in {"STAGE2_RAG_LLM_DEEP", "CACHE_HIT"}

        # Pydantic Validation rejects empty whitespace
        bad = await client.post("/api/v2/analyze", json={"text": "    \x00   "})
        assert bad.status_code == 422

        # Batch Async
        b = await client.post(
            "/api/v2/analyze/batch",
            json={
                "texts": [
                    "Fantastic laptop, zero lag and superb display!",
                    "im dead 💀 straight fire no cap",
                    "Oh great another crash right before submit 🙄",
                    "Terrible battery life, broke on day one.",
                ],
                "concurrency": 8,
            },
        )
        assert b.status_code == 200
        b_json = b.json()
        assert b_json["total_inputs"] == 4
        assert len(b_json["items"]) == 4

        # Social Engagement
        s = await client.post(
            "/api/v2/social/analyze",
            json={
                "caption": "Excited to share our Qdrant sentiment engine launch! 🚀",
                "comments": [
                    "sema mass launch bro 🔥",
                    "Super clean UI and fast latency!",
                ],
                "include_plot": False,
            },
        )
        assert s.status_code == 200
        s_json = s.json()
        assert s_json["comments_analyzed"] == 2
        assert s_json["dominant_sentiment"] == "POSITIVE"

        # Legacy v1 compatibility route
        v1 = await client.post("/api/text-analysis", json={"text": "Awesome experience overall!"})
        assert v1.status_code == 200
        v1_json = v1.json()
        assert "scores" in v1_json and "plot" in v1_json and "interpretation" in v1_json

        # Metrics
        m = await client.get("/api/v2/metrics")
        assert m.status_code == 200
        assert m.json()["telemetry"]["total_requests"] >= 4
