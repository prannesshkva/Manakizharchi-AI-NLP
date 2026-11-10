import asyncio
import json
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import httpx
from pydantic import BaseModel, Field

try:
    from langgraph.graph import END, START, StateGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

from app.core.cache import AsyncTTLCache, get_cache
from app.core.config import settings
from app.core.telemetry import PipelineTelemetry, get_telemetry, traceable
from app.models.schemas import (
    AnalyzeRequest,
    ChunkSentiment,
    FeatureExtractionResult,
    RetrievedContext,
    RoutingPath,
    SentimentLabel,
    SentimentResponse,
)
from app.nlp.preprocessor import clean_raw_text, extract_features
from app.nlp.vader_engine import Stage1Result, evaluate_stage1_vader
from app.rag.qdrant_store import HybridQdrantStore, get_qdrant_store
from app.services.plotting import render_sentiment_bar_async


class Stage2LLMVerdict(BaseModel):
    """Pydantic schema validating structured Stage-2 LLM output."""
    sentiment_label: SentimentLabel
    emotion_categories: List[str] = Field(min_length=1, max_length=4)
    confidence: float = Field(ge=0.0, le=1.0)
    interpretation: str = Field(min_length=5, max_length=600)


class SentimentAgentState(TypedDict, total=False):
    request_id: str
    raw_text: str
    force_stage2: bool
    include_pos_tags: bool
    include_plot: bool
    confidence_threshold: float
    t_start: float
    features: FeatureExtractionResult
    cache_key: str
    cache_hit: bool
    stage1: Stage1Result
    retrieved_context: List[RetrievedContext]
    final_response: SentimentResponse


class SentimentOrchestrator:
    """
    Two-Stage Agentic Sentiment Pipeline orchestrated with LangGraph StateGraph:
      - Stage 1: Pydantic validation -> NLP Noise Stripping & POS Extraction
                 -> SHA-256 Cache Lookup -> VADER Lexicon & Confidence Gating
      - Gate   : Route unambiguous high-confidence text directly out (<5ms)
                 or escalate ambiguous/slang/sarcasm/mixed text to Stage 2
      - Stage 2: Async Qdrant + BM25 Hybrid RAG Retrieval (RRF)
                 -> Async Cloud LLM Evaluation (Groq/Gemini) with Deterministic
                    Semantic Circuit-Breaker Fallback
    """

    def __init__(
        self,
        store: Optional[HybridQdrantStore] = None,
        cache: Optional[AsyncTTLCache] = None,
        telemetry: Optional[PipelineTelemetry] = None,
    ):
        self.store = store or get_qdrant_store()
        self.cache = cache or get_cache()
        self.telemetry = telemetry or get_telemetry()
        self._graph = self._compile_langgraph() if LANGGRAPH_AVAILABLE else None

    # ------------------------------------------------------------------
    # Graph Nodes
    # ------------------------------------------------------------------
    async def _cache_lookup_node(self, state: SentimentAgentState) -> Dict[str, Any]:
        canonical_text, _, _ = clean_raw_text(state["raw_text"])
        threshold = state["confidence_threshold"]
        cache_key = self.cache.build_key(
            canonical_text=canonical_text,
            threshold=threshold,
            force_stage2=state["force_stage2"],
            include_pos=state["include_pos_tags"],
        )
        cached: Optional[SentimentResponse] = await self.cache.get(cache_key)
        if cached is None:
            return {"cache_hit": False, "cache_key": cache_key}

        elapsed_ms = round((time.perf_counter() - state["t_start"]) * 1000.0, 3)
        plot_b64 = cached.plot_base64
        if state["include_plot"] and not plot_b64:
            plot_b64 = await render_sentiment_bar_async(cached.vader_scores)

        cloned = cached.model_copy(
            update={
                "request_id": state["request_id"],
                "cache_hit": True,
                "routing_path": RoutingPath.CACHE_HIT,
                "latency_ms": elapsed_ms,
                "plot_base64": plot_b64,
            }
        )
        self.telemetry.record(
            routing_path=RoutingPath.CACHE_HIT.value,
            sentiment_label=cloned.sentiment_label.value,
            latency_ms=elapsed_ms,
            ambiguity_detected=cloned.ambiguity_detected,
            ambiguity_reasons=cloned.ambiguity_reasons,
            original_chars=cloned.features.original_length,
            cleaned_chars=cloned.features.cleaned_length,
            llm_provider="in_memory_cache",
        )
        return {"cache_hit": True, "cache_key": cache_key, "final_response": cloned}

    async def _preprocess_node(self, state: SentimentAgentState) -> Dict[str, Any]:
        features = extract_features(state["raw_text"], include_pos_tags=state["include_pos_tags"])
        return {"features": features}

    async def _stage1_vader_node(self, state: SentimentAgentState) -> Dict[str, Any]:
        stage1 = evaluate_stage1_vader(
            raw_text=state["raw_text"],
            features=state["features"],
            confidence_threshold=state["confidence_threshold"],
        )
        return {"stage1": stage1}

    async def _fast_finalize_node(self, state: SentimentAgentState) -> Dict[str, Any]:
        s1: Stage1Result = state["stage1"]
        feats: FeatureExtractionResult = state["features"]
        label = s1.initial_label

        if label == SentimentLabel.POSITIVE:
            emotions = ["Happy / Satisfied", "Approval"]
            interp = (
                f"Stage-1 Fast Path resolved unambiguously positive sentiment "
                f"(compound={s1.vader_scores.compound:+.3f}, confidence={s1.confidence:.2%}) without requiring LLM escalation."
            )
        elif label == SentimentLabel.NEGATIVE:
            emotions = ["Disappointed / Critical", "Dissatisfaction"]
            interp = (
                f"Stage-1 Fast Path resolved unambiguously negative sentiment "
                f"(compound={s1.vader_scores.compound:+.3f}, confidence={s1.confidence:.2%}) without requiring LLM escalation."
            )
        else:
            emotions = ["Neutral / Objective"]
            interp = (
                f"Stage-1 Fast Path resolved straightforward neutral tone "
                f"(compound={s1.vader_scores.compound:+.3f}, confidence={s1.confidence:.2%})."
            )

        plot_b64 = await render_sentiment_bar_async(s1.vader_scores) if state["include_plot"] else None
        elapsed_ms = round((time.perf_counter() - state["t_start"]) * 1000.0, 3)

        response = SentimentResponse(
            request_id=state["request_id"],
            sentiment_label=label,
            emotion_categories=emotions,
            confidence=s1.confidence,
            routing_path=RoutingPath.STAGE1_VADER_FAST,
            ambiguity_detected=False,
            ambiguity_reasons=s1.ambiguity_reasons,
            cache_hit=False,
            latency_ms=elapsed_ms,
            vader_scores=s1.vader_scores,
            stage1_confidence=s1.confidence,
            interpretation=interp,
            llm_provider_used="vader_lexicon_fast_path",
            features=feats,
            retrieved_context=[],
            chunk_sentiments=s1.chunk_sentiments,
            plot_base64=plot_b64,
        )
        await self.cache.set(state["cache_key"], response)
        self.telemetry.record(
            routing_path=RoutingPath.STAGE1_VADER_FAST.value,
            sentiment_label=response.sentiment_label.value,
            latency_ms=elapsed_ms,
            ambiguity_detected=False,
            ambiguity_reasons=[],
            original_chars=feats.original_length,
            cleaned_chars=feats.cleaned_length,
            llm_provider="vader_lexicon_fast_path",
        )
        return {"final_response": response}

    async def _qdrant_rag_node(self, state: SentimentAgentState) -> Dict[str, Any]:
        feats: FeatureExtractionResult = state["features"]
        query = feats.canonical_text or state["raw_text"]
        # Augment query with detected ironic emoji semantics so vector search aligns closely
        if feats.slang_markers or feats.ironic_emojis:
            extras = " ".join(feats.slang_markers + feats.ironic_emojis)
            query = f"{query} {extras}".strip()

        contexts = await self.store.hybrid_search(query_text=query, top_k=4)
        return {"retrieved_context": contexts}

    async def _call_live_llm_stage2(
        self,
        raw_text: str,
        feats: FeatureExtractionResult,
        stage1: Stage1Result,
        contexts: List[RetrievedContext],
    ) -> Optional[Tuple[Stage2LLMVerdict, str]]:
        """
        Optional asynchronous Stage-2 LLM call (Groq or Gemini) with strict timeout
        and Pydantic schema enforcement. Returns None on missing key or circuit-breaker trip.
        """
        rag_summary = "\n".join(
            [
                f"- [{c.sentiment}] '{c.text}' -> Emotion: {c.emotion_label} (RRF={c.rrf_score:.4f})"
                for c in contexts
            ]
        ) or "No close RAG precedents found."

        chunks_info = "; ".join(
            [f"Chunk{c.chunk_index}('{c.text[:60]}' comp={c.compound:+.2f})" for c in stage1.chunk_sentiments]
        )

        system_prompt = (
            "You are Manakizharchi AI Stage-2 Sentiment Intelligence Reasoner. "
            "Resolve lexical ambiguity, slang, sarcasm, code-mixed Tamil/Tanglish terms, and mixed clause contrast. "
            "Respond strictly as compact JSON with keys: "
            '{"sentiment_label": "POSITIVE|NEGATIVE|NEUTRAL|MIXED|SARCASTIC", '
            '"emotion_categories": ["..."], "confidence": 0.0-1.0, "interpretation": "concise 1-2 sentence rationale"}'
        )
        user_prompt = (
            f"Input text: {feats.canonical_text}\n"
            f"Stage-1 VADER: compound={stage1.vader_scores.compound}, pos={stage1.vader_scores.positive}, "
            f"neg={stage1.vader_scores.negative}, neu={stage1.vader_scores.neutral}\n"
            f"Ambiguity signals: {stage1.ambiguity_reasons}\n"
            f"Detected slang: {feats.slang_markers} | Ironic emojis: {feats.ironic_emojis} | Negations: {feats.negation_terms}\n"
            f"Semantic chunks: {chunks_info}\n"
            f"Qdrant Hybrid RAG Precedents:\n{rag_summary}"
        )

        # 1. Try Groq Async API
        if settings.groq_api_key:
            try:
                async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
                    resp = await client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {settings.groq_api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": settings.groq_model,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            "temperature": 0.1,
                            "response_format": {"type": "json_object"},
                        },
                    )
                    if resp.status_code == 200:
                        content = resp.json()["choices"][0]["message"]["content"]
                        parsed = Stage2LLMVerdict.model_validate_json(content)
                        return parsed, "groq"
            except Exception:
                pass

        # 2. Try Gemini REST API
        if settings.gemini_api_key:
            try:
                async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
                    url = (
                        f"https://generativelanguage.googleapis.com/v1beta/models/"
                        f"gemini-2.0-flash:generateContent?key={settings.gemini_api_key}"
                    )
                    resp = await client.post(
                        url,
                        json={
                            "contents": [{"parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]}],
                            "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"},
                        },
                    )
                    if resp.status_code == 200:
                        text_out = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                        json_match = re.search(r"\{.*\}", text_out, flags=re.DOTALL)
                        if json_match:
                            parsed = Stage2LLMVerdict.model_validate_json(json_match.group(0))
                            return parsed, "gemini"
            except Exception:
                pass

        return None

    def _synthesize_rag_semantic_verdict(
        self,
        feats: FeatureExtractionResult,
        stage1: Stage1Result,
        contexts: List[RetrievedContext],
    ) -> Stage2LLMVerdict:
        """
        Deterministic Qdrant Hybrid RAG + Linguistic Heuristic Synthesizer.
        Used seamlessly when external LLM APIs are unconfigured or offline,
        resolving slang inversions, sarcasm cues, and mixed-clause polarity flips.
        """
        reasons = set(stage1.ambiguity_reasons)
        chunks = stage1.chunk_sentiments

        # 1. Weighted Qdrant Hybrid Precedent Vote
        vote_weights: Dict[str, float] = {}
        top_match = contexts[0] if contexts else None
        for idx, ctx in enumerate(contexts):
            # Prioritize candidates that genuinely overlap lexically or semantically
            sim_weight = ctx.rrf_score * (1.0 + max(0.0, ctx.dense_score) + min(1.5, ctx.sparse_score * 0.25))
            if idx == 0:
                sim_weight *= 1.35
            vote_weights[ctx.sentiment] = vote_weights.get(ctx.sentiment, 0.0) + sim_weight

        # 2. Inspect mixed-clause contrast cleanly before pure slang lookup
        chunk_compounds = [c.compound for c in chunks]
        has_opposing_chunks = (
            len(chunk_compounds) >= 2
            and max(chunk_compounds) >= 0.20
            and min(chunk_compounds) <= -0.15
        )
        contrast_matched_mixed = (
            bool(feats.contrast_markers)
            and len(chunk_compounds) >= 2
            and (
                (top_match is not None and top_match.sentiment == "MIXED" and top_match.dense_score >= 0.30)
                or any(w in feats.canonical_text.lower() for w in ("fast", "quick", "helpful", "brilliant", "gorgeous", "clean"))
                and min(chunk_compounds) <= -0.15
            )
        )
        if ("mixed_clause_contrast" in reasons and (has_opposing_chunks or (feats.contrast_markers and stage1.vader_scores.positive >= 0.12 and stage1.vader_scores.negative >= 0.12))) or contrast_matched_mixed:
            pos_chunks = [c.text for c in chunks if c.compound >= 0.0]
            neg_chunks = [c.text for c in chunks if c.compound < 0.0]
            pos_preview = pos_chunks[0][:55] if pos_chunks else "positive aspect"
            neg_preview = neg_chunks[-1][:55] if neg_chunks else "critical concern"
            return Stage2LLMVerdict(
                sentiment_label=SentimentLabel.MIXED,
                emotion_categories=["Ambivalent", "Balanced Critique"],
                confidence=round(max(0.79, min(0.94, 0.76 + abs(max(chunk_compounds) - min(chunk_compounds)) * 0.18)), 4),
                interpretation=(
                    f"Stage-2 Semantic Chunking resolved mixed-aspect polarity across contrast clauses "
                    f"('{pos_preview}' [+] vs '{neg_preview}' [-])."
                ),
            )

        # 3. Inspect Sarcasm & Irony cues (strict guardrail: require irony cues or strong sarcasm vector match)
        if "sarcasm_or_irony_cue" in reasons or (top_match is not None and top_match.sentiment == "SARCASTIC" and top_match.dense_score >= 0.65):
            cue_desc = (
                f"precedent '{top_match.text}' ({top_match.emotion_label})"
                if top_match and top_match.sentiment == "SARCASTIC"
                else "positive surface phrasing contradicted by negative event/emoji markers"
            )
            return Stage2LLMVerdict(
                sentiment_label=SentimentLabel.SARCASTIC,
                emotion_categories=["Sarcastic", "Frustrated / Ironic"],
                confidence=round(min(0.95, 0.80 + (top_match.dense_score * 0.12 if top_match else 0.05)), 4),
                interpretation=(
                    f"Stage-2 Confidence-Gated RAG detected lexical irony: {cue_desc}, "
                    f"overriding literal positive VADER tokens."
                ),
            )

        # 4. Slang / Tanglish / OOV Lexicon Resolution via Qdrant Hybrid RAG
        if ("slang_lexicon_gap" in reasons or "oov_expressive_gap" in reasons) and top_match is not None:
            # Determine strongest precedent sentiment, guarding against weak false-positive sarcasm drift
            safe_votes = {
                k: v for k, v in vote_weights.items()
                if k != "SARCASTIC" or "sarcasm_or_irony_cue" in reasons
            }
            best_label_str = max(safe_votes.items(), key=lambda x: x[1])[0] if safe_votes else top_match.sentiment
            if best_label_str == "SARCASTIC" and "sarcasm_or_irony_cue" not in reasons:
                best_label_str = stage1.initial_label.value
            try:
                resolved_label = SentimentLabel(best_label_str)
            except ValueError:
                resolved_label = SentimentLabel.NEUTRAL

            matching_emotions = [c.emotion_label for c in contexts[:2] if c.sentiment == resolved_label.value]
            if not matching_emotions:
                matching_emotions = [top_match.emotion_label]

            conf = round(min(0.95, max(0.76, 0.72 + min(0.22, top_match.dense_score * 0.25 + (0.08 if top_match.sparse_score > 0 else 0.0)))), 4)
            slang_str = ", ".join(feats.slang_markers + feats.ironic_emojis) or "expressive OOV terms"
            return Stage2LLMVerdict(
                sentiment_label=resolved_label,
                emotion_categories=matching_emotions[:2],
                confidence=conf,
                interpretation=(
                    f"Stage-2 Qdrant Hybrid RAG resolved lexical ambiguity ({slang_str}) using retrieved "
                    f"precedent '{top_match.text}' -> {resolved_label.value} ({top_match.emotion_label})."
                ),
            )

        # 5. Negation / Weak Polarity fallback combining VADER + top precedent nuance
        fallback_label = stage1.initial_label
        if top_match and top_match.sentiment != "SARCASTIC" and (top_match.dense_score >= 0.55 or top_match.sparse_score >= 3.0) and abs(stage1.vader_scores.compound) < 0.20:
            try:
                fallback_label = SentimentLabel(top_match.sentiment)
            except ValueError:
                pass

        return Stage2LLMVerdict(
            sentiment_label=fallback_label,
            emotion_categories=[top_match.emotion_label if top_match else "Contextual / Subtle Tone"],
            confidence=round(max(0.72, stage1.confidence + 0.18), 4),
            interpretation=(
                f"Stage-2 Hybrid RAG evaluated borderline text (VADER compound={stage1.vader_scores.compound:+.3f}) "
                f"against Qdrant precedents to classify as {fallback_label.value}."
            ),
        )

    async def _stage2_reasoner_node(self, state: SentimentAgentState) -> Dict[str, Any]:
        feats: FeatureExtractionResult = state["features"]
        stage1: Stage1Result = state["stage1"]
        contexts: List[RetrievedContext] = state.get("retrieved_context", [])

        live_result = await self._call_live_llm_stage2(state["raw_text"], feats, stage1, contexts)
        if live_result is not None:
            verdict, provider = live_result
        else:
            verdict = self._synthesize_rag_semantic_verdict(feats, stage1, contexts)
            provider = "rag_semantic_synthesizer"

        plot_b64 = await render_sentiment_bar_async(stage1.vader_scores) if state["include_plot"] else None
        elapsed_ms = round((time.perf_counter() - state["t_start"]) * 1000.0, 3)

        response = SentimentResponse(
            request_id=state["request_id"],
            sentiment_label=verdict.sentiment_label,
            emotion_categories=verdict.emotion_categories,
            confidence=round(verdict.confidence, 4),
            routing_path=RoutingPath.STAGE2_RAG_LLM_DEEP,
            ambiguity_detected=bool(stage1.ambiguity_reasons),
            ambiguity_reasons=(
                stage1.ambiguity_reasons
                if stage1.ambiguity_reasons
                else (["forced_stage2"] if state.get("force_stage2") else ["low_stage1_confidence"])
            ),
            cache_hit=False,
            latency_ms=elapsed_ms,
            vader_scores=stage1.vader_scores,
            stage1_confidence=stage1.confidence,
            interpretation=verdict.interpretation,
            llm_provider_used=provider,
            features=feats,
            retrieved_context=contexts,
            chunk_sentiments=stage1.chunk_sentiments,
            plot_base64=plot_b64,
        )

        await self.cache.set(state["cache_key"], response)
        self.telemetry.record(
            routing_path=RoutingPath.STAGE2_RAG_LLM_DEEP.value,
            sentiment_label=response.sentiment_label.value,
            latency_ms=elapsed_ms,
            ambiguity_detected=response.ambiguity_detected,
            ambiguity_reasons=response.ambiguity_reasons,
            original_chars=feats.original_length,
            cleaned_chars=feats.cleaned_length,
            llm_provider=provider,
        )
        return {"final_response": response}

    # ------------------------------------------------------------------
    # Conditional Graph Edges & Compilation
    # ------------------------------------------------------------------
    @staticmethod
    def _route_after_cache(state: SentimentAgentState) -> str:
        return "hit" if state.get("cache_hit") else "miss"

    @staticmethod
    def _route_after_stage1(state: SentimentAgentState) -> str:
        stage1 = state["stage1"]
        if state.get("force_stage2") or stage1.requires_stage2:
            return "stage2"
        return "fast"

    def _compile_langgraph(self):
        graph = StateGraph(SentimentAgentState)
        graph.add_node("cache_lookup", self._cache_lookup_node)
        graph.add_node("preprocess", self._preprocess_node)
        graph.add_node("stage1_vader", self._stage1_vader_node)
        graph.add_node("fast_finalize", self._fast_finalize_node)
        graph.add_node("qdrant_rag", self._qdrant_rag_node)
        graph.add_node("stage2_reasoner", self._stage2_reasoner_node)

        graph.add_edge(START, "cache_lookup")
        graph.add_conditional_edges(
            "cache_lookup",
            self._route_after_cache,
            {"hit": END, "miss": "preprocess"},
        )
        graph.add_edge("preprocess", "stage1_vader")
        graph.add_conditional_edges(
            "stage1_vader",
            self._route_after_stage1,
            {"fast": "fast_finalize", "stage2": "qdrant_rag"},
        )
        graph.add_edge("fast_finalize", END)
        graph.add_edge("qdrant_rag", "stage2_reasoner")
        graph.add_edge("stage2_reasoner", END)

        return graph.compile()

    # ------------------------------------------------------------------
    # Public Async Entry Point
    # ------------------------------------------------------------------
    @traceable(name="manakizharchi_two_stage_pipeline", run_type="chain")
    async def analyze(self, req: AnalyzeRequest) -> SentimentResponse:
        await self.store.initialize()

        initial_state: SentimentAgentState = {
            "request_id": str(uuid.uuid4())[:12],
            "raw_text": req.text,
            "force_stage2": req.force_stage2,
            "include_pos_tags": req.include_pos_tags,
            "include_plot": req.include_plot,
            "confidence_threshold": req.confidence_threshold if req.confidence_threshold is not None else settings.confidence_threshold,
            "t_start": time.perf_counter(),
        }

        if self._graph is not None:
            result_state = await self._graph.ainvoke(initial_state)
            return result_state["final_response"]

        # Deterministic async fallback if LangGraph is disabled
        c1 = await self._cache_lookup_node(initial_state)
        initial_state.update(c1)
        if initial_state.get("cache_hit"):
            return initial_state["final_response"]
        s1 = await self._preprocess_node(initial_state)
        initial_state.update(s1)
        v1 = await self._stage1_vader_node(initial_state)
        initial_state.update(v1)
        if self._route_after_stage1(initial_state) == "fast":
            f1 = await self._fast_finalize_node(initial_state)
            return f1["final_response"]
        r2 = await self._qdrant_rag_node(initial_state)
        initial_state.update(r2)
        s2 = await self._stage2_reasoner_node(initial_state)
        return s2["final_response"]


_SHARED_ORCHESTRATOR: Optional[SentimentOrchestrator] = None


def get_orchestrator() -> SentimentOrchestrator:
    global _SHARED_ORCHESTRATOR
    if _SHARED_ORCHESTRATOR is None:
        _SHARED_ORCHESTRATOR = SentimentOrchestrator()
    return _SHARED_ORCHESTRATOR
