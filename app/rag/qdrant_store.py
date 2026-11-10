import asyncio
import uuid
from typing import Any, Dict, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.core.embeddings import NormalizedEmbedder, get_embedder
from app.models.schemas import RetrievedContext
from app.nlp.preprocessor import extract_features, semantic_chunk_text
from data.ambiguity_corpus import SEED_AMBIGUITY_CORPUS


class HybridQdrantStore:
    """
    Production Hybrid Retrieval Engine backed by Qdrant (Dense Cosine Similarity)
    and BM25Okapi (Sparse Lexical Token Scoring), fused via Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, embedder: Optional[NormalizedEmbedder] = None):
        self.embedder = embedder or get_embedder()
        self.collection_name = settings.qdrant_collection
        self.client = QdrantClient(
            location=settings.qdrant_location,
            api_key=settings.qdrant_api_key or None,
        )
        self._lock = asyncio.Lock()
        self._corpus_records: List[Dict[str, Any]] = []
        self._bm25: Optional[BM25Okapi] = None
        self._initialized = False

    def _tokenize_for_bm25(self, text: str) -> List[str]:
        feats = extract_features(text, include_pos_tags=False)
        tokens = list(feats.lemmas or feats.tokens or text.lower().split())
        # Include raw lowered words too so exact slang forms like "mid", "tbh" match cleanly
        for t in feats.tokens:
            if t not in tokens:
                tokens.append(t)
        return tokens if tokens else text.lower().split()

    def _rebuild_bm25(self) -> None:
        tokenized = [self._tokenize_for_bm25(rec["text"]) for rec in self._corpus_records]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def initialize_sync(self) -> int:
        if self._initialized:
            return len(self._corpus_records)

        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.embedder.dim, distance=Distance.COSINE),
            )

        existing_count = self.client.count(collection_name=self.collection_name).count
        if existing_count == 0:
            texts = [item["text"] for item in SEED_AMBIGUITY_CORPUS]
            embeddings = self.embedder.encode(texts, normalize_embeddings=True)
            points = []
            for idx, item in enumerate(SEED_AMBIGUITY_CORPUS):
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"seed:{idx}:{item['text']}"))
                payload = {
                    "text": item["text"],
                    "emotion_label": item["emotion_label"],
                    "sentiment": item["sentiment"],
                    "category": item["category"],
                    "source": "seed_ambiguity_kb",
                }
                points.append(
                    PointStruct(id=point_id, vector=embeddings[idx].tolist(), payload=payload)
                )
                self._corpus_records.append(payload)

            self.client.upsert(collection_name=self.collection_name, points=points)
        else:
            # Hydrate local mirror for BM25 from persistent Qdrant collection
            scrolled, _ = self.client.scroll(collection_name=self.collection_name, limit=5000, with_payload=True)
            self._corpus_records = [p.payload for p in scrolled if p.payload]

        self._rebuild_bm25()
        self._initialized = True
        return len(self._corpus_records)

    async def initialize(self) -> int:
        if self._initialized:
            return len(self._corpus_records)
        async with self._lock:
            if self._initialized:
                return len(self._corpus_records)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self.initialize_sync)

    async def ingest_document(
        self,
        text: str,
        emotion_label: str,
        sentiment: str,
        category: str = "custom_domain",
        source: str = "user_ingest",
    ) -> int:
        await self.initialize()
        chunks = semantic_chunk_text(text)
        if not chunks:
            return 0

        async with self._lock:
            loop = asyncio.get_running_loop()

            def _do_upsert() -> int:
                embeddings = self.embedder.encode(chunks, normalize_embeddings=True)
                points = []
                for idx, chunk in enumerate(chunks):
                    point_id = str(uuid.uuid4())
                    payload = {
                        "text": chunk,
                        "emotion_label": emotion_label,
                        "sentiment": sentiment,
                        "category": category,
                        "source": source,
                    }
                    points.append(PointStruct(id=point_id, vector=embeddings[idx].tolist(), payload=payload))
                    self._corpus_records.append(payload)

                self.client.upsert(collection_name=self.collection_name, points=points)
                self._rebuild_bm25()
                return len(chunks)

            return await loop.run_in_executor(None, _do_upsert)

    def _hybrid_search_sync(self, query_text: str, top_k: int = 4) -> List[RetrievedContext]:
        if not self._initialized:
            self.initialize_sync()

        if not self._corpus_records:
            return []

        query_vec = self.embedder.encode([query_text], normalize_embeddings=True)[0].tolist()
        candidate_limit = min(max(top_k * 4, 12), len(self._corpus_records))

        # 1. Qdrant Dense Vector Search
        qdrant_res = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vec,
            limit=candidate_limit,
            with_payload=True,
        )
        dense_points = qdrant_res.points if hasattr(qdrant_res, "points") else qdrant_res

        dense_ranks: Dict[str, tuple[int, float, Dict[str, Any]]] = {}
        for rank_idx, pt in enumerate(dense_points, start=1):
            payload = pt.payload or {}
            txt = payload.get("text", "")
            if txt not in dense_ranks:
                dense_ranks[txt] = (rank_idx, float(pt.score), payload)

        # 2. Sparse Lexical BM25 Search
        sparse_ranks: Dict[str, tuple[int, float, Dict[str, Any]]] = {}
        if self._bm25 is not None:
            q_toks = self._tokenize_for_bm25(query_text)
            bm25_scores = self._bm25.get_scores(q_toks)
            indexed_scores = sorted(
                enumerate(bm25_scores),
                key=lambda x: x[1],
                reverse=True,
            )[:candidate_limit]
            for rank_idx, (doc_idx, score) in enumerate(indexed_scores, start=1):
                if score <= 0:
                    continue
                payload = self._corpus_records[doc_idx]
                txt = payload["text"]
                if txt not in sparse_ranks:
                    sparse_ranks[txt] = (rank_idx, float(score), payload)

        # 3. Reciprocal Rank Fusion (RRF, k = 60)
        k_rrf = 60.0
        fused: List[RetrievedContext] = []
        all_keys = set(dense_ranks.keys()) | set(sparse_ranks.keys())

        for txt in all_keys:
            rrf_score = 0.0
            dense_score = 0.0
            sparse_score = 0.0
            payload = None

            if txt in dense_ranks:
                r_d, d_sc, payload = dense_ranks[txt]
                rrf_score += 1.0 / (k_rrf + r_d)
                dense_score = d_sc

            if txt in sparse_ranks:
                r_s, s_sc, sp_payload = sparse_ranks[txt]
                rrf_score += 1.0 / (k_rrf + r_s)
                sparse_score = s_sc
                if payload is None:
                    payload = sp_payload

            if payload:
                fused.append(
                    RetrievedContext(
                        text=payload["text"],
                        emotion_label=payload["emotion_label"],
                        sentiment=payload["sentiment"],
                        category=payload["category"],
                        rrf_score=round(rrf_score, 6),
                        dense_score=round(dense_score, 4),
                        sparse_score=round(sparse_score, 4),
                    )
                )

        fused.sort(key=lambda item: item.rrf_score, reverse=True)
        return fused[:top_k]

    async def hybrid_search(self, query_text: str, top_k: int = 4) -> List[RetrievedContext]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self._hybrid_search_sync(query_text, top_k=top_k))

    def list_corpus(self, category: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        if not self._initialized:
            self.initialize_sync()
        items = self._corpus_records
        if category:
            items = [i for i in items if i.get("category") == category]
        return items[:limit]

    def corpus_count(self) -> int:
        if not self._initialized:
            self.initialize_sync()
        return len(self._corpus_records)


_SHARED_STORE: Optional[HybridQdrantStore] = None


def get_qdrant_store() -> HybridQdrantStore:
    global _SHARED_STORE
    if _SHARED_STORE is None:
        _SHARED_STORE = HybridQdrantStore()
    return _SHARED_STORE
