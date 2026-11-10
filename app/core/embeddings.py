import hashlib
import re
from typing import List, Union
import numpy as np

from app.core.config import settings

_SHARED_EMBEDDER = None


class NormalizedEmbedder:
    """
    L2-normalized dense text embedder supporting:
      1. Transformer embeddings (BAAI/bge-base-en-v1.5 or all-MiniLM-L6-v2) when
         installed and configured via EMBEDDING_BACKEND=transformer.
      2. Deterministic 384-dim Subword & Lexical Semantic Projection for zero-latency
         execution (~0.25ms/query) without network/GPU dependencies.
    """

    def __init__(self, model_name: str = None, backend: str = None, fast_dim: int = 384):
        self.model_name = model_name or settings.embedding_model
        self.requested_backend = (backend or settings.embedding_backend).lower()
        self.fast_dim = fast_dim
        self._model = None
        self._initialized = False
        self._active_backend = "fast_subword_dense"
        self._dim = fast_dim

    def _init_backend(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        if self.requested_backend == "transformer":
            try:
                from sentence_transformers import SentenceTransformer

                try:
                    self._model = SentenceTransformer(self.model_name, local_files_only=True)
                except Exception:
                    self._model = SentenceTransformer(self.model_name)
                sample = self._model.encode(["ping"], normalize_embeddings=True, show_progress_bar=False)
                self._dim = int(sample.shape[1])
                self._active_backend = f"transformer:{self.model_name}"
                return
            except Exception:
                self._model = None

        self._active_backend = "fast_subword_dense"
        self._dim = self.fast_dim

    @property
    def dim(self) -> int:
        self._init_backend()
        return self._dim

    @property
    def active_backend(self) -> str:
        self._init_backend()
        return self._active_backend

    def _project_token(self, token: str, weight: float, vec: np.ndarray) -> None:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
        # Scatter into 4 signed hash buckets (feature-hashing / count-sketch style)
        # for sub-millisecond deterministic orthogonality.
        dim = self._dim
        for k in range(4):
            raw = int.from_bytes(digest[k * 4 : (k + 1) * 4], "little")
            bucket = (raw >> 1) % dim
            sign = 1.0 if (raw & 1) == 0 else -1.0
            vec[bucket] += sign * weight

    def _fast_encode_single(self, text: str) -> np.ndarray:
        cleaned = re.sub(r"\s+", " ", (text or "").lower().strip())
        vec = np.zeros(self._dim, dtype=np.float32)
        if not cleaned:
            return vec

        words = re.findall(r"[a-z0-9_]+|[^\w\s]", cleaned)
        if not words:
            return vec

        # 1. Weighted unigrams (strong semantic signal)
        for word in words:
            w = 1.4 if len(word) >= 3 else 0.7
            self._project_token(f"w:{word}", w, vec)
            # Character 3-grams for morphological / spelling tolerance
            if len(word) >= 4:
                padded = f"^{word}$"
                for i in range(len(padded) - 2):
                    self._project_token(f"c3:{padded[i:i+3]}", 0.35, vec)

        # 2. Bigrams for phrase & negation idioms ("no cap", "not good", "straight fire")
        for i in range(len(words) - 1):
            self._project_token(f"bg:{words[i]}_{words[i+1]}", 1.8, vec)

        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
            vec = (vec / norm).astype(np.float32)
        return vec

    def encode(self, texts: Union[str, List[str]], normalize_embeddings: bool = True) -> np.ndarray:
        self._init_backend()
        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)
        if not batch:
            return np.zeros((0, self._dim), dtype=np.float32)

        if self._model is not None:
            try:
                emb = self._model.encode(
                    batch,
                    normalize_embeddings=normalize_embeddings,
                    show_progress_bar=False,
                )
                return np.asarray(emb, dtype=np.float32)
            except Exception:
                pass

        vectors = [self._fast_encode_single(t) for t in batch]
        return np.vstack(vectors).astype(np.float32)


def get_embedder() -> NormalizedEmbedder:
    global _SHARED_EMBEDDER
    if _SHARED_EMBEDDER is None:
        _SHARED_EMBEDDER = NormalizedEmbedder()
    return _SHARED_EMBEDDER
