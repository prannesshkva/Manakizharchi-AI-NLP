from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class SentimentLabel(str, Enum):
    POSITIVE = "POSITIVE"
    NEUTRAL = "NEUTRAL"
    NEGATIVE = "NEGATIVE"
    MIXED = "MIXED"
    SARCASTIC = "SARCASTIC"


class RoutingPath(str, Enum):
    CACHE_HIT = "CACHE_HIT"
    STAGE1_VADER_FAST = "STAGE1_VADER_FAST"
    STAGE2_RAG_LLM_DEEP = "STAGE2_RAG_LLM_DEEP"


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=15000, description="Unstructured input text to analyze")
    force_stage2: bool = Field(
        default=False,
        description="Bypass confidence gate and force Qdrant Hybrid RAG + deep semantic evaluation",
    )
    include_pos_tags: bool = Field(default=True, description="Include NLTK POS tags and lemmatization details")
    include_plot: bool = Field(default=False, description="Include Base64 PNG plot for v1 client compatibility")
    confidence_threshold: Optional[float] = Field(
        default=None,
        ge=0.1,
        le=0.99,
        description="Optional override for Stage-1 VADER confidence gate threshold",
    )

    @field_validator("text")
    @classmethod
    def clean_and_validate_text(cls, v: str) -> str:
        cleaned = v.replace("\x00", "").strip()
        if not cleaned:
            raise ValueError("Input text cannot be empty or whitespace-only after validation.")
        return cleaned


class BatchAnalyzeRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=2500, description="Batch of unstructured text inputs")
    concurrency: int = Field(default=32, ge=1, le=128, description="Max asyncio concurrent workers")
    confidence_threshold: Optional[float] = Field(default=None, ge=0.1, le=0.99)
    include_pos_tags: bool = Field(default=False, description="Omit full POS token arrays for compact batch payloads")
    include_items: bool = Field(default=True, description="Include individual per-item results")

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, items: List[str]) -> List[str]:
        cleaned = []
        for idx, item in enumerate(items):
            val = (item or "").replace("\x00", "").strip()
            if not val:
                raise ValueError(f"Item at index {idx} is empty or whitespace-only.")
            if len(val) > 5000:
                val = val[:5000]
            cleaned.append(val)
        return cleaned


class IngestDocumentRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=50000, description="Text or idiom to semantically chunk and index")
    emotion_label: str = Field(..., min_length=2, max_length=120, description="Target emotion tag (e.g. 'Excited', 'Ironic')")
    sentiment: SentimentLabel = Field(default=SentimentLabel.NEUTRAL, description="Canonical polarity label")
    category: str = Field(default="custom_domain", max_length=64)
    source: str = Field(default="user_ingest", max_length=120)

    @field_validator("text", "emotion_label")
    @classmethod
    def strip_text(cls, v: str) -> str:
        cleaned = v.replace("\x00", "").strip()
        if not cleaned:
            raise ValueError("Field cannot be blank.")
        return cleaned


class SocialAnalyzeRequest(BaseModel):
    post_url: Optional[str] = Field(default=None, description="Public Instagram post/reel URL")
    caption: Optional[str] = Field(default=None, max_length=10000, description="Optional direct caption text")
    comments: Optional[List[str]] = Field(
        default=None,
        max_length=500,
        description="Optional list of audience comments for live async batch scoring",
    )
    session_id: Optional[str] = Field(default=None, description="Optional Instagram session ID")
    include_plot: bool = Field(default=True, description="Generate audience sentiment distribution plot")

    @model_validator(mode="after")
    def require_input_source(self) -> "SocialAnalyzeRequest":
        has_url = bool(self.post_url and self.post_url.strip())
        has_caption = bool(self.caption and self.caption.strip())
        has_comments = bool(self.comments and any(c.strip() for c in self.comments))
        if not (has_url or has_caption or has_comments):
            raise ValueError("Provide at least one of: post_url, caption, or comments.")
        return self


class POSToken(BaseModel):
    token: str
    tag: str
    pos_group: str
    lemma: str


class FeatureExtractionResult(BaseModel):
    original_length: int
    cleaned_length: int
    noise_reduction_pct: float
    noise_tags_removed: List[str]
    canonical_text: str
    tokens: List[str]
    lemmas: List[str]
    pos_distribution: Dict[str, int]
    pos_tags_sample: List[POSToken] = Field(default_factory=list)
    semantic_chunks: List[str]
    negation_terms: List[str]
    slang_markers: List[str]
    contrast_markers: List[str]
    ironic_emojis: List[str]


class VaderScores(BaseModel):
    positive: float
    neutral: float
    negative: float
    compound: float


class ChunkSentiment(BaseModel):
    chunk_index: int
    text: str
    compound: float
    positive: float
    negative: float
    neutral: float


class RetrievedContext(BaseModel):
    text: str
    emotion_label: str
    sentiment: str
    category: str
    rrf_score: float
    dense_score: float
    sparse_score: float


class SentimentResponse(BaseModel):
    request_id: str
    sentiment_label: SentimentLabel
    emotion_categories: List[str]
    confidence: float
    routing_path: RoutingPath
    ambiguity_detected: bool
    ambiguity_reasons: List[str]
    cache_hit: bool
    latency_ms: float
    vader_scores: VaderScores
    stage1_confidence: float
    interpretation: str
    llm_provider_used: str
    features: FeatureExtractionResult
    retrieved_context: List[RetrievedContext] = Field(default_factory=list)
    chunk_sentiments: List[ChunkSentiment] = Field(default_factory=list)
    plot_base64: Optional[str] = None


class BatchAnalyzeResponse(BaseModel):
    total_inputs: int
    stage1_fast_count: int
    stage2_deep_count: int
    cache_hit_count: int
    stage1_fast_pct: float
    ambiguities_resolved: int
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    total_wall_time_ms: float
    avg_noise_reduction_pct: float
    sentiment_distribution: Dict[str, int]
    items: List[SentimentResponse] = Field(default_factory=list)


class IngestResponse(BaseModel):
    status: str
    chunks_indexed: int
    total_corpus_size: int
    category: str
