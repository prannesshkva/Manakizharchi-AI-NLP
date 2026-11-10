import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    app_name: str = "Manakizharchi AI"
    app_version: str = "2.0.0"
    app_description: str = (
        "Two-stage Hybrid NLP & Agentic Sentiment Intelligence System combining "
        "NLTK POS preprocessing, VADER confidence gating, Qdrant Hybrid RRF RAG, "
        "and asynchronous FastAPI + LangGraph orchestration."
    )

    # Confidence Gating & Chunking Parameters
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.68"))
    max_chunk_tokens: int = int(os.getenv("MAX_CHUNK_TOKENS", "48"))
    chunk_overlap_tokens: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "8"))

    # In-Memory TTL + LRU Caching
    cache_max_size: int = int(os.getenv("CACHE_MAX_SIZE", "2048"))
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))

    # Qdrant Vector Store Configuration
    qdrant_location: str = os.getenv("QDRANT_LOCATION", ":memory:")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "manakizharchi_ambiguity_kb")

    # Embedding Configuration
    embedding_backend: str = os.getenv("EMBEDDING_BACKEND", "auto")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")

    # Optional Cloud LLM Providers (Groq / Gemini)
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    llm_timeout_seconds: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "6.0"))

    # LangSmith Telemetry Configuration
    langsmith_tracing: bool = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    langsmith_endpoint: str = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    langsmith_api_key: str = os.getenv("LANGSMITH_API_KEY", "")
    langsmith_project: str = os.getenv("LANGSMITH_PROJECT", "manakizharchi-ai")


settings = Settings()

if settings.langsmith_api_key and settings.langsmith_tracing:
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
