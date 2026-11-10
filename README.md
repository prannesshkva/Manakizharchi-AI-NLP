# Manakizharchi AI v2.0

<p align="center">
  <strong>Hybrid NLP, Confidence-Gated Agentic Routing &amp; Qdrant RAG Sentiment Intelligence System</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-Async_ASGI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/LangGraph-StateGraph-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangGraph" />
  <img src="https://img.shields.io/badge/Qdrant-Hybrid_RRF-DC244C?style=for-the-badge&logo=qdrant&logoColor=white" alt="Qdrant" />
  <img src="https://img.shields.io/badge/NLTK_+_VADER-NLP_Pipeline-38BDF8?style=for-the-badge" alt="NLP" />
  <img src="https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge" alt="License" />
</p>

---

## Overview

**Manakizharchi AI** (*மனக்கிளர்ச்சி* — Affective Excitation / Emotional State) is an asynchronous, two-stage sentiment intelligence engine engineered to resolve **modern slang, sarcasm, code-mixed Tamil/Tanglish idioms, mixed-aspect clauses, and unstructured web noise** at production low latency.

Instead of sending every incoming text payload to expensive LLMs or relying solely on static bag-of-words lexicons that fail on irony and vernacular phrases, Manakizharchi AI pairs a **sub-millisecond VADER confidence gate** with **LangGraph conditional routing**, **Qdrant + BM25 Hybrid Retrieval (Reciprocal Rank Fusion)**, **NLTK POS feature extraction**, and **Pydantic v2 structured output enforcement**.

<p align="center">
  <img src="assets/architecture.svg" alt="Manakizharchi AI v2.0 Confidence-Gated Architecture" width="100%" />
</p>

---

## Core Architecture & Engineering Highlights

### 1. NLP Hygiene, POS Extraction & Semantic Chunking (`app/nlp/preprocessor.py`)
* **Unstructured Noise Normalization**: Strips HTML tags & entities, URLs, email handles, uppercase system/log metadata prefixes (`[ALERT]`, `[WARN]`, `[INFO]`), control/zero-width characters, elongation clutter (`"sooooo"` -> `"soo"`), and decomposes `#CamelCase` hashtags while preserving ironic emoji cues (`💀`, `🙄`, `🤡`, `😭`, `🔥`).
* **Linguistic Feature Extraction**: Runs NLTK tokenization, Treebank POS tagging, and WordNet POS-aware lemmatization (`_wordnet_pos`) to count expressive adjectives/adverbs/interjections alongside negation scope and contrast markers.
* **Discourse-Aware Semantic Chunking**: Splits compound statements along contrast conjunctions (`but`, `however`, `although`, `yet`, `whereas`) so opposing sentiment clauses remain isolated for chunk-level polarity inspection.

### 2. Stage-1 VADER Confidence Gating (`app/nlp/vader_engine.py`)
Computes compound polarity and per-chunk dispersion, calibrating Stage-1 confidence $C_1 \in [0, 1]$ minus explicit ambiguity penalties:

$$C_1 = \max\Big(0,\; \min\big(1,\; 0.58|s_{\text{comp}}| + 0.28|s_{\text{pos}} - s_{\text{neg}}| + 0.14(1 - s_{\text{neu}}) - \sum_k p_k\big)\Big)$$

* **Fast Path Exit (`STAGE1_VADER_FAST`)**: Clear, unambiguous texts ($C_1 \ge \tau$, default $\tau = 0.68$, zero critical flags) finalize immediately without touching Qdrant or LLMs.
* **Selective Escalation (`STAGE2_RAG_LLM_DEEP`)**: Texts exhibiting `slang_lexicon_gap`, `sarcasm_or_irony_cue`, `mixed_clause_contrast`, `oov_expressive_gap`, `negation_scope_shift`, or low confidence ($C_1 < \tau$) escalate down the Stage-2 conditional edge.

### 3. Stage-2 Qdrant Hybrid RRF & Agentic Orchestration (`app/rag/qdrant_store.py`, `app/agents/sentiment_agent.py`)
* **Hybrid Retrieval (Dense + Sparse + RRF)**: Queries Qdrant cosine similarity (`384d` fast subword dense projection or local `BAAI/bge-base-en-v1.5`) alongside sparse `BM25Okapi` lexical scoring, merged with Reciprocal Rank Fusion:

$$\text{RRF}(d) = \frac{1}{k + r_{\text{dense}}(d)} + \frac{1}{k + r_{\text{BM25}}(d)}, \qquad k = 60$$

* **LangGraph Execution & Deterministic Fallback**: Orchestrated as a compiled `StateGraph` (`cache_lookup -> [hit: END | miss: preprocess -> stage1_vader -> [fast | stage2]]`). Calls Groq (`llama-3.3-70b-versatile`) or Gemini (`gemini-2.0-flash`) asynchronously under strict Pydantic schema validation (`Stage2LLMVerdict`), automatically falling back to a deterministic RAG semantic synthesizer when API keys are omitted or rate-limited.
* **Pre-POS Async LRU+TTL Caching (`app/core/cache.py`)**: Keys canonical cleaned text via SHA-256 *before* NLTK POS tagging executes, returning repeated noisy variants (`CACHE_HIT`) cleanly.

---

## Verified Benchmark & System Performance

All numbers below are measured deterministically on CPU via `python scripts/benchmark_pipeline.py` (with zero external LLM API calls):

| Evaluation Suite | Metric | Baseline (Naive VADER) | Manakizharchi AI v2.0 | Verified Delta / SLA |
| :--- | :--- | :---: | :---: | :---: |
| **NLP Preprocessing Hygiene** *(16 noisy variant pairs)* | Token Inconsistency Rate | `71.92%` | `35.34%` | **50.86% Relative Reduction** |
| **6,250 Unstructured Inputs** *(Mixed Standard + Ambiguous)* | Overall Classification Accuracy | `68.34%` *(4,271 / 6,250)* | **`97.98%`** *(6,124 / 6,250)* | **+29.64% Accuracy Lift** |
| **Ambiguous Subset** *(n = 2,067: Slang, Sarcasm, Tanglish, Mixed)* | Subset Accuracy | `26.22%` *(542 / 2,067)* | **`94.05%`** *(1,944 / 2,067)* | **+67.83% Accuracy Lift** |
| **FastAPI Async HTTP Load** *(2,050 concurrent `/api/v2/analyze`)* | Latency Mean / P50 / P95 | — | **`115.24 ms`** / `104.47 ms` / `231.92 ms` | **197.0 req/s** &bull; `99.85%` Accuracy |

*Routing breakdown across the 2,050-request live ASGI load test*: `STAGE1_VADER_FAST: 663` | `STAGE2_RAG_LLM_DEEP: 378` | `CACHE_HIT: 1,009`.

---

## Interactive Web Dashboard (`frontend/`)

Served directly by FastAPI at `GET /` with zero frontend build step required:

1. **Hybrid Analyzer**: 1-click presets (Clear Positive, Gen-Z Slang, Sarcasm, Mixed Polarity, Tanglish Idiom, Noisy HTML+URLs), adjustable confidence threshold slider ($\tau$), NLTK POS tag pills, canonical hygiene tags, semantic chunk scores, and Qdrant RRF match cards.
2. **Batch Pipeline**: Paste multi-line unstructured inputs executed concurrently with tunable `asyncio.Semaphore` concurrency, displaying Stage-1 fast share, mean/P95 latency, and per-item routing verdicts.
3. **Social Engagement**: Evaluate captions and thread comment lists asynchronously in parallel (with optional public Instagram OG metadata extraction and dark-mode Matplotlib compound histograms).
4. **Qdrant KB & Telemetry**: Dynamically ingest custom idioms/domain phrases into Qdrant + BM25 in real time and inspect live FastAPI cache/latency/routing metrics.

---

## Repository Structure

```text
Manakizharchi-AI-NLP/
├── app/
│   ├── main.py                      # FastAPI ASGI entrypoint, lifespan bootstrap & dashboard mount
│   ├── core/
│   │   ├── config.py                # Typed environment configuration & LangSmith bootstrap
│   │   ├── embeddings.py            # NormalizedEmbedder (fast subword 384d & BAAI/bge-base-en-v1.5)
│   │   ├── cache.py                 # Async OrderedDict LRU+TTL cache with canonical SHA-256 keys
│   │   └── telemetry.py             # Latency percentiles (P50/P95/P99) & LangSmith traceable wrapper
│   ├── models/
│   │   └── schemas.py               # Strict Pydantic v2 request/response schemas & validators
│   ├── nlp/
│   │   ├── preprocessor.py          # Noise stripping, POS tagging, lemmatization & semantic chunker
│   │   └── vader_engine.py          # Stage-1 VADER scoring, chunk dispersion & confidence gate
│   ├── rag/
│   │   └── qdrant_store.py          # Qdrant dense cosine + BM25 sparse Hybrid RRF knowledge base
│   ├── agents/
│   │   └── sentiment_agent.py       # LangGraph StateGraph orchestrator + Groq/Gemini + fallback
│   ├── services/
│   │   ├── plotting.py              # Thread-safe Matplotlib dark-mode bar/histogram PNG generator
│   │   └── social_analyzer.py       # Async social post & parallel batch comment intelligence
│   └── api/
│       └── routes.py                # v2 REST routes & backward-compatible v1 endpoints
├── data/
│   └── ambiguity_corpus.py          # Curated affective KB (slang, Tanglish, sarcasm, mixed) & eval generator
├── frontend/
│   ├── index.html                   # 4-tab dark-mode Sentiment Intelligence UI
│   ├── style.css
│   └── app.js
├── assets/
│   └── architecture.svg             # Animated two-stage pipeline architecture diagram
├── scripts/
│   └── benchmark_pipeline.py        # Reproducible 6,250-input & 2,050-HTTP load benchmark
├── tests/
│   └── test_pipeline.py             # Pytest unit, RAG & end-to-end FastAPI integration suite
├── main.py                          # Root ASGI launcher (`uvicorn main:app`)
├── requirements.txt
├── Dockerfile
├── vercel.json
└── .env.example
```

---

## Quickstart

### 1. Local Installation

```bash
git clone https://github.com/prannesshkva/Manakizharchi-AI-NLP.git
cd Manakizharchi-AI-NLP

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
# source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

*(Zero API keys are required to run — optional `GROQ_API_KEY`, `GEMINI_API_KEY`, `QDRANT_LOCATION`, and `LANGSMITH_API_KEY` can be configured in `.env`.)*

### 2. Run the FastAPI Server & Dashboard

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open:
* **Web Dashboard**: [http://localhost:8000](http://localhost:8000)
* **Interactive Swagger API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

### 3. Run Verification Tests & Benchmark

```bash
# Run unit & end-to-end FastAPI test suite
pytest -v

# Run reproducible 6,250-input & 2,050-request async load benchmark
python scripts/benchmark_pipeline.py
```

### 4. Docker Deployment

```bash
docker build -t manakizharchi-ai .
docker run -p 8000:8000 --env-file .env manakizharchi-ai
```

---

## API Reference

| Method | Endpoint | Purpose |
| :--- | :--- | :--- |
| `GET` | `/health` | Service status, LangGraph toggle, Qdrant vector count, embedding dimension & cache stats |
| `POST` | `/api/v2/analyze` | Two-stage sentiment analysis with POS extraction, gating diagnostics & Qdrant RRF evidence |
| `POST` | `/api/v2/analyze/batch` | Async concurrent batch pipeline (up to 2,500 inputs via `asyncio.Semaphore`) |
| `POST` | `/api/v2/ingest` | Dynamically chunk and index new slang/domain phrases into Qdrant + BM25 |
| `GET` | `/api/v2/corpus` | Inspect indexed Qdrant affective precedent vectors |
| `POST` | `/api/v2/social/analyze` | Parallel comment thread evaluation, dominant verdict & Matplotlib chart |
| `GET` | `/api/v2/metrics` | Live latency percentiles (`P50`/`P95`/`P99`), noise reduction avg & routing counters |

### Example Request (`POST /api/v2/analyze`)

```bash
curl -X POST http://localhost:8000/api/v2/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "im literally dead 💀😂 this beat drop is straight fire no cap", "include_pos_tags": true}'
```

---

## Configuration (`.env.example`)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `CONFIDENCE_THRESHOLD` | `0.68` | Gate threshold $\tau$ for Stage-1 VADER fast exit vs. Stage-2 escalation |
| `CACHE_MAX_SIZE` | `2048` | Max canonical responses cached in the async LRU store |
| `CACHE_TTL_SECONDS` | `3600` | Cache entry time-to-live in seconds |
| `QDRANT_LOCATION` | `:memory:` | Qdrant storage (`:memory:`, local directory path, or Qdrant Cloud URL) |
| `EMBEDDING_BACKEND` | `auto` | `auto` / `fast` (384d subword projection) or `transformer` (`BAAI/bge-base-en-v1.5`) |
| `GROQ_API_KEY` | `""` | Optional Groq API key (`llama-3.3-70b-versatile`) for Stage-2 LLM reasoning |
| `GEMINI_API_KEY` | `""` | Optional Google Gemini API key (`gemini-2.0-flash`) fallback |
| `LANGSMITH_TRACING` | `false` | Enable `@traceable` LangSmith telemetry when `LANGSMITH_API_KEY` is provided |

---

## Author

* **Prannessh KVA** — [github.com/prannesshkva](https://github.com/prannesshkva)

## License

This project is licensed under the MIT License.
