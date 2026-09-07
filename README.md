# Enterprise Hybrid RAG Engine

[![P99 latency](https://img.shields.io/badge/P99_latency-%3C300ms-brightgreen)](scripts/benchmark_latency.py)
[![Cache hit](https://img.shields.io/badge/Cache_hit-%3C12ms-blue)](app/cache.py)
[![Zero-Docker](https://img.shields.io/badge/Mode-Zero--Docker%20Embedded-success)](app/retrieval.py)
[![Cost](https://img.shields.io/badge/Cost-%240%20Groq%20free%20tier-lightgrey)](app/engine.py)

Local hybrid retrieval for enterprise documents: **BM25 + dense Qdrant + Reciprocal Rank Fusion + Cross-Encoder reranking + in-memory semantic cache + Groq Llama-3.3-70B**. No Docker. Low RAM MiniLM models. Groq free-tier synthesis, with extractive local fallback if `GROQ_API_KEY` is unset.

## Architecture

```
Query
  │
  ├─ Semantic cache (cosine ≥ 0.92) ── hit ──► RAGResponse (<12ms target)
  │
  └─ miss
        ├─ Dense search     all-MiniLM-L6-v2 → embedded Qdrant (cosine, ./qdrant_local_data)
        ├─ Sparse search    BM25Okapi
        ├─ Fusion           RRF  score(d) = Σ 1 / (k + rank + 1)   k = 60
        ├─ Rerank           cross-encoder/ms-marco-MiniLM-L-6-v2
        └─ Generate         Instructor + Groq llama-3.3-70b-versatile
                            or local extractive synthesis
```

## Setup

Python 3.10+ recommended. From the repo root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Set `GROQ_API_KEY` in `.env`. Leave it empty to run generation fully offline.

```
GROQ_API_KEY=
LLM_MODEL=llama-3.3-70b-versatile
```

Startup indexes three sample enterprise docs: **Q3 Financials**, **Security Policy**, and **SLA Guarantee**.

## Run

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/query ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"What was Q3 enterprise revenue and operating margin?\"}"
```

`POST /query` returns `RAGResponse`: `answer`, `confidence_score`, `sources_used`, `cache_hit`.

`POST /ingest` accepts additional `{id, text, metadata}` documents.

## Tests and reproduction

```bash
pytest -q
python eval/eval_ragas.py
python scripts/benchmark_latency.py --base-url http://127.0.0.1:8000 --requests 40 --concurrency 8
```

The latency script reports **P50 / P95 / P99** and **cache hit rate** against `http://127.0.0.1:8000/query`. Warm the process with a few unique questions first; repeated questions should hit the semantic cache.

## Design notes

- **Embedded Qdrant** uses `QdrantClient(path="./qdrant_local_data")`. No container, no server process.
- **MiniLM-L6** encoder and reranker keep the retrieval stack under a few hundred MB of RAM after load.
- **Semantic cache** stores query vectors, answers, and UTC timestamps in process memory.
- **$0 path**: local models for retrieve/rerank; Groq free tier for schema-governed answers via `https://api.groq.com/openai/v1`.
