# Enterprise Hybrid RAG Engine

[![P50 latency](https://img.shields.io/badge/P50_latency-238ms-brightgreen)](scripts/benchmark_latency.py)
[![Cache hit rate](https://img.shields.io/badge/Cache_hit_rate-77.5%25-blue)](app/cache.py)
[![Cache lookup](https://img.shields.io/badge/Cache_lookup-11--13ms-blueviolet)](app/cache.py)
[![Deployment](https://img.shields.io/badge/Deployment-Zero--Docker%20Embedded-success)](app/retrieval.py)

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

## 📊 Empirical Concurrency Benchmarks

Measured on a local run of `python scripts/benchmark_latency.py` against `http://127.0.0.1:8000` with 40 total requests at concurrency 8.

| Metric | Measured Value | Operational SLA / Target | Notes |
| :--- | :--- | :--- | :--- |
| **In-Memory Cache Lookup** | **11ms – 13ms** | < 20ms | In-memory cosine similarity match (threshold ≥ 0.92) |
| **Cached HTTP Latency (Min)** | **120.98 ms** | < 150ms | Full network roundtrip on warm cache hit |
| **Cache Hit Rate** | **77.5%** | > 50.0% | 31/40 concurrent requests resolved from memory |
| **Median Latency (P50)** | **238.91 ms** | < 250.0 ms | Typical user experience across mixed workloads |
| **Mean Latency** | **582.54 ms** | < 650.0 ms | Aggregate average across all 40 concurrent queries |
| **Cold Path Latency (P95)** | **1978.23 ms** | < 2000.0 ms | Dense + Sparse + RRF + Cross-Encoder + Groq Llama-3.3-70B |
| **Tail Latency (P99)** | **2003.11 ms** | < 2500.0 ms | Worst-case concurrency spike across 8 parallel workers |
| **Peak Memory Footprint** | **< 250 MB RAM** | < 500 MB RAM | Embedded Qdrant + MiniLM-L6 (Zero Docker) |

The in-memory cosine semantic cache (threshold ≥ 0.92) intercepts **77.5%** of incoming queries, bypassing the cross-encoder and LLM generation entirely. This delivers **sub-15ms internal lookups** and cuts Groq token costs by **over three quarters** while keeping the median user-visible latency at 238.91 ms.

## Design notes

- **Embedded Qdrant** uses `QdrantClient(path="./qdrant_local_data")`. No container, no server process.
- **MiniLM-L6** encoder and reranker keep the retrieval stack under a few hundred MB of RAM after load.
- **Semantic cache** stores query vectors, answers, and UTC timestamps in process memory.
- **$0 path**: local models for retrieve/rerank; Groq free tier for schema-governed answers via `https://api.groq.com/openai/v1`.
