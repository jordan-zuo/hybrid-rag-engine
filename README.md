# hybrid-rag-engine: Local Hybrid Retrieval with LLM-as-a-Judge Evaluation

[![Cached P50](https://img.shields.io/badge/Cached_P50-14ms-brightgreen)](scripts/benchmark_latency.py)
[![Cold P95](https://img.shields.io/badge/Cold_P95-3.2s-yellow)](scripts/benchmark_latency.py)
[![Cache hit rate](https://img.shields.io/badge/Cache_hit_rate-100%25_%28repeat%29-blue)](app/cache.py)
[![Deployment](https://img.shields.io/badge/Deployment-Zero--Docker%20Embedded-success)](app/retrieval.py)

Hybrid RAG engine over enterprise documents. Dense MiniLM embeddings in embedded Qdrant cover semantic match. BM25 covers exact terms such as model numbers and statute names. Reciprocal Rank Fusion merges both rankings. A Cross-Encoder reranks the shortlist. An in-memory semantic cache serves repeats. Schema-constrained LLM synthesis writes the answers. Generation runs on llm7 mistral-Nemo-Instruct-2407. Grading runs on Groq qwen/qwen3.8-27b. No Docker. Retrieval and reranking run on local MiniLM models.

## Architecture

```
Query
  │
  ├─ Semantic cache (cosine >= 0.92) ── hit ──> RAGResponse (about 14 ms measured)
  │
  └─ miss
        ├─ Dense search     all-MiniLM-L6-v2 -> embedded Qdrant (cosine, ./qdrant_local_data)
        ├─ Sparse search    BM25Okapi over indexed documents
        ├─ Fusion           RRF  score(d) = SUM 1 / (k + rank + 1)   k = 60
        ├─ Rerank           cross-encoder/ms-marco-MiniLM-L-6-v2, top_k = 3
        └─ Generate         Instructor (JSON mode) + mistral-Nemo-Instruct-2407 via llm7.
                            LLM errors raise as 500s since no extractive fallback exists.
                            Empty retrieval or non-positive confidence returns a refusal
                            before the LLM is ever called.
```

`POST /query` returns a `RAGResponse` object with answer, confidence score, sources used, cache flag and latency. `POST /ingest` accepts extra documents with id, text and metadata. Startup indexes three sample documents covering Q3 Financials, Security Policy and SLA Guarantee.

## Architectural decisions and constraints

1. **No RAGAS and no LangChain eval chain.** RAGAS routes its judge through fixed OpenAI model assumptions. Against other endpoints the full run collapsed to NaN after a single bad model ID failed all 18 jobs. The replacement is `eval/evaluate.py`. Source ID checks against ground truth labels cover precision and recall. A plain OpenAI compatible judge call with a structured JSON verdict covers faithfulness and relevancy. The harness pulls in no third-party eval dependency.
2. **Split providers for generation and grading.** Generation runs on llm7 where mistral-Nemo-Instruct-2407 returned 200 with json_mode support. Grading runs on Groq qwen/qwen3.8-27b which reasons better and only needs six sequential eval time calls. Separate models also remove self-preference bias. Verified free on the current keys were mistral-Nemo-Instruct-2407 and codestral-latest. gemma4:31b and DeepSeek-V4-Flash-0731 returned 402.
3. **Instructor in JSON mode, not tool mode.** mistral-Nemo-Instruct-2407 supports json_mode but not tool calling. Default tool mode structured output failed with 400. The client is built with `mode=instructor.Mode.JSON`. Each request caps max_tokens at 300 because free tier output token budgets reject larger requests with 429.
4. **Fail loud synthesis.** An earlier revision caught all LLM exceptions and returned copied context sentences. Benchmarks looked fast while outages stayed hidden. That path is deleted. Missing keys and API errors raise visibly. Only the out of domain refusal answers without an LLM call.
5. **Module path and cold start.** The repo ships no installed package. Evaluation and tests run with PYTHONPATH pointed at the repo root. Loading MiniLM plus Cross-Encoder takes around 30 seconds on first use. Test suite wall time reflects this. Wait it out instead of treating it as a hang.

## Setup and reproduction

PowerShell, from the repo root.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and save the file afterwards. The server only reads what is on disk.

```
LLM_API_KEY=          # generation provider (llm7)
LLM_MODEL=mistral-Nemo-Instruct-2407
GROQ_API_KEY=         # eval judge provider (Groq; also used as JUDGE_API_KEY fallback)
JUDGE_MODEL=qwen/qwen3.8-27b
```

Run the server in one terminal and evaluate in another. Never run both at once. Embedded Qdrant takes an exclusive file lock, so stop the server before running eval.

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

```powershell
(Invoke-RestMethod http://127.0.0.1:8000/health | ConvertTo-Json)
pytest -q
```

```powershell
$env:PYTHONPATH = "."
python -m eval.evaluate                      # quality: needs server STOPPED
python scripts/benchmark_latency.py --base-url http://127.0.0.1:8000   # latency: needs server RUNNING
```

The benchmark defaults stay inside free-tier quota. Pass explicit flags for stress runs.

## Benchmarks

Cache and cold paths were measured separately on the llm7 free tier. A blended number across a bimodal distribution describes neither path, so the table reports both.

| Metric | Measured Value | Target | Notes |
| :--- | :--- | :--- | :--- |
| Cached latency (P50, 12 req, conc 1, 100 percent hits) | 14.50 ms | < 20 ms | Repeat workload. Full HTTP roundtrip with cache hit set |
| Cached latency (min) | 13.49 ms | < 150 ms | Floor across all runs |
| Cache hit rate (repeat workload) | 100.0% | > 50.0% | 12 of 12. Mixed 40/8 reached 75.0 percent |
| Cold latency (P50, 6 req, conc 1, fresh cache) | 2555.60 ms | < 4000 ms | Sequential genuine llm7 answers without queueing |
| Cold latency (P95) | 3217.11 ms | < 4000 ms | Worst sequential cold call took 3330.13 ms |
| Cold latency (mean) | 1891.50 ms | — | Below P50 because two duplicate queries hit cache mid run |
| Peak memory footprint | < 250 MB RAM | < 500 MB RAM | Embedded Qdrant plus MiniLM-L6 with no containers |

Cached hits return in about 14 ms. Cold llm7 inference costs 2.4 to 3.3 s per call. Every miss is a grounded answer verifiable through sources_used.

## Quality evaluations

`eval/dataset.json` holds six ground truth pairs. Each pairs a question with its reference answer and expected source ID. Precision and recall are exact label checks. Faithfulness and relevancy are qwen judge verdicts. The judge reads the answer with the retrieved contexts and the reference.

| Metric | Measured Score | Threshold | Status |
| :--- | :--- | :--- | :--- |
| Faithfulness (groundedness) | 1.00 | > 0.90 | Pass, 6/6 |
| Context precision | 1.00 | > 0.85 | Pass, 6/6 |
| Context recall | 1.00 | > 0.85 | Pass, 6/6 |
| Answer relevancy | 1.00 | > 0.90 | Pass, 6/6 |

Hallucination rate is 1 minus faithfulness. It measured 0.0 percent. Per-case verdicts and judge rationales sit in `eval/results.json` next to the generating and judging model IDs.

## Known limitations

- Free-tier output-token budgets bound generation concurrency. The benchmark defaults reflect that. Pushing concurrency past the quota surfaces provider 429s as 500s by design.
- Semantic cache is in-process memory. A server restart clears it. It is not shared across workers.
- `qdrant_local_data/` is single-access embedded storage. One process at a time. Server or eval, never both.
