from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from collections.abc import Sequence

import httpx
import numpy as np

QUERIES = [
    "What was Q3 enterprise revenue and operating margin?",
    "What was Q3 enterprise revenue and operating margin?",
    "How often must API keys be rotated?",
    "How often must API keys be rotated?",
    "What uptime does the SLA guarantee?",
    "What credit applies if uptime falls below 99.9%?",
    "Is the control environment SOC 2 Type II certified?",
    "What is the P1 incident response target?",
]


async def _one_query(client: httpx.AsyncClient, url: str, query: str) -> tuple[float, bool]:
    started = time.perf_counter()
    response = await client.post(url, json={"query": query})
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.raise_for_status()
    payload = response.json()
    return elapsed_ms, bool(payload.get("cache_hit"))


async def run_benchmark(base_url: str, concurrency: int, requests: int) -> None:
    url = f"{base_url.rstrip('/')}/query"
    queries = [QUERIES[i % len(QUERIES)] for i in range(requests)]
    latencies: list[float] = []
    cache_hits = 0

    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(timeout=60.0, limits=limits) as client:
        health = await client.get(f"{base_url.rstrip('/')}/health")
        health.raise_for_status()
        print(f"health: {health.json()}")

        semaphore = asyncio.Semaphore(concurrency)

        async def bound(query: str) -> tuple[float, bool]:
            async with semaphore:
                return await _one_query(client, url, query)

        results = await asyncio.gather(*[bound(query) for query in queries])

    for latency_ms, hit in results:
        latencies.append(latency_ms)
        cache_hits += int(hit)

    _print_report(latencies, cache_hits, requests, concurrency)


def _print_report(
    latencies: Sequence[float],
    cache_hits: int,
    requests: int,
    concurrency: int,
) -> None:
    arr = np.asarray(latencies, dtype=np.float64)
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    hit_rate = cache_hits / requests if requests else 0.0
    print("\n=== Hybrid RAG latency benchmark ===")
    print(f"requests={requests} concurrency={concurrency}")
    print(f"mean={statistics.fmean(latencies):.2f} ms")
    print(f"P50={p50:.2f} ms  P95={p95:.2f} ms  P99={p99:.2f} ms")
    print(f"cache_hits={cache_hits} cache_hit_rate={hit_rate:.1%}")
    print(f"min={arr.min():.2f} ms  max={arr.max():.2f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Async latency benchmark for /query")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--requests", type=int, default=40)
    args = parser.parse_args()
    asyncio.run(run_benchmark(args.base_url, args.concurrency, args.requests))


if __name__ == "__main__":
    main()
