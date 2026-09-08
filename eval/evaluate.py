"""LLM-as-a-Judge quantitative quality evaluation (no RAGAS).

Why this exists:
    Exact substring matching ("48.2 million" in answer) is brittle: paraphrases
    cause false negatives and keyword-stuffed hallucinations cause false
    positives. This harness uses a judge LLM for semantic verdicts instead.

What this does:
    1. Loads `eval/dataset.json` (question, ground_truth, source_id).
    2. Runs each question through `RAGEngine.query()`.
       Failures raise visibly (no silent extractive fallback in the engine).
    3. Context Precision (deterministic): expected `source_id` in
       `res.sources_used`.
    4. Faithfulness / Answer Relevancy / Context Recall (judge LLM):
       same model as generation (`settings.LLM_MODEL`) with a structured
       JSON prompt asking whether the answer is strictly grounded in the
       retrieved context and relevant to the query, and whether the
       retrieved context contains the facts needed for the ground truth.
    5. Saves pass rates + per-case verdicts to `eval/results.json`.

Usage:
    $env:PYTHONPATH = "."
    python -m eval.evaluate
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from app.config import settings
from app.engine import RAGEngine, RAGResponse
from app.main import SAMPLE_DOCUMENTS

DATASET_PATH = Path(__file__).parent / "dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.json"

# Generator lives on llm7 (settings.LLM_MODEL); the judge is a separate
# provider/model (settings.JUDGE_MODEL, Groq qwen3.8-27b) to avoid
# self-preference bias. Resolved at call time so .env changes apply.
def _judge_model() -> str:
    return settings.JUDGE_MODEL


def _judge_base_url() -> str:
    return settings.judge_base_url

JUDGE_SYSTEM_PROMPT = (
    "You are a strict RAG evaluation judge. Judge ONLY what is given. "
    "Return a single JSON object and nothing else."
)


def _build_judge_client() -> Any:
    if not settings.judge_api_key:
        raise RuntimeError(
            "No judge API key is configured (JUDGE_API_KEY / GROQ_API_KEY): "
            "the LLM-as-a-Judge harness requires a key."
        )
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The 'openai' package is required for the judge client.") from exc
    return OpenAI(api_key=settings.judge_api_key, base_url=_judge_base_url())


def _judge_prompt(
    question: str, answer: str, contexts: list[str], ground_truth: str
) -> str:
    context_block = "\n\n".join(f"[context {i + 1}]\n{text}" for i, text in enumerate(contexts))
    return (
        "Evaluate one RAG answer against its retrieved contexts.\n\n"
        f"Question:\n{question}\n\n"
        f"Answer under review:\n{answer}\n\n"
        f"Retrieved contexts:\n{context_block}\n\n"
        f"Reference ground truth:\n{ground_truth}\n\n"
        "Decide three booleans:\n"
        '1. "faithful": true ONLY if every factual claim in the answer is '
        "directly supported by the retrieved contexts (no hallucinations, "
        "no outside knowledge, no contradictions). Paraphrase is fine; "
        "unsupported details are not.\n"
        '2. "relevant": true ONLY if the answer actually addresses the question '
        "(not a refusal, not off-topic).\n"
        '3. "context_recall": true ONLY if the retrieved contexts contain the '
        "facts needed to produce the ground truth.\n\n"
        "Return ONLY this JSON object:\n"
        '{"faithful": true/false, "relevant": true/false, '
        '"context_recall": true/false, "reason": "<one short sentence>"}'
    )


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "y", "1", "pass"}:
            return True
        if text in {"false", "no", "n", "0", "fail"}:
            return False
    raise ValueError(f"Cannot coerce to bool: {value!r}")


def _parse_verdict(content: str) -> dict[str, Any]:
    text = (content or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Judge did not return JSON: {content!r}")
    payload = json.loads(text[start : end + 1])
    return {
        "faithful": _coerce_bool(payload["faithful"]),
        "relevant": _coerce_bool(payload["relevant"]),
        "context_recall": _coerce_bool(payload["context_recall"]),
        "reason": str(payload.get("reason") or ""),
    }


def _judge_case(
    client: Any, question: str, answer: str, contexts: list[str], ground_truth: str
) -> tuple[dict[str, Any], float]:
    start = time.perf_counter()
    completion = client.chat.completions.create(
        model=_judge_model(),
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _judge_prompt(question, answer, contexts, ground_truth),
            },
        ],
        temperature=0,
        max_tokens=300,
    )
    judge_latency_ms = (time.perf_counter() - start) * 1000
    content = completion.choices[0].message.content or ""
    # Let parse errors raise visibly: a malformed judge verdict is a failure,
    # not something to silently count as a pass.
    return _parse_verdict(content), judge_latency_ms


def run_evaluation() -> dict[str, Any]:
    judge_client = _build_judge_client()
    engine = RAGEngine()
    engine.index_documents(SAMPLE_DOCUMENTS)

    cases: list[dict[str, Any]] = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    total = len(cases)
    precision_hits = 0
    faithfulness_hits = 0
    relevancy_hits = 0
    recall_hits = 0

    print("Running LLM-as-a-Judge Quality Evaluation Suite...")
    print(f"Judge model: {_judge_model()} ({_judge_base_url()})")
    print("=" * 60)

    results: list[dict[str, Any]] = []
    for i, case in enumerate(cases, 1):
        question = str(case["question"])
        ground_truth = str(case.get("ground_truth") or "")
        expected_src = str(case.get("source_id") or "")

        query_start = time.perf_counter()
        res: RAGResponse = engine.query(question)
        query_latency_ms = (time.perf_counter() - query_start) * 1000

        retrieved = engine.retriever.search(question, top_k=3, top_fused=10)
        contexts = [str(item.get("text") or "") for item in retrieved if item.get("text")]

        # b. Context Precision: deterministic source match.
        precision_hit = bool(expected_src) and expected_src in (res.sources_used or [])
        if precision_hit:
            precision_hits += 1

        # c. Faithfulness / Relevancy / Context Recall: judge LLM verdict.
        # Judge/client failures raise visibly per-case and are recorded.
        try:
            verdict, judge_latency_ms = _judge_case(
                judge_client, question, res.answer, contexts, ground_truth
            )
            judge_error: str | None = None
        except Exception as exc:
            verdict = {"faithful": False, "relevant": False, "context_recall": False, "reason": ""}
            judge_latency_ms = 0.0
            judge_error = f"{type(exc).__name__}: {exc}"

        if verdict["faithful"]:
            faithfulness_hits += 1
        if verdict["relevant"]:
            relevancy_hits += 1
        if verdict["context_recall"]:
            recall_hits += 1

        passed = precision_hit and verdict["faithful"] and verdict["relevant"]
        status = "PASS" if passed else "FAIL"
        print(f"[{i}/{total}] {status} (query {query_latency_ms:.1f}ms, judge {judge_latency_ms:.1f}ms)")
        print(f"   Query:    {question}")
        print(f"   Answer:   {res.answer[:120]}...")
        print(f"   Sources:  {res.sources_used} (Expected: {expected_src})")
        print(
            f"   Judge:    faithful={verdict['faithful']} "
            f"relevant={verdict['relevant']} recall={verdict['context_recall']} "
            f"- {verdict['reason']}"
        )
        if judge_error:
            print(f"   Error:    {judge_error}")
        print()

        results.append(
            {
                "question": question,
                "expected_source": expected_src,
                "sources_used": list(res.sources_used or []),
                "context_precision_hit": precision_hit,
                "faithful": bool(verdict["faithful"]),
                "relevant": bool(verdict["relevant"]),
                "context_recall_hit": bool(verdict["context_recall"]),
                "judge_reason": verdict["reason"],
                "judge_error": judge_error,
                "query_latency_ms": round(query_latency_ms, 1),
                "judge_latency_ms": round(judge_latency_ms, 1),
            }
        )

    context_precision = precision_hits / total if total else 0.0
    faithfulness = faithfulness_hits / total if total else 0.0
    answer_relevancy = relevancy_hits / total if total else 0.0
    context_recall = recall_hits / total if total else 0.0

    print("=" * 60)
    print("LLM-AS-A-JUDGE QUALITY RESULTS:")
    print("=" * 60)
    print(f"Total Test Cases:       {total}")
    print(f"Context Precision:      {context_precision * 100:.1f}% ({precision_hits}/{total})")
    print(f"Faithfulness:           {faithfulness * 100:.1f}% ({faithfulness_hits}/{total})")
    print(f"Answer Relevancy:       {answer_relevancy * 100:.1f}% ({relevancy_hits}/{total})")
    print(f"Context Recall:         {context_recall * 100:.1f}% ({recall_hits}/{total})")
    print(f"Hallucination Rate:     {(1.0 - faithfulness) * 100:.1f}%")
    print("=" * 60)

    summary = {
        "total": total,
        "judge_model": _judge_model(),
        "judge_base_url": _judge_base_url(),
        "gen_model": settings.LLM_MODEL,
        "gen_base_url": settings.gen_base_url,
        "context_precision": round(context_precision, 4),
        "faithfulness": round(faithfulness, 4),
        "answer_relevancy": round(answer_relevancy, 4),
        "context_recall": round(context_recall, 4),
        "hallucination_rate": round(1.0 - faithfulness, 4) if total else 0.0,
        "cases": results,
    }
    RESULTS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved JSON summary to {RESULTS_PATH}")
    return summary


if __name__ == "__main__":
    summary = run_evaluation()
    ok = (
        summary["context_precision"] == 1.0
        and summary["faithfulness"] == 1.0
        and summary["answer_relevancy"] == 1.0
    )
    raise SystemExit(0 if ok else 1)
