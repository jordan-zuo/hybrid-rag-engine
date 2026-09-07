from __future__ import annotations

import json
import os
from pathlib import Path

from datasets import Dataset

from app.config import settings
from app.engine import RAGEngine
from app.main import SAMPLE_DOCUMENTS

ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "dataset.json"


def _load_cases() -> list[dict]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def _collect_predictions(engine: RAGEngine, cases: list[dict]) -> Dataset:
    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truths: list[str] = []
    for case in cases:
        question = case["question"]
        result = engine.query(question)
        retrieved = engine.retriever.search(question, top_k=3, top_fused=10)
        questions.append(question)
        answers.append(result.answer)
        contexts.append([item["text"] for item in retrieved])
        ground_truths.append(case["ground_truth"])
        print(f"- {question}\n  answer: {result.answer}\n  sources: {result.sources_used}\n")
    return Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )


def _configure_groq_for_ragas() -> None:
    if settings.GROQ_API_KEY:
        os.environ.setdefault("OPENAI_API_KEY", settings.GROQ_API_KEY)
        os.environ.setdefault("OPENAI_BASE_URL", settings.GROQ_BASE_URL)
        os.environ.setdefault("OPENAI_API_BASE", settings.GROQ_BASE_URL)


def _evaluate(dataset: Dataset) -> None:
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    _configure_groq_for_ragas()
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )
    scores = dict(result) if not hasattr(result, "to_pandas") else result
    print("=== Ragas evaluation ===")
    if hasattr(result, "to_pandas"):
        frame = result.to_pandas()
        print(frame)
        for metric in ("faithfulness", "answer_relevancy", "context_precision"):
            if metric in frame.columns:
                print(f"{metric}: {float(frame[metric].mean()):.4f}")
        return
    for metric in ("faithfulness", "answer_relevancy", "context_precision"):
        if metric in scores:
            print(f"{metric}: {scores[metric]}")


def main() -> None:
    engine = RAGEngine()
    engine.index_documents(SAMPLE_DOCUMENTS)
    dataset = _collect_predictions(engine, _load_cases())
    try:
        _evaluate(dataset)
    except Exception as exc:
        print("Ragas evaluate() could not complete:", exc)
        print("Predictions above can still be scored manually against ground_truth.")


if __name__ == "__main__":
    main()
