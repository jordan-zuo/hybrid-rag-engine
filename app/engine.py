from __future__ import annotations

import math
import re
import time
from typing import Any

from pydantic import BaseModel, Field

from app.cache import SemanticCache
from app.config import settings
from app.retrieval import HybridRetriever, tokenize


class GroundedAnswer(BaseModel):
    answer: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    sources_used: list[str] = Field(default_factory=list)


class RAGResponse(BaseModel):
    answer: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    sources_used: list[str] = Field(default_factory=list)
    cache_hit: bool = False
    latency_ms: float = 0.0


class RAGEngine:
    def __init__(self) -> None:
        self.retriever = HybridRetriever()
        self.cache = SemanticCache(embed_fn=self.retriever.embed)
        self._llm = _build_groq_client()

    def index_documents(self, docs: list[dict[str, Any]]) -> int:
        return self.retriever.index_documents(docs)

    def query(self, query: str, top_k: int = 3, top_fused: int = 10) -> RAGResponse:
        start_time = time.perf_counter()

        def _latency_ms() -> float:
            return round((time.perf_counter() - start_time) * 1000, 2)

        cached = self.cache.lookup(query)
        if cached is not None:
            return RAGResponse(
                answer=str(cached.get("answer") or ""),
                confidence_score=float(cached.get("confidence_score") or 0.0),
                sources_used=list(cached.get("sources_used") or []),
                cache_hit=True,
                latency_ms=_latency_ms(),
            )

        sources = self.retriever.search(query, top_k=top_k, top_fused=top_fused)
        sources = [
            item for item in sources
            if float(item.get("score") or 0.0) > 0.0
        ]
        top_score = float(sources[0].get("score") or 0.0) if sources else 0.0
        confidence_score = max(0.0, min(1.0, top_score))
        if confidence_score <= 0.0 or not sources:
            return RAGResponse(
                answer="This information is not available in the enterprise documents.",
                confidence_score=0.0,
                sources_used=[],
                cache_hit=False,
                latency_ms=_latency_ms(),
            )

        generated = self._generate(query, sources)
        response = RAGResponse(
            answer=generated.answer,
            confidence_score=generated.confidence_score,
            sources_used=generated.sources_used or [item["id"] for item in sources],
            cache_hit=False,
            latency_ms=_latency_ms(),
        )
        self.cache.store(query, response.model_dump())
        return response

    def stats(self) -> dict[str, Any]:
        return {
            "documents": self.retriever.document_count(),
            "cache_entries": len(self.cache),
            "collection": settings.COLLECTION_NAME,
            "dense_model": settings.DENSE_MODEL_NAME,
            "reranker_model": settings.RERANKER_MODEL_NAME,
            "llm_model": settings.LLM_MODEL if self._llm is not None else None,
            "synthesis": "groq" if self._llm is not None else "local",
            "docker": False,
        }

    def _generate(self, query: str, sources: list[dict[str, Any]]) -> GroundedAnswer:
        if not sources:
            return GroundedAnswer(
                answer="No relevant documents were found in the knowledge base.",
                confidence_score=0.0,
                sources_used=[],
            )
        if self._llm is None:
            return _local_synthesis(query, sources)

        context = "\n\n".join(f"[{item['id']}]\n{item['text']}" for item in sources)
        system_prompt = (
            "You are a strict enterprise search assistant. Only answer questions using "
            "the provided context. If the answer is not explicitly mentioned in the "
            "context, strictly reply: 'This information is not available in the "
            "enterprise documents.' Never use outside knowledge, speculate, or make "
            "assumptions."
        )
        prompt = (
            "Set confidence_score between 0 and 1. "
            "Put source ids in sources_used.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}"
        )
        try:
            return self._llm.chat.completions.create(
                model=settings.LLM_MODEL,
                response_model=GroundedAnswer,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception:
            return _local_synthesis(query, sources)


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _local_synthesis(query: str, sources: list[dict[str, Any]]) -> GroundedAnswer:
    query_tokens = set(tokenize(query))
    picked: list[str] = []
    seen: set[str] = set()
    for item in sources:
        sentences = [part.strip() for part in _SENTENCE_RE.split(item["text"]) if part.strip()]
        ranked = sorted(
            sentences,
            key=lambda sentence: len(query_tokens & set(tokenize(sentence))),
            reverse=True,
        )
        for sentence in ranked[:2]:
            key = sentence.lower()
            overlap = len(query_tokens & set(tokenize(sentence)))
            if key in seen or (query_tokens and overlap == 0):
                continue
            seen.add(key)
            picked.append(sentence)
            if len(picked) >= 4:
                break
        if len(picked) >= 4:
            break
    if not picked:
        picked = [item["text"].strip() for item in sources[:2] if item["text"].strip()]
    answer = " ".join(picked).strip() or (
        "Retrieved documents did not contain enough overlapping text to synthesize an answer."
    )
    top_score = float(sources[0].get("score") or 0.0)
    confidence = 1.0 / (1.0 + math.exp(-top_score))
    return GroundedAnswer(
        answer=answer,
        confidence_score=round(confidence, 4),
        sources_used=[item["id"] for item in sources],
    )


def _build_groq_client() -> Any:
    if not settings.GROQ_API_KEY:
        return None
    try:
        import instructor
        from openai import OpenAI
    except ImportError:
        return None
    return instructor.from_openai(
        OpenAI(api_key=settings.GROQ_API_KEY, base_url=settings.GROQ_BASE_URL)
    )
