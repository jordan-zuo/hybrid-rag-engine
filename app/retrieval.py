from __future__ import annotations

import re
import uuid
from collections import defaultdict
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.config import settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def point_id_for(doc_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, doc_id))


class HybridRetriever:
    """Dense Qdrant + BM25, fused with RRF, then Cross-Encoder reranked."""

    def __init__(self) -> None:
        self.client = QdrantClient(path=settings.QDRANT_STORAGE_PATH)
        self.dense_model = SentenceTransformer(settings.DENSE_MODEL_NAME)
        self.reranker = CrossEncoder(settings.RERANKER_MODEL_NAME)
        self._documents: dict[str, dict[str, Any]] = {}
        self._bm25: BM25Okapi | None = None
        self._bm25_ids: list[str] = []
        self._ensure_collection()
        self._rebuild_from_store()

    def embed(self, texts: str | list[str]) -> np.ndarray:
        return np.asarray(
            self.dense_model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

    def index_documents(self, docs: list[dict[str, Any]]) -> int:
        if not docs:
            return 0

        prepared: list[dict[str, Any]] = []
        for index, doc in enumerate(docs):
            text = (doc.get("text") or "").strip()
            if not text:
                continue
            doc_id = str(doc.get("id") or f"doc-{len(self._documents) + index}")
            prepared.append(
                {
                    "id": doc_id,
                    "text": text,
                    "metadata": dict(doc.get("metadata") or {}),
                }
            )
        if not prepared:
            return 0

        vectors = self.embed([item["text"] for item in prepared])
        points = [
            PointStruct(
                id=point_id_for(item["id"]),
                vector=vectors[i].tolist(),
                payload=item,
            )
            for i, item in enumerate(prepared)
        ]
        self.client.upsert(
            collection_name=settings.COLLECTION_NAME,
            points=points,
            wait=True,
        )
        for item in prepared:
            self._documents[item["id"]] = item
        self._fit_bm25()
        return len(prepared)

    def search(
        self,
        query: str,
        top_k: int = 3,
        top_fused: int = 10,
        query_embedding: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        if not self._documents:
            return []

        retrieve_k = max(settings.DENSE_TOP_K, settings.SPARSE_TOP_K, top_fused)
        if query_embedding is None:
            query_embedding = self.embed(query)

        dense_ids = self._dense_search(query_embedding, retrieve_k)
        sparse_ids = self._bm25_search(query, retrieve_k)
        fused = reciprocal_rank_fusion([dense_ids, sparse_ids], k=settings.RRF_K)
        candidates: list[dict[str, Any]] = []
        for doc_id, rrf_score in fused[:top_fused]:
            doc = self._documents.get(doc_id)
            if doc is None:
                continue
            candidates.append({**doc, "rrf_score": rrf_score})
        return self._rerank(query, candidates, top_k)

    def document_count(self) -> int:
        return len(self._documents)

    def _ensure_collection(self) -> None:
        existing = {collection.name for collection in self.client.get_collections().collections}
        if settings.COLLECTION_NAME in existing:
            return
        self.client.create_collection(
            collection_name=settings.COLLECTION_NAME,
            vectors_config=VectorParams(
                size=settings.VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )

    def _rebuild_from_store(self) -> None:
        self._documents = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=settings.COLLECTION_NAME,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                doc_id = str(payload.get("id") or point.id)
                text = str(payload.get("text") or "")
                if not text:
                    continue
                self._documents[doc_id] = {
                    "id": doc_id,
                    "text": text,
                    "metadata": dict(payload.get("metadata") or {}),
                }
            if offset is None:
                break
        self._fit_bm25()

    def _fit_bm25(self) -> None:
        self._bm25_ids = list(self._documents.keys())
        corpus = [tokenize(self._documents[doc_id]["text"]) for doc_id in self._bm25_ids]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def _dense_search(self, query_embedding: np.ndarray, limit: int) -> list[str]:
        if hasattr(self.client, "query_points"):
            results = self.client.query_points(
                collection_name=settings.COLLECTION_NAME,
                query=_as_list(query_embedding),
                limit=limit,
                with_payload=True,
            )
            hits = results.points
        else:
            hits = self.client.search(
                collection_name=settings.COLLECTION_NAME,
                query_vector=_as_list(query_embedding),
                limit=limit,
                with_payload=True,
            )

        ranked: list[str] = []
        for hit in hits:
            payload = hit.payload or {}
            doc_id = str(payload.get("id") or hit.id)
            if doc_id in self._documents:
                ranked.append(doc_id)
        return ranked

    def _bm25_search(self, query: str, limit: int) -> list[str]:
        if self._bm25 is None:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked: list[str] = []
        for idx in np.argsort(scores)[::-1]:
            if scores[idx] <= 0:
                break
            ranked.append(self._bm25_ids[int(idx)])
            if len(ranked) >= limit:
                break
        return ranked

    def _rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        pairs = [(query, item["text"]) for item in candidates]
        raw_scores = self.reranker.predict(pairs, show_progress_bar=False)
        scores = np.atleast_1d(np.asarray(raw_scores, dtype=np.float32))
        scored = [{**item, "score": float(score)} for item, score in zip(candidates, scores)]
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """score(d) = sum(1 / (k + rank + 1)) with 0-based rank per list."""
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def _as_list(embedding: np.ndarray) -> list[float]:
    return np.asarray(embedding, dtype=np.float32).reshape(-1).tolist()
