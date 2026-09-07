from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np

from app.config import settings

EmbedFn = Callable[[str | list[str]], np.ndarray]


class SemanticCache:
    """In-memory vector cache of query embeddings, answers, and timestamps."""

    def __init__(
        self,
        embed_fn: EmbedFn | None = None,
        threshold: float | None = None,
    ) -> None:
        self.embed_fn = embed_fn
        self.threshold = (
            settings.CACHE_SIMILARITY_THRESHOLD if threshold is None else threshold
        )
        self._embeddings: np.ndarray | None = None
        self._entries: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self._entries)

    def lookup(self, query: str) -> dict[str, Any] | None:
        if self._embeddings is None or not self._entries:
            return None
        vector = _unit_vector(self._embed(query))
        similarities = self._embeddings @ vector
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        if best_score < self.threshold:
            return None
        hit = dict(self._entries[best_idx]["response"])
        hit["cache_hit"] = True
        hit["cache_similarity"] = best_score
        return hit

    def store(self, query: str, response: dict[str, Any]) -> None:
        vector = _unit_vector(self._embed(query)).reshape(1, -1)
        if self._embeddings is None:
            self._embeddings = vector
        else:
            self._embeddings = np.vstack([self._embeddings, vector])
        self._entries.append(
            {
                "query": query,
                "response": dict(response),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def clear(self) -> None:
        self._embeddings = None
        self._entries = []

    def _embed(self, query: str) -> np.ndarray:
        if self.embed_fn is None:
            raise RuntimeError("SemanticCache requires an embed_fn")
        return np.asarray(self.embed_fn(query), dtype=np.float32)


def _unit_vector(embedding: np.ndarray) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm
