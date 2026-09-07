from app.cache import SemanticCache
from app.retrieval import reciprocal_rank_fusion, tokenize


def test_rrf_prefers_docs_high_in_both_lists() -> None:
    fused = reciprocal_rank_fusion([["b", "a", "c"], ["b", "d"]], k=60)
    assert fused[0][0] == "b"
    expected = 1.0 / (60 + 0 + 1) + 1.0 / (60 + 0 + 1)
    assert abs(fused[0][1] - expected) < 1e-9


def test_tokenize_lowercases_and_strips_punctuation() -> None:
    assert tokenize("Hello, RAG-2!") == ["hello", "rag", "2"]


def test_semantic_cache_hit_and_miss() -> None:
    vectors = {
        "q3 revenue": [1.0, 0.0],
        "q3 financial results": [0.99, 0.01],
        "sla uptime": [0.0, 1.0],
    }

    def embed_fn(text: str):
        return vectors[text]

    cache = SemanticCache(embed_fn=embed_fn, threshold=0.92)
    cache.store("q3 revenue", {"answer": "48.2 million", "confidence_score": 0.9, "sources_used": ["q3-financials"]})
    hit = cache.lookup("q3 financial results")
    miss = cache.lookup("sla uptime")
    assert hit is not None
    assert hit["answer"] == "48.2 million"
    assert hit["cache_hit"] is True
    assert miss is None
