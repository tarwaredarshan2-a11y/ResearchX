from __future__ import annotations

from config import DENSE_WEIGHT, RRF_K, SPARSE_WEIGHT, TOP_K
from utils.retriever import BM25Retriever, DenseRetriever, SearchResult


def weighted_rrf(
    dense_results: list[SearchResult],
    sparse_results: list[SearchResult],
    dense_weight: float = DENSE_WEIGHT,
    sparse_weight: float = SPARSE_WEIGHT,
    rrf_k: int = RRF_K,
    top_k: int = TOP_K,
) -> list[SearchResult]:
    by_id: dict[str, SearchResult] = {}
    scores: dict[str, float] = {}

    for result in dense_results:
        by_id[result.chunk_id] = result
        scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + dense_weight / (rrf_k + result.rank)
    for result in sparse_results:
        by_id.setdefault(result.chunk_id, result)
        scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + sparse_weight / (rrf_k + result.rank)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    fused = []
    for rank, (chunk_id, score) in enumerate(ranked[:top_k], start=1):
        original = by_id[chunk_id]
        fused.append(
            SearchResult(
                chunk_id=chunk_id,
                text=original.text,
                metadata=original.metadata,
                score=score,
                rank=rank,
                source="hybrid_rrf",
            )
        )
    return fused


class HybridRetriever:
    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever):
        self.dense = dense
        self.bm25 = bm25

    def search(self, query: str, top_k: int = TOP_K, paper_ids: list[str] | None = None) -> dict[str, list[SearchResult]]:
        dense_results = self.dense.search(query, top_k=top_k, paper_ids=paper_ids)
        sparse_results = self.bm25.search(query, top_k=top_k, paper_ids=paper_ids)
        hybrid_results = weighted_rrf(dense_results, sparse_results, top_k=top_k)
        return {"dense": dense_results, "bm25": sparse_results, "hybrid": hybrid_results}
