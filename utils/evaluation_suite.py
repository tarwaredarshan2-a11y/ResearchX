from __future__ import annotations

from dataclasses import dataclass

from utils.hybrid_retriever import weighted_rrf
from utils.retriever import BM25Retriever, DenseRetriever


@dataclass
class QueryLabel:
    query: str
    relevant_chunk_ids: set[str]


def precision_at_k(results: list, relevant: set[str], k: int = 5) -> float:
    if not results:
        return 0.0
    retrieved = [result.chunk_id for result in results[:k]]
    return len(set(retrieved) & relevant) / k


def recall_at_k(results: list, relevant: set[str], k: int = 5) -> float:
    if not relevant:
        return 0.0
    retrieved = [result.chunk_id for result in results[:k]]
    return len(set(retrieved) & relevant) / len(relevant)


def mrr(results: list, relevant: set[str]) -> float:
    for index, result in enumerate(results, start=1):
        if result.chunk_id in relevant:
            return 1.0 / index
    return 0.0


def evaluate_retrieval(dense: DenseRetriever, bm25: BM25Retriever, labels: list[QueryLabel]) -> dict:
    if not labels:
        return {"message": "No evaluation data available."}
    metrics = {"Dense Retrieval": [], "BM25": [], "Dense + BM25 + RRF": []}
    for label in labels:
        dense_results = dense.search(label.query, top_k=5)
        sparse_results = bm25.search(label.query, top_k=5)
        hybrid_results = weighted_rrf(dense_results, sparse_results, top_k=5)
        for name, results in [
            ("Dense Retrieval", dense_results),
            ("BM25", sparse_results),
            ("Dense + BM25 + RRF", hybrid_results),
        ]:
            metrics[name].append(
                {
                    "Precision@5": precision_at_k(results, label.relevant_chunk_ids),
                    "Recall@5": recall_at_k(results, label.relevant_chunk_ids),
                    "MRR": mrr(results, label.relevant_chunk_ids),
                }
            )
    return {name: average_metric(rows) for name, rows in metrics.items()}


def average_metric(rows: list[dict]) -> dict:
    keys = rows[0].keys()
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}


def rrf_ablation(dense_results: list, sparse_results: list) -> dict:
    experiments = [(1.0, 0.0), (0.75, 0.25), (0.65, 0.35), (0.5, 0.5), (0.25, 0.75), (0.0, 1.0)]
    return {
        f"{dense_weight:.2f}/{sparse_weight:.2f}": [
            result.chunk_id for result in weighted_rrf(dense_results, sparse_results, dense_weight, sparse_weight)
        ]
        for dense_weight, sparse_weight in experiments
    }


def chunk_ablation_settings() -> list[dict]:
    return [
        {"chunk_size": 600, "overlap": 100},
        {"chunk_size": 900, "overlap": 150},
        {"chunk_size": 1200, "overlap": 200},
    ]
