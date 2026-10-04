from utils.hybrid_retriever import weighted_rrf
from utils.pdf_loader import Chunk
from utils.retriever import BM25Retriever, SearchResult


def test_bm25_retrieves_exact_terms():
    chunks = [
        Chunk("c1", "Paper A", "p1", 1, "Methods", "We use a transformer model on CICIDS dataset."),
        Chunk("c2", "Paper B", "p2", 2, "Methods", "We discuss unrelated survey design."),
    ]
    retriever = BM25Retriever(chunks)

    results = retriever.search("CICIDS transformer")

    assert results[0].chunk_id == "c1"
    assert results[0].metadata["page"] == 1


def test_weighted_rrf_combines_dense_and_sparse_rankings():
    dense = [
        SearchResult("a", "alpha", {"paper": "A", "page": 1}, 0.9, 1, "dense"),
        SearchResult("b", "beta", {"paper": "B", "page": 2}, 0.8, 2, "dense"),
    ]
    sparse = [
        SearchResult("b", "beta", {"paper": "B", "page": 2}, 2.0, 1, "bm25"),
        SearchResult("c", "gamma", {"paper": "C", "page": 3}, 1.0, 2, "bm25"),
    ]

    results = weighted_rrf(dense, sparse, top_k=3)

    assert [result.chunk_id for result in results] == ["b", "a", "c"]
    assert all(result.source == "hybrid_rrf" for result in results)
