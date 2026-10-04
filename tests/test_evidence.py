from utils.evidence import make_evidence_records
from utils.retriever import SearchResult


def test_make_evidence_records_preserves_citation_data():
    result = SearchResult(
        "c1",
        "Method X achieved higher F1 on the benchmark dataset.",
        {"paper": "Paper A", "page": 7, "section": "Results"},
        0.02,
        1,
        "hybrid_rrf",
    )

    records = make_evidence_records("Which method performed best?", [result])

    assert records[0].paper == "Paper A"
    assert records[0].page == 7
    assert records[0].citation == "Paper A, p.7"
    assert records[0].confidence == "High"
