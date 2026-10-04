from utils.comparison import compare_papers
from utils.evidence import EvidenceRecord
from utils.research_gap import detect_candidate_gaps


def test_candidate_gap_detected_from_limitations():
    records = [
        EvidenceRecord(
            claim="What limitations appear?",
            paper="Paper A",
            page=6,
            section="Discussion",
            passage="A limitation of this study is the small dataset and limited real-world evaluation.",
        ),
        EvidenceRecord(
            claim="What limitations appear?",
            paper="Paper B",
            page=9,
            section="Conclusion",
            passage="Future work should evaluate larger dataset settings.",
        ),
    ]

    gaps = detect_candidate_gaps(records)

    assert gaps
    assert gaps[0]["confidence"] == "Medium"
    assert "Paper A" in gaps[0]["supporting_papers"]


def test_comparison_uses_not_reported_for_missing_fields():
    records = [
        EvidenceRecord(
            claim="Compare methods",
            paper="Paper A",
            page=2,
            section="Methods",
            passage="The method uses a transformer model.",
        )
    ]

    rows = compare_papers(records)

    assert rows[0]["Paper"] == "Paper A"
    assert rows[0]["Dataset"] == "Not reported"
