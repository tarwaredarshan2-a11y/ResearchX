from utils.claim_verifier import verify_claim
from utils.evidence import EvidenceRecord


def test_claim_verifier_marks_supported_claim_entailed():
    evidence = [
        EvidenceRecord(
            claim="Transformer model achieved higher F1 performance",
            paper="Paper A",
            page=5,
            section="Results",
            passage="The transformer model achieved higher F1 performance than the baseline.",
        )
    ]

    result = verify_claim("Transformer model achieved higher F1 performance", evidence)[0]

    assert result.verdict == "ENTAILED"


def test_claim_verifier_returns_neutral_without_evidence():
    result = verify_claim("Unsupported claim", [])[0]

    assert result.verdict == "NEUTRAL"
    assert "Insufficient evidence" in result.evidence
