from __future__ import annotations

from dataclasses import dataclass, asdict

from utils.evidence import EvidenceRecord
from utils.retriever import tokenize


@dataclass
class VerificationResult:
    claim: str
    source: str
    page: int
    evidence: str
    verdict: str
    confidence: str
    explanation: str
    review_status: str = "AI Suggested"

    def to_dict(self) -> dict:
        return asdict(self)


NEGATION_TERMS = {"not", "no", "never", "without", "fail", "fails", "failed", "lower", "worse", "decrease"}


def verify_claim(claim: str, evidence_records: list[EvidenceRecord]) -> list[VerificationResult]:
    if not evidence_records:
        return [
            VerificationResult(
                claim=claim,
                source="No source",
                page=0,
                evidence="Insufficient evidence found in the uploaded literature.",
                verdict="NEUTRAL",
                confidence="Low",
                explanation="No retrieved passage was available to support or contradict the claim.",
            )
        ]

    results = []
    claim_tokens = set(tokenize(claim))
    claim_negated = bool(claim_tokens & NEGATION_TERMS)
    for record in evidence_records:
        evidence_tokens = set(tokenize(record.passage))
        overlap = claim_tokens & evidence_tokens
        evidence_negated = bool(evidence_tokens & NEGATION_TERMS)
        if len(overlap) >= max(3, min(6, len(claim_tokens) // 3)):
            verdict = "CONTRADICTED" if claim_negated != evidence_negated and (claim_negated or evidence_negated) else "ENTAILED"
            confidence = record.confidence
            explanation = "The passage shares key terms with the claim and provides direct textual support."
            if verdict == "CONTRADICTED":
                explanation = "The passage contains overlapping claim terms but differs in polarity or reported direction."
        else:
            verdict = "NEUTRAL"
            confidence = "Low"
            explanation = "The passage is related but does not provide enough direct support for the claim."
        record.verification_status = verdict
        results.append(
            VerificationResult(
                claim=claim,
                source=record.paper,
                page=record.page,
                evidence=record.passage,
                verdict=verdict,
                confidence=confidence,
                explanation=explanation,
                review_status=record.review_status,
            )
        )
    return results
