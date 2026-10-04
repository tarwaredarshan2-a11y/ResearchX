from __future__ import annotations

import re
from dataclasses import dataclass, asdict

from utils.retriever import SearchResult


@dataclass
class EvidenceRecord:
    claim: str
    paper: str
    page: int
    section: str
    passage: str
    evidence_type: str = "Retrieved passage"
    verification_status: str = "NEUTRAL"
    confidence: str = "Medium"
    review_status: str = "AI Suggested"

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def citation(self) -> str:
        return f"{self.paper}, p.{self.page}"


def make_evidence_records(claim: str, results: list[SearchResult]) -> list[EvidenceRecord]:
    if not results:
        return []
    records = []
    for result in results:
        metadata = result.metadata or {}
        records.append(
            EvidenceRecord(
                claim=claim,
                paper=str(metadata.get("paper", "Unknown paper")),
                page=int(metadata.get("page", 0) or 0),
                section=str(metadata.get("section", "Unknown")),
                passage=trim_passage(result.text),
                confidence=score_to_confidence(result.score),
            )
        )
    return records


def trim_passage(text: str, limit: int = 700) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rsplit(" ", 1)[0] + "..."


def score_to_confidence(score: float) -> str:
    if score >= 0.012:
        return "High"
    if score >= 0.006:
        return "Medium"
    return "Low"


def evidence_table(records: list[EvidenceRecord]) -> list[dict]:
    return [record.to_dict() for record in records]
