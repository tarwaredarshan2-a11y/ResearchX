from __future__ import annotations

from collections import defaultdict

from utils.evidence import EvidenceRecord


GAP_KEYWORDS = ["limited", "limitation", "future work", "underexplored", "lack", "small dataset", "challenge", "not evaluated"]


def detect_candidate_gaps(evidence_records: list[EvidenceRecord]) -> list[dict]:
    candidates: dict[str, dict] = {}
    for record in evidence_records:
        passage_lower = record.passage.lower()
        if not any(keyword in passage_lower for keyword in GAP_KEYWORDS):
            continue
        title = classify_gap_title(passage_lower)
        gap = candidates.setdefault(
            title,
            {
                "title": title,
                "description": "A potential limitation pattern appears in the uploaded literature.",
                "why_it_appears": "Retrieved passages mention limitations, future work, missing evaluation, or constrained experimental settings.",
                "supporting_papers": set(),
                "supporting_pages": defaultdict(set),
                "evidence": [],
                "confidence": "Medium",
                "review_status": "AI Suggested",
            },
        )
        gap["supporting_papers"].add(record.paper)
        gap["supporting_pages"][record.paper].add(record.page)
        gap["evidence"].append({"paper": record.paper, "page": record.page, "passage": record.passage})

    formatted = []
    for gap in candidates.values():
        paper_count = len(gap["supporting_papers"])
        gap["confidence"] = "High" if paper_count >= 3 else "Medium" if paper_count >= 2 else "Low"
        gap["supporting_papers"] = sorted(gap["supporting_papers"])
        gap["supporting_pages"] = {
            paper: sorted(pages) for paper, pages in sorted(gap["supporting_pages"].items(), key=lambda item: item[0])
        }
        formatted.append(gap)
    return formatted


def classify_gap_title(text: str) -> str:
    if "dataset" in text or "data set" in text:
        return "Limited evaluation across datasets"
    if "general" in text or "real-world" in text:
        return "Limited real-world generalization evidence"
    if "metric" in text or "accuracy" in text or "f1" in text:
        return "Narrow evaluation metrics"
    return "Underexplored limitation reported in literature"
