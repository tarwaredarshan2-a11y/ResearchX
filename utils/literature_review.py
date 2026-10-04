from __future__ import annotations

from collections import Counter

from utils.comparison import compare_papers, extract_structured_facts
from utils.evidence import EvidenceRecord
from utils.retriever import tokenize


def build_literature_matrix(evidence_records: list[EvidenceRecord]) -> tuple[list[dict], dict]:
    comparison = compare_papers(evidence_records)
    matrix = []
    for row in comparison:
        matrix.append(
            {
                "Paper": row["Paper"],
                "Problem": infer_problem(evidence_records, row["Paper"]),
                "Method": row["Method"],
                "Dataset": row["Dataset"],
                "Model": infer_model(evidence_records, row["Paper"]),
                "Metrics": row["Metric"],
                "Key Findings": row["Finding"],
                "Limitations": row["Limitation"],
            }
        )

    summary = {
        "Common Approaches": common_terms(evidence_records, ["method", "approach", "model", "algorithm"]),
        "Common Findings": common_terms(evidence_records, ["achieved", "improved", "result", "performance"]),
        "Major Differences": "Compare datasets, metrics, and limitations in the matrix. Do not treat differences as contradictions without direct evidence.",
        "Common Limitations": common_terms(evidence_records, ["limited", "limitation", "future", "challenge"]),
    }
    return matrix, summary


def infer_problem(records: list[EvidenceRecord], paper: str) -> str:
    for record in records:
        if record.paper == paper and any(word in record.passage.lower() for word in ["problem", "challenge", "task"]):
            return record.passage[:260]
    return "Not reported"


def infer_model(records: list[EvidenceRecord], paper: str) -> str:
    for record in records:
        if record.paper == paper:
            facts = extract_structured_facts(record.passage)
            if facts["model"] != "Not reported":
                return facts["model"]
    return "Not reported"


def common_terms(records: list[EvidenceRecord], triggers: list[str]) -> str:
    selected = []
    for record in records:
        if any(trigger in record.passage.lower() for trigger in triggers):
            selected.extend(tokenize(record.passage))
    counts = Counter(token for token in selected if len(token) > 4)
    if not counts:
        return "Not enough evidence available."
    return ", ".join(term for term, _ in counts.most_common(6))
