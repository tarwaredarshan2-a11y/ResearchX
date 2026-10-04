from __future__ import annotations

import re

from utils.evidence import EvidenceRecord


FIELDS = ["method", "model", "dataset", "metric", "result", "limitation"]


def extract_structured_facts(passage: str) -> dict:
    lower = passage.lower()
    facts = {field: "Not reported" for field in FIELDS}
    if any(word in lower for word in ["method", "approach", "framework", "algorithm"]):
        facts["method"] = sentence_with(passage, ["method", "approach", "framework", "algorithm"])
    if any(word in lower for word in ["model", "cnn", "rnn", "lstm", "bert", "transformer", "svm"]):
        facts["model"] = sentence_with(passage, ["model", "cnn", "rnn", "lstm", "bert", "transformer", "svm"])
    if any(word in lower for word in ["dataset", "data set", "benchmark", "corpus"]):
        facts["dataset"] = sentence_with(passage, ["dataset", "data set", "benchmark", "corpus"])
    if any(word in lower for word in ["accuracy", "precision", "recall", "f1", "auc", "metric"]):
        facts["metric"] = sentence_with(passage, ["accuracy", "precision", "recall", "f1", "auc", "metric"])
    if any(word in lower for word in ["achieved", "outperform", "improved", "result", "%"]):
        facts["result"] = sentence_with(passage, ["achieved", "outperform", "improved", "result", "%"])
    if any(word in lower for word in ["limitation", "limited", "future work", "challenge", "however"]):
        facts["limitation"] = sentence_with(passage, ["limitation", "limited", "future work", "challenge", "however"])
    return facts


def sentence_with(text: str, needles: list[str]) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    for sentence in sentences:
        lower = sentence.lower()
        if any(needle in lower for needle in needles):
            return sentence[:260]
    return "Not reported"


def compare_papers(evidence_records: list[EvidenceRecord]) -> list[dict]:
    rows_by_paper: dict[str, dict] = {}
    for record in evidence_records:
        row = rows_by_paper.setdefault(
            record.paper,
            {
                "Paper": record.paper,
                "Method": "Not reported",
                "Dataset": "Not reported",
                "Metric": "Not reported",
                "Finding": "Not reported",
                "Limitation": "Not reported",
            },
        )
        facts = extract_structured_facts(record.passage)
        if row["Method"] == "Not reported" and facts["method"] != "Not reported":
            row["Method"] = facts["method"]
        if row["Dataset"] == "Not reported" and facts["dataset"] != "Not reported":
            row["Dataset"] = facts["dataset"]
        if row["Metric"] == "Not reported" and facts["metric"] != "Not reported":
            row["Metric"] = facts["metric"]
        if row["Finding"] == "Not reported" and facts["result"] != "Not reported":
            row["Finding"] = facts["result"]
        if row["Limitation"] == "Not reported" and facts["limitation"] != "Not reported":
            row["Limitation"] = facts["limitation"]
    return list(rows_by_paper.values())


def analyze_disagreements(evidence_records: list[EvidenceRecord]) -> list[dict]:
    rows = []
    passages_by_paper = {record.paper: record.passage for record in evidence_records}
    papers = list(passages_by_paper)
    for left_index, left in enumerate(papers):
        for right in papers[left_index + 1 :]:
            left_text = passages_by_paper[left].lower()
            right_text = passages_by_paper[right].lower()
            category = "Agreement"
            reason = "The retrieved evidence appears to discuss similar claims or methods."
            if ("dataset" in left_text) != ("dataset" in right_text):
                category = "Dataset Difference"
                reason = "One passage reports dataset context while the other does not."
            elif ("accuracy" in left_text or "f1" in left_text) != ("accuracy" in right_text or "f1" in right_text):
                category = "Metric Difference"
                reason = "The retrieved passages emphasize different reported metrics."
            elif ("not " in left_text) != ("not " in right_text):
                category = "Potential Contradiction"
                reason = "Reason for difference could not be established from the analyzed evidence."
            rows.append({"Paper A": left, "Paper B": right, "Category": category, "Reason": reason, "Review Status": "AI Suggested"})
    return rows
