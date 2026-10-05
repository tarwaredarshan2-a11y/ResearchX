from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from config import STATE_FILE, TOP_K
from utils.claim_verifier import verify_claim
from utils.comparison import analyze_disagreements, compare_papers
from utils.evidence import EvidenceRecord, make_evidence_records
from utils.hybrid_retriever import HybridRetriever
from utils.literature_review import build_literature_matrix
from utils.llm import LLMClient
from utils.pdf_loader import Paper, extract_pdf, save_uploaded_pdf
from utils.research_gap import detect_candidate_gaps
from utils.retriever import BM25Retriever, DenseRetriever


class ResearchOrchestrator:
    def __init__(self):
        self.state = load_state()
        self.dense = DenseRetriever()
        self.bm25 = BM25Retriever(self.dense.load_all_records())
        self.hybrid = HybridRetriever(self.dense, self.bm25)
        self.llm = LLMClient()

    def process_uploads(self, uploaded_files) -> list[Paper]:
        existing_hashes = {paper["paper_id"] for paper in self.state.get("papers", [])}
        processed: list[Paper] = []
        for uploaded in uploaded_files:
            saved, _ = save_uploaded_pdf(uploaded, existing_hashes)
            if saved.status == "Duplicate":
                processed.append(saved)
                continue
            if saved.status == "Failed":
                processed.append(saved)
                continue
            paper, chunks = extract_pdf(saved.path, saved.paper_id, saved.name)
            processed.append(paper)
            if paper.status == "Processed":
                self.dense.add_chunks(chunks)
                for existing in self.state["papers"]:
                    if existing["paper_id"] == paper.paper_id:
                        break
                else:
                    self.state["papers"].append(paper.__dict__)
                existing_hashes.add(paper.paper_id)
        self.bm25 = BM25Retriever(self.dense.load_all_records())
        self.hybrid = HybridRetriever(self.dense, self.bm25)
        save_state(self.state)
        return processed

    def delete_paper(self, paper_id: str) -> bool:
        paper_to_delete = None
        for paper in self.state.get("papers", []):
            if paper.get("paper_id") == paper_id:
                paper_to_delete = paper
                break
        if not paper_to_delete:
            return False

        # 1. Delete chunks from Vector DB
        self.dense.delete_paper_chunks(paper_id)

        # 2. Delete source file from uploads/ if it exists
        if paper_to_delete.get("path"):
            file_path = Path(paper_to_delete["path"])
            if file_path.is_file():
                try:
                    file_path.unlink()
                except Exception:
                    pass

        # 3. Remove paper record from state
        self.state["papers"] = [p for p in self.state["papers"] if p.get("paper_id") != paper_id]

        # 4. Re-index retrievers
        self.bm25 = BM25Retriever(self.dense.load_all_records())
        self.hybrid = HybridRetriever(self.dense, self.bm25)

        # 5. Reset last analysis so lingering output from deleted files disappears
        self.state["last_analysis"] = {}

        save_state(self.state)
        return True

    def clear_analysis(self) -> None:
        self.state["last_analysis"] = {}
        save_state(self.state)


    def analyze(self, question: str, paper_ids: list[str] | None = None) -> dict[str, Any]:
        if not self.state.get("papers"):
            return {"error": "No research papers yet. Upload PDF papers to begin."}
        if not question.strip():
            return {"error": "Enter a research question."}

        retrieval = self.hybrid.search(question, top_k=TOP_K, paper_ids=paper_ids)
        evidence = make_evidence_records(question, retrieval["hybrid"])
        verifications = verify_claim(question, evidence)
        for record, verification in zip(evidence, verifications):
            record.verification_status = verification.verdict
        answer = self._answer(question, evidence)
        comparison = compare_papers(evidence)
        disagreements = analyze_disagreements(evidence)
        matrix, matrix_summary = build_literature_matrix(evidence)
        gaps = detect_candidate_gaps(evidence)
        draft = generate_draft(question, evidence, matrix, gaps)

        self.state["last_analysis"] = {
            "question": question,
            "answer": answer,
            "evidence": [record.to_dict() for record in evidence],
            "verifications": [result.to_dict() for result in verifications],
            "comparison": comparison,
            "disagreements": disagreements,
            "matrix": matrix,
            "matrix_summary": matrix_summary,
            "gaps": gaps,
            "draft": draft,
            "paper_scope": paper_ids,
        }
        save_state(self.state)
        return self.state["last_analysis"]


    def _answer(self, question: str, evidence: list[EvidenceRecord]) -> str:
        if not evidence:
            return "Insufficient evidence found in the uploaded literature."
        context = "\n\n".join(f"[{item.paper}, p.{item.page}] {item.passage}" for item in evidence[:6])
        if self.llm.available:
            prompt = (
                "Answer the research question using only the evidence below. Cite every factual claim "
                "with [Paper, p.X]. If evidence is insufficient, say so.\n\n"
                f"Question: {question}\n\nEvidence:\n{context}"
            )
            try:
                return self.llm.generate(prompt)
            except Exception as exc:
                return extractive_answer(evidence) + f"\n\nLLM note: Gemini could not be used ({exc})."
        return extractive_answer(evidence)


def extractive_answer(evidence: list[EvidenceRecord]) -> str:
    lines = ["Evidence-grounded answer based on retrieved passages:"]
    for record in evidence[:5]:
        first_sentence = re.split(r"(?<=[.!?])\s+", record.passage)[0]
        lines.append(f"- {first_sentence} [{record.paper}, p.{record.page}]")
    return "\n".join(lines)


def generate_draft(question: str, evidence: list[EvidenceRecord], matrix: list[dict], gaps: list[dict]) -> str:
    if not evidence:
        return "No research draft yet.\nGenerate a draft from your analyzed literature."
    citations = sorted({f"{record.paper}, p.{record.page}" for record in evidence})
    gap_text = "\n".join(f"- {gap['title']}: {gap['description']}" for gap in gaps) or "- No candidate research gaps yet."
    matrix_text = "\n".join(f"- {row['Paper']}: {row['Key Findings']}" for row in matrix) or "- No matrix entries available."
    refs = "\n".join(f"- {citation}" for citation in citations)
    return f"""# Evidence-Supported Research Draft

## Abstract
This draft summarizes evidence retrieved for the research question: {question}. Claims are limited to uploaded literature. [RESULT TO BE INSERTED]

## Introduction
The analyzed literature is used to identify methods, findings, limitations, and candidate research gaps with page-level evidence.

## Related Work
{matrix_text}

## Methodology
ResearchX processes uploaded PDFs, chunks page-level text, retrieves evidence using dense retrieval and BM25, fuses rankings with weighted RRF, and verifies claims against retrieved passages.

## Results
[RESULT TO BE INSERTED]

## Discussion
The retrieved evidence supports cautious synthesis only. Candidate gaps should be reviewed by a human researcher before use.

## Candidate Research Gaps
{gap_text}

## Conclusion
The available evidence provides a structured starting point for literature analysis, but experimental claims require human validation and additional study.

## References
{refs}
"""


def draft_to_latex(markdown: str) -> str:
    latex = markdown
    latex = latex.replace("# Evidence-Supported Research Draft", "\\title{Evidence-Supported Research Draft}")
    latex = re.sub(r"^## (.+)$", r"\\section{\1}", latex, flags=re.MULTILINE)
    latex = latex.replace("[RESULT TO BE INSERTED]", "\\textbf{[RESULT TO BE INSERTED]}")
    return "\\documentclass[conference]{IEEEtran}\n\\begin{document}\n" + latex + "\n\\end{document}\n"


def load_state() -> dict[str, Any]:
    if not STATE_FILE.exists():
        return {"papers": [], "last_analysis": {}}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"papers": [], "last_analysis": {}}


def save_state(state: dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
