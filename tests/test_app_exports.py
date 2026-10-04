import json
from types import SimpleNamespace

import app
from app import (
    add_approved_gap_to_draft,
    build_bibtex,
    build_evidence_jsonld,
    literature_matrix_to_latex,
    persist_review_status,
)


def test_literature_matrix_latex_escapes_special_characters():
    latex = literature_matrix_to_latex(
        [
            {
                "Paper": "Paper_A & B",
                "Method": "Model_1",
                "Dataset": "Not reported",
                "Metrics": "F1: 91%",
                "Key Findings": "A result",
                "Limitations": "Not reported",
            }
        ]
    )

    assert r"Paper\_A \& B" in latex
    assert r"F1: 91\%" in latex
    assert r"\begin{tabular}" in latex


def test_bibtex_export_uses_unique_keys_and_discloses_missing_metadata():
    bibtex = build_bibtex(
        [
            {"name": "Research Paper.pdf"},
            {"name": "Research Paper.pdf"},
        ]
    )

    assert bibtex.count("@misc{") == 2
    assert "researchx_research_paper_1" in bibtex
    assert "researchx_research_paper_2" in bibtex
    assert bibtex.count("author and publication details not extracted") == 2
    escaped = build_bibtex([{"name": "A_B & C%{D}.pdf"}])
    assert r"A\_B \& C\%\{D\}" in escaped


def test_evidence_jsonld_keeps_page_and_paper_provenance():
    jsonld = json.loads(
        build_evidence_jsonld(
            [
                {
                    "paper": "Paper A",
                    "page": 7,
                    "claim": "The method improved F1.",
                    "passage": "The method improved F1 on the benchmark.",
                    "verification_status": "ENTAILED",
                    "confidence": "High",
                }
            ]
        )
    )

    assert jsonld["@context"] == "https://schema.org"
    paper, quotation = jsonld["@graph"]
    assert paper["name"] == "Paper A"
    assert quotation["isPartOf"]["@id"] == paper["@id"]
    assert quotation["pagination"] == "7"


def test_approved_gap_is_saved_with_page_citations_once(monkeypatch):
    monkeypatch.setattr(app, "save_state", lambda state: None)
    gap = {
        "title": "Limited evaluation across datasets",
        "description": "Evaluation across additional datasets may be useful.",
        "evidence": [{"paper": "Paper A", "page": 4}],
    }
    state = {
        "last_analysis": {
            "draft": "# Research draft\n\n## Candidate Research Gaps\n- AI suggestion."
        }
    }
    orchestrator = SimpleNamespace(state=state)

    persist_review_status(orchestrator, "gap", gap, "Approved")
    add_approved_gap_to_draft(orchestrator, gap)
    add_approved_gap_to_draft(orchestrator, gap)

    saved_draft = state["last_analysis"]["draft"]
    assert state["review_statuses"][app.review_key("gap", gap)] == "Approved"
    assert saved_draft.count("### Limited evaluation across datasets") == 1
    assert "[Paper A, p.4]" in saved_draft
