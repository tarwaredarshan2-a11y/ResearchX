# ResearchX: Evidence-Grounded AI Research Assistant

## Abstract

ResearchX is a Python and Streamlit application for evidence-grounded analysis of research papers. The system ingests multiple PDFs, preserves page-level provenance, retrieves evidence using dense retrieval and BM25, fuses results with weighted Reciprocal Rank Fusion, verifies claims against retrieved passages, supports multi-paper synthesis, suggests Candidate Research Gaps, and generates an evidence-supported academic draft. Experimental evaluation has not yet been conducted; therefore quantitative results are marked as [RESULT TO BE INSERTED].

## 1. Introduction

Students and early researchers often need to compare multiple papers, inspect evidence, and identify possible research directions. Generic PDF summarizers may produce summaries without sufficient traceability. ResearchX addresses this by making evidence and citations central to the workflow.

## 2. Related Work

ResearchX draws on retrieval-augmented generation, sparse lexical retrieval, dense embedding retrieval, rank fusion, and evidence-based literature review methods. This manuscript does not fabricate external citations. Formal citations should be inserted after a reviewed bibliography is prepared.

## 3. Methodology

Uploaded PDFs are processed with PyMuPDF. Text is extracted page by page and chunked using a default size of 900 characters with 150 characters overlap. Each chunk stores paper name, paper ID, page number, section label, and chunk ID.

Dense retrieval uses BAAI/bge-small-en-v1.5 embeddings persisted in ChromaDB. Sparse retrieval uses BM25. ResearchX combines rankings using weighted Reciprocal Rank Fusion with default dense weight 0.65, sparse weight 0.35, and RRF k value 60.

## 4. System Architecture

The implementation uses a compact modular architecture:

- Streamlit user interface with Workspace, Literature, Insights, and Draft sections.
- PDF loader for ingestion and chunking.
- Embedding and ChromaDB layer for persistent dense retrieval.
- BM25 lexical retriever.
- Hybrid retriever with weighted RRF.
- Evidence, claim verification, comparison, literature review, research gap, and evaluation utilities.
- Gemini wrapper for optional LLM synthesis through a centralized interface.

## 5. Experimental Setup

The evaluation framework supports Precision@5, Recall@5, MRR, citation correctness rate, unsupported claim rate, claim verification accuracy, and evidence support rate. Benchmark labels have not yet been added.

## 6. Results

[RESULT TO BE INSERTED]

No experimental performance claims are made until benchmark data is collected and evaluated.

## 7. Discussion

ResearchX emphasizes traceable evidence rather than unsupported generated answers. Candidate Research Gaps are derived from retrieved limitations and must be reviewed by a human researcher. Differences across studies are categorized conservatively and are not automatically treated as contradictions.

## 8. Limitations

The current implementation does not include OCR for scanned PDFs. Claim verification is heuristic unless extended with a dedicated NLI model. The system depends on the quality of extracted PDF text and available benchmark labels.

## 9. Conclusion

ResearchX implements a practical evidence-grounded research assistant suitable for a 3rd-year engineering project. It connects ingestion, hybrid retrieval, evidence display, verification, synthesis, Candidate Research Gap detection, and draft generation into one maintainable application.

## References

[REFERENCES TO BE INSERTED AFTER LITERATURE REVIEW]
