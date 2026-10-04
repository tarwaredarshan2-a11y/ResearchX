# ResearchX

ResearchX is an evidence-grounded AI research assistant for analyzing multiple research papers. It helps students and researchers move from uploaded PDFs to page-level evidence, claim verification, multi-paper synthesis, candidate research gaps, and an evidence-supported academic draft.

## Problem Statement

Normal PDF summarizers often produce fluent answers without showing the source evidence. ResearchX focuses on traceability: important claims are linked to extracted passages and page-level citations from uploaded papers.

## Objectives

- Ingest multiple research PDFs safely.
- Preserve paper, page, section, and chunk provenance.
- Retrieve evidence using dense retrieval, BM25, and weighted Reciprocal Rank Fusion.
- Verify generated claims as ENTAILED, NEUTRAL, or CONTRADICTED.
- Compare papers and create a literature review matrix.
- Suggest candidate research gaps from repeated limitations and evidence.
- Generate an IEEE-style draft without fabricating citations or results.

## Key Features

1. Evidence-grounded research with page-level citations.
2. Hybrid literature retrieval using dense embeddings, BM25, and RRF.
3. Claim verification using retrieved evidence.
4. Multi-paper comparison.
5. Evidence-based Candidate Research Gap detection.
6. Evidence-supported research drafting and LaTeX export.

## Workflow

Upload papers -> ask a research question -> retrieve evidence -> verify claims -> compare literature -> identify Candidate Research Gaps -> generate a research draft.

## Architecture

ResearchX is a compact Python and Streamlit application:

- `app.py`: four-section Streamlit UI.
- `utils/pdf_loader.py`: PDF extraction and chunking.
- `utils/embeddings.py`: BAAI/bge-small-en-v1.5 embedding wrapper.
- `utils/retriever.py`: Chroma dense retrieval and BM25 lexical retrieval.
- `utils/hybrid_retriever.py`: weighted Reciprocal Rank Fusion.
- `utils/evidence.py`: evidence records and page-level citations.
- `utils/claim_verifier.py`: claim-level verification.
- `utils/comparison.py`: multi-paper comparison and disagreement analysis.
- `utils/literature_review.py`: literature matrix synthesis.
- `utils/research_gap.py`: Candidate Research Gap detection.
- `utils/evaluation_suite.py`: retrieval metrics and ablation helpers.
- `utils/orchestrator.py`: end-to-end workflow.

## Technology Stack

- Python 3.10+
- Streamlit
- PyMuPDF
- BAAI/bge-small-en-v1.5 through sentence-transformers
- ChromaDB
- BM25
- Gemini through `utils/llm.py`
- `.env` for secrets

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

Create a `.env` file:

```bash
GEMINI_API_KEY=your_key_here
```

The API key is optional. Without it, ResearchX uses extractive evidence-grounded answers from retrieved passages. It does not fabricate missing research information.

## Running

```bash
streamlit run app.py
```

## Retrieval Methodology

ResearchX chunks each page with a default size of 900 characters and 150 characters of overlap. Dense retrieval uses `BAAI/bge-small-en-v1.5` embeddings stored in persistent ChromaDB. BM25 retrieves exact technical terms such as datasets, models, and abbreviations. Weighted RRF combines dense and sparse rankings:

```text
RRF(d) = 0.65 / (60 + dense_rank(d)) + 0.35 / (60 + sparse_rank(d))
```

## Evidence Verification

Evidence records include claim, paper, page, section, passage, evidence type, verification status, confidence, and review status. Claim verdicts are limited to ENTAILED, NEUTRAL, and CONTRADICTED.

## Research-Gap Methodology

ResearchX identifies Candidate Research Gaps from limitation-oriented evidence. It reports why a gap appears, supporting papers, supporting pages, passages, confidence, and review status. It does not claim to prove a true research gap automatically.

## Evaluation

The evaluation suite supports Precision@5, Recall@5, MRR, citation correctness rate, unsupported claim rate, claim verification accuracy, and evidence support rate. If benchmark data is unavailable, ResearchX reports: No evaluation data available.

## Limitations

- Scanned PDFs require OCR outside the current implementation.
- LLM output depends on Gemini availability and must be reviewed.
- Candidate Research Gaps are AI suggested and require human validation.
- Evaluation scores are not reported until labeled benchmark data is provided.

## Future Work

- Add optional OCR for scanned PDFs.
- Add benchmark import UI.
- Improve claim verification with a dedicated natural language inference model.
- Add structured PDF section detection for more paper layouts.
