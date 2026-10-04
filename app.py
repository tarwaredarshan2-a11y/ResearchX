from __future__ import annotations

import hashlib
import html
import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from config import APP_NAME, APP_SUBTITLE, REVIEW_STATUSES, VERIFICATION_STATUSES
from utils.evaluation_suite import chunk_ablation_settings
from utils.orchestrator import ResearchOrchestrator, draft_to_latex, save_state


st.set_page_config(page_title=APP_NAME, page_icon="RX", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Newsreader:opsz,wght@6..72,500;6..72,600&family=JetBrains+Mono:wght@400;500&family=Material+Symbols+Outlined:wght,FILL@400,500,600');

    :root {
        --rx-primary: #1746ad;
        --rx-primary-dark: #102f78;
        --rx-ink: #172033;
        --rx-muted: #65728a;
        --rx-border: #dfe5f0;
        --rx-page: #f6f8fc;
        --rx-card: #ffffff;
        --rx-teal: #08766c;
    }

    html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        color: var(--rx-ink);
    }
    .stApp { background: var(--rx-page); }
    [data-testid="stMainBlockContainer"] {
        max-width: 1580px;
        padding: 1.5rem 2.4rem 4rem;
        margin: 0 auto;
    }
    header[data-testid="stHeader"] {
        background: rgba(246, 248, 252, .88);
        border-bottom: 1px solid rgba(223, 229, 240, .75);
    }
    section[data-testid="stSidebar"] {
        background: #fff;
        border-right: 1px solid var(--rx-border);
    }
    section[data-testid="stSidebar"] > div {
        padding: 1.35rem 1rem 1rem;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        padding: .35rem .45rem;
        border-radius: 8px;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: #f1f4fa;
    }
    h1, h2, h3 {
        color: var(--rx-ink);
        letter-spacing: -.025em;
    }
    h1, h2 { font-family: 'Newsreader', Georgia, serif; }
    h1 { font-size: clamp(2rem, 3vw, 2.65rem); }
    h2 { font-size: 1.7rem; }
    [data-testid="stMetric"] {
        background: var(--rx-card);
        border: 1px solid var(--rx-border);
        border-radius: 12px;
        padding: .95rem 1rem;
        box-shadow: 0 1px 2px rgba(20, 35, 70, .025);
    }
    [data-testid="stMetricLabel"] { color: var(--rx-muted); }
    [data-testid="stMetricValue"] { color: var(--rx-ink); }
    [data-testid="stMetricDelta"] { font-size: 11px; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--rx-card);
        border-color: var(--rx-border);
        border-radius: 12px;
    }
    div.stButton > button, div.stDownloadButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        border-radius: 8px;
        border-color: var(--rx-border);
        font-weight: 600;
        transition: border-color .15s ease, background .15s ease, transform .15s ease;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover {
        border-color: var(--rx-primary);
        color: var(--rx-primary);
        transform: translateY(-1px);
    }
    div[data-testid="stFormSubmitButton"] > button[kind="primary"],
    div.stButton > button[kind="primary"] {
        background: var(--rx-primary);
        border-color: var(--rx-primary);
    }
    div[data-testid="stFileUploader"] {
        background: #fff;
        border-radius: 12px;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--rx-border);
        border-radius: 10px;
        overflow: hidden;
    }
    div[data-testid="stTabs"] button {
        font-weight: 600;
    }
    .rx-topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        border: 1px solid var(--rx-border);
        border-radius: 12px;
        padding: 11px 15px;
        margin: 0 0 27px;
        background: rgba(255,255,255,.85);
        box-shadow: 0 2px 8px rgba(23,32,51,.025);
    }
    .rx-topbar-title {
        color: var(--rx-ink);
        font-size: 12px;
        font-weight: 700;
    }
    .rx-topbar-sub {
        color: var(--rx-muted);
        font-size: 10px;
        margin-top: 3px;
    }
    .rx-topbar-status {
        color: var(--rx-muted);
        font: 500 10px 'JetBrains Mono', monospace;
        text-align: right;
        white-space: nowrap;
    }
    .rx-live-dot {
        display: inline-block;
        width: 7px;
        height: 7px;
        margin-right: 6px;
        border-radius: 50%;
        background: #12856d;
        box-shadow: 0 0 0 3px rgba(18,133,109,.11);
    }
    .rx-brand {
        display: flex;
        align-items: center;
        gap: 11px;
        padding: 7px 2px 18px;
        border-bottom: 1px solid var(--rx-border);
        margin-bottom: 20px;
    }
    .rx-brand-mark {
        display: grid;
        place-items: center;
        width: 38px;
        height: 38px;
        flex: 0 0 38px;
        border-radius: 11px;
        background: var(--rx-primary);
        color: white;
        font: 600 13px 'JetBrains Mono', monospace;
        letter-spacing: -.08em;
    }
    .rx-brand-name { color: var(--rx-ink); font-size: 14px; font-weight: 700; }
    .rx-brand-sub { color: var(--rx-muted); font-size: 11px; margin-top: 2px; }
    .rx-eyebrow {
        color: var(--rx-primary);
        font: 500 11px 'JetBrains Mono', monospace;
        letter-spacing: .11em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .rx-muted { color: var(--rx-muted); }
    .rx-citation {
        border-left: 3px solid var(--rx-primary);
        padding: 10px 13px;
        margin: 8px 0;
        background: #f7f9fd;
        border-radius: 0 8px 8px 0;
        color: var(--rx-ink);
    }
    .rx-verdict {
        color: var(--rx-teal);
        font: 500 11px 'JetBrains Mono', monospace;
        text-transform: uppercase;
        letter-spacing: .04em;
    }
    .rx-section-label {
        color: var(--rx-muted);
        font: 500 11px 'JetBrains Mono', monospace;
        letter-spacing: .08em;
        text-transform: uppercase;
    }
    .rx-hero {
        position: relative;
        overflow: hidden;
        padding: 26px 30px;
        margin-bottom: 19px;
        border-radius: 15px;
        background:
          radial-gradient(ellipse at 94% 20%, rgba(119,162,255,.26), transparent 32%),
          linear-gradient(115deg, #112b65 0%, #1746ad 60%, #2059c6 100%);
        color: #fff;
    }
    .rx-hero-eyebrow {
        color: #b8ccff;
        font: 500 10px 'JetBrains Mono', monospace;
        letter-spacing: .14em;
        text-transform: uppercase;
    }
    .rx-hero-title {
        max-width: 690px;
        margin-top: 8px;
        color: white;
        font: 600 clamp(25px, 3vw, 36px)/1.12 'Newsreader', Georgia, serif;
        letter-spacing: -.025em;
    }
    .rx-hero-copy {
        max-width: 680px;
        margin-top: 9px;
        color: #d8e3ff;
        font-size: 12px;
        line-height: 1.8;
    }
    .rx-hero-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 7px;
        margin-top: 17px;
    }
    .rx-hero-badge {
        border: 1px solid rgba(218,230,255,.25);
        border-radius: 99px;
        padding: 5px 9px;
        color: #edf3ff;
        background: rgba(255,255,255,.075);
        font: 400 9px 'JetBrains Mono', monospace;
    }
    .rx-section-head {
        display: flex;
        justify-content: space-between;
        align-items: end;
        gap: 12px;
        margin: 24px 0 11px;
    }
    .rx-section-head h3 {
        margin: 2px 0 0;
        font: 600 20px 'Newsreader', Georgia, serif;
        color: var(--rx-ink);
    }
    .rx-section-head p {
        margin: 0;
        color: var(--rx-muted);
        font-size: 10px;
    }
    .rx-feature-card {
        min-height: 147px;
        border: 1px solid var(--rx-border);
        border-radius: 11px;
        padding: 14px 14px 12px;
        background: #fff;
        box-shadow: 0 2px 8px rgba(23,32,51,.025);
    }
    .rx-feature-icon {
        display: grid;
        place-items: center;
        width: 31px;
        height: 31px;
        border-radius: 9px;
        margin-bottom: 10px;
        color: var(--rx-primary);
        background: #edf2ff;
        font: 20px 'Material Symbols Outlined';
    }
    .rx-feature-title {
        font-size: 11px;
        font-weight: 700;
        color: var(--rx-ink);
        margin-bottom: 4px;
    }
    .rx-feature-copy {
        color: var(--rx-muted);
        font-size: 10px;
        line-height: 1.6;
    }
    .rx-feature-meta {
        margin-top: 8px;
        color: var(--rx-teal);
        font: 500 9px 'JetBrains Mono', monospace;
        text-transform: uppercase;
        letter-spacing: .035em;
    }
    .rx-kicker {
        color: var(--rx-muted);
        font: 500 9px 'JetBrains Mono', monospace;
        letter-spacing: .08em;
        text-transform: uppercase;
    }
    .rx-paper-sheet {
        min-height: 220px;
        padding: 21px 23px;
        border: 1px solid var(--rx-border);
        border-radius: 8px;
        background: #fff;
        box-shadow: 0 4px 16px rgba(23,32,51,.045);
    }
    .rx-draft-outline {
        border-left: 2px solid #e1e7f1;
        margin: 11px 0 17px 3px;
        padding-left: 12px;
    }
    .rx-draft-outline-item {
        margin: 9px 0;
        color: #46536a;
        font-size: 10px;
    }
    @media (max-width: 700px) {
        [data-testid="stMainBlockContainer"] { padding: 1.2rem .8rem 3rem; }
        .rx-topbar { align-items: flex-start; padding: 10px; }
        .rx-topbar-status { font-size: 8px; }
        .rx-hero { padding: 22px 19px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_orchestrator() -> ResearchOrchestrator:
    return ResearchOrchestrator()


def main() -> None:
    orchestrator = get_orchestrator()
    papers = orchestrator.state.get("papers", [])

    with st.sidebar:
        st.markdown(
            """
            <div class="rx-brand">
              <div class="rx-brand-mark">RX</div>
              <div>
                <div class="rx-brand-name">ResearchX</div>
                <div class="rx-brand-sub">Evidence-grounded research</div>
              </div>
            </div>
            <div class="rx-section-label">Research workbench</div>
            """,
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Workspace navigation",
            ["Workspace", "Literature", "Insights", "Draft"],
            label_visibility="collapsed",
            key="page_navigation",
            format_func=lambda value: {
                "Workspace": "⌂   Workspace",
                "Literature": "▤   Literature",
                "Insights": "✦   Insights",
                "Draft": "¶   Draft",
            }[value],
        )
        st.divider()
        st.markdown(
            f"""
            <div class="rx-section-label">Library status</div>
            <p style="margin:.45rem 0 .2rem;font-size:13px;font-weight:600;color:#172033">
              {len(papers)} indexed {'paper' if len(papers) == 1 else 'papers'}
            </p>
            <p style="margin:0;font-size:11px;color:#65728a">
              BAAI/bge-small-en-v1.5 + BM25
            </p>
            """,
            unsafe_allow_html=True,
        )
        st.divider()
        st.caption("ResearchX · Evidence, not guesswork")

    render_product_header(page, orchestrator)

    if page == "Workspace":
        render_workspace(orchestrator)
    elif page == "Literature":
        render_literature(orchestrator)
    elif page == "Insights":
        render_insights(orchestrator)
    else:
        render_draft(orchestrator)


def render_product_header(page: str, orchestrator: ResearchOrchestrator) -> None:
    papers = orchestrator.state.get("papers", [])
    analysis = orchestrator.state.get("last_analysis", {})
    engine_status = "Gemini available" if orchestrator.llm.available else "Grounded extractive mode"
    current_question = analysis.get("question", "")
    question_status = (
        f"Latest query · {current_question[:46]}{'…' if len(current_question) > 46 else ''}"
        if current_question
        else engine_status
    )
    st.markdown(
        f"""
        <div class="rx-topbar">
          <div>
            <div class="rx-topbar-title">{APP_NAME} <span style="font-weight:400;color:#65728a">/ {page}</span></div>
            <div class="rx-topbar-sub">{APP_SUBTITLE}</div>
          </div>
          <div class="rx-topbar-status">
            <span class="rx-live-dot"></span>{len(papers)} indexed {'paper' if len(papers) == 1 else 'papers'}
            &nbsp;·&nbsp; {html.escape(question_status)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(eyebrow: str, title: str, description: str) -> None:
    st.markdown(f'<div class="rx-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    st.caption(description)


def set_active_page(page: str) -> None:
    st.session_state["page_navigation"] = page


def render_feature_catalog(analysis: dict) -> None:
    st.markdown(
        """
        <div class="rx-section-head">
          <div>
            <div class="rx-kicker">ResearchX toolkit · real project features</div>
            <h3>From papers to a defensible literature review</h3>
          </div>
          <p>Six connected steps · every insight traceable to a source</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    has_analysis = bool(analysis.get("evidence"))
    features = [
        (
            "manage_search",
            "Evidence-grounded Q&A",
            "Ask a research question and get an answer from retrieved paper passages with page-level citations.",
            "Latest answer saved" if has_analysis else "Run your first query to begin",
            "Workspace",
        ),
        (
            "hub",
            "Hybrid literature search",
            "Combine BGE-small semantic embeddings with BM25 keyword matching, merged using weighted RRF.",
            "Dense 65% · BM25 35%",
            "Workspace",
        ),
        (
            "fact_check",
            "Claim verification",
            "Check answer claims against retrieved passages as ENTAILED, NEUTRAL, or CONTRADICTED; review before citing.",
            "Heuristic · human review",
            "Literature",
        ),
        (
            "table_chart",
            "Cross-paper comparison",
            "Compare reported methods, datasets, models, metrics, findings, and limitations in one literature matrix.",
            "CSV · LaTeX export" if has_analysis else "Builds from retrieved papers",
            "Insights",
        ),
        (
            "lightbulb",
            "Candidate research gaps",
            "Surface limitation and future-work passages with supporting papers, page references, confidence, and review status.",
            "Reviewed suggestion" if has_analysis and analysis.get("gaps") else "Candidate · not a proven gap",
            "Insights",
        ),
        (
            "edit_document",
            "Academic draft & citations",
            "Edit your evidence-supported draft, add reviewed gaps, and export Markdown, LaTeX, and reference BibTeX.",
            "Saved locally · editable",
            "Draft",
        ),
    ]
    columns = st.columns(3, gap="medium")
    for index, (icon, title, description, badge, target_page) in enumerate(features):
        with columns[index % 3]:
            st.markdown(
                f"""
                <div class="rx-feature-card">
                  <div class="rx-feature-icon">{icon}</div>
                  <div class="rx-feature-title">{title}</div>
                  <div class="rx-feature-copy">{description}</div>
                  <div class="rx-feature-meta">{badge}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.button(
                f"Open {target_page.lower()} →",
                key=f"open_feature_{index}",
                on_click=set_active_page,
                args=(target_page,),
                use_container_width=True,
            )


def render_workspace(orchestrator: ResearchOrchestrator) -> None:
    render_page_header(
        "Research workspace",
        "Your literature, in context.",
        "Upload research papers, ask a question, and follow every answer back to its source.",
    )

    papers = orchestrator.state.get("papers", [])
    analysis = orchestrator.state.get("last_analysis", {})
    evidence = analysis.get("evidence", [])
    metrics = st.columns(4)
    metrics[0].metric("Indexed papers", len(papers))
    metrics[1].metric("Retrieved passages", len(evidence))
    metrics[2].metric("Verified claims", len(analysis.get("verifications", [])))
    metrics[3].metric("Candidate gaps", len(analysis.get("gaps", [])))

    st.markdown(
        """
        <section class="rx-hero">
          <div class="rx-hero-eyebrow">The evidence-first research workbench</div>
          <div class="rx-hero-title">Move from scattered papers to research you can stand behind.</div>
          <div class="rx-hero-copy">
            Build a private paper library, ask grounded questions, inspect the source behind each claim,
            and turn reviewed evidence into a structured draft.
          </div>
          <div class="rx-hero-badges">
            <span class="rx-hero-badge">PAGE-LEVEL CITATIONS</span>
            <span class="rx-hero-badge">DENSE + BM25 SEARCH</span>
            <span class="rx-hero-badge">NO FABRICATED BENCHMARK SCORES</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    render_feature_catalog(analysis)
    st.divider()

    upload_col, research_col = st.columns([0.85, 1.4], gap="large")
    with upload_col:
        st.markdown('<div class="rx-kicker">01 · Your source library</div>', unsafe_allow_html=True)
        st.subheader("Research papers")
        st.caption("Upload PDFs to extract, chunk, and index page-level text.")
        with st.container(border=True):
            uploaded = st.file_uploader(
                "Select PDF files",
                type=["pdf"],
                accept_multiple_files=True,
                help="Only PDF files are supported.",
                key="research_pdf_upload",
            )
            if st.button("Process selected papers", type="primary", disabled=not uploaded, use_container_width=True):
                with st.spinner("Extracting pages and indexing passages..."):
                    processed = orchestrator.process_uploads(uploaded)
                st.session_state["upload_results"] = [
                    {
                        "name": paper.name,
                        "status": paper.status,
                        "page_count": paper.page_count,
                        "error": paper.error,
                    }
                    for paper in processed
                ]
                st.rerun()

        for result in st.session_state.get("upload_results", []):
            if result["status"] == "Processed":
                st.success(f"{result['name']} · {result['page_count']} pages indexed")
            elif result["status"] == "Duplicate":
                st.info(f"{result['name']} is already in this library.")
            else:
                st.error(f"{result['name']}: {result['error'] or 'Processing failed.'}")

        st.markdown('<div class="rx-kicker">Indexed documents</div>', unsafe_allow_html=True)
        render_paper_library(papers, compact=True)
        if papers:
            total_pages = sum(int(paper.get("page_count", 0) or 0) for paper in papers)
            total_chunks = len(orchestrator.bm25.documents)
            st.caption(f"Index telemetry · {total_pages:,} extracted pages · {total_chunks:,} text passages")

    with research_col:
        st.markdown('<div class="rx-kicker">02 · Evidence-grounded query</div>', unsafe_allow_html=True)
        st.subheader("Ask your literature")
        st.caption("Ask a focused question. ResearchX retrieves source passages before drafting an answer.")
        suggestions = [
            "What are the key limitations?",
            "Compare evaluation metrics",
            "Summarize dataset splits",
        ]
        suggestion_cols = st.columns(len(suggestions))
        for index, suggestion in enumerate(suggestions):
            if suggestion_cols[index].button(suggestion, key=f"question_suggestion_{index}"):
                st.session_state["research_question"] = suggestion
                st.rerun()

        with st.container(border=True):
            with st.form("research_question_form"):
                question = st.text_input(
                    "Research question",
                    placeholder="e.g. Which methods perform best across these studies?",
                    key="research_question",
                )
                submitted = st.form_submit_button(
                    "✦  Analyze literature",
                    type="primary",
                    disabled=not papers,
                    use_container_width=True,
                )
            if submitted:
                with st.spinner("Retrieving evidence and verifying claims..."):
                    result = orchestrator.analyze(question)
                if "error" in result:
                    st.warning(result["error"])
                else:
                    st.rerun()

        analysis = orchestrator.state.get("last_analysis", {})
        if analysis:
            render_synthesis_canvas(orchestrator, analysis)
        elif not papers:
            st.info("Your synthesis canvas will appear here after you add papers and run your first research question.")


def render_paper_library(papers: list[dict], compact: bool = False) -> None:
    if not papers:
        st.info("Your library is empty. Upload a PDF to start building your evidence base.")
        return

    for index, paper in enumerate(papers):
        with st.container(border=True):
            st.markdown(f"**{paper.get('name', 'Untitled paper')}**")
            status = paper.get("status", "Unknown")
            page_count = paper.get("page_count", 0)
            if status == "Processed":
                st.caption(f"Indexed · {page_count} {'page' if page_count == 1 else 'pages'}")
            else:
                st.caption(status)
                if paper.get("error"):
                    st.error(paper["error"])
            if not compact and paper.get("paper_id"):
                st.caption(f"Document ID · {paper['paper_id'][:12]}")
            source = Path(paper.get("path", ""))
            if source.is_file():
                st.download_button(
                    "Download source PDF",
                    data=source.read_bytes(),
                    file_name=source.name,
                    mime="application/pdf",
                    key=f"paper_download_{index}_{paper.get('paper_id', '')}",
                    use_container_width=True,
                )


def render_synthesis_canvas(orchestrator: ResearchOrchestrator, analysis: dict) -> None:
    evidence = analysis.get("evidence", [])
    verifications = analysis.get("verifications", [])
    verdict_counts = {
        verdict: sum(item.get("verdict") == verdict for item in verifications)
        for verdict in ("ENTAILED", "NEUTRAL", "CONTRADICTED")
    }

    st.markdown('<div class="rx-kicker">03 · Synthesized research answer</div>', unsafe_allow_html=True)
    with st.container(border=True):
        title_col, status_col = st.columns([4, 1])
        with title_col:
            st.markdown("### Evidence-backed synthesis")
            st.caption(analysis.get("question", "Latest research question"))
        with status_col:
            st.caption(f"{len(evidence)} cited passages")
        st.markdown(analysis.get("answer", "Insufficient evidence found in the uploaded literature."))

        verdict_columns = st.columns(3)
        for index, verdict in enumerate(("ENTAILED", "NEUTRAL", "CONTRADICTED")):
            verdict_columns[index].metric(verdict.title(), verdict_counts[verdict])

        if verifications:
            with st.expander("Claim-by-claim verification notes", expanded=False):
                for verification in verifications:
                    st.markdown(
                        f"**{verification.get('verdict', 'NEUTRAL')}** · "
                        f"{verification.get('source', 'Unknown source')} · "
                        f"p.{verification.get('page', '?')}"
                    )
                    st.write(verification.get("explanation", ""))
                    st.text(verification.get("evidence", ""))
                    st.divider()

        if analysis.get("comparison"):
            st.markdown("**Quick cross-paper comparison**")
            st.dataframe(
                pd.DataFrame(analysis["comparison"]),
                use_container_width=True,
                hide_index=True,
            )

        action_columns = st.columns([1.25, 1, 1])
        action_columns[0].button(
            "Add answer to Related Work",
            type="primary",
            use_container_width=True,
            key="add_synthesis_to_draft",
            on_click=add_answer_to_draft,
            args=(orchestrator,),
        )
        action_columns[1].button(
            "Review evidence →",
            use_container_width=True,
            key="open_literature_from_synthesis",
            on_click=set_active_page,
            args=("Literature",),
        )
        action_columns[2].download_button(
            "Export evidence CSV",
            data=pd.DataFrame(evidence).to_csv(index=False) if evidence else "",
            file_name="researchx_evidence.csv",
            mime="text/csv",
            disabled=not evidence,
            use_container_width=True,
            key="export_workspace_evidence_csv",
        )

    with st.expander("Source passages and citations", expanded=True):
        render_evidence_cards(evidence[:5], orchestrator, show_review=False)


def add_answer_to_draft(orchestrator: ResearchOrchestrator) -> None:
    analysis = orchestrator.state.get("last_analysis", {})
    answer = analysis.get("answer", "").strip()
    if not answer:
        return

    citations = sorted(
        {
            f"[{item.get('paper', 'Unknown paper')}, p.{item.get('page', '?')}]"
            for item in analysis.get("evidence", [])
        }
    )
    citation_text = " ".join(citations)
    related_work_entry = (
        answer if citation_text in answer else f"{answer}\n\nSources: {citation_text}"
    )
    draft = analysis.get("draft", "")
    section_pattern = re.compile(r"(## Related Work\s*\n)(.*?)(?=\n## |\Z)", flags=re.DOTALL)
    match = section_pattern.search(draft)
    if match:
        current_section = match.group(2).strip()
        if answer not in current_section:
            new_section = f"{current_section}\n\n{related_work_entry}".strip()
            draft = f"{draft[:match.start(2)]}{new_section}{draft[match.end(2):]}"
    elif related_work_entry not in draft:
        draft = f"{draft.rstrip()}\n\n## Related Work\n\n{related_work_entry}\n"

    analysis["draft"] = draft
    orchestrator.state["last_analysis"] = analysis
    save_state(orchestrator.state)


def render_literature(orchestrator: ResearchOrchestrator) -> None:
    render_page_header(
        "Literature",
        "Sources behind the synthesis.",
        "Inspect indexed papers, trace claims to page-level passages, and compare the evidence across studies.",
    )
    papers = orchestrator.state.get("papers", [])
    analysis = orchestrator.state.get("last_analysis", {})
    evidence = analysis.get("evidence", [])
    verifications = analysis.get("verifications", [])
    summary = st.columns(4)
    summary[0].metric("Indexed papers", len(papers))
    summary[1].metric("Evidence passages", len(evidence))
    summary[2].metric("Claim verdicts", len(verifications))
    summary[3].metric("Compared studies", len(analysis.get("comparison", [])))

    action_left, action_right = st.columns([1, 3])
    action_left.button(
        "＋  Add papers",
        key="literature_add_papers",
        on_click=set_active_page,
        args=("Workspace",),
        use_container_width=True,
    )
    with action_right:
        if papers:
            st.download_button(
                "Download references · BibTeX",
                data=build_bibtex(papers),
                file_name="researchx_references.bib",
                mime="application/x-bibtex",
                key="literature_export_bibtex",
            )

    papers_tab, evidence_tab, comparison_tab = st.tabs(
        [
            f"Indexed papers · {len(papers)}",
            f"Evidence & verification · {len(evidence)}",
            f"Comparison · {len(analysis.get('comparison', []))}",
        ]
    )

    with papers_tab:
        if papers:
            table = [
                {
                    "Paper": paper.get("name", ""),
                    "Pages": paper.get("page_count", 0),
                    "Status": paper.get("status", "Unknown"),
                }
                for paper in papers
            ]
            st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
            st.caption("Source PDFs stay in this project's uploads folder.")
        else:
            st.info("No papers have been indexed yet. Add PDFs from the Workspace.")

    with evidence_tab:
        evidence = analysis.get("evidence", [])
        if not evidence:
            st.info("Analyze a research question in Workspace to retrieve page-level evidence.")
        else:
            csv_col, jsonld_col, review_filter_col, filter_col = st.columns([1, 1, 1.15, 1.5])
            with csv_col:
                st.download_button(
                    "Export evidence CSV",
                    data=pd.DataFrame(evidence).to_csv(index=False),
                    file_name="researchx_evidence.csv",
                    mime="text/csv",
                    key="export_evidence_csv",
                )
            with jsonld_col:
                st.download_button(
                    "Export JSON-LD",
                    data=build_evidence_jsonld(evidence),
                    file_name="researchx_evidence.jsonld",
                    mime="application/ld+json",
                    key="export_evidence_jsonld",
                )
            with filter_col:
                verification_filter = st.selectbox(
                    "Verification status",
                    ["All statuses", *VERIFICATION_STATUSES],
                    key="evidence_verification_filter",
                )
            with review_filter_col:
                review_filter = st.selectbox(
                    "Review status",
                    ["All review states", *REVIEW_STATUSES],
                    key="evidence_review_filter",
                )
            visible = [
                record for record in evidence
                if (
                    verification_filter == "All statuses"
                    or record.get("verification_status") == verification_filter
                )
                and (
                    review_filter == "All review states"
                    or orchestrator.state.get("review_statuses", {}).get(
                        review_key("evidence", record),
                        record.get("review_status", REVIEW_STATUSES[0]),
                    ) == review_filter
                )
            ]
            st.caption(f"Showing {len(visible)} of {len(evidence)} retrieved passages.")
            render_evidence_cards(visible, orchestrator, show_review=True)

    with comparison_tab:
        comparison = analysis.get("comparison", [])
        disagreements = analysis.get("disagreements", [])
        if not comparison:
            st.info("Analyze a question to build a cross-paper comparison from retrieved evidence.")
        else:
            st.subheader("Cross-paper comparison")
            st.dataframe(pd.DataFrame(comparison), use_container_width=True, hide_index=True)
            st.download_button(
                "Export comparison CSV",
                data=pd.DataFrame(comparison).to_csv(index=False),
                file_name="researchx_comparison.csv",
                mime="text/csv",
                key="export_comparison_csv",
            )
            st.subheader("Potential differences to review")
            if disagreements:
                st.dataframe(pd.DataFrame(disagreements), use_container_width=True, hide_index=True)
                st.caption(
                    "Difference labels are heuristic cues from retrieved passages, not independent statistical tests."
                )
            else:
                st.info("No paper-to-paper differences were identified in the retrieved passages.")
            st.markdown("**Synthesis context**")
            st.info(
                "This comparison reflects only the passages retrieved for the latest research question. "
                "A missing field means it was not reported in those passages, not that the paper never reports it."
            )


def render_evidence_cards(
    evidence: list[dict],
    orchestrator: ResearchOrchestrator,
    show_review: bool = True,
) -> None:
    if not evidence:
        st.info("No retrieved evidence matches this view.")
        return

    review_statuses = orchestrator.state.get("review_statuses", {})
    for record in evidence:
        key = review_key("evidence", record)
        status = review_statuses.get(key, record.get("review_status", REVIEW_STATUSES[0]))
        record["review_status"] = status
        with st.container(border=True):
            heading, verdict = st.columns([4, 1])
            with heading:
                st.markdown(f"**{record.get('paper', 'Unknown paper')} · p.{record.get('page', '?')}**")
                st.caption(f"{record.get('section', 'Unknown section')} · Confidence: {record.get('confidence', 'Unknown')}")
            with verdict:
                st.markdown(f'<div class="rx-verdict">{record.get("verification_status", "NEUTRAL")}</div>', unsafe_allow_html=True)
            if record.get("claim"):
                st.text(f"Claim: {record['claim']}")
            st.text(record.get("passage", ""))
            source_paper = next(
                (
                    paper
                    for paper in orchestrator.state.get("papers", [])
                    if paper.get("name") == record.get("paper")
                ),
                None,
            )
            if source_paper and Path(source_paper.get("path", "")).is_file():
                source_path = Path(source_paper["path"])
                st.download_button(
                    f"View source PDF · page {record.get('page', '?')}",
                    data=source_path.read_bytes(),
                    file_name=source_path.name,
                    mime="application/pdf",
                    key=f"source_pdf_{key}",
                )
            if show_review:
                selected = st.selectbox(
                    "Human review",
                    REVIEW_STATUSES,
                    index=REVIEW_STATUSES.index(status) if status in REVIEW_STATUSES else 0,
                    key=f"review_evidence_{key}",
                )
                if selected != status:
                    persist_review_status(orchestrator, "evidence", record, selected)
                    st.rerun()


def render_insights(orchestrator: ResearchOrchestrator) -> None:
    render_page_header(
        "Research insights",
        "Patterns worth a closer look.",
        "Explore the literature matrix and review candidate gaps surfaced by the evidence. AI suggestions are not proof.",
    )
    analysis = orchestrator.state.get("last_analysis", {})
    gaps = analysis.get("gaps", [])
    matrices = analysis.get("matrix", [])
    review_statuses = orchestrator.state.get("review_statuses", {})
    approved_gaps = sum(
        review_statuses.get(review_key("gap", gap), gap.get("review_status")) == "Approved"
        for gap in gaps
    )
    summary = st.columns(4)
    summary[0].metric("Papers in matrix", len(matrices))
    summary[1].metric("Candidate research gaps", len(gaps))
    summary[2].metric("Human approved", approved_gaps)
    summary[3].metric("Evidence passages", len(analysis.get("evidence", [])))

    matrix_tab, gaps_tab, evaluation_tab = st.tabs(
        [
            f"Literature matrix · {len(matrices)}",
            f"Candidate research gaps · {len(gaps)}",
            "Evaluation",
        ]
    )

    with matrix_tab:
        matrix = analysis.get("matrix", [])
        if not matrix:
            st.info("Analyze a research question to build a structured literature matrix.")
        else:
            st.dataframe(pd.DataFrame(matrix), use_container_width=True, hide_index=True)
            csv_col, latex_col = st.columns(2)
            with csv_col:
                st.download_button(
                    "Export matrix CSV",
                    data=pd.DataFrame(matrix).to_csv(index=False),
                    file_name="researchx_literature_matrix.csv",
                    mime="text/csv",
                    key="export_matrix_csv",
                )
            with latex_col:
                st.download_button(
                    "Export LaTeX table",
                    data=literature_matrix_to_latex(matrix),
                    file_name="researchx_literature_matrix.tex",
                    mime="text/plain",
                    key="export_matrix_latex",
                )
            summary = analysis.get("matrix_summary", {})
            if summary:
                st.subheader("Synthesis notes")
                summary_cols = st.columns(2)
                for index, (title, value) in enumerate(summary.items()):
                    with summary_cols[index % 2]:
                        with st.container(border=True):
                            st.caption(title)
                            st.write(value)

    with gaps_tab:
        gaps = analysis.get("gaps", [])
        if not gaps:
            st.info("Analyze literature to surface limitations and possible research gaps.")
        else:
            st.warning(
                "These are AI-suggested candidates extracted from limitation-oriented passages in the retrieved papers. "
                "They are not proof of novelty; check current and broader literature before claiming a research gap."
            )
            filter_col, review_filter_col, export_col = st.columns([1, 1, 1])
            with filter_col:
                confidence_filter = st.selectbox(
                    "Confidence",
                    ["All confidence levels", "High", "Medium", "Low"],
                    key="gap_confidence_filter",
                )
            with review_filter_col:
                status_filter = st.selectbox(
                    "Review status",
                    ["All review states", *REVIEW_STATUSES],
                    key="gap_review_filter",
                )
            with export_col:
                st.download_button(
                    "Export gaps CSV",
                    data=pd.DataFrame(
                        [
                            {
                                "Title": gap.get("title", ""),
                                "Description": gap.get("description", ""),
                                "Confidence": gap.get("confidence", ""),
                                "Review status": gap.get("review_status", REVIEW_STATUSES[0]),
                                "Supporting papers": "; ".join(gap.get("supporting_papers", [])),
                            }
                            for gap in gaps
                        ]
                    ).to_csv(index=False),
                    file_name="researchx_candidate_gaps.csv",
                    mime="text/csv",
                    key="export_gaps_csv",
                )
            visible_gaps = [
                gap for gap in gaps
                if (
                    confidence_filter == "All confidence levels"
                    or gap.get("confidence") == confidence_filter
                )
                and (
                    status_filter == "All review states"
                    or review_statuses.get(
                        review_key("gap", gap),
                        gap.get("review_status", REVIEW_STATUSES[0]),
                    ) == status_filter
                )
            ]
            st.caption(f"{len(visible_gaps)} candidate {'gap' if len(visible_gaps) == 1 else 'gaps'}")
            for gap in visible_gaps:
                render_gap_card(orchestrator, gap)

    with evaluation_tab:
        st.write(
            "Retrieval scores are shown only when labeled benchmark queries are available. "
            "This project currently has no benchmark labels loaded."
        )
        st.info("No evaluation data available.")
        with st.expander("Retrieval configuration and ablation settings"):
            st.caption("Hybrid retrieval uses dense and BM25 rankings combined with weighted Reciprocal Rank Fusion.")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Dense weight": dense, "Sparse weight": sparse}
                        for dense, sparse in [
                            (1.0, 0.0),
                            (0.75, 0.25),
                            (0.65, 0.35),
                            (0.5, 0.5),
                            (0.25, 0.75),
                            (0.0, 1.0),
                        ]
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.dataframe(
                pd.DataFrame(chunk_ablation_settings()),
                use_container_width=True,
                hide_index=True,
            )


def render_gap_card(orchestrator: ResearchOrchestrator, gap: dict) -> None:
    key = review_key("gap", gap)
    status = orchestrator.state.get("review_statuses", {}).get(
        key,
        gap.get("review_status", REVIEW_STATUSES[0]),
    )
    gap["review_status"] = status
    with st.container(border=True):
        title_col, confidence_col = st.columns([4, 1])
        with title_col:
            st.markdown(f"### {html.escape(gap.get('title', 'Candidate gap'))}")
        with confidence_col:
            st.caption(f"{gap.get('confidence', 'Unknown')} confidence")
        st.write(gap.get("description", ""))
        st.caption(gap.get("why_it_appears", ""))
        supporting = gap.get("supporting_papers", [])
        if supporting:
            st.caption(f"Supporting papers · {', '.join(supporting)}")
        with st.expander(f"Review supporting context ({len(gap.get('evidence', []))} passages)"):
            for item in gap.get("evidence", []):
                st.text(f"{item.get('paper', 'Unknown paper')} · p.{item.get('page', '?')}")
                st.text(item.get("passage", ""))
        controls = st.columns([2, 1, 1])
        selected = controls[0].selectbox(
            "Review status",
            REVIEW_STATUSES,
            index=REVIEW_STATUSES.index(status) if status in REVIEW_STATUSES else 0,
            key=f"review_gap_{key}",
        )
        if selected != status:
            persist_review_status(orchestrator, "gap", gap, selected)
            st.rerun()
        if controls[1].button("Reject", key=f"reject_gap_{key}", use_container_width=True):
            persist_review_status(orchestrator, "gap", gap, "Rejected")
            st.rerun()
        if controls[2].button(
            "Approve & add to draft",
            key=f"approve_gap_{key}",
            type="primary",
            use_container_width=True,
        ):
            persist_review_status(orchestrator, "gap", gap, "Approved")
            add_approved_gap_to_draft(orchestrator, gap)
            st.rerun()


def render_draft(orchestrator: ResearchOrchestrator) -> None:
    analysis = orchestrator.state.get("last_analysis", {})
    render_page_header(
        "Research draft",
        "A draft you can actually work with.",
        "Edit the evidence-supported draft, save it to this workspace, or export it for your writing workflow.",
    )
    draft = analysis.get("draft", "")
    if not draft:
        st.info("Your draft will be prepared after you analyze a research question in Workspace.")
        return

    if st.session_state.get("draft_source") != draft:
        st.session_state["draft_source"] = draft
        st.session_state["draft_editor"] = draft

    edit_col, preview_col = st.columns([1.05, 0.95], gap="large")
    with edit_col:
        st.subheader("Editor")
        edited_draft = st.text_area(
            "Draft content",
            key="draft_editor",
            height=560,
            label_visibility="collapsed",
        )
    with preview_col:
        st.subheader("Preview")
        with st.container(border=True):
            st.markdown(edited_draft)

    save_col, markdown_col, latex_col, bibtex_col = st.columns([1.2, 1, 1, 1])
    with save_col:
        if st.button("Save changes", type="primary", use_container_width=True):
            analysis["draft"] = edited_draft
            orchestrator.state["last_analysis"] = analysis
            save_state(orchestrator.state)
            st.session_state["draft_source"] = edited_draft
            st.success("Draft saved to this ResearchX workspace.")
    with markdown_col:
        st.download_button(
            "Export Markdown",
            data=edited_draft,
            file_name="researchx_draft.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with latex_col:
        st.download_button(
            "Export LaTeX",
            data=draft_to_latex(edited_draft),
            file_name="researchx_draft.tex",
            mime="text/plain",
            use_container_width=True,
        )
    with bibtex_col:
        st.download_button(
            "Export BibTeX",
            data=build_bibtex(orchestrator.state.get("papers", [])),
            file_name="researchx_references.bib",
            mime="application/x-bibtex",
            use_container_width=True,
        )
    st.caption("BibTeX entries use the uploaded PDF title only; author and publication details are not inferred.")


def review_key(kind: str, item: dict) -> str:
    if kind == "gap":
        identity = item.get("title", "")
    else:
        identity = "|".join(
            str(item.get(field, ""))
            for field in ("paper", "page", "claim", "passage")
        )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return f"{kind}:{digest}"


def persist_review_status(
    orchestrator: ResearchOrchestrator,
    kind: str,
    item: dict,
    status: str,
) -> None:
    key = review_key(kind, item)
    orchestrator.state.setdefault("review_statuses", {})[key] = status
    item["review_status"] = status

    analysis = orchestrator.state.get("last_analysis", {})
    collection = "gaps" if kind == "gap" else "evidence"
    for stored_item in analysis.get(collection, []):
        if review_key(kind, stored_item) == key:
            stored_item["review_status"] = status
    save_state(orchestrator.state)


def add_approved_gap_to_draft(orchestrator: ResearchOrchestrator, gap: dict) -> None:
    analysis = orchestrator.state.get("last_analysis", {})
    draft = analysis.get("draft", "")
    section = "## Approved Candidate Research Gaps"
    if section not in draft:
        draft = f"{draft.rstrip()}\n\n{section}\n"

    title = gap.get("title", "Candidate research gap")
    if f"### {title}" not in draft:
        citations = sorted(
            {
                f"[{item.get('paper', 'Unknown paper')}, p.{item.get('page', '?')}]"
                for item in gap.get("evidence", [])
            }
        )
        references = ", ".join(citations) if citations else "No page-level evidence recorded"
        entry = (
            f"\n### {title}\n"
            f"{gap.get('description', '')} "
            f"This candidate was approved for further human review. Supporting passages: {references}.\n"
        )
        draft = f"{draft.rstrip()}\n{entry}"

    analysis["draft"] = draft
    orchestrator.state["last_analysis"] = analysis
    save_state(orchestrator.state)


def literature_matrix_to_latex(matrix: list[dict]) -> str:
    columns = [
        ("Paper", "Paper"),
        ("Method", "Method"),
        ("Dataset", "Dataset"),
        ("Metrics", "Metrics"),
        ("Key Findings", "Key Findings"),
        ("Limitations", "Limitations"),
    ]

    def escape(value: object) -> str:
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
        }
        return "".join(replacements.get(character, character) for character in str(value or "")).replace("\n", " ")

    output = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Evidence-grounded literature comparison}",
        r"\begin{tabular}{p{2.2cm}p{2.5cm}p{2.2cm}p{1.5cm}p{3.2cm}p{3.2cm}}",
        r"\hline",
        " & ".join(escape(label) for _, label in columns) + r" \\",
        r"\hline",
    ]
    for row in matrix:
        output.append(" & ".join(escape(row.get(field, "")) for field, _ in columns) + r" \\")
    output.extend([r"\hline", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(output) + "\n"


def build_bibtex(papers: list[dict]) -> str:
    entries = []
    for index, paper in enumerate(papers, start=1):
        title = str(paper.get("name", f"Uploaded paper {index}"))
        key_base = re.sub(r"[^A-Za-z0-9]+", "_", Path(title).stem).strip("_").lower()
        key = f"researchx_{key_base or 'paper'}_{index}"
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
        }
        escaped_title = "".join(replacements.get(character, character) for character in title)
        entries.append(
            f"@misc{{{key},\n"
            f"  title = {{{escaped_title}}},\n"
            "  note = {Uploaded PDF; author and publication details not extracted}\n"
            "}"
        )
    return "\n\n".join(entries) + ("\n" if entries else "")


def build_evidence_jsonld(evidence: list[dict]) -> str:
    graph = []
    paper_ids: dict[str, str] = {}
    for record in evidence:
        paper = str(record.get("paper", "Unknown paper"))
        paper_id = paper_ids.setdefault(
            paper,
            f"urn:researchx:paper:{hashlib.sha256(paper.encode('utf-8')).hexdigest()}",
        )
        graph.append(
            {
                "@type": "Quotation",
                "text": record.get("passage", ""),
                "isPartOf": {"@id": paper_id},
                "pagination": str(record.get("page", "")),
                "headline": record.get("claim", ""),
                "additionalProperty": [
                    {
                        "@type": "PropertyValue",
                        "name": "Verification status",
                        "value": record.get("verification_status", "NEUTRAL"),
                    },
                    {
                        "@type": "PropertyValue",
                        "name": "Confidence",
                        "value": record.get("confidence", "Unknown"),
                    },
                ],
            }
        )
    papers = [
        {"@id": paper_id, "@type": "ScholarlyArticle", "name": paper}
        for paper, paper_id in paper_ids.items()
    ]
    return json.dumps(
        {"@context": "https://schema.org", "@graph": [*papers, *graph]},
        ensure_ascii=False,
        indent=2,
    )


if __name__ == "__main__":
    main()
