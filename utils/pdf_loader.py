from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from config import CHUNK_OVERLAP, CHUNK_SIZE, UPLOAD_DIR


SECTION_RE = re.compile(
    r"^\s*(abstract|introduction|related work|methodology|methods?|results?|discussion|"
    r"conclusion|limitations?|references)\s*$",
    re.IGNORECASE,
)


@dataclass
class Paper:
    paper_id: str
    name: str
    path: str
    page_count: int
    status: str
    error: str = ""


@dataclass
class Chunk:
    chunk_id: str
    paper: str
    paper_id: str
    page: int
    section: str
    text: str
    metadata: dict = field(default_factory=dict)

    def to_record(self) -> dict:
        metadata = {
            "paper": self.paper,
            "paper_id": self.paper_id,
            "page": self.page,
            "section": self.section,
            "chunk_id": self.chunk_id,
        }
        metadata.update(self.metadata)
        return {"id": self.chunk_id, "text": self.text, "metadata": metadata}


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_pdf_name(filename: str) -> str:
    stem = Path(filename).stem or "paper"
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_") or "paper"
    return safe[:120] + ".pdf"


def save_uploaded_pdf(uploaded_file: BinaryIO, existing_hashes: set[str] | None = None) -> tuple[Paper, bytes]:
    data = uploaded_file.read()
    digest = file_hash(data)
    name = getattr(uploaded_file, "name", "uploaded.pdf")

    if not name.lower().endswith(".pdf"):
        return Paper(digest, name, "", 0, "Failed", "Only PDF files are supported."), data

    if existing_hashes and digest in existing_hashes:
        return Paper(digest, name, "", 0, "Duplicate", "This PDF has already been processed."), data

    target = UPLOAD_DIR / f"{digest[:12]}_{safe_pdf_name(name)}"
    target.write_bytes(data)
    return Paper(digest, Path(name).stem, str(target), 0, "Saved"), data


def extract_pdf(path: str, paper_id: str | None = None, paper_name: str | None = None) -> tuple[Paper, list[Chunk]]:
    source = Path(path)
    paper_id = paper_id or file_hash(source.read_bytes())
    paper_name = paper_name or source.stem

    try:
        import pymupdf

        doc = pymupdf.open(source)
    except ImportError as exc:
        return Paper(paper_id, paper_name, str(source), 0, "Failed", f"PyMuPDF could not be loaded: {exc}"), []
    except Exception as exc:
        return Paper(paper_id, paper_name, str(source), 0, "Failed", f"Invalid or corrupted PDF: {exc}"), []

    try:
        page_count = doc.page_count
        if page_count == 0:
            return Paper(paper_id, paper_name, str(source), 0, "Failed", "Empty PDF."), []

        chunks: list[Chunk] = []
        any_text = False
        current_section = "Unknown"

        for page_index in range(page_count):
            page = doc.load_page(page_index)
            text = page.get_text("text") or ""
            text = normalize_text(text)
            if text:
                any_text = True
            current_section = detect_section(text, current_section)
            chunks.extend(chunk_text(text, paper_name, paper_id, page_index + 1, current_section))

        if not any_text:
            return Paper(
                paper_id,
                paper_name,
                str(source),
                page_count,
                "Failed",
                "No extractable text found. The PDF may be scanned.",
            ), []

        return Paper(paper_id, paper_name, str(source), page_count, "Processed"), chunks
    finally:
        doc.close()


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_section(text: str, fallback: str = "Unknown") -> str:
    for line in text.splitlines()[:12]:
        clean = re.sub(r"^\d+(\.\d+)*\s*", "", line.strip()).strip()
        match = SECTION_RE.match(clean)
        if match:
            return match.group(1).title()
    return fallback


def chunk_text(
    text: str,
    paper: str,
    paper_id: str,
    page: int,
    section: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    if not text.strip():
        return []

    chunks: list[Chunk] = []
    start = 0
    cleaned = text.strip()
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        if end < len(cleaned):
            boundary = max(cleaned.rfind(". ", start, end), cleaned.rfind("\n", start, end))
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        piece = cleaned[start:end].strip()
        if piece:
            raw_id = f"{paper_id}:{page}:{start}:{end}"
            chunk_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()
            chunks.append(Chunk(chunk_id, paper, paper_id, page, section, piece))
        if end >= len(cleaned):
            break
        start = max(0, end - overlap)
    return chunks
