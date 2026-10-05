from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from config import CHROMA_COLLECTION, TOP_K, VECTOR_DB_DIR
from utils.embeddings import EmbeddingModel
from utils.pdf_loader import Chunk


TOKEN_RE = re.compile(r"[A-Za-z0-9_+\-.]+")


@dataclass
class SearchResult:
    chunk_id: str
    text: str
    metadata: dict
    score: float
    rank: int
    source: str


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


class BM25Retriever:
    def __init__(self, chunks: Iterable[Chunk | dict] | None = None, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: list[dict] = []
        self.doc_tokens: list[list[str]] = []
        self.doc_freq: Counter[str] = Counter()
        self.avgdl = 0.0
        if chunks:
            self.index(chunks)

    def index(self, chunks: Iterable[Chunk | dict]) -> None:
        self.documents = [chunk.to_record() if isinstance(chunk, Chunk) else chunk for chunk in chunks]
        self.doc_tokens = [tokenize(doc["text"]) for doc in self.documents]
        self.doc_freq = Counter()
        for tokens in self.doc_tokens:
            self.doc_freq.update(set(tokens))
        self.avgdl = sum(len(tokens) for tokens in self.doc_tokens) / max(len(self.doc_tokens), 1)

    def search(self, query: str, top_k: int = TOP_K, paper_ids: list[str] | None = None) -> list[SearchResult]:
        if not self.documents:
            return []
        query_terms = tokenize(query)
        scores: list[tuple[int, float]] = []
        total_docs = len(self.documents)
        filter_set = set(paper_ids) if paper_ids else None
        for index, tokens in enumerate(self.doc_tokens):
            doc = self.documents[index]
            if filter_set and doc.get("metadata", {}).get("paper_id") not in filter_set:
                continue
            counts = Counter(tokens)
            score = 0.0
            doc_len = len(tokens) or 1
            for term in query_terms:
                if counts[term] == 0:
                    continue
                idf = math.log(1 + (total_docs - self.doc_freq[term] + 0.5) / (self.doc_freq[term] + 0.5))
                numerator = counts[term] * (self.k1 + 1)
                denominator = counts[term] + self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1))
                score += idf * numerator / denominator
            if score > 0:
                scores.append((index, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        results = []
        for rank, (index, score) in enumerate(scores[:top_k], start=1):
            doc = self.documents[index]
            results.append(SearchResult(doc["id"], doc["text"], doc["metadata"], score, rank, "bm25"))
        return results


class DenseRetriever:
    def __init__(self, embedding_model: EmbeddingModel | None = None, collection_name: str = CHROMA_COLLECTION):
        self.embedding_model = embedding_model or EmbeddingModel()
        self.collection_name = collection_name
        self._collection = None

    def _get_collection(self):
        if self._collection is not None:
            return self._collection
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as exc:
            raise RuntimeError("chromadb is not installed. Install requirements.txt to enable dense retrieval.") from exc
        client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR), settings=Settings(anonymized_telemetry=False))
        self._collection = client.get_or_create_collection(self.collection_name)
        return self._collection

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        collection = self._get_collection()
        records = [chunk.to_record() for chunk in chunks]
        embeddings = self.embedding_model.embed([record["text"] for record in records])
        ids = [record["id"] for record in records]
        try:
            existing = collection.get(ids=ids).get("ids", [])
            existing_ids = set(existing)
        except Exception:
            existing_ids = set()
        new_records = [(record, embedding) for record, embedding in zip(records, embeddings) if record["id"] not in existing_ids]
        if not new_records:
            return
        collection.add(
            ids=[record["id"] for record, _ in new_records],
            documents=[record["text"] for record, _ in new_records],
            metadatas=[record["metadata"] for record, _ in new_records],
            embeddings=[embedding for _, embedding in new_records],
        )

    def delete_paper_chunks(self, paper_id: str) -> None:
        collection = self._get_collection()
        try:
            collection.delete(where={"paper_id": paper_id})
        except Exception:
            data = collection.get()
            ids_to_del = [
                cid for cid, meta in zip(data.get("ids", []), data.get("metadatas", []))
                if (meta or {}).get("paper_id") == paper_id
            ]
            if ids_to_del:
                collection.delete(ids=ids_to_del)

    def search(self, query: str, top_k: int = TOP_K, paper_ids: list[str] | None = None) -> list[SearchResult]:
        collection = self._get_collection()
        query_embedding = self.embedding_model.embed([query])[0]
        
        where_filter = None
        if paper_ids:
            if len(paper_ids) == 1:
                where_filter = {"paper_id": paper_ids[0]}
            else:
                where_filter = {"paper_id": {"$in": paper_ids}}

        try:
            response = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter,
            )
        except Exception:
            # Fallback if where query fails or is empty
            response = collection.query(query_embeddings=[query_embedding], n_results=top_k * 3)

        ids = response.get("ids", [[]])[0]
        docs = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0] if response.get("distances") else [0.0] * len(ids)
        results = []
        filter_set = set(paper_ids) if paper_ids else None
        for rank, (chunk_id, doc, metadata, distance) in enumerate(zip(ids, docs, metadatas, distances), start=1):
            meta = metadata or {}
            if filter_set and meta.get("paper_id") not in filter_set:
                continue
            results.append(SearchResult(chunk_id, doc, meta, 1.0 / (1.0 + float(distance)), rank, "dense"))
            if len(results) >= top_k:
                break
        return results

    def load_all_records(self) -> list[dict]:
        collection = self._get_collection()
        data = collection.get()
        records = []
        for chunk_id, document, metadata in zip(data.get("ids", []), data.get("documents", []), data.get("metadatas", [])):
            records.append({"id": chunk_id, "text": document, "metadata": metadata or {}})
        return records

