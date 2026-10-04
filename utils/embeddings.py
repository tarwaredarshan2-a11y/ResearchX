from __future__ import annotations

import hashlib
import math

from config import EMBEDDING_MODEL


class EmbeddingError(RuntimeError):
    pass


class EmbeddingModel:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError(
                "sentence-transformers is not installed. Install requirements.txt to enable embeddings."
            ) from exc
        try:
            self._model = SentenceTransformer(self.model_name)
        except Exception as exc:
            raise EmbeddingError(f"Could not load embedding model {self.model_name}: {exc}") from exc
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [list(map(float, vector)) for vector in vectors]


class HashEmbeddingModel:
    """Small deterministic embedder for tests only."""

    def __init__(self, dimensions: int = 64):
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            values = [0.0] * self.dimensions
            for token in text.lower().split():
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:2], "big") % self.dimensions
                values[index] += 1.0
            norm = math.sqrt(sum(value * value for value in values)) or 1.0
            vectors.append([value / norm for value in values])
        return vectors
