"""Embedding backends. All return L2-normalised vectors so dot product == cosine similarity."""

import hashlib
import logging
import re
from typing import List, Optional, Protocol

import numpy as np

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str

    def embed(self, texts: List[str]) -> List[List[float]]: ...


def _normalise(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model %s", model_name)
        self.name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vecs = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return _normalise(np.asarray(vecs, dtype=np.float32)).tolist()


class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        from openai import OpenAI

        self.name = model
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        resp = self._client.embeddings.create(model=self._model, input=texts)
        return _normalise(np.array([d.embedding for d in resp.data], dtype=np.float32)).tolist()


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder.

    Needs no model download, so it is used by the unit tests and for offline smoke tests.
    It is NOT semantically meaningful; never use it for real evaluation numbers.
    """

    def __init__(self, dim: int = 384):
        self.name = "hash"
        self.dim = dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in re.findall(r"\w+", t.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        return _normalise(out).tolist()


def build_embedder(provider: str, model: str, openai_api_key: Optional[str] = None,
                   openai_model: str = "text-embedding-3-small") -> Embedder:
    provider = provider.lower()
    if provider == "hash":
        return HashEmbedder()
    if provider == "openai":
        if not openai_api_key:
            raise ValueError("EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY")
        return OpenAIEmbedder(openai_api_key, openai_model)
    return SentenceTransformerEmbedder(model)
