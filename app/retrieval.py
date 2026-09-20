"""Hybrid retrieval: ChromaDB vector search + BM25 keyword search, fused, optionally cross-encoder reranked.

Design notes
- Every candidate gets an *absolute* cosine score against the query (comparable across queries).
  That score, not the per-query normalised fusion score, is used for the relevance threshold.
- Fusion is either weighted Reciprocal Rank Fusion ("rrf") or min-max weighted sum ("minmax").
  `alpha` is the weight on the vector side (1.0 = vector only, 0.0 = BM25 only).
- The optional reranker is a cross-encoder (query and chunk scored jointly), which is a real
  second opinion, unlike re-scoring with the same bi-encoder embeddings.
"""

import logging
import math
import os
import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from app.embeddings import Embedder

logger = logging.getLogger(__name__)

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "how", "in", "is",
    "it", "its", "of", "on", "or", "that", "the", "to", "was", "were", "what", "when", "where", "which",
    "who", "why", "will", "with", "does", "do", "did", "this", "these", "those", "there", "their", "can",
}


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9][a-z0-9\-\.]*[a-z0-9]|[a-z0-9]", text.lower())
    filtered = [t for t in tokens if t not in STOP_WORDS]
    return filtered or tokens


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    va, vb = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(x, 50.0), -50.0)))


class RetrievalResult:
    def __init__(self, chunk_id: str, text: str, metadata: Dict[str, Any], *, vector_rank: Optional[int] = None,
                 bm25_rank: Optional[int] = None, cosine_score: float = 0.0, bm25_score: float = 0.0,
                 fused_score: float = 0.0, rerank_score: Optional[float] = None):
        self.chunk_id = chunk_id
        self.text = text
        self.metadata = metadata
        self.vector_rank = vector_rank
        self.bm25_rank = bm25_rank
        self.cosine_score = cosine_score
        self.bm25_score = bm25_score
        self.fused_score = fused_score
        self.rerank_score = rerank_score  # sigmoid(cross-encoder logit) in [0, 1], or None

    @property
    def filename(self) -> str:
        return str(self.metadata.get("filename", "unknown"))

    @property
    def page_number(self) -> int:
        return int(self.metadata.get("page_number", 1))

    @property
    def doc_id(self) -> str:
        return str(self.metadata.get("doc_id", "unknown"))

    @property
    def score(self) -> float:
        """Score shown to users: reranker score if present, else absolute cosine similarity."""
        return self.rerank_score if self.rerank_score is not None else self.cosine_score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "filename": self.filename,
            "page_number": self.page_number,
            "text": self.text,
            "score": round(self.score, 4),
            "cosine_score": round(self.cosine_score, 4),
            "bm25_score": round(self.bm25_score, 4),
            "citation_tag": f"[{self.filename}, p. {self.page_number}]",
        }


class CrossEncoderReranker:
    """Lazy wrapper around a sentence-transformers CrossEncoder."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        logger.info("Loading cross-encoder %s", model_name)
        self.name = model_name
        self._model = CrossEncoder(model_name)

    def score(self, query: str, texts: List[str]) -> List[float]:
        if not texts:
            return []
        logits = self._model.predict([(query, t) for t in texts], show_progress_bar=False)
        return [float(x) for x in np.asarray(logits).reshape(-1)]


class HybridRetriever:
    def __init__(
        self,
        embedder: Embedder,
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "rag_documents",
        default_alpha: float = 0.6,
        fusion_method: str = "rrf",
        reranker: Optional[Any] = None,
        min_cosine_score: float = 0.20,
        min_rerank_score: float = 0.05,
    ):
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self.embedder = embedder
        self.default_alpha = default_alpha
        self.fusion_method = fusion_method
        self.reranker = reranker
        self.min_cosine_score = min_cosine_score
        self.min_rerank_score = min_rerank_score

        os.makedirs(persist_directory, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=persist_directory, settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        self.records: Dict[str, Dict[str, Any]] = {}
        self._bm25 = None
        self._bm25_ids: List[str] = []
        self._load_from_chroma()

    # ------------------------------------------------------------------ indexing
    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.embedder.embed(texts)

    def _load_from_chroma(self) -> None:
        data = self.collection.get(include=["documents", "metadatas", "embeddings"])
        ids = data.get("ids") or []
        docs = data.get("documents")
        metas = data.get("metadatas")
        embs = data.get("embeddings")
        for i, cid in enumerate(ids):
            self.records[cid] = {
                "text": docs[i],
                "metadata": dict(metas[i]),
                "embedding": [float(x) for x in embs[i]] if embs is not None else None,
            }
        self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        from rank_bm25 import BM25Okapi

        self._bm25_ids = list(self.records.keys())
        corpus = [tokenize(self.records[c]["text"]) for c in self._bm25_ids]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def add_chunks(self, chunks: List[Any]) -> int:
        if not chunks:
            return 0
        embeddings = self.embed([c.text for c in chunks])
        self.collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )
        for c, e in zip(chunks, embeddings):
            self.records[c.chunk_id] = {"text": c.text, "metadata": dict(c.metadata), "embedding": e}
        self._rebuild_bm25()
        return len(chunks)

    def delete_document(self, doc_id_or_filename: str) -> int:
        ids = [
            cid for cid, r in self.records.items()
            if r["metadata"].get("doc_id") == doc_id_or_filename or r["metadata"].get("filename") == doc_id_or_filename
        ]
        if not ids:
            return 0
        self.collection.delete(ids=ids)
        for cid in ids:
            self.records.pop(cid, None)
        self._rebuild_bm25()
        return len(ids)

    def list_documents(self) -> List[Dict[str, Any]]:
        docs: Dict[str, Dict[str, Any]] = {}
        for r in self.records.values():
            m = r["metadata"]
            d = docs.setdefault(m["doc_id"], {
                "doc_id": m["doc_id"], "filename": m["filename"], "total_pages": m.get("total_pages", 0),
                "chunk_count": 0, "pages": set(),
            })
            d["chunk_count"] += 1
            d["pages"].add(m.get("page_number", 1))
        return [
            {"doc_id": d["doc_id"], "filename": d["filename"], "total_pages": d["total_pages"],
             "chunk_count": d["chunk_count"], "pages_indexed": len(d["pages"])}
            for d in docs.values()
        ]

    # ------------------------------------------------------------------ search
    def _allowed(self, cid: str, doc_ids: Optional[List[str]]) -> bool:
        return not doc_ids or self.records[cid]["metadata"].get("doc_id") in doc_ids

    def overview_chunks(self, doc_ids: Optional[List[str]] = None, max_pages: int = 2, limit: int = 6) -> List[RetrievalResult]:
        """Opening chunks of the selected documents; used for 'what is this document about?' questions."""
        picked = []
        for cid, r in self.records.items():
            m = r["metadata"]
            if self._allowed(cid, doc_ids) and int(m.get("page_number", 1)) <= max_pages:
                picked.append((m.get("doc_id"), int(m.get("page_number", 1)), int(m.get("chunk_index", 0)), cid))
        picked.sort()
        return [
            RetrievalResult(cid, self.records[cid]["text"], self.records[cid]["metadata"], cosine_score=1.0)
            for _, _, _, cid in picked[:limit]
        ]

    def search(
        self,
        query: str,
        top_k: int = 4,
        mode: str = "hybrid",
        alpha: Optional[float] = None,
        rerank: Optional[bool] = None,
        doc_ids: Optional[List[str]] = None,
        apply_threshold: bool = True,
    ) -> List[RetrievalResult]:
        """mode: 'vector' | 'bm25' | 'hybrid'."""
        if not self.records:
            return []
        mode = mode.lower()
        if mode not in {"vector", "bm25", "hybrid"}:
            raise ValueError("mode must be 'vector', 'bm25' or 'hybrid'")
        w_vec = {"vector": 1.0, "bm25": 0.0}.get(mode, self.default_alpha if alpha is None else float(alpha))
        w_vec = max(0.0, min(1.0, w_vec))
        use_rerank = (self.reranker is not None) if rerank is None else (rerank and self.reranker is not None)

        allowed_ids = [c for c in self.records if self._allowed(c, doc_ids)]
        if not allowed_ids:
            return []
        pool = min(max(top_k * 4, 20), len(allowed_ids))

        q_emb = self.embed([query])[0]

        # ---- vector ranks
        vec_rank: Dict[str, int] = {}
        if w_vec > 0:
            where = {"doc_id": {"$in": doc_ids}} if doc_ids else None
            res = self.collection.query(query_embeddings=[q_emb], n_results=pool, where=where)
            for rank, cid in enumerate(res["ids"][0], start=1):
                vec_rank[cid] = rank

        # ---- bm25 ranks
        bm_rank: Dict[str, int] = {}
        bm_raw: Dict[str, float] = {}
        q_tokens = tokenize(query)
        if w_vec < 1.0 and self._bm25 is not None and q_tokens:
            scores = self._bm25.get_scores(q_tokens)
            allowed_set = set(allowed_ids)
            order = [i for i in np.argsort(-scores) if self._bm25_ids[i] in allowed_set and scores[i] > 0][:pool]
            for rank, i in enumerate(order, start=1):
                bm_rank[self._bm25_ids[i]] = rank
                bm_raw[self._bm25_ids[i]] = float(scores[i])

        candidates = set(vec_rank) | set(bm_rank)
        if not candidates:
            return []

        # ---- fusion
        fused: Dict[str, float] = {}
        if self.fusion_method == "minmax":
            vmax = max(bm_raw.values()) if bm_raw else 1.0
            for cid in candidates:
                v = 1.0 / vec_rank[cid] if cid in vec_rank else 0.0
                b = bm_raw.get(cid, 0.0) / vmax if vmax else 0.0
                fused[cid] = w_vec * v + (1 - w_vec) * b
        else:  # weighted reciprocal rank fusion
            k = 60.0
            for cid in candidates:
                v = 1.0 / (k + vec_rank[cid]) if cid in vec_rank else 0.0
                b = 1.0 / (k + bm_rank[cid]) if cid in bm_rank else 0.0
                fused[cid] = w_vec * v + (1 - w_vec) * b

        results = []
        for cid in candidates:
            rec = self.records[cid]
            results.append(RetrievalResult(
                cid, rec["text"], rec["metadata"],
                vector_rank=vec_rank.get(cid), bm25_rank=bm_rank.get(cid),
                cosine_score=cosine(q_emb, rec["embedding"]) if rec["embedding"] is not None else 0.0,
                bm25_score=bm_raw.get(cid, 0.0), fused_score=fused[cid],
            ))
        results.sort(key=lambda r: r.fused_score, reverse=True)
        results = results[: max(top_k * 3, 10)]

        # ---- optional cross-encoder rerank
        if use_rerank:
            logits = self.reranker.score(query, [r.text for r in results])
            for r, logit in zip(results, logits):
                r.rerank_score = _sigmoid(logit)
            results.sort(key=lambda r: r.rerank_score, reverse=True)

        # ---- relevance threshold on an absolute score
        if apply_threshold:
            results = [r for r in results if self.passes_threshold(r)]
        return results[:top_k]

    def passes_threshold(self, r: RetrievalResult) -> bool:
        """True if the chunk is relevant enough to show to the LLM (absolute score, not per-query rank)."""
        if r.rerank_score is not None:
            return r.rerank_score >= self.min_rerank_score
        return r.cosine_score >= self.min_cosine_score
